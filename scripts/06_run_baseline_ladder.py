#!/usr/bin/env python3
"""
Phase 4 — Baseline ladder + external rescoring  (RFC-001 §6, RQ1/RQ2 support).

Compares alignment methods by generating a sequence pool from each and scoring it with
EXTERNAL oracles:
  B0  SFT reference            (no alignment)
  B1  CRISPGen-REINFORCE       (prior work; its generated pool, reused)
  B3  Single-objective DPO     (efficacy-only preferences)
  B4  Scalarized MODPO         (weighted-sum preference, w=0.5/0.5)
  B5  Pareto-DPO (ours)        (dominance preferences)

Efficacy  = CRISPRon SURROGATE (CNN trained on 300k real CRISPRon labels; fidelity
            reported).  Off-target = REAL chr22 structural near-match count (MM<=3,
            partial-genome Cas-OFFinder-style).  Also: Pareto hypervolume, efficacy-
            safety trade-off, diversity.

Outputs: results/json/baseline_ladder.json , results/csv/baseline_metrics.csv
"""
from __future__ import annotations
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT))
from _common import (new_policy_from_sft, dpo_train, load_pref, diversity_metrics, TOK)  # noqa
from _eval_oracles import get_crispron_surrogate, get_chr22_sites, offtarget_counts       # noqa
from models.policy_decoder import ARDecoder, DecoderConfig                                # noqa

# import pool loader + pair builder from 01 (module name starts with a digit)
_spec = importlib.util.spec_from_file_location(
    "build01", PROJECT / "scripts" / "01_build_preference_pairs.py")
build01 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(build01)

POOL = 512
BETA = 0.20                     # RQ2 sweet spot (fair operating point for all DPO variants)
EPOCHS = 12
SEED = 42
B5_CKPT = "pareto_dpo_beta_0.20.pt"   # was pareto_dpo_best.pt (beta=0.10, diversity-collapsed)


def precompute_ref_logp(df: pd.DataFrame, device) -> pd.DataFrame:
    ref = new_policy_from_sft(device).eval()

    @torch.no_grad()
    def lp(seqs):
        out = np.empty(len(seqs))
        for i in range(0, len(seqs), 2048):
            ids = TOK.encode(list(seqs[i:i + 2048]), device=device)
            out[i:i + ids.shape[0]] = ref.sequence_logprob(ids).float().cpu().numpy()
        return out
    df = df.copy()
    df["ref_logp_w"] = lp(df["y_w"].tolist())
    df["ref_logp_l"] = lp(df["y_l"].tolist())
    return df


def split_to_tensors(df):
    return (TOK.encode(df["y_w"].tolist()), TOK.encode(df["y_l"].tolist()),
            torch.tensor(df["ref_logp_w"].to_numpy(), dtype=torch.float32),
            torch.tensor(df["ref_logp_l"].to_numpy(), dtype=torch.float32))


def train_baseline_dpo(pref_df, val_split, device, amp):
    tr = split_to_tensors(precompute_ref_logp(pref_df, device))
    model = new_policy_from_sft(device)
    model, _ = dpo_train(model, tr, val_split, device, beta=BETA, epochs=EPOCHS,
                         seed=SEED, amp=amp)
    return model


def hypervolume_2d(eff01: np.ndarray, supp01: np.ndarray, ref=(0.0, 0.0)) -> float:
    """HV dominated by the non-dominated set of (eff01, supp01) above reference `ref`."""
    pts = np.stack([eff01, supp01], 1)
    pts = pts[(pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])]
    if len(pts) == 0:
        return 0.0
    order = np.argsort(-pts[:, 0])                      # eff descending
    pts = pts[order]
    hv, prev_supp = 0.0, ref[1]
    best_supp = ref[1]
    nd = []
    for e, s in pts:                                    # keep non-dominated (eff desc)
        if s > best_supp:
            nd.append((e, s)); best_supp = s
    prev_e = ref[0]
    for e, s in sorted(nd, key=lambda t: t[1]):         # supp ascending
        hv += (e - ref[0]) * (s - prev_supp)
        prev_supp = s
    return float(hv)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if amp:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    val_split = load_pref("val")

    # ── external oracles ────────────────────────────────────────────────────
    predict_eff, fidelity = get_crispron_surrogate(device)
    print(f"[oracle] CRISPRon surrogate fidelity: Spearman={fidelity['heldout_spearman']:.3f} "
          f"Pearson={fidelity['heldout_pearson']:.3f} RMSE={fidelity['heldout_rmse']:.2f}")
    sites = get_chr22_sites(n_sample=500_000, seed=0)
    print(f"[oracle] chr22 structural off-target: {len(sites):,} NGG sites (sampled, real)")

    # ── scored pool for building B3/B4 preferences ──────────────────────────
    pool = build01.load_pilot_pool(max_mm=3, ood_chrom="chr22")
    pool["const0"] = 0.0
    from scipy.stats import rankdata
    pool["scalar"] = 0.5 * rankdata(pool["s_eff"]) + 0.5 * rankdata(pool["s_off_id"])

    # B3: efficacy-only dominance ; B4: scalarized (single scalar) dominance
    b3_pairs = build01.build_pairs(pool, "s_eff", "const0", max_pairs=40000,
                                   fanout_cap=12, n_bins=6, seed=SEED)
    b4_pairs = build01.build_pairs(pool, "scalar", "const0", max_pairs=40000,
                                   fanout_cap=12, n_bins=6, seed=SEED)
    print(f"[prefs] B3(eff-only)={len(b3_pairs)} pairs | B4(scalarized)={len(b4_pairs)} pairs")

    # ── policies ────────────────────────────────────────────────────────────
    policies = {}
    policies["B0_SFT"] = new_policy_from_sft(device)
    print("[train] B3 single-objective DPO ..."); policies["B3_SingleObjDPO"] = train_baseline_dpo(b3_pairs, val_split, device, amp)
    print("[train] B4 scalarized MODPO ...");     policies["B4_ScalarizedMODPO"] = train_baseline_dpo(b4_pairs, val_split, device, amp)
    st = torch.load(PROJECT / "models" / "checkpoints" / B5_CKPT,
                    map_location=device, weights_only=False)
    b5 = ARDecoder(DecoderConfig(**st["config"])).to(device); b5.load_state_dict(st["model"])
    policies["B5_ParetoDPO"] = b5

    # ── generate pools ──────────────────────────────────────────────────────
    pools = {name: m.sample(POOL, device, seed=SEED) for name, m in policies.items()}
    # B1 = CRISPGen-REINFORCE generated pool (reuse existing generated sequences)
    crispgen = pd.read_csv(NOTEBOOK / "report" / "CRISPGen_Mass_Pool.csv", usecols=["seq"], nrows=200000)
    pools["B1_CRISPGen_REINFORCE"] = crispgen["seq"].sample(POOL, random_state=SEED).tolist()

    order = ["B0_SFT", "B1_CRISPGen_REINFORCE", "B3_SingleObjDPO",
             "B4_ScalarizedMODPO", "B5_ParetoDPO"]

    # ── score every pool with the SAME external oracles ─────────────────────
    scored = {}
    for name in order:
        seqs = pools[name]
        eff = predict_eff(seqs)                              # 0-100 (surrogate)
        off = offtarget_counts(seqs, sites, device, max_mm=3).astype(float)
        scored[name] = {"eff": eff, "off": off, "div": diversity_metrics(seqs)}
        print(f"[score] {name:<24} eff mean {eff.mean():5.1f} | chr22 off mean {off.mean():7.1f} "
              f"| hamming {scored[name]['div']['mean_pairwise_hamming_frac']:.3f}")

    # normalization for hypervolume (common across methods)
    all_off = np.concatenate([scored[n]["off"] for n in order])
    off_ref = float(np.percentile(all_off, 99) + 1e-9)      # fixed suppression scale

    rows = []
    for name in order:
        eff, off, div = scored[name]["eff"], scored[name]["off"], scored[name]["div"]
        eff01 = np.clip(eff / 100.0, 0, 1)
        supp01 = np.clip(1.0 - off / off_ref, 0, 1)          # higher = safer
        hv = hypervolume_2d(eff01, supp01)
        rows.append({
            "method": name,
            "eff_mean": round(float(eff.mean()), 2),
            "eff_pct_ge_50": round(float((eff >= 50).mean() * 100), 1),
            "chr22_offtarget_mean": round(float(off.mean()), 2),
            "chr22_offtarget_median": round(float(np.median(off)), 1),
            "suppression_mean01": round(float(supp01.mean()), 4),
            "pareto_hypervolume": round(hv, 4),
            "pos_entropy_bits": round(div["positional_entropy_bits"], 4),
            "mean_pairwise_hamming": round(div["mean_pairwise_hamming_frac"], 4),
            "unique_frac": round(div["unique_frac"], 4),
        })

    out = {"pool_size": POOL, "beta": BETA, "epochs": EPOCHS, "seed": SEED,
           "b5_checkpoint": B5_CKPT,
           "efficacy_oracle": "CRISPRon SURROGATE (CNN on 300k real labels)",
           "efficacy_oracle_fidelity": fidelity,
           "offtarget_oracle": f"real chr22 NGG structural near-match MM<=3 ({len(sites):,} sampled sites)",
           "hv_off_reference_p99": round(off_ref, 2),
           "results": rows, "wall_seconds": round(time.time() - t0, 1)}
    if amp:
        out["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    (PROJECT / "results" / "json" / "baseline_ladder.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(PROJECT / "results" / "csv" / "baseline_ladder.csv", index=False)

    # dump per-guide scored arrays so downstream analysis (set-coverage, bootstrapped
    # hypervolume CIs, GD/IGD; scripts/09) can recompute Pareto metrics without retraining
    npz_dir = PROJECT / "results" / "npz"; npz_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz_dir / "ladder_pools.npz",
             methods=np.array(order),
             off_ref=np.float64(off_ref),
             **{f"{n}__eff": scored[n]["eff"] for n in order},
             **{f"{n}__off": scored[n]["off"] for n in order})
    pd.DataFrame(rows).to_csv(PROJECT / "results" / "csv" / "baseline_metrics.csv", index=False)
    print(f"[done] {out['wall_seconds']}s | peak VRAM {out.get('peak_vram_mb','-')} MB "
          f"-> results/json/baseline_ladder.json + csv")


if __name__ == "__main__":
    main()
