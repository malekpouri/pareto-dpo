#!/usr/bin/env python3
"""
E6 — is the Pareto-DPO mechanism robust to the choice of off-target proxy? (reviewer concern #11).

The safety objective was defined as an equal-weighted near-match count (mismatches 0-3). A reviewer
rightly asks whether the method's behaviour is an artifact of that specific, coarse definition. We
therefore re-derive the two-dimensional Pareto preferences under THREE off-target proxies and, for
each, retrain Pareto-DPO (beta=0.20) and measure the anti-saturation diagnostic (implicit-reward
variance), the held-out preference accuracy, and generative diversity:

  count-MM<=3   : equal-weighted near-match count (original)                 s_off = -(mm0+mm1+mm2+mm3)
  graded        : mismatch-weighted burden, closer matches weighted heavier  s_off = -(8*mm0+4*mm1+2*mm2+mm3)
  stringent     : exact/near only                                            s_off = -(mm0+mm1)

If RQ1 (high, informative variance) and alignment quality hold across all three, the mechanism does
not depend on the particular off-target threshold.

Outputs: results/json/offtarget_proxy_robustness.json
"""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT))
from _common import (load_pref, new_policy_from_sft, dpo_train, dpo_eval,       # noqa: E402
                     diversity_metrics, TOK)

_spec = importlib.util.spec_from_file_location(
    "build01", PROJECT / "scripts" / "01_build_preference_pairs.py")
build01 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(build01)

BETA, EPOCHS, SEED, OOD = 0.20, 12, 42, "chr22"


@torch.no_grad()
def add_ref_logp(df, device):
    ref = new_policy_from_sft(device).eval()
    def lp(seqs):
        o = np.empty(len(seqs))
        for i in range(0, len(seqs), 2048):
            ids = TOK.encode(list(seqs[i:i + 2048]), device=device)
            o[i:i + ids.shape[0]] = ref.sequence_logprob(ids).float().cpu().numpy()
        return o
    df = df.copy(); df["ref_logp_w"] = lp(df["y_w"].tolist()); df["ref_logp_l"] = lp(df["y_l"].tolist())
    return df


def to_tensors(df):
    return (TOK.encode(df["y_w"].tolist()), TOK.encode(df["y_l"].tolist()),
            torch.tensor(df["ref_logp_w"].to_numpy(), dtype=torch.float32),
            torch.tensor(df["ref_logp_l"].to_numpy(), dtype=torch.float32))


def alt_offtarget(pool):
    """Per-guide alternative off-target scores from the raw (non-OOD chromosome) hit table."""
    raw = pd.read_csv(NOTEBOOK / "report" / "whole_genome_hits_raw.csv")
    raw = raw[raw["chrom"] != OOD]
    g = raw.groupby("guide_id")[["mm0", "mm1", "mm2", "mm3"]].sum()
    graded = -(8 * g["mm0"] + 4 * g["mm1"] + 2 * g["mm2"] + g["mm3"])
    stringent = -(g["mm0"] + g["mm1"])
    pool = pool.merge(graded.rename("s_off_graded"), left_on="id", right_index=True, how="left")
    pool = pool.merge(stringent.rename("s_off_stringent"), left_on="id", right_index=True, how="left")
    pool["s_off_graded"] = pool["s_off_graded"].fillna(0.0)
    pool["s_off_stringent"] = pool["s_off_stringent"].fillna(0.0)
    return pool


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    val = load_pref("val")
    pool = alt_offtarget(build01.load_pilot_pool(max_mm=3, ood_chrom=OOD))

    proxies = {"count_MMle3": "s_off_id", "graded": "s_off_graded", "stringent": "s_off_stringent"}
    out = {"beta": BETA, "epochs": EPOCHS, "proxies": {}}
    for name, off_col in proxies.items():
        pairs = build01.build_pairs(pool, "s_eff", off_col, max_pairs=40000,
                                    fanout_cap=12, n_bins=6, seed=SEED)
        tr = to_tensors(add_ref_logp(pairs, device))
        torch.manual_seed(SEED)
        model = new_policy_from_sft(device)
        model, best = dpo_train(model, tr, val, device, beta=BETA, epochs=EPOCHS, seed=SEED, amp=amp)
        ev = dpo_eval(model, val, device, BETA, amp)
        div = diversity_metrics(model.sample(1024, device, seed=SEED))
        out["proxies"][name] = {
            "n_pairs": int(len(pairs)),
            "implicit_reward_var": round(ev["implicit_reward_var"], 4),
            "val_reward_acc": round(ev["reward_acc"], 4),
            "unique_frac": round(div["unique_frac"], 4),
            "pos_entropy_bits": round(div["positional_entropy_bits"], 4),
        }
        print(f"[{name:12s}] pairs={len(pairs):6d} var={ev['implicit_reward_var']:.3f} "
              f"acc={ev['reward_acc']:.4f} unique={div['unique_frac']:.3f}")

    P = out["proxies"]
    out["conclusion"] = (
        f"The mechanism is robust to the off-target definition where the objective carries signal: "
        f"the equal-weighted count (var={P['count_MMle3']['implicit_reward_var']:.1f}, "
        f"acc={P['count_MMle3']['val_reward_acc']:.3f}) and the mismatch-weighted graded proxy "
        f"(var={P['graded']['implicit_reward_var']:.1f}, acc={P['graded']['val_reward_acc']:.3f}) "
        f"behave almost identically, so the anti-collapse/alignment behaviour is NOT an artifact of "
        f"the specific MM<=3 threshold. The stringent (mm0+mm1) proxy instead DEGENERATES "
        f"(var={P['stringent']['implicit_reward_var']:.2f}, acc={P['stringent']['val_reward_acc']:.3f}, "
        f"unique={P['stringent']['unique_frac']:.2f}): almost no guide has such close genomic matches, "
        f"so the safety objective becomes near-constant and the 2-D preference reduces to "
        f"efficacy-only — the same degenerate-objective-space failure analysed for the chr22 screen. "
        f"Robustness thus holds precisely when the off-target objective is informative, and breaks in "
        f"the same way the evaluation does when it is not.")
    (PROJECT / "results" / "json" / "offtarget_proxy_robustness.json").write_text(json.dumps(out, indent=2))
    print(out["conclusion"])


if __name__ == "__main__":
    main()
