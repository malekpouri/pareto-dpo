#!/usr/bin/env python3
"""
Phase 4 / RQ2 — beta-sweep: likelihood displacement vs diversity  (RFC-001 §2.5, RQ2).

Trains Pareto-DPO at beta in {0.05, 0.1, 0.2, 0.5} (all else fixed) and, for each,
records the alignment-vs-diversity trade-off:
  * KL(policy || ref) proxy and chosen-logprob displacement (pi_theta - pi_ref)
  * val reward-accuracy (alignment quality)
  * diversity of a freshly SAMPLED pool: positional Shannon entropy, 3-mer entropy,
    mean pairwise Hamming, unique fraction.

Outputs: results/json/rq2_beta_sweep.json , results/csv/rq2_beta_sweep.csv
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from _common import (load_pref, new_policy_from_sft, dpo_train, dpo_eval,  # noqa: E402
                     diversity_metrics)

BETAS = [0.05, 0.1, 0.2, 0.5]
EPOCHS = 12
POOL = 1024                     # sampled sequences for diversity
SEED = 42


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if amp:
        torch.cuda.reset_peak_memory_stats()
    train, val = load_pref("train"), load_pref("val")

    # reference (SFT) diversity baseline for context
    ref = new_policy_from_sft(device)
    ref_div = diversity_metrics(ref.sample(POOL, device, seed=SEED))
    print(f"[ref/SFT] diversity: pos_ent={ref_div['positional_entropy_bits']:.3f} bits | "
          f"hamming={ref_div['mean_pairwise_hamming_frac']:.3f} | unique={ref_div['unique_frac']:.3f}")

    rows, t0 = [], time.time()
    for beta in BETAS:
        torch.manual_seed(SEED)
        model = new_policy_from_sft(device)
        model, best_acc = dpo_train(model, train, val, device, beta=beta,
                                    epochs=EPOCHS, seed=SEED, amp=amp)
        ev = dpo_eval(model, val, device, beta, amp)
        div = diversity_metrics(model.sample(POOL, device, seed=SEED))
        row = {"beta": beta, "val_reward_acc": round(ev["reward_acc"], 4),
               "kl_to_ref": round(ev["kl_to_ref"], 3),
               "chosen_logprob_displacement": round(ev["disp_chosen"], 3),
               "implicit_reward_var": round(ev["implicit_reward_var"], 4),
               "reward_margin": round(ev["reward_margin"], 4),
               "pos_entropy_bits": round(div["positional_entropy_bits"], 4),
               "kmer3_entropy_bits": round(div["kmer3_entropy_bits"], 4),
               "mean_pairwise_hamming": round(div["mean_pairwise_hamming_frac"], 4),
               "unique_frac": round(div["unique_frac"], 4)}
        rows.append(row)
        print(f"[beta={beta}] acc={row['val_reward_acc']} KL={row['kl_to_ref']} "
              f"disp={row['chosen_logprob_displacement']} | "
              f"pos_ent={row['pos_entropy_bits']} hamming={row['mean_pairwise_hamming']} "
              f"unique={row['unique_frac']}")

    out = {"betas": BETAS, "epochs": EPOCHS, "pool_size": POOL, "seed": SEED,
           "ref_sft_diversity": {k: round(v, 4) for k, v in ref_div.items()
                                 if isinstance(v, float)},
           "sweep": rows, "wall_seconds": round(time.time() - t0, 1)}
    if amp:
        out["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    (PROJECT / "results" / "json" / "rq2_beta_sweep.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(PROJECT / "results" / "csv" / "rq2_beta_sweep.csv", index=False)
    print(f"[done] {out['wall_seconds']}s | peak VRAM {out.get('peak_vram_mb','-')} MB "
          f"-> results/json/rq2_beta_sweep.json + csv")


if __name__ == "__main__":
    main()
