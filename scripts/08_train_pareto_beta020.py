#!/usr/bin/env python3
"""
Train the Pareto-DPO policy at the RQ2 sweet-spot beta=0.20 and persist it as a
first-class checkpoint for the (fair) baseline ladder.

Phase-4 postmortem: the ladder's B5 used the beta=0.10 checkpoint, which the RQ2
sweep shows is diversity-collapsed (unique frac 0.10). beta=0.20 gives near-best
alignment (val reward-acc 0.791) AND high diversity (unique frac 0.83), so it is
the correct operating point for a fair method comparison.

Output: models/checkpoints/pareto_dpo_beta_0.20.pt  = {"model": state_dict,
        "config": DecoderConfig-as-dict, "beta": 0.20, "val_reward_acc": ...}
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT))
from _common import load_pref, new_policy_from_sft, dpo_train, dpo_eval, SFT  # noqa: E402

BETA = 0.20
# NB: 12 epochs — this MUST match the RQ2 beta-sweep budget (05_run_beta_sweep.py),
# because the "sweet spot" characterization (val acc 0.791, unique frac 0.828) is the
# 12-epoch model. Training longer (e.g. 20 ep) over-aligns and re-collapses diversity
# to ~0.54, which would silently move the operating point away from the one RQ2 chose.
EPOCHS = 12
SEED = 42


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if amp:
        torch.cuda.reset_peak_memory_stats()
    train, val, ood = load_pref("train"), load_pref("val"), load_pref("ood")

    torch.manual_seed(SEED)
    model = new_policy_from_sft(device)
    t0 = time.time()
    model, best_acc = dpo_train(model, train, val, device, beta=BETA,
                                epochs=EPOCHS, seed=SEED, amp=amp, verbose=True)

    val_ev = dpo_eval(model, val, device, BETA, amp)
    ood_ev = dpo_eval(model, ood, device, BETA, amp)

    cfg = torch.load(SFT, map_location="cpu", weights_only=False)["config"]
    out = PROJECT / "models" / "checkpoints" / "pareto_dpo_beta_0.20.pt"
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": cfg, "beta": BETA,
                "val_reward_acc": round(best_acc, 4),
                "ood_reward_acc": round(ood_ev["reward_acc"], 4)}, out)

    print(f"\n[done] beta={BETA} | best val_acc={best_acc:.3f} "
          f"val_KL={val_ev['kl_to_ref']:.1f} val_reward_var={val_ev['implicit_reward_var']:.2f} "
          f"| OOD acc={ood_ev['reward_acc']:.3f} | {time.time()-t0:.1f}s "
          f"| peak VRAM {torch.cuda.max_memory_allocated()/1e6 if amp else 0:.1f} MB")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
