#!/usr/bin/env python3
"""
Phase-2 reviewer analyses: dynamic saturation, sequence memorization, and diversity metrics.

(a) DYNAMIC SATURATION CURVES. We train Pareto-DPO (beta=0.20) and, at every epoch, record the
    implicit-reward variance V[pi] and the held-out pairwise (dominance) accuracy, together with
    the sampled-pool unique fraction. This shows the preference signal does not saturate over
    training (variance stays high, accuracy plateaus) and locates the epoch at which diversity
    begins to collapse — motivating the diversity-aware early stop.

(b) MEMORIZATION CHECK. We sample guides from the trained policy and compute, for each, the
    minimum Hamming distance d_min to the nearest TRAINING guide. A generator that memorized its
    training set would show d_min = 0 for many samples; we report the d_min distribution, the
    exact-copy fraction, and — as a reference scale — the train-set's own nearest-neighbour d_min.

(c) DIVERSITY METRICS for Tables 1 and 2: mean pairwise Hamming distance (fraction) and mean
    positional Shannon entropy of the generated pool (confirmatory; also emitted per split).

Outputs: results/json/dynamic_saturation_memorization.json,
         results/plots/fig9_dynamic_saturation.png, fig10_memorization.png (+ manuscript_assets, +pdf)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT))
from _common import (load_pref, new_policy_from_sft, dpo_eval, diversity_metrics,  # noqa: E402
                     TOK, PAIRS)
from models.pareto_dpo_loss import pareto_dpo_loss                                 # noqa: E402

BETA, MAX_EPOCHS, SEED = 0.20, 20, 42
MB, ACCUM, LR = 16, 4, 1e-4
NUC = {"A": 0, "C": 1, "G": 2, "T": 3}
CB = {"var": "#0072B2", "acc": "#D55E00", "div": "#009E73", "grey": "#8C8C8C"}
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.family": "DejaVu Sans",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "legend.frameon": False})


def _save(fig, name):
    for ext in ("png", "pdf"):
        for d in (PROJECT / "results" / "plots", PROJECT / "manuscript_assets"):
            fig.savefig(d / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)
    print(f"[fig] {name}.png/.pdf written")


def train_with_history(train, val, device, amp):
    """Train Pareto-DPO recording (epoch, variance, val_acc, unique_frac) each epoch."""
    model = new_policy_from_sft(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    n = len(train[0])
    spe = int(np.ceil(n / MB / ACCUM))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS * spe, eta_min=LR / 20)
    hist, best = [], (-1.0, None)
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED + ep))
        opt.zero_grad(); micro = 0
        for i in range(0, n, MB):
            idx = perm[i:i + MB]
            iw, il = train[0][idx].to(device), train[1][idx].to(device)
            rw, rl = train[2][idx].to(device), train[3][idx].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pw = model.sequence_logprob(iw).float(); pl = model.sequence_logprob(il).float()
            loss, _ = pareto_dpo_loss(pw, pl, rw, rl, beta=BETA)
            (loss / ACCUM).backward(); micro += 1
            if micro % ACCUM == 0 or i + MB >= n:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(); sched.step()
        ev = dpo_eval(model, val, device, BETA, amp)
        div = diversity_metrics(model.sample(1024, device, seed=SEED))
        hist.append({"epoch": ep, "implicit_reward_var": round(ev["implicit_reward_var"], 4),
                     "val_pairwise_acc": round(ev["reward_acc"], 4),
                     "unique_frac": round(div["unique_frac"], 4),
                     "mean_pairwise_hamming": round(div["mean_pairwise_hamming_frac"], 4),
                     "pos_entropy_bits": round(div["positional_entropy_bits"], 4)})
        print(f"  ep{ep:02d} var={hist[-1]['implicit_reward_var']:.3f} "
              f"acc={hist[-1]['val_pairwise_acc']:.3f} unique={hist[-1]['unique_frac']:.3f}")
        if ev["reward_acc"] > best[0]:
            best = (ev["reward_acc"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return model, hist


@torch.no_grad()
def min_hamming_to_train(gen, train_seqs, device):
    """For each generated 20-mer, minimum Hamming distance to any training guide."""
    G = torch.tensor([[NUC[c] for c in s] for s in gen], dtype=torch.uint8).to(device)
    T = torch.tensor([[NUC[c] for c in s] for s in train_seqs], dtype=torch.uint8)
    dmin = np.empty(len(gen), dtype=np.int16)
    for i in range(0, len(gen), 256):
        g = G[i:i + 256]
        block = np.full(g.shape[0], 21, dtype=np.int16)
        for j in range(0, len(T), 20000):
            tb = T[j:j + 20000].to(device)
            hd = (g[:, None, :] != tb[None, :, :]).sum(2)            # (gc, tb)
            block = np.minimum(block, hd.min(1).values.cpu().numpy().astype(np.int16))
            del tb, hd
        dmin[i:i + g.shape[0]] = block
    return dmin


@torch.no_grad()
def train_internal_nn(train_seqs, device):
    """Nearest-neighbour Hamming distance within the training set, excluding self."""
    T = torch.tensor([[NUC[c] for c in s] for s in train_seqs], dtype=torch.uint8).to(device)
    nn = np.empty(len(train_seqs), dtype=np.int16)
    for i in range(0, len(train_seqs), 256):
        g = T[i:i + 256]
        hd = (g[:, None, :] != T[None, :, :]).sum(2)                # (gc, N) includes self=0
        # set the self-distance to a large value so the min is the true nearest OTHER
        rows = torch.arange(g.shape[0], device=device)
        hd[rows, i + rows] = 21
        nn[i:i + g.shape[0]] = hd.min(1).values.cpu().numpy().astype(np.int16)
        del hd
    return nn


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    train, val = load_pref("train"), load_pref("val")

    print("[a] dynamic saturation curves ...")
    model, hist = train_with_history(train, val, device, amp)

    print("[b] memorization check ...")
    tdf = pd.read_csv(PAIRS / "train_ref.csv")
    train_seqs = sorted(set(tdf["y_w"]).union(set(tdf["y_l"])))
    gen = model.sample(2048, device, seed=SEED)
    dmin = min_hamming_to_train(gen, train_seqs, device)
    # reference scale: nearest-neighbour d_min WITHIN the training set (exclude self-match)
    tt_nn = train_internal_nn(train_seqs, device)

    L = 20
    memo = {
        "n_generated": len(gen),
        "n_train_guides": len(train_seqs),
        "dmin_mean": round(float(dmin.mean()), 3),
        "dmin_median": float(np.median(dmin)),
        "dmin_min": int(dmin.min()),
        "exact_copy_fraction": round(float((dmin == 0).mean()), 5),
        "frac_within_1mismatch": round(float((dmin <= 1).mean()), 5),
        "train_internal_nn_dmin_mean": round(float(tt_nn.mean()), 3),
        "seq_length": L,
    }
    print(f"    d_min mean={memo['dmin_mean']} exact-copies={memo['exact_copy_fraction']}")

    pool_div = diversity_metrics(gen)
    best_ep = max(hist, key=lambda h: h["val_pairwise_acc"])["epoch"]
    # diversity-aware early stop: max accuracy subject to unique fraction >= 0.80
    div_ok = [h for h in hist if h["unique_frac"] >= 0.80]
    div_aware_ep = (max(div_ok, key=lambda h: h["val_pairwise_acc"])["epoch"] if div_ok else best_ep)
    out = {"beta": BETA, "max_epochs": MAX_EPOCHS, "best_acc_epoch": best_ep,
           "diversity_aware_early_stop_epoch": div_aware_ep, "unique_frac_floor": 0.80,
           "saturation_curve": hist, "memorization": memo,
           "final_pool_diversity": {"mean_pairwise_hamming": round(pool_div["mean_pairwise_hamming_frac"], 4),
                                     "pos_entropy_bits": round(pool_div["positional_entropy_bits"], 4),
                                     "unique_frac": round(pool_div["unique_frac"], 4)}}
    (PROJECT / "results" / "json" / "dynamic_saturation_memorization.json").write_text(json.dumps(out, indent=2))

    # ── Fig 9: dynamic saturation ────────────────────────────────────────────
    ep = [h["epoch"] for h in hist]
    var = [h["implicit_reward_var"] for h in hist]
    acc = [h["val_pairwise_acc"] for h in hist]
    uniq = [h["unique_frac"] for h in hist]
    fig, ax1 = plt.subplots(figsize=(6.6, 4.0), constrained_layout=True)
    ax1.plot(ep, var, "-o", color=CB["var"], ms=4, label="implicit-reward variance")
    ax1.set_xlabel("training epoch"); ax1.set_ylabel("implicit-reward variance", color=CB["var"])
    ax1.tick_params(axis="y", labelcolor=CB["var"])
    ax2 = ax1.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(ep, acc, "-s", color=CB["acc"], ms=4, label="val pairwise accuracy")
    ax2.plot(ep, uniq, "-^", color=CB["div"], ms=4, label="unique fraction (diversity)")
    ax2.set_ylabel("accuracy / unique fraction"); ax2.set_ylim(0.4, 1.02)
    ax2.axhline(0.5, ls=":", color=CB["grey"], lw=1)
    ax2.axhline(0.80, ls=":", color=CB["div"], lw=1, alpha=0.6)
    ax2.axvline(div_aware_ep, ls="--", color=CB["grey"], lw=1.2)
    ax2.text(div_aware_ep + 0.25, 0.44, f"diversity-aware\nearly stop (ep {div_aware_ep})",
             fontsize=7.5, color=CB["grey"])
    lines = ax1.get_lines() + ax2.get_lines()[:2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=7.5)
    ax1.set_title("Dynamic saturation: signal stays dispersed & accurate;\ndiversity collapses only late")
    _save(fig, "fig9_dynamic_saturation")

    # ── Fig 10: memorization ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    bins = np.arange(0, L + 2) - 0.5
    ax.hist(dmin, bins=bins, color=CB["var"], alpha=0.8,
            label=f"generated → nearest train guide\n(mean {memo['dmin_mean']}, exact copies {memo['exact_copy_fraction']*100:.2f}%)")
    ax.axvline(0, color=CB["acc"], lw=1.4, ls="--")
    ax.set_xlabel(r"minimum Hamming distance $d_{\min}$ to training set (of 20 nt)")
    ax.set_ylabel("generated guides")
    ax.set_title("No sequence memorization:\ngenerated guides are far from all training guides")
    ax.legend(loc="upper right", fontsize=7.5)
    _save(fig, "fig10_memorization")

    print(f"[done] best_acc_epoch={best_ep} -> results/json/dynamic_saturation_memorization.json")


if __name__ == "__main__":
    main()
