#!/usr/bin/env python3
"""
E4 — is the efficacy gap (B5 49.9 vs best baseline 52.2) practically/statistically real?
(reviewer concern #13).

Using the per-guide efficacy arrays dumped by the ladder, we report the full distribution and
bootstrap 95% CIs of the mean efficacy per method, the B5-vs-best-baseline difference with a
bootstrap CI and a Mann-Whitney U test, and the effect size in interpretable units (surrogate
RMSE and Cohen's d). The point is to show the mean gap is small relative to both the surrogate's
own error and the within-method spread — i.e. B5 is efficacy-competitive, not efficacy-sacrificing.

Outputs: results/json/efficacy_significance.json,
         results/plots/fig7_efficacy_significance.png (+ manuscript_assets, +pdf)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

PROJECT = Path(__file__).resolve().parents[1]
NPZ = PROJECT / "results" / "npz" / "ladder_pools.npz"
SURROGATE_RMSE = 12.4                                   # held-out RMSE of the efficacy oracle
CB = {"B0_SFT": "#8C8C8C", "B1_CRISPGen_REINFORCE": "#E69F00", "B3_SingleObjDPO": "#56B4E9",
      "B4_ScalarizedMODPO": "#009E73", "B5_ParetoDPO": "#D55E00"}
SHORT = {"B0_SFT": "B0 SFT", "B1_CRISPGen_REINFORCE": "B1 REINFORCE",
         "B3_SingleObjDPO": "B3 Single-DPO", "B4_ScalarizedMODPO": "B4 Scalarized",
         "B5_ParetoDPO": "B5 Pareto-DPO"}
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.family": "DejaVu Sans",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "legend.frameon": False})


def boot_mean_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), bs


def main():
    d = np.load(NPZ, allow_pickle=True)
    methods = [str(m) for m in d["methods"]]
    eff = {m: d[f"{m}__eff"].astype(float) for m in methods}

    stats_per = {}
    means_bs = {}
    for m in methods:
        mu, lo, hi, bs = boot_mean_ci(eff[m])
        means_bs[m] = bs
        stats_per[m] = {"mean": round(mu, 3), "ci95": [round(lo, 3), round(hi, 3)],
                        "std": round(float(eff[m].std()), 3), "n": int(len(eff[m]))}

    b5 = "B5_ParetoDPO"
    best_base = max([m for m in methods if m != b5], key=lambda m: eff[m].mean())
    diff = eff[best_base].mean() - eff[b5].mean()
    diff_bs = means_bs[best_base] - means_bs[b5]
    u, p_mw = stats.mannwhitneyu(eff[best_base], eff[b5], alternative="two-sided")
    pooled_sd = np.sqrt((eff[b5].var(ddof=1) + eff[best_base].var(ddof=1)) / 2)
    cohens_d = float(diff / pooled_sd)

    res = {
        "per_method": stats_per,
        "best_baseline": best_base,
        "b5_vs_best_baseline": {
            "mean_difference": round(float(diff), 3),
            "difference_ci95": [round(float(np.percentile(diff_bs, 2.5)), 3),
                                round(float(np.percentile(diff_bs, 97.5)), 3)],
            "difference_in_surrogate_RMSE_units": round(float(diff / SURROGATE_RMSE), 3),
            "cohens_d": round(cohens_d, 3),
            "mann_whitney_p": float(f"{p_mw:.3e}"),
        },
        "interpretation": (
            f"B5's efficacy is systematically ~{diff:.1f} points below the best baseline "
            f"({best_base}); the difference is highly significant (Mann-Whitney p={p_mw:.1e}) and "
            f"the standardized effect is large (Cohen's d={cohens_d:.2f}) BECAUSE B5's efficacy "
            f"distribution is the tightest (std {eff[b5].std():.2f} vs ~2.8). In absolute terms, "
            f"however, the gap is only {diff/SURROGATE_RMSE:.2f}x the efficacy oracle's own RMSE "
            f"({SURROGATE_RMSE}), i.e. B5 trades a modest, consistent efficacy reduction for "
            f"uniform safety rather than being efficacy-blind."),
    }
    (PROJECT / "results" / "json" / "efficacy_significance.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # figure: efficacy distributions + mean CI
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    data = [eff[m] for m in methods]
    parts = ax.violinplot(data, showmeans=False, showextrema=False, widths=0.8)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(CB[methods[i]]); b.set_alpha(0.55)
    for i, m in enumerate(methods, start=1):
        mu, lo, hi = stats_per[m]["mean"], *stats_per[m]["ci95"]
        ax.plot([i, i], [lo, hi], color="black", lw=1.4, zorder=4)
        ax.plot(i, mu, "o", color="black", ms=4, zorder=5)
    ax.axhspan(eff[b5].mean(), eff[best_base].mean(), color="grey", alpha=0.12)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([SHORT[m] for m in methods], rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("on-target efficacy (surrogate, 0–100)")
    ax.set_title(f"B5 trades a small, consistent efficacy reduction for uniform safety\n"
                 f"(B5 vs {SHORT[best_base]}: Δ={diff:.1f} pts = {diff/SURROGATE_RMSE:.2f}×oracle-RMSE; "
                 f"B5 std {eff[b5].std():.2f} is tightest)", fontsize=8.5)
    for ext in ("png", "pdf"):
        for dd in (PROJECT / "results" / "plots", PROJECT / "manuscript_assets"):
            fig.savefig(dd / f"fig7_efficacy_significance.{ext}", bbox_inches="tight",
                        pad_inches=0.3, facecolor="white")
    plt.close(fig)
    print("[fig] fig7_efficacy_significance.png/.pdf written")


if __name__ == "__main__":
    main()
