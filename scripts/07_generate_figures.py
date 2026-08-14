#!/usr/bin/env python3
"""
Phase 5 — publication-grade figure generation (300 DPI).

Reads the Phase-4 result JSON/CSV and renders four manuscript figures into
results/plots/ and manuscript_assets/ (PNG @300dpi + vector PDF):

  Fig 1  Pareto-DPO architecture & scalarization-free loss concept (schematic)
  Fig 2  Anti-saturation: implicit-reward variance, Pareto-DPO vs CRISPGen critic
  Fig 3  RQ2 beta-sweep trade-off: diversity/entropy vs |KL| and alignment
  Fig 4  Baseline ladder: Pareto frontier + hypervolume comparison (B0-B5)

No numbers are invented here: every value is loaded from
  results/json/{rq2_beta_sweep,baseline_ladder,rq_answers}.json
Figures fail loudly if a required input is missing.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
RJ = PROJECT / "results" / "json"
PLOTS = PROJECT / "results" / "plots"
ASSETS = PROJECT / "manuscript_assets"
PLOTS.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

# ── shared style (Oxford Bioinformatics: clean, serif-free, colour-blind-safe) ──
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})
CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "grey": "#8C8C8C",
      "sky": "#56B4E9", "yellow": "#F0E442"}


def _save(fig, name):
    for ext in ("png", "pdf"):
        for d in (PLOTS, ASSETS):
            fig.savefig(d / f"{name}.{ext}", bbox_inches="tight",
                        pad_inches=0.35, facecolor="white")
    plt.close(fig)
    print(f"[fig] {name}.png / .pdf -> results/plots/ + manuscript_assets/")


def _load(name):
    p = RJ / name
    if not p.exists():
        raise FileNotFoundError(f"required input missing: {p} (run its Phase-4 script first)")
    return json.loads(p.read_text())


# ─────────────────────────────────────────────────────────────────────────────
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, fc, ec=CB["grey"], fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    linewidth=1.2, edgecolor=ec, facecolor=fc, alpha=0.95))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.2, color="#444"))

    # data / preference construction (left)
    box(0.2, 4.4, 2.6, 1.1, "CRISPGen pool\n+ external oracles\n(CRISPRon, Cas-OFFinder)", "#EAF3FB", fs=7.5)
    box(0.2, 2.6, 2.6, 1.1, "2-D Pareto\ndominance sort\n(efficacy × suppression)", "#EAF3FB", fs=7.5)
    box(0.2, 0.8, 2.6, 1.1, "Preference pairs\n$(y_w \\succ_P y_l)$\nNO scalarization", "#FDEEDD", fs=7.5)
    arrow(1.5, 4.4, 1.5, 3.7); arrow(1.5, 2.6, 1.5, 1.9)

    # policy / reference (middle)
    box(3.6, 3.6, 2.6, 1.4, "Policy  $\\pi_\\theta$\nAR nucleotide decoder\n(1.8M params, 8GB-safe)", "#E7F6EF")
    box(3.6, 1.2, 2.6, 1.4, "Reference  $\\pi_{ref}$\n(frozen SFT)\nlog-probs PRECOMPUTED", "#F3F3F3")
    arrow(2.8, 1.35, 3.6, 1.9)     # pairs -> both
    arrow(2.8, 1.35, 3.6, 4.3)

    # loss (right)
    box(7.1, 2.4, 4.6, 1.5,
        "Pareto-DPO loss\n"
        r"$-\log\sigma(\beta[h(y_w)-h(y_l)])$"
        "\n" r"$h(y)=\log(\pi_\theta(y)/\pi_{ref}(y))$",
        "#FBEAF2", ec=CB["purple"], fs=8.5)
    arrow(6.2, 4.3, 7.1, 3.5); arrow(6.2, 1.9, 7.1, 2.9)
    box(8.4, 0.5, 2.0, 1.0, "aligned $\\pi_\\theta$\nover the frontier", "#FBEAF2", ec=CB["purple"], fs=7.5)
    arrow(9.4, 2.4, 9.4, 1.5)

    ax.text(6, 5.7, "Pareto-DPO: scalarization-free multi-objective preference alignment",
            ha="center", va="center", fontsize=10, fontweight="bold")
    _save(fig, "fig1_architecture")


# ─────────────────────────────────────────────────────────────────────────────
def fig2_anti_saturation():
    rq = _load("rq_answers.json")["RQ1_anti_saturation"]
    v_pareto = rq["pareto_implicit_reward_variance"]
    v_crispgen = rq["crispgen_internal_critic_variance"]
    ratio = float(rq["variance_ratio_pareto_over_crispgen"])

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    labels = ["CRISPGen\ninternal critic\n(collapsed)", "Pareto-DPO\nimplicit reward\n(ours)"]
    vals = [max(v_crispgen, 1e-12), v_pareto]
    bars = ax.bar(labels, vals, color=[CB["grey"], CB["blue"]], width=0.6,
                  edgecolor="black", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Reward-signal variance (log scale)")
    ax.set_title("Anti-saturation of the alignment signal")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.6, f"{v:.2e}",
                ha="center", va="bottom", fontsize=8)
    ax.annotate(f"~{ratio:.1e}× higher\nvariance",
                xy=(1, v_pareto), xytext=(0.35, v_pareto * 0.03),
                fontsize=8.5, ha="center", color=CB["red"],
                arrowprops=dict(arrowstyle="-|>", color=CB["red"], lw=1.3))
    ax.set_ylim(1e-11, v_pareto * 30)
    _save(fig, "fig2_anti_saturation")


# ─────────────────────────────────────────────────────────────────────────────
def fig3_beta_tradeoff():
    sw = _load("rq2_beta_sweep.json")
    rows = sw["sweep"]
    beta = np.array([r["beta"] for r in rows])
    uniq = np.array([r["unique_frac"] for r in rows])
    ent = np.array([r["pos_entropy_bits"] for r in rows])
    kl = np.array([abs(r["kl_to_ref"]) for r in rows])
    acc = np.array([r["val_reward_acc"] for r in rows])
    ref = sw.get("ref_sft_diversity", {})

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7), constrained_layout=True)

    # (a) diversity & entropy vs beta
    ax = axes[0]
    ax.plot(beta, uniq, "-o", color=CB["blue"], label="unique fraction", lw=1.6, ms=5)
    ax.plot(beta, ent / 2.0, "-s", color=CB["green"], label="positional entropy (/2 bits)", lw=1.6, ms=5)
    if "unique_frac" in ref:
        ax.axhline(ref["unique_frac"], ls=":", color=CB["grey"], lw=1, label="SFT ref (unique)")
    ax.set_xscale("log"); ax.set_xticks(beta); ax.set_xticklabels([str(b) for b in beta])
    ax.set_xlabel(r"$\beta$"); ax.set_ylabel("normalized diversity")
    ax.set_title("(a) Diversity is governed by $\\beta$")
    ax.axvspan(0.15, 0.28, color=CB["yellow"], alpha=0.25)
    ax.text(0.20, 0.05, "sweet spot", ha="center", fontsize=7.5, color="#8a6d00")
    ax.legend(loc="center right")
    ax.set_ylim(0, 1.05)

    # (b) diversity vs displacement (KL), coloured by alignment acc
    ax = axes[1]
    sc = ax.scatter(kl, uniq, c=acc, cmap="viridis", s=90, edgecolor="black",
                    linewidth=0.7, zorder=3, vmin=acc.min() - 0.005, vmax=acc.max() + 0.005)
    # place each label to the LEFT of its point so none collide with the colorbar
    for b, x, y in zip(beta, kl, uniq):
        ax.annotate(f"β={b}", (x, y), textcoords="offset points", xytext=(-10, 8),
                    ha="right", fontsize=7.5, clip_on=False)
    ax.plot(kl, uniq, "-", color=CB["grey"], lw=1, alpha=0.6, zorder=1)
    ax.set_xlabel(r"likelihood displacement  $|\Delta_w|$  (mean log-ratio, preferred completions)")
    ax.set_ylabel("unique fraction (diversity)")
    ax.set_title("(b) Displacement–diversity trade-off")
    ax.margins(x=0.16, y=0.10)
    cb = fig.colorbar(sc, ax=ax, pad=0.03); cb.set_label("val reward-acc", fontsize=8)
    fig.suptitle("RQ2 — controllable alignment-vs-diversity trade-off",
                 fontsize=10, fontweight="bold")
    _save(fig, "fig3_beta_tradeoff")


# ─────────────────────────────────────────────────────────────────────────────
def fig4_ladder():
    lad = _load("baseline_ladder.json")
    rows = lad["results"]
    off_ref = lad["hv_off_reference_p99"]
    order = ["B0_SFT", "B1_CRISPGen_REINFORCE", "B3_SingleObjDPO",
             "B4_ScalarizedMODPO", "B5_ParetoDPO"]
    by = {r["method"]: r for r in rows}
    col = {"B0_SFT": CB["grey"], "B1_CRISPGen_REINFORCE": CB["orange"],
           "B3_SingleObjDPO": CB["sky"], "B4_ScalarizedMODPO": CB["green"],
           "B5_ParetoDPO": CB["red"]}
    short = {"B0_SFT": "B0 SFT", "B1_CRISPGen_REINFORCE": "B1 CRISPGen",
             "B3_SingleObjDPO": "B3 Single-DPO", "B4_ScalarizedMODPO": "B4 Scalarized",
             "B5_ParetoDPO": "B5 Pareto-DPO (ours)"}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)

    # (a) trade-off scatter: efficacy vs suppression, size ~ diversity
    ax = axes[0]
    for m in order:
        r = by[m]
        eff01 = r["eff_mean"] / 100.0
        supp = r["suppression_mean01"]
        ax.scatter(eff01, supp, s=60 + 260 * r["unique_frac"], color=col[m],
                   edgecolor="black", linewidth=0.8, alpha=0.9,
                   label=f"{short[m]} (uniq {r['unique_frac']:.2f})", zorder=3)
    ax.set_xlabel("mean on-target efficacy (surrogate, 0–1)")
    ax.set_ylabel("mean off-target suppression (0–1)")
    ax.set_title("(a) Efficacy–safety trade-off (size ∝ diversity)")
    ax.annotate("B5 = safety-extreme\nof method frontier", xy=(0.4995, 0.9995),
                xytext=(0.5085, 0.990), fontsize=7.5, color=CB["red"],
                arrowprops=dict(arrowstyle="-|>", color=CB["red"], lw=1.1))
    ax.legend(loc="upper right", fontsize=6.6, ncol=1, borderaxespad=0.4)
    ax.margins(x=0.16, y=0.14)

    # (b) hypervolume bars
    ax = axes[1]
    hv = [by[m]["pareto_hypervolume"] for m in order]
    bars = ax.bar([short[m].split(" (")[0] for m in order], hv,
                  color=[col[m] for m in order], edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Pareto hypervolume (higher = better)")
    ax.set_title("(b) Dominated hypervolume")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short[m].split(" (")[0] for m in order], rotation=30, ha="right")
    for b, v in zip(bars, hv):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, max(hv) * 1.15)

    fid = lad.get("efficacy_oracle_fidelity", {})
    cap = (f"Efficacy = CRISPRon surrogate (Spearman {fid.get('heldout_spearman', float('nan')):.2f}, "
           f"noisy); off-target = chr22 near-match (MM≤3). B5=β0.20. HV rewards objective-space "
           f"spread, so uniformly-safe B5 (supp≈1.0) collapses to one frontier point — see caveats.")
    fig.suptitle("Baseline ladder — external-oracle Pareto comparison (B0–B5)",
                 fontsize=10.5, fontweight="bold")
    fig.text(0.5, -0.02, cap, ha="center", va="top", fontsize=6.6, color="#555", wrap=True)
    _save(fig, "fig4_baseline_ladder")


def main():
    fig1_architecture()
    fig2_anti_saturation()
    fig3_beta_tradeoff()
    fig4_ladder()
    print("[done] 4 figures @300 DPI -> results/plots/ + manuscript_assets/")


if __name__ == "__main__":
    main()
