#!/usr/bin/env python3
"""
Rigorous multi-objective evaluation beyond a single hypervolume scalar (reviewer concern #1).

A reviewer correctly notes that reporting one empirical hypervolume — on which Pareto-DPO (B5)
ranks last — is a fragile basis for a Pareto-superiority claim. We therefore add three standard,
complementary set-based analyses computed from the *per-guide* scored pools dumped by the
baseline ladder (results/npz/ladder_pools.npz):

  1. Two-set coverage (C-metric, Zitzler & Thiele 1998): C(A,B) = fraction of B's points weakly
     dominated by at least one point of A. This measures actual Pareto dominance BETWEEN method
     pools and is invariant to objective-space spread (unlike hypervolume).
  2. Bootstrapped 95% CIs for the hypervolume, to test whether the HV differences among methods
     are statistically distinguishable at all given finite-pool sampling noise.
  3. Generational Distance (GD) and Inverted GD (IGD) to a reference front built from the union
     of all methods' non-dominated points.

We report whatever these show. Objectives here live in the (efficacy, suppression) PROXY space;
their real-world fidelity is bounded by the oracles (see offtarget_oracle_validation and the
manuscript Discussion) — these metrics establish dominance structure, not wet-lab superiority.

Outputs: results/json/pareto_metrics.json,
         results/plots/fig5_pareto_metrics.png (+ manuscript_assets, +pdf)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
NPZ = PROJECT / "results" / "npz" / "ladder_pools.npz"


def hypervolume_2d(eff01: np.ndarray, supp01: np.ndarray, ref=(0.0, 0.0)) -> float:
    """Dominated hypervolume of the non-dominated set of (eff01, supp01) above `ref`.
    Identical to the implementation used by the baseline ladder (scripts/06)."""
    pts = np.stack([eff01, supp01], 1)
    pts = pts[(pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(-pts[:, 0])]                  # eff descending
    nd, best_supp = [], ref[1]
    for e, s in pts:
        if s > best_supp:
            nd.append((e, s)); best_supp = s
    hv, prev_supp = 0.0, ref[1]
    for e, s in sorted(nd, key=lambda t: t[1]):        # supp ascending
        hv += (e - ref[0]) * (s - prev_supp)
        prev_supp = s
    return float(hv)

CB = {"B0_SFT": "#8C8C8C", "B1_CRISPGen_REINFORCE": "#E69F00", "B3_SingleObjDPO": "#56B4E9",
      "B4_ScalarizedMODPO": "#009E73", "B5_ParetoDPO": "#D55E00"}
SHORT = {"B0_SFT": "B0 SFT", "B1_CRISPGen_REINFORCE": "B1 REINFORCE",
         "B3_SingleObjDPO": "B3 Single-DPO", "B4_ScalarizedMODPO": "B4 Scalarized",
         "B5_ParetoDPO": "B5 Pareto-DPO"}
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.family": "DejaVu Sans",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "legend.frameon": False})


def points(eff, off, off_ref):
    eff01 = np.clip(eff / 100.0, 0, 1)
    supp01 = np.clip(1.0 - off / off_ref, 0, 1)
    return np.stack([eff01, supp01], axis=1)


def coverage(A, B):
    """C(A,B): fraction of B weakly dominated by >=1 point of A (maximization)."""
    if len(B) == 0:
        return 0.0
    ge = (A[:, None, 0] >= B[None, :, 0]) & (A[:, None, 1] >= B[None, :, 1])
    gt = (A[:, None, 0] > B[None, :, 0]) | (A[:, None, 1] > B[None, :, 1])
    dominated = (ge & gt).any(axis=0)         # for each b, any a dominating it
    return float(dominated.mean())


def nondominated(P):
    keep = np.ones(len(P), bool)
    for i in range(len(P)):
        if not keep[i]:
            continue
        dom = (P[:, 0] >= P[i, 0]) & (P[:, 1] >= P[i, 1]) & \
              ((P[:, 0] > P[i, 0]) | (P[:, 1] > P[i, 1]))
        if dom.any():
            keep[i] = False
    return P[keep]


def gd_igd(P, ref):
    d_P_to_ref = np.sqrt(((P[:, None, :] - ref[None, :, :]) ** 2).sum(2)).min(1)
    d_ref_to_P = np.sqrt(((ref[:, None, :] - P[None, :, :]) ** 2).sum(2)).min(1)
    return float(np.sqrt((d_P_to_ref ** 2).mean())), float(np.sqrt((d_ref_to_P ** 2).mean()))


def main(n_boot=2000, seed=0):
    d = np.load(NPZ, allow_pickle=True)
    methods = [str(m) for m in d["methods"]]
    off_ref = float(d["off_ref"])
    P = {m: points(d[f"{m}__eff"].astype(float), d[f"{m}__off"].astype(float), off_ref)
         for m in methods}
    rng = np.random.default_rng(seed)

    # 1. coverage matrix
    C = {a: {b: round(coverage(P[a], P[b]), 4) for b in methods} for a in methods}
    # B5 domination summary: is any B5 point dominated by another method?
    b5 = "B5_ParetoDPO"
    dominated_by_others = {m: round(coverage(P[m], P[b5]), 4) for m in methods if m != b5}

    # 2. bootstrapped hypervolume CI
    hv_ci = {}
    for m in methods:
        pts = P[m]
        base = hypervolume_2d(pts[:, 0], pts[:, 1])
        bs = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, len(pts), len(pts))
            bs[i] = hypervolume_2d(pts[idx, 0], pts[idx, 1])
        hv_ci[m] = {"hv": round(float(base), 4),
                    "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                             round(float(np.percentile(bs, 97.5)), 4)],
                    "boot_mean": round(float(bs.mean()), 4),
                    "boot_std": round(float(bs.std()), 4)}

    # 3. GD / IGD to union reference front
    ref = nondominated(np.concatenate([P[m] for m in methods], axis=0))
    gi = {m: dict(zip(("gd", "igd"), (round(x, 5) for x in gd_igd(P[m], ref)))) for m in methods}

    res = {
        "note": "objectives are PROXY (efficacy surrogate, chr22 MM<=3 suppression); "
                "metrics establish dominance structure, not wet-lab superiority.",
        "coverage_C_row_dominates_col": C,
        "b5_coverage_by_each_other_method_C(other,B5)": dominated_by_others,
        "hypervolume_bootstrap": hv_ci,
        "generational_distance": gi,
        "reference_front_size": int(len(ref)),
    }
    (PROJECT / "results" / "json" / "pareto_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.8), constrained_layout=True)

    # (a) suppression distribution per method
    supp = [P[m][:, 1] for m in methods]
    parts = ax[0].violinplot(supp, showmeans=True, showextrema=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(CB[methods[i]]); b.set_alpha(0.7)
    ax[0].set_xticks(range(1, len(methods) + 1))
    ax[0].set_xticklabels([SHORT[m] for m in methods], rotation=30, ha="right", fontsize=7)
    ax[0].set_ylabel("off-target suppression (0–1)")
    ax[0].set_title("(a) Distribution of safety\n(higher = safer)")

    # (b) coverage heatmap
    M = np.array([[C[a][b] for b in methods] for a in methods])
    im = ax[1].imshow(M, cmap="Reds", vmin=0, vmax=max(M.max(), 0.01))
    ax[1].set_xticks(range(len(methods))); ax[1].set_yticks(range(len(methods)))
    ax[1].set_xticklabels([SHORT[m] for m in methods], rotation=40, ha="right", fontsize=6.5)
    ax[1].set_yticklabels([SHORT[m] for m in methods], fontsize=6.5)
    ax[1].set_xlabel("… dominates this method"); ax[1].set_ylabel("this method …")
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax[1].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                       fontsize=6.5, color="black" if M[i, j] < 0.5*M.max()+1e-9 else "white")
    ax[1].set_title("(b) Two-set coverage $C(A,B)$")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    # (c) hypervolume with bootstrap 95% CI
    xs = np.arange(len(methods))
    hvs = [hv_ci[m]["hv"] for m in methods]
    los = [hv_ci[m]["hv"] - hv_ci[m]["ci95"][0] for m in methods]
    his = [hv_ci[m]["ci95"][1] - hv_ci[m]["hv"] for m in methods]
    ax[2].bar(xs, hvs, color=[CB[m] for m in methods], edgecolor="black", linewidth=0.7,
              yerr=[los, his], capsize=4, error_kw=dict(lw=1.1))
    ax[2].set_xticks(xs); ax[2].set_xticklabels([SHORT[m] for m in methods],
                                                rotation=30, ha="right", fontsize=7)
    ax[2].set_ylabel("hypervolume (95% bootstrap CI)")
    lo = min(hv_ci[m]["ci95"][0] for m in methods)
    hi = max(hv_ci[m]["ci95"][1] for m in methods)
    ax[2].set_ylim(lo - 0.02, hi + 0.02)
    ax[2].set_title("(c) Hypervolume gap is small\nand efficacy-driven")

    fig.suptitle("Set-based Pareto evaluation of the ladder — proxy objective space is "
                 f"degenerate (union front = {len(ref)} point)",
                 fontsize=10.5, fontweight="bold")
    for ext in ("png", "pdf"):
        for dd in (PROJECT / "results" / "plots", PROJECT / "manuscript_assets"):
            fig.savefig(dd / f"fig5_pareto_metrics.{ext}", bbox_inches="tight",
                        pad_inches=0.3, facecolor="white")
    plt.close(fig)
    print("[fig] fig5_pareto_metrics.png/.pdf written")


if __name__ == "__main__":
    main()
