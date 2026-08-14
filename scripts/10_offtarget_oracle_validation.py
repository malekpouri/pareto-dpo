#!/usr/bin/env python3
"""
Empirical grounding of the off-target *suppression* objective (reviewer concern #2).

The safety objective used to define Pareto dominance is a STRUCTURAL near-match count
(guide vs. genomic site, Hamming distance in the protospacer). A reviewer rightly asks
whether that structural proxy actually tracks *experimentally validated* off-target
cleavage. Here we test exactly that against wet-lab data:

  * CIRCLE-seq (Tsai et al.): 584,949 (sgRNA, off-target-site) pairs, 7,371 labelled as
    validated cleavage sites (label=1) vs. 577,578 negatives (label=0).

For each pair we compute the position-wise mismatch count between the 23/24-nt aligned
guide and candidate site (the same structural quantity that underlies the suppression
objective) and ask how well a low mismatch count predicts a *validated* off-target. If the
structural proxy is well grounded, (i) validated off-targets concentrate at low mismatch,
and (ii) the score −(mismatch count) discriminates label with high ROC-AUC.

This validates the ORACLE (the safety signal), not individual generated guides — novel
guides have no wet-lab label by construction; the point is that the signal Pareto-DPO
optimises is real.

Outputs: results/json/offtarget_oracle_validation.json,
         results/plots/fig6_offtarget_oracle_validation.png (+ manuscript_assets, +pdf)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = Path(__file__).resolve().parents[2]
CIRCLE = NOTEBOOK.parent / "I_1_CIRCLE_seq_10gRNA_wholeDataset.csv"

CB = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "grey": "#8C8C8C", "purple": "#CC79A7"}
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.family": "DejaVu Sans",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "legend.frameon": False})


def mismatch_counts(guides: np.ndarray, offs: np.ndarray) -> np.ndarray:
    """Position-wise mismatch count over aligned fixed-length guide/off-target strings.
    Non-nucleotide alignment markers (e.g. '_') and the PAM 'N' are ignored so that only
    protospacer base–base disagreements are counted."""
    IGNORE = set("_-N ")
    out = np.empty(len(guides), dtype=np.int16)
    for i, (g, o) in enumerate(zip(guides, offs)):
        m = 0
        for a, b in zip(g, o):
            if a in IGNORE or b in IGNORE:
                continue
            if a != b:
                m += 1
        out[i] = m
    return out


def main():
    df = pd.read_csv(CIRCLE)
    g = df["sgRNA_seq"].str.upper().to_numpy()
    o = df["off_seq"].str.upper().to_numpy()
    y = (df["label"].to_numpy() > 0).astype(int)
    mm = mismatch_counts(g, o)

    # score used by the suppression objective: fewer mismatches => more likely a real cut
    score = -mm.astype(float)
    auc = roc_auc_score(y, score)
    ap = average_precision_score(y, score)
    prevalence = float(y.mean())

    # how the near-match threshold (MM<=k, the screen used in-silico) captures validated sites
    recall_at = {k: float(((mm <= k) & (y == 1)).sum() / (y == 1).sum()) for k in (3, 4, 5, 6)}
    # precision of the MM<=3 screen (of sites it flags, how many are validated)
    flagged3 = (mm <= 3)
    prec_at3 = float(((mm <= 3) & (y == 1)).sum() / max(flagged3.sum(), 1))

    pos_mm, neg_mm = mm[y == 1], mm[y == 0]
    res = {
        "dataset": "CIRCLE-seq (Tsai et al.), whole dataset",
        "n_pairs": int(len(df)), "n_validated": int(y.sum()),
        "n_negative": int((y == 0).sum()), "prevalence": round(prevalence, 5),
        "roc_auc_neg_mismatch_predicts_validated": round(float(auc), 4),
        "average_precision": round(float(ap), 4),
        "validated_mismatch_mean": round(float(pos_mm.mean()), 3),
        "validated_mismatch_median": float(np.median(pos_mm)),
        "negative_mismatch_mean": round(float(neg_mm.mean()), 3),
        "recall_of_validated_at_MMle_k": {str(k): round(v, 4) for k, v in recall_at.items()},
        "precision_of_MMle3_screen": round(prec_at3, 4),
    }
    (PROJECT / "results" / "json" / "offtarget_oracle_validation.json").write_text(
        json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.7), constrained_layout=True)
    kmax = 12
    bins = np.arange(0, kmax + 2) - 0.5
    ax[0].hist(np.clip(neg_mm, 0, kmax), bins=bins, density=True, color=CB["grey"],
               alpha=0.65, label=f"non-validated (n={ (y==0).sum():,})")
    ax[0].hist(np.clip(pos_mm, 0, kmax), bins=bins, density=True, color=CB["red"],
               alpha=0.75, label=f"validated off-target (n={y.sum():,})")
    ax[0].axvline(3.5, ls="--", color=CB["blue"], lw=1.2)
    ax[0].text(3.6, ax[0].get_ylim()[1]*0.9, "MM≤3 screen", fontsize=7.5, color=CB["blue"])
    ax[0].set_xlabel("protospacer mismatch count"); ax[0].set_ylabel("density")
    ax[0].set_title("(a) Mismatch count only weakly\nseparates validated off-targets")
    ax[0].legend(loc="upper left", fontsize=7.5)

    fpr, tpr, _ = roc_curve(y, score)
    ax[1].plot(fpr, tpr, color=CB["red"], lw=1.8, label=f"AUC = {auc:.3f}")
    ax[1].plot([0, 1], [0, 1], ls=":", color=CB["grey"], lw=1)
    ax[1].set_xlabel("false-positive rate"); ax[1].set_ylabel("true-positive rate")
    ax[1].set_title("(b) Near-match count is a\nweak standalone predictor")
    ax[1].legend(loc="lower right")
    fig.suptitle("Empirical check of the structural off-target proxy against wet-lab data "
                 "(CIRCLE-seq)", fontsize=10, fontweight="bold")
    for ext in ("png", "pdf"):
        for d in (PROJECT / "results" / "plots", PROJECT / "manuscript_assets"):
            fig.savefig(d / f"fig6_offtarget_oracle_validation.{ext}",
                        bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)
    print("[fig] fig6_offtarget_oracle_validation.png/.pdf written")


if __name__ == "__main__":
    main()
