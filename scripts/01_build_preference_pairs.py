#!/usr/bin/env python3
"""
Phase 1 — Pareto preference-pair construction  (RFC-001 §3, scalarization-free).

Builds winner/loser preference pairs from the 2-D Pareto-dominance partial order over
two EXTERNAL, non-circular objectives:

    O1  s_eff : on-target efficacy      = CRISPRon score (0-100)          [maximize]
    O2  s_off : off-target suppression  = -(whole-genome near-match hits) [maximize]

Neither objective is the saturable internal CRISPGen critic (`eff`/`disc`), so the
preference signal is empirical/structural, not something the policy can hack (RFC §1).

Splits produced (written to data/preference_pairs/ as CSV + a JSON data card):
  * train.csv : dominance pairs among TRAIN guides, ID objective (all chroms except OOD)
  * val.csv   : dominance pairs among VAL   guides, ID objective
  * ood.csv   : dominance pairs among ALL   guides, OOD objective (held-out chromosome
                only) -> a genuine off-target distribution shift for RQ3

Runnable now on the 1,000-guide pilot (the only subset with BOTH CRISPRon and a
whole-genome off-target screen). Scaling to the 300k/3M pool requires computing the
off-target screen for those sequences first; the O(N^2) dominance step here is fine for
the pilot and must be replaced by the NSGA-II fronts + blocked pair emission at scale
(both already implemented below).
"""
from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

NOTEBOOK = Path(__file__).resolve().parents[2]                     # .../notebook
PROJECT = Path(__file__).resolve().parents[1]                     # .../ParetoDPO_Genomics
OUTDIR = PROJECT / "data" / "preference_pairs"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load the scored candidate pool (external objectives only)
# ─────────────────────────────────────────────────────────────────────────────
def load_pilot_pool(max_mm: int, ood_chrom: str) -> pd.DataFrame:
    """Return per-guide DataFrame: id, seq, s_eff, s_off_id, s_off_ood, off_id, off_ood."""
    cris_path = glob.glob(str(NOTEBOOK / "output_result_final*crispron.csv"))
    assert cris_path, "CRISPRon file not found next to the notebook"
    cris = pd.read_csv(cris_path[0])                              # ID, 30mer, CRISPRon
    cris["id"] = cris["ID"].str.rsplit("_p_", n=1).str[0]
    s_eff = cris.groupby("id")["CRISPRon"].mean().rename("s_eff")

    raw = pd.read_csv(NOTEBOOK / "report" / "whole_genome_hits_raw.csv")
    mm_cols = [f"mm{m}" for m in range(0, max_mm + 1)]
    raw["near"] = raw[mm_cols].sum(axis=1)                        # near-match count <= max_mm

    off_id = (raw.loc[raw["chrom"] != ood_chrom]
                 .groupby("guide_id")["near"].sum().rename("off_id"))
    off_ood = (raw.loc[raw["chrom"] == ood_chrom]
                  .groupby("guide_id")["near"].sum().rename("off_ood"))
    seq = raw.groupby("guide_id")["seq"].first().rename("seq")

    df = pd.concat([seq, s_eff, off_id, off_ood], axis=1)
    # guides absent from a chromosome contribute 0 near-matches there
    df["off_id"] = df["off_id"].fillna(0.0)
    df["off_ood"] = df["off_ood"].fillna(0.0)
    df = df.dropna(subset=["s_eff", "seq"]).reset_index().rename(columns={"index": "id"})
    df["s_off_id"] = -df["off_id"]                               # fewer hits -> higher suppression
    df["s_off_ood"] = -df["off_ood"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pareto machinery (maximize both objectives)
# ─────────────────────────────────────────────────────────────────────────────
def fast_non_dominated_sort(obj: np.ndarray) -> np.ndarray:
    """NSGA-II fronts. obj shape (N, K), maximization. Returns front index per point."""
    n = obj.shape[0]
    dominates = _dominance_matrix(obj)                            # dominates[i,j]=i≻j
    dom_count = dominates.sum(axis=0)                             # how many dominate i
    front = np.full(n, -1, dtype=int)
    current = np.where(dom_count == 0)[0]
    f = 0
    while current.size:
        front[current] = f
        nxt = []
        for i in current:
            dominated = np.where(dominates[i])[0]
            dom_count[dominated] -= 1
            nxt.extend(dominated[dom_count[dominated] == 0].tolist())
        current = np.unique(np.array(nxt, dtype=int)) if nxt else np.array([], dtype=int)
        f += 1
    return front


def _dominance_matrix(obj: np.ndarray) -> np.ndarray:
    """Boolean (N,N): entry [i,j] True iff point i Pareto-dominates point j (maximize).
    O(N^2 K) memory/time - fine for the pilot (N~1e3); stream in blocks at scale."""
    ge = (obj[:, None, :] >= obj[None, :, :]).all(axis=2)         # i>=j on all objectives
    gt = (obj[:, None, :] > obj[None, :, :]).any(axis=2)          # i>j on some objective
    dom = ge & gt
    np.fill_diagonal(dom, False)
    return dom


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pair emission: dominance pairs + hard-negative mining + stratified coverage
# ─────────────────────────────────────────────────────────────────────────────
def build_pairs(df: pd.DataFrame, eff_col: str, off_col: str, *,
                max_pairs: int, fanout_cap: int, n_bins: int,
                seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    obj = df[[eff_col, off_col]].to_numpy(float)
    n = len(df)
    dom = _dominance_matrix(obj)
    w_idx, l_idx = np.where(dom)                                  # every winner≻loser pair
    if w_idx.size == 0:
        return pd.DataFrame(columns=["x", "y_w", "y_l", "s_eff_w", "s_eff_l",
                                     "s_off_w", "s_off_l", "margin"])

    # normalized dominance margin (smaller = harder negative)
    span = np.clip(obj.max(0) - obj.min(0), 1e-9, None)
    d_eff = (obj[w_idx, 0] - obj[l_idx, 0]) / span[0]
    d_off = (obj[w_idx, 1] - obj[l_idx, 1]) / span[1]
    margin = 0.5 * (d_eff + d_off)

    # hard-negative bias: sample pairs with prob ∝ 1/(margin+eps), then stratify winners
    order = np.argsort(margin)                                    # hardest first
    w_idx, l_idx, margin = w_idx[order], l_idx[order], margin[order]

    # stratified coverage across efficacy × suppression bins of the WINNER
    eff_b = np.digitize(obj[w_idx, 0], np.quantile(obj[:, 0], np.linspace(0, 1, n_bins + 1)[1:-1]))
    off_b = np.digitize(obj[w_idx, 1], np.quantile(obj[:, 1], np.linspace(0, 1, n_bins + 1)[1:-1]))
    strata = eff_b * (n_bins + 1) + off_b

    keep = np.zeros(w_idx.size, dtype=bool)
    per_winner: dict[int, int] = {}
    # round-robin over strata so no region floods the loss
    by_stratum: dict[int, list[int]] = {}
    for k, s in enumerate(strata):
        by_stratum.setdefault(int(s), []).append(k)
    strata_keys = list(by_stratum)
    rng.shuffle(strata_keys)
    picked = 0
    exhausted = set()
    while picked < max_pairs and len(exhausted) < len(strata_keys):
        for s in strata_keys:
            if s in exhausted:
                continue
            bucket = by_stratum[s]
            advanced = False
            while bucket:
                k = bucket.pop(0)
                w = int(w_idx[k])
                if per_winner.get(w, 0) >= fanout_cap:
                    continue
                keep[k] = True
                per_winner[w] = per_winner.get(w, 0) + 1
                picked += 1
                advanced = True
                break
            if not bucket:
                exhausted.add(s)
            if picked >= max_pairs:
                break
        if not any(s not in exhausted for s in strata_keys):
            break

    sel = np.where(keep)[0]
    ids = df["id"].to_numpy()
    seqs = df["seq"].to_numpy()
    out = pd.DataFrame({
        "x": "",                                                 # unconditional (Decision 4)
        "y_w": seqs[w_idx[sel]], "y_l": seqs[l_idx[sel]],
        "id_w": ids[w_idx[sel]], "id_l": ids[l_idx[sel]],
        "s_eff_w": obj[w_idx[sel], 0], "s_eff_l": obj[l_idx[sel], 0],
        "s_off_w": obj[w_idx[sel], 1], "s_off_l": obj[l_idx[sel], 1],
        "margin": margin[sel],
    })
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Driver
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="pilot", choices=["pilot"],
                    help="scored candidate pool (pilot = 1000 guides w/ CRISPRon + WG screen)")
    ap.add_argument("--max-mm", type=int, default=3, help="max mismatch counted as near-match")
    ap.add_argument("--ood-chrom", default="chr22", help="held-out chromosome for OOD split")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-pairs", type=int, default=40000, help="cap per split")
    ap.add_argument("--fanout-cap", type=int, default=12, help="max losers per winner")
    ap.add_argument("--bins", type=int, default=6, help="strata per objective")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    df = load_pilot_pool(args.max_mm, args.ood_chrom)
    print(f"[pool] {len(df)} guides | s_eff(CRISPRon) "
          f"[{df.s_eff.min():.1f},{df.s_eff.max():.1f}] mean {df.s_eff.mean():.2f} | "
          f"off_id(near<= MM{args.max_mm}) mean {df.off_id.mean():.1f} max {int(df.off_id.max())} | "
          f"off_ood({args.ood_chrom}) mean {df.off_ood.mean():.2f} max {int(df.off_ood.max())}")

    fronts = fast_non_dominated_sort(df[["s_eff", "s_off_id"]].to_numpy(float))
    n_front = int((fronts == 0).sum())
    print(f"[pareto] ID front |F_1| = {n_front} non-dominated guides "
          f"({100*n_front/len(df):.1f}%); {fronts.max()+1} fronts total")

    # guide-level train/val split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(df))
    n_val = int(round(args.val_frac * len(df)))
    val_ids = set(df.iloc[perm[:n_val]]["id"])
    train_df = df[~df["id"].isin(val_ids)].reset_index(drop=True)
    val_df = df[df["id"].isin(val_ids)].reset_index(drop=True)

    splits = {
        "train": build_pairs(train_df, "s_eff", "s_off_id", max_pairs=args.max_pairs,
                             fanout_cap=args.fanout_cap, n_bins=args.bins, seed=args.seed),
        "val":   build_pairs(val_df, "s_eff", "s_off_id", max_pairs=args.max_pairs,
                             fanout_cap=args.fanout_cap, n_bins=args.bins, seed=args.seed + 1),
        "ood":   build_pairs(df, "s_eff", "s_off_ood", max_pairs=args.max_pairs,
                             fanout_cap=args.fanout_cap, n_bins=args.bins, seed=args.seed + 2),
    }

    card = {"source": args.source, "seed": args.seed, "max_mm": args.max_mm,
            "ood_chrom": args.ood_chrom, "n_guides": int(len(df)),
            "id_front_size": n_front, "objectives": {
                "s_eff": "CRISPRon (external, 0-100), maximize",
                "s_off": f"-(near-match hits MM<= {args.max_mm}) from whole-genome screen, maximize"},
            "splits": {}}
    for name, pdf in splits.items():
        path = OUTDIR / f"{name}.csv"
        pdf.to_csv(path, index=False)
        card["splits"][name] = {"n_pairs": int(len(pdf)),
                                "n_unique_winners": int(pdf["id_w"].nunique()) if len(pdf) else 0,
                                "mean_margin": float(pdf["margin"].mean()) if len(pdf) else None,
                                "file": str(path.relative_to(PROJECT))}
        print(f"[{name}] {len(pdf):>6} pairs | "
              f"{card['splits'][name]['n_unique_winners']} unique winners | "
              f"mean margin {card['splits'][name]['mean_margin']}")

    (OUTDIR / "data_card.json").write_text(json.dumps(card, indent=2))
    print(f"[done] wrote {OUTDIR.relative_to(PROJECT)}/(train|val|ood).csv + data_card.json")


if __name__ == "__main__":
    main()
