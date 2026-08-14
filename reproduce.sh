#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Pareto-DPO — one-command reproduction of every result, figure and JSON table.
#
# Runs the full seeded pipeline end to end:
#   preferences → SFT reference → offline ref log-probs → Pareto-DPO (β=0.1)
#   → β=0.20 sweet-spot checkpoint → β-sweep (RQ2) → baseline ladder → RQ stats
#   → publication figures.
#
# Usage:   bash reproduce.sh
# Python:  override the interpreter with  PY=/path/to/python bash reproduce.sh
# Total wall-clock is ~20 min on a single RTX 5050 (8 GB); peak VRAM < 0.6 GB.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"

echo "▶ [1/9] Build 2-D Pareto-dominance preference pairs (train/val/OOD chr22)"
"$PY" scripts/01_build_preference_pairs.py

echo "▶ [2/9] Supervised fine-tune the reference decoder (π_ref)"
"$PY" scripts/03_train_sft_reference.py

echo "▶ [3/9] Pre-compute frozen reference log-probabilities offline"
"$PY" scripts/02_precompute_ref_logprobs.py --ckpt models/checkpoints/sft_ref_base.pt

echo "▶ [4/9] Train Pareto-DPO (default β=0.1)"
"$PY" scripts/04_train_pareto_dpo.py

echo "▶ [5/9] Train the β=0.20 sweet-spot checkpoint (RQ2 operating point)"
"$PY" scripts/08_train_pareto_beta020.py

echo "▶ [6/9] RQ2 temperature sweep  → results/json/rq2_beta_sweep.json"
"$PY" scripts/05_run_beta_sweep.py

echo "▶ [7/9] Baseline ladder (B0–B5) → results/json/baseline_ladder.json"
"$PY" scripts/06_run_baseline_ladder.py

echo "▶ [8/11] Consolidated RQ statistics → results/json/rq_answers.json"
"$PY" scripts/07_rq_statistics.py

echo "▶ [9/16] Set-based Pareto metrics (coverage, bootstrap HV CI, GD/IGD) → Fig 5"
"$PY" scripts/09_pareto_metrics.py

echo "▶ [10/16] Empirical off-target oracle check vs CIRCLE-seq → Fig 6"
"$PY" scripts/10_offtarget_oracle_validation.py

echo "▶ [11/16] Efficacy distribution + significance (reviewer E4) → Fig 7"
"$PY" scripts/11_efficacy_significance.py

echo "▶ [12/16] Guide-clustered OOD transfer test (reviewer E3)"
"$PY" scripts/12_ood_clustered_test.py

echo "▶ [13/16] Shuffled-preference negative control (reviewer E1) → Fig 8"
"$PY" scripts/13_shuffled_pref_control.py

echo "▶ [14/18] Baseline tuning-parity + B4 weight sweep (reviewer E5)"
"$PY" scripts/14_baseline_tuning_parity.py

echo "▶ [15/18] Off-target-proxy robustness (reviewer E6)"
"$PY" scripts/15_offtarget_proxy_robustness.py

echo "▶ [16/18] Dynamic saturation curves + memorization check → Figs 9,10"
"$PY" scripts/16_dynamic_saturation_and_memorization.py

echo "▶ [17/18] Incomparable-pair causal ablation → Fig 11"
"$PY" scripts/17_incomparable_pair_ablation.py

echo "▶ [18/18] Publication figures @300 DPI → results/plots/ + manuscript_assets/"
"$PY" scripts/07_generate_figures.py

echo "✓ Done. Tables: results/json/*.json + results/csv/*.csv | Figures: results/plots/*.png"
