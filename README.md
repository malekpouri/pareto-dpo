<h1 align="center">Pareto-DPO</h1>
<p align="center">
  <b>Scalarization-Free Multi-Objective Preference Alignment<br/>for Generative Genomic Design</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg">
  <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg">
  <img alt="VRAM" src="https://img.shields.io/badge/VRAM-%3C0.6GB%20(8GB%20GPU)-76b900.svg">
  <img alt="Reproducible" src="https://img.shields.io/badge/reproducible-one--command-8A2BE2.svg">
</p>

> **⚠️ Manuscript withheld pending publication.** This repository contains **only the software
> implementation and reproducible pipeline**. The manuscript, its figures, the numerical results,
> and the reviewer correspondence have been intentionally removed until the associated paper is
> published. This README describes the code, not the paper's findings.

## What this is

Pareto-DPO is a **preference-construction method** for aligning a generative nucleotide-sequence
model to *multiple* competing objectives (e.g. on-target efficacy and off-target suppression for
CRISPR sgRNA design) **without scalarization**. The idea is simple and the optimiser is standard:

- Score candidate sequences on each objective with **external** oracles.
- Build a preference set from the **Pareto-dominance partial order** — a pair `(y_w, y_l)` is kept
  only if `y_w` dominates `y_l`; mutually incomparable candidates are excluded.
- Feed that preference set to **standard Direct Preference Optimization (DPO)**.

No weighting of objectives is chosen at any point. The whole pipeline is designed to run on a
single consumer GPU (**< 0.6 GB peak VRAM**) via offline reference-log-prob caching, a compact
autoregressive policy, and BF16 gradient accumulation.

## Install

```bash
git clone https://github.com/malekpouri/pareto-dpo && cd pareto-dpo
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # install a CUDA torch build for your GPU
pip install -e .                          # optional: import as a package
```

> **Note.** Running the pipeline requires input datasets that are **not** included here (CRISPRon
> scores, whole-genome near-match tables, and a GRCh38 reference). See the docstrings in
> `scripts/01_build_preference_pairs.py` and `scripts/_eval_oracles.py` for the expected inputs.

## Run

Regenerate the full pipeline end to end (seeded):

```bash
bash reproduce.sh                         # or:  PY=venv/bin/python bash reproduce.sh
```

<details>
<summary>Pipeline stages (scripts/)</summary>

| Stage | Script |
|---|---|
| Preference construction (2-D Pareto dominance) | `01_build_preference_pairs.py` |
| SFT reference decoder | `03_train_sft_reference.py` |
| Offline reference log-probs | `02_precompute_ref_logprobs.py` |
| Pareto-DPO training | `04_train_pareto_dpo.py`, `08_train_pareto_beta020.py` |
| Temperature sweep | `05_run_beta_sweep.py` |
| Baseline ladder + external rescoring | `06_run_baseline_ladder.py` |
| Consolidated statistics | `07_rq_statistics.py` |
| Set-based Pareto metrics | `09_pareto_metrics.py` |
| Off-target oracle check | `10_offtarget_oracle_validation.py` |
| Efficacy significance | `11_efficacy_significance.py` |
| Guide-clustered OOD test | `12_ood_clustered_test.py` |
| Shuffled-preference control | `13_shuffled_pref_control.py` |
| Baseline tuning parity | `14_baseline_tuning_parity.py` |
| Off-target-proxy robustness | `15_offtarget_proxy_robustness.py` |
| Dynamic saturation + memorization | `16_dynamic_saturation_and_memorization.py` |
| Incomparable-pair ablation | `17_incomparable_pair_ablation.py` |
| Figure generation | `07_generate_figures.py` |

</details>

## Repository layout

```
pareto-dpo/
├── models/
│   ├── policy_decoder.py     autoregressive nucleotide policy + tokenizer
│   └── pareto_dpo_loss.py    scalarization-free DPO loss (unit-tested)
├── scripts/                  numbered, seeded pipeline (01 → 17) + helpers
│   ├── _common.py            shared DPO train/eval + diversity metrics
│   └── _eval_oracles.py      external efficacy surrogate + off-target screen
├── reproduce.sh              one-command end-to-end driver
├── requirements.txt · setup.py · LICENSE
```

Generated artifacts (checkpoints, cached data, result tables, and figures) are produced by the
pipeline and are **git-ignored** — see `.gitignore`.

## Hardware footprint

Peak VRAM stays **< 0.6 GB** on a single 8 GB consumer GPU (reference: NVIDIA RTX 5050, 32 GB RAM,
i5-11400), via offline reference-log-prob caching, a ~1.8 M-parameter policy, and BF16 autocast
with gradient accumulation.

## Citation

Manuscript in preparation. A citation entry will be added here once the paper is published.

## License

Released under the [MIT License](LICENSE).
