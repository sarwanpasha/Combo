# Combo: measurable and predictable genetic interactions in combinatorial single-cell screens

Code for analyzing two-gene perturbations in two public single-cell screens:

* **Norman et al. (2019)**: CRISPR activation in K562 cells (loaded through the GEARS data interface).
* **Wessels et al. (2023)**: Cas13 RNA targeting (CaRPool-seq) in THP-1 cells (scPerturb, Zenodo record 13350497). Only endogenous genes are used; guide-array and transgene features stored in the count matrix are removed before normalization.

For every pair the additive expectation is `a_ij = mu_0 + alpha_i + alpha_j` and the interaction is `delta_ij = y_ij - a_ij`.
The code estimates how much of each interaction is measurement noise, characterizes the reproducible part, and evaluates
baselines and published models against the measured interaction.

## Contents

| File | Purpose |
|---|---|
| `setup_env.sh`, `setup_model_envs.sh` | Python environments (GEARS; CPA; scGPT; AttentionPert source) with small compatibility patches |
| `download_data.sh` | Downloads both datasets |
| `build_pseudobulk.py`, `build_wessels.py` | Pseudobulk arrays (`pseudobulk.npz`, `pseudobulk_wessels.npz`) |
| `make_splits.py` | Random and group-holdout 5-fold splits |
| `build_wessels_gears.py` | GEARS `PertData` object for the Wessels screen |
| `cells.py` | Cell-level loading aligned with the pseudobulk arrays |
| `noise_ceiling.py` | Split-half noise of every interaction and the noise ceiling |
| `crosshalf.py` | Cross-half evaluation (train on one half of the cells, score on the other), signal share, regression-dilution check, reproducibility of interaction descriptors |
| `target_check.py` | On-target activation or knockdown in pairs versus singles |
| `bio_insights.py` | Genetic interaction model (magnitude, dominance, classes), rescaling versus novel parts, direction of non-additivity, interaction components with matched-background GO enrichment, gene-level additive models, prioritization |
| `bio_programs.py` | Myeloid and erythroid program shifts of each pair and readout-gene buffering |
| `dominance_check.py` | Dominance hierarchy versus single-effect strength |
| `run_gears_v2.py` | GEARS per fold and seed (random, group-holdout and held-out-gene splits; validation pairs taken from training pairs) |
| `run_gears_published.py` | GEARS on its published simulation split, with its own metrics |
| `run_cpa.py`, `run_attentionpert.py`, `run_scgpt.py` | CPA, AttentionPert and scGPT on the same folds |
| `linear_pair_baselines.py` | Linear model of Ahlmann-Eltze et al. adapted to pairs, shared-gene and nearest-neighbor baselines |
| `score_all.py` | Scores all in-script baselines and every saved model prediction (interaction fraction, total R^2, per-pair scores, noise-corrected scores, own-backbone scores, common-subset scores, held-out-gene split over 20 gene draws) |
| `summarize_scores.py` | Compact summary of `scores_all.tsv` |
| `export_figdata.py` | Compact text export of the per-pair quantities used for figures |
| `bench.py`, `density_subsample.py`, `analysis_extra.py`, `run_gears.py`, `run_all.sh`, `submit_all.sh` | Earlier benchmark scripts (oracle bootstrap, rank sweep, pair-density subsampling) |
| `setup_slurm.sh`, `submit_models.sh` | Optional single-node Slurm setup (`GPU_SHARE=k` runs k jobs per GPU) and submission of all jobs as `Combo_*` |

## Usage

```bash
bash setup_env.sh && bash setup_model_envs.sh
bash download_data.sh
python build_pseudobulk.py && python build_wessels.py && python make_splits.py && python build_wessels_gears.py
python noise_ceiling.py --reps 20 && python crosshalf.py && python target_check.py
python bio_insights.py && python bio_programs.py && python dominance_check.py
# GPU models (Slurm): bash setup_slurm.sh && GROUP=gears bash submit_models.sh && GROUP=hog bash submit_models.sh && GROUP=ext bash submit_models.sh
python linear_pair_baselines.py
python score_all.py            # writes scores_all.tsv
```
