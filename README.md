# CombiPert-Bench

Evaluation code for two-gene (combinatorial) perturbation prediction on two public single-cell screens:

* **Norman et al. (2019)**: CRISPR activation in K562 cells (loaded through the GEARS data interface).
* **Wessels et al. (2023)**: Cas13 RNA-targeting (CaRPool-seq) in THP-1 cells (scPerturb, Zenodo record 13350497).

For every gene pair the additive prediction is `a_ij = mu_0 + alpha_i + alpha_j` (control pseudobulk plus both measured single-gene effects) and the interaction is `delta_ij = y_ij - a_ij`. The scorer reports

* total-expression R^2 over test pairs, and
* the interaction fraction `F_int = 1 - sum ||delta_hat - delta||^2 / sum ||delta||^2`, which is 0 for the additive baseline.

Models that predict their own single perturbations (for example GEARS) are scored against the measured (empirical) backbone and, for reference, against their own additive reconstruction.

## Contents

| File | Purpose |
|---|---|
| `setup_env.sh` | Creates a virtual environment, installs dependencies and applies two small compatibility patches to GEARS 0.1.2 |
| `download_data.sh` | Downloads both datasets |
| `build_pseudobulk.py`, `build_wessels.py` | Pseudobulk arrays (`pseudobulk.npz`, `pseudobulk_wessels.npz`) |
| `make_splits.py` | Random 5-fold and gene-disjoint 5-fold splits for both datasets |
| `build_wessels_gears.py` | GEARS `PertData` object for the Wessels screen |
| `run_gears.py` | Trains GEARS per fold and saves pair and single predictions |
| `bench.py` | Additive, ridge, MLP coordinate map, oracle and GEARS scoring; `--ranks` for a rank sweep |
| `density_subsample.py` | Interaction recovery as a function of training pairs per gene |
| `analysis_extra.py` | Bootstrap of the oracle ceiling, GO-initialized MLP, GO enrichment of interaction components |
| `run_all.sh` | Runs the full pipeline sequentially |
| `setup_slurm.sh`, `submit_all.sh` | Optional single-node Slurm setup and parallel submission of all jobs |

## Usage

```bash
bash setup_env.sh          # once
bash run_all.sh            # sequential run, logs in logs/
# or, with Slurm available:
bash submit_all.sh         # parallel jobs named Combo_*
```

GEARS training is the only step that needs a GPU. All other steps run on a CPU in minutes. Results are written to `bench_results.tsv`, `bench_ranks.tsv`, `density_subsample.tsv` and `analysis_extra.json`.
