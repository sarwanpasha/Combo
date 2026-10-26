#!/bin/bash
# Full CombiPert-Bench reproduction on one GPU box. Logs to logs/*.txt. Resumable (GEARS folds skip if saved).
set -uo pipefail
cd "$(dirname "$0")"; source activate.sh; mkdir -p logs
step(){ echo "=== $(date +%T) $1"; }
step download;            bash download_data.sh            2>&1 | tail -5  > logs/download.txt
step pseudobulk_norman;   python build_pseudobulk.py       > logs/build_pseudobulk.txt 2>&1
step pseudobulk_wessels;  python build_wessels.py          > logs/build_wessels.txt 2>&1
step splits;              python make_splits.py            > logs/make_splits.txt 2>&1
step bench_nogears;       python bench.py --out bench_nogears.tsv > logs/bench_nogears.txt 2>&1
step wessels_gears_data;  [ -d data_wessels_gears/wessels ] || python build_wessels_gears.py > logs/build_wessels_gears.txt 2>&1
F='Step [0-9]+ Train Loss|^Epoch [0-9]+:'
for DS in norman wessels; do for SP in rand gd; do
  step "gears $DS $SP"; DATASET=$DS SPLIT=$SP GEARS_EPOCHS=20 python run_gears.py 2>&1 | grep -vE "$F" > logs/gears_${DS}_${SP}.txt
done; done
step bench_final;         python bench.py --out bench_results.tsv > logs/bench_final.txt 2>&1
step rank_sweep;          python bench.py --ranks 4 8 16 24 32 48 --out bench_ranks.tsv > logs/bench_ranks.txt 2>&1
step density;           python density_subsample.py --reps 10 > logs/density.txt 2>&1
step extra;             python analysis_extra.py --boot 1000 > logs/analysis_extra.txt 2>&1
step ALL_DONE
