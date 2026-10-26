#!/bin/bash
# Submits the whole CombiPert-Bench pipeline as parallel Slurm jobs (names Combo_*), with dependencies.
#   Combo_prep  -> data download, pseudobulk, splits, Wessels GEARS PertData
#   Combo_G_<ds>_<split>_f<k> (20 jobs, parallel) -> GEARS per fold
#   Combo_bench_nogears, Combo_ranks, Combo_density (parallel, after prep)
#   Combo_bench_final (after all GEARS jobs)
set -euo pipefail
cd "$(dirname "$0")"; WD=$(pwd); mkdir -p logs
mk(){ # name cpus mem time cmd
cat > logs/$1.sbatch << EOF
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$2
#SBATCH --mem=$3
#SBATCH --time=$4
#SBATCH --output=$WD/logs/$1_%j.out
#SBATCH --error=$WD/logs/$1_%j.err
cd $WD
source $WD/activate.sh
export OMP_NUM_THREADS=$2 MKL_NUM_THREADS=$2 OPENBLAS_NUM_THREADS=$2
set -eo pipefail
$5
EOF
echo logs/$1.sbatch; }
sub(){ sbatch --parsable "$@"; }

PREP_DONE=0; [ -s gd_split_wessels.npz ] && [ -d data_wessels_gears/wessels ] && PREP_DONE=1
if [ $PREP_DONE = 0 ]; then
  P=$(sub $(mk Combo_prep 8 64G 02:00:00 "bash download_data.sh; python build_pseudobulk.py; python build_wessels.py; python make_splits.py; [ -d data_wessels_gears/wessels ] || python build_wessels_gears.py"))
  DEP="--dependency=afterok:$P"; echo "Combo_prep $P"
else DEP=""; fi

GIDS=""
for DS in norman wessels; do for SP in rand gd; do for F in 0 1 2 3 4; do
  J=$(sub $DEP $(mk Combo_G_${DS}_${SP}_f$F 3 24G 04:00:00 \
     "DATASET=$DS SPLIT=$SP FOLD=$F GEARS_EPOCHS=20 python run_gears.py 2>&1 | grep -vE 'Step [0-9]+ Train Loss|^Epoch [0-9]+:'"))
  GIDS="$GIDS:$J"
done; done; done
echo "GEARS jobs: $GIDS"
sub $DEP $(mk Combo_bench_nogears 4 32G 02:00:00 "python bench.py --out bench_nogears.tsv")
sub $DEP $(mk Combo_ranks 4 32G 04:00:00 "python bench.py --ranks 4 8 16 24 32 48 --out bench_ranks.tsv")
sub $DEP $(mk Combo_density 4 32G 06:00:00 "python density_subsample.py --reps 10")
sub $DEP $(mk Combo_extra 8 24G 04:00:00 "python analysis_extra.py --boot 1000")
sub --dependency=afterany$GIDS $(mk Combo_bench_final 4 32G 02:00:00 "python bench.py --out bench_results.tsv")
squeue -o "%.8i %.28j %.8T %.10M %.5C %R"
