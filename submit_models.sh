#!/bin/bash
# Submits all GPU model jobs (one GPU each) and CPU analyses as Slurm jobs named Combo_*.
# Requires: setup_env.sh, setup_slurm.sh and data preparation already done; extra envs from setup_model_envs.sh.
set -uo pipefail
cd "$(dirname "$0")"; WD=$(pwd); mkdir -p logs
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
mk(){ # name gpus cpus mem time env cmd ; skips jobs that are already queued or running
squeue -h -o %j | grep -qx "$1" && return
cat > logs/$1.sbatch << EOF
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$3
#SBATCH --mem=$4
#SBATCH --time=$5
$( [ "$2" -gt 0 ] && { [ -n "${GPU_SHARE:-}" ] && echo "#SBATCH --licenses=gpu:$2" || echo "#SBATCH --gres=gpu:$2"; } )
#SBATCH --output=$WD/logs/$1_%j.out
#SBATCH --error=$WD/logs/$1_%j.err
cd $WD
[ -n "\${GPU_SHARE:-}" ] && [ "$2" -gt 0 ] && export CUDA_VISIBLE_DEVICES=\$((SLURM_JOB_ID % $NGPU))
export PYTHONNOUSERSITE=1 PYTHONPATH="" TMPDIR=$WD/tmp
source $WD/$6/bin/activate
export OMP_NUM_THREADS=$3 MKL_NUM_THREADS=$3 OPENBLAS_NUM_THREADS=$3
$7
EOF
sbatch --parsable logs/$1.sbatch; }
F="2>&1 | grep -vE 'Step [0-9]+ Train Loss|^Epoch [0-9]+:'"
G=${GROUP:-all}
if [[ $G == all || $G == gears ]]; then
# GEARS v2 (validation = 10% of training pairs), 3 seeds on random splits, 1 seed on within-group splits
for S in 0 1 2; do for DS in ${DSL:-norman wessels}; do for f in 0 1 2 3 4; do
  mk Combo_Gv2_${DS}_rand_f${f}_s$S 1 6 40G 08:00:00 venv "DATASET=$DS SPLIT=rand FOLD=$f SEED=$S python run_gears_v2.py $F"
done; done; done
for DS in ${DSL:-norman wessels}; do for f in 0 1 2 3 4; do
  mk Combo_Gv2_${DS}_gd_f${f}_s0 1 6 40G 08:00:00 venv "DATASET=$DS SPLIT=gd FOLD=$f SEED=0 python run_gears_v2.py $F"
done; done
# GEARS published protocol (simulation split), 2 seeds
[[ "${DSL:-norman}" == *norman* ]] && mk Combo_Gpub 1 6 40G 12:00:00 venv "SEEDS=1,2 python run_gears_published.py $F"
fi
if [[ $G == all || $G == hog ]]; then
# GEARS on the held-out-gene split (10 gene draws, same draws as score_all.py)
for DS in ${DSL:-norman wessels}; do for s in 0 1 2 3 4 5 6 7 8 9; do
  mk Combo_Ghog_${DS}_s$s 1 6 40G 08:00:00 venv "DATASET=$DS SPLIT=hog FOLD=$s SEED=0 python run_gears_v2.py $F"
done; done
fi
if [[ $G == all || $G == ext ]]; then
# CPA, AttentionPert and scGPT on the random split
for DS in ${DSL:-norman wessels}; do for f in 0 1 2 3 4; do
  mk Combo_CPA_${DS}_rand_f$f 1 8 60G 12:00:00 venv_cpa "DATASET=$DS SPLIT=rand FOLD=$f SEED=0 MAX_EPOCHS=150 python run_cpa.py"
done; done
for DS in ${DSL:-norman wessels}; do
  mk Combo_ATT_${DS}_rand 1 8 60G 16:00:00 venv "DATASET=$DS SPLIT=rand SEED=0 python run_attentionpert.py $F"
done
# scGPT fine-tuning is the most expensive model: 5 epochs per fold, all folds for the THP-1 screen, folds 0 and 1 for K562
for f in 0 1 2 3 4; do mk Combo_SCGPT_wessels_rand_f$f 1 8 80G 24:00:00 venv_scgpt "DATASET=wessels SPLIT=rand FOLD=$f SEED=0 EPOCHS=5 python run_scgpt.py"; done
for f in 0 1; do mk Combo_SCGPT_norman_rand_f$f 1 8 80G 24:00:00 venv_scgpt "DATASET=norman SPLIT=rand FOLD=$f SEED=0 EPOCHS=5 python run_scgpt.py"; done
fi
if [[ $G == all || $G == cpu ]]; then
# CPU analyses
mk Combo_linear 0 8 32G 02:00:00 venv "python linear_pair_baselines.py"
mk Combo_noise 0 16 64G 04:00:00 venv "python noise_ceiling.py --reps 20"
mk Combo_target 0 8 64G 02:00:00 venv "python target_check.py"
mk Combo_bio 0 16 64G 06:00:00 venv "python bio_insights.py"
fi
squeue -o "%.6i %.32j %.9T %.9M %.4C %b %R"
