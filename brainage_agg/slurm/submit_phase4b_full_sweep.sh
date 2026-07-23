#!/bin/bash
# SLURM GPU array job: Phase 4b full sweep -- group_perm permutation-null
# correction for all three weight types (tstat, shap, hybrid), all 6
# aggregations, all 10 CNN seeds, both datasets (NKI, PPMI).
#
# Each array task handles ONE (dataset, aggregation) pair and processes all
# 10 seeds internally (existing seed-averaging behavior of run_voxel_importance.py
# is unchanged -- this just turns on --n-permutations for the full scope).
#
# 12 tasks total (2 datasets x 6 aggregations). Per-task wall time, based on
# validation job 281357 (d=7680, ~20.5 min/group_perm permutation):
#   - 5 of 6 aggregations (d_agg=7680):        ~10 seeds x (~19min PassA + 10x20.5min PassA'
#                                                + ~6min PassB) ~= ~48h
#   - concatenation (d_agg=23040, ~3x cost):   ~10 seeds x (~57min PassA + 10x61.5min PassA'
#                                                + ~6min PassB) ~= ~122h
# --time below gives generous headroom over the concatenation estimate.
#
# Array throttled to 2 concurrent tasks (%2): the gpu partition's only node
# (cgpu01) has no SLURM GRES/GPU accounting configured (`scontrol show node
# cgpu01` reports Gres=(null)), so nothing stops the scheduler from packing
# more array tasks onto it than its one ~14.6GB GPU can hold. Job 319349
# (2026-07-16) had 4 tasks land on it concurrently and silently lost one
# (task 3, NKI lme_slope): every subject's gradient computation hit CUDA OOM,
# the run logged "No attribution maps produced -- skipping" and still exited
# 0/COMPLETED, leaving that cell's output file untouched from a prior run.
# 2 tasks x ~4.3GB fits with headroom (the OOM needed 4 tasks x ~4.3GB
# against a 14.58GB card). Re-check `brainage_agg/slurm/logs/phase4b_*.out`
# for "CUDA out of memory" / "No attribution maps produced" after any future
# run regardless -- this failure mode does not show up as a failed job.
#
# Usage (from project root):
#   sbatch brainage_agg/slurm/submit_phase4b_full_sweep.sh

#SBATCH --job-name=phase4b_sweep
#SBATCH --output=brainage_agg/slurm/logs/phase4b_%A_%a.out
#SBATCH --error=brainage_agg/slurm/logs/phase4b_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=144:00:00
#SBATCH --array=0-11%2

set -euo pipefail

PROJECT_ROOT="$PWD"
LOG_DIR="$PROJECT_ROOT/brainage_agg/slurm/logs"
mkdir -p "$LOG_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS=(nki ppmi)
AGGS=(mean concatenation annualized_rate lme_slope difference lme_slope_change)

n_aggs=${#AGGS[@]}
ds_idx=$(( SLURM_ARRAY_TASK_ID / n_aggs ))
agg_idx=$(( SLURM_ARRAY_TASK_ID % n_aggs ))
DATASET=${DATASETS[$ds_idx]}
AGG=${AGGS[$agg_idx]}

echo "Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} starting on $(hostname) at $(date)"
echo "dataset=${DATASET} agg=${AGG}"

source "$PROJECT_ROOT/.venv/bin/activate"

python3 -m brainage_agg.experiment.run_voxel_importance \
    --device cuda \
    --dataset "$DATASET" \
    --agg "$AGG" \
    --n-subjects 50 \
    --n-seeds 10 \
    --weight-types tstat shap hybrid \
    --n-permutations 10

echo "Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} done at $(date)"
