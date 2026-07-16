#!/bin/bash
# SLURM GPU job: Phase 4a pilot -- compare index-permutation vs
# group-label-permutation null designs for architecture-bias correction
# (w_t only, small scale: 1 dataset x 1 aggregation x n_seeds x n_subjects).
#
# Usage (from project root):
#   sbatch brainage_agg/slurm/submit_permutation_null_pilot.sh
#   sbatch brainage_agg/slurm/submit_permutation_null_pilot.sh --topk 100 --n-permutations 30

#SBATCH --job-name=perm_null_pilot
#SBATCH --output=brainage_agg/slurm/logs/perm_null_pilot_%j.out
#SBATCH --error=brainage_agg/slurm/logs/perm_null_pilot_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT="$PWD"
LOG_DIR="$PROJECT_ROOT/brainage_agg/slurm/logs"
mkdir -p "$LOG_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ${SLURM_JOB_ID} starting on $(hostname) at $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Args: $*"

source "$PROJECT_ROOT/.venv/bin/activate"

python3 -m brainage_agg.experiment.run_permutation_null_pilot \
    --dataset nki --agg mean --n-subjects 5 --n-seeds 2 \
    --topk 50 --n-permutations 20 --device cuda \
    "$@"

echo "Job ${SLURM_JOB_ID} done at $(date)"
