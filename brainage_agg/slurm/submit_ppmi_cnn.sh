#!/bin/bash
# SLURM GPU job: extract CNN features for PPMI (all scalers × feature_sets × seeds).
# Processes 3 scalers × 4 feature_sets × 10 seeds = 120 NPZ files.
# Uses raw T1w NIfTI with non-zero voxel mask (no skull-stripping required).
# Dependency: PPMI_data/cohort_longitudinal.csv must exist.
#
# Usage (from project root):
#   sbatch brainage_agg/slurm/submit_ppmi_cnn.sh
#   sbatch brainage_agg/slurm/submit_ppmi_cnn.sh --overwrite

#SBATCH --job-name=ppmi_cnn
#SBATCH --output=brainage_agg/slurm/logs/ppmi_cnn_%j.out
#SBATCH --error=brainage_agg/slurm/logs/ppmi_cnn_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=36:00:00

set -euo pipefail

PROJECT_ROOT="$PWD"
LOG_DIR="$PROJECT_ROOT/brainage_agg/slurm/logs"
mkdir -p "$LOG_DIR"

echo "Job ${SLURM_JOB_ID} starting on $(hostname) at $(date)"
echo "Project root: ${PROJECT_ROOT}"

source "$PROJECT_ROOT/.venv/bin/activate"

python3 "$PROJECT_ROOT/scripts/extract_all_features.py" \
    --config "$PROJECT_ROOT/brainage_agg/configs/ppmi_extract_cnn.yaml" \
    --gpus 0 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --persistent-workers \
    "$@"

echo "Job ${SLURM_JOB_ID} done at $(date)"
