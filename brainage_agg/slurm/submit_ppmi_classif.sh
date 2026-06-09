#!/bin/bash
# SLURM array job: PPMI PD/HC classification (one task per extractor file).
# Array size: 1 (FS ROI) + N_CNN files — check with:
#   python brainage_agg/experiment/run_classification.py --config brainage_agg/ppmi_config.yaml --list
# Dependency: PPMI_data/features/ must contain all NPZ files.
#
# Usage (from project root):
#   N=$(python brainage_agg/experiment/run_classification.py \
#         --config brainage_agg/ppmi_config.yaml --list | head -1)
#   sbatch --array=0-$((N-1)) brainage_agg/slurm/submit_ppmi_classif.sh
#
# Merge afterwards:
#   python brainage_agg/experiment/run_classification.py \
#     --config brainage_agg/ppmi_config.yaml --positive-class PD --merge

#SBATCH --job-name=ppmi_classif
#SBATCH --output=brainage_agg/slurm/logs/ppmi_classif_%A_%a.out
#SBATCH --error=brainage_agg/slurm/logs/ppmi_classif_%A_%a.err
#SBATCH --array=0-0            # set correct upper bound before submitting (see usage above)
#SBATCH --partition=all
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT="$PWD"
LOG_DIR="$PROJECT_ROOT/brainage_agg/slurm/logs"
mkdir -p "$LOG_DIR"

echo "Task ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_MAX} starting on $(hostname) at $(date)"

source "$PROJECT_ROOT/.venv/bin/activate"

python3 "$PROJECT_ROOT/brainage_agg/experiment/run_classification.py" \
    --job-id "${SLURM_ARRAY_TASK_ID}" \
    --project-root "$PROJECT_ROOT" \
    --config "$PROJECT_ROOT/brainage_agg/ppmi_config.yaml" \
    --positive-class PD

echo "Task ${SLURM_ARRAY_TASK_ID} done at $(date)"
