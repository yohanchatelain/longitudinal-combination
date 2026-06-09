#!/bin/bash
# SLURM array job: G1 vs G2 span classification (one task per extractor).
#
# Usage (from project root):
#   sbatch brainage_agg/slurm/submit_span_classif.sh
#
# After the array completes, merge with:
#   python brainage_agg/experiment/run_span_classification.py --merge

#SBATCH --job-name=span_classif
#SBATCH --output=brainage_agg/slurm/logs/span_%A_%a.out
#SBATCH --error=brainage_agg/slurm/logs/span_%A_%a.err
#SBATCH --array=0-240
#SBATCH --partition=all
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

PROJECT_ROOT="$PWD"
LOG_DIR="$PROJECT_ROOT/brainage_agg/slurm/logs"
mkdir -p "$LOG_DIR"

echo "Task ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_MAX} starting on $(hostname) at $(date)"

source "$PROJECT_ROOT/.venv/bin/activate"

python3 "$PROJECT_ROOT/brainage_agg/experiment/run_span_classification.py" \
    --job-id "${SLURM_ARRAY_TASK_ID}" \
    --project-root "$PROJECT_ROOT"

echo "Task ${SLURM_ARRAY_TASK_ID} done at $(date)"
