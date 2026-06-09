#!/bin/bash
# Single SLURM job for the main factorial experiment (run.py).
#
# Runs all 241 extractors × 9 arm/agg combos × 3 bands × 5 CV folds sequentially.
# Results written to: brainage_agg/outputs/results.csv
#
# Usage (from project root):
#   sbatch brainage_agg/slurm/submit_run.sh
#   sbatch brainage_agg/slurm/submit_run.sh --dry-run   # quick smoke test
#   sbatch brainage_agg/slurm/submit_run.sh --skip-lme  # skip LME aggs

#SBATCH --job-name=brainage_run
#SBATCH --output=brainage_agg/slurm/logs/run_%j.out
#SBATCH --error=brainage_agg/slurm/logs/run_%j.err
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="$PWD"
LOG_DIR="$PROJECT_ROOT/brainage_agg/slurm/logs"
mkdir -p "$LOG_DIR"

echo "Job ${SLURM_JOB_ID} starting on $(hostname) at $(date)"
echo "Project root: ${PROJECT_ROOT}"

source "$PROJECT_ROOT/.venv/bin/activate"

python3 "$PROJECT_ROOT/brainage_agg/experiment/run.py" \
    --project-root "$PROJECT_ROOT" \
    "$@"

echo "Job ${SLURM_JOB_ID} done at $(date)"
