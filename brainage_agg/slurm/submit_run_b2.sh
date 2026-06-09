#!/bin/bash
# SLURM job for Arm B2 only (eTIV rate target).
# Appends B2 rows to the existing results.csv (Arms A and B already complete).
#
# Usage (from project root):
#   sbatch brainage_agg/slurm/submit_run_b2.sh
#   sbatch brainage_agg/slurm/submit_run_b2.sh --dry-run   # quick smoke test

#SBATCH --job-name=brainage_b2
#SBATCH --output=brainage_agg/slurm/logs/run_b2_%j.out
#SBATCH --error=brainage_agg/slurm/logs/run_b2_%j.err
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
    --arms B2 \
    --append \
    "$@"

echo "Job ${SLURM_JOB_ID} done at $(date)"
