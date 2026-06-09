#!/bin/bash
# SLURM job for PPMI factorial experiment (all arms, all bands PD/HC/all).
# Writes results to brainage_agg/ppmi_outputs/results.csv.
# Dependency: PPMI_data/features/ must contain ROI + CNN NPZ files.
#
# Usage (from project root):
#   sbatch brainage_agg/slurm/submit_ppmi_run.sh
#   sbatch brainage_agg/slurm/submit_ppmi_run.sh --dry-run   # quick smoke test
#   sbatch brainage_agg/slurm/submit_ppmi_run.sh --skip-lme  # skip LME aggregations

#SBATCH --job-name=ppmi_run
#SBATCH --output=brainage_agg/slurm/logs/ppmi_run_%j.out
#SBATCH --error=brainage_agg/slurm/logs/ppmi_run_%j.err
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
    --config "$PROJECT_ROOT/brainage_agg/ppmi_config.yaml" \
    "$@"

echo "Job ${SLURM_JOB_ID} done at $(date)"
