#!/bin/bash
# Extract PPMI FreeSurfer ROI features using Desikan-Killiany atlas.
# Reads directly from ppmi/<directory>/stats/lh.aparc.stats etc.
# Harmonized with NKI (both use DK atlas, lh.aparc.stats / rh.aparc.stats).
#
# Usage (from project root):
#   sbatch brainage_agg/slurm/submit_ppmi_roi.sh
#   sbatch brainage_agg/slurm/submit_ppmi_roi.sh --overwrite

#SBATCH --job-name=ppmi_roi
#SBATCH --output=brainage_agg/slurm/logs/ppmi_roi_%j.out
#SBATCH --error=brainage_agg/slurm/logs/ppmi_roi_%j.err
#SBATCH --partition=all
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT="$PWD"
LOG_DIR="$PROJECT_ROOT/brainage_agg/slurm/logs"
mkdir -p "$LOG_DIR"

echo "Job ${SLURM_JOB_ID} starting on $(hostname) at $(date)"
echo "Project root: ${PROJECT_ROOT}"

source "$PROJECT_ROOT/.venv/bin/activate"

python3 "$PROJECT_ROOT/scripts/extract_roi_features.py" \
    --config "$PROJECT_ROOT/brainage_agg/configs/ppmi_extract_roi.yaml" \
    --scalers none \
    "$@"

echo "Job ${SLURM_JOB_ID} done at $(date)"
