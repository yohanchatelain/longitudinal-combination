#!/bin/bash
#SBATCH --job-name=brainage_classif
#SBATCH --output=brainage_agg/outputs/classif_jobs/slurm_%A_%a.out
#SBATCH --error=brainage_agg/outputs/classif_jobs/slurm_%A_%a.err
#SBATCH --array=0-240          # adjust upper bound with: python run_classification.py --list
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT=$PWD
PYTHON=$(uv python dir)
echo "Job ${SLURM_ARRAY_TASK_ID} starting on $(hostname) with ${PYTHON}"

# mkdir -p brainage_agg/outputs/classif_jobs
source .venv/bin/activate

uv run brainage_agg/experiment/run_classification.py \
    --job-id "${SLURM_ARRAY_TASK_ID}" \
    --project-root "${PROJECT_ROOT}"

echo "Job ${SLURM_ARRAY_TASK_ID} done"
