#!/bin/bash
# SLURM job: architectural prior + task feature importance experiment.
#
# Runs all (dataset, arch, task, method) combinations:
#   datasets : nki, ppmi
#   archs    : double_conv, cov_pool
#   tasks    : prior, age, sex, severity (severity is PPMI-only)
#   methods  : rf, ridge (both by default via --method both)
#
# Outputs land in outputs/prior_experiment/{dataset}/{arch}/{task}_{method}/
#
# Usage (from project root):
#   sbatch brainage_agg/slurm/submit_prior_experiment.sh
#   sbatch brainage_agg/slurm/submit_prior_experiment.sh --n-seeds 5
#   sbatch brainage_agg/slurm/submit_prior_experiment.sh \
#       --dataset nki --arch double_conv --modes prior age --method rf

#SBATCH --job-name=prior_experiment
#SBATCH --output=brainage_agg/slurm/logs/prior_experiment_%j.out
#SBATCH --error=brainage_agg/slurm/logs/prior_experiment_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="$PWD"
LOG_DIR="$PROJECT_ROOT/brainage_agg/slurm/logs"
mkdir -p "$LOG_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ${SLURM_JOB_ID} starting on $(hostname) at $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Args: $*"

source "$PROJECT_ROOT/.venv/bin/activate"

python3 -m brainage_agg.experiment.run_prior_experiment \
    --device cuda \
    --method both \
    "$@"

echo "Job ${SLURM_JOB_ID} done at $(date)"
