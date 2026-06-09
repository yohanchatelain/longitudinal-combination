#!/bin/bash
# SLURM GPU job: map CNN feature importance to voxel regions via gradient attribution.
#
# Processes one (arch, scaler, channels) extractor group across all available seeds,
# for all aggregation types, on a sampled subset of subjects.  Outputs are written
# to <result_dir>/group_diff/voxel_importance/<agg>/.
#
# Usage (from project root):
#   bash brainage_agg/slurm/submit_voxel_importance.sh
#   bash brainage_agg/slurm/submit_voxel_importance.sh --dataset ppmi
#   bash brainage_agg/slurm/submit_voxel_importance.sh \
#       --arch double_conv --scaler zscore --channels t1_sobel \
#       --n-subjects 50 --n-seeds 5 --save-voxel-map
#
# Tip: the rank channel (t1_rank_sobel, t1_rank) is the preprocessing bottleneck
# (~2-3 min/subject on CPU).  Use --channels t1_sobel or --channels t1 for faster runs.

#SBATCH --job-name=voxel_importance
#SBATCH --output=brainage_agg/slurm/logs/voxel_importance_%j.out
#SBATCH --error=brainage_agg/slurm/logs/voxel_importance_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="$PWD"
LOG_DIR="$PROJECT_ROOT/brainage_agg/slurm/logs"
mkdir -p "$LOG_DIR"

# Reduce CUDA memory fragmentation; automatic fallback to CPU on OOM is in the script
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ${SLURM_JOB_ID} starting on $(hostname) at $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Args: $*"

source "$PROJECT_ROOT/.venv/bin/activate"

python3 -m brainage_agg.experiment.run_voxel_importance \
    --device cuda \
    "$@"

echo "Job ${SLURM_JOB_ID} done at $(date)"
