#!/usr/bin/env bash
#SBATCH --job-name=attr_validation
#SBATCH --dependency=singleton
# GPU partition has no configured GRES/TRES on this cluster.
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=outputs/attribution_validation/logs/null_%A_%a.out
#SBATCH --error=outputs/attribution_validation/logs/null_%A_%a.err

set -euo pipefail

run_id=${1:?usage: sbatch ... run_attribution_null.sbatch RUN_ID}
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
cohorts=(nki ppmi)
aggregations=(mean annualized_rate)
cohort_index=$((task_id / 2))
aggregation_index=$((task_id % 2))

# See run_attribution_validation.sbatch: no pyproject.toml anchors this repo
# as a uv project, so invoke the venv interpreter directly rather than
# relying on `uv run`'s ambient-shell-dependent resolution.
python="$SLURM_SUBMIT_DIR/.venv/bin/python"

"$python" -m brainage_agg.experiment.run_voxel_importance \
    --dataset "${cohorts[$cohort_index]}" \
    --arch double_conv \
    --scaler minmax \
    --channels t1_rank_sobel \
    --n-subjects 50 \
    --n-seeds 5 \
    --agg "${aggregations[$aggregation_index]}" \
    --weight-types tstat shap hybrid \
    --n-permutations 199 \
    --null-rng-seed 20260829 \
    --save-voxel-map \
    --strict-complete \
    --validation-output-dir "outputs/attribution_validation/${run_id}/null" \
    --device cuda
