#!/usr/bin/env bash
#SBATCH --job-name=attr_validation
# GPU partition has no configured GRES/TRES on this cluster; selecting the
# partition is the site-specific resource request.
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/attribution_validation/logs/%A_%a.out
#SBATCH --error=outputs/attribution_validation/logs/%A_%a.err

set -euo pipefail

run_id=${1:?usage: sbatch ... run_attribution_validation.sbatch RUN_ID}
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

# No pyproject.toml anchors this repo as a uv project, so `uv run` resolves
# its interpreter from ambient shell state (VIRTUAL_ENV) instead of always
# using ./.venv. That caused job 363859 to silently fall back to a stray
# python3.11 user-site install and fail every task. Invoke the venv
# interpreter directly so this is deterministic regardless of submission env.
python="$SLURM_SUBMIT_DIR/.venv/bin/python"

cohorts=(nki ppmi)
architectures=(double_conv cov_pool)
targets=(cerebellar_white_matter hippocampus putamen superior_frontal)
amplitudes=(0.02 0.05 0.10)
aggregations=(mean annualized_rate)

if (( task_id < 960 )); then
    seed=$((task_id % 10))
    aggregation_index=$(((task_id / 10) % 2))
    amplitude_index=$(((task_id / 20) % 3))
    target_index=$(((task_id / 60) % 4))
    architecture_index=$(((task_id / 240) % 2))
    cohort_index=$(((task_id / 480) % 2))
    "$python" -m brainage_agg.experiment.run_attribution_validation injection-cell \
        --run-id "$run_id" \
        --cohort "${cohorts[$cohort_index]}" \
        --architecture "${architectures[$architecture_index]}" \
        --target "${targets[$target_index]}" \
        --amplitude "${amplitudes[$amplitude_index]}" \
        --aggregation "${aggregations[$aggregation_index]}" \
        --seed "$seed" \
        --device cuda
else
    control_index=$((task_id - 960))
    seed=$((control_index % 10))
    architecture_index=$(((control_index / 10) % 2))
    cohort_index=$(((control_index / 20) % 2))
    "$python" -m brainage_agg.experiment.run_attribution_validation control-cell \
        --run-id "$run_id" \
        --cohort "${cohorts[$cohort_index]}" \
        --architecture "${architectures[$architecture_index]}" \
        --seed "$seed" \
        --device cuda
fi
