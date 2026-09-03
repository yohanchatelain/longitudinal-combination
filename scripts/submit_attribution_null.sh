#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?usage: scripts/submit_attribution_null.sh RUN_ID}
mkdir -p outputs/attribution_validation/logs
sbatch --array=0-3%2 slurm/run_attribution_null.sbatch "$run_id"
