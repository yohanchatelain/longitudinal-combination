#!/usr/bin/env bash
set -euo pipefail

run_id=${1:?usage: scripts/submit_attribution_validation.sh RUN_ID}
mkdir -p outputs/attribution_validation/logs

# The %2 throttle is part of the preregistered compute contract. Injection
# (tasks 0-959) and architecture-prior controls (960-999) share one array so
# their combined GPU concurrency cannot exceed two.
sbatch --array=0-999%2 slurm/run_attribution_validation.sbatch "$run_id"
