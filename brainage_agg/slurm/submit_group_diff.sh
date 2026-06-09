#!/bin/bash
# SLURM array job: one task per extractor (0 = FS ROI, 1-240 = CNN variants).
# After the array completes, the merge job runs automatically to produce
# summary.csv and heatmaps.
#
# Usage:
#   bash brainage_agg/slurm/submit_group_diff.sh [--skip-lme] [--n-permutations N]
#
# Options forwarded to group_diff.py:
#   --skip-lme          Skip lme_slope / lme_slope_change aggregations (faster)
#   --n-permutations N  Permutations for MANOVA (default 1000)

set -euo pipefail

PROJECT_ROOT="$PWD"
PYTHON="$PROJECT_ROOT/.venv/bin/python3"
GROUP_DIFF="$PROJECT_ROOT/brainage_agg/analysis/group_diff.py"
RESULT_DIR="$PROJECT_ROOT/brainage_agg/outputs"
LOG_DIR="$PROJECT_ROOT/brainage_agg/slurm/logs"

mkdir -p "$LOG_DIR"

# Parse optional arguments
EXTRA_ARGS=""
for arg in "$@"; do
    EXTRA_ARGS="$EXTRA_ARGS $arg"
done

# Count total extractors: 1 FS ROI + all CNN files
N_CNN=$(ls "$PROJECT_ROOT/outputs/features/features__"*.npz 2>/dev/null \
        | grep -v freesurfer_roi | wc -l)
N_TOTAL=$((1 + N_CNN))
ARRAY_END=$((N_TOTAL - 1))

echo "Submitting array job: 0-${ARRAY_END} (${N_TOTAL} extractors)"
echo "Extra args: ${EXTRA_ARGS:-none}"

# Task 0 (FS ROI) fits StatsmodelsLME for 657 features (~30 min).
# Submit it separately with a longer time limit so it doesn't timeout.
FS_ROI_JOB=$(sbatch --parsable \
    --job-name=group_diff_fs_roi \
    --output="$LOG_DIR/array_%j_0.out" \
    --error="$LOG_DIR/array_%j_0.err" \
    --partition=all \
    --cpus-per-task=4 \
    --mem=16G \
    --time=01:00:00 \
    --wrap="
cd '$PROJECT_ROOT'
source .venv/bin/activate
echo \"Task 0 / ${ARRAY_END}: \$(date)\"
python3 '$GROUP_DIFF' \
    --result-dir '$RESULT_DIR' \
    --extractor-idx 0 \
    --n-jobs 4 \
    $EXTRA_ARGS
echo \"Done: \$(date)\"
")

echo "FS ROI job submitted: $FS_ROI_JOB (60 min limit)"

# CNN tasks (1-N): VectorizedOLS, finish in ~5 min each
CNN_JOB=$(sbatch --parsable \
    --job-name=group_diff_cnn \
    --output="$LOG_DIR/array_%A_%a.out" \
    --error="$LOG_DIR/array_%A_%a.err" \
    --partition=all \
    --array="1-${ARRAY_END}" \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:15:00 \
    --wrap="
cd '$PROJECT_ROOT'
source .venv/bin/activate
echo \"Task \$SLURM_ARRAY_TASK_ID / ${ARRAY_END}: \$(date)\"
python3 '$GROUP_DIFF' \
    --result-dir '$RESULT_DIR' \
    --extractor-idx \$SLURM_ARRAY_TASK_ID \
    --n-jobs 4 \
    $EXTRA_ARGS
echo \"Done: \$(date)\"
")

echo "CNN array job submitted: $CNN_JOB (15 min limit, tasks 1-${ARRAY_END})"

# --- Merge job: runs after both FS ROI and CNN jobs complete ---
MERGE_JOB=$(sbatch --parsable \
    --job-name=group_diff_merge \
    --output="$LOG_DIR/merge_%j.out" \
    --error="$LOG_DIR/merge_%j.err" \
    --partition=all \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:15:00 \
    --dependency="afterok:${FS_ROI_JOB}:${CNN_JOB}" \
    --wrap="
cd '$PROJECT_ROOT'
source .venv/bin/activate
echo \"Merging results: \$(date)\"
python3 '$GROUP_DIFF' \
    --result-dir '$RESULT_DIR' \
    --merge
echo \"Merge done: \$(date)\"
")

echo "Merge job submitted: $MERGE_JOB (depends on ${FS_ROI_JOB} and ${CNN_JOB})"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $LOG_DIR/merge_${MERGE_JOB}.out"
