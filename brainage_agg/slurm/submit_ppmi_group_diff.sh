#!/bin/bash
# SLURM array job: PPMI PD vs HC group difference analysis.
# One task per extractor (0 = FS ROI, 1-240 = CNN variants).
# After all tasks complete, a merge job produces summary.csv and heatmaps.
#
# Usage (from project root):
#   bash brainage_agg/slurm/submit_ppmi_group_diff.sh [--skip-lme] [--n-permutations N]
#
# Results written to: brainage_agg/ppmi_outputs/group_diff/

set -euo pipefail

PROJECT_ROOT="$PWD"
SCRIPT="$PROJECT_ROOT/brainage_agg/analysis/group_diff.py"
RESULT_DIR="$PROJECT_ROOT/brainage_agg/ppmi_outputs"
FEATURE_DIR="$PROJECT_ROOT/PPMI_data/features"
LOG_DIR="$PROJECT_ROOT/brainage_agg/slurm/logs"

mkdir -p "$LOG_DIR"

# Parse optional arguments
EXTRA_ARGS=""
for arg in "$@"; do
    EXTRA_ARGS="$EXTRA_ARGS $arg"
done

# Count extractors: 1 FS ROI + CNN files
N_CNN=$(ls "$FEATURE_DIR/features__"*.npz 2>/dev/null | grep -v freesurfer_roi | wc -l)
N_TOTAL=$((1 + N_CNN))
ARRAY_END=$((N_TOTAL - 1))

echo "Submitting PPMI group_diff array: 0-${ARRAY_END} (${N_TOTAL} extractors)"

# Task 0 (FS ROI) uses StatsmodelsLME (~30 min)
FS_ROI_JOB=$(sbatch --parsable \
    --job-name=ppmi_gdiff_fsroi \
    --output="$LOG_DIR/ppmi_gdiff_%j_0.out" \
    --error="$LOG_DIR/ppmi_gdiff_%j_0.err" \
    --partition=all \
    --cpus-per-task=4 \
    --mem=16G \
    --time=01:00:00 \
    --wrap="
cd '$PROJECT_ROOT'
source .venv/bin/activate
echo 'Task 0 / ${ARRAY_END}: \$(date)'
python3 '$SCRIPT' \
    --dataset ppmi \
    --result-dir '$RESULT_DIR' \
    --feature-dir '$FEATURE_DIR' \
    --extractor-idx 0 \
    --n-jobs 4 \
    $EXTRA_ARGS
echo 'Done: \$(date)'
")

echo "FS ROI job: $FS_ROI_JOB (60 min)"

# CNN tasks (1-N): VectorizedOLS, ~5 min each
CNN_JOB=$(sbatch --parsable \
    --job-name=ppmi_gdiff_cnn \
    --output="$LOG_DIR/ppmi_gdiff_%A_%a.out" \
    --error="$LOG_DIR/ppmi_gdiff_%A_%a.err" \
    --partition=all \
    --array="1-${ARRAY_END}" \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:15:00 \
    --wrap="
cd '$PROJECT_ROOT'
source .venv/bin/activate
echo 'Task \$SLURM_ARRAY_TASK_ID / ${ARRAY_END}: \$(date)'
python3 '$SCRIPT' \
    --dataset ppmi \
    --result-dir '$RESULT_DIR' \
    --feature-dir '$FEATURE_DIR' \
    --extractor-idx \$SLURM_ARRAY_TASK_ID \
    $EXTRA_ARGS
echo 'Done: \$(date)'
")

echo "CNN array job: $CNN_JOB (15 min, tasks 1-${ARRAY_END})"

# Merge after both complete
MERGE_JOB=$(sbatch --parsable \
    --job-name=ppmi_gdiff_merge \
    --output="$LOG_DIR/ppmi_gdiff_merge_%j.out" \
    --error="$LOG_DIR/ppmi_gdiff_merge_%j.err" \
    --partition=all \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:15:00 \
    --dependency="afterok:${FS_ROI_JOB}:${CNN_JOB}" \
    --wrap="
cd '$PROJECT_ROOT'
source .venv/bin/activate
echo 'Merging PPMI group_diff results: \$(date)'
python3 '$SCRIPT' \
    --dataset ppmi \
    --result-dir '$RESULT_DIR' \
    --merge
echo 'Merge done: \$(date)'
")

echo "Merge job: $MERGE_JOB (depends on $FS_ROI_JOB and $CNN_JOB)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Output:  $RESULT_DIR/group_diff/"
