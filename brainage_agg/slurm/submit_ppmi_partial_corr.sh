#!/bin/bash
# SLURM array job: T2-M2 — Partial Spearman correlation with an UPDRS target.
# PD subjects only. One task per extractor (0 = FS ROI, 1-240 = CNN variants).
#
# Prerequisites:
#   python3 -m brainage_agg.data.add_updrs   # builds manifest_with_updrs.csv
#
# Usage (from project root):
#   bash brainage_agg/slurm/submit_ppmi_partial_corr.sh [--target TARGET] [--skip-lme]
#
# TARGET can be any column in manifest_with_updrs.csv, e.g.:
#   NP3TOT_first (default, severity)
#   NP3TOT_change, NP3TOT_change_rate, NHY_change, MOCA_change  (longitudinal)
#
# Results: brainage_agg/ppmi_outputs/partial_corr/          (NP3TOT_first)
#          brainage_agg/ppmi_outputs/partial_corr_<TARGET>/ (all others)

set -euo pipefail

PROJECT_ROOT="$PWD"
SCRIPT="$PROJECT_ROOT/brainage_agg/analysis/partial_corr.py"
RESULT_DIR="$PROJECT_ROOT/brainage_agg/ppmi_outputs"
FEATURE_DIR="$PROJECT_ROOT/PPMI_data/features"
LOG_DIR="$PROJECT_ROOT/brainage_agg/slurm/logs"

mkdir -p "$LOG_DIR"

# Parse --target from args; collect remaining args as EXTRA_ARGS
TARGET="NP3TOT_first"
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        *)        EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# Output directory: keep legacy name for NP3TOT_first, use target suffix otherwise
if [[ "$TARGET" == "NP3TOT_first" ]]; then
    OUT_DIR="$RESULT_DIR/partial_corr"
else
    OUT_DIR="$RESULT_DIR/partial_corr_${TARGET}"
fi

echo "Target:     $TARGET"
echo "Output dir: $OUT_DIR"

N_CNN=$(ls "$FEATURE_DIR/features__"*.npz 2>/dev/null | grep -v freesurfer_roi | wc -l)
N_TOTAL=$((1 + N_CNN))
ARRAY_END=$((N_TOTAL - 1))

echo "Submitting PPMI partial_corr array: 0-${ARRAY_END} (${N_TOTAL} extractors)"

# FS ROI (task 0): StatsmodelsLME takes ~30 min for 657 features
FS_ROI_JOB=$(sbatch --parsable \
    --job-name=ppmi_pcorr_fsroi \
    --output="$LOG_DIR/ppmi_pcorr_%j_0.out" \
    --error="$LOG_DIR/ppmi_pcorr_%j_0.err" \
    --partition=all \
    --cpus-per-task=4 \
    --mem=16G \
    --time=01:00:00 \
    --wrap="
cd '$PROJECT_ROOT'
source .venv/bin/activate
echo 'Task 0 / ${ARRAY_END}: \$(date)'
python3 '$SCRIPT' \
    --result-dir '$RESULT_DIR' \
    --feature-dir '$FEATURE_DIR' \
    --out-dir '$OUT_DIR' \
    --target '$TARGET' \
    --extractor-idx 0 \
    $EXTRA_ARGS
echo 'Done: \$(date)'
")

echo "FS ROI job: $FS_ROI_JOB (60 min)"

# CNN tasks (1-N): VectorizedOLS, ~5-10 min each
CNN_JOB=$(sbatch --parsable \
    --job-name=ppmi_pcorr_cnn \
    --output="$LOG_DIR/ppmi_pcorr_%A_%a.out" \
    --error="$LOG_DIR/ppmi_pcorr_%A_%a.err" \
    --partition=all \
    --array="1-${ARRAY_END}" \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:20:00 \
    --wrap="
cd '$PROJECT_ROOT'
source .venv/bin/activate
echo 'Task \$SLURM_ARRAY_TASK_ID / ${ARRAY_END}: \$(date)'
python3 '$SCRIPT' \
    --result-dir '$RESULT_DIR' \
    --feature-dir '$FEATURE_DIR' \
    --out-dir '$OUT_DIR' \
    --target '$TARGET' \
    --extractor-idx \$SLURM_ARRAY_TASK_ID \
    $EXTRA_ARGS
echo 'Done: \$(date)'
")

echo "CNN array job: $CNN_JOB (20 min, tasks 1-${ARRAY_END})"

# Merge
MERGE_JOB=$(sbatch --parsable \
    --job-name=ppmi_pcorr_merge \
    --output="$LOG_DIR/ppmi_pcorr_merge_%j.out" \
    --error="$LOG_DIR/ppmi_pcorr_merge_%j.err" \
    --partition=all \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:10:00 \
    --dependency="afterok:${FS_ROI_JOB}:${CNN_JOB}" \
    --wrap="
cd '$PROJECT_ROOT'
source .venv/bin/activate
echo 'Merging partial_corr results: \$(date)'
python3 '$SCRIPT' \
    --result-dir '$RESULT_DIR' \
    --out-dir '$OUT_DIR' \
    --target '$TARGET' \
    --merge
echo 'Merge done: \$(date)'
")

echo "Merge job: $MERGE_JOB (depends on $FS_ROI_JOB and $CNN_JOB)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Output:  $OUT_DIR/"
