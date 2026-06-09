#!/bin/bash
# SLURM array job: Ridge regression predicting all clinical targets from aggregated features.
# PD subjects only. One task per extractor (0 = FS ROI, 1-240 = CNN variants).
# Each task runs all 9 targets × their matched aggregations (severity→static, progression→change).
#
# Targets: NP3TOT_first/change/change_rate, NHY_first/change, MOCA_first/last/change,
#          disease_duration_first
#
# Prerequisites:
#   python3 -m brainage_agg.data.add_updrs   # builds manifest_with_updrs.csv
#
# Usage (from project root):
#   bash brainage_agg/slurm/submit_ppmi_updrs.sh
#
# Results: brainage_agg/ppmi_outputs/updrs_results.csv

set -euo pipefail

PROJECT_ROOT="$PWD"
SCRIPT="$PROJECT_ROOT/brainage_agg/experiment/run_updrs.py"
LOG_DIR="$PROJECT_ROOT/brainage_agg/slurm/logs"

mkdir -p "$LOG_DIR"

# Count extractors
N_CNN=$(ls "$PROJECT_ROOT/PPMI_data/features/features__"*.npz 2>/dev/null \
        | grep -v freesurfer_roi | wc -l)
N_TOTAL=$((1 + N_CNN))
ARRAY_END=$((N_TOTAL - 1))

echo "Submitting PPMI UPDRS regression array: 0-${ARRAY_END} (${N_TOTAL} extractors)"

# Array job: all extractors (FS ROI is fast for Ridge regression, ~2-5 min each)
JOB=$(sbatch --parsable \
    --job-name=ppmi_updrs \
    --output="$LOG_DIR/ppmi_updrs_%A_%a.out" \
    --error="$LOG_DIR/ppmi_updrs_%A_%a.err" \
    --partition=all \
    --array="0-${ARRAY_END}" \
    --cpus-per-task=2 \
    --mem=8G \
    --time=01:00:00 \
    --wrap="
cd '$PROJECT_ROOT'
source .venv/bin/activate
echo 'Task \$SLURM_ARRAY_TASK_ID / ${ARRAY_END}: \$(date)'
python3 '$SCRIPT' \
    --job-id \$SLURM_ARRAY_TASK_ID \
    --config '$PROJECT_ROOT/brainage_agg/ppmi_config.yaml'
echo 'Done: \$(date)'
")

echo "Array job submitted: $JOB"

# Merge after all tasks complete
MERGE_JOB=$(sbatch --parsable \
    --job-name=ppmi_updrs_merge \
    --output="$LOG_DIR/ppmi_updrs_merge_%j.out" \
    --error="$LOG_DIR/ppmi_updrs_merge_%j.err" \
    --partition=all \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:10:00 \
    --dependency="afterok:${JOB}" \
    --wrap="
cd '$PROJECT_ROOT'
source .venv/bin/activate
echo 'Merging UPDRS results: \$(date)'
python3 '$SCRIPT' \
    --merge \
    --config '$PROJECT_ROOT/brainage_agg/ppmi_config.yaml'
echo 'Merge done: \$(date)'
")

echo "Merge job: $MERGE_JOB (depends on $JOB)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Output:  $PROJECT_ROOT/brainage_agg/ppmi_outputs/updrs_results.csv"
