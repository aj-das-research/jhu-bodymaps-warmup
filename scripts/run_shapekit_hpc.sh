#!/usr/bin/env bash
# Run BodyMaps ShapeKit (CPU) on SuPreM vertebra predictions, then re-audit.
#
# Prerequisites:
#   bash scripts/setup_shapekit_hpc.sh
#   AbdomenAtlasDemoPredict/ already produced by run_inference_hpc.sh
#
# Usage:
#   bash scripts/run_shapekit_hpc.sh
#   CPU_COUNT=4 bash scripts/run_shapekit_hpc.sh
#   SKIP_AUDIT=1 bash scripts/run_shapekit_hpc.sh
#
# Resume only the incomplete case (case 6 done, case 31 killed by walltime):
#   rm -rf AbdomenAtlasDemoPredict_shapekit/BDMAP_00000031   # required; stale copy looks "done"
#   CASE=BDMAP_00000031 bash scripts/run_shapekit_hpc.sh
#
# Output is gitignored (AbdomenAtlasDemoPredict*). No GPU required.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SK_DIR="$ROOT/third_party/ShapeKit"
INPUT_FOLDER="${INPUT_FOLDER:-$ROOT/AbdomenAtlasDemoPredict}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-$ROOT/AbdomenAtlasDemoPredict_shapekit}"
LOG_FOLDER="${LOG_FOLDER:-$ROOT/reports/shapekit_logs}"
CPU_COUNT="${CPU_COUNT:-8}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"
# Optional: process only these case IDs (space-separated), via ShapeKit --csv
CASE="${CASE:-}"
CASES="${CASES:-$CASE}"

log() { echo "[shapekit-run] $*"; }

export TMPDIR="${HOME}/tmp/${SLURM_JOB_ID:-shapekit-run}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export MPLCONFIGDIR="${HOME}/.cache/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$LOG_FOLDER" "$OUTPUT_FOLDER"

if [ ! -f "$SK_DIR/main.py" ]; then
    log "ERROR: ShapeKit not found at $SK_DIR"
    log "  run: bash scripts/setup_shapekit_hpc.sh"
    exit 1
fi
if [ ! -d "$INPUT_FOLDER" ]; then
    log "ERROR: input folder missing: $INPUT_FOLDER"
    exit 1
fi

# --- conda: shapekit --------------------------------------------------------
if [ -f /apps/local/conda_init.sh ]; then
    # shellcheck disable=SC1091
    source /apps/local/conda_init.sh
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate shapekit

# Sanity: affine reference from config must exist for every case
AFFINE_REF="$(python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("$SK_DIR/config.yaml").read_text())
print(cfg.get("affine_reference_file_name", "liver.nii.gz"))
PY
)"

log "checking affine reference '$AFFINE_REF' under each case ..."
missing=0
for case_dir in "$INPUT_FOLDER"/*/; do
    [ -d "$case_dir" ] || continue
    ref_path="${case_dir}segmentations/${AFFINE_REF}"
    if [ ! -f "$ref_path" ]; then
        log "ERROR: missing $ref_path"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    log "Fix: configs/shapekit_vertebrae.yaml sets affine_reference_file_name: vertebrae_L1.nii.gz"
    log "     re-run: bash scripts/setup_shapekit_hpc.sh"
    exit 1
fi

CSV_ARGS=()
if [ -n "$CASES" ]; then
    CSV_PATH="$LOG_FOLDER/shapekit_cases_$$.csv"
    {
        echo "Inference ID"
        for cid in $CASES; do
            echo "$cid"
        done
    } > "$CSV_PATH"
    CSV_ARGS=(--csv "$CSV_PATH")
    log "cases:  $CASES  (csv filter $CSV_PATH)"
    # ShapeKit treats any non-empty output segmentations/ as done — wipe stale copies
    # for selected cases so a walltime-killed run can actually restart.
    for cid in $CASES; do
        stale="$OUTPUT_FOLDER/$cid"
        if [ -d "$stale" ]; then
            log "removing stale/incomplete output: $stale"
            rm -rf "$stale"
        fi
    done
fi

log "input:  $INPUT_FOLDER"
log "output: $OUTPUT_FOLDER"
log "logs:   $LOG_FOLDER"
log "cpus:   $CPU_COUNT"
log "config: $SK_DIR/config.yaml (affine ref = $AFFINE_REF)"

cd "$SK_DIR"
python -W ignore main.py \
    --input_folder  "$INPUT_FOLDER" \
    --output_folder "$OUTPUT_FOLDER" \
    --cpu_count     "$CPU_COUNT" \
    --log_folder    "$LOG_FOLDER" \
    "${CSV_ARGS[@]}"

log "ShapeKit finished."

if [ "$SKIP_AUDIT" = "1" ]; then
    log "SKIP_AUDIT=1 — not running audit"
    exit 0
fi

# --- re-audit with suprem (has our audit deps / pinned stack) ---------------
log "re-auditing with conda env 'suprem' ..."
conda activate suprem
cd "$ROOT"
python scripts/audit_predictions.py \
    --pred_dir "$OUTPUT_FOLDER" \
    --report reports/audit_shapekit.json

log "done."
log "  ShapeKit out: $OUTPUT_FOLDER"
log "  audit:        $ROOT/reports/audit_shapekit.json"
log "  compare vs:   $ROOT/reports/audit_before.json"
log "  (optional)    python scripts/compare_audits.py \\"
log "                  --before reports/audit_before.json \\"
log "                  --after  reports/audit_shapekit.json"
log "  Phase-B ask:  did FRAGMENTED drop? did L1 SIZE(0.44) / T9 volume dip survive?"
