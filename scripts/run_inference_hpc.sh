#!/usr/bin/env bash
# Run official SuPreM vertebrae inference on CIAI (Phase 3).
#
# Prerequisites:
#   - interactive GPU allocation (salloc ...)
#   - bash scripts/setup_env_hpc.sh already succeeded
#   - data/ populated via bash scripts/download_data.sh
#
# Usage (from anywhere):
#   bash scripts/run_inference_hpc.sh
#
# Writes predictions into:
#   <repo>/AbdomenAtlasDemoPredict/{BDMAP_*}/combined_labels.nii.gz
#   <repo>/AbdomenAtlasDemoPredict/{BDMAP_*}/segmentations/vertebrae_*.nii.gz

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INFER_DIR="$ROOT/third_party/SuPreM/direct_inference"
CKPT="$INFER_DIR/pretrained_checkpoints/swin_unetr_totalsegmentator_vertebrae.pth"
DATA="$ROOT/data/AbdomenAtlasDemo"
SAVE="$ROOT/AbdomenAtlasDemoPredict"
ENV_NAME=suprem

log() { echo "[infer] $*"; }

# --- Keep caches off full node /tmp (gpu-33 issue) -------------------------
export TMPDIR="${HOME}/tmp/${SLURM_JOB_ID:-infer}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
export XDG_CACHE_HOME="$TMPDIR/xdg-cache"
export TORCH_HOME="$TMPDIR/torch"
export HF_HOME="$TMPDIR/hf"
export TRANSFORMERS_CACHE="$TMPDIR/hf/transformers"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TRANSFORMERS_CACHE"

log "ROOT=$ROOT"
log "TMPDIR=$TMPDIR"
log "hostname=$(hostname)"

# --- Conda env -------------------------------------------------------------
if [ -f /apps/local/conda_init.sh ]; then
    # shellcheck disable=SC1091
    source /apps/local/conda_init.sh
fi
if ! command -v conda >/dev/null 2>&1; then
    log "ERROR: conda not found. Run: source /apps/local/conda_init.sh"
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- Sanity checks ---------------------------------------------------------
if [ ! -d "$INFER_DIR" ]; then
    log "ERROR: SuPreM missing at $INFER_DIR — run bash scripts/setup_env_hpc.sh"
    exit 1
fi
if [ ! -e "$CKPT" ]; then
    log "ERROR: checkpoint missing at $CKPT"
    log "       run: bash scripts/download_data.sh && bash scripts/setup_env_hpc.sh"
    exit 1
fi
if [ ! -d "$DATA" ]; then
    log "ERROR: demo data missing at $DATA — run bash scripts/download_data.sh"
    exit 1
fi

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available — are you inside an salloc GPU job?"
print(f"[infer] torch {torch.__version__}  cuda=True  ({torch.cuda.get_device_name(0)})")
PY

log "data cases:"
find "$DATA" -mindepth 1 -maxdepth 1 -type d | sort | sed 's|^|  |'

mkdir -p "$SAVE"
cd "$INFER_DIR"

log "starting inference (checkpoint load ~720MB — wait, do not Ctrl-C)"
log "  checkpoint: $CKPT"
log "  data:       $DATA"
log "  save_dir:   $SAVE"

# Official warm-up flags + explicit CUDA device for customize path.
python -W ignore inference.py \
    --save_dir "$SAVE" \
    --checkpoint "$CKPT" \
    --data_root_path "$DATA" \
    --customize \
    --device cuda \
    --num_workers 4

log "done. Predictions under: $SAVE"
find "$SAVE" -maxdepth 3 \( -name 'combined_labels.nii.gz' -o -name 'vertebrae_*.nii.gz' \) | sort | head -60
log "case count: $(find "$SAVE" -mindepth 1 -maxdepth 1 -type d | wc -l)"
