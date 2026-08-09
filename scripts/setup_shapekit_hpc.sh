#!/usr/bin/env bash
# Clone ShapeKit into third_party/ and create a separate conda env 'shapekit'.
# Isolated from 'suprem' so ShapeKit deps cannot disturb warm-up pins.
#
# Usage:
#   bash scripts/setup_shapekit_hpc.sh
#
# Then:
#   bash scripts/run_shapekit_hpc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SK_DIR="$ROOT/third_party/ShapeKit"
CFG_SRC="$ROOT/configs/shapekit_vertebrae.yaml"
ENV_NAME=shapekit

log() { echo "[shapekit-setup] $*"; }

export TMPDIR="${HOME}/tmp/${SLURM_JOB_ID:-shapekit-setup}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PIP_CACHE_DIR="${HOME}/.cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$ROOT/third_party"

if [ ! -f "$CFG_SRC" ]; then
    log "ERROR: missing $CFG_SRC"
    exit 1
fi

# --- conda ------------------------------------------------------------------
if [ -f /apps/local/conda_init.sh ]; then
    # shellcheck disable=SC1091
    source /apps/local/conda_init.sh
fi
if ! command -v conda >/dev/null 2>&1; then
    log "ERROR: conda not found. source /apps/local/conda_init.sh first"
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- clone -----------------------------------------------------------------
if [ -d "$SK_DIR/.git" ]; then
    log "ShapeKit already present at $SK_DIR"
else
    log "cloning ShapeKit -> $SK_DIR"
    git clone https://github.com/BodyMaps/ShapeKit.git "$SK_DIR"
fi

# --- project config (vertebrae-only; affine ref = L1, not liver) ------------
log "installing warm-up config -> $SK_DIR/config.yaml"
cp -f "$CFG_SRC" "$SK_DIR/config.yaml"
cat > "$SK_DIR/CONFIG_NOTE_BODYMAPS_WARMUP.txt" <<EOF
config.yaml was replaced by the BodyMaps warm-up copy:
  $CFG_SRC

Changes vs upstream:
  - target_organs: [vertebrae]
  - affine_reference_file_name: vertebrae_L1.nii.gz
    (upstream default liver.nii.gz is absent from AbdomenAtlasDemoPredict)
EOF

# --- env --------------------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "conda env '$ENV_NAME' already exists — skipping create"
else
    log "creating conda env '$ENV_NAME' (python 3.10)"
    conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"

# ShapeKit pins pytorch-lightning==1.6.4 (needs pip<24.1 for metadata).
log "pinning pip<24.1 and installing ShapeKit requirements"
pip install "pip>=23.0,<24.1"
while IFS= read -r requirement || [ -n "$requirement" ]; do
    [[ -z "$requirement" || "$requirement" =~ ^[[:space:]]*# ]] && continue
    pip install "$requirement" || log "WARNING: failed to install $requirement — skipping"
done < "$SK_DIR/requirements.txt"

# Soft numpy pin helpful with torch 1.11-era stacks; do not fail setup if it conflicts.
pip install "numpy>=1.21,<1.24" || true

log "smoke-checking imports"
python - <<'PY'
import importlib
for m in ("nibabel", "numpy", "yaml", "cc3d", "scipy", "skimage"):
    importlib.import_module(m)
    print(f"  OK {m}")
PY

log "done."
log "  ShapeKit: $SK_DIR"
log "  config:   $SK_DIR/config.yaml  (from configs/shapekit_vertebrae.yaml)"
log "  env:      conda activate $ENV_NAME"
log "  next:     bash scripts/run_shapekit_hpc.sh"
