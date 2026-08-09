#!/usr/bin/env bash
# Create / verify the 'suprem' conda env for the BodyMaps warm-up (official pins).
#
# Everything stays inside this repo:
#   third_party/SuPreM/          cloned upstream code
#   data/                        checkpoint + demo CTs (download_data.sh)
#
# Usage (from repo root or anywhere):
#   bash scripts/setup_env_hpc.sh
#
# Idempotent:
#   - skips conda create if env exists
#   - skips pip groups whose intended pins are already satisfied
#   - auto-sources CIAI conda if needed (/apps/local/conda_init.sh)
#
# Pitfalls handled:
#   1. Full node /tmp           -> TMPDIR + pip cache under $HOME
#   2. numpy>=1.24              -> re-pin numpy<1.24 for torch 1.11 / monai 0.9
#   3. bash readonly GROUPS     -> use PIP_GROUPS (never assign to GROUPS)
#   4. heredoc inside $(...)    -> use scripts/check_suprem_pkgs.py instead

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME=suprem
SUPREM_DIR="$ROOT/third_party/SuPreM"
CKPT="$ROOT/data/swin_unetr_totalsegmentator_vertebrae.pth"
LEGACY_SUPREM="$HOME/projects/SuPreM"
CHECK_PY="$ROOT/scripts/check_suprem_pkgs.py"

log() { echo "[setup] $*"; }

install_group() {
    local g="$1"
    case "$g" in
        torch)
            log "installing torch 1.11.0+cu113 (official pin, A100-capable)"
            pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
                --extra-index-url https://download.pytorch.org/whl/cu113
            ;;
        monai)
            log "installing monai[all]==0.9.0"
            pip install "monai[all]==0.9.0"
            ;;
        suprem_reqs)
            # pip>=24.1 rejects pytorch-lightning==1.6.4 metadata (torch>=1.8.*).
            log "pinning pip<24.1 (needed for pytorch-lightning==1.6.4)"
            pip install "pip>=23.0,<24.1"
            log "installing SuPreM requirements.txt"
            if ! pip install -r "$SUPREM_DIR/requirements.txt"; then
                log "WARNING: full requirements failed; retrying without pytorch-lightning"
                log "         (lightning is not used by direct_inference/)"
                grep -viE '^pytorch-lightning' "$SUPREM_DIR/requirements.txt" \
                    > "$TMPDIR/suprem_reqs_nole.txt"
                pip install -r "$TMPDIR/suprem_reqs_nole.txt"
                log "retrying pytorch-lightning==1.6.4 alone"
                pip install "pytorch-lightning==1.6.4" || \
                    log "WARNING: pytorch-lightning==1.6.4 still failed; continuing (inference OK)"
            fi
            ;;
        extras)
            log "installing audit/postprocessing deps (nibabel, cc3d, scipy)"
            pip install nibabel connected-components-3d scipy
            ;;
        numpy)
            log "re-pinning numpy<1.24 (required for this torch/monai era)"
            pip install "numpy>=1.21,<1.24"
            ;;
        *)
            log "WARNING: unknown install group '$g' — skipping"
            ;;
    esac
}

# --- 0. Temp / pip cache off the node /tmp ---------------------------------
export TMPDIR="${HOME}/tmp/${SLURM_JOB_ID:-setup}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export PIP_CACHE_DIR="${HOME}/.cache/pip"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$ROOT/third_party"
log "ROOT=$ROOT"
log "TMPDIR=$TMPDIR  PIP_CACHE_DIR=$PIP_CACHE_DIR"

# --- 1. Conda on PATH (CIAI: /apps/local) ----------------------------------
init_conda() {
    if command -v conda >/dev/null 2>&1; then
        return 0
    fi
    if [ -f /apps/local/conda_init.sh ]; then
        log "sourcing /apps/local/conda_init.sh"
        # shellcheck disable=SC1091
        source /apps/local/conda_init.sh
        return 0
    fi
    if command -v module >/dev/null 2>&1; then
        log "trying: module load anaconda3"
        module load anaconda3 2>/dev/null || true
    fi
    if ! command -v conda >/dev/null 2>&1; then
        log "ERROR: conda not found."
        log "  On CIAI:  source /apps/local/conda_init.sh"
        log "  Or:       module load anaconda3"
        exit 1
    fi
}

init_conda

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- 2. Create env only if missing -----------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "conda env '$ENV_NAME' already exists — skipping create"
else
    log "creating conda env '$ENV_NAME' (python 3.9)"
    conda create -n "$ENV_NAME" python=3.9 -y
fi
conda activate "$ENV_NAME"
log "active env: ${CONDA_DEFAULT_ENV:-?} (python $(python -V 2>&1 | cut -d' ' -f2))"

# --- 3. SuPreM inside this project -----------------------------------------
if [ ! -d "$SUPREM_DIR/.git" ]; then
    if [ -d "$LEGACY_SUPREM/.git" ] && [ ! -e "$SUPREM_DIR" ]; then
        log "reusing existing clone: $LEGACY_SUPREM -> $SUPREM_DIR"
        mv "$LEGACY_SUPREM" "$SUPREM_DIR"
    else
        log "cloning SuPreM into $SUPREM_DIR"
        git clone -q https://github.com/MrGiovanni/SuPreM "$SUPREM_DIR"
    fi
else
    log "SuPreM already present at $SUPREM_DIR"
fi

mkdir -p "$SUPREM_DIR/direct_inference/pretrained_checkpoints"
if [ -f "$CKPT" ]; then
    ln -sfn "$CKPT" "$SUPREM_DIR/direct_inference/pretrained_checkpoints/$(basename "$CKPT")"
    log "checkpoint linked into third_party/SuPreM/.../pretrained_checkpoints/"
else
    log "WARNING: checkpoint not found at $CKPT"
    log "         run: bash scripts/download_data.sh   then re-run this script"
fi

# --- 4. Check pins; install anything missing/wrong -------------------------
# Do NOT use the name GROUPS — bash provides a readonly GROUPS array (GIDs).
GROUPS_FILE="$TMPDIR/setup_pip_groups.$$"
set +e
python "$CHECK_PY" --groups-file "$GROUPS_FILE"
CHECK_EC=$?
set -e
if [ ! -f "$GROUPS_FILE" ]; then
    log "ERROR: package check did not write $GROUPS_FILE"
    exit 1
fi
PIP_GROUPS="$(cat "$GROUPS_FILE")"
rm -f "$GROUPS_FILE"
log "check exit=$CHECK_EC  pip groups to install: '${PIP_GROUPS:-<none>}'"

if [ -z "${PIP_GROUPS// }" ]; then
    log "skipping pip installs — all intended packages OK"
else
    log "installing missing/wrong packages now..."
    for g in $PIP_GROUPS; do
        install_group "$g"
    done
    # Dependency installs often pull numpy>=1.24; enforce the pin last.
    case " $PIP_GROUPS " in
        *" suprem_reqs "*|*" monai "*|*" extras "*|*" numpy "*)
            log "ensuring numpy>=1.21,<1.24 after dependency installs"
            pip install "numpy>=1.21,<1.24"
            ;;
    esac
fi

# --- 5. Final verification (must pass) -------------------------------------
log "final verification:"
set +e
python "$CHECK_PY"
FINAL_EC=$?
set -e
if [ "$FINAL_EC" -ne 0 ]; then
    log "ERROR: packages still missing/wrong after install (exit $FINAL_EC)"
    exit 1
fi

python - <<'PY'
import torch, monai, numpy, nibabel, cc3d, scipy
print(f"  torch      {torch.__version__}")
print(f"  cuda       {torch.cuda.is_available()}", end="")
print(f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available()
      else "  <- False is OK on login node; re-check on a GPU allocation")
print(f"  monai      {monai.__version__}")
print(f"  numpy      {numpy.__version__} (must be <1.24)")
print(f"  nibabel    {nibabel.__version__}")
print(f"  scipy      {scipy.__version__}")
print("  cc3d       ok")
assert torch.__version__.startswith("1.11.0"), torch.__version__
assert monai.__version__ == "0.9.0", monai.__version__
major, minor = (int(x) for x in numpy.__version__.split(".")[:2])
assert (major, minor) < (1, 24), numpy.__version__
PY

log "done."
log "  env:    conda activate $ENV_NAME"
log "  SuPreM: $SUPREM_DIR"
log "  Next:   run inference from third_party/SuPreM/direct_inference (Phase 3)"
