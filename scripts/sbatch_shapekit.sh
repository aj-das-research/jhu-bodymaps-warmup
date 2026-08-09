#!/bin/bash
#SBATCH -J shapekit-vert
#SBATCH -p cscc-cpu-p
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o reports/slurm/%x-%j.out
#SBATCH -e reports/slurm/%x-%j.err
#
# Batch ShapeKit on CIAI CPU partition (longer than gpu-debug-qos 3h interactive).
# ShapeKit is CPU-only; extra GPUs do not speed it up. With one case it uses 1 worker.
#
# Submit from a login node (ciai-login-*):
#   cd ~/projects/jhu-bodymaps-warmup
#   mkdir -p reports/slurm
#   CASE=BDMAP_00000031 sbatch scripts/sbatch_shapekit.sh
#
# Defaults to the incomplete demo case. Override:
#   CASE=BDMAP_00000031 sbatch scripts/sbatch_shapekit.sh
#   CASES="BDMAP_00000006 BDMAP_00000031" sbatch scripts/sbatch_shapekit.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f reports/slurm/shapekit-vert-<JOBID>.out
#   tail -f reports/shapekit_logs/postprocessing.log

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p reports/slurm reports/shapekit_logs

export TMPDIR="${HOME}/tmp/${SLURM_JOB_ID:-shapekit-batch}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export MPLCONFIGDIR="${HOME}/.cache/matplotlib"
export PIP_CACHE_DIR="${HOME}/.cache/pip"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

# Propagate case filter into the run script (default: finish the incomplete case).
export CASE="${CASE:-BDMAP_00000031}"
export CASES="${CASES:-$CASE}"
export CPU_COUNT="${CPU_COUNT:-${SLURM_CPUS_PER_TASK:-16}}"
export SKIP_AUDIT="${SKIP_AUDIT:-0}"

echo "[sbatch-shapekit] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[sbatch-shapekit] partition=${SLURM_JOB_PARTITION:-?} cpus=${CPU_COUNT} mem=${SLURM_MEM_PER_NODE:-?}MB"
echo "[sbatch-shapekit] cases=${CASES}"
echo "[sbatch-shapekit] start=$(date -Is)"

if [ -f /apps/local/conda_init.sh ]; then
    # shellcheck disable=SC1091
    source /apps/local/conda_init.sh
fi

bash scripts/run_shapekit_hpc.sh

echo "[sbatch-shapekit] end=$(date -Is)"
echo "[sbatch-shapekit] next: conda activate suprem && python scripts/compare_audits.py \\"
echo "         --before reports/audit_before.json --after reports/audit_shapekit.json"
