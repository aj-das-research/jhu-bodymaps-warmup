# HPC Guide (CIAI / MBZUAI)

Reusable notes for interactive and batch jobs on CIAI login nodes (`ciai-login-*`).
Keep this folder **cluster-general** — put project-specific workflows in each repo’s own README.

Official policies (on-campus / VPN): [HPC Wiki — Policies](https://hpc.mbzuai.ac.ae/wiki/policies.html)

## Quick start

```bash
# 1) Interactive GPU (max 3h on gpu-debug-qos) — prefer salloc
salloc -p long --qos=gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 03:00:00

# 2) On gpu-XX
hostname && nvidia-smi
export TMPDIR=$HOME/tmp/${SLURM_JOB_ID:-manual}
export MPLCONFIGDIR=$TMPDIR/matplotlib
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

# 3) Conda
source /apps/local/conda_init.sh
conda activate <your-env>
```

## Docs in this folder

| File | Purpose |
|------|---------|
| [links.txt](links.txt) | Official wiki / policy URLs |
| [interactive-commands.md](interactive-commands.md) | `salloc` / `srun`, QoS limits, TMPDIR, troubleshooting |
| [batch-jobs.md](batch-jobs.md) | `sbatch`, partitions (`long`, `cscc-cpu-p`), QoS, multi-GPU notes |
| [conda.md](conda.md) | `/apps/local` conda init, modules, cache tips |

## Changelog

- **2026-08-08** — Batch jobs: `cscc-cpu-p` for &gt;3h CPU work; `gpu-debug-qos` interactive still max 3h; multi-GPU does not help CPU-only codes.
- **2026-08-07** — Interactive GPU (`gpu-debug-qos`): prefer `salloc`; document `srun --pty` step-launch failures.
- **2026-08-07** — Full `/tmp` on some nodes (e.g. `gpu-33`): use `$HOME/tmp` / `--exclude`.
- **2026-08-07** — Conda via `source /apps/local/conda_init.sh` / `module load anaconda3`.
- **2026-08-07** — Scoped this folder to general CIAI usage only (project workflows stay in each repo README).
