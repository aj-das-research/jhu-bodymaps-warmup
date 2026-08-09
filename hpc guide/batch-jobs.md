# Batch jobs on CIAI (Slurm)

Use **`sbatch`** from a login node for work that needs more than the interactive
debug walltime. Do not run heavy jobs on `ciai-login-*` itself.

Official wiki (on-campus / VPN): [Policies](https://hpc.mbzuai.ac.ae/wiki/policies.html),
[CAMD Slurm](https://hpc.mbzuai.ac.ae/wiki/camd/slurm.html).

---

## Partitions and QoS (observed on CIAI)

| Partition | Default / allowed QoS | Typical use | Notes |
|-----------|------------------------|-------------|--------|
| `long` | `gpu-12` (default), also `gpu-debug-qos` | GPU nodes (`gpu-XX`, 4× A100 each) | Interactive shells are often **routed to `gpu-debug-qos`** → **max 3h** |
| `cscc-cpu-p` | `cscc-cpu-qos` | CPU nodes (`cn-XX`) | Good for CPU-only post-processing; long walltimes accepted in practice |
| `cscc-gpu-p` | (account-dependent) | CSCC GPU pool | Separate from `long` |

**Important:** more GPUs only help GPU code. CPU-only tools (e.g. ShapeKit) need
**longer walltime**, not `--gres=gpu:N`. With a single case, ShapeKit also uses
only **one** process, so `--cpus-per-task` above ~8–16 is mostly for memory/headroom.

Check what *you* can use:

```bash
sinfo -s
scontrol show partition long | grep -i AllowQos
scontrol show partition cscc-cpu-p | grep -i AllowQos
groups   # e.g. cscc-users → can use cscc-cpu-p

# Dry-run (does not queue a real job on most Slurm builds)
sbatch --test-only -p cscc-cpu-p --cpus-per-task=16 --mem=64G -t 12:00:00 --wrap='hostname'
sbatch --test-only -p long --qos=gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 03:00:00 --wrap='hostname'
```

If `--qos=gpu-12` returns `Invalid qos specification`, your account may only have
`gpu-debug-qos` on `long` — use **`cscc-cpu-p`** (or ask HPC for the right QoS)
for jobs longer than 3h.

---

## Multi-GPU note

`long` nodes expose up to **4 GPUs** (`gpu:a100-sxm4-40gb:4`). Example when your
account allows production GPU QoS:

```bash
#SBATCH -p long
#SBATCH --qos=gpu-12          # only if your account allows it
#SBATCH --gres=gpu:2          # or gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
```

Interactive multi-GPU under `gpu-debug-qos` is still capped at **3 hours**.

---

## Minimal CPU batch template

```bash
#!/bin/bash
#SBATCH -J my-cpu-job
#SBATCH -p cscc-cpu-p
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

mkdir -p logs
export TMPDIR=$HOME/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR"

source /apps/local/conda_init.sh
conda activate <your-env>
# your command here
```

Submit / monitor:

```bash
mkdir -p logs
sbatch job.sh
squeue -u $USER
tail -f logs/<jobname>-<jobid>.out
scancel <jobid>
```

---

## Minimal GPU batch template (production QoS)

Only if `sbatch --test-only ... --qos=gpu-12` succeeds for your account:

```bash
#!/bin/bash
#SBATCH -J my-gpu-job
#SBATCH -p long
#SBATCH --qos=gpu-12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

mkdir -p logs
export TMPDIR=$HOME/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
nvidia-smi
source /apps/local/conda_init.sh
conda activate <your-env>
```

---

## Interactive vs batch

| Need | Use |
|------|-----|
| Short GPU debug (≤3h) | `salloc` + `gpu-debug-qos` — see [interactive-commands.md](interactive-commands.md) |
| CPU work &gt;3h | `sbatch` on `cscc-cpu-p` |
| GPU training &gt;3h | `sbatch` on `long` with a production QoS your account allows (`gpu-12` if permitted) |
