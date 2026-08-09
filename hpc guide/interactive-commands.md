# Interactive jobs on CIAI (Slurm)

Use these from a **login node** (e.g. `ciai-login-2`). Do not run heavy GPU work on the login node itself.

Official reference: [HPC Wiki — Policies](https://hpc.mbzuai.ac.ae/wiki/policies.html) (on-campus / VPN).

---

## Correct interactive GPU command (debug QoS)

`gpu-debug-qos` allows **at most 3 hours**. Always set `-t` / `--time` to `03:00:00` or less.

This cluster sets Slurm `LaunchParameters=use_interactive_step`. For interactive shells, prefer **`salloc`** (not `srun --pty bash`). `srun --pty` often gets resources, then aborts the step with `task 0 launch failed: Unspecified error`.

### Recommended — `salloc` (1 GPU, 8 CPUs, 64G, up to 3h)

```bash
salloc -p long --qos=gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 03:00:00
```

With `use_interactive_step`, that should drop you into a shell **on the compute node**. Confirm:

```bash
hostname          # should be gpu-XX, not ciai-login-*
nvidia-smi
```

If you are still on the login node after `salloc`, open a shell on the allocation:

```bash
srun --jobid=$SLURM_JOB_ID --pty /bin/bash
```

Leave with `exit` (repeat if you nested shells). To free resources early: `scancel $SLURM_JOB_ID`.

Next: [conda init](conda.md) with `source /apps/local/conda_init.sh`.

### Shorter debug session

```bash
salloc -p long --qos=gpu-debug-qos --gres=gpu:1 --cpus-per-task=4 --mem=32G -t 01:00:00
```

### Optional — pin to a healthy idle node

If a specific node keeps failing step launch, exclude it or request an idle one:

```bash
salloc -p long --qos=gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 01:00:00 -w gpu-13
# or exclude a bad node:
salloc -p long --qos=gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 01:00:00 --exclude=gpu-33
```

---

## Why `srun --pty bash` aborts here

Symptom:

```text
srun: Interactive job detected (srun/salloc). Routed to QoS='gpu-debug-qos'.
srun: StepId=....0 aborted before step completely launched.
srun: error: task 0 launch failed: Unspecified error
```

What actually happened (job `153933`):

- Scheduler **did allocate** (`NodeList=gpu-33`, 1 GPU / 8 CPU / 64G).
- The **job step** that starts `bash` failed immediately (~1s).
- This is **not** a time-limit or QoS reject (those print a clear message before allocation).

Likely causes on this cluster:

1. **`use_interactive_step`** — interactive shells are meant to go through `salloc`, not a direct `srun --pty` allocate+launch.
2. **Node/prolog glitch** on the assigned node (here `gpu-33`) while launching the step.
3. Less often: broken X11/PTY from an IDE terminal (`PrologFlags` includes `X11`). Try from a plain SSH session if `salloc` still fails.

### Fallback if you still want `srun`

```bash
srun -p long --qos=gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 03:00:00 --pty /bin/bash
```

Prefer `salloc` first.

---

## `/tmp` full on compute node (`TMPDIR` warning)

Successful `salloc` may still print:

```text
error: Unable to create TMPDIR [/tmp/slurm-<user>-<jobid>]: No space left on device
error: Setting TMPDIR to /tmp
```

Seen on `gpu-33` (job `153935`). The session still starts, but local `/tmp` is full — this also helps explain earlier `srun --pty` step aborts on that node.

Mitigations:

```bash
# Prefer another node next time
salloc -p long --qos=gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 03:00:00 --exclude=gpu-33

# Inside any session: use home for temp
export TMPDIR=$HOME/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
df -h /tmp "$TMPDIR"
```

---

## What was wrong in the failed attempts

| Attempt | Problem |
|---------|---------|
| `srun ... -t 03:00:00 --pty bash` → step abort | Allocation OK (e.g. on `gpu-33`); interactive step/PTY launch failed. Use `salloc`. Node `/tmp` may be full. |
| `srun --qos=gpu-debug-qos --gres=gpu:1 --pty bash` (no `-t`) | **Rejected**: default time &gt; 180 min for `gpu-debug-qos`. Must pass `-t 3:00:00` or less. |
| `salloc` on `gpu-33` + TMPDIR errors | Allocation OK; node `/tmp` full — set `TMPDIR` under `$HOME` or `--exclude=gpu-33`. |

Use `--qos=gpu-debug-qos` (or `-q gpu-debug-qos`). Prefer the long form `--qos=` for clarity.

---

## Useful status / inspection commands

```bash
# Your jobs
squeue -u $USER

# Why a job is pending / where it ran
squeue -j <JOBID> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Partitions and node state
sinfo -s
sinfo -p long -N -l

# Recent job history for you
sacct -u $USER --starttime=today --format=JobID,JobName,Partition,QOS,State,Elapsed,ExitCode,AllocTRES%40

# Cancel a stuck job
scancel <JOBID>
```

---

## Batch job (non-interactive)

For runs longer than 3h, use **`sbatch`** — see **[batch-jobs.md](batch-jobs.md)**.

Short version for CPU-only work (`cscc-users`):

```bash
sbatch -p cscc-cpu-p --cpus-per-task=16 --mem=128G -t 12:00:00 --wrap='hostname'
```

GPU interactive debug stays on `gpu-debug-qos` (≤3h). Extra GPUs do not speed up CPU-only codes.

---

## Quick checklist — interactive GPU session

1. On a login node (`ciai-login-*`), not already inside another allocation.
2. Request with **`salloc`** (not `srun --pty`): partition `long`, `--qos=gpu-debug-qos`, `--gres=gpu:1`, `-t` ≤ `03:00:00`.
3. Prompt should show `gpu-XX`. Run `hostname` and `nvidia-smi`.
4. If `/tmp` warnings appear: `export TMPDIR=$HOME/tmp/$SLURM_JOB_ID && mkdir -p "$TMPDIR"`, or `--exclude` that node next time.
5. Init conda: `source /apps/local/conda_init.sh` (see [conda.md](conda.md)).
6. When done: `exit` (or `scancel $SLURM_JOB_ID`).
