# Conda on CIAI

Shared Anaconda lives under `/apps/local`. Initialize **after** you are on a compute node (post-`salloc`), or on the login node for light env management only.

---

## Init (recommended)

```bash
source /apps/local/conda_init.sh
```

Verify:

```bash
conda --version
conda env list
which python
```

Activate an environment:

```bash
conda activate <env_name>
```

---

## Alternatives

```bash
# Same base install via Lmod
module load anaconda3

# Anaconda with the 3.10 tree
source /apps/local/init_conda_3.10
```

Paths (for reference):

| Item | Path |
|------|------|
| Init script | `/apps/local/conda_init.sh` |
| Anaconda root | `/apps/local/anaconda3` |
| Anaconda 3.10 | `/apps/local/anaconda3.10` |
| Modulefile | `module load anaconda3` → `/apps/local/anaconda3` |

---

## TMPDIR / cache tip (compute nodes)

Some GPU nodes have a full `/tmp` (e.g. `gpu-33`). Point temp and caches at home before conda/pip/matplotlib work:

```bash
export TMPDIR=$HOME/tmp/${SLURM_JOB_ID:-manual}
export TMP="$TMPDIR" TEMP="$TMPDIR"
export MPLCONFIGDIR=$TMPDIR/matplotlib
export PIP_CACHE_DIR=$HOME/.cache/pip
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$PIP_CACHE_DIR"
```

---

## Typical session sequence

```bash
# 1) From login node
salloc -p long --qos=gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 03:00:00

# 2) On compute node (prompt shows gpu-XX)
hostname && nvidia-smi
export TMPDIR=$HOME/tmp/$SLURM_JOB_ID && mkdir -p "$TMPDIR"
source /apps/local/conda_init.sh
conda activate <env_name>

# 3) Run work, then leave
exit
```
