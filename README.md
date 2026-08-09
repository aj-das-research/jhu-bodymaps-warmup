# JHU BodyMaps RA warm-up: vertebrae segmentation + ShapeKit baseline

Working repo for the BodyMaps Research Assistant warm-up (SuPreM "Apply to Vertebrae
Segmentation"). Full spec in [docs/official_warmup_spec.md](docs/official_warmup_spec.md).

Pipeline: run the lab's pretrained Swin UNETR (TotalSegmentator vertebra classes, C1–L5)
on the two demo CT scans to produce `AbdomenAtlasDemoPredict/`, audit label errors, then
run ShapeKit as the standing post-processing baseline (`AbdomenAtlasDemoPredict_shapekit/`).
Static raw-vs-ShapeKit slice panels live under `reports/figures/raw_vs_shapekit/`.

## Layout
```
postprocessing_vertebrae.py     THE deliverable: evidence-edited refinement (stages 1-3 + audit)
scripts/download_data.sh        fetch demo CTs + checkpoint from cs.jhu.edu (not in git)
scripts/setup_env_hpc.sh        CIAI: conda suprem + third_party/SuPreM (idempotent)
scripts/run_inference_hpc.sh    CIAI: official --customize inference -> AbdomenAtlasDemoPredict/
scripts/audit_predictions.py    Phase A: volume / components / SIZE / ORDER flags
scripts/check_pred_grid.py      CT vs prediction grid check (ITK-SNAP overlay)
scripts/setup_shapekit_hpc.sh   Phase B: conda shapekit + third_party/ShapeKit
scripts/run_shapekit_hpc.sh     Phase B: ShapeKit CPU baseline + re-audit
scripts/sbatch_shapekit.sh      Phase B: long CPU batch job for ShapeKit
scripts/compare_audits.py       Phase B: before vs after audit diff
scripts/plot_slice_compare_panel.py   raw vs ShapeKit 3×2 diagnostic panels
scripts/render_error_panels.py  optional audit-driven PNG panels
configs/shapekit_vertebrae.yaml ShapeKit config (vertebrae-only; affine = L1)
third_party/SuPreM/             upstream clone (gitignored; created by setup_env_hpc.sh)
third_party/ShapeKit/           upstream clone (gitignored; created by setup_shapekit_hpc.sh)
notebooks/BodyMaps_RA_warmup.ipynb   Colab: env, inference, error audit, refinement, zip
docs/official_warmup_spec.md    the official tutorial, transcribed
docs/JHU_BodyMaps_RA_Prep_Guide.md   research briefing + positioning notes
hpc guide/                      reusable CIAI cluster notes (not project-specific)
reports/figures/raw_vs_shapekit/     diagnostic PNGs
data/                           gitignored; created by the download script
```

## HPC (CIAI) — env + inference

Cluster how-to (salloc, conda init, TMPDIR): [hpc guide/README.md](hpc%20guide/README.md).

Project workflow on a GPU node:

```bash
# Interactive GPU (prefer salloc; max 3h on gpu-debug-qos)
salloc -p long --qos=gpu-debug-qos --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 03:00:00

cd ~/projects/jhu-bodymaps-warmup
bash scripts/download_data.sh      # once, if data/ not present (login or GPU OK)
bash scripts/setup_env_hpc.sh      # conda env 'suprem' + third_party/SuPreM
bash scripts/run_inference_hpc.sh  # -> AbdomenAtlasDemoPredict/
```

`setup_env_hpc.sh` is idempotent: skips env create and pip groups that already match.
`run_inference_hpc.sh` keeps caches under `$HOME/tmp` (some nodes have full `/tmp`),
loads the ~720 MB checkpoint on CPU first, then runs with `--customize --device cuda`.

**Do not Ctrl-C** during checkpoint load — it can look idle for a few minutes on NFS.

Expected outputs:

```text
AbdomenAtlasDemoPredict/
  BDMAP_00000006/
    combined_labels.nii.gz
    segmentations/vertebrae_C1.nii.gz ... vertebrae_L5.nii.gz
  BDMAP_00000031/
    ...
```

After inference, activate the env in your shell if needed:

```bash
source /apps/local/conda_init.sh
conda activate suprem
```

### Phase A — audit + grid check (CPU, `suprem` env)

```bash
conda activate suprem
python scripts/audit_predictions.py \
    --pred_dir AbdomenAtlasDemoPredict \
    --report reports/audit_before.json

python scripts/check_pred_grid.py \
    --ct_dir data/AbdomenAtlasDemo \
    --pred_dir AbdomenAtlasDemoPredict \
    --report reports/grid_check.json
```

If grids match, ITK-SNAP can overlay without resampling. If not, keep submission
preds on their native grid and resample a *viewing copy* only.

### Phase B — ShapeKit baseline (CPU, separate `shapekit` env)

ShapeKit is **CPU-only** (extra GPUs do not help). One large case can exceed the
3h `gpu-debug-qos` interactive limit — submit a **batch CPU job** instead:

```bash
# once
bash scripts/setup_shapekit_hpc.sh

# from a login node (ciai-login-*): finishes incomplete case(s), then audits
mkdir -p reports/slurm
CASE=BDMAP_00000031 sbatch scripts/sbatch_shapekit.sh

squeue -u $USER
tail -f reports/slurm/shapekit-vert-<JOBID>.out
# when done:
conda activate suprem
python scripts/compare_audits.py \
    --before reports/audit_before.json \
    --after  reports/audit_shapekit.json
```

Short interactive runs (≤3h, small cases only):

```bash
bash scripts/run_shapekit_hpc.sh
# or one case: CASE=BDMAP_00000006 bash scripts/run_shapekit_hpc.sh
```

Falsifiable ask: ShapeKit should crush FRAGMENTED speckles; surviving SIZE / ordering
issues mark where delete-only cleanup is insufficient.

**Label-id note:** SuPreM `combined_labels` uses vertebrae ids **1–24** (L5=1 … C1=24).
ShapeKit rewrites the same anatomy with AbdomenAtlas ids **26–49**. Per-mask files
and audits are unaffected; ITK-SNAP / panels may show different palette colors for
the same vertebra.

Cluster batch notes: [hpc guide/batch-jobs.md](hpc%20guide/batch-jobs.md).

### Diagnostic panels — raw vs ShapeKit

```bash
conda activate suprem
python scripts/plot_slice_compare_panel.py
# -> reports/figures/raw_vs_shapekit/BDMAP_00000006/vertebrae_C1/axial_319_panel.png

# every axial plane where raw XOR ShapeKit is non-empty:
python scripts/plot_slice_compare_panel.py --diff_slices
```

See [reports/figures/README.md](reports/figures/README.md).

### Phase C — evidence-edited refinement (ours)

`postprocessing_vertebrae.py` (repo root) is the submission deliverable. Principle:
the CT is the only editor, anatomy only audits. Stage 1 triages disconnected
components by image evidence (speck / soft / bone-near bridged through a CT-bone
corridor / bone-far pooled); stage 2 re-arbitrates SUSPECT bands only: the column is
segmented by disc minima of a mean-HU profile sampled perpendicular to the body
centerline (spacing-regularized DP, anchored on audit-clean neighbors), then a
uniform-speed geodesic competition through the bone domain reassigns mass so fronts
meet at low-HU clefts. C1/C2 skipped (no disc), unresolved disc counts skip with a
flag, and a band reverts unless the audit strictly improves. Stage 3 fills enclosed
holes and applies bounded volume-preserving SDT smoothing (Dice >= 0.97 guard).

```bash
conda activate suprem   # needs: numpy scipy nibabel scikit-image connected-components-3d
python postprocessing_vertebrae.py \
    --pred_dir AbdomenAtlasDemoPredict \
    --ct_root  data/AbdomenAtlasDemo \
    --out_dir  AbdomenAtlasDemoPredict_refined \
    --report_dir reports
python scripts/compare_audits.py --before reports/audit_before.json --after reports/audit_refined.json
```

Results (identical parameters both cases; QA in `reports/*_postprocessing_qa.json`):

| case | before | ShapeKit | ours |
|---|---|---|---|
| BDMAP_00000006 | 10 FRAG, 1 ORDER | clean, but deletes C1 tips on CT-certified bone | all clean, tips preserved |
| BDMAP_00000031 | 21 FRAG, L1 SIZE 0.44 | 0 FRAG, L1 worsens to 11.6 cm3 (SIZE 0.20) | all clean, L1 restored to 64 cm3 at detected disc planes |

Figures: `reports/figures/refinement/` (3-way 3D + volumes, sagittal band overlay,
density profile with DP-chosen disc cuts).

## Local (view in ITK-SNAP)
```bash
bash scripts/download_data.sh
# Open Main Image:   data/AbdomenAtlasDemo/BDMAP_00000031/ct.nii.gz
# Open Segmentation: AbdomenAtlasDemoPredict/BDMAP_00000031/combined_labels.nii.gz
# (only if check_pred_grid.py reports MATCH)
```

## Colab (inference)
Open `notebooks/BodyMaps_RA_warmup.ipynb` in Colab (GPU runtime). It fetches data and
checkpoint directly from cs.jhu.edu, sets up the pinned python=3.9 env, runs inference,
and walks through the official warm-up tutorial steps.
