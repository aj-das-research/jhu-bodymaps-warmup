# JHU BodyMaps RA warm-up: vertebrae segmentation post-processing

Working repo for the BodyMaps Research Assistant warm-up (SuPreM "Apply to Vertebrae
Segmentation"). Full spec in [docs/official_warmup_spec.md](docs/official_warmup_spec.md).

Pipeline: run the lab's pretrained Swin UNETR (TotalSegmentator vertebra classes, C1-L5)
on the two demo CT scans to produce `AbdomenAtlasDemoPredict/`, then reduce the label
errors with `postprocessing_vertebrae.py` (largest-component filtering, fragment removal,
hole closing, superior-inferior ordering checks).

## Layout
```
postprocessing_vertebrae.py     the deliverable script (per-mask + combined-map modes)
scripts/download_data.sh        fetch demo CTs + checkpoint from cs.jhu.edu (not in git)
notebooks/BodyMaps_RA_warmup.ipynb   Colab: env, inference, error audit, refinement, zip
docs/official_warmup_spec.md    the official tutorial, transcribed
docs/JHU_BodyMaps_RA_Prep_Guide.md   research briefing + positioning notes
data/                           gitignored; created by the download script
```

## Local (view in ITK-SNAP)
```bash
bash scripts/download_data.sh
# then: ITK-SNAP -> File -> Open Main Image -> data/AbdomenAtlasDemo/BDMAP_00000006/ct.nii.gz
```

## Colab (inference)
Open `notebooks/BodyMaps_RA_warmup.ipynb` in Colab (GPU runtime). It fetches data and
checkpoint directly from cs.jhu.edu, sets up the pinned python=3.9 env, runs inference,
audits vertebra errors, applies post-processing, and zips the refined folder.

## Post-processing usage
```bash
python postprocessing_vertebrae.py \
    --pred_dir AbdomenAtlasDemoPredict \
    --out_dir  AbdomenAtlasDemoPredict_refined \
    --min_voxels 200 --closing_iters 1
```

## Submission checklist
- [ ] refined `AbdomenAtlasDemoPredict` (compressed)
- [ ] `postprocessing_vertebrae.py`
- [ ] CV -> zzhou82@jh.edu (state position: Research Assistant)
