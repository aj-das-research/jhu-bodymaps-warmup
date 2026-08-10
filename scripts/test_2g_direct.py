"""Apply stage 2g alone to an existing full-frame output (fast A/B).
Usage: python scripts/test_2g_direct.py SEG.nii.gz CT.nii.gz OUT.nii.gz"""
import json, sys
from pathlib import Path
import nibabel as nib
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import postprocessing_vertebrae as pv

seg_p, ct_p, out_p = sys.argv[1:4]
img = nib.load(seg_p)
zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
vox = float(abs(np.linalg.det(img.affine[:3, :3])))
seg_full = np.asarray(img.dataobj).astype(np.uint8)
ct_full = nib.load(ct_p).get_fdata(dtype=np.float32)
nz = np.nonzero(seg_full)
pad = [int(round(25.0 / z)) for z in zooms]
sl = tuple(slice(max(int(c.min()) - p, 0), min(int(c.max()) + p + 1, s))
           for c, p, s in zip(nz, pad, seg_full.shape))
seg = seg_full[sl].copy()
ct = np.ascontiguousarray(ct_full[sl])
del ct_full
qa = {"flags": []}
out = pv.stage2g_imbrication(seg, ct, img.affine, zooms, vox, qa)
print(json.dumps(qa.get("imbrication", {}), indent=1))
print("FLAGS:", qa["flags"])
seg_full[sl] = out
nib.save(nib.Nifti1Image(seg_full, img.affine, img.header), out_p)
print("saved", out_p)
