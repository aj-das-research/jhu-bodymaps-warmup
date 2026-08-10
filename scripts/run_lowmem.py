"""Run the postprocessing pipeline in memory-bounded phases.

The single-file deliverable runs everything in one process (~9 GB peak on a
0.7 mm whole-spine case); this driver executes the SAME stages in three
fresh processes with an on-disk checkpoint between them, for validation on
small machines. Results are identical: stages are pure functions of
(seg, raw, ct) and the checkpoint carries seg + accumulated QA verbatim.

Usage: python scripts/run_lowmem.py CASE_ID PHASE   (PHASE in 1|2|3)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import postprocessing_vertebrae as pv

CASE = sys.argv[1]
PHASE = int(sys.argv[2])
PRED = Path("AbdomenAtlasDemoPredict") / CASE
CT_ROOT = Path("data/AbdomenAtlasDemo")
OUT = Path("out_v9")
REPORTS = Path("reports/v9")
STATE = Path("state_v9"); STATE.mkdir(exist_ok=True)


def load_crop():
    loaded = pv.load_case(PRED, CT_ROOT)
    seg_img, seg_full, ct_full, _ = loaded
    zooms = tuple(float(z) for z in seg_img.header.get_zooms()[:3])
    vox = float(abs(np.linalg.det(seg_img.affine[:3, :3])))
    pad = np.maximum(np.round(pv.P["crop_margin_mm"] / np.asarray(zooms)).astype(int), 2)
    nz = np.nonzero(seg_full)
    lo = np.maximum(np.array([c.min() for c in nz]) - pad, 0)
    hi = np.minimum(np.array([c.max() for c in nz]) + pad + 1, seg_full.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    seg = seg_full[sl].copy()
    ct = np.ascontiguousarray(ct_full[sl])
    aff = seg_img.affine.copy()
    aff[:3, 3] = (seg_img.affine @ np.array([lo[0], lo[1], lo[2], 1.0]))[:3]
    shape = seg_full.shape
    del seg_full, ct_full
    return seg_img, seg, ct, aff, zooms, vox, sl, shape


def main():
    seg_img, seg_raw, ct, aff, zooms, vox, sl, full_shape = load_crop()
    qa_p = STATE / f"{CASE}_qa.json"
    seg_p = STATE / f"{CASE}_seg.npy"
    if PHASE == 1:
        qa = {"case": CASE, "params": pv.P, "records": [], "bands": [],
              "flags": [], "polish": [], "smooth": []}
        _, s0 = pv.audit(seg_raw, aff, vox)
        qa["audit_before_summary"] = s0
        raw = seg_raw.copy()
        seg = pv.stage1_triage(seg_raw, ct, zooms, vox, qa["records"])
        seg = pv.stage2a_islands(seg, vox, qa["records"])
        seg = pv.stage2b_arbitrate(seg, raw, ct, aff, zooms, vox, qa)
        seg = pv.stage2c_interface_polish(seg, ct, zooms, vox, qa)
        seg = pv.stage3_smooth(seg, ct, zooms, vox, qa)
        seg = pv.stage2d_reclaim_pool(seg, raw, ct, zooms, vox, qa)
        bone_f = ct >= pv.P["bone_hu"]
        qa["envelope"] = {
            "dropped_bone_cm3": round(float(((raw > 0) & (seg == 0) & bone_f).sum()) * vox / 1e3, 2),
            "dropped_subbone_cm3": round(float(((raw > 0) & (seg == 0) & ~bone_f).sum()) * vox / 1e3, 2),
            "recolored_cm3": round(float(((raw > 0) & (seg > 0) & (raw != seg)).sum()) * vox / 1e3, 2),
            "added_beyond_raw_cm3": round(float(((raw == 0) & (seg > 0)).sum()) * vox / 1e3, 2)}
        np.save(seg_p, seg)
        qa_p.write_text(json.dumps(qa, default=float))
    elif PHASE == 2:
        qa = json.loads(qa_p.read_text())
        seg = np.load(seg_p)
        seg = pv.stage2e_multiview_recolor(seg, ct, aff, zooms, vox, qa)
        np.save(seg_p, seg)
        qa_p.write_text(json.dumps(qa, default=float))
    else:
        qa = json.loads(qa_p.read_text())
        seg = np.load(seg_p)
        seg = pv.stage2f_skeleton_relabel(seg, ct, aff, zooms, vox, qa)
        seg = pv.stage2g_imbrication(seg, ct, aff, zooms, vox, qa)
        upv_t, upv_p = pv._imbrication_cm3(seg, ct, zooms, vox)
        qa["imbrication_upv_cm3"] = {"total": upv_t, "per_level": upv_p}
        rows, s1 = pv.audit(seg, aff, vox)
        qa["audit_after"] = s1
        qa["audit_rows_after"] = rows
        out_full = np.zeros(full_shape, dtype=np.uint8)
        out_full[sl] = seg
        pv.write_case(OUT, CASE, out_full, seg_img)
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / f"{CASE}_postprocessing_qa.json").write_text(
            json.dumps(qa, indent=1, default=float))
        print("audit_after:", s1)
        print("skeleton:", json.dumps(qa.get("skeleton", {})))
        print("imbrication:", json.dumps(qa.get("imbrication", {})))
    print(f"phase {PHASE} done: {CASE}")


if __name__ == "__main__":
    main()
