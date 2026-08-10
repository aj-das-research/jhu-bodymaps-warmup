"""Phase-1 rerun that checkpoints the segmentation AFTER STAGE 2b (and 2d)
for stage attribution of the spinous one-down defect. Compressed saves.

Usage: python scripts/run_stagedump.py CASE_ID
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import postprocessing_vertebrae as pv
from run_lowmem import load_crop, CASE  # noqa: F401  (CASE from argv)

STATE = Path("state_v8/dbg"); STATE.mkdir(parents=True, exist_ok=True)


def main():
    case = sys.argv[1]
    seg_img, seg_raw, ct, aff, zooms, vox, sl, full_shape = load_crop()
    qa = {"case": case, "params": pv.P, "records": [], "bands": [],
          "flags": [], "polish": [], "smooth": []}
    import json
    _, s0 = pv.audit(seg_raw, aff, vox)
    qa["audit_before_summary"] = s0
    raw = seg_raw.copy()
    seg = pv.stage1_triage(seg_raw, ct, zooms, vox, qa["records"])
    seg = pv.stage2a_islands(seg, vox, qa["records"])
    seg = pv.stage2b_arbitrate(seg, raw, ct, aff, zooms, vox, qa)
    np.savez_compressed(STATE / f"{case}_post2b.npz", seg=seg,
                        lo=np.array([s.start for s in sl]),
                        hi=np.array([s.stop for s in sl]))
    print("saved post2b", flush=True)
    seg = pv.stage2c_interface_polish(seg, ct, zooms, vox, qa)
    seg = pv.stage3_smooth(seg, ct, zooms, vox, qa)
    seg = pv.stage2d_reclaim_pool(seg, raw, ct, zooms, vox, qa)
    np.savez_compressed(STATE / f"{case}_post2d.npz", seg=seg,
                        lo=np.array([s.start for s in sl]),
                        hi=np.array([s.stop for s in sl]))
    print("saved post2d", flush=True)
    # phase-1-compatible checkpoint so run_lowmem phases 2/3 continue from it
    bone_f = ct >= pv.P["bone_hu"]
    qa["envelope"] = {
        "dropped_bone_cm3": round(float(((raw > 0) & (seg == 0) & bone_f).sum()) * vox / 1e3, 2),
        "dropped_subbone_cm3": round(float(((raw > 0) & (seg == 0) & ~bone_f).sum()) * vox / 1e3, 2),
        "recolored_cm3": round(float(((raw > 0) & (seg > 0) & (raw != seg)).sum()) * vox / 1e3, 2),
        "added_beyond_raw_cm3": round(float(((raw == 0) & (seg > 0)).sum()) * vox / 1e3, 2)}
    np.save(Path("state_v8") / f"{case}_seg.npy", seg)
    (Path("state_v8") / f"{case}_qa.json").write_text(json.dumps(qa, default=float))
    print("phase1 checkpoint written", flush=True)


if __name__ == "__main__":
    main()
