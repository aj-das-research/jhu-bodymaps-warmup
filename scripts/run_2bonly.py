"""Stages 1-2a-2b only, dumping band gate telemetry. Usage: ... CASE 1"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import postprocessing_vertebrae as pv
from run_lowmem import load_crop

def main():
    case = sys.argv[1]
    seg_img, seg_raw, ct, aff, zooms, vox, sl, full_shape = load_crop()
    qa = {"case": case, "params": pv.P, "records": [], "bands": [],
          "flags": [], "polish": [], "smooth": []}
    raw = seg_raw.copy()
    seg = pv.stage1_triage(seg_raw, ct, zooms, vox, qa["records"])
    seg = pv.stage2a_islands(seg, vox, qa["records"])
    seg = pv.stage2b_arbitrate(seg, raw, ct, aff, zooms, vox, qa)
    np.savez_compressed(Path("state_v8/dbg") / f"{case}_2bonly.npz", seg=seg,
                        lo=np.array([s.start for s in sl]),
                        hi=np.array([s.stop for s in sl]))
    for b in qa["bands"]:
        slim = {k: b.get(k) for k in ("levels", "badness_before_after",
                "imbrication_upv_cm3_before_after", "arch_mode",
                "arch_phase2_changed_mm3", "arch_rootless", "seed_cm3",
                "cleared_out_of_band_mm3", "disc_cuts_z") if k in b}
        print("BAND:", json.dumps(slim), flush=True)
    print("FLAGS:", qa["flags"], flush=True)

if __name__ == "__main__":
    main()
