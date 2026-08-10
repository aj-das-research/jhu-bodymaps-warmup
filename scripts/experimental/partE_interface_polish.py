"""P1b: pairwise interface re-arbitration (adaptive, evidence-local).

For every adjacent label pair whose interface is guillotine-flat (planarity
RMS below threshold), re-solve ONLY a small collar around that interface:
domain = the two labels' voxels inside the collar, seeds = their voxels just
outside it, priority = |grad HU| so the boundary settles on the joint-space
ridges. Random-walker fallback where the gradient ridge is too weak. Each
pair is gated: revert unless planarity improves and the volume shift stays
bounded. Runs on all pairs, both cases - the trigger is the data.

Usage: python partE_polish.py <case_id> <in_suffix> <out_suffix>
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import ndimage

import vertlib as V

PAR = {
    "planarity_trigger_mm": 3.5,
    "min_contact_mm2": 80.0,
    "collar_mm": 6.0,
    "seed_shell_mm": 2.5,
    "max_shift_frac": 0.15,   # max voxels a pair swap may move, vs smaller label
    "rw_beta": 130.0,
    "rw_max_voxels": 2.5e6,
}


def interfaces(seg):
    """dict (a,b)->list of voxel index arrays (concatenated later)."""
    out = {}
    for ax in range(3):
        s_hi = [slice(None)] * 3
        s_lo = [slice(None)] * 3
        s_hi[ax] = slice(1, None)
        s_lo[ax] = slice(None, -1)
        x, y = seg[tuple(s_hi)], seg[tuple(s_lo)]
        m = (x > 0) & (y > 0) & (x != y)
        idx = np.nonzero(m)
        a = np.minimum(x[m], y[m]).astype(np.int32)
        b = np.maximum(x[m], y[m]).astype(np.int32)
        base = np.zeros(3, dtype=np.int64)
        pts = np.c_[idx[0], idx[1], idx[2]] + base
        pts[:, ax] += 1  # voxel on the high side; collar dilation covers both
        key = a * 100 + b
        for k in np.unique(key):
            sel = key == k
            out.setdefault((int(k) // 100, int(k) % 100), []).append(pts[sel])
    return {k: np.vstack(v) for k, v in out.items()}


def planarity_mm(pts, zooms):
    P = pts * np.asarray(zooms)
    if len(P) < 50:
        return None
    Q = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    return float(np.sqrt(((Q @ Vt[2]) ** 2).mean()))


def polish(seg, ct, zooms, vox_mm3, log=print):
    out = seg.copy()
    grad_cache = {}
    pairs = interfaces(out)
    report = []
    adj = {(V.NAME_TO_ID[V.TOP_DOWN[k]], V.NAME_TO_ID[V.TOP_DOWN[k + 1]])
           for k in range(len(V.TOP_DOWN) - 1)}
    adj = {(min(a, b), max(a, b)) for a, b in adj}
    for (a, b), pts in sorted(pairs.items()):
        if (a, b) not in adj:
            continue
        name = f"{V.ID_TO_NAME[b]}|{V.ID_TO_NAME[a]}"
        area = len(pts) * (vox_mm3 ** (2 / 3))
        p0 = planarity_mm(pts, zooms)
        rec = {"pair": name, "contact_mm2": round(area, 0),
               "planarity_before": None if p0 is None else round(p0, 2)}
        report.append(rec)
        if p0 is None or area < PAR["min_contact_mm2"] or p0 >= PAR["planarity_trigger_mm"]:
            rec["action"] = "skip"
            continue
        # collar ROI around this interface
        lo = np.maximum(pts.min(0) - 2, 0)
        hi = np.minimum(pts.max(0) + 3, np.asarray(seg.shape))
        pad = np.ceil((PAR["collar_mm"] + PAR["seed_shell_mm"] + 2) / np.asarray(zooms)).astype(int)
        sl = tuple(slice(max(int(l - p), 0), min(int(h + p), n))
                   for l, h, p, n in zip(lo, hi, pad, seg.shape))
        sub = out[sl]
        iface = np.zeros(sub.shape, dtype=bool)
        iface[tuple((pts - [s.start for s in sl]).T)] = True
        d_if = ndimage.distance_transform_edt(~iface, sampling=zooms)
        collar = d_if <= PAR["collar_mm"]
        shell = (d_if > PAR["collar_mm"]) & (d_if <= PAR["collar_mm"] + PAR["seed_shell_mm"])
        ab = (sub == a) | (sub == b)
        R = collar & ab
        seeds = np.where(shell & ab, sub, 0).astype(np.int32)
        if not (seeds == a).any() or not (seeds == b).any() or not R.any():
            rec["action"] = "skip_no_seeds"
            continue
        key = tuple((s.start, s.stop) for s in sl)
        if key not in grad_cache:
            grad_cache.clear()  # keep memory flat; ROIs rarely repeat
            grad_cache[key] = ndimage.gaussian_gradient_magnitude(
                ct[sl].astype(np.float32), sigma=[1.0 / z for z in zooms])
        grad = grad_cache[key]
        from skimage.segmentation import watershed
        ws = watershed(grad, markers=seeds, mask=R | (seeds > 0))
        cand = sub.copy()
        m_new = R & (ws > 0)
        cand[m_new] = ws[m_new].astype(cand.dtype)
        rec.update(_gate(out, sl, sub, cand, a, b, zooms, "ws", p0))
        if rec.get("accepted"):
            out[sl] = cand
            continue
        # random-walker fallback on the same ROI
        if R.sum() <= PAR["rw_max_voxels"]:
            try:
                from skimage.segmentation import random_walker
                data = np.clip(ct[sl].astype(np.float32), -200, 1500) / 1500.0
                lab = np.where(shell & ab, sub, 0).astype(np.int32)
                lab[~(R | (lab > 0))] = -1  # outside: not solved
                rw = random_walker(data, lab, beta=PAR["rw_beta"], mode="cg_j")
                cand = sub.copy()
                m_new = R & (rw > 0)
                cand[m_new] = rw[m_new].astype(cand.dtype)
                rec.update(_gate(out, sl, sub, cand, a, b, zooms, "rw", p0))
                if rec.get("accepted"):
                    out[sl] = cand
            except Exception as e:  # solver failure: keep original
                rec["rw_error"] = str(e)[:80]
    return out, report


def _gate(out, sl, sub, cand, a, b, zooms, tag, p0):
    """Accept iff planarity improves and volume shift is bounded."""
    m = (cand == a) | (cand == b)
    swapped = int(((cand != sub) & m).sum())
    n_min = min(int((sub == a).sum()), int((sub == b).sum()))
    pts = []
    for ax in range(3):
        s_hi = [slice(None)] * 3
        s_lo = [slice(None)] * 3
        s_hi[ax] = slice(1, None)
        s_lo[ax] = slice(None, -1)
        x, y = cand[tuple(s_hi)], cand[tuple(s_lo)]
        mm = ((x == a) & (y == b)) | ((x == b) & (y == a))
        idx = np.nonzero(mm)
        pp = np.c_[idx[0], idx[1], idx[2]]
        pp[:, ax] += 1
        pts.append(pp)
    pts = np.vstack(pts) if pts else np.zeros((0, 3))
    p1 = planarity_mm(pts, zooms)
    res = {f"planarity_after_{tag}": None if p1 is None else round(p1, 2),
           f"swapped_vox_{tag}": swapped}
    ok = (p1 is not None and swapped > 0
          and swapped <= PAR["max_shift_frac"] * max(n_min, 1)
          and p1 > p0 + 0.3)
    res["accepted"] = bool(ok)
    res["action"] = tag if res["accepted"] else f"revert_{tag}"
    return res


def main():
    cid, sin, sout = sys.argv[1], sys.argv[2], sys.argv[3]
    d = V.prepare_case(cid)
    seg = np.load(f"{V.CACHE}/{cid}_{sin}.npy")
    out, report = polish(seg, d["ct"], d["zooms"], float(d["vox_mm3"]))
    for r in report:
        if r.get("action") != "skip":
            print(f"  {r['pair']:>8} before={r['planarity_before']} "
                  f"ws={r.get('planarity_after_ws')} rw={r.get('planarity_after_rw')} "
                  f"{r['action']}")
    changed = int((out != seg).sum())
    print(f"  polished voxels: {changed}")
    rows, s = V.audit(out, d["affine"], float(d["vox_mm3"]), verbose=False)
    print(f"  audit: {s}")
    np.save(f"{V.CACHE}/{cid}_{sout}.npy", out)
    V.save_json({"params": PAR, "report": report, "summary": s},
                f"{V.OUT}/{cid}_{sout}_qa.json")


if __name__ == "__main__":
    main()
