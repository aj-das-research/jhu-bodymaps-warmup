"""Process-ownership pass: downward-growth rule for posterior extensions.

Anatomy: the spinous process and inferior articular processes of vertebra k
grow DOWNWARD from k, overlapping level k+1's height. When the arch split
fails to separate a thick facet stack, those extensions end up labeled k+1.

Fix, per adjacent pair (upper u, lower l):
  1. ROI around their interface. Voxels of u/l farther than TRUST_MM from the
     interface are trusted anchors (their labels are far from the dispute).
  2. The disputed zone (union of u,l within TRUST_MM) is re-partitioned by
     multi-scale waist splitting: erode the union by increasing radii until
     each core connects to anchors of only ONE label; that label owns the
     core. A core still touching both at max erosion goes to the UPPER level
     (processes grow downward - stated anatomical prior).
  3. Shell voxels reconstruct by uniform BFS. Gated per pair: bounded swap,
     no new fragments, else revert.

Usage: python partF_processes.py <case_id> <in_suffix> <out_suffix>
"""
from __future__ import annotations

import sys

import cc3d
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

import vertlib as V

PAR = {
    "roi_mm": 22.0,
    "trust_mm": 7.0,
    "erode_steps_mm": [2.4, 3.2, 4.0],
    "min_core_mm3": 60.0,
    "max_shift_frac": 0.35,
}


def pair_interface_pts(seg, a, b):
    pts = []
    for ax in range(3):
        s_hi = [slice(None)] * 3
        s_lo = [slice(None)] * 3
        s_hi[ax] = slice(1, None)
        s_lo[ax] = slice(None, -1)
        x, y = seg[tuple(s_hi)], seg[tuple(s_lo)]
        m = ((x == a) & (y == b)) | ((x == b) & (y == a))
        if m.any():
            i = np.nonzero(m)
            pp = np.c_[i[0], i[1], i[2]]
            pp[:, ax] += 1
            pts.append(pp)
    return np.vstack(pts) if pts else None


def repartition_pair(seg, u_id, l_id, zooms, vox_mm3):
    """u_id = anatomically UPPER level. Returns (changed_voxels, accepted)."""
    pts = pair_interface_pts(seg, u_id, l_id)
    if pts is None or len(pts) < 30:
        return 0, False
    lo = np.maximum(pts.min(0), 0)
    hi = np.minimum(pts.max(0) + 1, np.asarray(seg.shape))
    pad = np.ceil((PAR["roi_mm"]) / np.asarray(zooms)).astype(int)
    sl = tuple(slice(max(int(a - p), 0), min(int(b + p), n))
               for a, b, p, n in zip(lo, hi, pad, seg.shape))
    sub = seg[sl]
    M = (sub == u_id) | (sub == l_id)
    iface = np.zeros(sub.shape, dtype=bool)
    iface[tuple((pts - [s.start for s in sl]).T)] = True
    d_if = ndimage.distance_transform_edt(~iface, sampling=zooms)
    disputed = M & (d_if <= PAR["trust_mm"])
    anchors = M & ~disputed
    if not disputed.any() or not (anchors & (sub == u_id)).any() \
            or not (anchors & (sub == l_id)).any():
        return 0, False

    # multi-scale waist splitting of the disputed zone (with anchors attached
    # so cores can reach their roots)
    edtM = ndimage.distance_transform_edt(M, sampling=zooms)
    new = np.where(anchors, sub, 0).astype(np.int32)
    remaining = disputed.copy()
    for t in PAR["erode_steps_mm"]:
        if not remaining.any():
            break
        core = M & (edtM >= t)
        cc = cc3d.connected_components((core | anchors).astype(np.uint8),
                                       connectivity=26)
        for cid in np.unique(cc[remaining & core]):
            if cid == 0:
                continue
            comp = cc == cid
            labs = set(np.unique(np.where(anchors & comp, sub, 0))) - {0}
            comp_dis = comp & remaining
            if float(comp_dis.sum()) * vox_mm3 < PAR["min_core_mm3"]:
                continue
            if labs == {u_id} or labs == {l_id}:
                new[comp_dis] = u_id if labs == {u_id} else l_id
                remaining &= ~comp_dis
        # cores connected to BOTH anchors stay for the next (deeper) erosion
    # still-ambiguous deep cores: downward-growth rule -> upper level
    deep_left = remaining & (edtM >= PAR["erode_steps_mm"][0])
    new[deep_left] = u_id
    remaining &= ~deep_left
    # shell reconstruction by uniform BFS through M
    ws = watershed(np.zeros(M.shape, dtype=np.uint8), markers=new, mask=M)
    cand = sub.copy()
    fill = M & (ws > 0)
    cand[fill] = ws[fill].astype(sub.dtype)

    changed = int((cand != sub).sum())
    n_min = min(int((sub == u_id).sum()), int((sub == l_id).sum()))
    if changed == 0 or changed > PAR["max_shift_frac"] * max(n_min, 1):
        return changed, False
    # fragment guard within ROI
    for lab in (u_id, l_id):
        _, c_new = np.unique(cc3d.connected_components(
            (cand == lab).astype(np.uint8), connectivity=26), return_counts=True)
        _, c_old = np.unique(cc3d.connected_components(
            (sub == lab).astype(np.uint8), connectivity=26), return_counts=True)
        if len(c_new) > len(c_old) + 1:
            return changed, False
    seg[sl] = cand
    return changed, True


def main():
    cid, sin, sout = sys.argv[1], sys.argv[2], sys.argv[3]
    d = V.prepare_case(cid)
    seg = np.load(f"{V.CACHE}/{cid}_{sin}.npy").copy()
    zooms, vox_mm3 = d["zooms"], float(d["vox_mm3"])
    total = 0
    for k in range(len(V.TOP_DOWN) - 1):
        u, l = V.NAME_TO_ID[V.TOP_DOWN[k]], V.NAME_TO_ID[V.TOP_DOWN[k + 1]]
        ch, ok = repartition_pair(seg, u, l, zooms, vox_mm3)
        if ch:
            print(f"  {V.TOP_DOWN[k]:>3}|{V.TOP_DOWN[k+1]:<3} "
                  f"{'applied' if ok else 'reverted'} ({round(ch*vox_mm3/1000,1)} cm3 zone)")
        total += ch if ok else 0
    from partD_smooth import absorb_orphans
    seg = absorb_orphans(seg, vox_mm3)
    rows, s = V.audit(seg, d["affine"], vox_mm3, verbose=False)
    print(f"  total reassigned-zone: {round(total*vox_mm3/1000,1)} cm3 | audit: {s}")
    np.save(f"{V.CACHE}/{cid}_{sout}.npy", seg)


if __name__ == "__main__":
    main()
