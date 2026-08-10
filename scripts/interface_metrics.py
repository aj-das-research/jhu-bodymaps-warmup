"""Interface and shape metrics for vertebra label volumes (v3-plan gates).

Per adjacent pair: contact area, planarity RMS (SVD residual of interface
voxels vs best-fit plane; anatomy interdigitates, so raw-like >= 4 mm is the
healthy range and near-planar values flag guillotines), and S-I overlap
(z-span overlap between the two labels; ~8-15 mm at T/L facets is healthy,
~0 means amputation, >~25 mm means bleed). Per level: volume + robust
log-volume trend residual.

Usage:
  python scripts/interface_metrics.py A=path/combined_labels.nii.gz \
         [B=path2 ...] [--json out.json]
Each named input becomes a column; pairs are reported top-down.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

NAMES_BOTTOM_UP = (["L5", "L4", "L3", "L2", "L1"]
                   + [f"T{i}" for i in range(12, 0, -1)]
                   + [f"C{i}" for i in range(7, 0, -1)])
ID_TO_NAME = {i + 1: n for i, n in enumerate(NAMES_BOTTOM_UP)}
TOP_DOWN = list(reversed(NAMES_BOTTOM_UP))


def all_interfaces(seg):
    """One sweep over 3 axes -> dict (a,b) -> float voxel points, a<b."""
    out = {}
    for ax in range(3):
        s_hi = [slice(None)] * 3
        s_lo = [slice(None)] * 3
        s_hi[ax] = slice(1, None)
        s_lo[ax] = slice(None, -1)
        x, y = seg[tuple(s_hi)], seg[tuple(s_lo)]
        m = (x > 0) & (y > 0) & (x != y)
        if not m.any():
            continue
        i = np.nonzero(m)
        a = np.minimum(x[m], y[m]).astype(np.int32)
        b = np.maximum(x[m], y[m]).astype(np.int32)
        pp = np.c_[i[0], i[1], i[2]].astype(np.float64)
        pp[:, ax] += 0.5
        key = a * 100 + b
        for k in np.unique(key):
            sel = key == k
            out.setdefault((int(k) // 100, int(k) % 100), []).append(pp[sel])
    return {k: np.vstack(v) for k, v in out.items()}


def planarity_mm(pts, zooms):
    if pts is None or len(pts) < 50:
        return None
    Q = pts * np.asarray(zooms)
    Q = Q - Q.mean(0)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    return float(np.sqrt(((Q @ Vt[2]) ** 2).mean()))


def analyze(path):
    img = nib.load(str(path))
    seg = np.asarray(img.dataobj).astype(np.uint8)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    vox_mm3 = float(abs(np.linalg.det(img.affine[:3, :3])))
    nz = np.nonzero(seg)
    sl = tuple(slice(int(c.min()), int(c.max()) + 2) for c in nz)
    seg = seg[sl]
    iface = all_interfaces(seg)
    present = {}
    for lid in range(1, 25):
        m = seg == lid
        n = int(m.sum())
        if n:
            zs = np.nonzero(m.any(axis=(0, 1)))[0]
            present[lid] = {"vox": n, "cm3": n * vox_mm3 / 1000.0,
                            "z0": int(zs[0]), "z1": int(zs[-1])}
    levels = {}
    order = [n for n in TOP_DOWN if lid_of(n) in present]
    logv = {n: np.log(present[lid_of(n)]["vox"]) for n in order}
    for i, n in enumerate(order):
        w = [logv[order[j]] for j in range(max(0, i - 2), min(len(order), i + 3))
             if j != i]
        res = abs(logv[n] - float(np.median(w))) if w else 0.0
        levels[n] = {"cm3": round(present[lid_of(n)]["cm3"], 1),
                     "logres": round(float(res), 3)}
    pairs = {}
    for k in range(len(order) - 1):
        u, l = order[k], order[k + 1]
        a, b = lid_of(u), lid_of(l)
        pts = iface.get((min(a, b), max(a, b)))
        pl = planarity_mm(pts, zooms)
        area = None if pts is None else len(pts) * (vox_mm3 ** (2 / 3))
        # S-I span overlap in mm (sign-corrected: u is anatomically upper)
        zu, zl = present[a], present[b]
        up = (zu["z0"] + zu["z1"]) >= (zl["z0"] + zl["z1"])  # u at larger z?
        ov = ((min(zu["z1"], zl["z1"]) - max(zu["z0"], zl["z0"]) + 1) * zooms[2])
        pairs[f"{u}|{l}"] = {
            "contact_mm2": None if area is None else round(area, 0),
            "planarity_mm": None if pl is None else round(pl, 2),
            "si_overlap_mm": round(float(ov), 1)}
    return {"levels": levels, "pairs": pairs, "zooms": zooms}


def lid_of(name):
    return NAMES_BOTTOM_UP.index(name) + 1


def main():
    args = [a for a in sys.argv[1:] if "=" in a]
    jout = None
    if "--json" in sys.argv:
        jout = sys.argv[sys.argv.index("--json") + 1]
    cols = {}
    for a in args:
        name, path = a.split("=", 1)
        cols[name] = analyze(Path(path))
    names = list(cols)
    all_pairs = []
    for c in names:
        for p in cols[c]["pairs"]:
            if p not in all_pairs:
                all_pairs.append(p)
    print(f"\n{'pair':>8} | " + " | ".join(f"{c:>26}" for c in names))
    print(f"{'':>8} | " + " | ".join(f"{'planar  s-i_ovl  contact':>26}" for _ in names))
    for p in all_pairs:
        row = []
        for c in names:
            d = cols[c]["pairs"].get(p)
            row.append("-" * 26 if d is None else
                       f"{str(d['planarity_mm']):>6} {str(d['si_overlap_mm']):>8} "
                       f"{str(d['contact_mm2']):>9}")
        print(f"{p:>8} | " + " | ".join(row))
    print(f"\n{'level':>6} | " + " | ".join(f"{c:>16}" for c in names))
    print(f"{'':>6} | " + " | ".join(f"{'cm3    logres':>16}" for _ in names))
    lvls = []
    for c in names:
        for n in cols[c]["levels"]:
            if n not in lvls:
                lvls.append(n)
    for n in lvls:
        row = []
        for c in names:
            d = cols[c]["levels"].get(n)
            row.append(" " * 16 if d is None else
                       f"{d['cm3']:>7} {d['logres']:>7}")
        print(f"{n:>6} | " + " | ".join(row))
    if jout:
        Path(jout).parent.mkdir(parents=True, exist_ok=True)
        Path(jout).write_text(json.dumps(cols, indent=1, default=float))
        print(f"\nwrote {jout}")


if __name__ == "__main__":
    main()
