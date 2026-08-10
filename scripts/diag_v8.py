"""v8 defect quantification: split nodules and mixed blades.

Invariant being tested: A LABEL BOUNDARY MAY ONLY PASS THROUGH THIN BONE
(joint clefts, necks) - NEVER THROUGH THE INTERIOR OF A THICK MASS. A facet
knob is one rigid piece; a boundary crossing bone thicker than ~2.5 mm is
cutting through a nodule or blade, which is anatomically impossible.

Metrics:
  A. per adjacent pair: interface area where bone-EDT >= 2.5 mm
     ("mass-cut area") + max thickness crossed.
  B. 3D supra-neck cores (bone EDT >= 2.0 mm, comps >= 200 mm3) carrying
     two labels; compact cores (PCA elong < 2.2) = split NODULES, elongated
     = fused blade STACKS (need slice-tracking split, not unification).
  C. parasagittal blade sweep (|x-cx| <= 8): 2D posterior-corridor cores
     with mixed labels per slice.

Renders: posterior zoom at requested levels with mass-cut interface voxels
in red - shows exactly where boundaries slice through solid bone.

Usage: python scripts/diag_v8.py CT SEG OUTDIR [--levels T6,T7,T8,T9,T10]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cc3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_lateral import SNAP, surface_view

NAMES_BOTTOM_UP = (["L5", "L4", "L3", "L2", "L1"]
                   + [f"T{i}" for i in range(12, 0, -1)]
                   + [f"C{i}" for i in range(7, 0, -1)])
NAME_TO_ID = {n: i + 1 for i, n in enumerate(NAMES_BOTTOM_UP)}
ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}
TOP_DOWN = list(reversed(NAMES_BOTTOM_UP))
CMAP = np.array([tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)) for c in SNAP])


def interfaces_mask(seg):
    m = np.zeros(seg.shape, dtype=bool)
    pair_at = {}
    for ax in range(3):
        s_hi = [slice(None)] * 3
        s_lo = [slice(None)] * 3
        s_hi[ax] = slice(1, None)
        s_lo[ax] = slice(None, -1)
        x, y = seg[tuple(s_hi)], seg[tuple(s_lo)]
        b = (x > 0) & (y > 0) & (x != y)
        m[tuple(s_hi)] |= b
        m[tuple(s_lo)] |= b
    return m


def main():
    ct_p, seg_p, outdir = sys.argv[1:4]
    levels = "T6,T7,T8,T9,T10"
    if "--levels" in sys.argv:
        levels = sys.argv[sys.argv.index("--levels") + 1]
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    img = nib.load(seg_p)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    vox = float(np.prod(zooms))
    seg = np.asarray(img.dataobj).astype(np.uint8)
    ct = np.asarray(nib.load(ct_p).dataobj)
    nz = np.nonzero(seg)
    sl = tuple(slice(int(c.min()) - 6, int(c.max()) + 6) for c in nz)
    seg, ct = seg[sl], ct[sl]
    bone = ct >= 150.0
    edtb = ndimage.distance_transform_edt(bone, sampling=zooms).astype(np.float32)

    # ---- A: mass-cut interface area per adjacent pair --------------------
    print("=== A. interface mass-cut (boundary crossing bone >= 2.5 mm) ===")
    iface = interfaces_mask(seg)
    adj = [(NAME_TO_ID[TOP_DOWN[k]], NAME_TO_ID[TOP_DOWN[k + 1]])
           for k in range(len(TOP_DOWN) - 1)]
    area_fac = vox ** (2 / 3)
    cut_all = np.zeros(seg.shape, dtype=bool)
    st = ndimage.generate_binary_structure(3, 1)
    for a, b in adj:
        pa = (seg == a) & ndimage.binary_dilation((seg == b), structure=st)
        pb = (seg == b) & ndimage.binary_dilation((seg == a), structure=st)
        pm = pa | pb
        if not pm.any():
            continue
        vals = edtb[pm]
        cut = pm & (edtb >= 2.5)
        cut_all |= cut
        if vals.size * area_fac < 30:
            continue
        print(f"  {ID_TO_NAME[a]:>3}|{ID_TO_NAME[b]:<3} contact={vals.size*area_fac:7.0f}mm2 "
              f"masscut={int(cut.sum())*area_fac:6.0f}mm2 "
              f"({100*cut.sum()/max(vals.size,1):4.1f}%) maxEDT={vals.max():.1f}mm")

    # ---- B: mixed supra-neck cores --------------------------------------
    print("\n=== B. supra-neck 3D cores (EDT>=2.0) with two labels ===")
    cores = edtb >= 2.0
    cc = cc3d.connected_components(cores.astype(np.uint8), connectivity=26)
    counts = np.bincount(cc.ravel()); counts[0] = 0
    n_nod = n_stack = 0
    for cid in np.nonzero(counts)[0]:
        if counts[cid] * vox < 200:
            continue
        m = cc == cid
        labs = seg[m]
        ids, cnt = np.unique(labs[labs > 0], return_counts=True)
        if ids.size < 2:
            continue
        share = cnt / cnt.sum()
        if share.min() < 0.10 or cnt.sum() < 0.5 * counts[cid]:
            continue
        pts = np.argwhere(m) * np.asarray(zooms)
        pts = pts - pts.mean(0)
        w = np.linalg.eigvalsh(pts.T @ pts / len(pts))
        elong = float(np.sqrt(max(w[2], 1e-9) / max(w[1], 1e-9)))
        kind = "STACK(elong)" if elong >= 2.2 else "NODULE(compact)"
        if kind.startswith("STACK"):
            n_stack += 1
        else:
            n_nod += 1
        names = "+".join(f"{ID_TO_NAME[int(i)]}:{s:.0%}" for i, s in zip(ids, share))
        print(f"  {kind:>15} vol={counts[cid]*vox/1e3:5.2f}cm3 elong={elong:.1f} {names}")
    print(f"  totals: {n_nod} split nodules, {n_stack} mixed stacks")

    # ---- C: parasagittal blade sweep ------------------------------------
    cxs = np.nonzero(seg.any(axis=(1, 2)))[0]
    cx = int(np.median(cxs))
    mixed_slices = 0
    for x in range(cx - 10, cx + 11):
        b2 = bone[x]
        if not b2.any():
            continue
        edt2 = ndimage.distance_transform_edt(b2, sampling=(zooms[1], zooms[2]))
        lab2, n2 = ndimage.label(edt2 >= 1.2)
        s2 = seg[x]
        for c in range(1, n2 + 1):
            m = lab2 == c
            if m.sum() * zooms[1] * zooms[2] < 60:
                continue
            labs = s2[m]
            ids, cnt = np.unique(labs[labs > 0], return_counts=True)
            if ids.size >= 2 and (cnt / cnt.sum()).min() >= 0.15:
                mixed_slices += 1
                break
    print(f"\n=== C. parasagittal sweep: {mixed_slices}/21 slices contain a "
          f"mixed-label 2D core ===")

    # ---- render: posterior zoom with mass-cut in red --------------------
    lids = [NAME_TO_ID[n] for n in levels.split(",")]
    zsel = np.nonzero(np.isin(seg, lids).any(axis=(0, 1)))[0]
    zsl = slice(max(int(zsel[0]) - 4, 0), min(int(zsel[-1]) + 5, seg.shape[2]))
    segz, cutz = seg[:, :, zsl], cut_all[:, :, zsl]
    cutz = ndimage.binary_dilation(cutz, structure=np.ones((2, 2, 2), bool))
    show = segz.copy()
    show[cutz & (segz > 0)] = 30
    cmap_x = np.vstack([CMAP, np.tile([[0.1, 0.1, 0.1]], (10, 1))])
    cmap_x[30] = [1.0, 0.1, 0.1]
    # posterior: cast along y from arch side
    dy_sign = 1  # assume +y posterior (validated earlier on this case)
    v = np.transpose(show[:, ::-1, :] if dy_sign > 0 else show, (1, 0, 2))
    zm = (zooms[1], zooms[0], zooms[2])
    hit, depth = surface_view(v, zm, "left")
    fillv = np.nanmax(depth) if np.isfinite(depth).any() else 0
    d = ndimage.gaussian_filter(np.where(np.isnan(depth), fillv, depth), 1.0)
    gy, gz = np.gradient(d, zm[1], zm[2])
    light = np.clip(0.35 + 0.65 / np.sqrt(1 + gy ** 2 + gz ** 2), 0, 1)
    rgb = cmap_x[np.clip(hit, 0, len(cmap_x) - 1)] * light[..., None]
    rgb[hit == 0] = 0
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
              extent=[0, v.shape[1] * zm[1], 0, v.shape[2] * zm[2]],
              aspect="equal", interpolation="nearest")
    ax.set_title(f"posterior {levels}: label boundaries crossing bone >= 2.5 mm in RED")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    p = outdir / "masscut_posterior.png"
    fig.savefig(p, dpi=130, facecolor="black")
    print("render:", p)


if __name__ == "__main__":
    main()
