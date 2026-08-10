"""Per-vertebra isolation sheets + change map for arch-phase comparison.

Isolation: for each requested level, a lateral surface render of that label
solid with both z-neighbors ghosted (dimmed), per input volume. Shape
completeness (pedicles, laminae, both articular process pairs, spinous) and
wrong-knob errors are directly visible.

Change map: lateral render of volume B with voxels that differ from A shown
saturated while agreeing voxels are dimmed - localizes exactly what a stage
changed.

Usage:
  python scripts/render_isolation.py OUT.png LVL1,LVL2,... A=path [B=path ...]
  python scripts/render_isolation.py --diff OUT.png A=path B=path
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

from render_lateral import SNAP, surface_view

NAMES_BOTTOM_UP = (["L5", "L4", "L3", "L2", "L1"]
                   + [f"T{i}" for i in range(12, 0, -1)]
                   + [f"C{i}" for i in range(7, 0, -1)])
NAME_TO_ID = {n: i + 1 for i, n in enumerate(NAMES_BOTTOM_UP)}
CMAP = np.array([tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))
                 for c in SNAP])


def load(path):
    img = nib.load(path)
    seg = np.asarray(img.dataobj).astype(np.uint8)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return seg, zooms


def shade(hit, depth, zooms, dim_mask=None):
    rgb = CMAP[np.clip(hit, 0, len(CMAP) - 1)]
    fill = np.nanmax(depth) if np.isfinite(depth).any() else 0.0
    d = np.where(np.isnan(depth), fill, depth)
    d = ndimage.gaussian_filter(d, 1.0)
    gy, gz = np.gradient(d, zooms[1], zooms[2])
    light = np.clip(0.35 + 0.65 / np.sqrt(1 + gy ** 2 + gz ** 2), 0, 1)
    rgb = rgb * light[..., None]
    if dim_mask is not None:
        rgb[dim_mask] *= 0.28
    rgb[hit == 0] = 0.0
    return rgb


def panel(ax, seg, zooms, focus_ids=None, diff2d=None, side="right"):
    hit, depth = surface_view(seg, zooms, side)
    dim = None
    if focus_ids is not None:
        dim = ~np.isin(hit, list(focus_ids)) & (hit > 0)
    if diff2d is not None:
        dim = ~diff2d & (hit > 0)
    rgb = shade(hit, depth, zooms, dim)
    ext = [0, seg.shape[1] * zooms[1], 0, seg.shape[2] * zooms[2]]
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=ext,
              aspect="equal", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


def crop_for(seg, lid, zooms, pad_mm=14.0):
    m = seg == lid
    if not m.any():
        return None
    nz = np.nonzero(m)
    pad = np.ceil(pad_mm / np.asarray(zooms)).astype(int)
    return tuple(slice(max(int(c.min() - p), 0), min(int(c.max() + p + 1), n))
                 for c, p, n in zip(nz, pad, seg.shape))


def main():
    argv = sys.argv[1:]
    if argv[0] == "--diff":
        out = argv[1]
        (na, pa), (nb, pb) = (a.split("=", 1) for a in argv[2:4])
        A, zooms = load(pa)
        B, _ = load(pb)
        nz = np.nonzero((A != B))
        if nz[0].size == 0:
            print("no differences"); return
        pad = np.ceil(20.0 / np.asarray(zooms)).astype(int)
        sl = tuple(slice(max(int(c.min() - p), 0), min(int(c.max() + p + 1), n))
                   for c, p, n in zip(nz, pad, A.shape))
        A, B = A[sl], B[sl]
        fig, axes = plt.subplots(2, 2, figsize=(11, 16))
        for r, (nm, vol) in enumerate(((na, A), (nb, B))):
            for c, side in enumerate(("left", "right")):
                hit, depth = surface_view(vol, zooms, side)
                other = B if nm == na else A
                hit_o, _ = surface_view(other, zooms, side)
                panel(axes[r, c], vol, zooms, diff2d=(hit != hit_o), side=side)
                axes[r, c].set_title(f"{nm} ({side}) - changed voxels saturated",
                                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out, dpi=110, facecolor="black")
        print("render:", out)
        return

    out, levels = argv[0], argv[1].split(",")
    vols = []
    for a in argv[2:]:
        nm, p = a.split("=", 1)
        seg, zooms = load(p)
        vols.append((nm, seg, zooms))
    fig, axes = plt.subplots(len(levels), len(vols),
                             figsize=(4.6 * len(vols), 5.4 * len(levels)),
                             squeeze=False)
    ref = vols[-1][1]
    for r, lv in enumerate(levels):
        lid = NAME_TO_ID[lv]
        sl = crop_for(ref, lid, vols[-1][2])
        for c, (nm, seg, zooms) in enumerate(vols):
            s = crop_for(seg, lid, zooms) if sl is None else sl
            if s is None:
                axes[r, c].axis("off"); continue
            focus = {lid}
            panel(axes[r, c], seg[s], zooms, focus_ids=focus, side="right")
            if c == 0:
                axes[r, c].set_ylabel(lv, fontsize=13, color="w")
            if r == 0:
                axes[r, c].set_title(nm, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor="black")
    print("render:", out)


if __name__ == "__main__":
    main()
