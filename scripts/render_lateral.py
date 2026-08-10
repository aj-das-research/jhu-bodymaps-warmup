"""ITK-SNAP-style lateral surface renders of vertebra label volumes.

Ray-cast along +x and -x (patient left/right): for each (y,z) the first
nonzero label defines the visible surface; simple Lambert shading from the
depth gradient. Fast, dependency-light, and directly comparable to the 3D
view screenshots used for error inspection (wrong-colored facet knobs are
obvious). Renders one row per input volume, two columns (left/right views).

Usage:
  python scripts/render_lateral.py OUT.png A=path1 [B=path2 ...] [--zcrop z0 z1]
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import ndimage

# ITK-SNAP default label colors (ids 1..24), matching the user's viewer
SNAP = ["#000000", "#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff",
        "#ff00ff", "#ffefd5", "#0000cd", "#cd853f", "#b8860b", "#529584",
        "#1e90ff", "#00bfff", "#a020f0", "#948d92", "#2e8b57", "#008b8b",
        "#32cd32", "#7cfc00", "#dda0dd", "#8b7500", "#c0c0c0", "#9370db",
        "#8b4513"]


def surface_view(seg, zooms, side):
    """Return (label2d, depth2d) for a lateral ray-cast along x."""
    nx = seg.shape[0]
    xs = np.arange(nx) if side == "left" else np.arange(nx)[::-1]
    hit = np.zeros(seg.shape[1:], dtype=np.uint8)
    depth = np.full(seg.shape[1:], np.nan, dtype=np.float32)
    todo = np.ones(seg.shape[1:], dtype=bool)
    for x in xs:
        sl = seg[x]
        m = todo & (sl > 0)
        if m.any():
            hit[m] = sl[m]
            depth[m] = x * zooms[0] if side == "left" else (nx - 1 - x) * zooms[0]
            todo &= ~m
        if not todo.any():
            break
    return hit, depth


def shaded_rgb(hit, depth, zooms):
    cmap = np.array([tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))
                     for c in SNAP])
    rgb = cmap[np.clip(hit, 0, len(cmap) - 1)]
    fill = np.nanmax(depth) if np.isfinite(depth).any() else 0.0
    d = np.where(np.isnan(depth), fill, depth)
    d = ndimage.gaussian_filter(d, 1.0)
    gy, gz = np.gradient(d, zooms[1], zooms[2])
    nrm = 1.0 / np.sqrt(1.0 + gy ** 2 + gz ** 2)
    light = np.clip(0.35 + 0.65 * nrm, 0, 1)
    rgb = rgb * light[..., None]
    rgb[hit == 0] = 0.0
    return rgb


def main():
    out = sys.argv[1]
    args = [a for a in sys.argv[2:] if "=" in a]
    zcrop = None
    if "--zcrop" in sys.argv:
        i = sys.argv.index("--zcrop")
        zcrop = (int(sys.argv[i + 1]), int(sys.argv[i + 2]))
    rows = []
    for a in args:
        name, path = a.split("=", 1)
        img = nib.load(path)
        seg = np.asarray(img.dataobj).astype(np.uint8)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        if zcrop:
            seg = seg[:, :, zcrop[0]:zcrop[1]]
        rows.append((name, seg, zooms))
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 5.2 * len(rows)),
                             squeeze=False)
    for r, (name, seg, zooms) in enumerate(rows):
        for c, side in enumerate(("left", "right")):
            hit, depth = surface_view(seg, zooms, side)
            rgb = shaded_rgb(hit, depth, zooms)
            ext = [0, seg.shape[1] * zooms[1], 0, seg.shape[2] * zooms[2]]
            axes[r, c].imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
                              extent=ext, aspect="equal",
                              interpolation="nearest")
            axes[r, c].set_title(f"{name} ({side} lateral)", fontsize=11)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor="black")
    print("render:", out)


if __name__ == "__main__":
    main()
