"""Multi-angle 3D gallery: raw vs processed, column and standalone sheets.

True 3D-look renders: the label volume is rotated about the S-I axis
(nearest-neighbor, label-safe) and surface ray-cast with Lambert shading at
each requested camera angle. Produces:
  {case}_column_5views.png     raw vs v8 at 5 angles (posterior, both
                               posterior obliques, lateral, anterior)
  {case}_standalone_sheet.png  full-page: one row per vertebra, raw vs v8
                               at posterior-oblique and lateral views,
                               neighbors ghosted for context

Usage: python scripts/render_3d_views.py CASE RAW.nii.gz V8.nii.gz OUTDIR
"""
from __future__ import annotations

import sys
from pathlib import Path

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
ID_TO_NAME = {i + 1: n for i, n in enumerate(NAMES_BOTTOM_UP)}
TOP_DOWN = list(reversed(NAMES_BOTTOM_UP))
CMAP = np.array([tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)) for c in SNAP])
ANGLES = [("posterior", 0), ("post-oblique R", 40), ("lateral R", 90),
          ("post-oblique L", -40), ("anterior", 180)]


def shade_hit(hit, depth, zm, dim_mask=None):
    fv = np.nanmax(depth) if np.isfinite(depth).any() else 0.0
    d = ndimage.gaussian_filter(np.where(np.isnan(depth), fv, depth), 1.0)
    gy, gz = np.gradient(d, zm[1], zm[2])
    light = np.clip(0.35 + 0.65 / np.sqrt(1 + gy ** 2 + gz ** 2), 0, 1)
    rgb = CMAP[np.clip(hit, 0, len(CMAP) - 1)] * light[..., None]
    if dim_mask is not None:
        rgb[dim_mask & (hit > 0)] *= 0.22
    rgb[hit == 0] = 0
    return rgb


def view_at(vol, zooms, angle_deg, focus_ids=None):
    """Rotate about S-I axis then ray-cast from the posterior direction."""
    if angle_deg != 0:
        vol = ndimage.rotate(vol, angle_deg, axes=(0, 1), order=0,
                             reshape=True, prefilter=False)
    v = np.transpose(vol[:, ::-1, :], (1, 0, 2))
    zm = (zooms[1], zooms[0], zooms[2])
    hit, depth = surface_view(v, zm, "left")
    dim = None
    if focus_ids is not None:
        dim = ~np.isin(hit, list(focus_ids))
    return shade_hit(hit, depth, zm, dim), v.shape, zm


def imshow_r(ax, rgb, shape, zm, title=None):
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
              aspect=zm[2] / zm[1], interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=11, color="w")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("black")


def main():
    case, raw_p, v8_p, outdir = sys.argv[1:5]
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    img = nib.load(v8_p)
    zooms0 = tuple(float(z) for z in img.header.get_zooms()[:3])
    v8 = np.asarray(img.dataobj).astype(np.uint8)
    raw = np.asarray(nib.load(raw_p).dataobj).astype(np.uint8)
    nz = np.nonzero(v8)
    sl = tuple(slice(max(int(c.min()) - 6, 0), int(c.max()) + 7) for c in nz)
    v8, raw = v8[sl], raw[sl]
    # downsample to ~1.4-1.5 mm for fast rotation, label-safe
    f = [max(int(round(1.4 / z)), 1) for z in zooms0]
    v8d = v8[::f[0], ::f[1], ::f[2]]
    rawd = raw[::f[0], ::f[1], ::f[2]]
    zooms = tuple(z * ff for z, ff in zip(zooms0, f))

    # ---- column sheet: 2 rows (raw, v8) x 5 angles ----------------------
    fig, axes = plt.subplots(2, len(ANGLES), figsize=(4.2 * len(ANGLES), 15))
    for r, (nm, vol) in enumerate((("RAW", rawd), ("PROCESSED v8", v8d))):
        for c, (aname, ang) in enumerate(ANGLES):
            rgb, shp, zm = view_at(vol, zooms, ang)
            imshow_r(axes[r, c], rgb, shp, zm,
                     f"{nm} - {aname}" if True else None)
    fig.suptitle(f"{case}: raw model output vs processed v8, five 3D views",
                 fontsize=15, color="w")
    fig.patch.set_facecolor("black")
    fig.tight_layout()
    p1 = outdir / f"{case}_column_5views.png"
    fig.savefig(p1, dpi=120, facecolor="black")
    plt.close(fig)
    print("saved", p1)

    # ---- standalone sheet: rows = levels, 4 cols ------------------------
    present = [ID_TO_NAME[i] for i in sorted(np.unique(v8d)) if i > 0]
    rows = [n for n in TOP_DOWN if n in present]
    ncol = 4
    fig, axes = plt.subplots(len(rows), ncol,
                             figsize=(3.4 * ncol, 2.9 * len(rows)),
                             squeeze=False)
    name_to_id = {v: k for k, v in ID_TO_NAME.items()}
    for ri, lname in enumerate(rows):
        lid = name_to_id[lname]
        m = v8d == lid
        if not m.any():
            for ax in axes[ri]:
                ax.axis("off")
            continue
        nzl = np.nonzero(m)
        pad = [int(round(12.0 / z)) for z in zooms]
        csl = tuple(slice(max(int(c.min()) - p, 0), min(int(c.max()) + p + 1, s))
                    for c, p, s in zip(nzl, pad, v8d.shape))
        panels = [("raw obl", rawd[csl], 35), ("v8 obl", v8d[csl], 35),
                  ("raw lat", rawd[csl], 90), ("v8 lat", v8d[csl], 90)]
        for ci, (pname, vol, ang) in enumerate(panels):
            rgb, shp, zm = view_at(vol, zooms, ang, focus_ids={lid})
            imshow_r(axes[ri, ci], rgb, shp, zm,
                     f"{lname} {pname}" if ci == 0 or True else None)
    fig.suptitle(f"{case}: standalone vertebrae, raw vs processed v8 "
                 f"(posterior-oblique and lateral; neighbors dimmed)",
                 fontsize=14, color="w")
    fig.patch.set_facecolor("black")
    fig.tight_layout()
    p2 = outdir / f"{case}_standalone_sheet.png"
    fig.savefig(p2, dpi=110, facecolor="black")
    plt.close(fig)
    print("saved", p2)


if __name__ == "__main__":
    main()
