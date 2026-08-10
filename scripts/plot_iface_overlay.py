"""CT-underlay overlays of a label pair's boundary, before vs after.

For each requested pair: locate the A|B interface in the AFTER volume, take
the sagittal slice through its centroid, and draw the CT with both labels'
contours - BEFORE dashed, AFTER solid. Judges where a polish/arbitration
stage actually put the boundary relative to the visible disc/joint space.

Usage:
  python scripts/plot_iface_overlay.py OUT.png CT.nii.gz BEFORE.nii.gz \
      AFTER.nii.gz T8|T9,T9|T10,...
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

NAMES_BOTTOM_UP = (["L5", "L4", "L3", "L2", "L1"]
                   + [f"T{i}" for i in range(12, 0, -1)]
                   + [f"C{i}" for i in range(7, 0, -1)])
NAME_TO_ID = {n: i + 1 for i, n in enumerate(NAMES_BOTTOM_UP)}
COL = {"upper": "#ff5252", "lower": "#40c4ff"}


def main():
    out, ct_p, a_p, b_p, pair_arg = sys.argv[1:6]
    ct = np.asarray(nib.load(ct_p).dataobj)
    A = np.asarray(nib.load(a_p).dataobj).astype(np.uint8)
    B = np.asarray(nib.load(b_p).dataobj).astype(np.uint8)
    zooms = tuple(float(z) for z in nib.load(b_p).header.get_zooms()[:3])
    pairs = [p.split("|") for p in pair_arg.split(",")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.4 * len(pairs), 6.8),
                             squeeze=False)
    for c, (u, l) in enumerate(pairs):
        ax = axes[0, c]
        ui, li = NAME_TO_ID[u], NAME_TO_ID[l]
        m = (B == ui) | (B == li)
        nz = np.nonzero(m)
        if nz[0].size == 0:
            ax.axis("off"); continue
        x0 = int(np.median(nz[0]))
        pad = np.ceil(10.0 / np.asarray(zooms)).astype(int)
        sy = slice(max(int(nz[1].min() - pad[1]), 0), int(nz[1].max() + pad[1]))
        sz = slice(max(int(nz[2].min() - pad[2]), 0), int(nz[2].max() + pad[2]))
        img = np.clip(ct[x0, sy, sz], -250, 1300).T
        ax.imshow(img, cmap="gray", origin="lower", aspect=zooms[2] / zooms[1],
                  interpolation="bilinear")
        for seg, ls, lw, al in ((A, "--", 1.1, 0.9), (B, "-", 1.6, 1.0)):
            for lid, key in ((ui, "upper"), (li, "lower")):
                sl2 = (seg[x0, sy, sz] == lid).T.astype(float)
                if sl2.any():
                    ax.contour(sl2, levels=[0.5], colors=[COL[key]],
                               linestyles=ls, linewidths=lw, alpha=al)
        ax.set_title(f"{u}|{l}  (x={x0})  dashed=before solid=after",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("render:", out)


if __name__ == "__main__":
    main()
