"""Diagnose truncated posterior processes (spinous / transverse) per level.

For each level in the band: define a SPINOUS corridor (posterior midline) and
a TRANSVERSE corridor (posterolateral wings), then measure, per corridor:
  - the level's current mass and its caudal reach,
  - frontier contacts (which label, pool, or unlabeled CT bone borders it),
  - RECLAIMABLE bone: unlabeled CT bone geodesically contiguous with the
    level's process mass (masked dilation, capped), i.e. bone the image says
    continues but no label owns,
  - RIB-RISK: whether uncapped growth explodes (costovertebral connection).

Renders: posterior surface view (raw vs final), midsagittal CT strip with
label contours (blade ownership in profile), and prints the accounting table.

Usage: python scripts/diag_processes.py CT.nii.gz RAW.nii.gz FINAL.nii.gz OUTDIR
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
from render_lateral import SNAP

NAMES_BOTTOM_UP = (["L5", "L4", "L3", "L2", "L1"]
                   + [f"T{i}" for i in range(12, 0, -1)]
                   + [f"C{i}" for i in range(7, 0, -1)])
NAME_TO_ID = {n: i + 1 for i, n in enumerate(NAMES_BOTTOM_UP)}
ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}
BONE_HU = 150.0
STRUCT6 = ndimage.generate_binary_structure(3, 1)
CMAP = np.array([tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)) for c in SNAP])


def centerline(seg):
    nz = seg.shape[2]
    cx = np.full(nz, np.nan); cy = np.full(nz, np.nan)
    for z in np.nonzero(seg.any(axis=(0, 1)))[0]:
        pts = np.nonzero(seg[:, :, z])
        cx[z], cy[z] = pts[0].mean(), pts[1].mean()
    ok = ~np.isnan(cx)
    idx = np.nonzero(ok)[0]
    cx[ok] = ndimage.gaussian_filter1d(cx[ok], 12)
    cy[ok] = ndimage.gaussian_filter1d(cy[ok], 12)
    for arr in (cx, cy):
        arr[:idx[0]] = arr[idx[0]]; arr[idx[-1]:] = arr[idx[-1]]
        m = np.isnan(arr)
        arr[m] = np.interp(np.nonzero(m)[0], idx, arr[idx])
    return cx, cy


def grow_geodesic(seed, domain, zooms, max_mm):
    it = int(np.ceil(max_mm / min(zooms)))
    return ndimage.binary_dilation(seed, structure=STRUCT6, iterations=it,
                                   mask=domain | seed)


def main():
    ct_p, raw_p, fin_p, outdir = sys.argv[1:5]
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fin_img = nib.load(fin_p)
    zooms = tuple(float(z) for z in fin_img.header.get_zooms()[:3])
    fin = np.asarray(fin_img.dataobj).astype(np.uint8)
    raw = np.asarray(nib.load(raw_p).dataobj).astype(np.uint8)
    ct = np.asarray(nib.load(ct_p).dataobj)
    # crop to band ROI
    band_ids = [NAME_TO_ID[n] for n in
                ("T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2")]
    m = np.isin(fin, band_ids)
    nzc = np.nonzero(m)
    pad = np.ceil(30.0 / np.asarray(zooms)).astype(int)
    sl = tuple(slice(max(int(c.min() - p), 0), min(int(c.max() + p), n))
               for c, p, n in zip(nzc, pad, fin.shape))
    fin, raw, ct = fin[sl], raw[sl], ct[sl]
    bone = ct >= BONE_HU
    cx, cy = centerline(np.isin(fin, band_ids))
    X = np.arange(fin.shape[0])[:, None, None] * 1.0
    Y = np.arange(fin.shape[1])[None, :, None] * 1.0
    dx = (X - cx[None, None, :]) * zooms[0]
    dy = (Y - cy[None, None, :]) * zooms[1]
    # corridors (y grows posterior in this dataset's orientation: verify by
    # checking where arch mass sits relative to centerline)
    dyf = np.broadcast_to(dy, fin.shape)
    selb = np.isin(fin, band_ids) & (np.abs(dyf) > 12)
    arch_side = np.sign(np.median(dyf[selb]))
    dyp = dy * arch_side  # positive = posterior
    SPI = (np.abs(dx) < 9.0) & (dyp > 18.0)
    TRV = (np.abs(dx) >= 12.0) & (np.abs(dx) < 40.0) & (dyp > 2.0) & (dyp < 30.0)
    unl_bone = bone & (fin == 0)
    pool = unl_bone & (raw > 0)
    print(f"arch side sign={arch_side:+.0f}; ROI {fin.shape}; "
          f"unlabeled bone {unl_bone.sum()*np.prod(zooms)/1e3:.1f} cm3 "
          f"(of it raw-labeled pool {pool.sum()*np.prod(zooms)/1e3:.1f} cm3)")
    vox = float(np.prod(zooms))
    st = ndimage.generate_binary_structure(3, 1)
    print(f"\n{'lvl':>4} | {'corr':>4} | {'own cm3':>8} | {'reach mm':>8} | "
          f"{'next-lvl contact mm2':>20} | {'unlab-contig cm3 (30mm)':>23} | {'uncapped cm3':>12}")
    results = {}
    for name in ("T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "L1", "L2"):
        lid = NAME_TO_ID[name]
        nxt = NAME_TO_ID[NAMES_BOTTOM_UP[lid - 2]] if lid >= 2 else 0  # caudal neighbor
        for cname, corr in (("SPI", SPI), ("TRV", TRV)):
            own = (fin == lid) & corr
            if not own.any():
                print(f"{name:>4} | {cname:>4} | {'-':>8}")
                continue
            zs = np.nonzero(own.any(axis=(0, 1)))[0]
            reach = (zs.max() - zs.min()) * zooms[2]
            shell = ndimage.binary_dilation(own, structure=st) & ~own
            cnx = int((shell & (fin == nxt)).sum()) * (vox ** (2 / 3))
            grow = grow_geodesic(own, unl_bone, zooms, 30.0) & unl_bone
            big = grow_geodesic(own, unl_bone, zooms, 80.0) & unl_bone
            results[(name, cname)] = dict(own=own.sum() * vox / 1e3,
                                          reach=reach, cnx=cnx,
                                          grow=grow.sum() * vox / 1e3,
                                          big=big.sum() * vox / 1e3)
            r = results[(name, cname)]
            print(f"{name:>4} | {cname:>4} | {r['own']:>8.1f} | {r['reach']:>8.0f} | "
                  f"{r['cnx']:>20.0f} | {r['grow']:>23.2f} | {r['big']:>12.1f}")

    # ---- renders ---------------------------------------------------------
    from render_lateral import surface_view, shaded_rgb
    fig, axes = plt.subplots(1, 4, figsize=(26, 10))
    for i, (nm, vol) in enumerate((("raw", raw), ("final", fin))):
        # posterior view: ray-cast along y from the arch side
        v = vol[:, ::-1, :] if arch_side > 0 else vol
        vt = np.transpose(v, (1, 0, 2))         # cast along axis 0 = y
        hit, depth = surface_view(vt, (zooms[1], zooms[0], zooms[2]), "left")
        rgb = shaded_rgb(hit, depth, (zooms[1], zooms[0], zooms[2]))
        axes[i].imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
                       extent=[0, vol.shape[0] * zooms[0], 0, vol.shape[2] * zooms[2]],
                       aspect="equal", interpolation="nearest")
        axes[i].set_title(f"{nm} posterior view")
        axes[i].set_xticks([]); axes[i].set_yticks([])
    # midsagittal CT strip with label contours, raw and final
    x0 = int(round(np.nanmedian(cx)))
    img = np.clip(ct[x0], -250, 1400).T
    for j, (nm, vol) in enumerate((("raw", raw), ("final", fin))):
        ax = axes[2 + j]
        ax.imshow(img, cmap="gray", origin="lower", aspect=zooms[2] / zooms[1],
                  interpolation="bilinear")
        sl2 = vol[x0].T
        for lid in band_ids + [NAME_TO_ID["T4"], NAME_TO_ID["L3"]]:
            mm2 = (sl2 == lid).astype(float)
            if mm2.any():
                ax.contour(mm2, levels=[0.5], colors=[CMAP[lid]], linewidths=1.4)
        ax.set_title(f"{nm} midsagittal x={x0} (blade profiles)")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    p = outdir / "diag_processes.png"
    fig.savefig(p, dpi=110, facecolor="white")
    print("\nrender:", p)


if __name__ == "__main__":
    main()
