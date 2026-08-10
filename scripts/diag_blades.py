"""Spinous-process ownership diagnosis at FULL resolution.

The reported defect (high-res ITK-SNAP review of v8): spinous blades are
assigned one level DOWN through T9..L2 - T9's blade wears T10's color, ...,
L1's blade wears L2's color, so L2 appears to own two blades. The earlier
gallery renders downsampled to ~1.4 mm and masked exactly this.

This tool measures blade identity against the one anatomical invariant that
cannot lie: A SPINOUS BLADE ATTACHES TO ITS OWN VERTEBRA AT ITS ROOT. The
root (lamina junction) sits at its vertebra's own disc band; only the TIP
imbricates downward. So for every level we locate its labeled mass inside
the posterior spinous corridor, take the mass's ROOT STRIP (the part at the
corridor's anterior boundary, where blades join the ring), and ask: which
level's BODY BAND contains that root? Bodies are disc-cut-validated and
correct, so root-band membership is trusted identity. A level whose corridor
root sits in the band of the level ABOVE is wearing its neighbor's blade.

Outputs (all FULL resolution, no downsampling):
  A_posterior_fullres.png     posterior surface zoom, raw | v8
  B_oblique_fullres.png       posterior-oblique +/-40deg, raw | v8
  C_midsag_slab_fullres.png   sagittal surface of |x|<=5mm slab with body
                              bands + measured blade roots drawn per level
  D_parasag_slices.png        parasagittal label slices raw | v8
  blade_root_table.json       per-level root-z, band owner, verdict
Usage:
  python scripts/diag_blades.py CT SEG1 NAME1 SEG2 NAME2 OUTDIR [--lo L3] [--hi T7]
"""
from __future__ import annotations

import json
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
NAME_TO_ID = {n: i + 1 for i, n in enumerate(NAMES_BOTTOM_UP)}
ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}
CMAP = np.array([tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)) for c in SNAP])
BONE_HU = 150.0


def shade(hit, depth, zm2):
    fv = np.nanmax(depth) if np.isfinite(depth).any() else 0.0
    d = ndimage.gaussian_filter(np.where(np.isnan(depth), fv, depth), 1.0)
    gy, gz = np.gradient(d, zm2[0], zm2[1])
    light = np.clip(0.35 + 0.65 / np.sqrt(1 + gy ** 2 + gz ** 2), 0, 1)
    rgb = CMAP[np.clip(hit, 0, len(CMAP) - 1)] * light[..., None]
    rgb[hit == 0] = 0.06
    return rgb


def posterior_view(vol, zooms, angle=0, psign=-1):
    if angle:
        vol = ndimage.rotate(vol, angle, axes=(0, 1), order=0,
                             reshape=True, prefilter=False)
    v = vol[:, ::-1, :] if psign > 0 else vol      # cast from the blade side
    v = np.transpose(v, (1, 0, 2))
    zm = (zooms[1], zooms[0], zooms[2])
    hit, depth = surface_view(v, zm, "left")
    return shade(hit, depth, (zm[1], zm[2])), (zm[1], zm[2])


def sagittal_view(vol, zooms):
    hit, depth = surface_view(vol, zooms, "left")   # cast along +x
    return shade(hit, depth, (zooms[1], zooms[2])), (zooms[1], zooms[2])


def imshow_r(ax, rgb, zm2, title):
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
              aspect=zm2[1] / zm2[0], interpolation="nearest")
    ax.set_title(title, fontsize=11, color="w")
    ax.set_xticks([]); ax.set_yticks([])


def centerline(seg):
    cx = np.full(seg.shape[2], np.nan)
    cy = np.full(seg.shape[2], np.nan)
    for z in range(seg.shape[2]):
        m = seg[:, :, z] > 0
        if m.sum() > 20:
            i, j = np.nonzero(m)
            cx[z], cy[z] = i.mean(), j.mean()
    ok = ~np.isnan(cx)
    idx = np.arange(seg.shape[2])
    cx = np.interp(idx, idx[ok], cx[ok])
    cy = np.interp(idx, idx[ok], cy[ok])
    return (ndimage.gaussian_filter1d(cx, 6.0),
            ndimage.gaussian_filter1d(cy, 6.0))


def body_bands(seg, cx, cy, zooms):
    """Per-level trusted z-band from the BODY mass (voxels within 18 mm of
    the column centerline in-plane). Bodies are disc-cut validated."""
    rb = 18.0
    bands = {}
    xx = np.arange(seg.shape[0], dtype=np.float32)
    yy = np.arange(seg.shape[1], dtype=np.float32)
    for lid in [int(v) for v in np.unique(seg) if v > 0]:
        m = seg == lid
        zs = np.nonzero(m.any(axis=(0, 1)))[0]
        acc_z, acc_w = [], []
        for z in zs:
            i, j = np.nonzero(m[:, :, z])
            d2 = ((i - cx[z]) * zooms[0]) ** 2 + ((j - cy[z]) * zooms[1]) ** 2
            n = int((d2 <= rb * rb).sum())
            if n:
                acc_z.append(z); acc_w.append(n)
        if not acc_z:
            continue
        acc_z = np.asarray(acc_z); acc_w = np.asarray(acc_w, dtype=np.float64)
        cw = np.cumsum(acc_w) / acc_w.sum()
        z0 = float(acc_z[np.searchsorted(cw, 0.04)])
        z1 = float(acc_z[np.searchsorted(cw, 0.96)])
        bands[lid] = (z0, z1)
    return bands


def posterior_sign(seg, cx, cy, zooms):
    """+1 if posterior (spinous side) is at increasing y, else -1. The arch
    pulls the whole-vertebra centroid posterior of the body centroid."""
    rb = 18.0
    d_all, d_body = [], []
    for z in range(0, seg.shape[2], 4):
        i, j = np.nonzero(seg[:, :, z] > 0)
        if i.size < 50:
            continue
        d2 = ((i - cx[z]) * zooms[0]) ** 2 + ((j - cy[z]) * zooms[1]) ** 2
        sel = d2 <= rb * rb
        if sel.sum() < 50:
            continue
        d_all.append(j.mean()); d_body.append(j[sel].mean())
    return 1 if np.mean(np.asarray(d_all) - np.asarray(d_body)) >= 0 else -1


def corridor_mask(seg, bone, cx, cy, zooms, psign, canal_mm=12.0,
                  half_mm=7.0):
    """Midline spinous-corridor bone mask + its per-z front index y0.
    Corridor = bone posterior of (body posterior edge + canal depth), within
    half_mm of the midline. Body-edge based: DISH cannot fuse it away."""
    nz = seg.shape[2]
    yB = np.full(nz, np.nan)
    rb = 18.0
    for z in range(nz):
        i, j = np.nonzero(seg[:, :, z] > 0)
        if i.size < 20:
            continue
        d2 = ((i - cx[z]) * zooms[0]) ** 2 + ((j - cy[z]) * zooms[1]) ** 2
        sel = d2 <= rb * rb
        if sel.sum() < 20:
            continue
        yB[z] = np.percentile(j[sel], 5 if psign < 0 else 95)
    ok = ~np.isnan(yB)
    idx = np.arange(nz)
    yB = np.interp(idx, idx[ok], yB[ok])
    yB = ndimage.median_filter(yB, 11)
    off = canal_mm / zooms[1]
    y0 = np.round(yB - off if psign < 0 else yB + off).astype(int)
    hw = max(int(round(half_mm / zooms[0])), 2)
    corr = np.zeros(seg.shape, dtype=bool)
    for z in range(nz):
        xc = int(round(cx[z]))
        xs = slice(max(xc - hw, 0), xc + hw + 1)
        if psign < 0:
            corr[xs, :max(y0[z] + 1, 0), z] = bone[xs, :max(y0[z] + 1, 0), z]
        else:
            corr[xs, y0[z]:, z] = bone[xs, y0[z]:, z]
    return corr, y0


def upward_violation(seg, corr, bands, zooms, lids, tol_mm=5.0):
    """PHYSICS METER (chain-proof, object-free): in the midline spinous
    corridor the only structures are spinous blades and interspinous bone.
    Blades angle CAUDALLY - a blade droops below its vertebra, never reaches
    above it. So a corridor voxel whose label's own body band lies BELOW the
    voxel (band top + tol < z) is anatomically impossible: that is exactly a
    blade wearing the label of the level below. Returns per-level cm3 of
    violating volume + the violation mask (for overlays)."""
    tol = int(round(tol_mm / zooms[2]))
    viol = np.zeros(seg.shape, dtype=bool)
    per = {}
    zidx = np.arange(seg.shape[2])
    for lid in lids:
        if lid not in bands:
            continue
        ztop = bands[lid][1]
        m = (seg == lid) & corr
        bad = m & (zidx[None, None, :] > ztop + tol)
        v = float(bad.sum()) * np.prod(zooms) / 1e3
        per[ID_TO_NAME[lid]] = round(v, 2)
        viol |= bad
    per["TOTAL"] = round(float(sum(per.values())), 2)
    return per, viol


def trace_roots(seg, corr, y0, psign, bands, zooms, lids):
    """Per level: corridor mass, its ROOT-STRIP z (the 7 mm of corridor
    nearest the canal, where blades join the ring), band owner of that root,
    number of blade pieces."""
    strip = np.zeros(seg.shape, dtype=bool)
    s1 = int(round(7.0 / zooms[1]))
    for z in range(seg.shape[2]):
        if psign < 0:
            a, b = max(y0[z] - s1, 0), max(y0[z] + 1, 0)
        else:
            a, b = y0[z], y0[z] + s1
        strip[:, a:b, z] = corr[:, a:b, z]

    def owner(zc):
        for lid, (a, b) in bands.items():
            if a - 3 <= zc <= b + 3:
                return lid
        # nearest band center
        best, bd = 0, 1e9
        for lid, (a, b) in bands.items():
            d = abs(zc - 0.5 * (a + b))
            if d < bd:
                bd, best = d, lid
        return best

    rows = []
    for lid in lids:
        m = (seg == lid) & corr
        v_cm3 = float(m.sum()) * np.prod(zooms) / 1e3
        if v_cm3 < 0.2:
            rows.append({"level": ID_TO_NAME[lid], "corridor_cm3": round(v_cm3, 2),
                         "verdict": "no corridor mass"})
            continue
        r = (seg == lid) & strip
        src = r if r.sum() > 30 else m
        zc = float(np.nonzero(src)[2].mean())
        ow = owner(zc)
        band = bands.get(lid)
        n_pieces = int(ndimage.label(m, structure=np.ones((3, 3, 3), bool))[1])
        # count only substantial pieces
        lab_m, nm = ndimage.label(m, structure=np.ones((3, 3, 3), bool))
        cnt = np.bincount(lab_m.ravel()); cnt[0] = 0
        n_big = int((cnt * np.prod(zooms) / 1e3 >= 0.25).sum())
        rows.append({
            "level": ID_TO_NAME[lid], "corridor_cm3": round(v_cm3, 2),
            "root_z_idx": round(zc, 1),
            "own_band_z": [round(b, 1) for b in band] if band else None,
            "root_band_owner": ID_TO_NAME[ow],
            "verdict": "OK" if ow == lid else f"SHIFTED(root in {ID_TO_NAME[ow]})",
            "blade_pieces_ge_0.25cm3": n_big, "pieces_total": n_pieces})
    return rows


def main():
    ct_p, seg1_p, name1, seg2_p, name2, outdir = sys.argv[1:7]
    lo_name = sys.argv[sys.argv.index("--lo") + 1] if "--lo" in sys.argv else "L3"
    hi_name = sys.argv[sys.argv.index("--hi") + 1] if "--hi" in sys.argv else "T7"
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    img = nib.load(seg2_p)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    segB = np.asarray(img.dataobj).astype(np.uint8)
    segA = np.asarray(nib.load(seg1_p).dataobj).astype(np.uint8)
    ct = np.asarray(nib.load(ct_p).dataobj)

    lids = list(range(NAME_TO_ID[lo_name], NAME_TO_ID[hi_name] + 1))
    # z-crop around the clean (segB) extent of the requested levels
    zsel = np.nonzero(np.isin(segB, lids).any(axis=(0, 1)))[0]
    pad = int(round(10.0 / zooms[2]))
    zsl = slice(max(int(zsel[0]) - pad, 0), min(int(zsel[-1]) + pad + 1, segB.shape[2]))
    segA, segB, ct = segA[:, :, zsl], segB[:, :, zsl], ct[:, :, zsl]
    # xy-crop to segB extent (+25 mm)
    nz = np.nonzero(segB)
    padxy = [int(round(25.0 / z)) for z in zooms[:2]]
    xsl = slice(max(int(nz[0].min()) - padxy[0], 0), int(nz[0].max()) + padxy[0] + 1)
    ysl = slice(max(int(nz[1].min()) - padxy[1], 0), int(nz[1].max()) + padxy[1] + 1)
    segA, segB, ct = segA[xsl, ysl], segB[xsl, ysl], ct[xsl, ysl]
    # scrub out-of-window labels (raw scatter) for rendering clarity
    keep = np.isin(segA, list(range(1, 25)))
    segA = np.where(keep, segA, 0).astype(np.uint8)
    bone = ct >= BONE_HU

    cx, cy = centerline(segB)
    psign = posterior_sign(segB, cx, cy, zooms)
    print(f"posterior side: {'+y' if psign > 0 else '-y'}")
    bands = body_bands(segB, cx, cy, zooms)
    corr, y0 = corridor_mask(segB, bone, cx, cy, zooms, psign)

    report = {"zooms": zooms, "posterior_sign": psign,
              "levels": [ID_TO_NAME[l] for l in lids],
              "bands_zidx": {ID_TO_NAME[l]: [round(a, 1), round(b, 1)]
                             for l, (a, b) in bands.items()}}
    viols = {}
    for nm, sg in ((name1, segA), (name2, segB)):
        rows = trace_roots(sg, corr, y0, psign, bands, zooms, lids)
        upv, vmask = upward_violation(sg, corr, bands, zooms, lids)
        viols[nm] = vmask
        report[nm] = rows
        report[f"{nm}_upward_violation_cm3"] = upv
        print(f"=== {nm}: blade root-band audit ===")
        for r in rows:
            print("  ", json.dumps(r))
        print(f"=== {nm}: UPWARD-VIOLATION (blade wearing lower level's "
              f"label) cm3 ===")
        print("  ", json.dumps(upv))
    (outdir / "blade_root_table.json").write_text(json.dumps(report, indent=1))

    # ---- E: violation overlay, midsagittal slab -------------------------
    hw5 = max(int(round(7.0 / zooms[0])), 2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 13))
    for ax, (nm, sg) in zip(axes, ((name1, segA), (name2, segB))):
        slab = np.zeros_like(sg)
        vs = np.zeros_like(sg)
        for z in range(sg.shape[2]):
            xc = int(round(cx[z]))
            xsl = slice(max(xc - hw5, 0), xc + hw5 + 1)
            slab[xsl, :, z] = sg[xsl, :, z]
            vs[xsl, :, z] = viols[nm][xsl, :, z]
        show = slab.copy()
        show[vs > 0] = 30
        cme = np.vstack([CMAP, np.tile([[0.08, 0.08, 0.08]], (10, 1))])
        cme[30] = [1.0, 0.12, 0.12]
        hit, depth = surface_view(show, zooms, "left")
        fv = np.nanmax(depth) if np.isfinite(depth).any() else 0.0
        d = ndimage.gaussian_filter(np.where(np.isnan(depth), fv, depth), 1.0)
        gy, gz = np.gradient(d, zooms[1], zooms[2])
        light = np.clip(0.4 + 0.6 / np.sqrt(1 + gy ** 2 + gz ** 2), 0, 1)
        rgb = cme[np.clip(hit, 0, len(cme) - 1)] * light[..., None]
        rgb[hit == 0] = 0.05
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
                  aspect=zooms[2] / zooms[1], interpolation="nearest")
        for lid in lids:
            if lid in bands:
                ax.axhline(bands[lid][1], color=CMAP[lid], lw=0.7, alpha=0.8)
        ax.set_title(f"{nm} - RED = corridor bone wearing a LOWER level's "
                     f"label (impossible)", fontsize=10, color="w")
        ax.set_xticks([]); ax.set_yticks([])
    fig.patch.set_facecolor("black")
    fig.savefig(outdir / "E_violation_overlay.png", dpi=170,
                facecolor="black", bbox_inches="tight")
    plt.close(fig)

    # ---- A: posterior full-res ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 13))
    for ax, (nm, sg) in zip(axes, ((name1, segA), (name2, segB))):
        rgb, zm2 = posterior_view(sg, zooms, psign=psign)
        imshow_r(ax, rgb, zm2, f"{nm} - posterior (full res)")
    _legend(fig, lids)
    fig.patch.set_facecolor("black")
    fig.savefig(outdir / "A_posterior_fullres.png", dpi=170, facecolor="black",
                bbox_inches="tight")
    plt.close(fig)

    # ---- B: obliques -----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 22))
    for r_, ang in enumerate((40, -40)):
        for c_, (nm, sg) in enumerate(((name1, segA), (name2, segB))):
            rgb, zm2 = posterior_view(sg, zooms, ang, psign=psign)
            imshow_r(axes[r_, c_], rgb, zm2, f"{nm} - oblique {ang:+d}deg")
    _legend(fig, lids)
    fig.patch.set_facecolor("black")
    fig.savefig(outdir / "B_oblique_fullres.png", dpi=150, facecolor="black",
                bbox_inches="tight")
    plt.close(fig)

    # ---- C: midsagittal slab surface + bands + roots ---------------------
    hw = max(int(round(5.0 / zooms[0])), 2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 13))
    for ax, (nm, sg) in zip(axes, ((name1, segA), (name2, segB))):
        slab = np.zeros_like(sg)
        for z in range(sg.shape[2]):
            xc = int(round(cx[z]))
            slab[max(xc - hw, 0):xc + hw + 1, :, z] = \
                sg[max(xc - hw, 0):xc + hw + 1, :, z]
        rgb, zm2 = sagittal_view(slab, zooms)
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
                  aspect=zm2[1] / zm2[0], interpolation="nearest")
        for lid in lids:
            if lid not in bands:
                continue
            a, b = bands[lid]
            col = CMAP[lid]
            ax.axhline(a, color=col, lw=0.8, alpha=0.85)
            ax.axhline(b, color=col, lw=0.8, alpha=0.85, ls=":")
            ax.text(2, 0.5 * (a + b), ID_TO_NAME[lid], color=col, fontsize=9,
                    va="center", fontweight="bold")
        rows = report[nm]
        for r_ in rows:
            if "root_z_idx" not in r_:
                continue
            lid = NAME_TO_ID[r_["level"]]
            zc = r_["root_z_idx"]
            bad = not r_["verdict"].startswith("OK")
            ax.plot([slab.shape[1] * 0.82], [zc], marker="<",
                    color=CMAP[lid], ms=9, mec="r" if bad else "w",
                    mew=1.6 if bad else 0.6)
        ax.set_title(f"{nm} - midsagittal slab; lines = body bands, "
                     f"arrows = measured blade ROOT (red edge = shifted)",
                     fontsize=10, color="w")
        ax.set_xticks([]); ax.set_yticks([])
    fig.patch.set_facecolor("black")
    fig.savefig(outdir / "C_midsag_slab_fullres.png", dpi=170,
                facecolor="black", bbox_inches="tight")
    plt.close(fig)

    # ---- D: parasagittal slices -----------------------------------------
    offs_mm = (-3.0, -1.0, 1.0, 3.0)
    fig, axes = plt.subplots(2, len(offs_mm), figsize=(4.2 * len(offs_mm), 22))
    for r_, (nm, sg) in enumerate(((name1, segA), (name2, segB))):
        for c_, off in enumerate(offs_mm):
            sl2 = np.zeros(sg.shape[1:], dtype=np.uint8)
            for z in range(sg.shape[2]):
                x = int(round(cx[z] + off / zooms[0]))
                if 0 <= x < sg.shape[0]:
                    sl2[:, z] = sg[x, :, z]
            rgb = CMAP[np.clip(sl2, 0, len(CMAP) - 1)]
            rgb[sl2 == 0] = 0.05
            axes[r_, c_].imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
                                aspect=zooms[2] / zooms[1],
                                interpolation="nearest")
            axes[r_, c_].set_title(f"{nm} x={off:+.0f}mm", fontsize=10, color="w")
            axes[r_, c_].set_xticks([]); axes[r_, c_].set_yticks([])
    fig.patch.set_facecolor("black")
    fig.savefig(outdir / "D_parasag_slices.png", dpi=150, facecolor="black",
                bbox_inches="tight")
    plt.close(fig)
    print("figures ->", outdir)


def _legend(fig, lids):
    handles = [plt.Line2D([0], [0], marker="s", ls="", ms=10,
                          mfc=CMAP[l], mec="w", mew=0.3,
                          label=f"{ID_TO_NAME[l]} (id {l})")
               for l in sorted(lids, reverse=True)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(lids), 8),
               facecolor="black", labelcolor="w", fontsize=9, frameon=False)


if __name__ == "__main__":
    main()
