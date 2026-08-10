"""Account for every raw-labeled voxel the pipeline dropped or recolored.

The envelope rule: the raw prediction is the outer boundary; inside it we fix
CLASS, we do not delete real bone. This diagnostic finds all voxels with
raw>0 and final==0 (the dropped pool), groups them into components, and for
each reports: volume, bone fraction, raw majority label, the AXIAL-RING link
(the level whose labeled mass the component touches through 2D bone
connectivity within its own axial slices - in the axial projection a
vertebra is one closed ring, so a true process chunk connects to its own
ring in-slice), the 3D bone-geodesic nearest level, and the s-position vote.
Also per-corridor totals of final-vs-raw disagreement (recolor map).

Renders: posterior + lateral views with the dropped pool in saturated red
over a dimmed final segmentation.

Usage: python scripts/diag_pool.py CT RAW FINAL OUTDIR
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
ID_TO_NAME = {i + 1: n for i, n in enumerate(NAMES_BOTTOM_UP)}
BONE_HU = 150.0
CMAP = np.array([tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)) for c in SNAP])


def main():
    ct_p, raw_p, fin_p, outdir = sys.argv[1:5]
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    fin_img = nib.load(fin_p)
    zooms = tuple(float(z) for z in fin_img.header.get_zooms()[:3])
    vox = float(np.prod(zooms))
    fin = np.asarray(fin_img.dataobj).astype(np.uint8)
    raw = np.asarray(nib.load(raw_p).dataobj).astype(np.uint8)
    ct = np.asarray(nib.load(ct_p).dataobj)
    nzc = np.nonzero(raw > 0)
    pad = np.ceil(15.0 / np.asarray(zooms)).astype(int)
    sl = tuple(slice(max(int(c.min() - p), 0), min(int(c.max() + p), n))
               for c, p, n in zip(nzc, pad, fin.shape))
    fin, raw, ct = fin[sl], raw[sl], ct[sl]
    bone = ct >= BONE_HU

    dropped = (raw > 0) & (fin == 0)
    drop_bone = dropped & bone
    drop_soft = dropped & ~bone
    print(f"dropped raw-labeled mass: {dropped.sum()*vox/1e3:.2f} cm3 "
          f"(bone {drop_bone.sum()*vox/1e3:.2f}, sub-bone-HU {drop_soft.sum()*vox/1e3:.2f})")
    recolored = (raw > 0) & (fin > 0) & (raw != fin)
    print(f"recolored (raw label changed): {recolored.sum()*vox/1e3:.2f} cm3")
    added = (raw == 0) & (fin > 0)
    print(f"added beyond raw envelope: {added.sum()*vox/1e3:.2f} cm3 "
          f"(bridges + bounded smoothing)")

    cc = cc3d.connected_components(drop_bone.astype(np.uint8), connectivity=26)
    ncomp = int(cc.max())
    counts = np.bincount(cc.ravel()); counts[0] = 0
    order = np.argsort(counts)[::-1]
    print(f"\ndropped BONE components: {ncomp}; showing all >= 100 mm3")
    print(f"{'#':>3} | {'cm3':>6} | {'raw-major':>9} | {'axial-ring link':>15} | "
          f"{'3D geodesic link (10mm)':>23}")
    st2 = ndimage.generate_binary_structure(2, 1)
    st3 = ndimage.generate_binary_structure(3, 1)
    total_linked = 0.0
    for comp_id in order:
        if counts[comp_id] * vox < 100.0 or comp_id == 0:
            continue
        comp = cc == comp_id
        objs = ndimage.find_objects(comp.astype(np.uint8))[0]
        rl = raw[comp]
        ids, cnt = np.unique(rl[rl > 0], return_counts=True)
        raw_major = ID_TO_NAME[int(ids[cnt.argmax()])]
        # axial-ring link: per slice, 2D bone component containing comp
        # voxels; which final labels does it include?
        ring_votes = {}
        z0, z1 = objs[2].start, objs[2].stop
        for z in range(z0, z1):
            cz = comp[:, :, z]
            if not cz.any():
                continue
            lab2, n2 = ndimage.label(bone[:, :, z], structure=st2)
            hit = np.unique(lab2[cz]); hit = hit[hit > 0]
            for h in hit:
                labs = fin[:, :, z][lab2 == h]
                for l in np.unique(labs[labs > 0]):
                    ring_votes[int(l)] = ring_votes.get(int(l), 0) + 1
        ring = max(ring_votes, key=ring_votes.get) if ring_votes else 0
        # 3D bone-geodesic link within 10 mm
        padv = np.ceil(12.0 / np.asarray(zooms)).astype(int)
        slc = tuple(slice(max(s.start - int(p), 0), min(s.stop + int(p), n))
                    for s, p, n in zip(objs, padv, comp.shape))
        it = int(np.ceil(10.0 / min(zooms)))
        reach = ndimage.binary_dilation(comp[slc], structure=st3, iterations=it,
                                        mask=bone[slc] | comp[slc])
        labs = fin[slc][reach]
        ids3, cnt3 = np.unique(labs[labs > 0], return_counts=True)
        geo = ID_TO_NAME[int(ids3[cnt3.argmax()])] if ids3.size else "-"
        ring_n = ID_TO_NAME[ring] if ring else "-"
        print(f"{comp_id:>3} | {counts[comp_id]*vox/1e3:>6.2f} | {raw_major:>9} | "
              f"{ring_n:>15} | {geo:>23}")
        if ring or ids3.size:
            total_linked += counts[comp_id] * vox / 1e3
    print(f"\nlinked (recoverable) dropped bone >= 100mm3: {total_linked:.2f} cm3")

    # render: pool in red over dimmed final, posterior + lateral
    fig, axes = plt.subplots(1, 2, figsize=(15, 11))
    views = [("posterior", lambda v: np.transpose(v, (1, 0, 2)),
              (zooms[1], zooms[0], zooms[2])),
             ("right lateral", lambda v: v, zooms)]
    show = fin.copy()
    POOL_ID = 30
    show[drop_bone] = POOL_ID
    cmap_x = np.vstack([CMAP, np.tile([[0.1, 0.1, 0.1]], (10, 1))])
    cmap_x[POOL_ID] = [1.0, 0.15, 0.15]
    for ax, (nm, tf, zm) in zip(axes, views):
        v = tf(show)
        hit, depth = surface_view(v, zm, "left")
        rgb = cmap_x[np.clip(hit, 0, len(cmap_x) - 1)]
        fillv = np.nanmax(depth) if np.isfinite(depth).any() else 0.0
        d = ndimage.gaussian_filter(np.where(np.isnan(depth), fillv, depth), 1.0)
        gy, gz = np.gradient(d, zm[1], zm[2])
        light = np.clip(0.35 + 0.65 / np.sqrt(1 + gy ** 2 + gz ** 2), 0, 1)
        rgb = rgb * light[..., None]
        dim = (hit > 0) & (hit != POOL_ID)
        rgb[dim] *= 0.45
        rgb[hit == 0] = 0
        ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower",
                  extent=[0, v.shape[1] * zm[1], 0, v.shape[2] * zm[2]],
                  aspect="equal", interpolation="nearest")
        ax.set_title(f"dropped raw-labeled bone (red) - {nm}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    p = outdir / "diag_pool.png"
    fig.savefig(p, dpi=110, facecolor="black")
    print("render:", p)


if __name__ == "__main__":
    main()
