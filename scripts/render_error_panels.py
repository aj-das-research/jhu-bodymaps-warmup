"""Render an image panel for every flagged vertebra in an audit report.

For each mask carrying flags (FRAGMENTED / SIZE / ORDER / ORGAN_OVERLAP),
writes a PNG with sagittal and coronal crops centered on the vertebra:
CT in a bone window underneath, connected components of the mask each in a
distinct color (so fragmentation is visible at a glance), title with the
flags and volume. Run once per prediction folder (raw / shapekit / refined)
and compare folders side by side.

Usage:
    python scripts/render_error_panels.py \
        --pred_dir AbdomenAtlasDemoPredict \
        --ct_root data/AbdomenAtlasDemo \
        --report reports/audit_before.json \
        --out reports/figures/raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import cc3d
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BONE_WINDOW = (-200.0, 1200.0)
CROP_MM = 90.0          # half-extent of the crop around the centroid, in mm
COMPONENT_COLORS = ["#ff3b30", "#34c759", "#007aff", "#ff9500", "#af52de",
                    "#5ac8fa", "#ffcc00", "#ff2d55", "#a2845e", "#8e8e93"]


def _window(ct: np.ndarray) -> np.ndarray:
    lo, hi = BONE_WINDOW
    return np.clip((ct - lo) / (hi - lo), 0, 1)


def _crop_range(center: float, half_vox: float, size: int) -> slice:
    lo = max(int(center - half_vox), 0)
    hi = min(int(center + half_vox) + 1, size)
    return slice(lo, hi)


def render_mask(ct_path: Path, mask_path: Path, title: str, out_png: Path) -> None:
    ct_img = nib.load(str(ct_path))
    ct = np.asarray(ct_img.dataobj).astype(np.float32)
    mask = np.asarray(nib.load(str(mask_path)).dataobj) > 0
    if not mask.any():
        return
    if mask.shape != ct.shape:
        print(f"[warn] grid mismatch for {mask_path.name}; rendering mask without CT")
        ct = np.zeros_like(mask, dtype=np.float32)

    comps = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    n_comp = int(comps.max())
    # Order component ids by size (largest first) for stable coloring.
    sizes = np.bincount(comps.ravel()); sizes[0] = 0
    order = {int(cid): rank for rank, cid in
             enumerate(np.argsort(sizes)[::-1][:max(n_comp, 1)])}

    center = ndimage.center_of_mass(mask)
    spac = np.sqrt((ct_img.affine[:3, :3] ** 2).sum(axis=0))
    halves = [CROP_MM / max(s, 1e-3) for s in spac]

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, (fixed_axis, name) in zip(axes, [(0, "sagittal"), (1, "coronal")]):
        idx = int(round(center[fixed_axis]))
        sl = [slice(None)] * 3
        sl[fixed_axis] = idx
        plane_axes = [a for a in range(3) if a != fixed_axis]
        crop = tuple(
            _crop_range(center[a], halves[a], mask.shape[a]) if a in plane_axes else idx
            for a in range(3)
        )
        ct_plane = _window(ct[crop])
        comp_plane = comps[crop]
        ax.imshow(ct_plane.T, cmap="gray", origin="lower", interpolation="nearest")
        for cid in np.unique(comp_plane):
            if cid == 0:
                continue
            color = COMPONENT_COLORS[order.get(int(cid), 0) % len(COMPONENT_COLORS)]
            overlay = np.zeros((*comp_plane.T.shape, 4))
            m = (comp_plane.T == cid)
            overlay[m] = matplotlib.colors.to_rgba(color, alpha=0.55)
            ax.imshow(overlay, origin="lower", interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=130)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render panels for flagged vertebrae.")
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--ct_root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path,
                        help="Audit JSON produced by audit_predictions.py on --pred_dir.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--all", action="store_true",
                        help="Render every mask, not only flagged ones.")
    args = parser.parse_args()

    summaries = json.loads(args.report.read_text())
    n = 0
    for case in summaries:
        ct_path = args.ct_root / case["case"] / "ct.nii.gz"
        for r in case["masks"]:
            if not r["flags"] and not args.all:
                continue
            if r["voxels"] == 0:
                continue
            mask_path = args.pred_dir / case["case"] / "segmentations" / f"{r['name']}.nii.gz"
            if not mask_path.exists():
                continue
            flags = " ".join(r["flags"]) or "no flags"
            title = (f"{case['case']}  {r['name']}  vol={r.get('volume_cm3', '?')}cm3  "
                     f"components={r['components']}  [{flags}]")
            out_png = args.out / case["case"] / f"{r['name']}.png"
            render_mask(ct_path, mask_path, title, out_png)
            n += 1
    print(f"\n{n} panel(s) in {args.out}")


if __name__ == "__main__":
    main()