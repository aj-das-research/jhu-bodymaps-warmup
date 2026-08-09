#!/usr/bin/env python
"""3×2 diagnostic panel: CT physics + raw vs ShapeKit masks.

Default target from the warm-up case study:
  BDMAP_00000006 / vertebrae_C1 / axial slice 319

Layout
  [ CT bone window ] [ HU heatmap + bone contour ]
  [ raw on CT      ] [ shapekit on CT            ]
  [ diff on CT     ] [ correction map (edits)    ]

Default output (structured):
  reports/figures/raw_vs_shapekit/<case>/<vertebra>/<axis>_<slice>_panel.png

Usage:
  python scripts/plot_slice_compare_panel.py
  python scripts/plot_slice_compare_panel.py --pdf
  python scripts/plot_slice_compare_panel.py --diff_slices
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

AXIS_NAME = {0: "sagittal", 1: "coronal", 2: "axial"}


def _plane(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    idx = int(np.clip(idx, 0, vol.shape[axis] - 1))
    return np.take(vol, idx, axis=axis)


def _aspect(zooms: tuple, axis: int) -> float:
    """imshow aspect so pixels are isotropic in mm (rows/cols of the plane)."""
    a_cols, a_rows = [a for a in range(3) if a != axis]
    return float(zooms[a_rows] / max(zooms[a_cols], 1e-6))


def _window(ct: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((ct - lo) / (hi - lo), 0, 1)


def _load_mask(pred_root: Path, case: str, vertebra: str) -> np.ndarray:
    path = pred_root / case / "segmentations" / f"{vertebra}.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(f"missing mask: {path}")
    return np.asarray(nib.load(str(path)).dataobj) > 0


def default_out_path(figures_root: Path, left: str, right: str,
                     case: str, vertebra: str, axis: int, slice_idx: int) -> Path:
    return (figures_root / f"{left}_vs_{right}" / case / vertebra /
            f"{AXIS_NAME[axis]}_{slice_idx}_panel.png")


def _parse_slices(text: str | None) -> list[int]:
    if not text:
        return []
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def diff_slice_indices(raw: np.ndarray, other: np.ndarray, axis: int) -> list[tuple[int, int, int]]:
    """Return (idx, n_deleted, n_added) for planes where masks differ."""
    removed = raw & ~other
    added = other & ~raw
    sum_axes = tuple(i for i in range(3) if i != axis)
    rem_counts = removed.sum(axis=sum_axes)
    add_counts = added.sum(axis=sum_axes)
    rows: list[tuple[int, int, int]] = []
    for i in range(raw.shape[axis]):
        ro, ao = int(rem_counts[i]), int(add_counts[i])
        if ro or ao:
            rows.append((i, ro, ao))
    return rows


def _overlay_mask(ax, ct_win, mask, aspect, color=(0.2, 0.85, 1.0), alpha=0.45):
    ax.imshow(ct_win.T, cmap="gray", origin="lower", aspect=aspect,
              interpolation="nearest", vmin=0, vmax=1)
    if mask.any():
        rgba = np.zeros((*mask.T.shape, 4))
        rgba[mask.T] = (*color, alpha)
        ax.imshow(rgba, origin="lower", aspect=aspect, interpolation="nearest")
    ax.axis("off")


def plot_panel(ct: np.ndarray, raw: np.ndarray, other: np.ndarray, zooms: tuple,
               axis: int, slice_idx: int, *, bone_lo: float, bone_hi: float,
               bone_hu: float, left_label: str, right_label: str,
               title: str) -> plt.Figure:
    ct2 = _plane(ct, axis, slice_idx).astype(np.float32)
    raw2 = _plane(raw, axis, slice_idx)
    oth2 = _plane(other, axis, slice_idx)
    asp = _aspect(zooms, axis)
    win = _window(ct2, bone_lo, bone_hi)

    removed = raw2 & ~oth2
    added = oth2 & ~raw2
    kept = raw2 & oth2

    fig, axes = plt.subplots(3, 2, figsize=(11, 14))
    fig.suptitle(title, fontsize=12, y=0.995)

    axes[0, 0].imshow(win.T, cmap="gray", origin="lower", aspect=asp,
                      interpolation="nearest", vmin=0, vmax=1)
    axes[0, 0].set_title(f"CT bone window [{bone_lo:.0f}, {bone_hi:.0f}] HU")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(ct2.T, cmap="turbo", origin="lower", aspect=asp,
                           interpolation="nearest",
                           vmin=float(np.percentile(ct2, 1)),
                           vmax=float(np.percentile(ct2, 99)))
    if (ct2 > bone_hu).any():
        axes[0, 1].contour(ct2.T > bone_hu, levels=[0.5], colors=["white"],
                           linewidths=0.6, origin="lower")
    axes[0, 1].set_title(f"HU heatmap + bone contour (HU>{bone_hu:.0f})")
    axes[0, 1].axis("off")
    cbar = fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.02)
    cbar.set_label("HU")

    _overlay_mask(axes[1, 0], win, raw2, asp, color=(0.15, 0.75, 1.0))
    axes[1, 0].set_title(f"{left_label} mask on CT")

    _overlay_mask(axes[1, 1], win, oth2, asp, color=(0.3, 1.0, 0.45))
    axes[1, 1].set_title(f"{right_label} mask on CT")

    axes[2, 0].imshow(win.T, cmap="gray", origin="lower", aspect=asp,
                      interpolation="nearest", vmin=0, vmax=1)
    overlay = np.zeros((*win.T.shape, 4))
    overlay[kept.T] = (0.25, 0.45, 1.0, 0.55)
    overlay[removed.T] = (1.0, 0.2, 0.15, 0.75)
    overlay[added.T] = (0.15, 0.9, 0.3, 0.75)
    axes[2, 0].imshow(overlay, origin="lower", aspect=asp, interpolation="nearest")
    axes[2, 0].set_title("Diff on CT  (blue=kept, red=removed, green=added)")
    axes[2, 0].axis("off")

    corr = np.zeros((*win.T.shape, 3))
    corr[removed.T] = (1.0, 0.15, 0.1)
    corr[added.T] = (0.1, 0.95, 0.25)
    axes[2, 1].imshow(corr, origin="lower", aspect=asp, interpolation="nearest")
    n_rem, n_add = int(removed.sum()), int(added.sum())
    axes[2, 1].set_title(f"{right_label} correction map  (−{n_rem} / +{n_add} vox)")
    axes[2, 1].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def _write_one(ct, raw, other, zooms, args, slice_idx: int, out: Path | None = None) -> Path:
    out_path = out or default_out_path(
        args.figures_root, args.left_name, args.right_name,
        args.case, args.vertebra, args.axis, slice_idx,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = (f"{args.case}  {args.vertebra}  "
             f"{AXIS_NAME[args.axis]} slice {slice_idx}  "
             f"({args.left_name} vs {args.right_name})")
    fig = plot_panel(
        ct, raw, other, zooms, args.axis, slice_idx,
        bone_lo=args.bone_lo, bone_hi=args.bone_hi, bone_hu=args.bone_hu,
        left_label=args.left_name, right_label=args.right_name, title=title,
    )
    fig.savefig(str(out_path), dpi=160)
    print(f"wrote {out_path}")
    if args.pdf:
        pdf = out_path.with_suffix(".pdf")
        fig.savefig(str(pdf))
        print(f"wrote {pdf}")
    plt.close(fig)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ct_root", type=Path, default=Path("data/AbdomenAtlasDemo"))
    p.add_argument("--raw", type=Path, default=Path("AbdomenAtlasDemoPredict"))
    p.add_argument("--other", type=Path, default=None,
                   help="Right-hand prediction root (default: --shapekit).")
    p.add_argument("--shapekit", type=Path, default=Path("AbdomenAtlasDemoPredict_shapekit"),
                   help="ShapeKit prediction root (used when --other is omitted).")
    p.add_argument("--left_name", default="raw", help="Tag for left/raw folder in paths.")
    p.add_argument("--right_name", default="shapekit",
                   help="Tag for right/ShapeKit folder in paths.")
    p.add_argument("--case", default="BDMAP_00000006")
    p.add_argument("--vertebra", default="vertebrae_C1")
    p.add_argument("--axis", type=int, default=2, choices=(0, 1, 2))
    p.add_argument("--slice", type=int, default=319, dest="slice_idx")
    p.add_argument("--diff_slices", action="store_true",
                   help="Write a panel for every plane where left XOR right is non-empty.")
    p.add_argument("--also_slices", type=str, default="",
                   help="Extra comma-separated slice indices to plot (e.g. 325,326).")
    p.add_argument("--bone_hu", type=float, default=150.0)
    p.add_argument("--bone_lo", type=float, default=-200.0)
    p.add_argument("--bone_hi", type=float, default=1200.0)
    p.add_argument("--figures_root", type=Path, default=Path("reports/figures"))
    p.add_argument("--out", type=Path, default=None,
                   help="Override output PNG path (single-slice mode only).")
    p.add_argument("--pdf", action="store_true", help="Also write a PDF sibling.")
    args = p.parse_args()

    other_root = args.other if args.other is not None else args.shapekit

    ct_path = args.ct_root / args.case / "ct.nii.gz"
    if not ct_path.is_file():
        print(f"ERROR: CT not found: {ct_path}", file=sys.stderr)
        return 1

    ct_img = nib.load(str(ct_path))
    ct = np.asarray(ct_img.dataobj)
    zooms = tuple(float(z) for z in ct_img.header.get_zooms()[:3])
    raw = _load_mask(args.raw, args.case, args.vertebra)
    other = _load_mask(other_root, args.case, args.vertebra)
    if raw.shape != ct.shape or other.shape != ct.shape:
        print(f"ERROR: shape mismatch ct{ct.shape} raw{raw.shape} other{other.shape}",
              file=sys.stderr)
        return 1

    slices: list[int] = []
    if args.diff_slices:
        rows = diff_slice_indices(raw, other, args.axis)
        print(f"{len(rows)} {AXIS_NAME[args.axis]} slices with edits "
              f"({args.left_name} vs {args.right_name}):")
        print(f"{'idx':>5} {'del':>6} {'add':>6} {'edit':>6}")
        for idx, n_del, n_add in rows:
            print(f"{idx:5d} {n_del:6d} {n_add:6d} {n_del + n_add:6d}")
            slices.append(idx)
    else:
        slices.append(args.slice_idx)

    for s in _parse_slices(args.also_slices):
        if s not in slices:
            slices.append(s)

    if args.diff_slices or len(slices) > 1:
        if args.out is not None:
            print("WARNING: --out ignored in multi-slice mode; using structured paths",
                  file=sys.stderr)
        for s in slices:
            _write_one(ct, raw, other, zooms, args, s)
    else:
        _write_one(ct, raw, other, zooms, args, slices[0], out=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
