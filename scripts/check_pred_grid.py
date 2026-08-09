#!/usr/bin/env python
"""Compare CT vs prediction grids (shape / spacing / affine).

ITK-SNAP "Open Segmentation" needs an exact grid match. Also prints voxel
volume (mm^3) so audit cm^3 figures can be reconciled with the CT spacing.

Usage:
    python scripts/check_pred_grid.py \\
        --ct_dir data/AbdomenAtlasDemo \\
        --pred_dir AbdomenAtlasDemoPredict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def _info(path: Path) -> dict:
    img = nib.load(str(path))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    shape = tuple(int(s) for s in img.shape[:3])
    vox_mm3 = float(abs(np.linalg.det(img.affine[:3, :3])))
    return {
        "path": str(path),
        "shape": shape,
        "zooms_mm": tuple(round(z, 4) for z in zooms),
        "voxel_mm3": round(vox_mm3, 4),
        "axcodes": "".join(nib.aff2axcodes(img.affine)),
    }


def _close(a, b, tol: float = 1e-3) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def check_case(ct_case: Path, pred_case: Path) -> dict:
    ct_path = ct_case / "ct.nii.gz"
    combined = pred_case / "combined_labels.nii.gz"
    # Prefer L1 as a representative per-mask file; fall back to any vertebra.
    seg_dir = pred_case / "segmentations"
    mask_path = seg_dir / "vertebrae_L1.nii.gz"
    if not mask_path.is_file() and seg_dir.is_dir():
        cands = sorted(seg_dir.glob("vertebrae_*.nii.gz"))
        mask_path = cands[0] if cands else None

    out = {"case": ct_case.name, "ok": False, "files": {}, "issues": []}
    if not ct_path.is_file():
        out["issues"].append(f"missing CT: {ct_path}")
        return out
    if not combined.is_file():
        out["issues"].append(f"missing prediction: {combined}")
        return out

    ct = _info(ct_path)
    pred = _info(combined)
    out["files"]["ct"] = ct
    out["files"]["combined_labels"] = pred
    if mask_path and mask_path.is_file():
        out["files"][mask_path.name] = _info(mask_path)

    if ct["shape"] != pred["shape"]:
        out["issues"].append(f"shape mismatch CT{ct['shape']} vs pred{pred['shape']}")
    if not _close(ct["zooms_mm"], pred["zooms_mm"]):
        out["issues"].append(
            f"spacing mismatch CT{ct['zooms_mm']} vs pred{pred['zooms_mm']}"
        )
    if mask_path and mask_path.is_file():
        m = out["files"][mask_path.name]
        if m["shape"] != ct["shape"]:
            out["issues"].append(
                f"per-mask shape mismatch CT{ct['shape']} vs {mask_path.name}{m['shape']}"
            )

    out["ok"] = not out["issues"]
    out["itk_snap_overlay_ok"] = out["ok"]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ct_dir", type=Path, default=Path("data/AbdomenAtlasDemo"))
    p.add_argument("--pred_dir", type=Path, default=Path("AbdomenAtlasDemoPredict"))
    p.add_argument("--report", type=Path, default=None,
                   help="Optional JSON path under reports/")
    args = p.parse_args()

    if not args.ct_dir.is_dir():
        print(f"ERROR: CT dir not found: {args.ct_dir}", file=sys.stderr)
        return 1
    if not args.pred_dir.is_dir():
        print(f"ERROR: pred dir not found: {args.pred_dir}", file=sys.stderr)
        return 1

    cases = sorted(d for d in args.ct_dir.iterdir() if d.is_dir())
    results = []
    all_ok = True
    for ct_case in cases:
        pred_case = args.pred_dir / ct_case.name
        if not pred_case.is_dir():
            print(f"\n=== {ct_case.name}: SKIP (no pred folder) ===")
            all_ok = False
            continue
        r = check_case(ct_case, pred_case)
        results.append(r)
        status = "MATCH" if r["ok"] else "MISMATCH"
        print(f"\n=== {r['case']}: {status} ===")
        for key, info in r["files"].items():
            print(f"  {key:22s} shape={info['shape']}  zooms={info['zooms_mm']}  "
                  f"voxel={info['voxel_mm3']} mm^3  axcodes={info['axcodes']}")
        if r["ok"]:
            print("  ITK-SNAP: Open Main Image = CT, Open Segmentation = combined_labels "
                  "(grids match; no resample needed)")
        else:
            all_ok = False
            for issue in r["issues"]:
                print(f"  ISSUE: {issue}")
            print("  ITK-SNAP: grids differ — resample labels to CT before overlay "
                  "(keep submission preds on their native grid)")

    print(f"\nSUMMARY: {'all grids match' if all_ok else 'one or more mismatches'} "
          f"across {len(results)} case(s)")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2))
        print(f"Report written to {args.report}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
