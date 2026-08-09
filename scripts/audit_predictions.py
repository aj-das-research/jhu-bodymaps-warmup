"""Audit vertebra segmentation predictions.

Reports, per case and per vertebra mask: voxel count, number of 3D connected
components, and the largest-component fraction. Flags EMPTY, FRAGMENTED, and
superior-inferior ordering violations. Run before and after post-processing;
the diff is the evidence for the submission note.

Usage:
    python scripts/audit_predictions.py --pred_dir AbdomenAtlasDemoPredict
    python scripts/audit_predictions.py --pred_dir ... --report audit_before.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage

_REGION_RANK = {"C": 0, "T": 1, "L": 2, "S": 3}
_VERT_RE = re.compile(r"vertebrae?_([CTLS])(\d+)", re.IGNORECASE)


def _vertebra_sort_key(name: str) -> tuple:
    m = _VERT_RE.search(name)
    if not m:
        return (99, 0)
    return (_REGION_RANK.get(m.group(1).upper(), 98), int(m.group(2)))


def voxel_volume_mm3(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(affine[:3, :3])))


def _world_si_coord(centroid_vox: tuple, affine: np.ndarray) -> float:
    v = np.array([centroid_vox[0], centroid_vox[1], centroid_vox[2], 1.0])
    return float((affine @ v)[2])


def flag_ordering(centroids_world_si: dict) -> list:
    ordered = sorted(centroids_world_si.items(), key=lambda kv: _vertebra_sort_key(kv[0]))
    if len(ordered) < 3:
        return []
    names = [n for n, _ in ordered]
    si = [z for _, z in ordered]
    decreasing = si[-1] < si[0]
    violations = []
    for i in range(1, len(si)):
        step = si[i] - si[i - 1]
        if (decreasing and step > 0) or (not decreasing and step < 0):
            violations.append(names[i])
    return violations


def audit_mask(path: Path) -> dict:
    img = nib.load(str(path))
    mask = np.asarray(img.dataobj) > 0
    voxels = int(mask.sum())
    entry = {"name": path.name.replace(".nii.gz", ""), "voxels": voxels,
             "volume_cm3": round(voxels * voxel_volume_mm3(img.affine) / 1000.0, 1),
             "components": 0, "largest_fraction": 0.0, "flags": []}
    if voxels == 0:
        entry["flags"].append("EMPTY")
        return entry
    labeled = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    entry["components"] = int((counts > 0).sum())
    entry["largest_fraction"] = round(float(counts.max()) / voxels, 4)
    entry["si_world"] = _world_si_coord(ndimage.center_of_mass(mask), img.affine)
    if entry["components"] > 1:
        entry["flags"].append(f"FRAGMENTED({entry['components']})")
    return entry


def audit_case(case: Path, organ_gt_root: Path | None = None) -> dict:
    seg_dir = case / "segmentations"
    vert_files = sorted(
        (p for p in seg_dir.glob("*.nii.gz") if p.name.lower().startswith("vertebrae")),
        key=lambda p: _vertebra_sort_key(p.name),
    )
    results = [audit_mask(f) for f in vert_files]

    # Organ-GT exclusion check: the demo ships ground-truth ORGAN labels
    # (no vertebrae). A vertebra voxel inside any GT organ is an objective,
    # GT-certified false positive. Requires matching grids.
    if organ_gt_root is not None:
        gt_path = organ_gt_root / case.name / "combined_labels.nii.gz"
        if gt_path.exists():
            gt_img = nib.load(str(gt_path))
            organs = np.asarray(gt_img.dataobj) > 0
            vox_cm3 = voxel_volume_mm3(gt_img.affine) / 1000.0
            for f, r in zip(vert_files, results):
                if r["voxels"] == 0:
                    continue
                mask = np.asarray(nib.load(str(f)).dataobj) > 0
                if mask.shape != organs.shape:
                    r["flags"].append("GRID_MISMATCH_VS_GT")
                    continue
                ov = int((mask & organs).sum())
                r["organ_overlap_cm3"] = round(ov * vox_cm3, 2)
                if r["organ_overlap_cm3"] > 0.5:
                    r["flags"].append(f"ORGAN_OVERLAP({r['organ_overlap_cm3']})")
        else:
            print(f"[warn] no organ GT at {gt_path}", file=sys.stderr)
    centroids = {r["name"]: r["si_world"] for r in results if "si_world" in r}
    ordering_violations = flag_ordering(centroids)
    for r in results:
        if r["name"] in ordering_violations:
            r["flags"].append("ORDER")

    # Size-smoothness along the column: a vertebra dramatically smaller than
    # BOTH anatomical neighbors is a misassignment suspect (mass absorbed by a
    # neighboring level), not ordinary speckle. Ratio reported for all interior
    # vertebrae; flagged only under a conservative threshold.
    present = [r for r in results if r["voxels"] > 0]
    for i in range(1, len(present) - 1):
        prev_v, curr, next_v = present[i - 1], present[i], present[i + 1]
        smaller_neighbor = min(prev_v["voxels"], next_v["voxels"])
        if smaller_neighbor > 0:
            ratio = round(curr["voxels"] / smaller_neighbor, 2)
            curr["size_ratio_vs_neighbors"] = ratio
            if ratio < 0.6:
                curr["flags"].append(f"SIZE({ratio})")
    return {"case": case.name, "masks": results,
            "n_masks": len(results),
            "n_fragmented": sum(1 for r in results if any(f.startswith("FRAG") for f in r["flags"])),
            "n_empty": sum(1 for r in results if "EMPTY" in r["flags"]),
            "n_size_anomalies": sum(1 for r in results if any(f.startswith("SIZE") for f in r["flags"])),
            "n_organ_overlaps": sum(1 for r in results if any(f.startswith("ORGAN") for f in r["flags"])),
            "n_order_violations": len(ordering_violations)}


def print_case(summary: dict) -> None:
    print(f"\n=== {summary['case']}: {summary['n_masks']} vertebra masks | "
          f"{summary['n_fragmented']} fragmented, {summary['n_empty']} empty, "
          f"{summary['n_size_anomalies']} size anomalies, "
          f"{summary['n_order_violations']} ordering violations ===")
    for r in summary["masks"]:
        flags = " ".join(r["flags"])
        marker = "  <-- " + flags if flags else ""
        ratio = r.get("size_ratio_vs_neighbors")
        ratio_s = f" ratio={ratio:4.2f}" if ratio is not None else "           "
        print(f"  {r['name']:22s} vol={r['volume_cm3']:7.1f}cm3 "
              f"components={r['components']:2d} "
              f"largest={r['largest_fraction']:.3f}{ratio_s}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit vertebra predictions.")
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--case", action="append", default=None,
                        help="Case id(s) only (repeatable or comma-separated). Default: all.")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional JSON output path for before/after comparison.")
    parser.add_argument("--organ_gt", type=Path, default=None,
                        help="Root of shipped ORGAN ground truth (e.g. data/AbdomenAtlasDemo); "
                             "enables the vertebra-inside-organ false-positive check.")
    args = parser.parse_args()

    cases = sorted(p for p in args.pred_dir.iterdir() if p.is_dir())
    if args.case:
        want = {p.strip() for v in args.case for p in v.split(",") if p.strip()}
        cases = [p for p in cases if p.name in want]
    if not cases:
        print(f"No case folders under {args.pred_dir}", file=sys.stderr)
        sys.exit(1)

    summaries = []
    for case in cases:
        summary = audit_case(case, organ_gt_root=args.organ_gt)
        summaries.append(summary)
        print_case(summary)

    total_frag = sum(s["n_fragmented"] for s in summaries)
    total_empty = sum(s["n_empty"] for s in summaries)
    total_order = sum(s["n_order_violations"] for s in summaries)
    print(f"\nTOTAL: {total_frag} fragmented, {total_empty} empty, "
          f"{total_order} ordering violations across {len(summaries)} case(s)")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        for s in summaries:  # strip non-serializable float wrappers
            for r in s["masks"]:
                r.pop("si_world", None)
        args.report.write_text(json.dumps(summaries, indent=2))
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()