"""Post-process predicted vertebra segmentations.

Cleans the common failure modes in automatic vertebra segmentation
(e.g. TotalSegmentator output on the AbdomenAtlasDemo scans):

  1. off-spine speckle          -> keep the largest connected component
  2. small floating fragments   -> drop components below a voxel threshold
  3. 1-2 voxel holes            -> optional light morphological closing
  4. wrong superior-inferior    -> flag vertebrae whose centroid breaks the
     level ordering                expected head-to-pelvis ordering (report
                                   only; relabeling stays manual by design)

Input layouts supported (auto-detected per case):
  A) per-mask   : <case>/segmentations/vertebrae_*.nii.gz  (binary masks)
  B) combined   : <case>/<seg_name>                        (one multi-label map)

Reuses cc3d (fast 3D connected components), nibabel (NIfTI IO), and
scipy.ndimage. No plotting, no network, deterministic.

Usage:
  python postprocessing_vertebrae.py --pred_dir AbdomenAtlasDemoPredict \
      --out_dir AbdomenAtlasDemoPredict_refined
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("postprocessing_vertebrae")

# Region order along the spine, superior -> inferior. Used only to sort the
# vertebrae so the ordering check walks them head-to-pelvis.
_REGION_RANK = {"C": 0, "T": 1, "L": 2, "S": 3}
_VERT_RE = re.compile(r"vertebrae?_([CTLS])(\d+)", re.IGNORECASE)


def _vertebra_sort_key(name: str) -> tuple:
    """Sort key mapping e.g. 'vertebrae_L3' -> (2, 3). Unknown names sort last."""
    m = _VERT_RE.search(name)
    if not m:
        return (99, 0)
    return (_REGION_RANK.get(m.group(1).upper(), 98), int(m.group(2)))


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 26-connected component of a binary mask."""
    if not mask.any():
        return mask
    labeled = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # background
    return labeled == int(counts.argmax())


def remove_small_fragments(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Drop connected components smaller than min_voxels."""
    labeled = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    keep = np.zeros(mask.shape, dtype=bool)
    for cid, count in enumerate(np.bincount(labeled.ravel())):
        if cid != 0 and count >= min_voxels:
            keep |= labeled == cid
    return keep


def clean_binary_mask(mask: np.ndarray, min_voxels: int, closing_iters: int) -> np.ndarray:
    """Full single-vertebra cleanup: largest component, fragment removal, closing."""
    mask = keep_largest_component(mask)
    mask = remove_small_fragments(mask, min_voxels)
    if closing_iters > 0 and mask.any():
        mask = ndimage.binary_closing(mask, iterations=closing_iters)
    return mask


def _world_si_coord(centroid_vox: tuple, affine: np.ndarray) -> float:
    """Map a voxel centroid to the world superior-inferior (RAS z) coordinate."""
    v = np.array([centroid_vox[0], centroid_vox[1], centroid_vox[2], 1.0])
    return float((affine @ v)[2])


def flag_ordering(centroids_world_si: dict) -> list:
    """Return vertebra names whose S-I position breaks monotonic spine ordering.

    Vertebrae are sorted head-to-pelvis by name; their world S-I coordinate
    should then be monotonic (sign depends on orientation, so we accept either
    consistently increasing or decreasing and flag the local violations).
    """
    ordered = sorted(centroids_world_si.items(), key=lambda kv: _vertebra_sort_key(kv[0]))
    if len(ordered) < 3:
        return []
    names = [n for n, _ in ordered]
    si = [z for _, z in ordered]
    decreasing = si[-1] < si[0]  # infer the expected direction from the endpoints
    violations = []
    for i in range(1, len(si)):
        step = si[i] - si[i - 1]
        if (decreasing and step > 0) or (not decreasing and step < 0):
            violations.append(names[i])
    return violations


def process_case_permask(seg_dir: Path, out_dir: Path, min_voxels: int, closing_iters: int) -> None:
    """Clean every vertebrae_*.nii.gz mask in a TotalSegmentator-style folder."""
    vert_files = sorted(
        (p for p in seg_dir.glob("*.nii.gz") if p.name.lower().startswith("vertebrae")),
        key=lambda p: _vertebra_sort_key(p.name),
    )
    if not vert_files:
        logger.warning("No vertebrae_*.nii.gz in %s", seg_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    centroids = {}
    for f in vert_files:
        img = nib.load(str(f))
        mask = np.asarray(img.dataobj) > 0
        if not mask.any():
            logger.info("%s empty, skipped", f.name)
            continue
        cleaned = clean_binary_mask(mask, min_voxels, closing_iters)
        out_img = nib.Nifti1Image(cleaned.astype(np.uint8), img.affine, img.header)
        nib.save(out_img, str(out_dir / f.name))
        if cleaned.any():
            centroids[f.stem.replace(".nii", "")] = _world_si_coord(
                ndimage.center_of_mass(cleaned), img.affine
            )

    for name in flag_ordering(centroids):
        logger.warning("%s breaks superior-inferior ordering (review manually)", name)


def process_case_combined(seg_path: Path, out_path: Path, labels, min_voxels: int, closing_iters: int) -> None:
    """Clean a set of vertebra labels inside one combined multi-label map."""
    img = nib.load(str(seg_path))
    seg = np.asarray(img.dataobj)
    out = seg.copy()
    for label in labels:
        mask = seg == label
        if not mask.any():
            continue
        out[mask] = 0  # clear, then write back only the cleaned voxels
        out[clean_binary_mask(mask, min_voxels, closing_iters)] = label
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out.astype(seg.dtype), img.affine, img.header), str(out_path))


def process_case(case: Path, out_root: Path, args) -> None:
    seg_dir = case / "segmentations"
    if seg_dir.is_dir():
        process_case_permask(seg_dir, out_root / case.name / "segmentations",
                             args.min_voxels, args.closing_iters)
        return
    combined = case / args.seg_name
    if combined.exists():
        labels = [int(x) for x in args.labels.split(",")] if args.labels else list(range(1, 26))
        process_case_combined(combined, out_root / case.name / args.seg_name, labels,
                              args.min_voxels, args.closing_iters)
        return
    logger.warning("Skipping %s: no segmentations/ folder and no %s", case.name, args.seg_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process vertebra segmentations.")
    parser.add_argument("--pred_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--seg_name", default="combined_labels.nii.gz",
                        help="Combined multi-label filename (combined-map layout only).")
    parser.add_argument("--labels", default="",
                        help="Comma-separated vertebra label ids for the combined map.")
    parser.add_argument("--min_voxels", type=int, default=200,
                        help="Drop connected components smaller than this.")
    parser.add_argument("--closing_iters", type=int, default=1,
                        help="Binary-closing iterations (0 disables).")
    args = parser.parse_args()

    cases = sorted(p for p in args.pred_dir.iterdir() if p.is_dir())
    if not cases:
        logger.error("No case folders under %s", args.pred_dir)
        return
    logger.info("Processing %d case(s)", len(cases))
    for case in cases:
        process_case(case, args.out_dir, args)
    logger.info("Done. Refined predictions in %s", args.out_dir)


if __name__ == "__main__":
    main()
