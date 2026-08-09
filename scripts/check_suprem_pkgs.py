#!/usr/bin/env python
"""Report whether the active env has the warm-up pip pins.

Prints a human-readable table to stdout.
Writes space-separated install-group tokens to --groups-file (if given):
  torch | monai | suprem_reqs | extras | numpy

Exit code: 0 if all intended pins OK, 1 if anything is missing/wrong.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys


def ver(dist: str):
    try:
        import importlib.metadata as md

        return md.version(dist)
    except Exception:
        return None


def has_mod(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--groups-file", help="Write needed install groups here")
    args = p.parse_args()

    rows = []  # (name, have, want, ok)
    need = []

    tv, vv, av = ver("torch"), ver("torchvision"), ver("torchaudio")
    torch_ok = (
        tv == "1.11.0+cu113"
        and vv is not None
        and vv.startswith("0.12.0")
        and av is not None
        and av.startswith("0.11.0")
    )
    rows.append(("torch", tv or "-", "1.11.0+cu113", torch_ok))
    rows.append(("torchvision", vv or "-", "0.12.0+cu113", bool(vv and vv.startswith("0.12.0"))))
    rows.append(("torchaudio", av or "-", "0.11.0+cu113", bool(av and av.startswith("0.11.0"))))
    if not torch_ok:
        need.append("torch")

    mv = ver("monai")
    monai_ok = mv == "0.9.0"
    rows.append(("monai", mv or "-", "0.9.0", monai_ok))
    if not monai_ok:
        need.append("monai")

    suprem_dists = [
        ("h5py", "h5py"),
        ("tqdm", "tqdm"),
        ("fastremap", "fastremap"),
        ("SimpleITK", "SimpleITK"),
        ("einops", "einops"),
        ("timm", "timm"),
        ("ml-collections", "ml_collections"),
        ("pytorch-lightning", "pytorch_lightning"),
        ("opencv-python", "cv2"),
        ("pandas", "pandas"),
        ("glob2", "glob2"),
        ("elasticdeform", "elasticdeform"),
    ]
    missing_suprem = []
    for dist, mod in suprem_dists:
        ok = ver(dist) is not None or has_mod(mod)
        have = ver(dist) or ("importable" if has_mod(mod) else "-")
        rows.append((dist, have, "installed", ok))
        if not ok:
            missing_suprem.append(dist)
    # pytorch-lightning is in SuPreM requirements.txt but unused by direct_inference/.
    # Soft requirement: show status, do not fail the env if missing.
    pl = ver("pytorch-lightning")
    rows.append(("pytorch-lightning", pl or "-", "1.6.4 (optional)", True))
    if missing_suprem:
        need.append("suprem_reqs")

    for dist, mod in [
        ("nibabel", "nibabel"),
        ("connected-components-3d", "cc3d"),
        ("scipy", "scipy"),
    ]:
        ok = ver(dist) is not None or has_mod(mod)
        have = ver(dist) or ("importable" if has_mod(mod) else "-")
        rows.append((dist, have, "installed", ok))
        if not ok and "extras" not in need:
            need.append("extras")

    nv = ver("numpy")
    numpy_ok = False
    if nv is not None:
        try:
            major, minor = (int(x) for x in nv.split(".")[:2])
            numpy_ok = (major, minor) < (1, 24)
        except ValueError:
            numpy_ok = False
    rows.append(("numpy", nv or "-", "<1.24", numpy_ok))
    if not numpy_ok:
        need.append("numpy")

    # stable unique order
    order = ["torch", "monai", "suprem_reqs", "extras", "numpy"]
    groups = [g for g in order if g in need]

    print("[setup] package check (active interpreter):")
    print(f"  {'package':28} {'have':18} {'want':14} status")
    for name, have, want, ok in rows:
        status = "OK" if ok else "MISSING/WRONG"
        print(f"  {name:28} {str(have):18} {want:14} {status}")

    if groups:
        print(f"[setup] install groups needed: {' '.join(groups)}")
    else:
        print("[setup] all intended pip packages already satisfied")

    if args.groups_file:
        with open(args.groups_file, "w", encoding="utf-8") as f:
            f.write(" ".join(groups))

    return 1 if groups else 0


if __name__ == "__main__":
    sys.exit(main())
