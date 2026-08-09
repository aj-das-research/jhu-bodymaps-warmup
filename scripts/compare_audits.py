#!/usr/bin/env python
"""Diff two audit_predictions.py JSON reports (before vs after post-process).

Highlights Phase-B questions:
  - Did FRAGMENTED / EMPTY / ORDER counts drop?
  - Did SIZE flags and volume_cm3 at level-boundary levels survive?

Usage:
    python scripts/compare_audits.py \\
        --before reports/audit_before.json \\
        --after  reports/audit_shapekit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _flag_kind(flag: str) -> str:
    if flag.startswith("FRAGMENTED"):
        return "FRAGMENTED"
    if flag.startswith("SIZE"):
        return "SIZE"
    return flag


def _index(report: list) -> dict:
    out = {}
    for case in report:
        masks = {m["name"]: m for m in case.get("masks", [])}
        out[case["case"]] = masks
    return out


def _count_flags(report: list) -> Counter:
    c = Counter()
    for case in report:
        for m in case.get("masks", []):
            for f in m.get("flags", []):
                c[_flag_kind(f)] += 1
    return c


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument(
        "--watch",
        nargs="*",
        default=[
            "vertebrae_T8",
            "vertebrae_T9",
            "vertebrae_T10",
            "vertebrae_T11",
            "vertebrae_T12",
            "vertebrae_L1",
            "vertebrae_L2",
            "vertebrae_C2",
        ],
        help="Vertebra names to print volume / flags side-by-side",
    )
    args = p.parse_args()

    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    b_idx, a_idx = _index(before), _index(after)
    b_flags, a_flags = _count_flags(before), _count_flags(after)

    kinds = sorted(set(b_flags) | set(a_flags))
    print("=== flag totals ===")
    print(f"{'flag':12s}  {'before':>8s}  {'after':>8s}  {'delta':>8s}")
    for k in kinds:
        b, a = b_flags.get(k, 0), a_flags.get(k, 0)
        print(f"{k:12s}  {b:8d}  {a:8d}  {a - b:+8d}")

    print("\n=== watched levels (volume_cm3 / largest_fraction / flags) ===")
    for case in sorted(set(b_idx) | set(a_idx)):
        print(f"\n--- {case} ---")
        for name in args.watch:
            bm = b_idx.get(case, {}).get(name)
            am = a_idx.get(case, {}).get(name)
            if bm is None and am is None:
                continue

            def fmt(m):
                if m is None:
                    return "MISSING"
                flags = ",".join(m.get("flags") or ["-"]) or "-"
                return (f"{m.get('volume_cm3', '?'):>6} cm3  "
                        f"frac={m.get('largest_fraction', '?'):.4f}  "
                        f"cc={m.get('components', '?')}  [{flags}]")

            print(f"  {name}")
            print(f"    before: {fmt(bm)}")
            print(f"    after:  {fmt(am)}")
            if bm and am and bm.get("volume_cm3") != am.get("volume_cm3"):
                dv = am["volume_cm3"] - bm["volume_cm3"]
                print(f"    delta_cm3: {dv:+.1f}")

    print("\n=== Phase-B reading ===")
    frag_b, frag_a = b_flags.get("FRAGMENTED", 0), a_flags.get("FRAGMENTED", 0)
    size_b, size_a = b_flags.get("SIZE", 0), a_flags.get("SIZE", 0)
    print(f"  FRAGMENTED: {frag_b} -> {frag_a}  "
          f"({'crushed' if frag_a < frag_b else 'unchanged/up'})")
    print(f"  SIZE:       {size_b} -> {size_a}  "
          f"({'survived' if size_a >= size_b and size_b > 0 else 'changed'})")
    print("  If FRAGMENTED drops but SIZE / T9–L1 volume dip survive → Tier-2 "
          "reassignment (not delete-only) is justified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
