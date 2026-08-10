"""Phantom unit test for the pedicle-root arch rebuild (arch_repartition).

Builds a 3-level synthetic column at 1 mm isotropic with the topology that
matters: bodies, pedicles, laminae, spinous processes, and inferior/superior
articular processes joined ACROSS levels by thin bone-continuous facet
bridges (the structure that breaks naive floods). Phase-1 is simulated with
correct bodies and a fully scrambled arch (everything label 1; the lowest
spinous left unlabeled to test pool reclaim). Ground truth is known by
construction, so arch accuracy is exact.

Run:  python tests/test_arch_phantom.py
Outputs: reports/debug/phantom/arch_phantom_<mode>.png + a pass/fail table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import postprocessing_vertebrae as pv

ZOOMS = (1.0, 1.0, 1.0)
SHAPE = (100, 120, 120)
CX, CY = 50, 45
ZC = [26, 60, 94]                # level centers, z increases caudally
BODY_R = 16.0
ZONE_R = 18.0                    # matches seed_radius_mm + 4 frontier
FACET_X = [(36, 42), (58, 64)]   # left / right facet-column x ranges


def _grid():
    x, y = np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), indexing="ij")
    return x, y, (x - CX) ** 2 + (y - CY) ** 2


def build_phantom():
    """Returns (vert_mask, gt_labels, body_zone) - gt by construction."""
    x, y, d2 = _grid()
    V = np.zeros(SHAPE, dtype=bool)
    gt = np.zeros(SHAPE, dtype=np.int32)

    def put(mask2d, z0, z1, lid):
        for z in range(z0, z1):
            V[:, :, z] |= mask2d
            g = gt[:, :, z]
            g[mask2d & (g == 0)] = lid

    for i, zc in enumerate(ZC, start=1):
        body = d2 <= BODY_R ** 2
        put(body, zc - 14, zc + 14, i)
        ped = np.zeros(SHAPE[:2], dtype=bool)
        for xa, xb in FACET_X:
            ped |= (x >= xa) & (x < xb) & (y >= 56) & (y < 68)
        put(ped, zc - 6, zc + 6, i)
        lam = (x >= 38) & (x < 62) & (y >= 68) & (y < 74)
        put(lam, zc - 7, zc + 7, i)
        spi = (x >= 47) & (x < 53) & (y >= 74) & (y < 86)
        put(spi, zc - 4, zc + 4, i)

    proc2d = np.zeros(SHAPE[:2], dtype=bool)
    for xa, xb in FACET_X:
        proc2d |= (x >= xa) & (x < xb) & (y >= 63) & (y < 69)
    bridge2d = np.zeros(SHAPE[:2], dtype=bool)
    for xa, _ in FACET_X:
        bridge2d |= (x >= xa + 2) & (x < xa + 4) & (y >= 65) & (y < 67)
    for i in (1, 2):                       # interface i | i+1
        zc = ZC[i - 1]
        # processes attach to their own lamina / pedicle through a THICK
        # multi-voxel junction (real pars is 8-15 mm); only the facet joint
        # itself is a thin bridge. The inferior process hangs LONG
        # (lumbar-like): its tip is ~25 mm from its own pedicle root but
        # ~10 mm from the neighbor's rooted superior process - the
        # asymmetry that breaks plain distance races.
        put(proc2d, zc + 3, zc + 22, i)          # long inferior process of i
        put(bridge2d, zc + 22, zc + 25, i)       # thin facet bridge (gt: i, cosmetic)
        put(proc2d, zc + 25, ZC[i] - 2, i + 1)   # superior process of i+1

    body_zone = np.zeros(SHAPE, dtype=bool)
    body_zone[d2 <= ZONE_R ** 2] = True
    return V, gt, body_zone & V


def scrambled_phase1(V, gt, body_zone):
    """Correct bodies (z-segment cuts), arch all label 1, lowest spinous 0."""
    z = np.arange(SHAPE[2])[None, None, :]
    seg_z = np.where(z < 43, 1, np.where(z < 77, 2, 3))
    lab = np.where(V, np.broadcast_to(seg_z, SHAPE), 0).astype(np.int32)
    A = V & ~body_zone
    lab[A] = 1                                   # the "arch river" failure
    x, y = np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), indexing="ij")
    spi3 = np.zeros(SHAPE, dtype=bool)
    for z3 in range(ZC[2] - 4, ZC[2] + 4):
        spi3[:, :, z3] = (x >= 47) & (x < 53) & (y >= 74)
    lab[spi3 & A] = 0                            # unreached pool
    return lab


def region_masks(V, gt, body_zone):
    """Named arch regions for scoring."""
    A = V & ~body_zone
    regs = {}
    for i, zc in enumerate(ZC, start=1):
        m = np.zeros(SHAPE, dtype=bool)
        m[:, :, zc - 7:zc + 7] = True
        regs[f"lamina+spinous L{i}"] = A & (gt == i) & m
    for i in (1, 2):
        zc = ZC[i - 1]
        m = np.zeros(SHAPE, dtype=bool)
        m[:, :, zc + 7:zc + 22] = True
        regs[f"inf process L{i}"] = A & (gt == i) & m
        m2 = np.zeros(SHAPE, dtype=bool)
        m2[:, :, zc + 25:ZC[i] - 2] = True
        regs[f"sup process L{i + 1}"] = A & (gt == i + 1) & m2
    return regs


def main():
    V, gt, body_zone = build_phantom()
    A = V & ~body_zone
    lab0 = scrambled_phase1(V, gt, body_zone)
    vox_mm3 = 1.0
    regs = region_masks(V, gt, body_zone)
    results = {}
    failures = []
    for mode in ("hier", "core", "edt", "uniform"):
        pv.P["arch_cost"] = mode
        out, rec = pv.arch_repartition(lab0.copy(), body_zone, A, [1, 2, 3],
                                       ZOOMS, vox_mm3)
        results[mode] = (out, rec)
        acc = float((out[A] == gt[A]).mean())
        print(f"\n[{mode}] {rec['arch_mode']}  roots={rec.get('arch_roots_cm3')}"
              f"  contested={rec['arch_contested_mm3']}mm3"
              f"  changed={rec.get('arch_phase2_changed_mm3')}mm3"
              f"  arch-accuracy={acc:.4f}")
        if not (out[body_zone] == lab0[body_zone]).all():
            failures.append(f"{mode}: bodies edited")
        for name, m in regs.items():
            lid = int(name.split("L")[-1])
            frac = float((out[m] == lid).mean()) if m.any() else float("nan")
            tag = "ok " if frac >= 0.98 else "LOW"
            print(f"    {tag} {name:22s} correct={frac:.3f}  ({int(m.sum())} vox)")
            if mode in ("hier", "core") and frac < 0.98:
                failures.append(f"{mode}: {name} correct={frac:.3f}")
        if mode == "core":
            r = rec.get("arch_roots_cm3", {})
            if len(r) != 3:
                failures.append(f"core: expected roots for 3 levels, got {r}")
            if rec.get("arch_rootless"):
                failures.append(f"core: rootless levels {rec['arch_rootless']}")
            spi3 = regs["lamina+spinous L3"] & (lab0 == 0)
            if spi3.any() and float((out[spi3] == 3).mean()) < 0.95:
                failures.append("core: unlabeled L3 spinous not reclaimed")

    render(V, gt, body_zone, lab0, results)
    print()
    if failures:
        print("FAIL")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("PASS: pedicle-root arch rebuild correct on phantom (hier+core); "
          "edt/uniform modes reported above as ablations.")


def render(V, gt, body_zone, lab0, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#101010", "#d1495b", "#66a182", "#2e6f95"])
    outdir = Path(__file__).resolve().parents[1] / "reports" / "debug" / "phantom"
    outdir.mkdir(parents=True, exist_ok=True)
    views = [("sagittal x=39 (facet column)", lambda a: a[39, :, :].T),
             ("sagittal x=50 (midline)", lambda a: a[50, :, :].T),
             ("coronal y=66 (facet plane)", lambda a: a[:, 66, :].T)]
    panels = [("phase-1 scrambled", lab0), ("ground truth", gt),
              ("rebuilt (core)", results["core"][0]),
              ("rebuilt (edt)", results["edt"][0]),
              ("rebuilt (uniform)", results["uniform"][0])]
    fig, axes = plt.subplots(len(views), len(panels),
                             figsize=(4.2 * len(panels), 3.9 * len(views)))
    for r, (vname, take) in enumerate(views):
        for c, (pname, vol) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(take(vol), cmap=cmap, vmin=0, vmax=3, origin="lower",
                      interpolation="nearest")
            zone = take(body_zone.astype(np.uint8))
            ax.contour(zone, levels=[0.5], colors="w", linewidths=0.6, alpha=0.6)
            if r == 0:
                ax.set_title(pname, fontsize=11)
            if c == 0:
                ax.set_ylabel(vname, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Arch rebuild from pedicle roots - 3-level phantom "
                 "(white contour = body/arch frontier)", fontsize=12)
    fig.tight_layout()
    p = outdir / "arch_phantom.png"
    fig.savefig(p, dpi=110)
    print(f"render: {p}")


if __name__ == "__main__":
    main()
