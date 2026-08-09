"""Evidence-edited, anatomy-audited post-processing for vertebra segmentations.

Refines SuPreM / TotalSegmentator-style vertebra predictions (C1..L5) on the
AbdomenAtlasDemo scans. Design principle: fix MODEL errors, never "fix"
PATIENT reality - the CT image is the only editor; anatomical priors only
flag suspects and freeze what is already consistent.

Pipeline (all actions QA-logged per case):

  Stage 1  evidence triage of disconnected components, per vertebra:
             speck (<30 mm3)            -> removed
             soft tissue (bone frac<=.5)-> removed (hallucination)
             bone near main (<=5 mm)    -> re-attached through a CT-bone
                                           corridor at the geometric neck of
                                           the gap (unclaimed voxels only),
                                           else kept + POSSIBLE_FRAGMENT
             bone far from main         -> removed to the unclaimed pool,
                                           logged REASSIGN_CANDIDATE
  Stage 2a island guard: a component enclosed >=80% by one other label and
           small relative to it is reassigned to that label (vertebrae do
           not interpenetrate).
  Stage 2b disc-aware boundary re-arbitration of SUSPECT bands only:
           - suspicion: audit flags (FRAGMENTED / SIZE / ORDER), inflated or
             starved size vs neighbors, robust log-volume trend residual,
             or >=20% of the raw label lost to the pool; contiguous closure
             forms the band, everything else is frozen.
           - the CT's own density segments the column: mean HU sampled on
             disks PERPENDICULAR to the body centerline dips at every
             intervertebral disc. A spacing-regularized DP picks exactly the
             expected number of minima between the frozen anchor levels
             (vertebral heights vary smoothly - a periodicity prior), so the
             image, not the labels, decides where bodies begin and end.
           - each level seeds inside its disc-bounded segment; a uniform-
             speed geodesic competition (synchronized layer growth through
             the thresholded bone domain) floods the band; fronts meet at
             low-HU clefts, which is where anatomy separates. Fragments and
             pooled mass are reclaimed by whichever level reaches them
             through bone first.
           - cut surfaces tilt with the column: segment membership uses the
             first-order arc-length projection onto the centerline tangent,
             clipped to +/-6 mm so it stays monotone off-axis.
           - arch phase: posterior elements are repartitioned by erosion
             splitting - eroding the arch domain ~2.2 mm disconnects it at
             the thin facet waists while laminae stay attached to their own
             pedicle; each eroded component joins the level whose body it
             touches, then the shell is reconstructed by uniform BFS.
           - guards: C1/C2 skipped (no disc plane at the atlas/axis), band
             skipped when the expected disc count cannot be resolved, and
             the whole band reverts unless the audit strictly improves.
  Stage 3  interface regularization (iterated 26-neighborhood majority vote
           on label-label interfaces), orphan-component absorption, enclosed-
           hole + directional pit filling, and bounded volume-preserving smoothing
           (Gaussian-smoothed SDT, volume-preserving iso-level by bisection,
           changes confined to +/-1.5 mm, additions bone-gated; a label
           reverts if Dice vs its pre-smooth mask drops below 0.97 or its
           component count increases).
  Stage 4  audit of the result (fragmentation, size-smoothness, ordering)
           written to the QA report; flags never edit voxels.

Verified on the two AbdomenAtlasDemo cases (identical parameters):
  BDMAP_00000006 (clean): 10 fragmented + 1 ORDER -> all clean; C1 caudal
    articular tips on CT-certified bone are preserved (the delete-only
    baseline removes them).
  BDMAP_00000031 (sick):  21 fragmented, mass misassignment T8..L2 with
    L1 at 23.3 cm3 vs ~60 cm3 neighbors -> all clean; L1 restored to
    ~65 cm3 by re-arbitration at the detected disc planes; the ShapeKit
    baseline instead deepens the error (L1 11.6 cm3, SIZE 0.20).

Dependencies: numpy, scipy, nibabel, scikit-image, cc3d (connected-components-3d).

Usage:
  python postprocessing_vertebrae.py \
      --pred_dir AbdomenAtlasDemoPredict \
      --ct_root  data/AbdomenAtlasDemo \
      --out_dir  AbdomenAtlasDemoPredict_refined \
      [--report_dir reports] [--case BDMAP_00000031]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage, signal
from skimage.segmentation import watershed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("postprocessing_vertebrae")

# SuPreM combined_labels ids: 1=L5 ... 24=C1 (bottom-up)
NAMES_BOTTOM_UP = (["L5", "L4", "L3", "L2", "L1"]
                   + [f"T{i}" for i in range(12, 0, -1)]
                   + [f"C{i}" for i in range(7, 0, -1)])
ID_TO_NAME = {i + 1: n for i, n in enumerate(NAMES_BOTTOM_UP)}
NAME_TO_ID = {n: i for i, n in ID_TO_NAME.items()}
TOP_DOWN = list(reversed(NAMES_BOTTOM_UP))
STRUCT6 = ndimage.generate_binary_structure(3, 1)
STRUCT26 = np.ones((3, 3, 3), dtype=bool)

P = {
    # evidence gates
    "bone_hu": 150.0, "speck_mm3": 30.0, "soft_bone_fraction": 0.50,
    "near_mm": 5.0, "corridor_pad_mm": 6.0, "bridge_cap_mm3": 500.0,
    # islands
    "enclosure": 0.80, "max_rel_to_host": 0.5, "min_labeled_shell": 0.5,
    # suspicion / bands
    "suspect_size_lo": 0.60, "suspect_size_hi": 1.80, "suspect_logres": 0.30,
    "suspect_pool_frac": 0.20, "band_gap_close": 3,
    # disc profile / DP
    "profile_sigma_mm": 2.5, "profile_radius_mm": 10.0, "min_body_height_mm": 12.0,
    "max_body_height_mm": 45.0, "dp_beta": 2.0, "pin_lo_mm": 8.0, "pin_hi_mm": 35.0,
    # arbitration domain
    "seed_margin_mm": 3.0, "seed_radius_mm": 14.0, "halo_mm": 2.5, "band_pad_mm": 4.0,
    "arch_erode_mm": 2.2,
    # smoothing
    "sigma_mm": 1.2, "max_dev_mm": 1.5, "vol_tol": 0.01, "min_dice": 0.97,
    "smooth_pad_mm": 5.0,
    # io
    "crop_margin_mm": 25.0,
}


# ------------------------------------------------------------------ utils --
def _bbox_pad(sl, pad_vox, shape):
    return tuple(slice(max(s.start - int(p), 0), min(s.stop + int(p), n))
                 for s, p, n in zip(sl, pad_vox, shape))


def _components(mask):
    cc = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    counts = np.bincount(cc.ravel())
    counts[0] = 0
    return cc, counts


def audit(seg, affine, vox_mm3):
    """Repo-convention audit: FRAGMENTED / SIZE(<0.6) / ORDER / EMPTY flags."""
    rows = []
    objs = ndimage.find_objects(seg)
    for name in TOP_DOWN:
        lid = NAME_TO_ID[name]
        sl = objs[lid - 1] if lid - 1 < len(objs) else None
        if sl is None:
            rows.append({"name": f"vertebrae_{name}", "voxels": 0, "volume_cm3": 0.0,
                         "components": 0, "largest_fraction": 0.0, "flags": ["EMPTY"]})
            continue
        mask = seg[sl] == lid
        vox = int(mask.sum())
        _, counts = _components(mask)
        com = ndimage.center_of_mass(mask)
        w = affine @ np.array([com[0] + sl[0].start, com[1] + sl[1].start,
                               com[2] + sl[2].start, 1.0])
        ncomp = int((counts > 0).sum())
        rows.append({"name": f"vertebrae_{name}", "voxels": vox,
                     "volume_cm3": round(vox * vox_mm3 / 1000.0, 1),
                     "components": ncomp,
                     "largest_fraction": round(float(counts.max()) / vox, 4),
                     "si_world": float(w[2]),
                     "flags": [f"FRAGMENTED({ncomp})"] if ncomp > 1 else []})
    si = [(r["name"], r["si_world"]) for r in rows if "si_world" in r]
    if len(si) >= 3:
        dec = si[-1][1] < si[0][1]
        for i in range(1, len(si)):
            step = si[i][1] - si[i - 1][1]
            if (dec and step > 0) or (not dec and step < 0):
                next(r for r in rows if r["name"] == si[i][0])["flags"].append("ORDER")
    present = [r for r in rows if r["voxels"] > 0]
    for i in range(1, len(present) - 1):
        smaller = min(present[i - 1]["voxels"], present[i + 1]["voxels"])
        if smaller > 0:
            ratio = round(present[i]["voxels"] / smaller, 2)
            present[i]["size_ratio_vs_neighbors"] = ratio
            if ratio < 0.6:
                present[i]["flags"].append(f"SIZE({ratio})")
    summary = {k: sum(1 for r in rows if any(f.startswith(pre) for f in r["flags"]))
               for k, pre in [("n_fragmented", "FRAG"), ("n_empty", "EMPTY"),
                              ("n_size", "SIZE"), ("n_order", "ORDER")]}
    return rows, summary


# ---------------------------------------------------------------- stage 1 --
def stage1_triage(seg, ct, zooms, vox_mm3, records):
    out = seg.copy()
    bone = ct >= P["bone_hu"]
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        cc, counts = _components(out == lid)
        if (counts > 0).sum() <= 1:
            continue
        main_id = int(counts.argmax())
        main = cc == main_id
        union_sl = ndimage.find_objects((cc > 0).astype(np.uint8))[0]
        pad_vox = np.ceil(P["near_mm"] / np.asarray(zooms)) + 2
        union_sl = _bbox_pad(union_sl, pad_vox, seg.shape)
        dist_main = ndimage.distance_transform_edt(~main[union_sl], sampling=zooms)
        for comp_id in np.nonzero(counts)[0]:
            if comp_id == main_id:
                continue
            comp = cc == comp_id
            vol = float(counts[comp_id]) * vox_mm3
            hu = ct[comp]
            rec = {"stage": 1, "label": ID_TO_NAME[lid], "vol_mm3": round(vol, 1),
                   "hu_mean": round(float(hu.mean()), 1),
                   "bone_fraction": round(float((hu >= P["bone_hu"]).mean()), 3)}
            if vol < P["speck_mm3"]:
                out[comp] = 0
                rec["action"] = "REMOVED_SPECK"
            elif rec["bone_fraction"] <= P["soft_bone_fraction"]:
                out[comp] = 0
                rec["action"] = "REMOVED_SOFT"
            else:
                d = float(dist_main[comp[union_sl]].min())
                rec["dist_to_main_mm"] = round(d, 2)
                if d <= P["near_mm"]:
                    added = _bridge(out, bone, main, comp, lid, zooms)
                    if added is not None:
                        out[comp] = lid
                        out[added] = lid
                        rec["action"] = "BRIDGED"
                        rec["bridge_added_mm3"] = round(float(added.sum()) * vox_mm3, 1)
                    else:
                        rec["action"] = "KEPT_POSSIBLE_FRAGMENT"
                else:
                    out[comp] = 0
                    rec["action"] = "REMOVED_FAR_REASSIGN_CANDIDATE"
            records.append(rec)
    return out


def _bridge(seg, bone, main, comp, lid, zooms):
    nz_c = ndimage.find_objects(comp.astype(np.uint8))[0]
    nz_m = ndimage.find_objects(main.astype(np.uint8))[0]
    joint = tuple(slice(min(a.start, b.start), max(a.stop, b.stop))
                  for a, b in zip(nz_c, nz_m))
    pad_vox = np.ceil(P["corridor_pad_mm"] / np.asarray(zooms)) + 1
    joint = _bbox_pad(joint, pad_vox, seg.shape)
    seg_j, bone_j = seg[joint], bone[joint]
    comp_j, main_j = comp[joint], main[joint]
    d_comp = ndimage.distance_transform_edt(~comp_j, sampling=zooms)
    d_main = ndimage.distance_transform_edt(~main_j, sampling=zooms)
    neck = (d_comp <= P["near_mm"]) & (d_main <= P["near_mm"])
    domain = (bone_j & ((seg_j == 0) | (seg_j == lid)) & neck) | comp_j | main_j
    reach = ndimage.binary_propagation(main_j, mask=domain, structure=STRUCT26)
    if not (reach & comp_j).any():
        return None
    fill = reach & (seg_j == 0) & bone_j & neck
    if float(fill.sum()) * abs(np.prod(zooms)) > P["bridge_cap_mm3"]:
        return None
    corridor = np.zeros_like(comp)
    corridor[joint] = fill
    return corridor


# --------------------------------------------------------------- stage 2a --
def stage2a_islands(seg, vox_mm3, records):
    out = seg.copy()
    vol = {int(l): int((seg == l).sum()) for l in np.unique(seg) if l != 0}
    for lid in sorted(vol):
        cc, counts = _components(out == lid)
        if (counts > 0).sum() <= 1:
            continue
        main_id = int(counts.argmax())
        for comp_id in np.nonzero(counts)[0]:
            if comp_id == main_id:
                continue
            comp = cc == comp_id
            sl = _bbox_pad(ndimage.find_objects(comp.astype(np.uint8))[0],
                           np.array([2, 2, 2]), seg.shape)
            comp_l = comp[sl]
            shell = ndimage.binary_dilation(comp_l, structure=STRUCT26) & ~comp_l
            neigh = out[sl][shell]
            labeled = neigh[neigh != 0]
            if neigh.size == 0 or labeled.size / neigh.size < P["min_labeled_shell"]:
                continue
            ids, cnt = np.unique(labeled, return_counts=True)
            host = int(ids[cnt.argmax()])
            frac = float(cnt.max()) / labeled.size
            rec = {"stage": "2a", "label": ID_TO_NAME[lid], "host": ID_TO_NAME[host],
                   "vol_mm3": round(float(counts[comp_id]) * vox_mm3, 1),
                   "enclosure": round(frac, 3)}
            if (host != lid and frac >= P["enclosure"]
                    and counts[comp_id] <= P["max_rel_to_host"] * vol.get(host, 0)):
                out[comp] = host
                rec["action"] = "ISLAND_REASSIGNED"
            else:
                rec["action"] = "ISLAND_FLAG_ONLY"
            records.append(rec)
    return out


# --------------------------------------------------------------- stage 2b --
def find_suspects(seg, raw, affine, vox_mm3):
    rows, _ = audit(seg, affine, vox_mm3)
    info = {r["name"].replace("vertebrae_", ""): r for r in rows if r["voxels"] > 0}
    order = [n for n in TOP_DOWN if n in info]
    suspects = set()
    logv = {n: np.log(max(info[n]["voxels"], 1)) for n in order}
    for i, n in enumerate(order):
        w = [logv[order[j]] for j in range(max(0, i - 2), min(len(order), i + 3)) if j != i]
        if w and abs(logv[n] - float(np.median(w))) > P["suspect_logres"]:
            suspects.add(n)
    for n, r in info.items():
        ratio = r.get("size_ratio_vs_neighbors")
        if ratio is not None and (ratio < P["suspect_size_lo"] or ratio > P["suspect_size_hi"]):
            suspects.add(n)
        if r["components"] > 1 or "ORDER" in r["flags"]:
            suspects.add(n)
    for lid in np.unique(raw):
        if lid == 0:
            continue
        raw_n = int((raw == lid).sum())
        pool_n = int(((raw == lid) & (seg == 0)).sum())
        if raw_n and pool_n / raw_n >= P["suspect_pool_frac"]:
            suspects.add(ID_TO_NAME[int(lid)])
    idx = {n: i for i, n in enumerate(order)}
    s_idx = sorted(idx[n] for n in suspects if n in idx)
    bands = []
    for i in s_idx:
        if bands and i - bands[-1][-1] <= P["band_gap_close"] + 1:
            bands[-1] = list(range(bands[-1][0], i + 1))
        else:
            bands.append([i])
    return [[order[j] for j in range(b[0], b[-1] + 1)] for b in bands], sorted(suspects), order


def column_profile(seg, raw, ct, zooms):
    """Body centerline (largest in-plane bone component) + perpendicular
    mean-HU profile: bodies read high, discs dip, even under kyphosis."""
    bone = ct >= P["bone_hu"]
    M = (seg > 0) | ((raw > 0) & bone)
    Mb = M & bone
    nz = seg.shape[2]
    cx = np.full(nz, np.nan); cy = np.full(nz, np.nan)
    for z in np.nonzero(M.any(axis=(0, 1)))[0]:
        sl2 = Mb[:, :, z] if Mb[:, :, z].any() else M[:, :, z]
        lab, n = ndimage.label(sl2)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        pts = np.nonzero(lab == sizes.argmax())
        cx[z], cy[z] = pts[0].mean(), pts[1].mean()
    w = max(int(40.0 / zooms[2]) | 1, 5)
    ker = np.ones(w) / w
    ok = ~np.isnan(cx)
    cxs, cys = cx.copy(), cy.copy()
    cxs[ok] = np.convolve(np.pad(cx[ok], w // 2, mode="edge"), ker, "valid")
    cys[ok] = np.convolve(np.pad(cy[ok], w // 2, mode="edge"), ker, "valid")
    zmm = np.asarray(zooms)
    Pmm = np.c_[cxs * zmm[0], cys * zmm[1], np.arange(nz) * zmm[2]]
    idx = np.nonzero(ok)[0]
    if idx.size < 5:
        return cxs, cys, np.zeros(nz)
    T = np.zeros_like(Pmm)
    grad = np.gradient(ndimage.gaussian_filter1d(Pmm[idx], sigma=8.0 / zmm[2], axis=0), axis=0)
    grad /= np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-6)
    T[idx] = grad
    r = P["profile_radius_mm"]
    aa, bb = np.meshgrid(np.arange(-r, r + 1.2, 1.2), np.arange(-r, r + 1.2, 1.2))
    keep = aa ** 2 + bb ** 2 <= r ** 2
    offs = np.c_[aa[keep], bb[keep]]
    rho = np.full(nz, np.nan)
    hu = ct.astype(np.float32)
    for z in idx:
        t = T[z]
        ref = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0.0, 1, 0])
        u = np.cross(t, ref); u /= np.linalg.norm(u)
        v = np.cross(t, u)
        pts_mm = Pmm[z] + offs[:, :1] * u + offs[:, 1:2] * v
        vals = ndimage.map_coordinates(hu, (pts_mm / zmm).T, order=1, mode="nearest")
        rho[z] = float(np.clip(vals, 0, 800).mean())
    rho_s = ndimage.gaussian_filter1d(np.where(np.isnan(rho), 0.0, rho),
                                      sigma=max(P["profile_sigma_mm"] / zooms[2], 0.8))
    return cxs, cys, rho_s


def disc_minima(rho_s, z_lo, z_hi, n_cuts, zooms, pin_lo, pin_hi):
    """Exactly n_cuts minima chosen by spacing-regularized DP with anchor pins."""
    seg_rho = rho_s[z_lo:z_hi]
    if seg_rho.size < 5 or seg_rho.max() <= 0 or n_cuts <= 0:
        return [] if n_cuts == 0 else None
    dz = zooms[2]
    peaks, props = signal.find_peaks(-seg_rho, distance=max(int(6.0 / dz), 2),
                                     prominence=1e-6)
    if peaks.size < n_cuts:
        return None
    z = peaks.astype(float) * dz
    p = props["prominences"] / props["prominences"].max()
    m, K = peaks.size, n_cuts
    min_h, max_h = P["min_body_height_mm"], P["max_body_height_mm"]
    span = (z_hi - z_lo) * dz
    NEG = -1e18
    first_ok = (z >= P["pin_lo_mm"]) & (z <= P["pin_hi_mm"]) if pin_lo else np.ones(m, bool)
    last_ok = ((z >= span - P["pin_hi_mm"]) & (z <= span - P["pin_lo_mm"])
               if pin_hi else np.ones(m, bool))
    if not first_ok.any() or not last_ok.any():
        return None
    dp = np.full((K, m, m), NEG)
    bk = np.zeros((K, m, m), dtype=np.int32)
    for j in range(m):
        if first_ok[j]:
            dp[0, j, :] = p[j]
    for k in range(1, K):
        for j in range(m):
            for i in range(j):
                h = z[j] - z[i]
                if not (min_h <= h <= max_h):
                    continue
                if k == 1:
                    best_prev, best_i2 = dp[0, i, 0], 0
                else:
                    best_prev, best_i2 = NEG, 0
                    for i2 in range(i):
                        h_prev = z[i] - z[i2]
                        if not (min_h <= h_prev <= max_h):
                            continue
                        s = dp[k - 1, i, i2] - P["dp_beta"] * np.log(h / h_prev) ** 2
                        if s > best_prev:
                            best_prev, best_i2 = s, i2
                if best_prev > NEG / 2 and best_prev + p[j] > dp[k, j, i]:
                    dp[k, j, i] = best_prev + p[j]
                    bk[k, j, i] = best_i2
    flat = dp[K - 1].copy()
    flat[~last_ok, :] = NEG
    if flat.max() <= NEG / 2:
        return None
    j, i = np.unravel_index(int(flat.argmax()), flat.shape)
    picks = [j]
    k = K - 1
    while k >= 1:
        picks.append(i)
        j, i, k = i, int(bk[k, j, i]), k - 1
    return sorted(int(peaks[q]) + z_lo for q in set(picks))


def arbitrate_band(seg, raw, ct, band, neighbors, cxs, cys, rho_s, zooms,
                   vox_mm3, qa):
    """Re-arbitrate one contiguous suspect band (v2 geometry).

    v2 upgrades over the flat-z version:
      - ARC-LENGTH OBLIQUE CUTS: segment membership uses s = first-order
        projection onto the column centerline tangent, so cut planes tilt with
        kyphosis instead of clipping tilted bodies with axial guillotines.
      - AGREEMENT-EXTENDED SEEDS: body-core seeds grow by geodesic propagation
        through voxels whose CURRENT label already matches the segment (and
        whose s lies in-segment), giving each level arch seeds wherever the
        model already agreed; only genuinely contested bone is left to race.
      - EDT-PRIORITY WATERSHED: flood priority = -distance-to-background, the
        classic touching-object splitter: thick masses (bodies, laminae) are
        claimed first and fronts meet at thin waists, which anatomically are
        the facet joints and the pars, not arbitrary mid-arch loci.
    """
    bone = ct >= P["bone_hu"]
    band_ids = [NAME_TO_ID[n] for n in band]
    band_mask = np.isin(seg, band_ids)
    pool = (np.isin(raw, band_ids)) & (seg == 0) & bone
    zsel = np.nonzero((band_mask | pool).any(axis=(0, 1)))[0]
    pad = int(P["band_pad_mm"] / zooms[2]) + 1
    z0 = max(int(zsel[0]) - pad, 0)
    z1 = min(int(zsel[-1]) + pad + 1, seg.shape[2])
    rec = {"levels": band, "z_range_crop": [int(z0), int(z1)]}
    qa["bands"].append(rec)

    def centroid_z(name):
        zs = np.nonzero((seg == NAME_TO_ID[name]).any(axis=(0, 1)))[0]
        return float(zs.mean()) if zs.size else None

    ups = sorted((NAME_TO_ID[n], centroid_z(n)) for n in band if centroid_z(n) is not None)
    z_up = True if len(ups) < 2 else ups[-1][1] >= ups[0][1]
    rec["z_increases_toward_head"] = bool(z_up)
    lo_name, hi_name = (neighbors if z_up else neighbors[::-1])
    w_lo = centroid_z(lo_name) if lo_name else None
    w_hi = centroid_z(hi_name) if hi_name else None
    win_lo = int(w_lo) if w_lo is not None else int(zsel[0]) + int(4.0 / zooms[2])
    win_hi = int(w_hi) if w_hi is not None else int(zsel[-1]) - int(4.0 / zooms[2])
    n_expect = (len(band_ids) - 1) + (w_lo is not None) + (w_hi is not None)
    found = disc_minima(rho_s, win_lo, win_hi, n_expect, zooms,
                        w_lo is not None, w_hi is not None)
    if found is None:
        qa["flags"].append(f"DISC_COUNT_UNRESOLVED band={band}: flag only, no edit")
        rec["skipped"] = True
        return None
    lo_cut = found.pop(0) if w_lo is not None else z0
    hi_cut = found.pop(-1) if w_hi is not None else z1
    cuts_z = [int(lo_cut)] + [int(f) for f in found] + [int(hi_cut)]
    rec["disc_cuts_z"] = cuts_z

    # ---- arc-length field ----------------------------------------------
    zmm = np.asarray(zooms, dtype=np.float64)
    nzc = seg.shape[2]
    okc = ~np.isnan(cxs)
    Pmm = np.c_[np.where(okc, cxs, 0) * zmm[0], np.where(okc, cys, 0) * zmm[1],
                np.arange(nzc) * zmm[2]]
    T = np.zeros_like(Pmm)
    idxc = np.nonzero(okc)[0]
    g = np.gradient(ndimage.gaussian_filter1d(Pmm[idxc], sigma=8.0 / zmm[2], axis=0), axis=0)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)
    T[idxc] = g
    step_len = np.zeros(nzc)
    step_len[idxc[1:]] = np.linalg.norm(np.diff(Pmm[idxc], axis=0), axis=1)
    s_axis = np.cumsum(step_len)
    s_cuts = [float(s_axis[c]) for c in cuts_z]

    tilt_pad = int(10.0 / zooms[2])
    z0e, z1e = max(cuts_z[0] - tilt_pad, 0), min(cuts_z[-1] + tilt_pad + 1, nzc)
    slab = slice(z0e, z1e)
    seg_s, bone_s = seg[:, :, slab], bone[:, :, slab]
    band_s, pool_s = band_mask[:, :, slab], pool[:, :, slab]
    xx, yy = np.meshgrid(np.arange(seg.shape[0], dtype=np.float32),
                         np.arange(seg.shape[1], dtype=np.float32), indexing="ij")
    s_f = np.empty(seg_s.shape, dtype=np.float32)
    for z in range(z0e, z1e):
        dx = (xx - np.float32(cxs[z] if okc[z] else 0)) * np.float32(zmm[0])
        dy = (yy - np.float32(cys[z] if okc[z] else 0)) * np.float32(zmm[1])
        # oblique correction, clipped to +/-6 mm: full first-order tilt at the
        # body radius (where cuts matter), saturated at rib/process radius
        # where the per-z linearization folds (validated visually)
        corr = np.clip(dx * np.float32(T[z, 0]) + dy * np.float32(T[z, 1]), -6.0, 6.0)
        s_f[:, :, z - z0e] = np.float32(s_axis[z]) + corr

    ids_by_z = sorted(band_ids) if z_up else sorted(band_ids, reverse=True)
    seg_ranges = list(zip(s_cuts[:-1], s_cuts[1:]))
    assign = dict(zip(ids_by_z, seg_ranges))
    s_lo_all, s_hi_all = s_cuts[0], s_cuts[-1]

    # ---- seeds: body core + agreement extension ------------------------
    markers = np.zeros(seg_s.shape, dtype=np.int32)
    r2 = P["seed_radius_mm"] ** 2
    margin = P["seed_margin_mm"]
    d2_slab = np.empty(seg_s.shape, dtype=np.float32)
    for z in range(z0e, z1e):
        dx = (xx - np.float32(cxs[z] if okc[z] else 0)) * np.float32(zmm[0])
        dy = (yy - np.float32(cys[z] if okc[z] else 0)) * np.float32(zmm[1])
        d2_slab[:, :, z - z0e] = dx * dx + dy * dy
    eligible_s = band_s | pool_s
    for lid, (sa, sb) in assign.items():
        seeds = (bone_s & eligible_s & (d2_slab <= r2)
                 & (s_f >= sa + margin) & (s_f <= sb - margin))
        if not seeds.any():
            er = ndimage.binary_erosion(seg_s == lid, structure=STRUCT6, iterations=2)
            seeds = er if er.any() else (seg_s == lid)
            qa["flags"].append(f"SEED_FALLBACK_{ID_TO_NAME[lid]}")
        markers[seeds] = lid
    rec["seed_cm3"] = {ID_TO_NAME[l]: round(float((markers == l).sum()) * vox_mm3 / 1000.0, 1)
                       for l in assign}

    # ---- domain + EDT-priority competition ------------------------------
    st = ndimage.generate_binary_structure(3, 1)
    halo_it = int(max(np.round(P["halo_mm"] / np.asarray(zooms))))
    grown = ndimage.binary_dilation(band_s | pool_s, structure=st, iterations=max(halo_it, 1))
    halo = grown & (seg_s == 0) & bone_s & ~pool_s
    D = (band_s | pool_s | halo) & (s_f >= s_lo_all) & (s_f <= s_hi_all)
    markers[~D] = 0
    # uniform-speed geodesic competition: distance is the cost. (A depth
    # priority was tried and convicted: laminae form one connected deep river,
    # so the first label entering ran the whole posterior column.)
    labels_ws = watershed(np.zeros(D.shape, dtype=np.uint8), markers=markers, mask=D)

    # ---- phase 2: arch repartition by erosion-splitting ------------------
    # The flood race misassigns posterior elements (spinous/laminae) because
    # facet joints are bone-continuous. Geometry fix: in the arch domain,
    # erode by ~2 mm - thin facet waists disconnect, laminae/spinous stay
    # connected to their own pedicle - then attach each eroded component to
    # the level whose phase-1 BODY it touches and reconstruct by uniform BFS.
    rb2 = (P["seed_radius_mm"] + 4.0) ** 2
    body_zone = d2_slab <= rb2
    A = D & ~body_zone
    if A.any():
        edtA = ndimage.distance_transform_edt(A, sampling=zooms).astype(np.float32)
        core_arch = A & (edtA >= P["arch_erode_mm"])
        cc_arch = cc3d.connected_components(core_arch.astype(np.uint8), connectivity=26)
        n_reassigned = 0
        markers2 = np.where(D & body_zone, labels_ws, 0).astype(np.int32)
        st1 = ndimage.generate_binary_structure(3, 1)
        dil_it = int(np.ceil((P["arch_erode_mm"] + 1.0) / min(zooms)))
        for comp_id in range(1, int(cc_arch.max()) + 1):
            comp = cc_arch == comp_id
            n_comp = int(comp.sum())
            if n_comp < 20:
                continue
            ring = ndimage.binary_dilation(comp, structure=st1, iterations=dil_it) & body_zone & D
            votes = labels_ws[ring]
            votes = votes[votes > 0]
            if votes.size < 10:
                continue
            ids, cnt = np.unique(votes, return_counts=True)
            if cnt.max() / votes.size >= 0.7:
                lid = int(ids[cnt.argmax()])
                markers2[comp] = lid
                n_reassigned += n_comp
        labels_arch = watershed(np.zeros(D.shape, dtype=np.uint8), markers=markers2, mask=D)
        changed = int(((labels_arch != labels_ws) & (labels_arch > 0)).sum())
        keep = labels_arch > 0
        labels_ws = np.where(keep, labels_arch, labels_ws)
        rec["arch_phase2_changed_mm3"] = round(changed * vox_mm3, 1)

    out = seg.copy()
    out[band_mask] = 0  # includes out-of-band overreach, which is cleared
    out_s = out[:, :, slab]
    sel = labels_ws > 0
    out_s[sel] = labels_ws[sel].astype(seg.dtype)
    rec["unreached_pool_mm3"] = round(float((D & ~sel & (seg_s == 0)).sum()) * vox_mm3, 1)
    rec["cleared_out_of_band_mm3"] = round(float((band_mask & (out == 0)).sum()) * vox_mm3, 1)
    return out



def stage2b_arbitrate(seg, raw, ct, affine, zooms, vox_mm3, qa):
    bands, suspects, present = find_suspects(seg, raw, affine, vox_mm3)
    qa["suspects"] = suspects
    if not bands:
        return seg
    cxs, cys, rho_s = column_profile(seg, raw, ct, zooms)

    def badness(s, names):
        rows_, _ = audit(s, affine, vox_mm3)
        bad = 0
        for r in rows_:
            nm = r["name"].replace("vertebrae_", "")
            if nm in names:
                bad += sum(1 for f in r["flags"]
                           if f.startswith(("SIZE", "FRAG", "EMPTY", "ORDER")))
                ratio = r.get("size_ratio_vs_neighbors")
                if ratio is not None and ratio > P["suspect_size_hi"]:
                    bad += 1
        return bad

    out = seg
    for band in bands:
        if {"C1", "C2"} & set(band):
            qa["flags"].append(f"ATLAS_AXIS_SKIP band={band}: no disc plane at C1/C2")
            continue
        i0, i1 = present.index(band[0]), present.index(band[-1])
        above = present[i0 - 1] if i0 > 0 else None
        below = present[i1 + 1] if i1 + 1 < len(present) else None
        cand = arbitrate_band(out, raw, ct, band, (below, above), cxs, cys, rho_s,
                              zooms, vox_mm3, qa)
        if cand is None:
            continue
        b0, b1 = badness(out, set(band)), badness(cand, set(band))
        qa["bands"][-1]["badness_before_after"] = [b0, b1]
        if b1 >= b0:
            qa["flags"].append(f"REVERTED band={band}: badness {b0}->{b1}")
        else:
            out = cand
    return out


def majority_filter(seg, iters=3):
    """Iterated 26-neighborhood majority vote on label-label interface voxels.

    Collapses voxel-scale interdigitation between adjacent labels (the source
    of spurious Euler handles). Label<->label swaps only; the mode must
    strictly beat the current label locally, so the filter converges and is
    approximately volume-preserving."""
    offs = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
    out = seg.copy()
    for _ in range(iters):
        diff = np.zeros(out.shape, dtype=bool)
        for ax in range(3):
            s_hi = [slice(None)] * 3; s_lo = [slice(None)] * 3
            s_hi[ax] = slice(1, None); s_lo[ax] = slice(None, -1)
            x, y = out[tuple(s_hi)], out[tuple(s_lo)]
            b = (x > 0) & (y > 0) & (x != y)
            diff[tuple(s_hi)] |= b
            diff[tuple(s_lo)] |= b
        diff &= out > 0
        pts = np.nonzero(diff)
        n = pts[0].size
        if n == 0:
            break
        counts = np.zeros((n, 25), dtype=np.uint8)
        shp = out.shape
        for dx, dy, dz in offs:
            xx = np.clip(pts[0] + dx, 0, shp[0] - 1)
            yy = np.clip(pts[1] + dy, 0, shp[1] - 1)
            zz = np.clip(pts[2] + dz, 0, shp[2] - 1)
            counts[np.arange(n), out[xx, yy, zz]] += 1
        cur = out[pts]
        lab_counts = counts[:, 1:]
        mode = lab_counts.argmax(axis=1) + 1
        swap = (mode != cur) & (lab_counts.max(axis=1) > counts[np.arange(n), cur])
        if not swap.any():
            break
        out[pts[0][swap], pts[1][swap], pts[2][swap]] = mode[swap].astype(out.dtype)
    return out


def absorb_orphans(seg, vox_mm3, max_mm3=1200.0):
    """Absorb non-largest components below max_mm3 into the label dominating
    their shell (remove if speck-sized and shell-free)."""
    out = seg.copy()
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        cc, counts = _components(out == lid)
        if (counts > 0).sum() <= 1:
            continue
        main_id = int(counts.argmax())
        for comp_id in np.nonzero(counts)[0]:
            if comp_id == main_id or counts[comp_id] * vox_mm3 > max_mm3:
                continue
            comp = cc == comp_id
            sl = _bbox_pad(ndimage.find_objects(comp.astype(np.uint8))[0],
                           np.array([2, 2, 2]), seg.shape)
            comp_l = comp[sl]
            shell = ndimage.binary_dilation(comp_l, iterations=2) & ~comp_l
            votes = out[sl][shell]
            votes = votes[(votes != 0) & (votes != lid)]
            if votes.size >= 5:
                ids, cnt = np.unique(votes, return_counts=True)
                out[comp] = int(ids[cnt.argmax()])
            elif counts[comp_id] * vox_mm3 < 100.0:
                out[comp] = 0
    return out


# ---------------------------------------------------------------- stage 3 --
def stage3_smooth(seg, ct, zooms, vox_mm3, qa):
    seg = majority_filter(seg, iters=3)
    seg = absorb_orphans(seg, vox_mm3)
    bone = ct >= P["bone_hu"]
    pad_vox = np.ceil(P["smooth_pad_mm"] / np.asarray(zooms))
    objs = ndimage.find_objects(seg)
    per_label = {}
    for lid in [int(v) for v in np.unique(seg) if v != 0]:
        sl = _bbox_pad(objs[lid - 1], pad_vox, seg.shape)
        m = seg[sl] == lid
        others = (seg[sl] > 0) & ~m
        m1 = ndimage.binary_fill_holes(m)
        # directional pit/window filling: 2D holes in sagittal and coronal
        # slices catch wall windows and tunnels that are not 3D-enclosed;
        # axial is EXCLUDED so the vertebral canal is never filled.
        for ax in (0, 1):
            fill2d = np.zeros_like(m1)
            for k in range(m1.shape[ax]):
                idx = [slice(None)] * 3
                idx[ax] = k
                fill2d[tuple(idx)] = ndimage.binary_fill_holes(m1[tuple(idx)])
            m1 |= fill2d & ~others
        m_f = ndimage.binary_fill_holes(m1)
        m_f &= ~others
        d_in = ndimage.distance_transform_edt(m_f, sampling=zooms)
        d_out = ndimage.distance_transform_edt(~m_f, sampling=zooms)
        sdt = (d_in - d_out).astype(np.float32)
        s = ndimage.gaussian_filter(sdt, sigma=[P["sigma_mm"] / z for z in zooms])
        target = int(m_f.sum())
        lo, hi = -2.0, 2.0
        tau = 0.0
        for _ in range(40):
            tau = 0.5 * (lo + hi)
            n = int((s > tau).sum())
            if abs(n - target) <= P["vol_tol"] * target:
                break
            lo, hi = (tau, hi) if n > target else (lo, tau)
        per_label[lid] = (sl, m_f, sdt, s, float(tau))
        qa["smooth"].append({"label": ID_TO_NAME[lid],
                             "holes_filled_mm3": round(float((m_f & ~m).sum()) * vox_mm3, 1),
                             "tau_mm": round(float(tau), 3)})
    out = np.zeros_like(seg)
    step = max(64, int(120 / zooms[2]))
    for c0 in range(0, seg.shape[2], step):
        c1 = min(c0 + step, seg.shape[2])
        score = np.full((seg.shape[0], seg.shape[1], c1 - c0), -np.inf, dtype=np.float32)
        lab = np.zeros((seg.shape[0], seg.shape[1], c1 - c0), dtype=seg.dtype)
        for lid, (sl, m_f, sdt, s, tau) in per_label.items():
            i0, i1 = max(sl[2].start, c0), min(sl[2].stop, c1)
            if i0 >= i1:
                continue
            zloc = slice(i0 - sl[2].start, i1 - sl[2].start)
            s_l, sdt_l, m_l = s[:, :, zloc], sdt[:, :, zloc], m_f[:, :, zloc]
            band = np.abs(sdt_l) <= P["max_dev_mm"]
            claim = np.where(band, s_l > tau, m_l)
            marg = np.where(band, s_l - tau, np.where(m_l, 1e3, -1e3)).astype(np.float32)
            claim &= (m_l | bone[sl[0], sl[1], i0:i1])
            sc = score[sl[0], sl[1], i0 - c0:i1 - c0]
            lb = lab[sl[0], sl[1], i0 - c0:i1 - c0]
            upd = claim & (marg > sc)
            sc[upd] = marg[upd]
            lb[upd] = lid
        out[:, :, c0:c1] = lab
    for lid, (sl, m_f, sdt, s, tau) in per_label.items():
        new = out[sl] == lid
        denom = int(new.sum()) + int(m_f.sum())
        dice = 2 * int((new & m_f).sum()) / denom if denom else 1.0
        _, c_new = _components(new)
        _, c_old = _components(m_f)
        rec = next(r for r in qa["smooth"] if r["label"] == ID_TO_NAME[lid])
        rec["dice_vs_presmooth"] = round(dice, 4)
        if dice < P["min_dice"] or int((c_new > 0).sum()) > int((c_old > 0).sum()):
            out[sl][m_f] = lid
            out[sl][(out[sl] == lid) & ~m_f] = 0
            rec["reverted"] = True
    return absorb_orphans(out, vox_mm3)


# --------------------------------------------------------------------- io --
def load_case(case_dir: Path, ct_root: Path):
    combined = case_dir / "combined_labels.nii.gz"
    if combined.exists():
        seg_img = nib.load(str(combined))
        seg = np.asarray(seg_img.dataobj).astype(np.uint8)
    else:
        seg_dir = case_dir / "segmentations"
        files = sorted(seg_dir.glob("vertebrae_*.nii.gz"))
        if not files:
            return None
        seg_img = nib.load(str(files[0]))
        seg = np.zeros(seg_img.shape, dtype=np.uint8)
        for f in files:
            name = f.name.replace("vertebrae_", "").replace(".nii.gz", "")
            if name in NAME_TO_ID:
                seg[np.asarray(nib.load(str(f)).dataobj) > 0] = NAME_TO_ID[name]
    ct_path = ct_root / case_dir.name / "ct.nii.gz"
    if not ct_path.exists():
        log.error("%s: no CT at %s (CT evidence is required)", case_dir.name, ct_path)
        return None
    ct_img = nib.load(str(ct_path))
    ct = np.asarray(ct_img.dataobj)
    if ct.shape != seg.shape:
        log.error("%s: CT grid %s != prediction grid %s", case_dir.name, ct.shape, seg.shape)
        return None
    return seg_img, seg, np.clip(ct, -1024, 3071).astype(np.int16), ct_img


def write_case(out_dir: Path, case_id: str, seg_full, seg_img):
    case_out = out_dir / case_id
    (case_out / "segmentations").mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(seg_full.astype(np.uint8), seg_img.affine, seg_img.header),
             str(case_out / "combined_labels.nii.gz"))
    for lid, name in ID_TO_NAME.items():
        m = (seg_full == lid).astype(np.uint8)
        nib.save(nib.Nifti1Image(m, seg_img.affine, seg_img.header),
                 str(case_out / "segmentations" / f"vertebrae_{name}.nii.gz"))


def process_case(case_dir: Path, ct_root: Path, out_dir: Path, report_dir: Path):
    loaded = load_case(case_dir, ct_root)
    if loaded is None:
        return None
    seg_img, seg_full, ct_full, _ = loaded
    zooms = tuple(float(z) for z in seg_img.header.get_zooms()[:3])
    vox_mm3 = float(abs(np.linalg.det(seg_img.affine[:3, :3])))
    # crop to the spine region (+margin) for all processing
    pad = np.maximum(np.round(P["crop_margin_mm"] / np.asarray(zooms)).astype(int), 2)
    nz = np.nonzero(seg_full)
    lo = np.maximum(np.array([c.min() for c in nz]) - pad, 0)
    hi = np.minimum(np.array([c.max() for c in nz]) + pad + 1, seg_full.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    seg = seg_full[sl].copy()
    ct = ct_full[sl]
    aff = seg_img.affine.copy()
    aff[:3, 3] = (seg_img.affine @ np.array([lo[0], lo[1], lo[2], 1.0]))[:3]

    qa = {"case": case_dir.name, "params": P, "records": [], "bands": [],
          "flags": [], "smooth": []}
    _, s0 = audit(seg, aff, vox_mm3)
    log.info("%s input audit: %s", case_dir.name, s0)
    raw = seg.copy()
    seg = stage1_triage(seg, ct, zooms, vox_mm3, qa["records"])
    seg = stage2a_islands(seg, vox_mm3, qa["records"])
    seg = stage2b_arbitrate(seg, raw, ct, aff, zooms, vox_mm3, qa)
    seg = stage3_smooth(seg, ct, zooms, vox_mm3, qa)
    rows, s1 = audit(seg, aff, vox_mm3)
    qa["audit_before"], qa["audit_after"] = s0, s1
    qa["audit_rows_after"] = rows
    log.info("%s output audit: %s", case_dir.name, s1)

    out_full = np.zeros_like(seg_full, dtype=np.uint8)
    out_full[sl] = seg
    write_case(out_dir, case_dir.name, out_full, seg_img)
    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"{case_dir.name}_postprocessing_qa.json").write_text(
            json.dumps(qa, indent=1, default=float))
    return s0, s1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred_dir", required=True, type=Path)
    ap.add_argument("--ct_root", required=True, type=Path,
                    help="Root holding <case>/ct.nii.gz (evidence gates need HU).")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--report_dir", type=Path, default=Path("reports"))
    ap.add_argument("--case", action="append", default=None)
    args = ap.parse_args()
    cases = sorted(p for p in args.pred_dir.iterdir() if p.is_dir())
    if args.case:
        want = {c for v in args.case for c in v.split(",")}
        cases = [c for c in cases if c.name in want]
    if not cases:
        log.error("no case folders under %s", args.pred_dir)
        return
    for case_dir in cases:
        process_case(case_dir, args.ct_root, args.out_dir, args.report_dir)
    log.info("done: refined predictions in %s", args.out_dir)


if __name__ == "__main__":
    main()
