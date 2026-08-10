"""Cinematic exploded-view comparison GIF: raw (left) vs ShapeKit-Pro (right).

Storyboard, both columns always in sync:
  A. the five hardest vertebrae (T10..L2, the repaired spinous chain)
     spin assembled through a full turn
  B. they separate vertically into an exploded view (level tags appear)
  C. the exploded stack spins a full turn, every vertebra clear
  D. the camera zooms in and keeps turning, then pulls back
  E. crossfade to a TOP-DOWN view of every separated vertebra, spinning
     in-plane, then crossfade back
  F. the stack merges again, closing the loop seamlessly

Usage:
  python scripts/render_explode_gif.py RAW.nii.gz PRO.nii.gz OUT.gif [--test]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_lateral import SNAP

CMAP = np.array([tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)) for c in SNAP])
LEVELS = [(8, "T10"), (7, "T11"), (6, "T12"), (5, "L1"), (4, "L2")]
BG = 12
ISO_MM = 1.0
DUR = 100          # ms per frame
H_STACK, GAP_SEP = 470, 32
XFADE = 4


def ease(t):
    return 3 * t * t - 2 * t * t * t


def resample_iso(vol, zooms, target):
    f = [max(int(round(target / z)), 1) for z in zooms]
    return vol[::f[0], ::f[1], ::f[2]]


def posterior_cast(vol):
    """Cast along +y after flip: viewer on the -y (posterior) side.
    Returns hit(nx,nz), depth(nx,nz)."""
    nx, ny, nz = vol.shape
    hit = np.zeros((nx, nz), dtype=np.uint8)
    depth = np.full((nx, nz), np.inf, dtype=np.float32)
    todo = np.ones((nx, nz), dtype=bool)
    for y in range(ny):
        sl = vol[:, y, :]
        m = todo & (sl > 0)
        if m.any():
            hit[m] = sl[m]
            depth[m] = y
            todo &= ~m
        if not todo.any():
            break
    return hit, depth


def axial_cast(vol):
    """Cast along -z from the top (superior). Returns hit(nx,ny), depth."""
    b = vol > 0
    rev = b[:, :, ::-1]
    idx = np.argmax(rev, axis=2)
    any_ = rev.any(axis=2)
    zidx = vol.shape[2] - 1 - idx
    hit = np.take_along_axis(vol, zidx[..., None], axis=2)[..., 0]
    hit[~any_] = 0
    depth = np.where(any_, idx, np.inf).astype(np.float32)
    return hit, depth


def shade(hit, depth):
    fin = np.isfinite(depth)
    fv = depth[fin].max() if fin.any() else 0.0
    d = ndimage.gaussian_filter(np.where(fin, depth, fv), 1.0)
    gy, gx = np.gradient(d)
    light = np.clip(0.35 + 0.65 / np.sqrt(1 + gx ** 2 + gy ** 2), 0, 1)
    rgb = CMAP[np.clip(hit, 0, len(CMAP) - 1)] * light[..., None]
    return rgb, depth, hit > 0


def resize3(srgb, sdep, sm, size):
    r = np.asarray(Image.fromarray((np.clip(srgb, 0, 1) * 255).astype(np.uint8))
                   .resize(size, Image.NEAREST)) / 255.0
    dpi = np.asarray(Image.fromarray(
        np.where(np.isfinite(sdep), sdep, 1e6).astype(np.float32))
        .resize(size, Image.NEAREST))
    m = np.asarray(Image.fromarray(sm.astype(np.uint8) * 255)
                   .resize(size, Image.NEAREST)) > 127
    return r, np.where(m, dpi, np.inf), m


def composite(sprites, offsets, H, W):
    rgb = np.full((H, W, 3), BG / 255.0, dtype=np.float32)
    dep = np.full((H, W), np.inf, dtype=np.float32)
    for (srgb, sdep, sm), (r0, c0) in zip(sprites, offsets):
        h, w = sm.shape
        rr = slice(max(r0, 0), min(r0 + h, H))
        cc = slice(max(c0, 0), min(c0 + w, W))
        if rr.stop <= rr.start or cc.stop <= cc.start:
            continue
        sr = slice(rr.start - r0, rr.stop - r0)
        scc = slice(cc.start - c0, cc.stop - c0)
        m = sm[sr, scc] & (sdep[sr, scc] < dep[rr, cc])
        sub = rgb[rr, cc]; sub[m] = srgb[sr, scc][m]; rgb[rr, cc] = sub
        dsub = dep[rr, cc]; dsub[m] = sdep[sr, scc][m]; dep[rr, cc] = dsub
    return rgb


def font(size):
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()


def main():
    raw_p, pro_p, out_p = sys.argv[1:4]
    test = "--test" in sys.argv
    img = nib.load(pro_p)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    pro = np.asarray(img.dataobj).astype(np.uint8)
    raw = np.asarray(nib.load(raw_p).dataobj).astype(np.uint8)
    ids = [lid for lid, _ in LEVELS]
    sel = np.isin(pro, ids)
    nzm = np.nonzero(sel)
    pad = [int(round(8.0 / z)) for z in zooms]
    box = tuple(slice(max(int(c.min()) - p, 0), int(c.max()) + p + 1)
                for c, p in zip(nzm, pad))
    vols = {}
    for tag, seg in (("raw", raw), ("pro", pro)):
        g = np.where(np.isin(seg[box], ids), seg[box], 0).astype(np.uint8)
        g = resample_iso(g, zooms, ISO_MM)
        nx, ny, nzv = g.shape
        d = int(np.ceil(np.hypot(nx, ny)))
        px, py = (d - nx) // 2 + 1, (d - ny) // 2 + 1
        vols[tag] = np.pad(g, ((px, px), (py, py), (0, 0)))
    shp = vols["pro"].shape
    zext = {}
    for lid, _ in LEVELS:
        zz = np.nonzero((vols["pro"] == lid).any(axis=(0, 1)))[0]
        zext[lid] = (int(zz.min()), int(zz.max()))

    sc = H_STACK / shp[2]
    pw = int(round(shp[0] * sc))
    BASE = 2 * GAP_SEP + 8
    PH = H_STACK + 4 * GAP_SEP + 16
    PW = pw + 20
    M, HDR, CAP = 10, 26, 24
    LM = 46
    W = 2 * PW + 3 * M + LM
    H = HDR + PH + CAP + 2 * M
    fnt_t, fnt_s, fnt_l = font(17), font(14), font(13)

    # ---- storyboard ------------------------------------------------------
    KEY = []
    ang = 0.0
    def add(n, dspin, sep_fn, zoom_fn, view, cap):
        nonlocal ang
        for i in range(n):
            t = (i + 1) / n
            KEY.append((ang + dspin * t, sep_fn(t), zoom_fn(t), view, cap))
        ang += dspin
    add(24, 360, lambda t: 0.0, lambda t: 1.0, "post",
        "the five hardest levels, assembled")
    add(10, 120, lambda t: ease(t), lambda t: 1.0, "post", "exploded view")
    add(24, 360, lambda t: 1.0, lambda t: 1.0, "post",
        "exploded view, every vertebra clear")
    add(8, 60, lambda t: 1.0, lambda t: 1 + 0.75 * ease(t), "post",
        "detail zoom")
    add(14, 180, lambda t: 1.0, lambda t: 1.75, "post", "detail zoom")
    add(6, 45, lambda t: 1.0, lambda t: 1.75 - 0.75 * ease(t), "post",
        "detail zoom")
    add(18, 360, lambda t: 1.0, lambda t: 1.0, "top",
        "top-down view, per vertebra")
    rem = (360 - (ang % 360)) % 360
    add(9, rem + 360, lambda t: 1 - ease(t), lambda t: 1.0, "post",
        "reassembled")

    def render(tag, spin, sep, view):
        vr = ndimage.rotate(vols[tag], spin % 360, axes=(0, 1), order=0,
                            reshape=False, prefilter=False) \
            if (spin % 360) > 1e-6 else vols[tag]
        sprites, offs = [], []
        for k, (lid, name) in enumerate(LEVELS):
            v = np.where(vr == lid, vr, 0)
            shift = int(round(sep * (k - 2) * GAP_SEP))
            if view == "post":
                hit, depth = posterior_cast(v)
                srgb, sdep, sm = shade(hit.T[::-1], depth.T[::-1])
                size = (pw, H_STACK)
                srgb, sdep, sm = resize3(srgb, sdep, sm, size)
                offs.append((BASE + shift, 10))
            else:
                zl, zh = zext[lid]
                hit, depth = axial_cast(v[:, :, zl:zh + 1])
                srgb, sdep, sm = shade(hit.T[::-1], depth.T[::-1])
                slot = int(H_STACK / 5 * 0.94)
                f = slot / sm.shape[0]
                size = (max(int(sm.shape[1] * f), 1), max(int(sm.shape[0] * f), 1))
                srgb, sdep, sm = resize3(srgb, sdep, sm, size)
                zc_img = (shp[2] - 1 - 0.5 * (zl + zh)) * sc
                offs.append((BASE + int(round(zc_img - sm.shape[0] / 2)) + shift,
                             10 + (pw - sm.shape[1]) // 2))
            sprites.append((srgb, sdep, sm))
        return composite(sprites, offs, PH, PW)

    def frame(spin, sep, zoom, view, cap):
        canvas = Image.new("RGB", (W, H), (BG, BG, BG))
        dr = ImageDraw.Draw(canvas)
        x_raw, x_pro = M + LM, M + LM + PW + M
        for tag, x0 in (("raw", x_raw), ("pro", x_pro)):
            im = Image.fromarray(
                (np.clip(render(tag, spin, sep, view), 0, 1) * 255).astype(np.uint8))
            if zoom > 1.001:
                zw, zh = int(PW / zoom), int(PH / zoom)
                cx, cy = PW // 2, PH // 2
                im = im.crop((cx - zw // 2, cy - zh // 2,
                              cx + zw // 2, cy + zh // 2)).resize(
                    (PW, PH), Image.LANCZOS)
            canvas.paste(im, (x0, HDR + M))
        dr.text((x_raw + PW // 2, M + 2), "RAW MODEL OUTPUT", font=fnt_t,
                fill=(235, 235, 235), anchor="mt")
        dr.text((x_pro + PW // 2, M + 2), "SHAPEKIT-PRO", font=fnt_t,
                fill=(120, 230, 180), anchor="mt")
        if sep > 0.55 and zoom < 1.15:
            for k, (lid, name) in enumerate(LEVELS):
                zl, zh = zext[lid]
                zc_img = (shp[2] - 1 - 0.5 * (zl + zh)) * sc
                ry = HDR + M + BASE + int(zc_img) \
                    + int(round(sep * (k - 2) * GAP_SEP))
                col = tuple(int(c * 255) for c in CMAP[lid])
                dr.text((M + 20, ry), name, font=fnt_l, fill=col, anchor="mm")
        dr.text((W // 2, H - M - 4), cap, font=fnt_s,
                fill=(190, 190, 190), anchor="ms")
        return canvas

    if test:
        for j in (0, 28, 45, 62, 76, 88, 100, 112):
            if j >= len(KEY):
                continue
            spin, sep, zoom, view, cap = KEY[j]
            frame(spin, sep, zoom, view, cap).save(f"/tmp/explode_{j:03d}.png")
            print("test frame", j, view, f"sep={sep:.2f} zoom={zoom:.2f}")
        return

    frames = []
    prev_view = "post"
    for j, (spin, sep, zoom, view, cap) in enumerate(KEY):
        f = frame(spin, sep, zoom, view, cap)
        if view != prev_view and frames:
            base = frames[-1].convert("RGB")
            for b in range(1, XFADE + 1):
                frames.append(Image.blend(base, f, b / (XFADE + 1))
                              .convert("P", palette=Image.ADAPTIVE, colors=256))
        frames.append(f.convert("P", palette=Image.ADAPTIVE, colors=256))
        prev_view = view
        if (j + 1) % 10 == 0:
            print(f"frame {j + 1}/{len(KEY)}", flush=True)
    frames[0].save(out_p, save_all=True, append_images=frames[1:],
                   duration=DUR, loop=0, optimize=True)
    print("saved", out_p, Path(out_p).stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
