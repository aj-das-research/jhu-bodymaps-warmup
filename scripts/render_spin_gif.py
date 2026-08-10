"""Animated 360-degree comparison GIF for the README.

Layout per frame (2 x 2):
    row 1: full spine, raw model output | ShapeKit-Pro
    row 2: the T9..L2 spinous region    | same, repaired
Both rows spin about the superior-inferior axis in sync.

Usage:
  python scripts/render_spin_gif.py RAW.nii.gz PRO.nii.gz OUT.gif
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_lateral import SNAP, surface_view

CMAP = np.array([tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)) for c in SNAP])
N_FRAMES = 30
ZOOM_IDS = (4, 5, 6, 7, 8, 9)          # L2..T9, the repaired spinous chain
BG = 12                                 # background gray (0..255)


def resample_iso(vol, zooms, target_mm):
    f = [max(int(round(target_mm / z)), 1) for z in zooms]
    out = vol[::f[0], ::f[1], ::f[2]]
    zm = tuple(z * ff for z, ff in zip(zooms, f))
    return out, zm


def pad_for_spin(vol):
    """Pad xy so any rotation stays inside the frame (reshape=False)."""
    nx, ny, nz = vol.shape
    d = int(np.ceil(np.hypot(nx, ny)))
    px, py = (d - nx) // 2 + 1, (d - ny) // 2 + 1
    return np.pad(vol, ((px, px), (py, py), (0, 0)))


def render_frame(vol, zm, angle):
    """Rotate about the S-I axis, cast from -y (posterior at angle 0)."""
    if angle:
        vol = ndimage.rotate(vol, angle, axes=(0, 1), order=0,
                             reshape=False, prefilter=False)
    v = np.transpose(vol, (1, 0, 2))
    zmv = (zm[1], zm[0], zm[2])
    hit, depth = surface_view(v, zmv, "left")
    fv = np.nanmax(depth) if np.isfinite(depth).any() else 0.0
    d = ndimage.gaussian_filter(np.where(np.isnan(depth), fv, depth), 1.0)
    gy, gz = np.gradient(d, zmv[1], zmv[2])
    light = np.clip(0.35 + 0.65 / np.sqrt(1 + gy ** 2 + gz ** 2), 0, 1)
    rgb = CMAP[np.clip(hit, 0, len(CMAP) - 1)] * light[..., None]
    rgb[hit == 0] = BG / 255.0
    # (x', z) -> image rows=z (flip so head is up), cols=x'
    img = (np.transpose(rgb, (1, 0, 2))[::-1] * 255).astype(np.uint8)
    return img


def to_pil(img, height):
    im = Image.fromarray(img)
    w = int(round(im.width * height / im.height))
    return im.resize((w, height), Image.NEAREST)


def font(size):
    try:
        from matplotlib import font_manager
        fp = font_manager.findfont("DejaVu Sans")
        return ImageFont.truetype(fp, size)
    except Exception:
        return ImageFont.load_default()


def main():
    raw_p, pro_p, out_p = sys.argv[1:4]
    img = nib.load(pro_p)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    pro = np.asarray(img.dataobj).astype(np.uint8)
    raw = np.asarray(nib.load(raw_p).dataobj).astype(np.uint8)
    raw = np.where(raw <= 24, raw, 0).astype(np.uint8)

    # crops follow the CLEAN volume so both columns share a frame
    nz = np.nonzero(pro)
    sl = tuple(slice(max(int(c.min()) - 4, 0), int(c.max()) + 5) for c in nz)
    raw_f, pro_f = raw[sl], pro[sl]
    zsel = np.nonzero(np.isin(pro, ZOOM_IDS).any(axis=(0, 1)))[0]
    zpad = int(round(8.0 / zooms[2]))
    zsl = slice(max(int(zsel[0]) - zpad, 0), int(zsel[-1]) + zpad + 1)
    xy = np.nonzero(np.isin(pro[:, :, zsl], ZOOM_IDS))
    p2 = [int(round(10.0 / z)) for z in zooms[:2]]
    xsl = slice(max(int(xy[0].min()) - p2[0], 0), int(xy[0].max()) + p2[0] + 1)
    ysl = slice(max(int(xy[1].min()) - p2[1], 0), int(xy[1].max()) + p2[1] + 1)
    raw_z, pro_z = raw[xsl, ysl, zsl], pro[xsl, ysl, zsl]

    vols = {}
    for k, (v, mm) in {"rf": (raw_f, 1.8), "pf": (pro_f, 1.8),
                       "rz": (raw_z, 1.0), "pz": (pro_z, 1.0)}.items():
        vv, zm = resample_iso(v, zooms, mm)
        vols[k] = (pad_for_spin(vv), zm)

    H1, H2, GAP, M = 430, 300, 10, 8
    fnt_t = font(17)
    fnt_s = font(14)
    frames = []
    for i in range(N_FRAMES):
        ang = i * 360.0 / N_FRAMES
        p = {k: to_pil(render_frame(v, zm, ang), H1 if k in ("rf", "pf") else H2)
             for k, (v, zm) in vols.items()}
        w1 = max(p["rf"].width, p["pf"].width)
        w2 = max(p["rz"].width, p["pz"].width)
        cw = max(w1, w2)
        W = 2 * cw + GAP + 2 * M
        H = 26 + H1 + 24 + H2 + 2 * M
        canvas = Image.new("RGB", (W, H), (BG, BG, BG))
        dr = ImageDraw.Draw(canvas)
        cx1, cx2 = M + cw // 2, M + cw + GAP + cw // 2
        dr.text((cx1, M + 2), "RAW MODEL OUTPUT", font=fnt_t,
                fill=(235, 235, 235), anchor="mt")
        dr.text((cx2, M + 2), "SHAPEKIT-PRO", font=fnt_t,
                fill=(120, 230, 180), anchor="mt")
        y1 = M + 26
        canvas.paste(p["rf"], (cx1 - p["rf"].width // 2, y1))
        canvas.paste(p["pf"], (cx2 - p["pf"].width // 2, y1))
        y2 = y1 + H1 + 24
        dr.text((W // 2, y1 + H1 + 3),
                "zoom: the T9..L2 spinous region",
                font=fnt_s, fill=(200, 200, 200), anchor="mt")
        canvas.paste(p["rz"], (cx1 - p["rz"].width // 2, y2))
        canvas.paste(p["pz"], (cx2 - p["pz"].width // 2, y2))
        frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=256))
        print(f"frame {i + 1}/{N_FRAMES}", flush=True)

    frames[0].save(out_p, save_all=True, append_images=frames[1:],
                   duration=130, loop=0, optimize=True)
    print("saved", out_p, Path(out_p).stat().st_size / 1e6, "MB")


if __name__ == "__main__":
    main()
