"""Cut a court out of a chip as an axis-aligned crop.

Given the chip's affine transform and a court footprint (four corners in
lon/lat, from footprints.py via 01_fetch_osm.py), this produces a fixed-size
image of that court with its long axis vertical, plus a margin so the fence
and immediate surroundings are visible. The classifier in 04/05 and the
labeling contact sheets in 03 both use this, so a crop looks identical
whether it is being labeled or classified.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    from common import obb_norm_to_pixels
except ImportError:  # pragma: no cover
    from scripts.common import obb_norm_to_pixels  # type: ignore

CROP_W, CROP_H = 160, 320       # pixels; court long axis vertical
CROP_MARGIN = 0.3               # extra context around the footprint (fraction of each side)


def lonlat_ring_to_pixels(ring: list[list[float]], sidecar: dict) -> list[tuple[float, float]]:
    """Footprint corners (lon, lat) -> chip pixel (col, row) via the sidecar's CRS and transform."""
    from pyproj import Transformer

    tf = sidecar["transform"]
    a, b, c, d, e, f = tf
    det = a * e - b * d
    to_proj = Transformer.from_crs("EPSG:4326", sidecar["crs"], always_xy=True)
    out = []
    for lon, lat in ring[:4]:
        x, y = to_proj.transform(lon, lat)
        # invert x = a*col + b*row + c ; y = d*col + e*row + f
        col = (e * (x - c) - b * (y - f)) / det
        row = (-d * (x - c) + a * (y - f)) / det
        out.append((col, row))
    return out


def pixels_to_norm_obb(pts: list[tuple[float, float]], size: int) -> list[list[float]]:
    return [[round(x / size, 5), round(y / size, 5)] for x, y in pts]


def crop_court(im: Image.Image, pts: list[tuple[float, float]], out_w: int = CROP_W, out_h: int = CROP_H,
               margin: float = CROP_MARGIN) -> Image.Image:
    """Axis-aligned crop of the rotated rectangle ``pts`` (4 pixel corners in order).

    Uses one affine resampling pass: output (u, v) maps to the input point
    centre + short_axis * ((u/out_w - 0.5) * W') + long_axis * ((v/out_h - 0.5) * L').
    Areas outside the chip come out black.
    """
    (x0, y0), (x1, y1), (x2, y2), _ = pts
    e1 = (x1 - x0, y1 - y0)
    e2 = (x2 - x1, y2 - y1)
    l1, l2 = math.hypot(*e1), math.hypot(*e2)
    if l1 >= l2:
        long_len, short_len, long_v, short_v = l1, l2, e1, e2
    else:
        long_len, short_len, long_v, short_v = l2, l1, e2, e1
    if long_len < 1 or short_len < 1:
        return Image.new("RGB", (out_w, out_h), (0, 0, 0))
    lx, ly = long_v[0] / long_len, long_v[1] / long_len
    sx, sy = short_v[0] / short_len, short_v[1] / short_len
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    Lp, Wp = long_len * (1 + margin), short_len * (1 + margin)
    a, b = sx * Wp / out_w, lx * Lp / out_h
    d, e = sy * Wp / out_w, ly * Lp / out_h
    c = cx - 0.5 * (sx * Wp + lx * Lp)
    f = cy - 0.5 * (sy * Wp + ly * Lp)
    return im.convert("RGB").transform((out_w, out_h), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)


def crop_from_chip(png_path: Path | str, sidecar: dict, ring_lonlat: list[list[float]],
                   out_w: int = CROP_W, out_h: int = CROP_H) -> tuple[Image.Image, list[tuple[float, float]]]:
    pts = lonlat_ring_to_pixels(ring_lonlat, sidecar)
    with Image.open(png_path) as im:
        return crop_court(im, pts, out_w, out_h), pts


def contact_sheet(crops: list[Image.Image], cols: int = 5, label_start: int = 1,
                  cell_w: int = CROP_W, cell_h: int = CROP_H, pad: int = 8) -> Image.Image:
    """Grid of numbered crops for quick human (or Claude Code) labeling."""
    rows = math.ceil(len(crops) / cols)
    sheet = Image.new("RGB", (cols * (cell_w + pad) + pad, rows * (cell_h + pad + 18) + pad), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for i, im in enumerate(crops):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + pad + 18)
        sheet.paste(im.resize((cell_w, cell_h)), (x, y + 18))
        draw.rectangle([x, y, x + 34, y + 17], fill=(255, 200, 0))
        draw.text((x + 3, y), str(label_start + i), fill=(0, 0, 0), font=font)
    return sheet
