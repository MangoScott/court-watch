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

# Every crop covers at least this much ground (metres, long x short axis), so
# absolute size survives: a tennis court (36.6 x 18.3 m fence line, 23.8 x 11 m
# playing lines) fills most of the frame and a single pickleball court
# (13.4 x 6.1 m) stays small. Before this, crops were scaled to the OSM
# footprint, so a tennis court mapped by its playing lines looked the same as
# a pickleball court mapped by its own. The 2:1 ground aspect matches CROP_W:CROP_H.
GROUND_LONG_M = 46.0
GROUND_SHORT_M = 23.0
CROP_VERSION = 2                # bump when crop geometry changes so cached training crops are rebuilt


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


def ground_window_px(sidecar: dict) -> tuple[float, float]:
    """Minimum crop window (long, short) in chip pixels for this chip's ground resolution."""
    gsd = sidecar.get("gsd")
    if not gsd:
        a, b, _, d, e, _ = sidecar["transform"]
        gsd = (math.hypot(a, d) + math.hypot(b, e)) / 2.0
    return GROUND_LONG_M / float(gsd), GROUND_SHORT_M / float(gsd)


def crop_court(im: Image.Image, pts: list[tuple[float, float]], out_w: int = CROP_W, out_h: int = CROP_H,
               margin: float = CROP_MARGIN, min_long: float = 0.0, min_short: float = 0.0) -> Image.Image:
    """Axis-aligned crop of the rotated rectangle ``pts`` (4 pixel corners in order).

    The window is the footprint plus ``margin``, enlarged to at least
    ``min_long`` x ``min_short`` pixels (see ``ground_window_px``). When the
    minimum applies, the window keeps the min_long:min_short aspect and grows
    just enough to contain the footprint, so nothing is stretched.

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
    if min_long > 0 and min_short > 0:
        scale = max(Lp / min_long, Wp / min_short, 1.0)
        Lp, Wp = min_long * scale, min_short * scale
    a, b = sx * Wp / out_w, lx * Lp / out_h
    d, e = sy * Wp / out_w, ly * Lp / out_h
    c = cx - 0.5 * (sx * Wp + lx * Lp)
    f = cy - 0.5 * (sy * Wp + ly * Lp)
    return im.convert("RGB").transform((out_w, out_h), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)


def crop_from_chip(png_path: Path | str, sidecar: dict, ring_lonlat: list[list[float]],
                   out_w: int = CROP_W, out_h: int = CROP_H) -> tuple[Image.Image, list[tuple[float, float]]]:
    pts = lonlat_ring_to_pixels(ring_lonlat, sidecar)
    min_long, min_short = ground_window_px(sidecar)
    with Image.open(png_path) as im:
        return crop_court(im, pts, out_w, out_h, min_long=min_long, min_short=min_short), pts


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
