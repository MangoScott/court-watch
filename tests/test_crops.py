import json
import math

from PIL import Image

from conftest import SITE_LAT, SITE_LON, court_obb, make_site, TENNIS_W, TENNIS_H, SIZE
from common import read_json
from crops import contact_sheet, crop_court, crop_from_chip, lonlat_ring_to_pixels, pixels_to_norm_obb


def test_crop_court_puts_long_axis_vertical_and_centres_court():
    im = Image.new("RGB", (512, 512), (0, 0, 0))
    # bright rotated rectangle 100 x 40 px centred at (256, 256), 30 degrees
    from PIL import ImageDraw
    a = math.radians(30)
    ux, uy, vx, vy = math.cos(a), math.sin(a), -math.sin(a), math.cos(a)
    pts = [(256 + sx * ux * 50 + sy * vx * 20, 256 + sx * uy * 50 + sy * vy * 20) for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    ImageDraw.Draw(im).polygon(pts, fill=(255, 255, 255))
    crop = crop_court(im, pts, out_w=80, out_h=160, margin=0.5)
    assert crop.size == (80, 160)
    px = crop.load()
    assert px[40, 80][0] > 200                      # centre is court
    assert px[40, 5][0] < 30 and px[5, 80][0] < 30  # margin is background
    # court occupies ~2/3 of each axis: edges at 1/6 and 5/6
    assert px[40, int(160 * 0.20)][0] > 200 and px[40, int(160 * 0.10)][0] < 30


def test_lonlat_roundtrip_with_sidecar(tmp_path):
    site_id = "s"
    obb = court_obb(0.3, 0.5, TENNIS_W, TENNIS_H)
    chips, _ = make_site(tmp_path, site_id, SITE_LAT, SITE_LON, {2023: [{"class": "tennis", "confidence": 1, "obb": obb}]})
    sidecar = read_json(chips / site_id / "2023.json")
    # build a lon/lat ring from the pixel obb via 06's helper, then invert it
    from conftest import load_script
    cd = load_script("06_change_detection")
    ring = cd.obb_to_lonlat(obb, sidecar)
    pts = lonlat_ring_to_pixels(ring, sidecar)
    expect = [(x * SIZE, y * SIZE) for x, y in obb]
    for (px, py), (ex, ey) in zip(pts, expect):
        assert abs(px - ex) < 0.05 and abs(py - ey) < 0.05
    norm = pixels_to_norm_obb(pts, SIZE)
    assert abs(norm[0][0] - obb[0][0]) < 1e-3
    crop, _ = crop_from_chip(chips / site_id / "2023.png", sidecar, ring)
    assert crop.size == (160, 320)


def test_contact_sheet_layout():
    crops = [Image.new("RGB", (160, 320), (i * 10, 0, 0)) for i in range(7)]
    sheet = contact_sheet(crops, cols=5)
    assert sheet.width == 5 * 168 + 8 and sheet.height == 2 * (320 + 8 + 18) + 8


def test_small_footprint_gets_fixed_ground_window():
    """A court mapped by its playing lines (24 x 11 m) must not fill the frame like a
    pitch-sized footprint does: the window is at least GROUND_LONG_M x GROUND_SHORT_M."""
    from crops import GROUND_LONG_M, GROUND_SHORT_M, ground_window_px
    im = Image.new("RGB", (512, 512), (0, 0, 0))
    from PIL import ImageDraw
    # 24 x 11 m court at 0.6 m/px = 40 x 18 px, axis aligned, centred
    ImageDraw.Draw(im).rectangle([256 - 9, 256 - 20, 256 + 9, 256 + 20], fill=(255, 255, 255))
    pts = [(247, 236), (265, 236), (265, 276), (247, 276)]
    sidecar = {"gsd": 0.6, "transform": [0.6, 0, 0, 0, -0.6, 0]}
    min_long, min_short = ground_window_px(sidecar)
    assert abs(min_long - GROUND_LONG_M / 0.6) < 1e-6 and abs(min_short - GROUND_SHORT_M / 0.6) < 1e-6
    tight = crop_court(im, pts)
    fixed = crop_court(im, pts, min_long=min_long, min_short=min_short)
    frac = lambda c: sum(1 for v in c.convert("L").getdata() if v > 128) / (c.width * c.height)
    assert frac(tight) > 0.5                       # old behaviour: court fills most of the crop
    assert 0.15 < frac(fixed) < 0.35               # 24*11 / (46*23) = 0.25 of the frame
    # centre is still court, and the window is not stretched (aspect follows the crop)
    assert fixed.load()[80, 160][0] > 200


def test_ground_window_from_transform_when_gsd_missing():
    from crops import GROUND_LONG_M, ground_window_px
    ml, _ = ground_window_px({"transform": [1.0, 0, 0, 0, -1.0, 0]})
    assert abs(ml - GROUND_LONG_M) < 1e-6


def test_large_footprint_keeps_margin_and_aspect():
    """A footprint bigger than the minimum window is still fully inside the crop."""
    im = Image.new("RGB", (512, 512), (0, 0, 0))
    from PIL import ImageDraw
    ImageDraw.Draw(im).rectangle([256 - 40, 256 - 100, 256 + 40, 256 + 100], fill=(255, 255, 255))   # 48 x 120 m
    pts = [(216, 156), (296, 156), (296, 356), (216, 356)]
    crop = crop_court(im, pts, min_long=46 / 0.6, min_short=23 / 0.6)
    px = crop.load()
    assert px[80, 160][0] > 200
    assert px[80, 3][0] < 30 and px[3, 160][0] < 30   # margin still visible on both axes
