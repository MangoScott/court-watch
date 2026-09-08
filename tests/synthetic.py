"""Synthetic site generator: chips, sidecars and detections for tests and the demo.

The synthetic site mirrors Sawyer Point: eight tennis courts in a row in 2019
and 2021 (as in the real imagery), and by 2023 three of them are hybrid and
the other five footprints hold 18 dedicated pickleball courts (4+4+4+3+3). Coordinates match
data/validation/sawyer_point.yaml so validate.py can be exercised offline.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SITE_LAT, SITE_LON = 39.10279, -84.49653
SIZE = 512


def load_script(name: str):
    """Import a numbered script module, e.g. load_script('06_change_detection')."""
    return importlib.import_module(name)


def court_obb(cx: float, cy: float, w: float, h: float) -> list[list[float]]:
    return [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]


def tennis_centres():
    # eight courts in a row, 40 px apart, centred vertically
    return [((100 + i * 40) / SIZE, 0.5) for i in range(8)]


TENNIS_W, TENNIS_H = 20 / SIZE, 44 / SIZE
PB_W, PB_H = 9 / SIZE, 20 / SIZE


def pickleball_obbs(cx: float, cy: float, n: int) -> list[list[list[float]]]:
    """n pickleball courts packed 2 per row inside a tennis footprint."""
    out = []
    rows = (n + 1) // 2
    for k in range(n):
        col, row = k % 2, k // 2
        x = cx + (col - 0.5) * (PB_W + 1 / SIZE)
        y = cy + (row - (rows - 1) / 2) * (PB_H + 1 / SIZE)
        out.append(court_obb(x, y, PB_W, PB_H))
    return out


def sawyer_detections() -> dict[int, list[dict]]:
    centres = tennis_centres()
    d2019 = [{"class": "tennis", "confidence": 0.93, "obb": court_obb(cx, cy, TENNIS_W, TENNIS_H), "notes": ""} for cx, cy in centres]
    d2021 = [{"class": "tennis", "confidence": 0.85, "obb": court_obb(cx, cy, TENNIS_W, TENNIS_H), "notes": ""} for cx, cy in centres]
    d2023 = []
    split = [4, 4, 4, 3, 3]
    for i, (cx, cy) in enumerate(centres):
        if i >= 5:
            d2023.append({"class": "hybrid", "confidence": 0.88, "obb": court_obb(cx, cy, TENNIS_W, TENNIS_H), "notes": ""})
        else:
            for obb in pickleball_obbs(cx, cy, split[i]):
                d2023.append({"class": "pickleball", "confidence": 0.8, "obb": obb, "notes": ""})
    return {2019: d2019, 2021: d2021, 2023: d2023}


def write_chip(png: Path, courts: list[dict]) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(0)
    arr = rng.integers(70, 110, (SIZE, SIZE, 3), dtype=np.uint8)
    im = Image.fromarray(arr, "RGB")
    d = ImageDraw.Draw(im)
    for c in courts:
        pts = [(x * SIZE, y * SIZE) for x, y in c["obb"]]
        d.polygon(pts, fill=(60, 120, 90) if c["class"] != "pickleball" else (40, 90, 160), outline=(230, 230, 230))
    png.parent.mkdir(parents=True, exist_ok=True)
    im.save(png, "PNG")


def make_site(root: Path, site_id: str, lat: float, lon: float, detections: dict[int, list[dict]],
              source: str = "test:synthetic") -> tuple[Path, Path]:
    """Create chips/<site>/<year>.png+json and detections/raw/<site>/<year>.json."""
    from common import chip_transform, utm_epsg
    from pyproj import Transformer

    chips = root / "chips"
    dets = root / "detections"
    epsg = utm_epsg(lon, lat)
    cx, cy = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    for year, courts in detections.items():
        png = chips / site_id / f"{year}.png"
        write_chip(png, courts)
        (chips / site_id / f"{year}.json").write_text(json.dumps({
            "site_id": site_id, "year": year, "imagery_date": f"{year}-07-01", "crs": f"EPSG:{epsg}",
            "transform": chip_transform(cx, cy), "gsd": 0.6, "size": SIZE, "source_gsd": 0.6,
            "coverage": 1.0, "center_lat": lat, "center_lon": lon, "stac_item_id": f"fake_{year}",
        }))
        rec = {"site_id": site_id, "year": year, "image": str(png), "width": SIZE, "height": SIZE,
               "imagery_date": f"{year}-07-01", "source": source, "unusable": False, "chip_notes": "",
               "courts": courts}
        (dets / "raw" / site_id).mkdir(parents=True, exist_ok=True)
        (dets / "raw" / site_id / f"{year}.json").write_text(json.dumps(rec))
    return chips, dets
