"""Shared helpers for the court-watch pipeline.

Nothing here talks to the network. Paths, the class list, the detection JSON
schema, chip geometry helpers, and small file utilities live here so every
numbered script agrees on them.

Detection JSON (written by 03_label_with_claude.py and 05_detect.py, read by
06_change_detection.py and spot_check.py)::

    {
      "site_id": "oh_3909750_-8449660",
      "year": 2023,
      "image": "data/chips/oh_3909750_-8449660/2023.png",
      "width": 512, "height": 512,
      "source": "claude:claude-opus-5",          # or "yolo:data/models/best.pt"
      "imagery_date": "2023-06-11",
      "unusable": false,                          # clouds, no data, wrong place
      "chip_notes": "free text from the model",
      "courts": [
        {"class": "hybrid", "confidence": 0.83,
         "obb": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],  # normalized 0-1 image coords
         "notes": "faded pickleball lines on north half"}
      ]
    }
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OSM_DIR = DATA / "osm"
CHIPS_DIR = DATA / "chips"
LABELS_DIR = DATA / "labels"
YOLO_DIR = DATA / "yolo"
MODELS_DIR = DATA / "models"
DETECTIONS_DIR = DATA / "detections"
CHANGE_DIR = DATA / "change"
REVIEW_DIR = DATA / "review"
OUTPUT_DIR = DATA / "output"
BOUNDARIES_DIR = DATA / "boundaries"
SITE_DIR = ROOT / "site"

# Order matters: the index is the YOLO class id.
CLASSES = ["tennis", "hybrid", "pickleball", "padel", "removed"]
CLASS_IDS = {name: i for i, name in enumerate(CLASSES)}
# Sentinel used for unknowns; never a detector output.
UNKNOWN = "unknown"

CHIP_SIZE = 512          # pixels
CHIP_GSD = 0.6           # metres per pixel; all years resampled to this grid
SITE_CLUSTER_M = 60.0    # OSM features closer than this share a site/chip

log = logging.getLogger("court_watch")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# small file helpers
# ---------------------------------------------------------------------------

def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path | str, obj: Any, indent: int = 2) -> None:
    """Write JSON via a temp file + rename so an interrupted run never leaves
    a half-written file that a later resume would treat as done."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, default=_json_default)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _json_default(o: Any) -> Any:
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_csv(path: Path | str, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path: Path | str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# site ids and chip paths
# ---------------------------------------------------------------------------

def make_site_id(state: str, lat: float, lon: float) -> str:
    """Stable id from the site centroid rounded to ~1 m (5 decimals).

    Re-running the OSM fetch after edits keeps ids stable unless the cluster
    centroid moves more than a metre, so cached chips stay attached.
    """
    return f"{state.lower()}_{round(lat * 1e5):d}_{round(lon * 1e5):d}"


def chip_dir(site_id: str, chips_dir: Path = CHIPS_DIR) -> Path:
    return Path(chips_dir) / site_id


def chip_png(site_id: str, year: int, chips_dir: Path = CHIPS_DIR) -> Path:
    return chip_dir(site_id, chips_dir) / f"{year}.png"


def chip_sidecar(site_id: str, year: int, chips_dir: Path = CHIPS_DIR) -> Path:
    return chip_dir(site_id, chips_dir) / f"{year}.json"


def list_chips(chips_dir: Path = CHIPS_DIR) -> list[dict]:
    """Every chip that has both a PNG and a sidecar. Sorted by site, year.

    Returns dicts with site_id, year, png, sidecar (paths) and the sidecar
    contents merged in (imagery_date, transform, crs, coverage, ...).
    """
    chips_dir = Path(chips_dir)
    out = []
    if not chips_dir.exists():
        return out
    for site_path in sorted(p for p in chips_dir.iterdir() if p.is_dir()):
        for sidecar in sorted(site_path.glob("*.json")):
            if sidecar.name == "stac_items.json":
                continue
            stem = sidecar.stem
            if not stem.isdigit():
                continue
            png = sidecar.with_suffix(".png")
            if not png.exists():
                continue
            meta = read_json(sidecar)
            rec = {"site_id": site_path.name, "year": int(stem), "png": png, "sidecar": sidecar}
            rec.update({k: v for k, v in meta.items() if k not in rec})
            out.append(rec)
    return out


def build_chip_manifest(chips_dir: Path = CHIPS_DIR) -> Path:
    """Rebuild chips/manifest.csv from sidecars. Cheap; called after fetches."""
    rows = []
    for c in list_chips(chips_dir):
        rows.append({
            "site_id": c["site_id"], "year": c["year"],
            "imagery_date": c.get("imagery_date", ""),
            "png": str(Path(c["png"]).relative_to(ROOT)) if str(c["png"]).startswith(str(ROOT)) else str(c["png"]),
            "coverage": c.get("coverage", ""),
            "gsd": c.get("gsd", ""), "crs": c.get("crs", ""),
            "stac_item_id": c.get("stac_item_id", ""),
        })
    path = Path(chips_dir) / "manifest.csv"
    write_csv(path, rows, ["site_id", "year", "imagery_date", "png", "coverage", "gsd", "crs", "stac_item_id"])
    return path


# ---------------------------------------------------------------------------
# geometry helpers (pure math, no GIS libs needed)
# ---------------------------------------------------------------------------

def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing lon/lat."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return (32600 if lat >= 0 else 32700) + zone


def chip_transform(cx: float, cy: float, size: int = CHIP_SIZE, gsd: float = CHIP_GSD) -> list[float]:
    """Affine (a, b, c, d, e, f) for a north-up chip centred on projected (cx, cy).

    Matches rasterio's Affine ordering: x = a*col + b*row + c, y = d*col + e*row + f.
    """
    half = size * gsd / 2.0
    return [gsd, 0.0, cx - half, 0.0, -gsd, cy + half]


def pixel_to_world(transform: list[float], col: float, row: float) -> tuple[float, float]:
    a, b, c, d, e, f = transform
    return a * col + b * row + c, d * col + e * row + f


def obb_norm_to_pixels(obb: list[list[float]], width: int, height: int) -> list[tuple[float, float]]:
    return [(float(x) * width, float(y) * height) for x, y in obb]


def clamp01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else float(v)


def normalize_obb(obb: Any) -> list[list[float]] | None:
    """Coerce model output into 4 [x, y] points in [0, 1]. None if hopeless."""
    try:
        pts = [[clamp01(float(p[0])), clamp01(float(p[1]))] for p in obb]
    except (TypeError, ValueError, IndexError):
        return None
    if len(pts) != 4:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if max(xs) - min(xs) < 0.005 or max(ys) - min(ys) < 0.005:
        return None  # degenerate
    return pts


def yolo_obb_line(class_name: str, obb: list[list[float]]) -> str:
    """One YOLO-OBB label line: `cls x1 y1 x2 y2 x3 y3 x4 y4` normalized."""
    cid = CLASS_IDS[class_name]
    coords = " ".join(f"{v:.6f}" for pt in obb for v in pt)
    return f"{cid} {coords}"
