#!/usr/bin/env python
"""Step 2b: newer, sharper imagery from the Ohio Statewide Imagery Program (OSIP).

NAIP (step 2) stops at the 2023 flight for Ohio. OSIP publishes 6-inch (0.15 m)
orthoimagery through ArcGIS ImageServer services, including the OSIP 4
statewide collection that began in spring 2025. This script pulls chips from
those services onto the same fixed per-site grid as the NAIP chips, so every
downstream step (crops, classifier, tracking, site) works unchanged. The
acquisition year comes from the mosaic catalog for the exact point, never
from the service name alone.

Two modes:

``--probe``   Ask the OSIP REST directories what services exist, save their
              metadata to data/osip/probe.json, and (with --lat/--lon) query
              the catalog and export one test chip for that point. Run this
              first on a new machine; the state's servers are not reachable
              from every network.
``--fetch``   For each site (same selection flags as 02_fetch_naip.py), query
              the catalog for imagery covering the site, and export a chip per
              acquisition year that is not already on disk.

Usage::

    python scripts/02b_fetch_osip.py --probe --lat 39.10279 --lon -84.49653
    python scripts/02b_fetch_osip.py --fetch --state OH --near 39.10279,-84.49653
    python scripts/02b_fetch_osip.py --fetch --state OH --limit 300 --min-year 2024
    python scripts/02b_fetch_osip.py --fetch --state OH --gsd 0.3 --size 1024   # sharper chips

Chips land in data/chips/<site_id>/<year>.png with a sidecar marking
``source: osip`` and the service used. If a NAIP chip already exists for that
year the OSIP one is skipped unless --overwrite-source is passed.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHIPS_DIR, CHIP_GSD, CHIP_SIZE, DATA, OSM_DIR, build_chip_manifest, chip_dir, chip_png, chip_sidecar,
    chip_transform, log, read_csv, read_json, setup_logging, utm_epsg, write_json_atomic,
)

OSIP_DIR = DATA / "osip"
DIRECTORIES = [
    "https://geo.oit.ohio.gov/arcgis/rest/services/OSIP",
    "https://geo1.oit.ohio.gov/arcgis/rest/services/OSIP",
]
# Preferred statewide mosaics, best first. Others discovered by --probe can be
# passed with --service.
DEFAULT_SERVICES = [
    "https://geo.oit.ohio.gov/arcgis/rest/services/OSIP/OSIP_6in_best_avail/ImageServer",
    "https://geo.oit.ohio.gov/arcgis/rest/services/OSIP/osip_best_avail_1ft/ImageServer",
]
UA = {"User-Agent": "court-watch/0.1 (github.com/MangoScott/court-watch)"}


def get_json(url: str, params: dict | None = None, retries: int = 3) -> dict:
    params = dict(params or {}, f="json")
    delay = 3
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=120)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(f"ArcGIS error: {data['error']}")
            return data
        except (requests.RequestException, ValueError, RuntimeError) as e:
            if attempt == retries - 1:
                raise
            log.warning("%s failed (%s); retry in %ds", url, e, delay)
            time.sleep(delay)
            delay *= 2
    return {}


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def list_services(directory: str) -> list[dict]:
    d = get_json(directory)
    out = []
    for s in d.get("services", []):
        name = s.get("name", "")
        url = f"{directory.rsplit('/services', 1)[0]}/services/{name}/{s.get('type')}"
        out.append({"name": name, "type": s.get("type"), "url": url})
    return out


def describe(service_url: str) -> dict:
    m = get_json(service_url)
    keep = ("name", "description", "serviceDescription", "extent", "pixelSizeX", "pixelSizeY", "bandCount",
            "capabilities", "maxImageHeight", "maxImageWidth", "spatialReference", "fields", "minValues", "maxValues",
            "copyrightText", "currentVersion", "hasRasterAttributeTable", "defaultMosaicMethod", "allowedMosaicMethods")
    return {k: m.get(k) for k in keep if k in m}


def years_in(text: str) -> list[int]:
    return sorted({int(y) for y in re.findall(r"(?<!\d)(20[0-3]\d)(?!\d)", text or "")})


# ---------------------------------------------------------------------------
# catalog + export
# ---------------------------------------------------------------------------

def catalog_at_point(service_url: str, lat: float, lon: float) -> list[dict]:
    """Mosaic catalog rows whose footprint contains the point (needs Catalog capability)."""
    d = get_json(f"{service_url}/query", {
        "geometry": json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}}),
        "geometryType": "esriGeometryPoint", "inSR": 4326, "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*", "returnGeometry": "false",
    })
    return [f.get("attributes", {}) for f in d.get("features", [])]


DATE_KEYS = ("AcquisitionDate", "ACQUISITIONDATE", "acquisitiondate", "Acquisition_Date", "DATE", "Date", "FlightDate",
             "CaptureDate", "ImageDate", "BeginDate", "EndDate")
YEAR_KEYS = ("Year", "YEAR", "year", "ImageYear", "FlightYear", "OSIP_Year", "Vintage")


def row_year(row: dict) -> tuple[int | None, str | None]:
    """(year, iso_date) from a catalog row: date fields (epoch ms or text), year fields, or the name."""
    for k in DATE_KEYS:
        v = row.get(k)
        if v in (None, "", 0):
            continue
        try:
            if isinstance(v, (int, float)):
                dt = datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(v)[:10])
            return dt.year, dt.date().isoformat()
        except (ValueError, OSError, OverflowError):
            continue
    for k in YEAR_KEYS:
        v = row.get(k)
        if v not in (None, ""):
            try:
                return int(str(v)[:4]), None
            except ValueError:
                pass
    ys = years_in(" ".join(str(row.get(k, "")) for k in ("Name", "NAME", "name", "Tag", "GroupName")))
    return (ys[-1], None) if ys else (None, None)


def export_chip(service_url: str, lat: float, lon: float, size: int, gsd: float, raster_id: int | None = None) -> bytes:
    """PNG bytes on the site's fixed UTM grid (same as 02_fetch_naip.py)."""
    from pyproj import Transformer

    epsg = utm_epsg(lon, lat)
    cx, cy = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
    half = size * gsd / 2.0
    params = {
        "bbox": f"{cx - half},{cy - half},{cx + half},{cy + half}", "bboxSR": epsg, "imageSR": epsg,
        "size": f"{size},{size}", "format": "png", "pixelType": "U8", "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }
    if raster_id is not None:
        params["mosaicRule"] = json.dumps({"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": [int(raster_id)]})
    r = requests.get(f"{service_url}/exportImage", params=params, headers=UA, timeout=180)
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image"):
        raise RuntimeError(f"exportImage returned {r.headers.get('content-type')}: {r.text[:200]}")
    return r.content


def save_chip(png_bytes: bytes, path: Path) -> float:
    """Write PNG as RGB and return coverage (fraction of non-black pixels)."""
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(png_bytes)) as im:
        rgb = im.convert("RGB")
        arr = np.asarray(rgb)
        coverage = float((arr.max(axis=2) > 0).mean())
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(path, "PNG", optimize=True)
    return coverage


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def probe(lat: float | None, lon: float | None, out_dir: Path, services: list[str]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"probed_at": datetime.now(timezone.utc).isoformat(), "directories": {}, "services": {}, "point": None}
    for d in DIRECTORIES:
        try:
            svcs = list_services(d)
            report["directories"][d] = svcs
            log.info("%s: %d services; recent-looking: %s", d, len(svcs),
                     [s["name"] for s in svcs if any(y >= 2024 for y in years_in(s["name"]))])
        except Exception as e:
            report["directories"][d] = {"error": str(e)}
            log.warning("directory %s failed: %s", d, e)
    for s in services:
        try:
            report["services"][s] = describe(s)
            m = report["services"][s]
            log.info("%s: pixel %s m, caps %s, fields %s", s.rsplit("/", 2)[-2], m.get("pixelSizeX"), m.get("capabilities"),
                     [f.get("name") for f in (m.get("fields") or [])][:25])
        except Exception as e:
            report["services"][s] = {"error": str(e)}
            log.warning("describe %s failed: %s", s, e)
    if lat is not None and lon is not None:
        pt: dict = {"lat": lat, "lon": lon, "catalog": {}, "chips": {}}
        for s in services:
            try:
                rows = catalog_at_point(s, lat, lon)
                pt["catalog"][s] = rows
                log.info("%s catalog at point: %d rows; years %s", s.rsplit("/", 2)[-2], len(rows),
                         sorted({row_year(r)[0] for r in rows}))
            except Exception as e:
                pt["catalog"][s] = {"error": str(e)}
                log.warning("catalog query failed for %s: %s", s, e)
            try:
                png = export_chip(s, lat, lon, CHIP_SIZE, CHIP_GSD)
                name = s.rsplit("/", 2)[-2]
                p = out_dir / f"probe_{name}.png"
                cov = save_chip(png, p)
                pt["chips"][s] = {"file": str(p), "coverage": cov}
                log.info("exported test chip %s (coverage %.2f)", p, cov)
            except Exception as e:
                pt["chips"][s] = {"error": str(e)}
                log.warning("exportImage failed for %s: %s", s, e)
        report["point"] = pt
    write_json_atomic(out_dir / "probe.json", report)
    log.info("probe written to %s", out_dir / "probe.json")
    return 0


def fetch(args) -> int:
    import importlib
    naip = importlib.import_module("02_fetch_naip")
    sites = naip.load_sites(args)
    log.info("%d sites; services %s", len(sites), args.service)
    stats = {"sites": 0, "fetched": 0, "skipped": 0, "no_imagery": 0, "failed": 0}
    for i, site in enumerate(sites, 1):
        sid, lat, lon = site["site_id"], float(site["lat"]), float(site["lon"])
        stats["sites"] += 1
        got_any = False
        for svc in args.service:
            try:
                rows = catalog_at_point(svc, lat, lon)
            except Exception as e:
                log.warning("%s: catalog failed on %s: %s", sid, svc, e)
                continue
            by_year: dict[int, dict] = {}
            for r in rows:
                y, date = row_year(r)
                if y is None or (args.min_year and y < args.min_year):
                    continue
                by_year.setdefault(y, {"row": r, "date": date})
            for year, info in sorted(by_year.items()):
                png, sc = chip_png(sid, year, args.chips_dir), chip_sidecar(sid, year, args.chips_dir)
                if png.exists() and sc.exists():
                    existing = read_json(sc)
                    if existing.get("source", "naip") != "osip" and not args.overwrite_source:
                        stats["skipped"] += 1
                        got_any = True
                        continue
                    if existing.get("source") == "osip":
                        stats["skipped"] += 1
                        got_any = True
                        continue
                rid = info["row"].get("OBJECTID") or info["row"].get("objectid")
                try:
                    data = export_chip(svc, lat, lon, args.size, args.gsd, raster_id=rid if args.lock_raster else None)
                    cov = save_chip(data, png)
                except Exception as e:
                    log.error("%s %d: export failed: %s", sid, year, e)
                    stats["failed"] += 1
                    continue
                from pyproj import Transformer
                epsg = utm_epsg(lon, lat)
                cx, cy = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)
                write_json_atomic(sc, {
                    "site_id": sid, "year": year, "imagery_date": info["date"] or str(year), "source": "osip",
                    "service": svc, "catalog_row": {k: v for k, v in info["row"].items() if k.lower() not in ("shape",)},
                    "crs": f"EPSG:{epsg}", "transform": chip_transform(cx, cy, args.size, args.gsd), "gsd": args.gsd,
                    "size": args.size, "source_gsd": None, "coverage": round(cov, 4), "low_coverage": cov < args.min_coverage,
                    "center_lat": lat, "center_lon": lon, "fetched_at": datetime.now(timezone.utc).isoformat(),
                })
                stats["fetched"] += 1
                got_any = True
            if got_any and not args.all_services:
                break
        if not got_any:
            stats["no_imagery"] += 1
        if i % 25 == 0 or i == len(sites):
            log.info("[%d/%d] %s", i, len(sites), stats)
    build_chip_manifest(args.chips_dir)
    log.info("done: %s", stats)
    return 0 if stats["failed"] == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--service", action="append", help="ImageServer URL(s); default: statewide best-available mosaics")
    ap.add_argument("--all-services", action="store_true", help="fetch from every service, not just the first with imagery")
    ap.add_argument("--lock-raster", action="store_true", help="export each catalog raster individually (per-year mosaics)")
    ap.add_argument("--min-year", type=int, default=None, help="only acquisition years >= this (e.g. 2024)")
    ap.add_argument("--overwrite-source", action="store_true", help="replace an existing NAIP chip for the same year")
    # site selection (shared with 02_fetch_naip.py)
    ap.add_argument("--state", default="OH")
    ap.add_argument("--sites", type=Path)
    ap.add_argument("--site-id")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--near")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--include-indoor", action="store_true")
    ap.add_argument("--chips-dir", type=Path, default=CHIPS_DIR)
    ap.add_argument("--out-dir", type=Path, default=OSIP_DIR)
    ap.add_argument("--size", type=int, default=CHIP_SIZE)
    ap.add_argument("--gsd", type=float, default=CHIP_GSD)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)
    args.service = args.service or DEFAULT_SERVICES
    if args.probe:
        return probe(args.lat, args.lon, args.out_dir, args.service)
    if args.fetch:
        return fetch(args)
    ap.error("pass --probe or --fetch")
    return 2


if __name__ == "__main__":
    sys.exit(main())
