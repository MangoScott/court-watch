#!/usr/bin/env python
"""Step 2: download a NAIP chip per site per year from the Planetary Computer.

For every site in ``data/osm/<STATE>_sites.csv`` this searches the ``naip``
STAC collection for items covering the site centroid, then for each year reads
a 512 x 512 px window resampled onto a fixed ground grid (0.6 m/px, UTM zone of
the site, north-up, centred on the site). Because the grid is fixed per site,
chips from different years line up pixel for pixel, which is what makes the
change detection in step 6 simple.

Outputs (in data/chips/<site_id>/)::

    stac_items.json     cached STAC search result (item id, datetime, asset href, bbox)
    <year>.png          RGB chip
    <year>.json         sidecar: imagery_date, stac_item_id, crs, transform, gsd, coverage, ...
    ../manifest.csv     rebuilt at the end from all sidecars

Everything is cached: existing PNG+sidecar pairs are skipped, and the STAC
search is only repeated with --refresh-stac. Safe to interrupt and rerun.

Usage::

    python scripts/02_fetch_naip.py --state OH                # all sites
    python scripts/02_fetch_naip.py --state OH --limit 20     # first 20 sites (smoke test)
    python scripts/02_fetch_naip.py --site-id oh_3909750_-8449660
    python scripts/02_fetch_naip.py --state OH --workers 4
    python scripts/02_fetch_naip.py --lat 39.0975 --lon -84.4966 --site-id sawyer_point   # ad-hoc point

NAIP is flown per state every 2-3 years (Ohio: 2011, 2013, 2015, 2017, 2019,
2021, 2023, ...). Pre-2018 imagery is 1 m so those chips are upsampled to the
0.6 m grid; the sidecar records the native ``source_gsd``.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHIPS_DIR, CHIP_GSD, CHIP_SIZE, OSM_DIR, build_chip_manifest, chip_dir, chip_png, chip_sidecar,
    chip_transform, log, read_csv, read_json, setup_logging, utm_epsg, write_json_atomic,
)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# GDAL settings that make remote COG windows fast and avoid directory listings.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("VSI_CACHE", "TRUE")


# ---------------------------------------------------------------------------
# STAC
# ---------------------------------------------------------------------------

def search_items(lat: float, lon: float, retries: int = 4) -> list[dict]:
    """All NAIP items whose footprint covers the point, newest first.

    Returned dicts are plain JSON (no pystac objects) so they cache cleanly.
    Hrefs are stored unsigned; sign at read time (signatures expire).
    """
    from pystac_client import Client

    delay = 5
    for attempt in range(retries):
        try:
            catalog = Client.open(STAC_URL)
            search = catalog.search(collections=["naip"], intersects={"type": "Point", "coordinates": [lon, lat]},
                                    limit=100)
            items = []
            for it in search.items():
                asset = it.assets.get("image")
                if asset is None:
                    continue
                props = it.properties
                items.append({
                    "id": it.id,
                    "datetime": it.datetime.isoformat() if it.datetime else props.get("datetime"),
                    "year": int(props.get("naip:year") or (it.datetime.year if it.datetime else 0)),
                    "gsd": props.get("gsd"),
                    "href": asset.href,
                    "bbox": list(it.bbox) if it.bbox else None,
                    "proj_epsg": props.get("proj:epsg") or (props.get("proj:code", "").replace("EPSG:", "") or None),
                })
            items.sort(key=lambda d: d["datetime"] or "", reverse=True)
            return items
        except Exception as e:  # network / STAC hiccups
            if attempt == retries - 1:
                raise
            log.warning("STAC search failed (%s); retry in %ds", e, delay)
            time.sleep(delay)
            delay *= 2
    return []


def pick_item_per_year(items: list[dict], lat: float, lon: float) -> dict[int, dict]:
    """One item per year. Where a point sits on a tile seam and several items
    match, prefer the one whose bbox centre is closest to the point (largest
    margin, so the 300 m window is least likely to run off the tile)."""
    by_year: dict[int, list[dict]] = {}
    for it in items:
        by_year.setdefault(it["year"], []).append(it)
    out = {}
    for year, its in by_year.items():
        def margin(it):
            if not it["bbox"]:
                return 0
            w, s, e, n = it["bbox"]
            return min(lon - w, e - lon, lat - s, n - lat)
        out[year] = max(its, key=margin)
    return out


# ---------------------------------------------------------------------------
# raster read
# ---------------------------------------------------------------------------

def read_chip(href: str, lat: float, lon: float, size: int = CHIP_SIZE, gsd: float = CHIP_GSD):
    """Read a north-up chip on the site's fixed UTM grid.

    Returns (rgb uint8 array HxWx3, meta dict). ``coverage`` is the fraction of
    pixels with data; chips straddling a tile edge have coverage < 1.
    """
    import numpy as np
    import planetary_computer as pc
    import rasterio
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.vrt import WarpedVRT

    epsg = utm_epsg(lon, lat)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = to_utm.transform(lon, lat)
    transform = chip_transform(cx, cy, size, gsd)
    signed = pc.sign(href)
    with rasterio.Env():
        with rasterio.open(signed) as src:
            src_gsd = float(abs(src.transform.a))
            with WarpedVRT(src, crs=f"EPSG:{epsg}", transform=Affine(*transform), width=size, height=size,
                           resampling=Resampling.bilinear, nodata=0) as vrt:
                data = vrt.read(indexes=[1, 2, 3], out_dtype="uint8")
    rgb = np.transpose(data, (1, 2, 0))
    coverage = float((rgb.max(axis=2) > 0).mean())
    meta = {
        "crs": f"EPSG:{epsg}", "transform": transform, "gsd": gsd, "size": size,
        "source_gsd": src_gsd, "coverage": round(coverage, 4),
        "center_lat": lat, "center_lon": lon,
    }
    return rgb, meta


def save_png(rgb, path: Path) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    Image.fromarray(rgb, mode="RGB").save(tmp, format="PNG", optimize=True)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# per-site driver
# ---------------------------------------------------------------------------

def fetch_site(site: dict, chips_dir: Path, years: set[int] | None, refresh_stac: bool,
               size: int, gsd: float, min_coverage: float, retries: int = 3) -> dict:
    sid, lat, lon = site["site_id"], float(site["lat"]), float(site["lon"])
    d = chip_dir(sid, chips_dir)
    d.mkdir(parents=True, exist_ok=True)
    stac_cache = d / "stac_items.json"
    if stac_cache.exists() and not refresh_stac:
        items = read_json(stac_cache)["items"]
    else:
        items = search_items(lat, lon)
        write_json_atomic(stac_cache, {"site_id": sid, "lat": lat, "lon": lon,
                                       "searched_at": datetime.now(timezone.utc).isoformat(), "items": items})
    per_year = pick_item_per_year(items, lat, lon)
    stats = {"site_id": sid, "years": 0, "fetched": 0, "skipped": 0, "failed": 0, "low_coverage": 0}
    for year, item in sorted(per_year.items()):
        if years and year not in years:
            continue
        stats["years"] += 1
        png, sidecar = chip_png(sid, year, chips_dir), chip_sidecar(sid, year, chips_dir)
        if png.exists() and sidecar.exists():
            stats["skipped"] += 1
            continue
        delay = 3
        for attempt in range(retries):
            try:
                rgb, meta = read_chip(item["href"], lat, lon, size, gsd)
                break
            except Exception as e:
                if attempt == retries - 1:
                    log.error("%s %d: read failed: %s", sid, year, e)
                    rgb = None
                else:
                    log.warning("%s %d: %s; retry in %ds", sid, year, e, delay)
                    time.sleep(delay)
                    delay *= 2
        if rgb is None:
            stats["failed"] += 1
            continue
        if meta["coverage"] < min_coverage:
            stats["low_coverage"] += 1
            log.warning("%s %d: coverage %.2f < %.2f, keeping but flagging", sid, year, meta["coverage"], min_coverage)
        save_png(rgb, png)
        meta.update({
            "site_id": sid, "year": year, "imagery_date": (item["datetime"] or "")[:10],
            "stac_item_id": item["id"], "href": item["href"], "low_coverage": meta["coverage"] < min_coverage,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })
        write_json_atomic(sidecar, meta)
        stats["fetched"] += 1
    return stats


def load_sites(args) -> list[dict]:
    if args.lat is not None and args.lon is not None:
        sid = args.site_id or f"adhoc_{round(args.lat * 1e5)}_{round(args.lon * 1e5)}"
        return [{"site_id": sid, "lat": args.lat, "lon": args.lon}]
    csv_path = args.sites or (OSM_DIR / f"{args.state.upper()}_sites.csv")
    sites = read_csv(csv_path)
    if args.site_id:
        sites = [s for s in sites if s["site_id"] == args.site_id]
        if not sites:
            raise SystemExit(f"site {args.site_id} not in {csv_path}")
    if not args.include_indoor:
        sites = [s for s in sites if str(s.get("any_indoor", "")).lower() not in ("true", "1")]
    if args.limit:
        sites = sites[: args.limit]
    return sites


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="OH")
    ap.add_argument("--sites", type=Path, help="sites CSV (default data/osm/<STATE>_sites.csv)")
    ap.add_argument("--site-id", help="only this site id")
    ap.add_argument("--lat", type=float, help="ad-hoc point instead of the sites CSV")
    ap.add_argument("--lon", type=float)
    ap.add_argument("--years", help="comma-separated years to fetch (default all available)")
    ap.add_argument("--limit", type=int, help="only the first N sites")
    ap.add_argument("--chips-dir", type=Path, default=CHIPS_DIR)
    ap.add_argument("--size", type=int, default=CHIP_SIZE)
    ap.add_argument("--gsd", type=float, default=CHIP_GSD, help="metres per pixel of the output grid")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=2, help="parallel sites (be polite to the PC API)")
    ap.add_argument("--refresh-stac", action="store_true", help="re-run STAC searches even if cached")
    ap.add_argument("--include-indoor", action="store_true", help="include sites OSM tags as indoor/covered")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    sites = load_sites(args)
    years = {int(y) for y in args.years.split(",")} if args.years else None
    log.info("%d sites to fetch into %s", len(sites), args.chips_dir)

    totals = {"years": 0, "fetched": 0, "skipped": 0, "failed": 0, "low_coverage": 0}
    t0 = time.time()

    def work(site):
        try:
            return fetch_site(site, args.chips_dir, years, args.refresh_stac, args.size, args.gsd, args.min_coverage)
        except Exception as e:  # keep going; the site can be retried on rerun
            log.error("site %s failed: %s", site["site_id"], e)
            return {"site_id": site["site_id"], "years": 0, "fetched": 0, "skipped": 0, "failed": 1, "low_coverage": 0}

    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, st in enumerate(ex.map(work, sites), 1):
            for k in totals:
                totals[k] += st.get(k, 0)
            if i % 25 == 0 or i == len(sites):
                log.info("[%d/%d] %s  (%.0fs)", i, len(sites), totals, time.time() - t0)

    manifest = build_chip_manifest(args.chips_dir)
    log.info("done: %s; manifest %s", totals, manifest)
    return 0 if totals["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
