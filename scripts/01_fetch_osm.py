#!/usr/bin/env python
"""Step 1: fetch tennis, pickleball and padel court locations from OpenStreetMap.

Queries Overpass for ``leisure=pitch`` features whose ``sport`` tag mentions
tennis, pickleball or padel inside a US state, then groups features that sit
within 60 m of each other into *sites*. One site = one imagery chip in later
steps, so a bank of eight individually mapped courts becomes one site and is
never double counted.

Outputs (in data/osm/)::

    <STATE>.gpkg          layer "features": one row per OSM feature
                          layer "sites":    one row per cluster (point centroid)
    <STATE>_sites.csv     site_id, lat, lon, n_features, sports, extent_m, oversized
    raw/<STATE>_<date>.json   the untouched Overpass response (cached)

Usage::

    python scripts/01_fetch_osm.py --state OH
    python scripts/01_fetch_osm.py --state OH --use-cache      # re-parse last raw response
    python scripts/01_fetch_osm.py --bbox 39.08,-84.52,39.12,-84.47 --state OH   # small test area

The raw response is cached, so re-running only re-clusters unless you pass
--refresh. Overpass mirrors are tried in order with backoff.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    OSM_DIR, SITE_CLUSTER_M, CHIP_SIZE, CHIP_GSD, log, make_site_id, setup_logging, write_csv,
)

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
SPORT_RE = r"tennis|pickleball|padel|paddle"


def overpass_query(state: str | None, bbox: tuple[float, float, float, float] | None) -> str:
    if bbox:
        s, w, n, e = bbox
        area_filter = f"({s},{w},{n},{e})"
        area_def = ""
    else:
        area_def = f'area["ISO3166-2"="US-{state.upper()}"]["admin_level"="4"]->.a;\n'
        area_filter = "(area.a)"
    return (
        "[out:json][timeout:600][maxsize:1073741824];\n"
        f"{area_def}"
        "(\n"
        f'  nwr["leisure"="pitch"]["sport"~"{SPORT_RE}",i]["sport"!~"^table_tennis$",i]{area_filter};\n'
        f'  nwr["leisure"="sports_centre"]["sport"~"^(tennis|pickleball|padel)$",i]{area_filter};\n'
        ");\n"
        "out geom tags;\n"
    )


def fetch_overpass(query: str, urls: list[str] = OVERPASS_URLS, retries: int = 4) -> dict:
    delay = 10
    last_err: Exception | None = None
    for attempt in range(retries):
        for url in urls:
            try:
                log.info("Overpass %s (attempt %d)", url, attempt + 1)
                r = requests.post(url, data={"data": query}, timeout=900,
                                  headers={"User-Agent": "court-watch/0.1 (github.com/MangoScott/court-watch)"})
                if r.status_code in (429, 504, 502, 503):
                    raise requests.HTTPError(f"{r.status_code} from {url}")
                r.raise_for_status()
                data = r.json()
                if "elements" not in data:
                    raise ValueError("no elements in response")
                return data
            except (requests.RequestException, ValueError) as e:
                last_err = e
                log.warning("Overpass failed: %s", e)
        log.info("sleeping %ds before retry", delay)
        time.sleep(delay)
        delay = min(delay * 2, 300)
    raise RuntimeError(f"Overpass failed after {retries} rounds: {last_err}")


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def element_geometry(el: dict):
    """Shapely geometry (EPSG:4326) for a node/way/relation from `out geom`."""
    from shapely.geometry import LineString, MultiPolygon, Point, Polygon
    from shapely.ops import unary_union

    t = el.get("type")
    if t == "node":
        return Point(el["lon"], el["lat"])
    if t == "way":
        coords = [(p["lon"], p["lat"]) for p in el.get("geometry", []) if p]
        if len(coords) >= 4 and coords[0] == coords[-1]:
            poly = Polygon(coords)
            return poly if poly.is_valid else poly.buffer(0)
        if len(coords) >= 2:
            return LineString(coords)
        return None
    if t == "relation":
        polys = []
        for m in el.get("members", []):
            if m.get("type") != "way" or m.get("role") == "inner":
                continue
            coords = [(p["lon"], p["lat"]) for p in m.get("geometry", []) if p]
            if len(coords) >= 4 and coords[0] == coords[-1]:
                polys.append(Polygon(coords))
        if polys:
            geom = unary_union([p if p.is_valid else p.buffer(0) for p in polys])
            return geom if isinstance(geom, (Polygon, MultiPolygon)) else geom.convex_hull
        if "bounds" in el:
            b = el["bounds"]
            return Point((b["minlon"] + b["maxlon"]) / 2, (b["minlat"] + b["maxlat"]) / 2)
    return None


def primary_sport(tags: dict) -> str:
    sport = (tags.get("sport") or "").lower()
    sport = ";".join(t for t in re.split(r"[;,]", sport) if t.strip() not in ("table_tennis", "beach_tennis"))
    for s in ("padel", "pickleball", "tennis"):
        if s in sport:
            return s
    if "paddle" in sport:
        return "padel"
    return sport or "unknown"


def parse_elements(elements: list[dict], fetched_at: str):
    """Overpass elements -> GeoDataFrame of features (EPSG:4326)."""
    import geopandas as gpd

    rows, geoms = [], []
    for el in elements:
        tags = el.get("tags") or {}
        geom = element_geometry(el)
        if geom is None or geom.is_empty:
            continue
        rows.append({
            "osm_type": el["type"], "osm_id": int(el["id"]),
            "osm_ref": f"{el['type']}/{el['id']}",
            "sport": primary_sport(tags), "sport_raw": tags.get("sport", ""),
            "name": tags.get("name", ""), "surface": tags.get("surface", ""),
            "lit": tags.get("lit", ""), "access": tags.get("access", ""),
            "hoops": tags.get("hoops", ""), "indoor": tags.get("indoor", "") or tags.get("covered", ""),
            "tags": json.dumps(tags, sort_keys=True), "fetched_at": fetched_at,
        })
        geoms.append(geom)
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    # outdoor only for v1; keep the rows but mark them so nothing is silently lost
    gdf["indoor_flag"] = gdf["indoor"].str.lower().isin(["yes", "true", "1"])
    return gdf


def cluster_sites(features, state: str, cluster_m: float = SITE_CLUSTER_M):
    """Group features within ``cluster_m`` of each other into sites.

    Returns (features with site_id column, sites GeoDataFrame of centroids).
    ``extent_m`` is the longest side of the site's bounding box; ``oversized``
    means it will not fit in one chip (CHIP_SIZE * CHIP_GSD metres).
    """
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    if len(features) == 0:
        sites = gpd.GeoDataFrame(columns=["site_id", "lat", "lon", "n_features", "sports", "extent_m", "oversized", "state"],
                                 geometry=[], crs="EPSG:4326")
        features = features.copy()
        features["site_id"] = pd.Series(dtype=str)
        return features, sites

    utm = features.estimate_utm_crs()
    proj = features.to_crs(utm)
    blobs = proj.geometry.buffer(cluster_m / 2.0).union_all()
    blobs = gpd.GeoDataFrame(geometry=list(getattr(blobs, "geoms", [blobs])), crs=utm)
    blobs["cluster"] = np.arange(len(blobs))
    joined = gpd.sjoin(proj[["geometry"]], blobs, how="left", predicate="intersects")
    joined = joined[~joined.index.duplicated(keep="first")]
    proj["cluster"] = joined["cluster"].reindex(proj.index).fillna(-1).astype(int)

    site_rows, site_geoms = [], []
    site_ids = {}
    chip_extent = CHIP_SIZE * CHIP_GSD
    for cid, grp in proj.groupby("cluster"):
        union = grp.geometry.union_all()
        minx, miny, maxx, maxy = union.bounds
        extent = max(maxx - minx, maxy - miny)
        centroid_proj = union.centroid
        centroid = gpd.GeoSeries([centroid_proj], crs=utm).to_crs("EPSG:4326").iloc[0]
        sid = make_site_id(state, centroid.y, centroid.x)
        site_ids[cid] = sid
        sports = sorted(set(features.loc[grp.index, "sport"]))
        site_rows.append({
            "site_id": sid, "lat": round(centroid.y, 6), "lon": round(centroid.x, 6),
            "n_features": len(grp), "sports": ";".join(sports),
            "has_tennis": "tennis" in sports, "has_pickleball": "pickleball" in sports, "has_padel": "padel" in sports,
            "extent_m": round(extent, 1), "oversized": extent > chip_extent * 0.85,
            "any_indoor": bool(features.loc[grp.index, "indoor_flag"].any()),
            "osm_refs": ";".join(features.loc[grp.index, "osm_ref"]), "state": state.upper(),
        })
        site_geoms.append(centroid)
    features = features.copy()
    features["site_id"] = proj["cluster"].map(site_ids)
    sites = gpd.GeoDataFrame(site_rows, geometry=site_geoms, crs="EPSG:4326")
    return features, sites


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="OH", help="two-letter state code (default OH)")
    ap.add_argument("--bbox", help="south,west,north,east to restrict the query (testing)")
    ap.add_argument("--out-dir", type=Path, default=OSM_DIR)
    ap.add_argument("--cluster-m", type=float, default=SITE_CLUSTER_M)
    ap.add_argument("--use-cache", action="store_true", help="re-parse the newest raw response instead of querying")
    ap.add_argument("--refresh", action="store_true", help="query Overpass even if a raw response exists today")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    state = args.state.upper()
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    tag = f"{state}_bbox" if args.bbox else state
    raw_path = raw_dir / f"{tag}_{today}.json"

    existing = sorted(raw_dir.glob(f"{tag}_*.json"))
    if args.use_cache and existing:
        raw_path = existing[-1]
        log.info("using cached Overpass response %s", raw_path)
        data = json.loads(raw_path.read_text())
    elif raw_path.exists() and not args.refresh:
        log.info("raw response for today exists (%s); pass --refresh to re-query", raw_path)
        data = json.loads(raw_path.read_text())
    else:
        bbox = tuple(float(x) for x in args.bbox.split(",")) if args.bbox else None
        query = overpass_query(state, bbox)
        log.debug("query:\n%s", query)
        data = fetch_overpass(query)
        raw_path.write_text(json.dumps(data))
        log.info("saved raw response to %s (%d elements)", raw_path, len(data.get("elements", [])))

    fetched_at = re.sub(r"_(\d{8})\.json$", r"\1", raw_path.name)[-8:]
    fetched_at = f"{fetched_at[:4]}-{fetched_at[4:6]}-{fetched_at[6:]}"
    features = parse_elements(data["elements"], fetched_at)
    log.info("%d features: %s", len(features), features["sport"].value_counts().to_dict() if len(features) else {})
    features, sites = cluster_sites(features, state, args.cluster_m)
    log.info("%d sites (%d oversized for one chip, %d flagged indoor)",
             len(sites), int(sites["oversized"].sum()) if len(sites) else 0,
             int(sites["any_indoor"].sum()) if len(sites) else 0)

    gpkg = args.out_dir / f"{tag}.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    features.to_file(gpkg, layer="features", driver="GPKG")
    sites.to_file(gpkg, layer="sites", driver="GPKG")
    write_csv(args.out_dir / f"{tag}_sites.csv", sites.drop(columns="geometry").to_dict("records"))
    log.info("wrote %s and %s", gpkg, args.out_dir / f"{tag}_sites.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
