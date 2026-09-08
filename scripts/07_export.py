#!/usr/bin/env python
"""Step 7: apply reviewer overrides and export the dataset and the map site.

Reads ``data/change/court_tracks_raw.json`` (step 6) and, if present,
``data/review/overrides.csv`` (hand corrections: site_id, track_id, year,
class, reviewer, note, n_courts), re-derives transitions, and writes:

data/output/
    courts.parquet          GeoParquet, one polygon per court footprint
    courts.geojson          same, GeoJSON
    transitions.csv         every class change, with flags
    summary_by_state.csv    counts per class + key transitions
    summary_by_county.csv   same by county (Census county boundaries, cached)

site/data/  (consumed by the static MapLibre site in site/)
    courts_points.geojson   one point per court with history for popups
    sites.json              per-site years, imagery dates, chip image paths
    summary.json            headline numbers
    chips/<site>/<year>.jpg before/after imagery

Usage::

    python scripts/07_export.py                  # everything
    python scripts/07_export.py --no-county      # skip the Census download
    python scripts/07_export.py --no-site        # dataset only
    python scripts/07_export.py --state OH --jpeg-quality 80
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BOUNDARIES_DIR, CHANGE_DIR, CHIPS_DIR, CLASSES, OUTPUT_DIR, REVIEW_DIR, SITE_DIR, UNKNOWN, log, read_csv,
    read_json, setup_logging, write_csv, write_json_atomic,
)
from tracking import apply_overrides, class_counts, summarize_track, tracks_from_dicts  # noqa: E402

COUNTY_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"
STATE_FIPS = {"OH": "39"}  # extend as states are added


def load_overrides(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = read_csv(path)
    log.info("%d override rows from %s", len(rows), path)
    return rows


def rebuild_records(raw: dict, overrides: list[dict], flag_conf: float) -> dict[str, dict]:
    sites = {}
    for sid, s in raw["sites"].items():
        tracks = tracks_from_dicts(s["courts"])
        tracks = apply_overrides(tracks, overrides, sid)
        by_id = {r["track_id"]: r for r in s["courts"]}
        recs = []
        for t in tracks:
            r = summarize_track(t, flag_conf)
            old = by_id[t.track_id]
            r.update({k: old.get(k) for k in ("site_id", "court_id", "ring_lonlat", "lat", "lon")})
            r["n_overrides"] = sum(1 for o in t.observations if o.source.startswith("review:"))
            recs.append(r)
        sites[sid] = {**s, "courts": recs}
    return sites


def county_frame(state: str, cache_dir: Path):
    """County polygons for the state from Census cartographic boundaries."""
    import geopandas as gpd
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    gpkg = cache_dir / "counties.gpkg"
    if not gpkg.exists():
        log.info("downloading county boundaries from %s", COUNTY_URL)
        r = requests.get(COUNTY_URL, timeout=300)
        r.raise_for_status()
        zpath = cache_dir / "counties.zip"
        zpath.write_bytes(r.content)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(cache_dir / "counties_shp")
        shp = next((cache_dir / "counties_shp").glob("*.shp"))
        gdf = gpd.read_file(shp)[["STATEFP", "COUNTYFP", "GEOID", "NAME", "STUSPS", "geometry"]]
        gdf.to_file(gpkg, driver="GPKG")
    gdf = gpd.read_file(gpkg)
    fips = STATE_FIPS.get(state.upper())
    if fips:
        gdf = gdf[gdf["STATEFP"] == fips]
    return gdf.to_crs("EPSG:4326")


def build_geodataframe(sites: dict[str, dict], state: str):
    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    rows, geoms = [], []
    for sid, s in sites.items():
        for r in s["courts"]:
            ring = r.get("ring_lonlat")
            if ring and len(ring) >= 4:
                geom = Polygon(ring)
            elif r.get("lon") is not None:
                geom = Point(r["lon"], r["lat"])
            else:
                continue
            rows.append({
                "court_id": r["court_id"], "site_id": sid, "state": state.upper(),
                "lat": r.get("lat"), "lon": r.get("lon"),
                "current_class": r["current_class"], "current_confidence": r["current_confidence"],
                "current_n_courts": r["current_n_courts"], "current_year": r["current_year"],
                "current_imagery_date": r.get("current_imagery_date"),
                "first_year": r["first_year"], "first_class": r["first_class"], "ever_tennis": r["ever_tennis"],
                "years_observed": ",".join(str(y) for y in s["years"]),
                "history": json.dumps([{k: h[k] for k in ("year", "class", "confidence", "n_courts", "imagery_date", "inferred", "interpolated", "source")} for h in r["history"]]),
                "transitions": json.dumps(r["transitions"]),
                "n_transitions": len(r["transitions"]),
                "needs_review": bool(r["needs_review"]), "n_overrides": int(r.get("n_overrides", 0)),
            })
            geoms.append(geom)
    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")


def summarize(gdf, by: list[str]) -> list[dict]:
    out = []
    if len(gdf) == 0:
        return out
    for key, grp in gdf.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        recs = grp.to_dict("records")
        counts = class_counts(recs)
        trans = [t for r in recs for t in json.loads(r["transitions"])]
        row = dict(zip(by, key))
        row.update({
            "footprints": counts["footprints"],
            **{f"current_{c}": counts.get(c, 0) for c in CLASSES},
            f"current_{UNKNOWN}": counts.get(UNKNOWN, 0),
            "pickleball_courts": counts["pickleball_courts"],
            "pickleball_footprints_uncounted": counts["pickleball_footprints_uncounted"],
            "ever_tennis": sum(1 for r in recs if r["ever_tennis"]),
            "tennis_to_hybrid": sum(1 for t in trans if t["from"] == "tennis" and t["to"] == "hybrid"),
            "tennis_or_hybrid_to_pickleball": sum(1 for t in trans if t["from"] in ("tennis", "hybrid") and t["to"] == "pickleball"),
            "tennis_or_hybrid_to_padel": sum(1 for t in trans if t["from"] in ("tennis", "hybrid") and t["to"] == "padel"),
            "tennis_or_hybrid_to_removed": sum(1 for t in trans if t["from"] in ("tennis", "hybrid") and t["to"] == "removed"),
            "needs_review": counts["needs_review"],
            "reviewed_courts": sum(1 for r in recs if r["n_overrides"] > 0),
        })
        out.append(row)
    return out


def export_site_data(gdf, sites: dict[str, dict], chips_dir: Path, site_dir: Path, summary_state: list[dict],
                     jpeg_quality: int, state: str) -> None:
    from PIL import Image

    data_dir = site_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pts = gdf.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # centroid of tiny polygons in lon/lat is fine here
        pts["geometry"] = pts.geometry.centroid
    keep = ["court_id", "site_id", "current_class", "current_confidence", "current_n_courts", "current_year",
            "first_year", "first_class", "ever_tennis", "history", "transitions", "needs_review", "n_overrides", "county"]
    keep = [k for k in keep if k in pts.columns]
    pts[keep + ["geometry"]].to_file(data_dir / "courts_points.geojson", driver="GeoJSON")

    site_index = {}
    for sid, s in sites.items():
        out_dir = data_dir / "chips" / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        images = {}
        for year in s["years"]:
            src = chips_dir / sid / f"{year}.png"
            dst = out_dir / f"{year}.jpg"
            if src.exists() and not dst.exists():
                with Image.open(src) as im:
                    im.convert("RGB").save(dst, "JPEG", quality=jpeg_quality, optimize=True)
            if dst.exists():
                images[str(year)] = f"data/chips/{sid}/{year}.jpg"
        site_index[sid] = {
            "lat": s["lat"], "lon": s["lon"], "years": s["years"], "imagery_dates": s["imagery_dates"],
            "images": images, "n_courts": len(s["courts"]),
            "courts": [{"court_id": r["court_id"], "class": r["current_class"], "obb": r["obb"],
                        "n_courts": r["current_n_courts"], "history": [{"year": h["year"], "class": h["class"], "obb": h["obb"], "n_courts": h["n_courts"]} for h in r["history"]]}
                       for r in s["courts"]],
        }
    write_json_atomic(data_dir / "sites.json", site_index, indent=None)
    headline = summary_state[0] if summary_state else {}
    write_json_atomic(data_dir / "summary.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(), "state": state.upper(),
        "n_sites": len(sites), "n_courts": int(len(gdf)), **{k: v for k, v in headline.items() if k != "state"},
        "classes": CLASSES,
    })
    log.info("site data -> %s", data_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks", type=Path, default=CHANGE_DIR / "court_tracks_raw.json")
    ap.add_argument("--overrides", type=Path, default=REVIEW_DIR / "overrides.csv")
    ap.add_argument("--chips-dir", type=Path, default=CHIPS_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--site-dir", type=Path, default=SITE_DIR)
    ap.add_argument("--state", default="OH")
    ap.add_argument("--flag-conf", type=float, default=0.6)
    ap.add_argument("--no-county", action="store_true")
    ap.add_argument("--no-site", action="store_true")
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    raw = read_json(args.tracks)
    overrides = load_overrides(args.overrides)
    sites = rebuild_records(raw, overrides, args.flag_conf)
    gdf = build_geodataframe(sites, args.state)
    log.info("%d courts across %d sites", len(gdf), len(sites))

    gdf["county"] = None
    if not args.no_county and len(gdf):
        try:
            counties = county_frame(args.state, BOUNDARIES_DIR)
            import geopandas as gpd
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                centroids = gdf.geometry.centroid
            pts = gpd.GeoDataFrame(gdf[["court_id"]], geometry=centroids, crs=gdf.crs)
            joined = gpd.sjoin(pts, counties[["NAME", "GEOID", "geometry"]], how="left", predicate="within")
            joined = joined[~joined.index.duplicated(keep="first")]
            gdf["county"] = joined["NAME"].reindex(gdf.index).values
            gdf["county_geoid"] = joined["GEOID"].reindex(gdf.index).values
        except Exception as e:  # network or file trouble should not block the export
            log.warning("county join skipped: %s", e)
    gdf["county"] = gdf["county"].fillna(UNKNOWN) if len(gdf) else gdf["county"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(args.out_dir / "courts.parquet")
    gdf.to_file(args.out_dir / "courts.geojson", driver="GeoJSON")
    trans_rows = []
    for _, r in gdf.iterrows():
        for t in json.loads(r["transitions"]):
            trans_rows.append({"court_id": r["court_id"], "site_id": r["site_id"], "county": r["county"],
                               **{k: (";".join(v) if isinstance(v, list) else v) for k, v in t.items()}})
    write_csv(args.out_dir / "transitions.csv", trans_rows)
    by_state = summarize(gdf, ["state"])
    write_csv(args.out_dir / "summary_by_state.csv", by_state)
    write_csv(args.out_dir / "summary_by_county.csv", summarize(gdf, ["state", "county"]))
    if by_state:
        s = by_state[0]
        log.info("HEADLINE %s: %d footprints; tennis %d, hybrid %d, pickleball footprints %d (%d courts), padel %d, removed %d, unknown %d; %d need review",
                 s["state"], s["footprints"], s["current_tennis"], s["current_hybrid"], s["current_pickleball"],
                 s["pickleball_courts"], s["current_padel"], s["current_removed"], s["current_unknown"], s["needs_review"])
    log.info("dataset -> %s", args.out_dir)
    if not args.no_site:
        export_site_data(gdf, sites, args.chips_dir, args.site_dir, by_state, args.jpeg_quality, args.state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
