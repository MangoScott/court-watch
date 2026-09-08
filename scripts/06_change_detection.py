#!/usr/bin/env python
"""Step 6: compare detections across years and record what happened to each court.

For every site with detections in ``data/detections/raw/`` this matches court
footprints across years (see tracking.py), infers removals from
disappearance, and writes one record per court footprint with its full class
history and transitions. Output is *raw*: model results only, no human
corrections. Corrections are applied in step 7.

Outputs (data/change/)::

    court_tracks_raw.json   {"sites": {site_id: {"lat","lon","years","courts":[...]}}}
    court_tracks_raw.csv    one row per court footprint (flat, for eyeballing)
    transitions_raw.csv     one row per class change, with flag reasons
    review_queue.csv        flagged transitions and low-confidence courts, most suspicious first

Usage::

    python scripts/06_change_detection.py
    python scripts/06_change_detection.py --site-id oh_3909750_-8449660 -v
    python scripts/06_change_detection.py --det-dir data/detections --flag-conf 0.7
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHANGE_DIR, CHIPS_DIR, DETECTIONS_DIR, log, obb_norm_to_pixels, pixel_to_world, read_json, setup_logging,
    write_csv, write_json_atomic,
)
from tracking import summarize_track, track_site  # noqa: E402


def load_detections(det_dir: Path, site_id: str | None = None) -> dict[str, dict[int, dict]]:
    raw = det_dir / "raw"
    out: dict[str, dict[int, dict]] = {}
    if not raw.exists():
        return out
    for site_path in sorted(p for p in raw.iterdir() if p.is_dir()):
        if site_id and site_path.name != site_id:
            continue
        per_year = {}
        for f in sorted(site_path.glob("*.json")):
            if f.stem.isdigit():
                per_year[int(f.stem)] = read_json(f)
        if per_year:
            out[site_path.name] = per_year
    return out


def obb_to_lonlat(obb: list[list[float]], sidecar: dict) -> list[list[float]] | None:
    """Normalized OBB -> lon/lat ring using the chip's affine transform + CRS."""
    try:
        from pyproj import Transformer
    except ImportError:  # pragma: no cover
        return None
    tf = sidecar.get("transform")
    crs = sidecar.get("crs")
    if not tf or not crs:
        return None
    size = int(sidecar.get("size", 512))
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    ring = []
    for col, row in obb_norm_to_pixels(obb, size, size):
        x, y = pixel_to_world(tf, col, row)
        lon, lat = to_ll.transform(x, y)
        ring.append([round(lon, 7), round(lat, 7)])
    ring.append(ring[0])
    return ring


def process_site(site_id: str, per_year: dict[int, dict], chips_dir: Path, flag_conf: float) -> dict:
    tracks = track_site(per_year)
    records = [summarize_track(t, flag_conf) for t in tracks]
    # georeference using the sidecar of the latest year that has one
    sidecar = None
    lat = lon = None
    for year in sorted(per_year, reverse=True):
        sc = chips_dir / site_id / f"{year}.json"
        if sc.exists():
            sidecar = read_json(sc)
            lat, lon = sidecar.get("center_lat"), sidecar.get("center_lon")
            break
    for r in records:
        r["site_id"] = site_id
        r["court_id"] = f"{site_id}:{r['track_id']}"
        r["ring_lonlat"] = obb_to_lonlat(r["obb"], sidecar) if sidecar else None
        if r["ring_lonlat"]:
            xs = [p[0] for p in r["ring_lonlat"][:-1]]
            ys = [p[1] for p in r["ring_lonlat"][:-1]]
            r["lon"], r["lat"] = round(sum(xs) / 4, 7), round(sum(ys) / 4, 7)
        else:
            r["lon"], r["lat"] = lon, lat
    years = sorted(y for y, d in per_year.items())
    return {
        "site_id": site_id, "lat": lat, "lon": lon,
        "years": years,
        "unusable_years": sorted(y for y, d in per_year.items() if d.get("unusable")),
        "imagery_dates": {y: per_year[y].get("imagery_date") for y in years},
        "sources": sorted({per_year[y].get("source", "") for y in years}),
        "courts": records,
    }


def flat_rows(sites: dict[str, dict]):
    court_rows, trans_rows = [], []
    for sid, s in sites.items():
        for r in s["courts"]:
            court_rows.append({
                "court_id": r["court_id"], "site_id": sid, "lat": r.get("lat"), "lon": r.get("lon"),
                "current_class": r["current_class"], "current_confidence": r["current_confidence"],
                "current_n_courts": r["current_n_courts"], "current_year": r["current_year"],
                "first_year": r["first_year"], "first_class": r["first_class"], "ever_tennis": r["ever_tennis"],
                "n_transitions": len(r["transitions"]), "needs_review": r["needs_review"],
                "history": " > ".join(f"{h['year']}:{h['class']}" for h in r["history"]),
            })
            for t in r["transitions"]:
                trans_rows.append({"court_id": r["court_id"], "site_id": sid, **{k: (";".join(v) if isinstance(v, list) else v) for k, v in t.items()}})
    return court_rows, trans_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det-dir", type=Path, default=DETECTIONS_DIR)
    ap.add_argument("--chips-dir", type=Path, default=CHIPS_DIR)
    ap.add_argument("--out-dir", type=Path, default=CHANGE_DIR)
    ap.add_argument("--site-id")
    ap.add_argument("--flag-conf", type=float, default=0.6, help="flag transitions below this confidence")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    dets = load_detections(args.det_dir, args.site_id)
    log.info("%d sites with detections", len(dets))
    sites = {}
    for sid, per_year in dets.items():
        sites[sid] = process_site(sid, per_year, args.chips_dir, args.flag_conf)
        if args.verbose:
            for r in sites[sid]["courts"]:
                log.debug("%s %s: %s", sid, r["track_id"], " > ".join(f"{h['year']}:{h['class']}" for h in r["history"]))

    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "flag_conf": args.flag_conf,
           "det_dir": str(args.det_dir), "sites": sites}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.out_dir / "court_tracks_raw.json", out)
    court_rows, trans_rows = flat_rows(sites)
    write_csv(args.out_dir / "court_tracks_raw.csv", court_rows)
    write_csv(args.out_dir / "transitions_raw.csv", trans_rows)
    queue = sorted([t for t in trans_rows if str(t["flagged"]) == "True"], key=lambda t: t["confidence"])
    write_csv(args.out_dir / "review_queue.csv", queue)

    n_courts = len(court_rows)
    by_class: dict[str, int] = {}
    for r in court_rows:
        by_class[r["current_class"]] = by_class.get(r["current_class"], 0) + 1
    log.info("%d court footprints; current classes %s; %d transitions (%d flagged)",
             n_courts, by_class, len(trans_rows), len(queue))
    log.info("wrote %s", args.out_dir / "court_tracks_raw.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
