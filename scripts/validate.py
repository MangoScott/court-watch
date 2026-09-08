#!/usr/bin/env python
"""Check pipeline output against sites whose history is known first-hand.

Reads YAML files in data/validation/ (see sawyer_point.yaml), finds the
nearest site in data/change/court_tracks_raw.json, and compares per-year
class counts with the expectations. Exits non-zero if any check fails, so it
can gate the "expand beyond Ohio" decision.

Expectation keys per year (or ``latest``)::

    tennis / hybrid / pickleball / padel / removed   number of court footprints in that class
    pickleball_courts                                sum of individual pickleball courts (n_courts)
    tennis_footprints                                tennis + hybrid footprints

Usage::

    python scripts/validate.py                          # all files in data/validation/
    python scripts/validate.py data/validation/sawyer_point.yaml --tracks data/change/court_tracks_raw.json
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CHANGE_DIR, CLASSES, DATA, log, read_json, setup_logging  # noqa: E402


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_site(tracks: dict, lat: float, lon: float, radius_m: float):
    best, best_d = None, float("inf")
    for sid, s in tracks["sites"].items():
        if s.get("lat") is None:
            continue
        d = haversine_m(lat, lon, s["lat"], s["lon"])
        if d < best_d:
            best, best_d = sid, d
    if best is None or best_d > radius_m:
        return None, best_d
    return best, best_d


def counts_for_year(site: dict, year: int) -> dict[str, int]:
    c = {k: 0 for k in CLASSES}
    c.update({"pickleball_courts": 0, "tennis_footprints": 0})
    for court in site["courts"]:
        obs = next((h for h in court["history"] if h["year"] == year), None)
        if obs is None:
            continue
        cls = obs["class"]
        if cls in c:
            c[cls] += 1
        if cls == "pickleball" and obs.get("n_courts") is not None:
            c["pickleball_courts"] += int(obs.get("n_courts") or 1)
        if cls in ("tennis", "hybrid"):
            c["tennis_footprints"] += 1
    return c


def check(cfg: dict, tracks: dict) -> bool:
    sid, dist = nearest_site(tracks, cfg["lat"], cfg["lon"], cfg.get("radius_m", 300))
    print(f"\n== {cfg['name']} ==")
    if sid is None:
        print(f"  FAIL: no site within {cfg.get('radius_m', 300)} m (nearest {dist:.0f} m). Is 01/02/05/06 done for this area?")
        return False
    site = tracks["sites"][sid]
    print(f"  site {sid} ({dist:.0f} m away), years {site['years']}")
    ok = True
    tol = int(cfg.get("tolerance", 0))
    for year_key, expect in cfg["expect"].items():
        year = max(site["years"]) if year_key == "latest" else int(year_key)
        if year not in site["years"]:
            near = min(site["years"], key=lambda y: abs(y - year)) if site["years"] else None
            print(f"  {year_key}: no imagery for {year}; nearest available year is {near}")
            if near is None:
                ok = False
                continue
            year = near
        got = counts_for_year(site, year)
        for k, want in expect.items():
            g = got.get(k, 0)
            passed = abs(g - int(want)) <= tol
            ok &= passed
            print(f"  {'PASS' if passed else 'FAIL'} {year} {k}: expected {want}, got {g}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*", type=Path)
    ap.add_argument("--tracks", type=Path, default=CHANGE_DIR / "court_tracks_raw.json")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)
    configs = args.configs or sorted((DATA / "validation").glob("*.yaml"))
    if not args.tracks.exists():
        log.error("%s missing; run 06_change_detection.py first", args.tracks)
        return 2
    tracks = read_json(args.tracks)
    all_ok = True
    for p in configs:
        cfg = yaml.safe_load(p.read_text())
        all_ok &= check(cfg, tracks)
    print("\nALL PASS" if all_ok else "\nSOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
