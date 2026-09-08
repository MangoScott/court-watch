#!/usr/bin/env python
"""Generate synthetic demo data and build the site so it can be previewed
without downloading imagery or calling any API.

Creates a fake Sawyer Point (8 tennis courts -> 3 hybrid + 18 pickleball) plus
a couple of other sites under data/demo/, runs 06_change_detection.py and
07_export.py against them, and writes site/data/. Then::

    python scripts/make_demo.py
    python -m http.server -d site 8000     # open http://localhost:8000

Delete site/data/ (or rerun 07_export.py on real data) to replace it.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from synthetic import TENNIS_H, TENNIS_W, court_obb, make_site, sawyer_detections  # noqa: E402
from common import SITE_DIR, make_site_id  # noqa: E402


def main() -> int:
    demo = ROOT / "data" / "demo"
    if demo.exists():
        shutil.rmtree(demo)
    sites = {
        make_site_id("OH", 39.0975, -84.4966): (39.0975, -84.4966, sawyer_detections()),
        make_site_id("OH", 39.9612, -82.9988): (39.9612, -82.9988, {   # Columbus: two courts, one removed
            2019: [{"class": "tennis", "confidence": 0.9, "obb": court_obb(0.4, 0.5, TENNIS_W, TENNIS_H), "notes": ""},
                   {"class": "tennis", "confidence": 0.9, "obb": court_obb(0.4 + 40 / 512, 0.5, TENNIS_W, TENNIS_H), "notes": ""}],
            2023: [{"class": "tennis", "confidence": 0.9, "obb": court_obb(0.4, 0.5, TENNIS_W, TENNIS_H), "notes": ""}],
            2025: [{"class": "tennis", "confidence": 0.9, "obb": court_obb(0.4, 0.5, TENNIS_W, TENNIS_H), "notes": ""}],
        }),
        make_site_id("OH", 41.4993, -81.6944): (41.4993, -81.6944, {   # Cleveland: padel build and a low-confidence hybrid
            2023: [{"class": "tennis", "confidence": 0.55, "obb": court_obb(0.5, 0.5, TENNIS_W, TENNIS_H), "notes": "shadow"}],
            2025: [{"class": "hybrid", "confidence": 0.5, "obb": court_obb(0.5, 0.5, TENNIS_W, TENNIS_H), "notes": "faint lines"},
                   {"class": "padel", "confidence": 0.9, "obb": court_obb(0.7, 0.5, 17 / 512, 34 / 512), "notes": ""}],
        }),
    }
    for sid, (lat, lon, dets) in sites.items():
        make_site(demo, sid, lat, lon, dets, source="demo:synthetic")
    py = sys.executable
    subprocess.check_call([py, ROOT / "scripts" / "06_change_detection.py", "--det-dir", demo / "detections",
                           "--chips-dir", demo / "chips", "--out-dir", demo / "change"])
    (SITE_DIR / "data").mkdir(exist_ok=True)
    shutil.rmtree(SITE_DIR / "data" / "chips", ignore_errors=True)
    subprocess.check_call([py, ROOT / "scripts" / "07_export.py", "--tracks", demo / "change" / "court_tracks_raw.json",
                           "--chips-dir", demo / "chips", "--out-dir", demo / "output", "--site-dir", SITE_DIR, "--no-county"])
    import json
    summary_path = SITE_DIR / "data" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["demo"] = True
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"demo data in {demo}; site data in {SITE_DIR / 'data'}. Serve with: python -m http.server -d site 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
