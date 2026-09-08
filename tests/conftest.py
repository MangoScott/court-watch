"""Pytest fixtures. The synthetic site lives in synthetic.py so make_demo.py can use it without pytest."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))          # tests/ (synthetic.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from synthetic import *  # noqa: E402,F401,F403  (ROOT, load_script, court_obb, make_site, ... re-exported)
from synthetic import SITE_LAT, SITE_LON, make_site, sawyer_detections  # noqa: E402


@pytest.fixture
def sawyer_site(tmp_path: Path):
    site_id = "oh_3909750_-8449660"
    chips, dets = make_site(tmp_path, site_id, SITE_LAT, SITE_LON, sawyer_detections())
    return {"root": tmp_path, "site_id": site_id, "chips": chips, "dets": dets}
