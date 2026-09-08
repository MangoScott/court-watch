import json

from shapely import affinity
from shapely.geometry import Point, Polygon

from conftest import load_script
from footprints import dedupe_footprints, footprints_for_feature, rectangle_params


def rect(cx, cy, L, W, ang):
    p = Polygon([(-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2)])
    return affinity.translate(affinity.rotate(p, ang, origin=(0, 0)), cx, cy)


def test_single_court_keeps_geometry_and_angle():
    (fp,) = footprints_for_feature(rect(10, 20, 37, 19, 30), "tennis")
    assert abs(fp.length - 37) < 0.1 and abs(fp.width - 19) < 0.1
    assert abs(fp.angle - 30) < 0.5 and not fp.subdivided and not fp.guessed
    assert abs(fp.cx - 10) < 0.01 and abs(fp.cy - 20) < 0.01


def test_bank_is_split_into_standard_courts():
    fps = footprints_for_feature(rect(0, 0, 160, 40, 0), "tennis")
    assert len(fps) == 8 and all(f.subdivided for f in fps)
    # courts' long axis is across the bank (north-south here), centres 20 m apart along x
    xs = sorted(round(f.cx, 1) for f in fps)
    assert abs((xs[1] - xs[0]) - 20.0) < 0.1
    assert all(abs(f.angle - 90) < 0.5 for f in fps)
    assert abs(fps[0].length - 36.6) < 0.01 and abs(fps[0].width - 18.3) < 0.01


def test_grid_and_pickleball_bank_and_node():
    assert len(footprints_for_feature(rect(0, 0, 80, 80, 0), "tennis")) == 8
    pb = footprints_for_feature(rect(0, 0, 60, 20, 0), "pickleball")
    assert len(pb) == 6 and abs(pb[0].length - 18.3) < 0.01
    (node,) = footprints_for_feature(Point(1, 2), "tennis")
    assert node.guessed and abs(node.length - 36.6) < 0.01


def test_oversized_is_capped_and_flagged():
    fps = footprints_for_feature(rect(0, 0, 400, 300, 10), "tennis")
    assert fps and all(f.oversized for f in fps) and len(fps) <= 24


def test_dedupe_prefers_real_polygons():
    individual = footprints_for_feature(rect(-70, 0, 37, 19, 0), "tennis")
    bank = footprints_for_feature(rect(0, 0, 160, 40, 0), "tennis")
    kept = dedupe_footprints(individual + bank)
    assert len(kept) == 8 and not kept[0].subdivided


def test_rectangle_params_degenerate():
    cx, cy, L, W, ang = rectangle_params(Point(3, 4))
    assert (cx, cy, L, W) == (3, 4, 0, 0)


def test_derive_courts_from_osm_fixture():
    from pathlib import Path
    m = load_script("01_fetch_osm")
    data = json.loads((Path(__file__).parent / "fixtures" / "overpass_small.json").read_text())
    feats = m.parse_elements(data["elements"], "2026-01-01")
    feats, sites = m.cluster_sites(feats, "OH")
    courts = m.derive_courts(feats, sites)
    assert len(courts) >= 5
    row = courts.iloc[0]
    ring = json.loads(row["ring"])
    assert len(ring) == 4 and all(len(p) == 2 for p in ring)
    assert row["court_id"].startswith(row["site_id"] + ":f")
    assert set(courts["sport"]) <= {"tennis", "pickleball", "padel"}
