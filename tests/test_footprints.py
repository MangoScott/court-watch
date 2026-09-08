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
    assert row["court_id"].startswith(row["site_id"] + ":f") and len(row["court_id"].split(":f")[1]) == 6
    assert set(courts["sport"]) <= {"tennis", "pickleball", "padel"}
    assert set(courts["role"]) <= {"court", "pb_child"}
    assert {"parent_id", "n_children", "n_overlay", "derived"} <= set(courts.columns)


def sawyer_like():
    """3 tennis courts (lines, 24 x 12) each with 2 pickleball courts inside, plus
    5 former courts holding 4,4,4,4,2 pickleball courts (lines, 13.4 x 6), all
    in one row 17 m apart, as OSM has Sawyer Point."""
    import math
    from footprints import Footprint
    fps = []
    for i in range(3):
        cx = i * 17.0
        fps.append(Footprint(cx, 0.0, 24.1, 12.0, 90.0, "tennis", 1, 0, osm_ref=f"way/t{i}"))
        for j in (-1, 1):
            fps.append(Footprint(cx + j * 3.2, 0.0, 13.4, 6.0, 90.0, "pickleball", 1, 0, osm_ref=f"way/o{i}{j}"))
    counts = [4, 4, 4, 4, 2]
    for i, n in enumerate(counts):
        cx = 60.0 + i * 17.0
        for k in range(n):
            col, row = k % 2, k // 2
            fps.append(Footprint(cx + (col - 0.5) * 7.0, (row - 0.5) * 15.0 if n > 2 else 0.0, 13.4, 6.0, 90.0,
                                 "pickleball", 1, 0, osm_ref=f"way/p{i}{k}"))
    return fps


def test_aggregate_pickleball_sawyer_point():
    from footprints import aggregate_pickleball
    out = aggregate_pickleball(sawyer_like())
    courts = [f for f in out if f.role == "court"]
    children = [f for f in out if f.role == "pb_child"]
    assert len(courts) == 8 and len(children) == 24
    tennis = [f for f in courts if f.sport == "tennis"]
    assert len(tennis) == 3 and all(f.n_overlay == 2 for f in tennis)
    parents = [f for f in courts if f.sport == "pickleball"]
    assert len(parents) == 5 and sorted(f.n_children for f in parents) == [2, 4, 4, 4, 4]
    assert all(f.derived == "pickleball_cluster" for f in parents)
    assert all(out[c.parent].role == "court" for c in children)


def test_lone_pickleball_courts_stay_courts():
    from footprints import Footprint, aggregate_pickleball
    fps = [Footprint(0, 0, 13.4, 6.0, 90.0, "pickleball", 1, 0), Footprint(8, 0, 13.4, 6.0, 90.0, "pickleball", 1, 0)]
    out = aggregate_pickleball(fps)
    assert len(out) == 2 and all(f.role == "court" and f.n_children == 1 for f in out)
