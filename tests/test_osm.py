import json
from pathlib import Path

from conftest import load_script

FIX = Path(__file__).parent / "fixtures" / "overpass_small.json"


def test_parse_and_cluster():
    m = load_script("01_fetch_osm")
    data = json.loads(FIX.read_text())
    feats = m.parse_elements(data["elements"], "2026-01-01")
    assert len(feats) == 6
    assert set(feats["sport"]) == {"tennis", "pickleball", "padel"}
    assert feats["indoor_flag"].sum() == 1
    feats, sites = m.cluster_sites(feats, "OH")
    assert len(sites) == 3
    bank = sites[sites["n_features"] == 4].iloc[0]
    assert bank["has_tennis"] and bank["has_pickleball"] and not bank["oversized"]
    assert bank["site_id"].startswith("oh_")
    assert feats["site_id"].notna().all()


def test_primary_sport_ignores_table_tennis():
    m = load_script("01_fetch_osm")
    assert m.primary_sport({"sport": "table_tennis"}) == "unknown"
    assert m.primary_sport({"sport": "tennis;table_tennis"}) == "tennis"
    assert m.primary_sport({"sport": "paddle_tennis"}) == "tennis"
    assert m.primary_sport({"sport": "Pickleball"}) == "pickleball"


def test_query_mentions_state_and_sports():
    m = load_script("01_fetch_osm")
    q = m.overpass_query("OH", None)
    assert 'US-OH' in q and "tennis|pickleball|padel" in q and "out geom" in q
    q2 = m.overpass_query("OH", (39.0, -84.6, 39.2, -84.4))
    assert "(39.0,-84.6,39.2,-84.4)" in q2 and "area" not in q2
