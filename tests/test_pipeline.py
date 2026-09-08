"""End-to-end over the synthetic Sawyer Point site: 06 -> 07 -> validate -> spot_check -> yolo export."""
import csv
import json
import subprocess
import sys
from pathlib import Path

from conftest import ROOT, SITE_LAT, load_script

PY = sys.executable
S = ROOT / "scripts"


def run(*args):
    r = subprocess.run([PY, *map(str, args)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def test_change_detection_export_validate_spotcheck(sawyer_site, tmp_path):
    change = tmp_path / "change"
    run(S / "06_change_detection.py", "--det-dir", sawyer_site["dets"], "--chips-dir", sawyer_site["chips"], "--out-dir", change)
    tracks = json.loads((change / "court_tracks_raw.json").read_text())
    site = tracks["sites"][sawyer_site["site_id"]]
    assert len(site["courts"]) == 8
    assert site["courts"][0]["ring_lonlat"] is not None
    assert abs(site["courts"][0]["lat"] - SITE_LAT) < 0.002
    assert (change / "review_queue.csv").exists()

    # validate.py against the committed Sawyer Point expectations
    run(S / "validate.py", ROOT / "data/validation/sawyer_point.yaml", "--tracks", change / "court_tracks_raw.json")

    # export, no county download, into temp dirs
    out = tmp_path / "output"
    site_dir = tmp_path / "site"
    run(S / "07_export.py", "--tracks", change / "court_tracks_raw.json", "--chips-dir", sawyer_site["chips"],
        "--out-dir", out, "--site-dir", site_dir, "--no-county")
    gj = json.loads((out / "courts.geojson").read_text())
    assert len(gj["features"]) == 8
    assert (out / "courts.parquet").exists()
    with open(out / "summary_by_state.csv") as f:
        (row,) = list(csv.DictReader(f))
    assert row["current_hybrid"] == "3" and row["current_pickleball"] == "5" and row["pickleball_courts"] == "18"
    assert row["tennis_to_hybrid"] == "3" and row["tennis_or_hybrid_to_pickleball"] == "5"
    assert (site_dir / "data" / "courts_points.geojson").exists()
    sites_json = json.loads((site_dir / "data" / "sites.json").read_text())
    assert set(sites_json[sawyer_site["site_id"]]["images"]) == {"2019", "2021", "2023"}
    assert (site_dir / "data" / "chips" / sawyer_site["site_id"] / "2023.jpg").exists()

    # overrides change the export but not the raw tracks
    ov = tmp_path / "overrides.csv"
    tid = site["courts"][7]["track_id"]
    ov.write_text("site_id,track_id,year,class,reviewer,note\n"
                  f"{sawyer_site['site_id']},{tid},2023,tennis,scott,lines repainted\n")
    run(S / "07_export.py", "--tracks", change / "court_tracks_raw.json", "--chips-dir", sawyer_site["chips"],
        "--out-dir", out, "--site-dir", site_dir, "--no-county", "--overrides", ov)
    with open(out / "summary_by_state.csv") as f:
        (row,) = list(csv.DictReader(f))
    assert row["current_hybrid"] == "2" and row["current_tennis"] == "1" and row["reviewed_courts"] == "1"
    raw_again = json.loads((change / "court_tracks_raw.json").read_text())
    assert raw_again == tracks

    # spot check grid over detections
    sc = tmp_path / "spot"
    r = run(S / "spot_check.py", "--source", "detections", "--det-dir", sawyer_site["dets"], "--chips-dir",
            sawyer_site["chips"], "--n", "2", "--seed", "1", "--out-dir", sc)
    html = (sc / "index.html").read_text()
    assert html.count('class="card"') == 2 and (sc / "img").exists()


def test_yolo_export(sawyer_site, tmp_path):
    m = load_script("03_label_with_claude")
    labels = tmp_path / "labels"
    (labels / "raw").mkdir(parents=True)
    for f in (sawyer_site["dets"] / "raw" / sawyer_site["site_id"]).glob("*.json"):
        rec = json.loads(f.read_text())
        (labels / "raw" / f"{rec['site_id']}_{rec['year']}.json").write_text(json.dumps(rec))
    stats = m.export_yolo(labels, tmp_path / "yolo")
    assert stats["train"] + stats["val"] == 3
    assert stats["boxes"]["pickleball"] == 18 and stats["boxes"]["tennis"] == 16
    yaml_text = (tmp_path / "yolo" / "dataset.yaml").read_text()
    assert "0: tennis" in yaml_text and "4: removed" in yaml_text
    label_files = list((tmp_path / "yolo" / "labels").rglob("*.txt"))
    assert len(label_files) == 3
    first = label_files[0].read_text().splitlines()[0].split()
    assert len(first) == 9


def test_choose_chips_prefers_latest(sawyer_site):
    m = load_script("03_label_with_claude")
    from common import list_chips
    chips = list_chips(sawyer_site["chips"])
    assert len(chips) == 3
    picked = m.choose_chips(chips, 1, 0, None, False, None)
    assert picked[0]["year"] == 2023
    assert len(m.choose_chips(chips, None, 0, None, True, sawyer_site["site_id"])) == 3


def test_validate_skips_expectations_needing_newer_imagery(sawyer_site, tmp_path):
    change = tmp_path / "change"
    run(S / "06_change_detection.py", "--det-dir", sawyer_site["dets"], "--chips-dir", sawyer_site["chips"], "--out-dir", change)
    cfg = tmp_path / "v.yaml"
    cfg.write_text("name: t\nlat: 39.10279\nlon: -84.49653\nradius_m: 300\nexpect:\n  2019: {tennis: 8}\n"
                   "  latest:\n    requires_imagery_from: 2030\n    tennis: 0\n")
    r = run(S / "validate.py", cfg, "--tracks", change / "court_tracks_raw.json")
    assert "SKIP latest" in r.stdout and "ALL PASS" in r.stdout


def test_rare_only_and_recent_sampling(sawyer_site):
    m = load_script("03_make_crops")
    from common import list_chips
    chips = list_chips(sawyer_site["chips"])
    courts = {sawyer_site["site_id"]: [
        {"court_id": "a", "sport": "tennis", "n_overlay": "0", "ring": [], "subdivided": "", "guessed": ""},
        {"court_id": "b", "sport": "pickleball", "n_overlay": "0", "ring": [], "subdivided": "", "guessed": ""},
        {"court_id": "c", "sport": "tennis", "n_overlay": "2", "ring": [], "subdivided": "", "guessed": ""},
    ]}
    cands = m.all_candidates(chips, courts)
    assert len(cands) == 9
    rare = m.sample_candidates(cands, 100, 0, set(), None, rare_only=True)
    assert {c["court"]["court_id"] for c in rare} == {"b", "c"}
    recent = m.sample_candidates(cands, 100, 0, set(), None, recent=1)
    assert {c["chip"]["year"] for c in recent} == {2023}
