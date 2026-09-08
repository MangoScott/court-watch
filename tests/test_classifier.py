"""Smoke test for the free-path detector: train the crop classifier on a tiny
synthetic set, then classify the synthetic Sawyer Point chips. Skipped when
torch is not installed (the Tests workflow installs it)."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from conftest import ROOT, SITE_LAT, SITE_LON, court_obb, load_script, make_site, sawyer_detections, tennis_centres, TENNIS_W, TENNIS_H  # noqa: E402

S = ROOT / "scripts"


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def write_courts_csv(path: Path, site_id: str, chips: Path):
    """Court footprints CSV in the shape 01_fetch_osm.py writes, from the fixture geometry."""
    cd = load_script("06_change_detection")
    from common import read_json
    sidecar = read_json(chips / site_id / "2019.json")
    rows = []
    for k, (cx, cy) in enumerate(tennis_centres()):
        ring = cd.obb_to_lonlat(court_obb(cx, cy, TENNIS_W, TENNIS_H), sidecar)[:4]
        rows.append({"court_id": f"{site_id}:f{k:02d}", "site_id": site_id, "osm_ref": "way/1", "sport": "tennis",
                     "lat": ring[0][1], "lon": ring[0][0], "length_m": 26, "width_m": 12, "angle_deg": 90,
                     "n_in_feature": 8, "subdivided": "True", "guessed": "False", "oversized": "False",
                     "ring": json.dumps(ring)})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_sheets_train_classify(tmp_path):
    site_id = "oh_3909750_-8449660"
    chips, dets = make_site(tmp_path, site_id, SITE_LAT, SITE_LON, sawyer_detections())
    courts_csv = tmp_path / "OH_courts.csv"
    write_courts_csv(courts_csv, site_id, chips)

    # contact sheets
    sheets = tmp_path / "sheets"
    run(S / "03_make_crops.py", "--courts", courts_csv, "--chips-dir", chips, "--sheets-dir", sheets,
        "--labels", tmp_path / "labels.csv", "--sheets", "1")
    assert (sheets / "sheet_001.jpg").exists()
    idx = json.loads((sheets / "sheet_001.json").read_text())["crops"]
    assert len(idx) == 20 and idx[0]["crop_id"].count(":") == 2

    # labels from the synthetic truth (fixture draws tennis/hybrid green, pickleball blue)
    truth = {}
    for year, courts in sawyer_detections().items():
        for k, (cx, cy) in enumerate(tennis_centres()):
            classes = {c["class"] for c in courts if abs((c["obb"][0][0] + c["obb"][2][0]) / 2 - cx) < 0.02}
            truth[f"{site_id}:f{k:02d}:{year}"] = "pickleball" if "pickleball" in classes else "tennis"
    labels = tmp_path / "labels.csv"
    with open(labels, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop_id", "class", "labeler", "note"])
        for cid, cls in truth.items():
            w.writerow([cid, cls, "test", ""])
    crops_dir = tmp_path / "crops"
    run(S / "03_make_crops.py", "--courts", courts_csv, "--chips-dir", chips, "--labels", labels,
        "--crops-dir", crops_dir, "--export")
    assert len(list((crops_dir / "train").glob("*/*.jpg"))) == 24

    # train briefly (val split is by site, so with one site everything is train; that is fine for a smoke test)
    model = tmp_path / "classifier.pt"
    run(S / "04_train_classifier.py", "--crops-dir", crops_dir / "train", "--out", model, "--epochs", "2",
        "--batch", "8", "--workers", "0", "--no-pretrained")
    assert model.exists() and (tmp_path / "classifier_report.json").exists()

    # classify all chips with the classifier backend
    det_out = tmp_path / "det_out"
    run(S / "05_detect.py", "--backend", "classifier", "--chips-dir", chips, "--det-dir", det_out,
        "--classifier", model, "--courts", courts_csv, "--batch-size", "8")
    rec = json.loads((det_out / "raw" / site_id / "2023.json").read_text())
    assert len(rec["courts"]) == 8
    c = rec["courts"][0]
    assert c["class"] in ("tennis", "hybrid", "pickleball", "padel", "removed")
    assert "probs" in c and c["court_id"].startswith(site_id)
    assert all(0 <= v <= 1 for pt in c["obb"] for v in pt)
    # pickleball footprints carry unknown court counts
    for court in rec["courts"]:
        if court["class"] == "pickleball":
            assert court["n_courts"] is None
