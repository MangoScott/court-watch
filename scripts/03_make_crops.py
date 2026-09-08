#!/usr/bin/env python
"""Step 3 (free path): build labeling contact sheets and the training crop set.

No model is involved. Court footprints come from OSM (``data/osm/<STATE>_courts.csv``,
step 1) and imagery from the chips (step 2). Each court in each year is cut
out as a 160 x 320 crop with the court upright (see crops.py).

Two modes:

``--sheets N``   Sample N crops that are not yet labeled and lay them out as
                 numbered contact sheets (20 per sheet) in data/labels/sheets/.
                 A person, or Claude Code looking at the images, records one
                 class per number in data/labels/manual/labels.csv::

                     crop_id,class,labeler,note
                     oh_3909750_-8449660:f03:2023,hybrid,scott,faint lines

                 Sampling favours the newest year of each site and sites OSM
                 tags as pickleball/padel so all classes show up.

``--export``     Regenerate the crops for every labeled row into
                 data/crops/train/<class>/<crop_id>.jpg for 04_train_classifier.py.
                 Rows labeled ``unknown`` go to class ``unusable`` so the model
                 learns to abstain on trees, shadow and blur.

Usage::

    python scripts/03_make_crops.py --state OH --sheets 30      # 600 crops to label
    python scripts/03_make_crops.py --export
    python scripts/03_make_crops.py --state OH --sheets 10 --seed 3 --sport pickleball

crop_id is ``<court_id>:<year>``. Chips and sheets are only read/written under
data/, nothing is downloaded.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHIPS_DIR, CLASSES, DATA, LABELS_DIR, OSM_DIR, list_chips, log, read_csv, read_json, setup_logging, write_csv,
)
from crops import CROP_H, CROP_W, contact_sheet, crop_from_chip  # noqa: E402

CROPS_DIR = DATA / "crops"
PER_SHEET = 20


def load_courts(courts_csv: Path) -> dict[str, list[dict]]:
    """site_id -> list of court rows (ring parsed)."""
    by_site: dict[str, list[dict]] = {}
    for r in read_csv(courts_csv):
        if r.get("role", "court") != "court":
            continue  # pickleball children are folded into their parent footprint
        r["ring"] = json.loads(r["ring"])
        by_site.setdefault(r["site_id"], []).append(r)
    return by_site


def load_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["crop_id"]: r for r in read_csv(path) if r.get("crop_id")}


def crop_id(court_id: str, year: int) -> str:
    return f"{court_id}:{year}"


def parse_crop_id(cid: str) -> tuple[str, str, int]:
    site_id, fk, year = cid.rsplit(":", 2)
    return site_id, f"{site_id}:{fk}", int(year)


def all_candidates(chips: list[dict], courts: dict[str, list[dict]]) -> list[dict]:
    out = []
    for c in chips:
        if c.get("low_coverage"):
            continue
        for court in courts.get(c["site_id"], []):
            out.append({"crop_id": crop_id(court["court_id"], c["year"]), "chip": c, "court": court})
    return out


def sample_candidates(cands: list[dict], n: int, seed: int, labeled: set[str], sport: str | None) -> list[dict]:
    rng = random.Random(seed)
    cands = [c for c in cands if c["crop_id"] not in labeled and (not sport or c["court"]["sport"] == sport)]
    latest_year = {}
    for c in cands:
        latest_year[c["chip"]["site_id"]] = max(latest_year.get(c["chip"]["site_id"], 0), c["chip"]["year"])
    latest = [c for c in cands if c["chip"]["year"] == latest_year[c["chip"]["site_id"]]]
    older = [c for c in cands if c["chip"]["year"] != latest_year[c["chip"]["site_id"]]]
    def is_rare(c):
        court = c["court"]
        return court["sport"] in ("pickleball", "padel") or int(court.get("n_overlay") or 0) > 0
    rare = [c for c in latest if is_rare(c)]
    common_latest = [c for c in latest if c not in rare]
    rng.shuffle(rare); rng.shuffle(common_latest); rng.shuffle(older)
    picked = rare[: max(1, n // 5)]
    picked += common_latest[: max(0, int(n * 0.6) - len(picked))]
    picked += older[: max(0, n - len(picked))]
    if len(picked) < n:
        picked += (common_latest + rare)[len(picked) - len(older):][: n - len(picked)]
    return picked[:n]


def make_sheets(picked: list[dict], out_dir: Path, chips_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("sheet_*.json"))
    start = len(existing) + 1
    n_sheets = 0
    for s in range(0, len(picked), PER_SHEET):
        part = picked[s: s + PER_SHEET]
        images, index = [], []
        for i, c in enumerate(part, 1):
            chip = c["chip"]
            im, _ = crop_from_chip(chip["png"], read_json(chip["sidecar"]), c["court"]["ring"])
            images.append(im)
            index.append({"n": i, "crop_id": c["crop_id"], "site_id": chip["site_id"], "year": chip["year"],
                          "sport_osm": c["court"]["sport"], "imagery_date": chip.get("imagery_date"),
                          "lat": c["court"]["lat"], "lon": c["court"]["lon"],
                          "subdivided": c["court"]["subdivided"], "guessed": c["court"]["guessed"],
                          "n_children": c["court"].get("n_children", ""), "n_overlay": c["court"].get("n_overlay", "")})
        sheet_no = start + n_sheets
        contact_sheet(images).save(out_dir / f"sheet_{sheet_no:03d}.jpg", "JPEG", quality=88)
        (out_dir / f"sheet_{sheet_no:03d}.json").write_text(json.dumps(
            {"sheet": sheet_no, "created_at": datetime.now(timezone.utc).isoformat(), "crops": index}, indent=1))
        n_sheets += 1
    return n_sheets


def export_training(labels: dict[str, dict], chips_dir: Path, courts: dict[str, list[dict]], out_dir: Path) -> dict:
    chips = {(c["site_id"], c["year"]): c for c in list_chips(chips_dir)}
    court_by_id = {court["court_id"]: court for cs in courts.values() for court in cs}
    stats = {c: 0 for c in CLASSES + ["unusable"]}
    stats["skipped"] = 0
    for cid, row in labels.items():
        cls = (row.get("class") or "").strip()
        if cls == "unknown":
            cls = "unusable"   # "cannot tell" crops teach the model to abstain
        if cls not in CLASSES + ["unusable"]:
            stats["skipped"] += 1
            continue
        site_id, court_id, year = parse_crop_id(cid)
        chip, court = chips.get((site_id, year)), court_by_id.get(court_id)
        if chip is None or court is None:
            stats["skipped"] += 1
            continue
        dst = out_dir / "train" / cls / f"{cid.replace(':', '__')}.jpg"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            im, _ = crop_from_chip(chip["png"], read_json(chip["sidecar"]), court["ring"])
            im.save(dst, "JPEG", quality=92)
        stats[cls] += 1
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="OH")
    ap.add_argument("--courts", type=Path, help="courts CSV (default data/osm/<STATE>_courts.csv)")
    ap.add_argument("--chips-dir", type=Path, default=CHIPS_DIR)
    ap.add_argument("--labels", type=Path, default=LABELS_DIR / "manual" / "labels.csv")
    ap.add_argument("--sheets-dir", type=Path, default=LABELS_DIR / "sheets")
    ap.add_argument("--crops-dir", type=Path, default=CROPS_DIR)
    ap.add_argument("--sheets", type=int, help="number of contact sheets to create (20 crops each)")
    ap.add_argument("--sport", choices=["tennis", "pickleball", "padel"], help="only courts OSM tags with this sport")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", action="store_true", help="write labeled crops to data/crops/train/<class>/")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    courts_csv = args.courts or (OSM_DIR / f"{args.state.upper()}_courts.csv")
    courts = load_courts(courts_csv)
    labels = load_labels(args.labels)
    log.info("%d sites with footprints, %d labeled crops", len(courts), len(labels))

    if args.sheets:
        chips = list_chips(args.chips_dir)
        cands = all_candidates(chips, courts)
        already = set(labels) | {c["crop_id"] for j in args.sheets_dir.glob("sheet_*.json") for c in read_json(j)["crops"]} if args.sheets_dir.exists() else set(labels)
        picked = sample_candidates(cands, args.sheets * PER_SHEET, args.seed, already, args.sport)
        n = make_sheets(picked, args.sheets_dir, args.chips_dir)
        log.info("%d candidate crops, %d sheets written to %s", len(cands), n, args.sheets_dir)
    if args.export:
        stats = export_training(labels, args.chips_dir, courts, args.crops_dir)
        log.info("training crops: %s -> %s", stats, args.crops_dir / "train")
    if not args.sheets and not args.export:
        ap.error("nothing to do: pass --sheets N and/or --export")
    return 0


if __name__ == "__main__":
    sys.exit(main())
