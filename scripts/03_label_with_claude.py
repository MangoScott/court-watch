#!/usr/bin/env python
"""Step 3: label chips with the Claude vision API and export a YOLO-OBB dataset.

Samples chips from data/chips/, sends each to Claude with the five class
definitions (see claude_labeler.py) and stores the untouched response in
``data/labels/raw/<site_id>_<year>.json``. Chips that already have a raw label
are never re-sent, so the run can be interrupted freely.

Two ways to call the API:

* default: synchronous, one request per chip, results land immediately.
* ``--batch``: submit everything through the Message Batches API at half price.
  Batch ids are saved in ``data/labels/batches/``; run again with ``--collect``
  (any time within 29 days) to pull results into ``raw/``.

Afterwards ``--export-yolo`` (or the standalone run with no new chips) writes
``data/yolo/`` with images/, labels/ and dataset.yaml, split by site so the
same court never appears in both train and val.

Usage::

    python scripts/03_label_with_claude.py --sample 300 --dry-run     # cost estimate only
    python scripts/03_label_with_claude.py --sample 300                # sync labeling
    python scripts/03_label_with_claude.py --sample 2000 --batch       # submit batch
    python scripts/03_label_with_claude.py --collect                   # fetch batch results
    python scripts/03_label_with_claude.py --export-yolo               # (re)build data/yolo
    python scripts/03_label_with_claude.py --site-id oh_3909750_-8449660 --all-years

Sampling: ``--sample N`` draws 60% from each site's latest chip (where hybrids
and pickleball conversions show up) and 40% from earlier years, with sites the
OSM data tags as pickleball or padel always included so the rarer classes
appear in training. Needs ANTHROPIC_API_KEY (or an `ant auth login` profile).
"""
from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHIPS_DIR, CLASSES, LABELS_DIR, OSM_DIR, YOLO_DIR, list_chips, log, read_csv, read_json,
    setup_logging, write_json_atomic, yolo_obb_line,
)
import claude_labeler as cl  # noqa: E402


def label_path(labels_dir: Path, site_id: str, year: int) -> Path:
    return labels_dir / "raw" / f"{site_id}_{year}.json"


def choose_chips(chips: list[dict], n: int | None, seed: int, sites_csv: Path | None,
                 all_years: bool, site_id: str | None) -> list[dict]:
    if site_id:
        chips = [c for c in chips if c["site_id"] == site_id]
    chips = [c for c in chips if not c.get("low_coverage")]
    if all_years or n is None:
        return chips
    rng = random.Random(seed)
    latest_by_site: dict[str, dict] = {}
    for c in chips:
        if c["site_id"] not in latest_by_site or c["year"] > latest_by_site[c["site_id"]]["year"]:
            latest_by_site[c["site_id"]] = c
    latest = list(latest_by_site.values())
    older = [c for c in chips if c is not latest_by_site.get(c["site_id"])]

    must = []
    if sites_csv and Path(sites_csv).exists():
        rare = {r["site_id"] for r in read_csv(sites_csv)
                if str(r.get("has_pickleball", "")).lower() == "true" or str(r.get("has_padel", "")).lower() == "true"}
        must = [c for c in latest if c["site_id"] in rare]
    rest_latest = [c for c in latest if c not in must]
    rng.shuffle(rest_latest)
    rng.shuffle(older)
    n_latest = max(0, int(round(n * 0.6)) - len(must))
    picked = must + rest_latest[:n_latest]
    picked += older[: max(0, n - len(picked))]
    if len(picked) < n:  # not enough older chips; top up with latest
        picked += rest_latest[n_latest: n_latest + (n - len(picked))]
    return picked[:n]


def record_for(chip: dict, parsed: dict) -> dict:
    rec = {
        "site_id": chip["site_id"], "year": chip["year"],
        "image": str(chip["png"]), "width": chip.get("size", 512), "height": chip.get("size", 512),
        "imagery_date": chip.get("imagery_date"), "labeled_at": datetime.now(timezone.utc).isoformat(),
    }
    rec.update(parsed)
    return rec


# ---------------------------------------------------------------------------
# sync + batch labeling
# ---------------------------------------------------------------------------

def label_sync(chips: list[dict], labels_dir: Path, model: str, effort: str, upscale: int) -> None:
    import anthropic
    client = anthropic.Anthropic()
    t0 = time.time()
    n_in = n_out = 0
    for i, chip in enumerate(chips, 1):
        out = label_path(labels_dir, chip["site_id"], chip["year"])
        parsed = cl.label_chip(client, chip["png"], chip["year"], chip.get("imagery_date"),
                               model=model, effort=effort, size=chip.get("size", 512), gsd=chip.get("gsd", 0.6),
                               upscale=upscale)
        write_json_atomic(out, record_for(chip, parsed))
        u = parsed.get("usage") or {}
        n_in += u.get("input_tokens") or 0
        n_out += u.get("output_tokens") or 0
        log.info("[%d/%d] %s %d: %d courts (%s) tokens in=%d out=%d", i, len(chips), chip["site_id"], chip["year"],
                 len(parsed["courts"]), ",".join(sorted({c["class"] for c in parsed["courts"]})) or "-", n_in, n_out)
    log.info("labeled %d chips in %.0fs", len(chips), time.time() - t0)


def label_batch(chips: list[dict], labels_dir: Path, model: str, effort: str, upscale: int, chunk: int = 2000) -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    bdir = labels_dir / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(chips), chunk):
        part = chips[start: start + chunk]
        requests = []
        index = {}
        for chip in part:
            cid = f"{chip['site_id']}__{chip['year']}"
            index[cid] = {"site_id": chip["site_id"], "year": chip["year"], "png": str(chip["png"]),
                          "imagery_date": chip.get("imagery_date"), "size": chip.get("size", 512), "gsd": chip.get("gsd", 0.6)}
            params = cl.build_params(chip["png"], chip["year"], chip.get("imagery_date"), model, effort,
                                     chip.get("size", 512), chip.get("gsd", 0.6), upscale=upscale)
            requests.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**params)))
        batch = client.messages.batches.create(requests=requests)
        write_json_atomic(bdir / f"{batch.id}.json", {
            "batch_id": batch.id, "model": model, "effort": effort, "upscale": upscale,
            "submitted_at": datetime.now(timezone.utc).isoformat(), "n": len(part), "collected": False,
            "chips": index,
        })
        log.info("submitted batch %s with %d requests (%s)", batch.id, len(part), batch.processing_status)
    log.info("run again with --collect to retrieve results (usually within an hour)")


def collect_batches(labels_dir: Path, wait: bool) -> None:
    import anthropic
    client = anthropic.Anthropic()
    bdir = labels_dir / "batches"
    pending = [p for p in sorted(bdir.glob("msgbatch_*.json")) if not read_json(p).get("collected")]
    if not pending:
        log.info("no uncollected batches in %s", bdir)
        return
    for p in pending:
        meta = read_json(p)
        bid = meta["batch_id"]
        while True:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status == "ended":
                break
            log.info("%s: %s (%s processing)", bid, b.processing_status, b.request_counts.processing)
            if not wait:
                break
            time.sleep(60)
        if b.processing_status != "ended":
            continue
        ok = err = 0
        for result in client.messages.batches.results(bid):
            chip = meta["chips"].get(result.custom_id)
            if chip is None:
                continue
            out = label_path(labels_dir, chip["site_id"], chip["year"])
            if result.result.type == "succeeded":
                parsed = cl.message_to_record(result.result.message, meta["model"])
                chip_rec = {"site_id": chip["site_id"], "year": chip["year"], "png": chip["png"],
                            "imagery_date": chip.get("imagery_date"), "size": chip.get("size", 512)}
                rec = record_for(chip_rec, parsed)
                rec["batch_id"] = bid
                write_json_atomic(out, rec)
                ok += 1
            else:
                err += 1
                log.warning("%s: %s -> %s", bid, result.custom_id, result.result.type)
        meta["collected"] = True
        meta["collected_at"] = datetime.now(timezone.utc).isoformat()
        meta["succeeded"], meta["errored"] = ok, err
        write_json_atomic(p, meta)
        log.info("%s: %d saved, %d errored (errored chips will be re-sent next run)", bid, ok, err)


# ---------------------------------------------------------------------------
# YOLO export
# ---------------------------------------------------------------------------

def export_yolo(labels_dir: Path, yolo_dir: Path, val_frac: float = 0.2, min_conf: float = 0.0,
                copy: bool = True) -> dict:
    """Write images/{train,val}, labels/{train,val}, dataset.yaml.

    Split is by site (hash of site_id) so all years of a court share a split.
    Unusable chips and chips with zero courts are kept as negatives (empty
    label file), which teaches the model about basketball courts and parks.
    """
    raw = sorted((labels_dir / "raw").glob("*.json"))
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    for split in ("train", "val"):
        (yolo_dir / "images" / split).mkdir(parents=True)
        (yolo_dir / "labels" / split).mkdir(parents=True)
    stats = {"train": 0, "val": 0, "boxes": {c: 0 for c in CLASSES}, "skipped_unusable": 0}
    for p in raw:
        rec = read_json(p)
        if rec.get("unusable"):
            stats["skipped_unusable"] += 1
            continue
        img = Path(rec["image"])
        if not img.exists():
            log.warning("missing image %s", img)
            continue
        h = int(hashlib.sha1(rec["site_id"].encode()).hexdigest(), 16) % 1000
        split = "val" if h < val_frac * 1000 else "train"
        stem = f"{rec['site_id']}_{rec['year']}"
        dst = yolo_dir / "images" / split / f"{stem}.png"
        if copy:
            shutil.copyfile(img, dst)
        else:
            dst.symlink_to(img.resolve())
        lines = []
        for c in rec.get("courts", []):
            if float(c.get("confidence", 0)) < min_conf:
                continue
            lines.append(yolo_obb_line(c["class"], c["obb"]))
            stats["boxes"][c["class"]] += 1
        (yolo_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[split] += 1
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (yolo_dir / "dataset.yaml").write_text(
        f"# generated by 03_label_with_claude.py on {datetime.now(timezone.utc).date()}\n"
        f"path: {yolo_dir.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    log.info("YOLO dataset: %s", stats)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chips-dir", type=Path, default=CHIPS_DIR)
    ap.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    ap.add_argument("--yolo-dir", type=Path, default=YOLO_DIR)
    ap.add_argument("--sites-csv", type=Path, default=OSM_DIR / "OH_sites.csv", help="for oversampling pickleball/padel sites")
    ap.add_argument("--sample", type=int, help="number of chips to label")
    ap.add_argument("--all-years", action="store_true", help="label every chip (or every chip of --site-id)")
    ap.add_argument("--site-id")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=cl.DEFAULT_MODEL)
    ap.add_argument("--effort", default=cl.DEFAULT_EFFORT, choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--upscale", type=int, default=2, help="resize factor for the image sent to Claude (1 = native)")
    ap.add_argument("--batch", action="store_true", help="use the Message Batches API (50%% cheaper, async)")
    ap.add_argument("--collect", action="store_true", help="collect finished batches into raw/")
    ap.add_argument("--wait", action="store_true", help="with --collect: poll until batches end")
    ap.add_argument("--export-yolo", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true", help="print selection and cost estimate, call nothing")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    if args.collect:
        collect_batches(args.labels_dir, args.wait)

    if args.sample or args.all_years or args.site_id:
        chips = list_chips(args.chips_dir)
        chosen = choose_chips(chips, args.sample, args.seed, args.sites_csv, args.all_years, args.site_id)
        todo = [c for c in chosen if not label_path(args.labels_dir, c["site_id"], c["year"]).exists()]
        log.info("%d chips available, %d selected, %d already labeled, %d to send",
                 len(chips), len(chosen), len(chosen) - len(todo), len(todo))
        est = cl.estimate_cost(len(todo), args.model, args.batch, upscale=args.upscale)
        log.info("estimated cost: $%.2f (%s, %s%s)", est["usd"], args.model, "batch" if args.batch else "sync",
                 f", upscale {args.upscale}x" if args.upscale > 1 else "")
        if args.dry_run:
            for c in todo[:10]:
                print(c["site_id"], c["year"], c["png"])
            print("...")
            return 0
        if todo:
            if args.batch:
                label_batch(todo, args.labels_dir, args.model, args.effort, args.upscale)
            else:
                label_sync(todo, args.labels_dir, args.model, args.effort, args.upscale)

    if args.export_yolo or not (args.sample or args.all_years or args.site_id or args.collect):
        export_yolo(args.labels_dir, args.yolo_dir, args.val_frac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
