#!/usr/bin/env python
"""Step 5: run the detector on every chip for every year.

Writes one detection JSON per chip to ``data/detections/raw/<site_id>/<year>.json``
in the same schema as the Claude labels (see common.py), so step 6 does not
care which backend produced them. Existing files are skipped.

Backends::

    --backend yolo    (default) data/models/best.pt from 04_train_obb.py
    --backend claude  label every chip with Claude directly. No training needed;
                      costs roughly $0.01-0.02 per chip. Fine for one state,
                      and useful as a second opinion on the YOLO output.

Usage::

    python scripts/05_detect.py                          # yolo, all chips
    python scripts/05_detect.py --site-id oh_3909750_-8449660
    python scripts/05_detect.py --backend claude --batch   # submit Claude batches
    python scripts/05_detect.py --backend claude --collect # collect them
    python scripts/05_detect.py --conf 0.2 --overwrite    # rerun with a lower threshold
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHIPS_DIR, CLASSES, DETECTIONS_DIR, MODELS_DIR, list_chips, log, normalize_obb, setup_logging,
    write_json_atomic,
)


def detection_path(det_dir: Path, site_id: str, year: int) -> Path:
    return det_dir / "raw" / site_id / f"{year}.json"


def base_record(chip: dict, source: str) -> dict:
    return {
        "site_id": chip["site_id"], "year": chip["year"], "image": str(chip["png"]),
        "width": chip.get("size", 512), "height": chip.get("size", 512),
        "imagery_date": chip.get("imagery_date"), "source": source,
        "unusable": bool(chip.get("low_coverage", False)),
        "chip_notes": "low imagery coverage" if chip.get("low_coverage") else "",
        "detected_at": datetime.now(timezone.utc).isoformat(), "courts": [],
    }


def run_yolo(chips: list[dict], det_dir: Path, weights: Path, conf: float, iou: float, device, batch: int) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed: pip install -r requirements-train.txt")
    if not weights.exists():
        raise SystemExit(f"{weights} missing; run 04_train_obb.py or use --backend claude")
    model = YOLO(str(weights))
    names = model.names  # {id: name}
    source = f"yolo:{weights}"
    for start in range(0, len(chips), batch):
        part = chips[start: start + batch]
        results = model.predict([str(c["png"]) for c in part], imgsz=part[0].get("size", 512), conf=conf, iou=iou,
                                device=device, verbose=False)
        for chip, res in zip(part, results):
            rec = base_record(chip, source)
            h, w = res.orig_shape
            obb = getattr(res, "obb", None)
            if obb is not None and len(obb) > 0:
                pts = obb.xyxyxyxy.cpu().numpy()   # N x 4 x 2 pixels
                cls = obb.cls.cpu().numpy().astype(int)
                confs = obb.conf.cpu().numpy()
                for p, ci, cf in zip(pts, cls, confs):
                    name = names.get(int(ci), CLASSES[int(ci)] if int(ci) < len(CLASSES) else None)
                    norm = normalize_obb([[x / w, y / h] for x, y in p])
                    if name in CLASSES and norm:
                        rec["courts"].append({"class": name, "confidence": round(float(cf), 3),
                                              "obb": [[round(x, 5), round(y, 5)] for x, y in norm], "notes": ""})
            write_json_atomic(detection_path(det_dir, chip["site_id"], chip["year"]), rec)
        log.info("[%d/%d] detected", min(start + batch, len(chips)), len(chips))


def run_claude(chips: list[dict], det_dir: Path, model: str, effort: str, upscale: int, use_batch: bool) -> None:
    import claude_labeler as cl
    import anthropic

    client = anthropic.Anthropic()
    if use_batch:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        bdir = det_dir / "batches"
        bdir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(chips), 2000):
            part = chips[start: start + 2000]
            reqs, index = [], {}
            for chip in part:
                cid = f"{chip['site_id']}__{chip['year']}"
                index[cid] = {k: (str(v) if k == "png" else v) for k, v in chip.items() if k in ("site_id", "year", "png", "imagery_date", "size", "gsd", "low_coverage")}
                params = cl.build_params(chip["png"], chip["year"], chip.get("imagery_date"), model, effort,
                                         chip.get("size", 512), chip.get("gsd", 0.6), upscale=upscale)
                reqs.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**params)))
            b = client.messages.batches.create(requests=reqs)
            write_json_atomic(bdir / f"{b.id}.json", {"batch_id": b.id, "model": model, "n": len(part),
                                                       "submitted_at": datetime.now(timezone.utc).isoformat(),
                                                       "collected": False, "chips": index})
            log.info("submitted batch %s (%d chips); run with --collect later", b.id, len(part))
        return
    for i, chip in enumerate(chips, 1):
        parsed = cl.label_chip(client, chip["png"], chip["year"], chip.get("imagery_date"), model=model,
                               effort=effort, size=chip.get("size", 512), gsd=chip.get("gsd", 0.6), upscale=upscale)
        rec = base_record(chip, parsed["source"])
        rec.update({k: v for k, v in parsed.items() if k != "source"})
        rec["unusable"] = rec["unusable"] or bool(chip.get("low_coverage"))
        write_json_atomic(detection_path(det_dir, chip["site_id"], chip["year"]), rec)
        log.info("[%d/%d] %s %d: %d courts", i, len(chips), chip["site_id"], chip["year"], len(rec["courts"]))


def collect_claude(det_dir: Path, wait: bool) -> None:
    import claude_labeler as cl
    import anthropic
    from common import read_json

    client = anthropic.Anthropic()
    bdir = det_dir / "batches"
    for p in sorted(bdir.glob("msgbatch_*.json")):
        meta = read_json(p)
        if meta.get("collected"):
            continue
        while True:
            b = client.messages.batches.retrieve(meta["batch_id"])
            if b.processing_status == "ended" or not wait:
                break
            log.info("%s: %s", meta["batch_id"], b.processing_status)
            import time
            time.sleep(60)
        if b.processing_status != "ended":
            log.info("%s still %s", meta["batch_id"], b.processing_status)
            continue
        ok = err = 0
        for result in client.messages.batches.results(meta["batch_id"]):
            chip = meta["chips"].get(result.custom_id)
            if chip is None:
                continue
            if result.result.type != "succeeded":
                err += 1
                continue
            parsed = cl.message_to_record(result.result.message, meta["model"])
            rec = base_record(chip, parsed["source"])
            rec.update({k: v for k, v in parsed.items() if k != "source"})
            rec["unusable"] = rec["unusable"] or bool(chip.get("low_coverage"))
            rec["batch_id"] = meta["batch_id"]
            write_json_atomic(detection_path(det_dir, chip["site_id"], int(chip["year"])), rec)
            ok += 1
        meta.update({"collected": True, "succeeded": ok, "errored": err,
                     "collected_at": datetime.now(timezone.utc).isoformat()})
        write_json_atomic(p, meta)
        log.info("%s: %d saved, %d errored", meta["batch_id"], ok, err)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["yolo", "claude"], default="yolo")
    ap.add_argument("--chips-dir", type=Path, default=CHIPS_DIR)
    ap.add_argument("--det-dir", type=Path, default=DETECTIONS_DIR)
    ap.add_argument("--site-id")
    ap.add_argument("--years", help="comma-separated years")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--overwrite", action="store_true")
    # yolo
    ap.add_argument("--weights", type=Path, default=MODELS_DIR / "best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    # claude
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--upscale", type=int, default=2)
    ap.add_argument("--batch", action="store_true", help="claude: submit via Batches API")
    ap.add_argument("--collect", action="store_true", help="claude: collect finished batches")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    if args.collect:
        collect_claude(args.det_dir, args.wait)
        return 0

    chips = list_chips(args.chips_dir)
    if args.site_id:
        chips = [c for c in chips if c["site_id"] == args.site_id]
    if args.years:
        ys = {int(y) for y in args.years.split(",")}
        chips = [c for c in chips if c["year"] in ys]
    if not args.overwrite:
        chips = [c for c in chips if not detection_path(args.det_dir, c["site_id"], c["year"]).exists()]
    if args.limit:
        chips = chips[: args.limit]
    log.info("%d chips to detect with %s", len(chips), args.backend)
    if not chips:
        return 0
    if args.backend == "yolo":
        run_yolo(chips, args.det_dir, args.weights, args.conf, args.iou, args.device, args.batch_size)
    else:
        run_claude(chips, args.det_dir, args.model, args.effort, args.upscale, args.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
