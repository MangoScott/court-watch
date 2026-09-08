#!/usr/bin/env python
"""Step 4: train a YOLOv8-OBB detector on the Claude-labeled chips.

Reads ``data/yolo/dataset.yaml`` (from 03_label_with_claude.py --export-yolo)
and fine-tunes a pretrained oriented-bounding-box model. The best weights are
copied to ``data/models/best.pt`` for 05_detect.py.

Requires ``pip install -r requirements-train.txt`` (ultralytics + torch). A GPU
helps but is not required for the Ohio-sized dataset; on CPU expect a few
hours for 50 epochs on ~1000 chips.

Usage::

    python scripts/04_train_obb.py                       # yolov8n-obb, 50 epochs
    python scripts/04_train_obb.py --model yolov8s-obb.pt --epochs 100 --device 0
    python scripts/04_train_obb.py --resume              # continue an interrupted run

Notes
-----
* imgsz is 512 to match the chips; do not let ultralytics rescale to 640.
* Rotation and flip augmentations are on: courts are north-up in NAIP, but
  the model should not learn that as a rule.
* The ``removed`` class is rare and visually weak in single images. Expect
  poor mAP for it; change detection (step 6) derives removals from
  disappearance rather than from this class.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS_DIR, YOLO_DIR, log, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=YOLO_DIR / "dataset.yaml")
    ap.add_argument("--model", default="yolov8n-obb.pt", help="pretrained OBB weights (n/s/m/l/x)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="cuda index, 'cpu', or None for auto")
    ap.add_argument("--project", type=Path, default=MODELS_DIR / "runs")
    ap.add_argument("--name", default="court_obb")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed: pip install -r requirements-train.txt")
        return 2
    if not args.data.exists():
        log.error("%s missing; run 03_label_with_claude.py --export-yolo first", args.data)
        return 2

    if args.resume:
        last = args.project / args.name / "weights" / "last.pt"
        model = YOLO(str(last))
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(
            data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device=args.device, project=str(args.project), name=args.name, exist_ok=True,
            degrees=180.0, flipud=0.5, fliplr=0.5, mosaic=1.0, scale=0.2,
            patience=15, seed=42, verbose=True,
        )
    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else args.project / args.name
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best, MODELS_DIR / "best.pt")
        log.info("best weights -> %s", MODELS_DIR / "best.pt")
    else:
        log.warning("no best.pt found under %s", run_dir)
    metrics = model.val(data=str(args.data), imgsz=args.imgsz, device=args.device)
    try:
        log.info("val mAP50=%.3f mAP50-95=%.3f", metrics.box.map50, metrics.box.map)
    except AttributeError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
