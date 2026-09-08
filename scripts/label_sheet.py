#!/usr/bin/env python
"""Record labels for one contact sheet into data/labels/manual/labels.csv.

Usage::

    python scripts/label_sheet.py 7 t t h pb t ? t t x t t t h t t t t t pb t --labeler claude-code

Give exactly one token per crop on the sheet, in order 1..N:
  t = tennis, h = hybrid, pb = pickleball, pd = padel, x = removed,
  ? = unknown / cannot tell (row is written with class "unknown" and skipped by training)
Optional ``--notes "3: faint lines" "12: tree cover"`` attach notes by crop number.
Rows for a crop_id already in the CSV are replaced.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS_DIR  # noqa: E402

CODES = {"t": "tennis", "h": "hybrid", "pb": "pickleball", "pd": "padel", "x": "removed", "?": "unknown"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet", type=int)
    ap.add_argument("codes", nargs="+")
    ap.add_argument("--labeler", default="claude-code")
    ap.add_argument("--notes", nargs="*", default=[])
    ap.add_argument("--sheets-dir", type=Path, default=LABELS_DIR / "sheets")
    ap.add_argument("--labels", type=Path, default=LABELS_DIR / "manual" / "labels.csv")
    args = ap.parse_args()

    meta = json.loads((args.sheets_dir / f"sheet_{args.sheet:03d}.json").read_text())
    crops = meta["crops"]
    if len(args.codes) != len(crops):
        sys.exit(f"sheet {args.sheet} has {len(crops)} crops, got {len(args.codes)} codes")
    notes = {}
    for n in args.notes:
        k, _, v = n.partition(":")
        notes[int(k)] = v.strip()
    existing = {}
    fields = ["crop_id", "class", "labeler", "note"]
    if args.labels.exists():
        for r in csv.DictReader(open(args.labels)):
            if r.get("crop_id"):
                existing[r["crop_id"]] = r
    for c, code in zip(crops, args.codes):
        if code not in CODES:
            sys.exit(f"unknown code {code!r}")
        existing[c["crop_id"]] = {"crop_id": c["crop_id"], "class": CODES[code], "labeler": args.labeler,
                                  "note": notes.get(c["n"], "")}
    args.labels.parent.mkdir(parents=True, exist_ok=True)
    with open(args.labels, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in existing.values():
            w.writerow({k: r.get(k, "") for k in fields})
    from collections import Counter
    print(f"sheet {args.sheet}: {dict(Counter(CODES[c] for c in args.codes))}; total labeled rows {len(existing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
