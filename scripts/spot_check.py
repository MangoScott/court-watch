#!/usr/bin/env python
"""Render a random sample of labels or detections as an HTML grid for review.

Draws each court's oriented box on its chip, colour-coded by class, and writes
a self-contained HTML page with a verdict form per chip (ok / wrong class /
bad box / missed court / not a court) and a notes field. Verdicts persist in
the browser (localStorage) and the "Export CSV" button downloads them as
``spot_check_verdicts.csv`` in the same shape as ``data/review/overrides.csv``
so accepted corrections can be pasted straight in.

Usage::

    python scripts/spot_check.py                          # 50 random Claude labels
    python scripts/spot_check.py --source detections      # 50 random detector outputs
    python scripts/spot_check.py --n 100 --seed 7 --class hybrid
    python scripts/spot_check.py --site-id oh_3909750_-8449660 --source detections

Output: data/review/spot_check_<source>_<timestamp>/index.html (open in a browser).
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHIPS_DIR, CLASSES, DETECTIONS_DIR, LABELS_DIR, REVIEW_DIR, log, obb_norm_to_pixels, read_json, setup_logging,
)

COLORS = {"tennis": "#2ecc71", "hybrid": "#f39c12", "pickleball": "#e74c3c", "padel": "#9b59b6", "removed": "#7f8c8d"}


def load_records(source: str, labels_dir: Path, det_dir: Path) -> list[dict]:
    recs = []
    if source == "labels":
        files = sorted((labels_dir / "raw").glob("*.json"))
    else:
        files = sorted((det_dir / "raw").glob("*/*.json"))
    for f in files:
        r = read_json(f)
        r["_file"] = str(f)
        recs.append(r)
    return recs


def draw(rec: dict, out_path: Path, scale: int = 1) -> None:
    from PIL import Image, ImageDraw

    with Image.open(rec["image"]) as im:
        im = im.convert("RGB")
        if scale > 1:
            im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        for i, c in enumerate(rec.get("courts", [])):
            pts = obb_norm_to_pixels(c["obb"], im.width, im.height)
            col = COLORS.get(c["class"], "#ffffff")
            d.polygon(pts, outline=col, width=2)
            x, y = min(p[0] for p in pts), min(p[1] for p in pts)
            d.text((x + 2, max(0, y - 12)), f"{i + 1} {c['class'][:2]} {c.get('confidence', 0):.2f}", fill=col)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(out_path, "PNG")


def render_html(recs: list[dict], out_dir: Path, source: str, chips_dir: Path) -> Path:
    cards = []
    for i, r in enumerate(recs):
        sc = chips_dir / r["site_id"] / f"{r['year']}.json"
        lat = lon = None
        if sc.exists():
            m = read_json(sc)
            lat, lon = m.get("center_lat"), m.get("center_lon")
        maps = f'<a href="https://www.google.com/maps/@{lat},{lon},19z/data=!3m1!1e3" target="_blank">satellite</a>' if lat else ""
        courts = "".join(
            f'<li><span class="sw" style="background:{COLORS.get(c["class"], "#fff")}"></span>'
            f'{j + 1}. <b>{html.escape(c["class"])}</b> {c.get("confidence", 0):.2f} '
            f'<small>{html.escape(c.get("notes", "") or "")}</small></li>'
            for j, c in enumerate(r.get("courts", [])))
        key = f"{r['site_id']}_{r['year']}"
        cards.append(f"""
<div class="card" data-key="{key}" data-site="{r['site_id']}" data-year="{r['year']}">
  <img src="img/{key}.png" loading="lazy">
  <div class="meta">
    <div><b>{r['site_id']}</b> {r['year']} <small>{html.escape(str(r.get('imagery_date') or ''))}</small> {maps}</div>
    <div class="notes">{html.escape(r.get('chip_notes', '') or '')}{' <b>UNUSABLE</b>' if r.get('unusable') else ''}</div>
    <ul>{courts or '<li><i>no courts</i></li>'}</ul>
    <div class="verdict">
      <label><input type="radio" name="v{i}" value="ok"> ok</label>
      <label><input type="radio" name="v{i}" value="wrong_class"> wrong class</label>
      <label><input type="radio" name="v{i}" value="bad_box"> bad box</label>
      <label><input type="radio" name="v{i}" value="missed"> missed court</label>
      <label><input type="radio" name="v{i}" value="not_court"> not a court</label>
      <select class="cls"><option value="">correct class…</option>{''.join(f'<option>{c}</option>' for c in CLASSES)}<option>unknown</option></select>
      <input class="note" placeholder="note">
    </div>
  </div>
</div>""")
    legend = " ".join(f'<span class="sw" style="background:{COLORS[c]}"></span>{c}' for c in CLASSES)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>spot check {source}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#ddd}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(540px,1fr));gap:16px}}
.card{{background:#1c1c1c;border-radius:8px;padding:8px;display:flex;gap:10px}}
.card img{{width:512px;height:512px;image-rendering:auto;border-radius:4px;flex:none}}
.meta{{font-size:13px;flex:1;min-width:0}} .meta ul{{padding-left:16px;margin:6px 0}} .notes{{color:#aaa;margin:4px 0}}
.sw{{display:inline-block;width:10px;height:10px;margin-right:4px;border-radius:2px}}
.verdict label{{display:inline-block;margin-right:8px}} .verdict input.note{{width:95%;margin-top:4px}}
.card.done{{outline:2px solid #2ecc71}} .card.bad{{outline:2px solid #e74c3c}}
header{{display:flex;gap:16px;align-items:center;margin-bottom:12px;position:sticky;top:0;background:#111;padding:8px 0}}
button{{padding:6px 12px}} a{{color:#6cf}}
</style></head><body>
<header><h2 style="margin:0">Spot check: {source} ({len(recs)} chips)</h2><div>{legend}</div>
<span id="progress"></span><button id="export">Export CSV</button><button id="clear">Clear</button></header>
<div class="grid">{''.join(cards)}</div>
<script>
const KEY='spotcheck:{source}:{out_dir.name}';
let state=JSON.parse(localStorage.getItem(KEY)||'{{}}');
function save(){{localStorage.setItem(KEY,JSON.stringify(state));
  const n=Object.values(state).filter(v=>v.verdict).length;document.getElementById('progress').textContent=n+'/{len(recs)} reviewed';}}
document.querySelectorAll('.card').forEach(card=>{{
  const k=card.dataset.key; const s=state[k]||{{}};
  card.querySelectorAll('input[type=radio]').forEach(r=>{{ if(s.verdict===r.value) r.checked=true;
    r.onchange=()=>{{state[k]={{...state[k],verdict:r.value,site_id:card.dataset.site,year:card.dataset.year}};paint(card,r.value);save();}}; }});
  const sel=card.querySelector('.cls'); sel.value=s.cls||''; sel.onchange=()=>{{state[k]={{...state[k],cls:sel.value}};save();}};
  const note=card.querySelector('.note'); note.value=s.note||''; note.oninput=()=>{{state[k]={{...state[k],note:note.value}};save();}};
  if(s.verdict) paint(card,s.verdict);
}});
function paint(card,v){{card.classList.toggle('done',v==='ok');card.classList.toggle('bad',v&&v!=='ok');}}
document.getElementById('export').onclick=()=>{{
  let csv='site_id,track_id,year,class,reviewer,note,verdict\\n';
  for(const [k,v] of Object.entries(state)){{ if(!v.verdict) continue;
    const esc=x=>'"'+String(x||'').replace(/"/g,'""')+'"';
    csv+=[esc(v.site_id),'',esc(v.year),esc(v.cls),'scott',esc(v.note),esc(v.verdict)].join(',')+'\\n'; }}
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{{type:'text/csv'}}));a.download='spot_check_verdicts.csv';a.click();}};
document.getElementById('clear').onclick=()=>{{if(confirm('Clear all verdicts?')){{state={{}};save();location.reload();}}}};
save();
</script></body></html>"""
    out = out_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["labels", "detections"], default="labels")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--class", dest="cls", choices=CLASSES, help="only chips containing this class")
    ap.add_argument("--site-id")
    ap.add_argument("--min-conf", type=float, default=None, help="only chips with a court below this confidence")
    ap.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    ap.add_argument("--det-dir", type=Path, default=DETECTIONS_DIR)
    ap.add_argument("--chips-dir", type=Path, default=CHIPS_DIR)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--scale", type=int, default=1, help="upscale factor for the drawn image")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    recs = load_records(args.source, args.labels_dir, args.det_dir)
    if args.site_id:
        recs = [r for r in recs if r["site_id"] == args.site_id]
    if args.cls:
        recs = [r for r in recs if any(c["class"] == args.cls for c in r.get("courts", []))]
    if args.min_conf is not None:
        recs = [r for r in recs if any(c.get("confidence", 1) < args.min_conf for c in r.get("courts", []))]
    recs = [r for r in recs if Path(r["image"]).exists()]
    if not recs:
        log.error("no %s records found", args.source)
        return 1
    rng = random.Random(args.seed)
    sample = rng.sample(recs, min(args.n, len(recs)))
    sample.sort(key=lambda r: (r["site_id"], r["year"]))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (REVIEW_DIR / f"spot_check_{args.source}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in sample:
        draw(r, out_dir / "img" / f"{r['site_id']}_{r['year']}.png", args.scale)
    manifest = [{"site_id": r["site_id"], "year": r["year"], "file": r["_file"]} for r in sample]
    (out_dir / "sample.json").write_text(json.dumps(manifest, indent=1))
    page = render_html(sample, out_dir, args.source, args.chips_dir)
    log.info("wrote %s (%d chips)", page, len(sample))
    print(page)
    return 0


if __name__ == "__main__":
    sys.exit(main())
