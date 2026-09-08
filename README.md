# court-watch

What happened to every tennis court in Ohio, from the sky.

The pipeline finds tennis, pickleball and padel courts in OpenStreetMap, pulls a
NAIP aerial chip for each one for every year of imagery, cuts every court out
as an upright crop using the OSM footprint, classifies each crop with a small
CNN trained on a few hundred hand-labeled crops, tracks each court footprint
across years, and publishes a court-level dataset plus a static map with a
before/after slider. The headline number is how many courts are still
technically tennis but have pickleball lines painted on them (`hybrid`).

**It costs nothing to run.** Imagery (USDA NAIP via Microsoft's Planetary
Computer) and locations (OpenStreetMap) are free; the classifier trains and
runs on CPU on GitHub's free Actions runners; the map is on GitHub Pages.
Labeling is a few hundred crops on contact sheets, done by Claude Code (covered
by a Claude subscription) or by a person. The Claude vision API path
(`03_label_with_claude.py`, `--backend claude`) is still in the repo as an
optional paid upgrade, but nothing depends on it.

Status: pipeline written and unit-tested end to end on synthetic data. It has
not yet been run against live Overpass or the Planetary Computer (the
development sandbox had no access to them). The first real run is the Ohio
run described below, and the gate before anything else is
`scripts/validate.py` passing on Sawyer Point Park.

## Court classes

| class | meaning |
|---|---|
| `tennis` | clean tennis court, no pickleball lines |
| `hybrid` | tennis footprint with any visible pickleball lines, even one faded set (when in doubt between tennis and hybrid, hybrid) |
| `pickleball` | dedicated pickleball courts, tennis lines gone or footprint reconfigured |
| `padel` | enclosed 20 x 10 m court with glass or mesh walls |
| `removed` | former court location now something else |

Basketball courts are the main false positive. The labeling prompt names them explicitly and the YOLO dataset keeps chips with zero courts as negatives.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # steps 1-3, 6-7, spot check, tests
pip install -r requirements-train.txt      # steps 4-5 (ultralytics + torch), optional
export ANTHROPIC_API_KEY=...               # or `ant auth login`
pytest                                     # offline tests, ~5 s
```

Preview the map site with synthetic data before spending anything:

```bash
python scripts/make_demo.py
python -m http.server -d site 8000         # http://localhost:8000
```

## Hosting and automation

The site is served free by GitHub Pages at
**https://mangoscott.github.io/court-watch/**: a results page (headline
numbers, changes by year, before/after imagery of every court that changed,
county table, downloads) and a map (`map.html`) with a per-court slider. `.github/workflows/pages.yml`
redeploys it on every push to the default branch. Until real results are
committed to `site/data/`, it deploys a clearly labelled synthetic demo.

`.github/workflows/pipeline.yml` runs the whole pipeline on GitHub's servers
from the Actions tab ("Run pipeline" -> "Run workflow"). No secrets needed.
Each run fetches the validation sites plus the next `limit` sites. If fewer
than `min_labels` crops are labeled yet, it makes contact sheets in
`data/labels/sheets/`, commits them, and stops; once
`data/labels/manual/labels.csv` has enough rows it trains the classifier,
classifies every chip, runs change detection and validation, exports, and
commits `site/data/` so the map updates. All of `data/` is saved as a workflow
artifact so the next run resumes instead of re-downloading. GitHub jobs are
capped at 6 hours, so the full state is done in chunks by raising `offset`
run by run.

`.github/workflows/tests.yml` runs the offline test suite on every push.

## Running Ohio (by hand)

Every step is resumable: rerun the same command after an interruption and it
picks up where it stopped. Nothing is ever re-downloaded or re-labeled.

```bash
# 1. Locations from OSM (one Overpass query, cached in data/osm/raw/)
python scripts/01_fetch_osm.py --state OH

# 2. Imagery. Start with the Sawyer Point site to check the chips look right.
#    Find its site_id in data/osm/OH_sites.csv (nearest to 39.0975, -84.4966),
#    then fetch everything.
python scripts/02_fetch_naip.py --lat 39.0975 --lon -84.4966 --site-id sawyer_point_test
python scripts/02_fetch_naip.py --state OH --workers 4

# 3. Contact sheets of court crops to label (20 per sheet), then fill in
#    data/labels/manual/labels.csv (crop_id,class,labeler,note) and export.
python scripts/03_make_crops.py --state OH --sheets 30
python scripts/03_make_crops.py --export

# 4. Train the crop classifier (CPU is fine, minutes)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python scripts/04_train_classifier.py

# 5. Classify every court in every year
python scripts/05_detect.py --backend classifier
python scripts/spot_check.py --source detections

#    Paid alternative (needs ANTHROPIC_API_KEY): Claude labels or detects directly.
#    python scripts/03_label_with_claude.py --sample 400 --dry-run
#    python scripts/05_detect.py --backend claude --batch

# 6. Change detection -> data/change/court_tracks_raw.json + review_queue.csv
python scripts/06_change_detection.py

# The gate. Must pass before touching another state.
python scripts/validate.py

# 7. Apply reviewer overrides, export dataset + site
python scripts/07_export.py
```

Then commit `site/` (its `data/` folder is rebuilt by step 7) and serve it
with GitHub Pages. It is plain HTML, MapLibre from a CDN, basemap from
OpenFreeMap; no keys, no backend.

## How the pieces fit

```
OSM features ──01──▶ sites (60 m clusters) ──02──▶ chips/<site>/<year>.png (+ .json sidecar)
                                                      │
                                    ┌─────────────────┴──────────────────┐
                                    ▼ 03 (sample)                        ▼ 05
                       labels/raw/<site>_<year>.json           detections/raw/<site>/<year>.json
                                    │                                    │
                                    ▼ 03 --export-yolo                   ▼ 06
                          yolo/ ──04──▶ models/best.pt ──▶ 05     change/court_tracks_raw.json
                                                                         │  + review/overrides.csv
                                                                         ▼ 07
                                             output/courts.{parquet,geojson}, summary_*.csv, site/data/
```

**Sites, not features.** OSM maps some banks of courts as one polygon and others
as one polygon per court. `01_fetch_osm.py` clusters features within 60 m into
a site and everything downstream is keyed by `site_id` + `year`. One chip per
site per year, so a court is never counted twice from overlapping chips.

**Fixed grid per site.** `02_fetch_naip.py` resamples every year onto the same
512 px, 0.6 m/px, north-up UTM grid centred on the site. Older 1 m NAIP is
upsampled (the sidecar records `source_gsd`). Because the grid never changes,
`tracking.py` can match courts across years in pixel space.

**Footprints from OSM, classes from a classifier.** `footprints.py` turns each
OSM court polygon into one rotated rectangle per court, splitting banks by
standard court dimensions (36.6 x 18.3 m tennis, 18.3 x 9.1 m pickleball) and
flagging what it had to guess. `crops.py` cuts each footprint out of a chip as
an upright 160 x 320 crop. The classifier only has to answer "which of the five
classes is this court?", which is a much easier problem than finding courts
from scratch, so a few hundred labels are enough.

OSM reflects today's geometry, so a converted tennis court shows up as four
small pickleball polygons. `aggregate_pickleball` folds contiguous pickleball
courts back into tennis-sized parent footprints (so the same footprint is
classified as tennis in 2019 and pickleball in 2023) and records the OSM
pickleball count on the parent. Pickleball courts OSM draws inside a tennis
court are attached to it as an overlay count, which is the hybrid signature.
At Sawyer Point this yields exactly 3 tennis footprints with 2 overlays each
and 5 pickleball parents holding 18 courts. Where the classifier sees
pickleball on a footprint OSM knows nothing about, the count is reported as
unknown rather than guessed.

**Three detector backends, one schema.** Classifier, YOLO-OBB and Claude
detections are written in the same JSON shape (see the docstring in
`scripts/common.py`), so change detection does not care which produced them.

**Tracking.** For each site, detections are matched year to year by overlap.
A tennis footprint that becomes four pickleball courts stays one track with a
`tennis -> pickleball` transition and `n_courts = 4`, so `pickleball_courts` in
the summaries counts individual pickleball courts while `current_pickleball`
counts footprints. A track with no detection in a usable year becomes
`removed` (inferred, flagged); if it is detected again later with the same
class the gap is treated as a detector miss instead. Reversions such as
`hybrid -> tennis` are kept but flagged.

**Raw vs reviewed.** `labels/raw`, `detections/raw` and `change/court_tracks_raw.json`
are never edited. Corrections go in `data/review/overrides.csv`:

```
site_id,track_id,year,class,reviewer,note,n_courts
oh_3909750_-8449660,c003,2023,hybrid,scott,lines visible in person,
```

`07_export.py` applies them at export time and records `n_overrides` per court,
so every published number is either a model output or a named person's call.
`spot_check.py` produces a grid with a verdict form whose CSV export is in this
shape.

**Unknowns stay unknown.** A chip marked unusable (clouds, no data) is skipped
for that year rather than counted as evidence; a court whose latest year is an
inferred observation is exported as `unknown` rather than guessed.

## Validation

`data/validation/sawyer_point.yaml` encodes what Scott knows about Sawyer Point
Park in Cincinnati: 8 tennis courts in 2019; today 3 hybrids and 18 dedicated
pickleball courts on the other 5 footprints. `scripts/validate.py` finds the
nearest site in the change-detection output and checks the counts per year.
The lat/lon in that file is approximate; adjust it after step 1 if the nearest
site is not the courts. Add more YAML files for other courts Scott knows.

## Costs

Zero with the default (classifier) path. The optional Claude API path costs
roughly $0.01-0.02 per chip; `03_label_with_claude.py --dry-run` prints the
estimate before anything is sent.

## Layout

```
scripts/
  common.py               paths, classes, detection JSON schema, chip geometry helpers
  tracking.py             cross-year matching, transitions, overrides (pure functions)
  footprints.py           OSM polygons -> individual court rectangles (pure geometry)
  crops.py                chip + footprint -> upright court crop, contact sheets
  claude_labeler.py       (optional, paid) prompt and response handling for the Claude API
  01_fetch_osm.py         Overpass -> data/osm/<STATE>.gpkg (+ _sites.csv, _courts.csv)
  02_fetch_naip.py        Planetary Computer STAC -> data/chips/
  03_make_crops.py        contact sheets to label; labeled crops -> data/crops/train/
  03_label_with_claude.py (optional, paid) Claude labels -> data/labels/raw, data/yolo/
  04_train_classifier.py  crop classifier -> data/models/classifier.pt
  04_train_obb.py         (optional) YOLOv8-OBB -> data/models/best.pt
  05_detect.py            classifier (default), yolo or claude -> data/detections/raw/
  06_change_detection.py  -> data/change/court_tracks_raw.json, review_queue.csv
  07_export.py            overrides -> data/output/, site/data/
  spot_check.py           random sample -> HTML review grid
  validate.py             known-site checks (the Ohio gate)
  make_demo.py            synthetic data so the site can be previewed
site/                     static site: index.html + results.js (results), map.html + app.js (map), style.css, results.css
tests/                    offline pytest suite with a synthetic Sawyer Point
data/                     gitignored except validation/ and README.md
```

Every script's `--help` is its documentation.

## Known gaps

* Not yet run on live data. Expect small fixes in `01` (Overpass response
  quirks) and `02` (tile seams, odd CRSs) on first contact.
* Court geometry is only as good as OSM. Courts mapped as a single node get a
  guessed north-up footprint (flagged `guessed`); banks are split by
  arithmetic (flagged `subdivided`). Courts not in OSM at all are invisible
  to this pipeline; that undercount is documented, not estimated.
* Pickleball court counts inside converted footprints are unknown in the
  classifier path (`pickleball_footprints_uncounted` in the summaries). The
  Claude backend fills them in.
* Sites wider than about 260 m (`oversized` in `OH_sites.csv`) do not fit in
  one chip; v1 accepts the crop and flags them.
* County summaries need one Census download (cached); `--no-county` skips it.
