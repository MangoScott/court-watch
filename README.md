# court-watch

What happened to every tennis court in Ohio, from the sky.

The pipeline finds tennis, pickleball and padel courts in OpenStreetMap, pulls a
NAIP aerial chip for each one for every year of imagery, labels the chips with
Claude's vision API, trains a YOLOv8-OBB detector on those labels, tracks each
court footprint across years, and publishes a court-level dataset plus a static
map with a before/after slider. The headline number is how many courts are still
technically tennis but have pickleball lines painted on them (`hybrid`).

Status: pipeline written and unit-tested end to end on synthetic data. It has
not yet been run against live Overpass, Planetary Computer, or the Claude API
(the development sandbox had no access to them). The first real run is the
Ohio run described below, and the gate before anything else is
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

The map is served free by GitHub Pages at
**https://mangoscott.github.io/court-watch/**. `.github/workflows/pages.yml`
redeploys it on every push to the default branch. Until real results are
committed to `site/data/`, it deploys a clearly labelled synthetic demo.

`.github/workflows/pipeline.yml` runs the whole pipeline on GitHub's servers
from the Actions tab ("Run pipeline" -> "Run workflow"). It needs one thing
set up once: the repository secret `ANTHROPIC_API_KEY` (Settings -> Secrets
and variables -> Actions -> New repository secret). Each run fetches the
validation sites plus the next `limit` sites, labels every chip with Claude,
runs change detection and validation, exports, commits `site/data/` so the map
updates, and saves all of `data/` as a workflow artifact so the next run
resumes instead of re-downloading. GitHub jobs are capped at 6 hours, so the
full state is done in chunks by raising `offset` run by run.

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

# 3. Label a training sample with Claude. --dry-run prints the cost first.
python scripts/03_label_with_claude.py --sample 400 --dry-run
python scripts/03_label_with_claude.py --sample 400 --batch     # half price, async
python scripts/03_label_with_claude.py --collect --wait          # pull results
python scripts/spot_check.py                                    # 50 labels -> HTML grid
python scripts/03_label_with_claude.py --export-yolo             # data/yolo/

# 4. Train (skip 4 and use `05_detect.py --backend claude` to label every chip
#    directly instead; for one state this costs tens of dollars, not hundreds)
python scripts/04_train_obb.py --epochs 50

# 5. Detect on every chip for every year
python scripts/05_detect.py                          # yolo
python scripts/05_detect.py --backend claude --batch # or Claude, via batches
python scripts/spot_check.py --source detections

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

**Two detector backends, one schema.** Claude labels and YOLO detections are
written in the same JSON shape (see the docstring in `scripts/common.py`), so
change detection does not care which produced them. For Ohio it is reasonable
to skip training and run `05_detect.py --backend claude --batch` on every chip;
YOLO matters when scaling to the whole country.

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

## Costs (rough)

Claude labeling with the default settings (Opus 5, image upscaled 2x, medium
effort) is about $0.02 per chip synchronous or $0.01 through the Batches API.
Ohio has on the order of 2-4k tennis sites and 7-9 NAIP years, so labeling
every chip directly is in the low hundreds of dollars; a 400-chip training
sample is a few dollars. `--dry-run` prints the estimate before anything is
sent. Pass `--model claude-sonnet-5` or `--upscale 1` to cut it.

## Layout

```
scripts/
  common.py               paths, classes, detection JSON schema, chip geometry helpers
  tracking.py             cross-year matching, transitions, overrides (pure functions)
  claude_labeler.py       prompt, output schema, request/response handling
  01_fetch_osm.py         Overpass -> data/osm/<STATE>.gpkg (+ _sites.csv)
  02_fetch_naip.py        Planetary Computer STAC -> data/chips/
  03_label_with_claude.py sample + label -> data/labels/raw, export data/yolo/
  04_train_obb.py         YOLOv8-OBB -> data/models/best.pt
  05_detect.py            yolo or claude -> data/detections/raw/
  06_change_detection.py  -> data/change/court_tracks_raw.json, review_queue.csv
  07_export.py            overrides -> data/output/, site/data/
  spot_check.py           random sample -> HTML review grid
  validate.py             known-site checks (the Ohio gate)
  make_demo.py            synthetic data so the site can be previewed
site/                     static MapLibre map (index.html, app.js, style.css)
tests/                    offline pytest suite with a synthetic Sawyer Point
data/                     gitignored except validation/ and README.md
```

Every script's `--help` is its documentation.

## Known gaps

* Not yet run on live data. Expect small fixes in `01` (Overpass response
  quirks) and `02` (tile seams, odd CRSs) on first contact.
* Claude's box coordinates are approximate. The spot check is there to measure
  how approximate; if boxes are too loose for YOLO training, the OSM court
  polygons (`data/osm/OH.gpkg`, layer `features`) can seed the geometry and
  Claude can classify only.
* Sites wider than about 260 m (`oversized` in `OH_sites.csv`) do not fit in
  one chip; v1 accepts the crop and flags them.
* County summaries need one Census download (cached); `--no-county` skips it.
