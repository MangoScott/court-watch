# court-watch

Tracks US tennis courts that have been converted to pickleball or padel, removed, or painted over with pickleball lines (hybrids). Built from public aerial imagery. The output is a court-level dataset and a static map site, refreshed roughly once a year when new imagery drops.

## Why this exists

The New York Times did a one-off version of this in September 2025 (Ethan Singer, "How Pickleball Took Over Thousands of Tennis Courts, as Seen From the Sky"). They started from known pickleball locations and counted pickleball courts. We start from the tennis side: every tennis court, what happened to it, and when. The number nobody has published is how many courts are still technically tennis but have pickleball lines painted over them. That hybrid count is the headline.

## Owner

Scott Glasgow. Technical background, tennis player, runs @ScottPlaysTennis. Scott reviews spot-checks and validates results against courts he knows in person. He is not going to hand-label thousands of images, so labeling must be automated with Claude's vision API and then spot-checked.

## Scope for v1

* Ohio only. Validate against Sawyer Point Park in Cincinnati (8 tennis courts in 2019, now 24 pickleball courts with 6 drawn on top of the 3 remaining tennis courts). If the pipeline does not correctly classify Sawyer Point, it is not ready.
* Outdoor courts only.
* Expand to the full US once Ohio numbers look right.

## Court classes

1. `tennis` - clean tennis court, no pickleball lines
2. `hybrid` - tennis court footprint with any visible pickleball lines, even one faded set. Err toward hybrid when unsure between tennis and hybrid.
3. `pickleball` - dedicated pickleball courts, tennis lines gone or footprint reconfigured
4. `padel` - enclosed 20x10m court with glass or mesh walls
5. `removed` - former court location now something else (parking, turf, basketball, empty)

Basketball courts, shade structures, and resurfacing are the common false positives. Do not let basketball courts become tennis detections.

## Pipeline

1. Locations. Query OpenStreetMap Overpass for `leisure=pitch` + `sport=tennis` (and `sport=pickleball`, `sport=padel`) within the target state. Save to GeoPackage with OSM id, geometry, tags, and fetch date.
2. Imagery. For each location, fetch NAIP chips (512px, centered on the court) for every available year via the Microsoft Planetary Computer STAC API (`naip` collection, COGs). Cache chips locally, never re-download. Store imagery date alongside each chip.
3. Labeling. Send chips to the Claude vision API to label the training set. Prompt it with the five class definitions above and ask for oriented bounding boxes plus a confidence. Export to YOLO-OBB format. Write a `spot_check.py` script that samples 50 random labels into an HTML grid for Scott to review.
4. Detection. Train YOLOv8-OBB on the labeled chips. Run on all chips for all years.
5. Change detection. Per location, compare detections across years. Record transitions (e.g. `tennis -> hybrid` in 2021, `hybrid -> pickleball` in 2023). Flag low-confidence transitions for review.
6. Output. GeoParquet and GeoJSON of every court with current class, class history, and imagery dates. Summary CSV by state and county. Static map site (MapLibre, no backend) deployable to GitHub Pages with a before/after image slider per court.

## Principles

* Python. Keep it simple. Prefer scripts in `scripts/` over frameworks.
* Write the README as you go. Every script has a docstring explaining what it does and how to run it.
* Everything cached and resumable. Imagery downloads and API calls will be interrupted; never lose work.
* Store raw model outputs separately from reviewed results. Credibility depends on being able to show the numbers are right.
* Start small, validate, then scale. Never run the full US before Ohio is verified.
* Do not invent counts. If something cannot be determined from imagery, mark it unknown.

## Repo layout

```
court-watch/
  CLAUDE.md
  README.md
  requirements.txt
  scripts/
    01_fetch_osm.py
    02_fetch_naip.py
    03_label_with_claude.py
    04_train_obb.py
    05_detect.py
    06_change_detection.py
    07_export.py
    spot_check.py
  data/          (gitignored except small samples)
  site/          (static map, GitHub Pages)
```

## Conventions

* Run scripts from the repo root: `python scripts/01_fetch_osm.py --state OH`.
* Shared helpers live in `scripts/common.py`, `scripts/tracking.py`, `scripts/claude_labeler.py`. Numbered scripts are thin CLIs over those modules.
* Every per-chip artifact (chip PNG, sidecar JSON, label JSON, detection JSON) is keyed by `site_id` and imagery `year`. A site is a cluster of OSM court features within 60 m; one chip per site per year.
* Raw model output goes in `data/labels/raw/` and `data/detections/raw/`. Human corrections go in `data/review/overrides.csv`. Nothing overwrites raw output.
* Tests run offline: `pytest`. Network-dependent code paths are isolated behind small functions so they can be mocked.
