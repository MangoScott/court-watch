# data/

Everything here is generated and gitignored, except `validation/` (hand-written
expectations for known sites) and `samples/` (tiny fixtures).

```
data/
  osm/            01_fetch_osm.py   -> <STATE>.gpkg (layers: features, sites), <STATE>_sites.csv, raw/ Overpass JSON
  chips/          02_fetch_naip.py  -> <site_id>/<year>.png + <year>.json sidecar, stac_items.json, manifest.csv
  labels/         03_label_with_claude.py -> raw/<site_id>_<year>.json (untouched Claude output), batches/
  yolo/           03_label_with_claude.py -> YOLO-OBB dataset (images/, labels/, dataset.yaml)
  models/         04_train_obb.py   -> best.pt and training runs
  detections/     05_detect.py      -> raw/<site_id>/<year>.json (untouched detector output)
  change/         06_change_detection.py -> court_tracks_raw.json
  review/         spot_check.py HTML grids, overrides.csv (human corrections; the only hand-edited file)
  output/         07_export.py      -> courts.parquet, courts.geojson, summary_*.csv, transitions.csv
  boundaries/     cached Census county boundaries
  validation/     expected results for known sites (committed)
```

Raw model output is never modified. Corrections go in `review/overrides.csv` and are
applied at export time, so every published number can be traced back to either a
model output or a named reviewer.
