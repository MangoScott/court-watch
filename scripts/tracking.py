"""Match court detections across years and derive class histories.

Pure functions over the detection JSON described in ``common.py``. No I/O, so
this is unit-tested offline and shared by 06_change_detection.py (raw tracks)
and 07_export.py (tracks after human overrides are applied).

Because every chip for a site is resampled onto the same ground grid
(02_fetch_naip.py), matching can be done in pixel space: a court that did not
move sits on the same pixels in every year.

Algorithm, per site
-------------------
1. Years are processed in order. Each *track* is one court footprint through
   time. For each year, every detection is matched to the track whose last
   observed polygon overlaps it (IoU >= ``iou_min``) or contains its centroid.
2. If several detections match one track and they are all pickleball/padel,
   the footprint was subdivided: they are merged into one observation with
   ``n_courts`` set (a tennis court that became four pickleball courts is one
   track with a ``tennis -> pickleball`` transition and ``n_courts = 4``).
3. A track with no match in a year that has a usable chip gets an inferred
   ``removed`` observation. If the court is detected again later with the
   same class as before, the gap is treated as a detector miss and filled in
   (``interpolated = True``) rather than a removal and rebuild.
4. Transitions are recorded wherever consecutive observations differ.
   Reversions (anything back to ``tennis``, ``removed`` -> anything) and any
   step with confidence below ``flag_conf`` are flagged for review.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

try:  # scripts/ may or may not be on sys.path depending on how we are run
    from common import CLASSES, UNKNOWN, obb_norm_to_pixels
except ImportError:  # pragma: no cover
    from scripts.common import CLASSES, UNKNOWN, obb_norm_to_pixels  # type: ignore

SUBDIVIDABLE = {"pickleball", "padel"}
# A detection with this class means "this court could not be seen this year"
# (trees, shadow, clouds). It masks the footprint for the year: no observation
# is recorded and the absence is not treated as evidence of removal.
UNUSABLE = "unusable"
# Transitions that are physically unlikely; when the model reports one we keep
# it but flag it, because it is more often a misclassification than a rebuild.
REVERSION_TARGETS = {"tennis"}


@dataclass
class Observation:
    year: int
    cls: str
    confidence: float
    obb: list[list[float]]          # normalized image coords
    n_courts: int | None = 1          # None = footprint is pickleball but the court count is unknown
    inferred: bool = False          # removed because nothing was detected
    interpolated: bool = False      # gap filled between two matching detections
    imagery_date: str | None = None
    notes: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "year": self.year, "class": self.cls, "confidence": round(self.confidence, 3),
            "obb": self.obb, "n_courts": self.n_courts, "inferred": self.inferred,
            "interpolated": self.interpolated, "imagery_date": self.imagery_date,
            "notes": self.notes, "source": self.source,
        }


@dataclass
class Track:
    track_id: str
    observations: list[Observation] = field(default_factory=list)

    @property
    def last(self) -> Observation:
        return self.observations[-1]

    def last_detected(self) -> Observation | None:
        for o in reversed(self.observations):
            if not o.inferred:
                return o
        return None


def obb_polygon(obb: list[list[float]], width: int, height: int) -> Polygon:
    poly = Polygon(obb_norm_to_pixels(obb, width, height))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def iou(a: Polygon, b: Polygon) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    if inter <= 0:
        return 0.0
    return inter / (a.area + b.area - inter)


def merged_obb(polys: list[Polygon], width: int, height: int) -> list[list[float]]:
    hull = unary_union(polys).minimum_rotated_rectangle
    coords = list(hull.exterior.coords)[:4]
    if len(coords) < 4:  # degenerate union, fall back to bbox
        minx, miny, maxx, maxy = unary_union(polys).bounds
        coords = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    return [[x / width, y / height] for x, y in coords]


def track_site(
    detections_by_year: dict[int, dict],
    iou_min: float = 0.3,
    centroid_buffer_px: float = 2.0,
) -> list[Track]:
    """Build tracks for one site.

    ``detections_by_year`` maps year -> detection JSON (see common.py). Years
    whose chip is marked ``unusable`` are skipped entirely: they neither add
    detections nor count as evidence of removal.
    """
    years = sorted(y for y, d in detections_by_year.items() if not d.get("unusable"))
    tracks: list[Track] = []
    counter = itertools.count(1)

    for year in years:
        det = detections_by_year[year]
        w, h = int(det.get("width", 512)), int(det.get("height", 512))
        source = det.get("source", "")
        imagery_date = det.get("imagery_date")
        all_courts = [c for c in det.get("courts", []) if c.get("obb")]
        courts = [c for c in all_courts if c.get("class") in CLASSES]
        masks = [obb_polygon(c["obb"], w, h) for c in all_courts if c.get("class") == UNUSABLE]
        polys = [obb_polygon(c["obb"], w, h) for c in courts]
        used = [False] * len(courts)
        masked: set[int] = set()   # track indices that are unobservable this year
        for ti, tr in enumerate(tracks):
            ref = tr.last_detected() or tr.last
            ref_poly = obb_polygon(ref.obb, w, h)
            if any(iou(ref_poly, m) >= iou_min or m.buffer(centroid_buffer_px).contains(ref_poly.centroid) for m in masks):
                masked.add(ti)

        # 1. candidate matches per track
        for tr in tracks:
            ref = tr.last_detected() or tr.last
            ref_poly = obb_polygon(ref.obb, w, h)
            ref_buf = ref_poly.buffer(centroid_buffer_px)
            cand = []
            for i, (c, p) in enumerate(zip(courts, polys)):
                if used[i]:
                    continue
                score = iou(ref_poly, p)
                if score >= iou_min or ref_buf.contains(p.centroid):
                    cand.append((score, i))
            if not cand:
                continue
            cand.sort(reverse=True)
            cand_classes = {courts[i]["class"] for _, i in cand}
            if len(cand) > 1 and cand_classes <= SUBDIVIDABLE:
                idxs = [i for _, i in cand]
                cls = max(cand_classes, key=lambda k: sum(1 for i in idxs if courts[i]["class"] == k))
                conf = sum(float(courts[i].get("confidence", 0)) for i in idxs) / len(idxs)
                tr.observations.append(Observation(
                    year=year, cls=cls, confidence=conf,
                    obb=merged_obb([polys[i] for i in idxs], w, h),
                    n_courts=len(idxs), imagery_date=imagery_date, source=source,
                    notes="; ".join(filter(None, (courts[i].get("notes", "") for i in idxs)))[:500],
                ))
                for i in idxs:
                    used[i] = True
            else:
                _, i = cand[0]
                c = courts[i]
                tr.observations.append(Observation(
                    year=year, cls=c["class"], confidence=float(c.get("confidence", 0)),
                    obb=c["obb"], n_courts=_n_courts(c),
                    imagery_date=imagery_date, source=source, notes=c.get("notes", "") or "",
                ))
                used[i] = True

        # 2. unmatched detections start new tracks
        for i, c in enumerate(courts):
            if used[i]:
                continue
            tracks.append(Track(
                track_id=f"c{next(counter):03d}",
                observations=[Observation(
                    year=year, cls=c["class"], confidence=float(c.get("confidence", 0)),
                    obb=c["obb"], n_courts=_n_courts(c),
                    imagery_date=imagery_date, source=source, notes=c.get("notes", "") or "",
                )],
            ))

        # 3. tracks with no observation this year -> inferred removed (unless masked)
        for ti, tr in enumerate(tracks):
            if tr.last.year != year and ti not in masked:
                ref = tr.last_detected() or tr.last
                tr.observations.append(Observation(
                    year=year, cls="removed", confidence=0.5, obb=ref.obb,
                    n_courts=0, inferred=True, imagery_date=imagery_date,
                    source=source, notes="no detection on this footprint",
                ))

    for tr in tracks:
        _fill_detector_gaps(tr)
    return tracks


def _n_courts(c: dict) -> int | None:
    """Court count for a detection: explicit value, None if explicitly unknown, else 1."""
    if "n_courts" in c and c["n_courts"] is None:
        return None
    try:
        return int(c.get("n_courts", 1) or 1)
    except (TypeError, ValueError):
        return 1


def _fill_detector_gaps(tr: Track) -> None:
    """Turn `A, removed(inferred)..., A` into `A, A(interpolated)..., A`."""
    obs = tr.observations
    i = 0
    while i < len(obs):
        if obs[i].inferred:
            j = i
            while j < len(obs) and obs[j].inferred:
                j += 1
            if 0 < i and j < len(obs) and obs[i - 1].cls == obs[j].cls:
                for k in range(i, j):
                    obs[k].cls = obs[i - 1].cls
                    obs[k].inferred = False
                    obs[k].interpolated = True
                    obs[k].n_courts = obs[i - 1].n_courts
                    obs[k].confidence = min(obs[i - 1].confidence, obs[j].confidence, 0.5)
                    obs[k].notes = "detector miss; same class before and after"
            i = j
        else:
            i += 1


def transitions(tr: Track, flag_conf: float = 0.6) -> list[dict]:
    out = []
    for prev, cur in zip(tr.observations, tr.observations[1:]):
        if prev.cls == cur.cls:
            continue
        conf = min(prev.confidence, cur.confidence)
        reasons = []
        if conf < flag_conf:
            reasons.append("low_confidence")
        if cur.cls in REVERSION_TARGETS or prev.cls == "removed":
            reasons.append("reversion")
        if cur.inferred:
            reasons.append("inferred_removal")
        out.append({
            "from": prev.cls, "to": cur.cls,
            "year_from": prev.year, "year_to": cur.year,
            "confidence": round(conf, 3),
            "n_courts_to": cur.n_courts,
            "flagged": bool(reasons), "flag_reasons": reasons,
        })
    return out


def summarize_track(tr: Track, flag_conf: float = 0.6) -> dict:
    """Flatten a track into the court record used by 06/07."""
    obs = tr.observations
    trans = transitions(tr, flag_conf)
    latest = obs[-1]
    ever_tennis = any(o.cls in ("tennis", "hybrid") for o in obs)
    return {
        "track_id": tr.track_id,
        "current_class": latest.cls if not latest.inferred or latest.cls == "removed" else UNKNOWN,
        "current_confidence": round(latest.confidence, 3),
        "current_n_courts": latest.n_courts,
        "current_year": latest.year,
        "current_imagery_date": latest.imagery_date,
        "first_year": obs[0].year,
        "first_class": obs[0].cls,
        "ever_tennis": ever_tennis,
        "history": [o.to_dict() for o in obs],
        "transitions": trans,
        "needs_review": any(t["flagged"] for t in trans) or latest.confidence < flag_conf,
        "obb": (tr.last_detected() or latest).obb,
    }


def apply_overrides(tracks: list[Track], overrides: list[dict], site_id: str) -> list[Track]:
    """Apply human corrections (from review/overrides.csv) to observations.

    Each override row: site_id, track_id, year, class, reviewer, note.
    ``class`` may be any of CLASSES or ``unknown``. Only rows for ``site_id``
    are applied; unknown track/year pairs are ignored with a warning.
    """
    import logging
    log = logging.getLogger("court_watch")
    by_key = {(r["track_id"], int(r["year"])): r for r in overrides if r.get("site_id") == site_id}
    if not by_key:
        return tracks
    for tr in tracks:
        for o in tr.observations:
            r = by_key.pop((tr.track_id, o.year), None)
            if r is None:
                continue
            new_cls = (r.get("class") or "").strip()
            if new_cls not in CLASSES and new_cls != UNKNOWN:
                log.warning("override %s/%s/%s has unknown class %r", site_id, tr.track_id, o.year, new_cls)
                continue
            o.cls = new_cls
            o.confidence = 1.0
            o.inferred = False
            o.interpolated = False
            o.source = f"review:{r.get('reviewer', 'unknown')}"
            o.notes = (r.get("note") or "").strip()
            if r.get("n_courts"):
                try:
                    o.n_courts = int(r["n_courts"])
                except ValueError:
                    pass
    for (tid, year) in by_key:
        log.warning("override for %s/%s/%s matched no observation", site_id, tid, year)
    return tracks


def tracks_from_dicts(records: list[dict]) -> list[Track]:
    """Inverse of summarize_track for the fields needed to re-derive history."""
    out = []
    for rec in records:
        obs = [Observation(
            year=int(h["year"]), cls=h["class"], confidence=float(h["confidence"]),
            obb=h["obb"], n_courts=_n_courts(h), inferred=bool(h.get("inferred")),
            interpolated=bool(h.get("interpolated")), imagery_date=h.get("imagery_date"),
            notes=h.get("notes", ""), source=h.get("source", ""),
        ) for h in rec["history"]]
        out.append(Track(track_id=rec["track_id"], observations=obs))
    return out


def class_counts(records: list[dict]) -> dict[str, Any]:
    """Counts used for summaries. ``pickleball_courts`` sums n_courts so a
    tennis footprint split into four pickleball courts counts as four."""
    counts = {c: 0 for c in CLASSES}
    counts[UNKNOWN] = 0
    pb_courts = 0
    pb_unknown = 0
    for r in records:
        counts[r["current_class"]] = counts.get(r["current_class"], 0) + 1
        if r["current_class"] == "pickleball":
            n = r.get("current_n_courts")
            if n is None or (isinstance(n, float) and n != n):   # None, or NaN from a dataframe
                pb_unknown += 1          # footprint is pickleball, individual count not determined
            else:
                pb_courts += int(n or 1)
    counts["pickleball_courts"] = pb_courts
    counts["pickleball_footprints_uncounted"] = pb_unknown
    counts["footprints"] = len(records)
    counts["needs_review"] = sum(1 for r in records if r.get("needs_review"))
    return counts
