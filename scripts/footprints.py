"""Court footprints from OpenStreetMap geometry.

OSM maps some tennis facilities as one polygon per court and others as one
polygon around a whole bank of courts. This module turns either into a list of
individual court footprints (rotated rectangles) using standard court sizes,
so the rest of the pipeline can work court by court without any model drawing
boxes. All maths is in a projected CRS (metres); callers project in and out.

Standard sizes (metres, long x short). "unit" is the paved footprint of one
court including run-outs; "pitch" is the typical centre-to-centre spacing
between adjacent courts in a bank::

    tennis      unit 36.6 x 18.3   pitch 38.0 x 19.5   (lines 23.77 x 10.97)
    pickleball  unit 18.3 x  9.1   pitch 19.0 x 10.0   (lines 13.41 x  6.10)
    padel       unit 20.0 x 10.0   pitch 21.0 x 11.0

Footprints carry flags so downstream code can tell how much to trust them:
``subdivided`` (bank split by arithmetic), ``guessed`` (OSM node only, a
north-up default rectangle), ``oversized`` (polygon far larger than any bank,
probably a whole park or a mis-tag; clipped to a cap).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

UNIT = {"tennis": (36.6, 18.3), "pickleball": (18.3, 9.1), "padel": (20.0, 10.0)}
PITCH = {"tennis": (38.0, 19.5), "pickleball": (19.0, 10.0), "padel": (21.0, 11.0)}
# OSM mappers trace either the fenced/paved compound (UNIT/PITCH above) or just
# the painted lines. Line-traced banks are much tighter, so both styles are tried.
LINES_UNIT = {"tennis": (23.77, 10.97), "pickleball": (13.41, 6.10), "padel": (20.0, 10.0)}
LINES_PITCH = {"tennis": (25.5, 14.6), "pickleball": (14.5, 7.6), "padel": (21.0, 11.0)}
MAX_COURTS_PER_FEATURE = 24


@dataclass
class Footprint:
    cx: float           # centre, projected metres
    cy: float
    length: float       # along the court's long axis, metres
    width: float
    angle: float        # direction of the long axis, degrees counter-clockwise from +x (east)
    sport: str
    n_in_feature: int
    index_in_feature: int
    subdivided: bool = False
    guessed: bool = False
    oversized: bool = False
    role: str = "court"          # "court" (tracked and classified) or "pb_child" (folded into a parent)
    parent: int | None = None    # index of the parent footprint in the site list, for pb_child
    n_children: int = 0          # pickleball courts folded into this footprint (count known from OSM)
    n_overlay: int = 0           # OSM pickleball courts drawn inside this tennis footprint (hybrid hint)
    derived: str = ""            # "" or "pickleball_cluster"
    osm_ref: str = ""

    def corners(self) -> list[tuple[float, float]]:
        return rect_corners(self.cx, self.cy, self.length, self.width, self.angle)

    def polygon(self) -> Polygon:
        return Polygon(self.corners())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["angle"] = round(d["angle"], 2)
        for k in ("cx", "cy", "length", "width"):
            d[k] = round(d[k], 2)
        return d


def rect_corners(cx: float, cy: float, length: float, width: float, angle_deg: float) -> list[tuple[float, float]]:
    """Four corners in order around the rectangle. Long axis at ``angle_deg``."""
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), math.sin(a)          # long axis
    vx, vy = -math.sin(a), math.cos(a)         # short axis
    hl, hw = length / 2.0, width / 2.0
    return [
        (cx - ux * hl - vx * hw, cy - uy * hl - vy * hw),
        (cx + ux * hl - vx * hw, cy + uy * hl - vy * hw),
        (cx + ux * hl + vx * hw, cy + uy * hl + vy * hw),
        (cx - ux * hl + vx * hw, cy - uy * hl + vy * hw),
    ]


def rectangle_params(geom: BaseGeometry) -> tuple[float, float, float, float, float]:
    """(cx, cy, length, width, angle_deg) of the minimum rotated rectangle."""
    rect = geom.minimum_rotated_rectangle
    if rect.geom_type != "Polygon":  # degenerate (line or point)
        c = geom.centroid
        return c.x, c.y, 0.0, 0.0, 0.0
    pts = list(rect.exterior.coords)[:4]
    e1 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    e2 = (pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
    l1, l2 = math.hypot(*e1), math.hypot(*e2)
    if l1 >= l2:
        length, width, ax = l1, l2, e1
    else:
        length, width, ax = l2, l1, e2
    angle = math.degrees(math.atan2(ax[1], ax[0]))
    if angle < 0:
        angle += 180.0
    if angle >= 180.0:
        angle -= 180.0
    c = rect.centroid
    return c.x, c.y, length, width, angle


def _grid_option(L: float, W: float, pl: float, pw: float) -> tuple[int, int, float]:
    """Courts with long axis along L: rows along L, cols along W. Returns (rows, cols, residual)."""
    rows = max(1, round(L / pl))
    cols = max(1, round(W / pw))
    resid = abs(L - rows * pl) / pl + abs(W - cols * pw) / pw
    return rows, cols, resid


def footprints_for_feature(geom: BaseGeometry, sport: str) -> list[Footprint]:
    """Split one OSM feature (projected) into court footprints.

    A polygon close to one court's size is returned as-is. Larger polygons are
    split into a grid of standard courts, choosing the orientation whose grid
    best explains the polygon's dimensions. Nodes get a guessed north-up court.
    """
    sport = sport if sport in UNIT else "tennis"
    ul, uw = UNIT[sport]
    pl, pw = PITCH[sport]

    if geom.is_empty:
        return []
    if isinstance(geom, Point) or geom.area < 1.0:
        c = geom.centroid
        return [Footprint(c.x, c.y, ul, uw, 90.0, sport, 1, 0, guessed=True)]

    cx, cy, L, W, angle = rectangle_params(geom)

    # one court (allow generous slack: fences, run-outs, sloppy tracing)
    if L <= ul * 1.35 and W <= uw * 1.6:
        return [Footprint(cx, cy, L, W, angle, sport, 1, 0)]

    # bank or grid of courts: try both orientations and both mapping styles,
    # keep the grid whose spacing best explains the polygon's dimensions
    best = None
    for style_unit, style_pitch in ((UNIT, PITCH), (LINES_UNIT, LINES_PITCH)):
        spl, spw = style_pitch[sport]
        for along_L in (True, False):
            rows, cols, resid = _grid_option(L, W, spl, spw) if along_L else _grid_option(W, L, spl, spw)
            if best is None or resid < best[0]:
                best = (resid, rows, cols, along_L, style_unit[sport])
    _, rows, cols, along_L, (ul, uw) = best
    n = rows * cols
    oversized = n > MAX_COURTS_PER_FEATURE
    if oversized:
        scale = math.sqrt(MAX_COURTS_PER_FEATURE / n)
        rows, cols = max(1, int(rows * scale)), max(1, int(cols * scale))
        n = rows * cols

    a = math.radians(angle)
    ux, uy = math.cos(a), math.sin(a)      # along L
    vx, vy = -math.sin(a), math.cos(a)     # along W
    out = []
    if along_L:
        cell_l, cell_w = L / rows, W / cols
        court_angle = angle
        for r in range(rows):
            for c in range(cols):
                off_l = (r + 0.5) * cell_l - L / 2.0
                off_w = (c + 0.5) * cell_w - W / 2.0
                out.append(Footprint(cx + ux * off_l + vx * off_w, cy + uy * off_l + vy * off_w,
                                     min(cell_l, ul), min(cell_w, uw), court_angle, sport, n, len(out),
                                     subdivided=True, oversized=oversized))
    else:
        cell_l, cell_w = W / rows, L / cols   # court long axis along W
        court_angle = (angle + 90.0) % 180.0
        for r in range(rows):
            for c in range(cols):
                off_w = (r + 0.5) * cell_l - W / 2.0    # along W (court long axis)
                off_l = (c + 0.5) * cell_w - L / 2.0    # along L (court short axis)
                out.append(Footprint(cx + ux * off_l + vx * off_w, cy + uy * off_l + vy * off_w,
                                     min(cell_l, ul), min(cell_w, uw), court_angle, sport, n, len(out),
                                     subdivided=True, oversized=oversized))
    return out


def dedupe_footprints(fps: list[Footprint], min_dist: float = 6.0) -> list[Footprint]:
    """Drop footprints whose centres fall within ``min_dist`` m of an earlier
    one (OSM sometimes has both a court polygon and a node for the same court,
    or a bank polygon plus individual courts). Non-guessed, non-subdivided
    footprints win."""
    order = sorted(fps, key=lambda f: (f.guessed, f.subdivided))
    kept: list[Footprint] = []
    for f in order:
        if any(math.hypot(f.cx - k.cx, f.cy - k.cy) < min_dist for k in kept):
            continue
        kept.append(f)
    return kept


def _connected_groups(polys: list[Polygon], gap: float) -> list[list[int]]:
    """Indices grouped by touching (within ``gap`` metres), union-find style."""
    n = len(polys)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    buffered = [p.buffer(gap / 2.0) for p in polys]
    for i in range(n):
        for j in range(i + 1, n):
            if buffered[i].intersects(buffered[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def split_cluster(rect: Polygon, n_children: int) -> list[Footprint]:
    """Tennis-sized parents for a block of pickleball courts.

    Pickleball courts painted on former tennis courts are usually 4 per court
    (sometimes 2, 3 or 6) and the block of painted lines is narrower than the
    slab. For a single row (block no longer than ~45 m across) the number of
    parents comes from the block width at a tight 18.3 m pitch (plus run-out),
    sanity-checked against the child count; multi-row blocks fall back to the
    generic grid.
    """
    cx, cy, L, W, angle = rectangle_params(rect)
    if W > 45:
        return footprints_for_feature(rect, "tennis")
    k = max(1, round((L + 4.0) / 18.3))
    if n_children and n_children / k > 6:
        k = math.ceil(n_children / 6)
    if n_children:
        k = min(k, n_children)
    a = math.radians(angle)
    ux, uy = math.cos(a), math.sin(a)
    cell = L / k
    out = []
    for i in range(k):
        off = (i + 0.5) * cell - L / 2.0
        out.append(Footprint(cx + ux * off, cy + uy * off, max(W, LINES_UNIT["tennis"][0]), cell,
                             (angle + 90.0) % 180.0, "tennis", k, i, subdivided=True))
    return out


def aggregate_pickleball(fps: list[Footprint], gap: float = 6.0, min_cluster: int = 3) -> list[Footprint]:
    """Fold OSM pickleball courts into tennis-sized footprints where they clearly
    sit on former tennis courts, so the same footprint can be classified in
    every year (tennis in 2019, pickleball in 2023) and counts come from OSM.

    * A pickleball court whose centre lies inside a tennis/padel footprint
      becomes a ``pb_child`` of it and bumps the parent's ``n_overlay`` (pickleball
      lines drawn on a tennis court: the hybrid signature).
    * Contiguous groups of >= ``min_cluster`` pickleball courts whose union is
      at least tennis-court sized are re-cut into standard tennis footprints
      (``derived = "pickleball_cluster"``, sport "pickleball"); each child is
      attached to the nearest parent and the parent's ``n_children`` is the
      known pickleball count. Parents that end up with no children are dropped.
    * Everything else (a lone pickleball court, a pair) stays a court of its own
      with ``n_children = 1``.

    Returns the full list (parents, courts and children) with ``parent`` set to
    list indices, ready for ``derive_courts`` to give ids.
    """
    tennis = [f for f in fps if f.sport != "pickleball"]
    pb = [f for f in fps if f.sport == "pickleball"]
    out: list[Footprint] = list(tennis)
    tpolys = [f.polygon() for f in tennis]
    free: list[Footprint] = []
    for f in pb:
        c = Point(f.cx, f.cy)
        hit = next((i for i, p in enumerate(tpolys) if p.contains(c)), None)
        if hit is not None:
            f.role, f.parent = "pb_child", hit
            tennis[hit].n_overlay += 1
            out.append(f)
        else:
            free.append(f)
    if not free:
        return out
    for group in _connected_groups([f.polygon() for f in free], gap):
        members = [free[i] for i in group]
        if len(members) >= min_cluster:
            union = unary_union([m.polygon() for m in members])
            _, _, L, W, _ = rectangle_params(union)
            if L >= 22 and W >= 10:
                parents = split_cluster(union.minimum_rotated_rectangle, len(members))
                for p in parents:
                    p.sport, p.subdivided, p.derived = "pickleball", True, "pickleball_cluster"
                    p.osm_ref = ";".join(sorted({m.osm_ref for m in members if m.osm_ref}))[:200]
                assigned: dict[int, list[Footprint]] = {}
                for m in members:
                    k = min(range(len(parents)), key=lambda i: math.hypot(parents[i].cx - m.cx, parents[i].cy - m.cy))
                    assigned.setdefault(k, []).append(m)
                for k, kids in assigned.items():
                    parent = parents[k]
                    parent.n_children = len(kids)
                    out.append(parent)
                    pidx = len(out) - 1
                    for m in kids:
                        m.role, m.parent = "pb_child", pidx
                        out.append(m)
                continue
        for m in members:
            m.n_children = 1
            out.append(m)
    return out
