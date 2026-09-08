"""Label a NAIP chip with the Claude vision API.

Shared by 03_label_with_claude.py (build the training set) and
05_detect.py --backend claude (label everything without training a model).

The request is a single image plus a fixed prompt describing the five court
classes. The response is constrained to a JSON schema via
``output_config.format`` so parsing never depends on prose. The raw response
text and token usage are stored verbatim next to the parsed result so the
labels can always be audited.

Coordinates: the model returns oriented bounding boxes as four corner points
in normalized image coordinates (0-1, origin top-left). ``parse_output``
clamps, validates and drops degenerate boxes; nothing else is altered.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

try:
    from common import CHIP_GSD, CHIP_SIZE, CLASSES, normalize_obb
except ImportError:  # pragma: no cover
    from scripts.common import CHIP_GSD, CHIP_SIZE, CLASSES, normalize_obb  # type: ignore

log = logging.getLogger("court_watch")

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "medium"
MAX_TOKENS = 8000

# $/1M tokens (input, output). Only used for --dry-run estimates.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

SYSTEM_PROMPT = """You are an expert at reading aerial and satellite imagery of sports facilities in the United States.

You will be shown one NAIP aerial image chip. Your job is to find every racket-sport court footprint in it and classify each one. Be precise about geometry and honest about confidence. If something cannot be determined from the image, say so in the notes and lower the confidence; never guess a count you cannot see.

CLASSES
1. tennis - a clean tennis court: full tennis line set, no pickleball lines. Playing lines are 23.77 x 10.97 m (roughly a 2.2:1 rectangle); the fenced or paved footprint including run-outs is about 36 x 18 m.
2. hybrid - a tennis court footprint with ANY visible pickleball lines painted over or inside the tennis lines, even one faded set. Pickleball lines are a small 13.4 x 6.1 m rectangle, usually two or four of them inside one tennis court, often in a contrasting color. When you are unsure between tennis and hybrid, choose hybrid.
3. pickleball - a dedicated pickleball court: tennis lines gone or the footprint reconfigured (fences or paint dividing the slab into pickleball-sized courts). Report EACH pickleball court as its own box, even when several share one slab.
4. padel - an enclosed 20 x 10 m court with glass or mesh walls; usually a crisp rectangle with a dark or shadowed wall outline and a turf-colored floor.
5. removed - a former court location that is now something else (parking, grass, turf, basketball, bare slab, construction). Only use this when a court footprint is still visible (slab, fence line, ghost lines) but it is clearly not a court any more. If there is no evidence a court was ever there, report nothing.

COMMON FALSE POSITIVES - DO NOT REPORT THESE AS COURTS
- Basketball courts: about 28 x 15 m, a key/lane and three-point arc at each end, hoops with shadows. Never classify a basketball court as tennis.
- Shade structures, pavilions and roofed courts: report only what is visible; if a roof hides the surface, describe it in notes with low confidence.
- Freshly resurfaced courts: a uniformly dark or unusually bright surface with faint lines is still a court; classify by the lines you can see.
- Parking stalls, pool decks, running tracks, playgrounds, tennis backboards without a court.

GEOMETRY
- Return an oriented bounding box (obb) for each court as four corner points in normalized image coordinates, x and y each between 0 and 1 with (0,0) at the top-left of the image. List corners in order around the rectangle (clockwise).
- For tennis and hybrid the box is the painted playing-line rectangle of ONE court (not the fenced compound, not the whole bank). Banks of courts get one box per court.
- For pickleball the box is one pickleball court's lines.
- For padel the box is the wall enclosure.
- For removed the box is the old footprint as best you can see it.

CONFIDENCE
- 0.9+ : lines clearly visible, class unambiguous.
- 0.6-0.9: class likely but lines faint, partial shadow, or resolution limits.
- below 0.6: you can see a court but the class is a guess. Explain in notes.

If the image is unusable (clouds, no data, blank, obviously the wrong place) set unusable to true, return an empty courts list and explain in chip_notes."""

USER_PROMPT = """Image facts: {size}x{size} px, {gsd} m per pixel, so the chip covers about {extent_m:.0f} m on each side. Imagery year: {year}{date_part}. North is up.

At this scale a tennis court's playing lines are about {tennis_w:.0f} x {tennis_l:.0f} px, a pickleball court's lines about {pb_w:.0f} x {pb_l:.0f} px, and a basketball court about {bb_w:.0f} x {bb_l:.0f} px.

Find and classify every racket-sport court in this image. Return only the JSON described by the schema."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "unusable": {"type": "boolean", "description": "True if the image cannot be interpreted."},
        "chip_notes": {"type": "string", "description": "Short notes about the whole image: setting, shadows, anything ambiguous."},
        "courts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "class": {"type": "string", "enum": CLASSES},
                    "confidence": {"type": "number", "description": "0 to 1"},
                    "obb": {
                        "type": "array",
                        "description": "Four [x, y] corner points, normalized 0-1, clockwise.",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                    "notes": {"type": "string"},
                },
                "required": ["class", "confidence", "obb", "notes"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["unusable", "chip_notes", "courts"],
    "additionalProperties": False,
}


def user_prompt(year: int | str, imagery_date: str | None, size: int = CHIP_SIZE, gsd: float = CHIP_GSD) -> str:
    return USER_PROMPT.format(
        size=size, gsd=gsd, extent_m=size * gsd, year=year,
        date_part=f" (acquired {imagery_date})" if imagery_date else "",
        tennis_w=10.97 / gsd, tennis_l=23.77 / gsd,
        pb_w=6.1 / gsd, pb_l=13.4 / gsd,
        bb_w=15.0 / gsd, bb_l=28.0 / gsd,
    )


def encode_png(path: Path | str, upscale: int = 1) -> str:
    """Base64 PNG. ``upscale`` > 1 resizes with Lanczos before encoding; it adds
    no information but makes faint pickleball lines easier for the model to
    resolve at the cost of ~upscale**2 more image tokens."""
    if upscale <= 1:
        with open(path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("ascii")
    import io
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB").resize((im.width * upscale, im.height * upscale), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def build_params(
    png_path: Path | str,
    year: int | str,
    imagery_date: str | None,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    size: int = CHIP_SIZE,
    gsd: float = CHIP_GSD,
    max_tokens: int = MAX_TOKENS,
    upscale: int = 1,
) -> dict[str, Any]:
    """Parameters for ``client.messages.create`` (also valid inside a batch Request)."""
    size, gsd = size * upscale, gsd / upscale  # what the model actually sees
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encode_png(png_path, upscale)}},
                {"type": "text", "text": user_prompt(year, imagery_date, size, gsd)},
            ],
        }],
    }


def parse_output(text: str) -> dict[str, Any]:
    """Validate the model's JSON into the courts structure used downstream.

    Raises ValueError if the text is not JSON or not an object. Invalid
    individual courts are dropped and counted in ``dropped``.
    """
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    courts_out = []
    dropped = 0
    for c in data.get("courts") or []:
        if not isinstance(c, dict):
            dropped += 1
            continue
        cls = c.get("class")
        obb = normalize_obb(c.get("obb"))
        if cls not in CLASSES or obb is None:
            dropped += 1
            continue
        try:
            conf = float(c.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = 0.0 if conf < 0 else 1.0 if conf > 1 else conf
        courts_out.append({
            "class": cls, "confidence": round(conf, 3), "obb": [[round(x, 5), round(y, 5)] for x, y in obb],
            "notes": str(c.get("notes") or "")[:500],
        })
    return {
        "unusable": bool(data.get("unusable", False)),
        "chip_notes": str(data.get("chip_notes") or "")[:1000],
        "courts": courts_out,
        "dropped": dropped,
    }


def message_to_record(message: Any, model: str) -> dict[str, Any]:
    """Turn an SDK Message (or batch result message) into the raw/parsed pair
    stored on disk. Never raises on model content; failures become unusable."""
    text = ""
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text += block.text
    usage = getattr(message, "usage", None)
    rec: dict[str, Any] = {
        "source": f"claude:{model}",
        "model": getattr(message, "model", model),
        "stop_reason": getattr(message, "stop_reason", None),
        "request_id": getattr(message, "_request_id", None),
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        } if usage is not None else None,
        "raw_text": text,
    }
    if rec["stop_reason"] == "refusal":
        rec.update({"unusable": True, "chip_notes": "model refused", "courts": [], "dropped": 0})
        return rec
    try:
        rec.update(parse_output(text))
    except ValueError as e:
        rec.update({"unusable": True, "chip_notes": f"unparseable output: {e}", "courts": [], "dropped": 0})
    return rec


def label_chip(client: Any, png_path: Path | str, year: int, imagery_date: str | None,
               model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT,
               size: int = CHIP_SIZE, gsd: float = CHIP_GSD, retries: int = 4,
               upscale: int = 1) -> dict[str, Any]:
    """Synchronous single-chip label with backoff on transient API errors."""
    import anthropic

    params = build_params(png_path, year, imagery_date, model, effort, size, gsd, upscale=upscale)
    delay = 5.0
    for attempt in range(retries + 1):
        try:
            message = client.messages.create(**params)
            return message_to_record(message, model)
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            if attempt >= retries:
                raise
            wait = delay
            if isinstance(e, anthropic.RateLimitError):
                try:
                    wait = float(e.response.headers.get("retry-after", delay))
                except (TypeError, ValueError, AttributeError):
                    pass
            log.warning("API error %s; retrying in %.0fs", type(e).__name__, wait)
            time.sleep(wait)
            delay = min(delay * 2, 120)
    raise RuntimeError("unreachable")


def estimate_cost(n_chips: int, model: str = DEFAULT_MODEL, batch: bool = False,
                  size: int = CHIP_SIZE, upscale: int = 1) -> dict[str, float]:
    """Rough $ estimate. Image tokens ~ w*h/750; prompt ~1100 tokens; output ~400."""
    inp, out = PRICES.get(model, PRICES[DEFAULT_MODEL])
    image_tokens = (size * upscale) ** 2 / 750
    per_chip_in = image_tokens + 1100
    per_chip_out = 400
    cost = n_chips * (per_chip_in * inp + per_chip_out * out) / 1e6
    if batch:
        cost *= 0.5
    return {"chips": n_chips, "input_tokens": n_chips * per_chip_in,
            "output_tokens": n_chips * per_chip_out, "usd": round(cost, 2)}
