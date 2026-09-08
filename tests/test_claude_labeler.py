import json
from types import SimpleNamespace

import pytest

import claude_labeler as cl
from common import CLASSES, yolo_obb_line


def test_parse_output_drops_invalid_and_clamps():
    text = json.dumps({
        "unusable": False, "chip_notes": "park",
        "courts": [
            {"class": "hybrid", "confidence": 1.4, "obb": [[-0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]], "notes": "n"},
            {"class": "basketball", "confidence": 0.9, "obb": [[0, 0], [1, 0], [1, 1], [0, 1]], "notes": ""},
            {"class": "tennis", "confidence": 0.9, "obb": [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], "notes": ""},
            {"class": "tennis", "confidence": "0.7", "obb": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.3]], "notes": ""},
        ],
    })
    out = cl.parse_output(text)
    assert len(out["courts"]) == 1
    assert out["dropped"] == 3
    c = out["courts"][0]
    assert c["confidence"] == 1.0 and c["obb"][0] == [0.0, 0.1]


def test_parse_output_rejects_non_object():
    with pytest.raises(ValueError):
        cl.parse_output("[1,2]")


def test_message_to_record_handles_refusal_and_garbage():
    msg = SimpleNamespace(content=[], usage=None, stop_reason="refusal", model="m")
    rec = cl.message_to_record(msg, "m")
    assert rec["unusable"] and rec["courts"] == []
    msg = SimpleNamespace(content=[SimpleNamespace(type="text", text="not json")], usage=SimpleNamespace(input_tokens=1, output_tokens=2),
                          stop_reason="end_turn", model="m")
    rec = cl.message_to_record(msg, "m")
    assert rec["unusable"] and "unparseable" in rec["chip_notes"] and rec["usage"]["output_tokens"] == 2


def test_build_params_shape(tmp_path):
    from PIL import Image
    png = tmp_path / "c.png"
    Image.new("RGB", (512, 512)).save(png)
    p = cl.build_params(png, 2023, "2023-06-01", upscale=2)
    assert p["model"] == cl.DEFAULT_MODEL
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert p["output_config"]["format"]["schema"]["properties"]["courts"]["items"]["properties"]["class"]["enum"] == CLASSES
    img, txt = p["messages"][0]["content"]
    assert img["source"]["media_type"] == "image/png" and len(img["source"]["data"]) > 100
    assert "1024x1024 px, 0.3 m per pixel" in txt["text"]


def test_yolo_line_format():
    line = yolo_obb_line("pickleball", [[0.1, 0.2], [0.3, 0.2], [0.3, 0.5], [0.1, 0.5]])
    parts = line.split()
    assert parts[0] == "2" and len(parts) == 9


def test_cost_estimate_batch_halves():
    a = cl.estimate_cost(100)
    b = cl.estimate_cost(100, batch=True)
    assert abs(a["usd"] / 2 - b["usd"]) < 0.02
