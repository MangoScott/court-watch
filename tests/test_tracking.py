from conftest import court_obb, sawyer_detections, TENNIS_W, TENNIS_H
from tracking import Track, apply_overrides, class_counts, summarize_track, track_site, tracks_from_dicts


def det(year, courts, unusable=False):
    return {"width": 512, "height": 512, "imagery_date": f"{year}-06-01", "unusable": unusable, "courts": courts}


def test_sawyer_point_history():
    d = {y: det(y, c) for y, c in sawyer_detections().items()}
    tracks = track_site(d)
    assert len(tracks) == 8, "eight tennis footprints should stay eight tracks"
    recs = [summarize_track(t) for t in tracks]
    counts = class_counts(recs)
    assert counts["hybrid"] == 3
    assert counts["pickleball"] == 5
    assert counts["pickleball_courts"] == 18
    assert counts["tennis"] == 0
    kinds = sorted(tuple((t["from"], t["to"]) for t in r["transitions"]) for r in recs)
    assert kinds.count((("tennis", "hybrid"),)) == 3
    assert kinds.count((("tennis", "pickleball"),)) == 5
    assert all(t["year_to"] == 2023 for r in recs for t in r["transitions"])
    assert all(t["year_to"] == 2023 for r in recs for t in r["transitions"])
    for r in recs:
        assert not r["needs_review"], r


def test_gap_is_interpolated_not_removed():
    box = court_obb(0.3, 0.5, TENNIS_W, TENNIS_H)
    d = {
        2017: det(2017, [{"class": "tennis", "confidence": 0.9, "obb": box}]),
        2019: det(2019, []),
        2021: det(2021, [{"class": "tennis", "confidence": 0.9, "obb": box}]),
    }
    (tr,) = track_site(d)
    assert [o.cls for o in tr.observations] == ["tennis", "tennis", "tennis"]
    assert tr.observations[1].interpolated
    assert summarize_track(tr)["transitions"] == []


def test_disappearance_is_removed_and_flagged():
    box = court_obb(0.3, 0.5, TENNIS_W, TENNIS_H)
    d = {
        2019: det(2019, [{"class": "tennis", "confidence": 0.9, "obb": box}]),
        2021: det(2021, []),
        2023: det(2023, []),
    }
    (tr,) = track_site(d)
    r = summarize_track(tr)
    assert r["current_class"] == "removed"
    assert len(r["transitions"]) == 1
    assert r["transitions"][0]["flagged"] and "inferred_removal" in r["transitions"][0]["flag_reasons"]


def test_unusable_year_is_ignored():
    box = court_obb(0.3, 0.5, TENNIS_W, TENNIS_H)
    d = {
        2019: det(2019, [{"class": "tennis", "confidence": 0.9, "obb": box}]),
        2021: det(2021, [], unusable=True),
        2023: det(2023, [{"class": "hybrid", "confidence": 0.9, "obb": box}]),
    }
    (tr,) = track_site(d)
    assert [o.year for o in tr.observations] == [2019, 2023]


def test_reversion_is_flagged():
    box = court_obb(0.3, 0.5, TENNIS_W, TENNIS_H)
    d = {
        2019: det(2019, [{"class": "hybrid", "confidence": 0.9, "obb": box}]),
        2021: det(2021, [{"class": "tennis", "confidence": 0.9, "obb": box}]),
    }
    (tr,) = track_site(d)
    t = summarize_track(tr)["transitions"][0]
    assert t["flagged"] and "reversion" in t["flag_reasons"]


def test_new_court_and_neighbor_do_not_merge():
    a = court_obb(0.3, 0.5, TENNIS_W, TENNIS_H)
    b = court_obb(0.3 + 40 / 512, 0.5, TENNIS_W, TENNIS_H)
    d = {
        2019: det(2019, [{"class": "tennis", "confidence": 0.9, "obb": a}]),
        2021: det(2021, [{"class": "tennis", "confidence": 0.9, "obb": a}, {"class": "tennis", "confidence": 0.9, "obb": b}]),
    }
    tracks = track_site(d)
    assert len(tracks) == 2
    assert sorted(t.observations[0].year for t in tracks) == [2019, 2021]


def test_overrides_roundtrip():
    d = {y: det(y, c) for y, c in sawyer_detections().items()}
    recs = [summarize_track(t) for t in track_site(d)]
    tracks = tracks_from_dicts(recs)
    tid = tracks[0].track_id
    tracks = apply_overrides(tracks, [
        {"site_id": "s", "track_id": tid, "year": "2023", "class": "removed", "reviewer": "scott", "note": "paved"},
        {"site_id": "other", "track_id": tid, "year": "2023", "class": "tennis", "reviewer": "x", "note": ""},
    ], "s")
    r = summarize_track(tracks[0])
    assert r["current_class"] == "removed"
    assert r["history"][-1]["source"] == "review:scott"
    assert r["history"][-1]["confidence"] == 1.0


def test_unknown_pickleball_count_is_not_invented():
    box = court_obb(0.3, 0.5, TENNIS_W, TENNIS_H)
    d = {
        2019: det(2019, [{"class": "tennis", "confidence": 0.9, "obb": box}]),
        2023: det(2023, [{"class": "pickleball", "confidence": 0.9, "obb": box, "n_courts": None}]),
    }
    (tr,) = track_site(d)
    r = summarize_track(tr)
    assert r["current_n_courts"] is None
    counts = class_counts([r])
    assert counts["pickleball"] == 1 and counts["pickleball_courts"] == 0 and counts["pickleball_footprints_uncounted"] == 1
    # roundtrip through dict form keeps None
    r2 = summarize_track(tracks_from_dicts([r])[0])
    assert r2["current_n_courts"] is None


def test_class_counts_treats_nan_as_unknown():
    rec = {"current_class": "pickleball", "current_n_courts": float("nan"), "needs_review": False}
    counts = class_counts([rec])
    assert counts["pickleball_courts"] == 0 and counts["pickleball_footprints_uncounted"] == 1


def test_unusable_observation_masks_year_without_removal():
    box = court_obb(0.3, 0.5, TENNIS_W, TENNIS_H)
    d = {
        2019: det(2019, [{"class": "tennis", "confidence": 0.9, "obb": box}]),
        2021: det(2021, [{"class": "unusable", "confidence": 0.8, "obb": box}]),
        2023: det(2023, [{"class": "hybrid", "confidence": 0.9, "obb": box}]),
    }
    (tr,) = track_site(d)
    assert [(o.year, o.cls) for o in tr.observations] == [(2019, "tennis"), (2023, "hybrid")]
    assert len(summarize_track(tr)["transitions"]) == 1


def test_smoothing_removes_single_year_island_without_probs():
    box = court_obb(0.3, 0.5, TENNIS_W, TENNIS_H)
    d = {y: det(y, [{"class": c, "confidence": 0.8, "obb": box}]) for y, c in
         [(2017, "tennis"), (2019, "hybrid"), (2021, "tennis"), (2023, "tennis")]}
    (tr,) = track_site(d)
    assert [o.cls for o in tr.observations] == ["tennis"] * 4
    assert tr.observations[1].smoothed and tr.observations[1].raw_cls == "hybrid"
    assert summarize_track(tr)["transitions"] == []
    (raw,) = track_site(d, smooth=False)
    assert [o.cls for o in raw.observations] == ["tennis", "hybrid", "tennis", "tennis"]


def test_smoothing_with_probs_keeps_real_change():
    box = court_obb(0.3, 0.5, TENNIS_W, TENNIS_H)
    P = lambda t, h, p: {"tennis": t, "hybrid": h, "pickleball": p, "padel": 0, "removed": 0, "unusable": 0}
    d = {
        2017: det(2017, [{"class": "tennis", "confidence": 0.9, "obb": box, "probs": P(0.9, 0.1, 0)}]),
        2019: det(2019, [{"class": "hybrid", "confidence": 0.55, "obb": box, "probs": P(0.45, 0.55, 0)}]),   # flicker
        2021: det(2021, [{"class": "tennis", "confidence": 0.8, "obb": box, "probs": P(0.8, 0.2, 0)}]),
        2022: det(2022, [{"class": "pickleball", "confidence": 0.95, "obb": box, "probs": P(0.03, 0.02, 0.95), "n_courts": None}]),
        2023: det(2023, [{"class": "pickleball", "confidence": 0.96, "obb": box, "probs": P(0.02, 0.02, 0.96), "n_courts": None}]),
    }
    (tr,) = track_site(d)
    classes = [o.cls for o in tr.observations]
    assert classes == ["tennis", "tennis", "tennis", "pickleball", "pickleball"]
    r = summarize_track(tr)
    assert [(t["from"], t["to"], t["year_to"]) for t in r["transitions"]] == [("tennis", "pickleball", 2022)]
    assert r["history"][1]["smoothed"] and r["history"][1]["raw_class"] == "hybrid"
