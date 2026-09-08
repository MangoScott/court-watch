from conftest import load_script


def test_row_year_parsing():
    m = load_script("02b_fetch_osip")
    assert m.row_year({"AcquisitionDate": 1745000000000}) == (2025, "2025-04-18")
    assert m.row_year({"AcquisitionDate": "2025-03-30T00:00:00"}) == (2025, "2025-03-30")
    assert m.row_year({"Year": "2019"}) == (2019, None)
    assert m.row_year({"Name": "HAMI_2023_6in_tile_123"}) == (2023, None)
    assert m.row_year({"Name": "nothing here"}) == (None, None)


def test_years_in():
    m = load_script("02b_fetch_osip")
    assert m.years_in("OSIP_2025_Franklin_6in") == [2025]
    assert m.years_in("osip_best_avail_1ft") == []


def test_preflight_false_when_unreachable(monkeypatch):
    m = load_script("02b_fetch_osip")
    import requests
    def boom(*a, **k):
        raise requests.ConnectionError("nope")
    monkeypatch.setattr(m.requests, "get", boom)
    assert m.preflight(["https://example.invalid/ImageServer"]) is False
