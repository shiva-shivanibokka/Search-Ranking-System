"""Unit tests for the ORCAS downloader/sampler.

These tests are pure/offline: no network access, no real download, and no
writes under data/raw/. The download function is monkeypatched wherever main()
is exercised.
"""

from __future__ import annotations


def test_parse_orcas_line():
    from scripts.download_orcas import parse_orcas_line

    assert parse_orcas_line("1\tweather today\tD123\thttp://x") == {
        "qid": "1",
        "query": "weather today",
        "did": "D123",
        "url": "http://x",
    }
    assert parse_orcas_line("bad") is None


def test_sample_orcas_is_deterministic_and_sized():
    from scripts.download_orcas import sample_orcas

    lines = [f"{i}\tquery {i}\tD{i}\thttp://x/{i}\n" for i in range(100)]

    sample_a = sample_orcas(lines, 10, seed=1)
    sample_b = sample_orcas(lines, 10, seed=1)

    assert len(sample_a) == 10
    assert all(isinstance(row, dict) for row in sample_a)
    assert sample_a == sample_b


def test_main_requires_license_optin(monkeypatch, capsys):
    from scripts.download_orcas import main

    called = {"download": False}

    def _fake_download(*args, **kwargs):
        called["download"] = True
        return []

    monkeypatch.setattr("scripts.download_orcas.download_and_sample", _fake_download)

    ret = main([])

    assert ret != 0
    assert called["download"] is False
    captured = capsys.readouterr()
    assert "non-commercial" in captured.out.lower()


def test_propensity_is_monotonic_decreasing():
    from scripts.calibrate_orcas import propensity_curve

    p = propensity_curve(10, eta=1.0)

    # Check that propensity values are strictly decreasing and positive
    for rank in range(1, 10):
        assert p[rank] > p[rank + 1] > 0, f"Not monotonically decreasing: p[{rank}]={p[rank]}, p[{rank+1}]={p[rank+1]}"


def test_query_popularity_and_click_volume():
    from scripts.calibrate_orcas import query_popularity, mean_clicks_per_query

    # Test data: 2 queries total, one clicked 2 distinct docs, one clicked 1 doc
    # Query A: 2 clicks (different docs), Query B: 1 click
    # Mean = (2 + 1) / 2 = 1.5
    rows = [
        {"qid": "A", "query": "weather today", "did": "D1", "url": "http://x"},
        {"qid": "A", "query": "Weather Today", "did": "D2", "url": "http://y"},  # same Q, different doc
        {"qid": "B", "query": "news", "did": "D3", "url": "http://z"},
    ]

    pop = query_popularity(rows)
    assert pop["weather today"] == 2, "Query popularity should normalize case"
    assert pop["news"] == 1
    assert len(pop) == 2

    mean_clicks = mean_clicks_per_query(rows)
    assert mean_clicks == 1.5, f"Expected mean 1.5, got {mean_clicks}"


def test_calibrate_output_shape():
    from scripts.calibrate_orcas import calibrate

    rows = [
        {"qid": "A", "query": "test", "did": "D1", "url": "http://x"},
        {"qid": "A", "query": "test", "did": "D2", "url": "http://y"},
    ]

    result = calibrate(rows, max_rank=10, eta=1.0)

    # Check required keys
    required_keys = {"eta", "propensity", "mean_clicks_per_query", "query_popularity", "source", "notes"}
    assert set(result.keys()) == required_keys, f"Missing or extra keys: {set(result.keys())}"

    # Check propensity keys are strings (for JSON serialization)
    assert all(isinstance(k, str) for k in result["propensity"].keys()), "Propensity keys must be strings"
    assert set(result["propensity"].keys()) == {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}, "Propensity should have ranks 1..10 as string keys"

    # Check source
    assert result["source"] == "ORCAS"

    # Check notes mentions that eta is literature-based and ORCAS has no position column
    assert "literature" in result["notes"].lower(), "Notes should mention literature assumption"
    assert "position" in result["notes"].lower(), "Notes should mention position column"
    assert "orcas" in result["notes"].lower(), "Notes should mention ORCAS"
