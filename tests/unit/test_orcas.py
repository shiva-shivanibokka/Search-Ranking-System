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
