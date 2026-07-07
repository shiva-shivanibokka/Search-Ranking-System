"""Unit tests for corrected (propensity-weighted, non-degenerate) retraining.

These tests are pure/offline: `retriever` and `build_lambdarank_features` are
stubbed/monkeypatched so no real BM25/two-tower model ever loads. They exist
to lock in the fix for the Critical audit bug where every retrain label was
hard-coded to `1` (a degenerate label set with no ranking signal): labels
must come from `clicked` (impressions LEFT JOIN clicks gives real negatives)
and clicked rows must carry an inverse-propensity-score (IPS) weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.retrain_from_clicks as retrain


class _StubRetriever:
    """Fake retriever: returns deterministic Candidate rows, no model load."""

    def get_candidates(self, query_text, doc_ids, ranks):
        return [
            retrain.Candidate(doc_id=doc_id, text=f"doc-{doc_id}", score=1.0, retrieval_rank=rank)
            for doc_id, rank in zip(doc_ids, ranks)
        ]


def _fake_build_lambdarank_features(query_text, candidates, bm25, bm25_pid_list, pid_to_len):
    """Deterministic (n, 7) matrix; shape-correct stand-in for the real
    shared feature builder (which requires a real bm25 index to run)."""
    return np.array([[float(c.doc_id)] * 7 for c in candidates], dtype=np.float32)


def test_matrix_has_multiple_labels_per_group(monkeypatch):
    monkeypatch.setattr(retrain, "build_lambdarank_features", _fake_build_lambdarank_features)

    labeled = pd.DataFrame(
        {
            "request_id": ["r1", "r1", "r1"],
            "query_text": ["q", "q", "q"],
            "doc_id": [10, 20, 30],
            "rank_shown": [1, 2, 3],
            "clicked": [True, False, False],
        }
    )
    propensity = {1: 1.0, 2: 0.5, 3: 0.33}

    X, y, weights, groups = retrain.build_training_matrix(
        labeled,
        _StubRetriever(),
        bm25=None,
        bm25_pid_list=[],
        pid_to_len={},
        propensity=propensity,
    )

    assert groups == [3]
    # The core assertion this whole task exists for: labels are NOT all-1.
    assert set(y[:3]) == {0.0, 1.0}
    assert len(np.unique(y)) > 1

    assert y[0] == 1.0
    assert weights[0] == pytest.approx(1.0 / propensity[1])
    assert y[1] == 0.0
    assert weights[1] == 1.0
    assert y[2] == 0.0
    assert weights[2] == 1.0


def test_matrix_multiple_request_groups(monkeypatch):
    """Two request_ids -> two groups, each sized by its own shown-set."""
    monkeypatch.setattr(retrain, "build_lambdarank_features", _fake_build_lambdarank_features)

    labeled = pd.DataFrame(
        {
            "request_id": ["r1", "r1", "r2"],
            "query_text": ["q1", "q1", "q2"],
            "doc_id": [10, 20, 30],
            "rank_shown": [1, 2, 1],
            "clicked": [True, False, False],
        }
    )
    propensity = {1: 1.0, 2: 0.5}

    X, y, weights, groups = retrain.build_training_matrix(
        labeled,
        _StubRetriever(),
        bm25=None,
        bm25_pid_list=[],
        pid_to_len={},
        propensity=propensity,
    )

    assert groups == [2, 1]
    assert X.shape == (3, 7)
    assert len(y) == 3
    assert len(weights) == 3


def test_is_degenerate_all_ones():
    assert retrain.is_degenerate(np.array([1.0, 1.0, 1.0], dtype=np.float32)) is True


def test_is_degenerate_all_zeros():
    assert retrain.is_degenerate(np.array([0.0, 0.0], dtype=np.float32)) is True


def test_is_degenerate_mixed_is_false():
    assert retrain.is_degenerate(np.array([1.0, 0.0, 1.0], dtype=np.float32)) is False


def test_main_aborts_on_degenerate_labels(monkeypatch, capsys):
    """Even if build_training_matrix somehow produced all-1 labels, main()
    must refuse to train/publish on them (belt-and-suspenders guard)."""
    labeled = pd.DataFrame(
        {
            "request_id": ["r1", "r1", "r1"],
            "query_text": ["q", "q", "q"],
            "doc_id": [10, 20, 30],
            "rank_shown": [1, 2, 3],
            "clicked": [True, True, True],
        }
    )

    monkeypatch.setattr(retrain, "THRESHOLD", 1)
    monkeypatch.setattr(retrain, "get_engine", lambda: object())
    monkeypatch.setattr(retrain, "load_labeled_impressions", lambda engine: labeled)
    monkeypatch.setattr(retrain, "_load_retriever", lambda: _StubRetriever())
    monkeypatch.setattr(retrain, "_load_bm25", lambda: (None, [], {}))
    monkeypatch.setattr(retrain, "_load_propensity", lambda: {1: 1.0, 2: 0.5, 3: 0.33})

    def _degenerate_matrix(*args, **kwargs):
        X = np.zeros((3, 7), dtype=np.float32)
        y = np.ones(3, dtype=np.float32)
        weights = np.ones(3, dtype=np.float32)
        groups = [3]
        return X, y, weights, groups

    monkeypatch.setattr(retrain, "build_training_matrix", _degenerate_matrix)

    def _boom(*args, **kwargs):
        raise AssertionError("must not be called when labels are degenerate")

    monkeypatch.setattr(retrain, "train_and_save", _boom)
    monkeypatch.setattr(retrain, "_publish", _boom)

    result = retrain.main()
    captured = capsys.readouterr()

    assert result == 0
    assert "Degenerate labels" in captured.out


def test_main_below_threshold_skips_everything(monkeypatch, capsys):
    monkeypatch.setattr(retrain, "THRESHOLD", 1000)
    monkeypatch.setattr(retrain, "get_engine", lambda: object())
    monkeypatch.setattr(
        retrain,
        "load_labeled_impressions",
        lambda engine: pd.DataFrame(
            {"request_id": [], "query_text": [], "doc_id": [], "rank_shown": [], "clicked": []}
        ),
    )

    def _boom(*args, **kwargs):
        raise AssertionError("must not be called below threshold")

    # No prior retrain -> high-water mark is 0, so new_rows == total (0) < THRESHOLD.
    monkeypatch.setattr(retrain, "_load_last_trained", lambda: 0)
    monkeypatch.setattr(retrain, "build_training_matrix", _boom)
    monkeypatch.setattr(retrain, "train_and_save", _boom)
    monkeypatch.setattr(retrain, "_publish", _boom)

    result = retrain.main()
    captured = capsys.readouterr()

    assert result == 0
    assert "nothing to do" in captured.out
