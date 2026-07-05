"""Unit tests for the real promotion gate (scripts/promote.py).

These lock in the fix for the Critical audit bug where the retraining DAG
compared a freshly-evaluated staging NDCG@10 against a STALE MLflow metric
pulled from whatever the last logged prod training run happened to record
(apples-to-oranges: different eval slice, different point in time). The
fix: evaluate BOTH prod and staging under the exact same `eval_fn` call
inside `evaluate_and_gate`, so the two numbers are directly comparable.

All tests here use a stub `eval_fn` — the real `training.evaluate.run_evaluation`
is expensive (loads every model, runs real queries, logs MLflow) and must
never be invoked from unit tests.
"""

from __future__ import annotations

import pytest

import scripts.promote as promote

CONFIG_KEY = "Hybrid(RRF)+LambdaRank"


def test_gate_rejects_no_improvement(tmp_path, monkeypatch):
    """delta below margin -> promote is False."""
    prod_path = tmp_path / "prod.json"
    staging_path = tmp_path / "staging.json"
    prod_path.write_text("prod-model")
    staging_path.write_text("staging-model")

    # evaluate_model_ndcg swaps model_path into the prod slot before calling
    # eval_fn, so the stub can't tell prod from staging by content alone —
    # use a call-order side-effect counter instead (first call = prod).
    calls = {"n": 0}

    def _stub_eval_fn(num_queries):
        calls["n"] += 1
        if calls["n"] == 1:
            return {CONFIG_KEY: {"NDCG@10": 0.50}}
        return {CONFIG_KEY: {"NDCG@10": 0.505}}

    # evaluate_model_ndcg's file-swap targets the real prod slot
    # (models/lambdarank/lambdarank.json) inside evaluate_and_gate; patch
    # that module-level constant to a tmp path so this test never touches
    # the real repo's model files.
    monkeypatch.setattr(promote, "PROD_MODEL_SLOT", tmp_path / "lambdarank.json")

    result = promote.evaluate_and_gate(
        prod_path, staging_path, margin=0.01, num_queries=6980, eval_fn=_stub_eval_fn
    )

    assert result["prod_ndcg"] == pytest.approx(0.50)
    assert result["staging_ndcg"] == pytest.approx(0.505)
    assert result["delta"] == pytest.approx(0.005)
    assert result["promote"] is False


def test_gate_accepts_improvement(tmp_path, monkeypatch):
    """delta at/above margin -> promote is True."""
    prod_path = tmp_path / "prod.json"
    staging_path = tmp_path / "staging.json"
    prod_path.write_text("prod-model")
    staging_path.write_text("staging-model")

    calls = {"n": 0}

    def _stub_eval_fn(num_queries):
        calls["n"] += 1
        if calls["n"] == 1:
            return {CONFIG_KEY: {"NDCG@10": 0.50}}
        return {CONFIG_KEY: {"NDCG@10": 0.53}}

    monkeypatch.setattr(promote, "PROD_MODEL_SLOT", tmp_path / "lambdarank.json")

    result = promote.evaluate_and_gate(
        prod_path, staging_path, margin=0.01, num_queries=6980, eval_fn=_stub_eval_fn
    )

    assert result["prod_ndcg"] == pytest.approx(0.50)
    assert result["staging_ndcg"] == pytest.approx(0.53)
    assert result["delta"] == pytest.approx(0.03)
    assert result["promote"] is True


def test_evaluate_model_restores_original(tmp_path, monkeypatch):
    """The original prod-slot file must be back in place after
    evaluate_model_ndcg returns, even on the happy path."""
    prod_slot = tmp_path / "lambdarank.json"
    prod_slot.write_text("ORIGINAL")
    monkeypatch.setattr(promote, "PROD_MODEL_SLOT", prod_slot)

    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("CANDIDATE")

    def _stub_eval_fn(num_queries):
        # while eval_fn runs, the prod slot must contain the swapped-in candidate
        assert prod_slot.read_text() == "CANDIDATE"
        return {CONFIG_KEY: {"NDCG@10": 0.42}}

    ndcg = promote.evaluate_model_ndcg(
        candidate_path, num_queries=6980, config_key=CONFIG_KEY, eval_fn=_stub_eval_fn
    )

    assert ndcg == pytest.approx(0.42)
    assert prod_slot.read_text() == "ORIGINAL"
    # the candidate file itself must be untouched (swap, not move)
    assert candidate_path.read_text() == "CANDIDATE"


def test_evaluate_model_restores_original_even_if_eval_fn_raises(tmp_path, monkeypatch):
    """If eval_fn raises mid-evaluation, the original prod-slot file must
    still be restored (try/finally, not just happy-path cleanup)."""
    prod_slot = tmp_path / "lambdarank.json"
    prod_slot.write_text("ORIGINAL")
    monkeypatch.setattr(promote, "PROD_MODEL_SLOT", prod_slot)

    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("CANDIDATE")

    def _boom_eval_fn(num_queries):
        raise RuntimeError("eval blew up")

    with pytest.raises(RuntimeError, match="eval blew up"):
        promote.evaluate_model_ndcg(
            candidate_path, num_queries=6980, config_key=CONFIG_KEY, eval_fn=_boom_eval_fn
        )

    assert prod_slot.read_text() == "ORIGINAL"
    assert candidate_path.read_text() == "CANDIDATE"


def test_evaluate_model_restores_original_when_prod_slot_did_not_exist(tmp_path, monkeypatch):
    """If there was no prod model at all beforehand, evaluate_model_ndcg must
    leave the slot empty again afterward (no leftover candidate copy)."""
    prod_slot = tmp_path / "lambdarank.json"
    monkeypatch.setattr(promote, "PROD_MODEL_SLOT", prod_slot)

    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("CANDIDATE")

    def _stub_eval_fn(num_queries):
        return {CONFIG_KEY: {"NDCG@10": 0.1}}

    promote.evaluate_model_ndcg(
        candidate_path, num_queries=6980, config_key=CONFIG_KEY, eval_fn=_stub_eval_fn
    )

    assert not prod_slot.exists()
    assert candidate_path.read_text() == "CANDIDATE"
