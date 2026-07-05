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

from pathlib import Path

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


def test_evaluate_model_ndcg_preserves_prod_when_backup_move_fails(tmp_path, monkeypatch):
    """If the FIRST step of the swap -- `PROD_MODEL_SLOT.replace(backup_path)`
    -- itself raises (e.g. a locked/permission-denied file, plausible on
    Windows), no backup was ever created. The original prod-slot file must
    be left completely untouched: `finally` must not infer "swap completed"
    from `PROD_MODEL_SLOT.exists()` alone, since that's also true when the
    swap never started."""
    prod_slot = tmp_path / "lambdarank.json"
    prod_slot.write_text("ORIGINAL")
    monkeypatch.setattr(promote, "PROD_MODEL_SLOT", prod_slot)

    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("CANDIDATE")

    real_replace = Path.replace
    calls = []

    def _flaky_replace(self, target):
        calls.append(self)
        if len(calls) == 1:
            raise PermissionError("simulated locked file")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _flaky_replace)

    def _stub_eval_fn(num_queries):
        return {CONFIG_KEY: {"NDCG@10": 0.5}}

    with pytest.raises(PermissionError, match="simulated locked file"):
        promote.evaluate_model_ndcg(
            candidate_path, num_queries=6980, config_key=CONFIG_KEY, eval_fn=_stub_eval_fn
        )

    # The original must survive: no backup was ever created, so nothing may
    # be deleted from PROD_MODEL_SLOT.
    assert prod_slot.exists()
    assert prod_slot.read_text() == "ORIGINAL"
    assert candidate_path.read_text() == "CANDIDATE"


def test_evaluate_model_ndcg_when_model_is_prod_slot(tmp_path, monkeypatch):
    """Real DAG usage: `model_path` IS `PROD_MODEL_SLOT` (same file), exactly
    like `airflow_dags/retraining_dag.py`'s `prod_path`. Swapping the file
    onto itself would move it away and then try to copy from the now-empty
    original path — destroying the only copy. Must instead evaluate in
    place and leave the file untouched."""
    prod_slot = tmp_path / "lambdarank.json"
    prod_slot.write_text("PROD-CONTENT")
    monkeypatch.setattr(promote, "PROD_MODEL_SLOT", prod_slot)

    def _stub_eval_fn(num_queries):
        # the slot must still hold the real prod content while eval runs
        assert prod_slot.read_text() == "PROD-CONTENT"
        return {CONFIG_KEY: {"NDCG@10": 0.61}}

    ndcg = promote.evaluate_model_ndcg(
        prod_slot, num_queries=6980, config_key=CONFIG_KEY, eval_fn=_stub_eval_fn
    )

    assert ndcg == pytest.approx(0.61)
    assert prod_slot.exists()
    assert prod_slot.read_text() == "PROD-CONTENT"


def test_evaluate_and_gate_prod_equals_slot(tmp_path, monkeypatch):
    """Real DAG usage: `evaluate_and_gate` is called with
    `prod_path == PROD_MODEL_SLOT` (see `airflow_dags/retraining_dag.py`,
    `prod_path = PROJECT_ROOT / "models/lambdarank/lambdarank.json"`).
    Must produce a correct gate decision AND must not lose or corrupt the
    prod-slot file or the staging file."""
    prod_slot = tmp_path / "lambdarank.json"
    prod_slot.write_text("PROD-CONTENT")
    monkeypatch.setattr(promote, "PROD_MODEL_SLOT", prod_slot)

    staging_path = tmp_path / "staging.json"
    staging_path.write_text("STAGING-CONTENT")

    calls = {"n": 0}

    def _stub_eval_fn(num_queries):
        calls["n"] += 1
        if calls["n"] == 1:
            return {CONFIG_KEY: {"NDCG@10": 0.40}}
        return {CONFIG_KEY: {"NDCG@10": 0.45}}

    result = promote.evaluate_and_gate(
        prod_slot, staging_path, margin=0.01, num_queries=6980, eval_fn=_stub_eval_fn
    )

    assert result["prod_ndcg"] == pytest.approx(0.40)
    assert result["staging_ndcg"] == pytest.approx(0.45)
    assert result["delta"] == pytest.approx(0.05)
    assert result["promote"] is True

    # no data loss: prod slot and staging file both intact afterward
    assert prod_slot.exists()
    assert prod_slot.read_text() == "PROD-CONTENT"
    assert staging_path.read_text() == "STAGING-CONTENT"
