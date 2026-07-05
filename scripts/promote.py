"""Real promotion gate: prod vs staging under one eval harness.

Fixes the Critical audit bug in airflow_dags/retraining_dag.py where the
retrain DAG compared a freshly-evaluated staging NDCG@10 against a STALE
MLflow metric — whatever `final_dev_ndcg10` happened to be logged on the
last training run tagged 'lambdarank_training'. That comparison is
apples-to-oranges: different eval slice, different point in time, and it
silently produces `prod_ndcg == 0.0` (and therefore a spurious PROMOTE) if
no such run exists yet.

The fix: evaluate BOTH the current production model and the staging
candidate with the exact same `eval_fn` call (`training.evaluate.
run_evaluation` by default), one right after the other, so the two NDCG@10
numbers are directly comparable. `evaluate_model_ndcg` does this by
temporarily swapping a given model file into the real production slot
(`models/lambdarank/lambdarank.json`, which is where `run_evaluation`
always loads the LambdaRank model from), running the eval, and restoring
whatever was in that slot beforehand — even if the eval raises.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from training.evaluate import run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The one slot `training.evaluate.run_evaluation` always loads the
# LambdaRank model from (cfg.lambdarank.save_dir / "lambdarank.json").
PROD_MODEL_SLOT = PROJECT_ROOT / "models" / "lambdarank" / "lambdarank.json"

DEFAULT_CONFIG_KEY = "Hybrid(RRF)+LambdaRank"


def evaluate_model_ndcg(
    model_path: Path,
    num_queries: int,
    config_key: str = DEFAULT_CONFIG_KEY,
    eval_fn: Callable[..., dict] = run_evaluation,
) -> float:
    """Swap `model_path` into the production model slot, run `eval_fn`, and
    return NDCG@10 for `config_key`. The slot is ALWAYS restored to
    whatever it held before this call — including when `eval_fn` raises —
    so a gate evaluation never permanently clobbers production.

    In real DAG usage (`airflow_dags/retraining_dag.py`), the prod model
    IS `PROD_MODEL_SLOT` — `evaluate_and_gate` is called with
    `prod_path == PROD_MODEL_SLOT`. Swapping a file onto itself
    (move-away-then-copy-from-itself) is not a no-op: it destroys the only
    copy before the copy step can read it. Detect that case up front and
    evaluate in place instead of swapping.
    """
    model_path = Path(model_path)
    if model_path.resolve() == PROD_MODEL_SLOT.resolve():
        results = eval_fn(num_queries=num_queries)
        return float(results[config_key]["NDCG@10"])

    backup_path = PROD_MODEL_SLOT.parent / f"{PROD_MODEL_SLOT.name}.promote_backup"
    had_prod = PROD_MODEL_SLOT.exists()
    PROD_MODEL_SLOT.parent.mkdir(parents=True, exist_ok=True)

    swapped = False
    try:
        if had_prod:
            PROD_MODEL_SLOT.replace(backup_path)
            swapped = True
        shutil.copy2(model_path, PROD_MODEL_SLOT)
        results = eval_fn(num_queries=num_queries)
        return float(results[config_key]["NDCG@10"])
    finally:
        # Invariant: never delete PROD_MODEL_SLOT unless a backup exists to
        # restore from. `swapped` (not `had_prod` or `.exists()`) is the only
        # reliable signal that the original was actually moved to
        # `backup_path` -- `PROD_MODEL_SLOT.exists()` can't tell "swap never
        # started" (original still there) from "swap completed" (candidate
        # there), and if `PROD_MODEL_SLOT.replace(backup_path)` itself raises
        # (e.g. a locked file on Windows), the original is still sitting at
        # PROD_MODEL_SLOT with no backup to restore from.
        if swapped:
            # Backup exists: remove whatever candidate copy landed (if any)
            # and restore the original from the backup.
            if PROD_MODEL_SLOT.exists():
                PROD_MODEL_SLOT.unlink()
            backup_path.replace(PROD_MODEL_SLOT)
        elif not had_prod:
            # No prod model to begin with and no backup was ever made: clean
            # up any candidate copy so the slot ends up empty again, exactly
            # as before this call.
            if PROD_MODEL_SLOT.exists():
                PROD_MODEL_SLOT.unlink()
        # else: had_prod was True but the swap never completed (the initial
        # `replace` raised before it could move the original) -- the
        # original is still intact at PROD_MODEL_SLOT and there is no
        # backup, so leave it completely untouched.


def evaluate_and_gate(
    prod_path: Path,
    staging_path: Path,
    margin: float,
    num_queries: int,
    eval_fn: Callable[..., dict] = run_evaluation,
) -> dict:
    """Evaluate `prod_path` and `staging_path` under the SAME `eval_fn` and
    decide whether staging should be promoted.

    Both NDCG@10 numbers come from identical eval conditions (same
    num_queries, same config_key, back-to-back calls), which is the fix
    for the apples-to-oranges bug: no more comparing a fresh staging
    number against a stale MLflow-logged prod number.
    """
    prod_ndcg = evaluate_model_ndcg(prod_path, num_queries, eval_fn=eval_fn)
    staging_ndcg = evaluate_model_ndcg(staging_path, num_queries, eval_fn=eval_fn)
    delta = staging_ndcg - prod_ndcg

    return {
        "prod_ndcg": prod_ndcg,
        "staging_ndcg": staging_ndcg,
        "delta": delta,
        "promote": delta >= margin,
    }
