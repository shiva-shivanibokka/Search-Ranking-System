"""
Airflow DAG: Automated LambdaRank Retraining Pipeline

Trigger conditions (checked daily at 2am):
  1. New click events >= RETRAINING_CLICK_THRESHOLD (default 1000)
     → Retrain LambdaRank on accumulated click data as implicit labels
  2. Scheduled weekly full retrain regardless of click count

Pipeline steps:
  check_click_threshold
    ↓ (if threshold reached)
  extract_click_features
    ↓
  train_lambdarank_with_clicks
    ↓
  evaluate_new_model
    ↓
  promote_if_better (NDCG@10 must improve by ≥1%)
    ↓ (if promoted)
  hot_reload_ranking_service
    ↓
  notify_completion
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import mlflow

logger = logging.getLogger(__name__)

# ── Project root (resolves correctly regardless of Airflow CWD) ───────────────
PROJECT_ROOT = Path(os.getenv("AIRFLOW_PROJECT_ROOT", "/opt/airflow/project"))

# ── DAG Config ────────────────────────────────────────────────────────────────

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

CLICK_THRESHOLD = int(os.getenv("RETRAINING_CLICK_THRESHOLD", "1000"))
NDCG_IMPROVEMENT_THRESHOLD = float(os.getenv("RETRAINING_NDCG_IMPROVEMENT", "0.01"))
FEEDBACK_URL = os.getenv("FEEDBACK_URL", "http://feedback:8004")
RANKING_URL = os.getenv("RANKING_URL", "http://ranking:8003")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
POSTGRES_DSN = (
    f"postgresql+psycopg2://"
    f"{os.getenv('POSTGRES_USER', 'searchuser')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'searchpass')}@"
    f"{os.getenv('POSTGRES_HOST', 'postgres')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'search_ranking')}"
)


# ── Task Functions ────────────────────────────────────────────────────────────


def check_click_threshold(**context):
    """
    Check if enough new click events have accumulated since last retrain.
    Returns 'retrain' or 'skip' to branch the DAG.
    """
    try:
        resp = requests.get(f"{FEEDBACK_URL}/stats", timeout=10)
        resp.raise_for_status()
        stats = resp.json()
        total_clicks = stats["total_clicks"]
        logger.info(f"Total clicks: {total_clicks}, threshold: {CLICK_THRESHOLD}")

        context["task_instance"].xcom_push(key="total_clicks", value=total_clicks)

        if total_clicks >= CLICK_THRESHOLD:
            logger.info("Threshold reached. Proceeding with retraining.")
            return "extract_click_features"
        else:
            logger.info(
                f"Threshold not reached ({total_clicks}/{CLICK_THRESHOLD}). Skipping."
            )
            return "skip_retraining"
    except Exception as e:
        logger.error(f"Failed to check click threshold: {e}")
        return "skip_retraining"


def extract_click_features(**context):
    """
    Extract training features from labeled impressions (impressions LEFT
    JOIN clicks), delegating to scripts.retrain_from_clicks so this DAG and
    the free-tier CI retrain job (scripts/retrain_from_clicks.py) never
    drift: same query, same feature builder, same labeling.

    Labels are NOT all-1: clicked -> 1.0, shown-but-not-clicked -> 0.0 (a
    real negative from impression_logs), IPS-weighted by
    1/propensity[rank] on clicked rows to correct for position bias.
    """
    import sys

    sys.path.insert(0, "/opt/airflow/project")

    from sqlalchemy import create_engine

    from scripts.retrain_from_clicks import (
        _load_bm25,
        _load_propensity,
        _load_retriever,
        build_training_matrix,
        load_labeled_impressions,
    )

    engine = create_engine(POSTGRES_DSN)
    labeled = load_labeled_impressions(engine)
    logger.info(f"Loaded {len(labeled)} labeled impression rows")

    retriever = _load_retriever()
    bm25, bm25_pid_list, pid_to_len = _load_bm25()
    propensity = _load_propensity()

    X, y, weights, groups = build_training_matrix(
        labeled, retriever, bm25, bm25_pid_list, pid_to_len, propensity
    )

    np.save(PROJECT_ROOT / "data/processed/click_train_X.npy", X)
    np.save(PROJECT_ROOT / "data/processed/click_train_y.npy", y)
    np.save(PROJECT_ROOT / "data/processed/click_train_weights.npy", weights)
    with open(PROJECT_ROOT / "data/processed/click_groups.json", "w") as f:
        json.dump(groups, f)

    context["task_instance"].xcom_push(key="num_click_features", value=len(X))
    logger.info(f"Extracted {len(X)} labeled feature rows across {len(groups)} groups")


def train_lambdarank_with_clicks(**context):
    """Retrain LambdaRank on the propensity-weighted feature matrix, via the
    shared scripts.retrain_from_clicks.train_and_save helper (single source
    of truth for DMatrix construction / params, so this DAG and the CI
    retrain job stay in lockstep). Aborts (no train, no staging artifact) if
    the labels that came out are degenerate.

    scripts.retrain_from_clicks.train_and_save always writes to the
    production path models/lambdarank/lambdarank.json (that is its contract
    for the lightweight CI job, which has no staging/prod distinction). This
    DAG DOES have a staging/prod distinction — evaluate_new_model/
    promote_if_better must be able to compare the new candidate against the
    model that is genuinely still in production. So the real production file
    is backed up before training and restored immediately after the new
    model is moved to the staging path, ensuring train_and_save's write
    never actually clobbers production here.
    """
    import sys

    sys.path.insert(0, "/opt/airflow/project")

    from scripts.retrain_from_clicks import is_degenerate, train_and_save

    X = np.load(PROJECT_ROOT / "data/processed/click_train_X.npy")
    y = np.load(PROJECT_ROOT / "data/processed/click_train_y.npy")
    weights = np.load(PROJECT_ROOT / "data/processed/click_train_weights.npy")
    with open(PROJECT_ROOT / "data/processed/click_groups.json") as f:
        groups = json.load(f)

    if is_degenerate(y):
        logger.warning("Degenerate labels — abort (no train, no staging artifact).")
        context["task_instance"].xcom_push(key="staging_run_id", value=None)
        return

    prod_path = PROJECT_ROOT / "models/lambdarank/lambdarank.json"
    staging_path = PROJECT_ROOT / "models/lambdarank/lambdarank_staging.json"
    prod_backup_path = PROJECT_ROOT / "models/lambdarank/lambdarank_prod_backup.json"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("neural-search-ranking")

    if prod_path.exists():
        prod_path.replace(prod_backup_path)

    try:
        with mlflow.start_run(run_name="lambdarank_click_retrain") as run:
            model_path = train_and_save(X, y, weights, groups)  # writes to prod_path

            # Move the newly-trained candidate to staging (not production yet).
            Path(model_path).replace(staging_path)
            mlflow.log_artifact(str(staging_path))
            run_id = run.info.run_id
    finally:
        # Restore the real production model regardless of outcome above —
        # train_and_save must never be the thing that promotes a model.
        if prod_backup_path.exists():
            prod_backup_path.replace(prod_path)

    context["task_instance"].xcom_push(key="staging_run_id", value=run_id)
    logger.info(f"Retrained LambdaRank saved to staging. MLflow run: {run_id}")


def evaluate_new_model(**context):
    """
    Evaluate staging model on dev set and compare to current production model.
    Pushes ndcg_delta to XCom for promotion decision.
    """
    import sys

    sys.path.insert(0, "/opt/airflow/project")

    from training.evaluate import run_evaluation

    # Temporarily swap in staging model for evaluation
    staging_path = PROJECT_ROOT / "models/lambdarank/lambdarank_staging.json"
    prod_path = PROJECT_ROOT / "models/lambdarank/lambdarank.json"
    backup_path = PROJECT_ROOT / "models/lambdarank/lambdarank_backup.json"

    if prod_path.exists():
        prod_path.rename(backup_path)
    staging_path.rename(prod_path)

    try:
        results = run_evaluation(num_queries=1000)
        staging_ndcg = results.get("TwoTower+LambdaRank", {}).get("NDCG@10", 0.0)
    finally:
        # Restore production model — always clean up regardless of eval outcome
        if prod_path.exists():
            prod_path.rename(staging_path)
        if backup_path.exists():
            backup_path.rename(prod_path)

    # Get current prod NDCG from MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=["1"],
        filter_string="tags.mlflow.runName = 'lambdarank_training'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    prod_ndcg = (
        float(runs[0].data.metrics.get("final_dev_ndcg10", 0.0)) if runs else 0.0
    )

    ndcg_delta = staging_ndcg - prod_ndcg
    context["task_instance"].xcom_push(key="staging_ndcg", value=staging_ndcg)
    context["task_instance"].xcom_push(key="prod_ndcg", value=prod_ndcg)
    context["task_instance"].xcom_push(key="ndcg_delta", value=ndcg_delta)

    logger.info(
        f"Staging NDCG@10: {staging_ndcg:.4f}, Prod NDCG@10: {prod_ndcg:.4f}, Delta: {ndcg_delta:.4f}"
    )


def promote_if_better(**context):
    """
    Promote staging → production if NDCG@10 improved by >= threshold.
    Gate: improvement must be >= NDCG_IMPROVEMENT_THRESHOLD (default 0.01).
    """
    ti = context["task_instance"]
    ndcg_delta = ti.xcom_pull(key="ndcg_delta", task_ids="evaluate_new_model")
    staging_ndcg = ti.xcom_pull(key="staging_ndcg", task_ids="evaluate_new_model")

    if ndcg_delta >= NDCG_IMPROVEMENT_THRESHOLD:
        staging_path = PROJECT_ROOT / "models/lambdarank/lambdarank_staging.json"
        prod_path = PROJECT_ROOT / "models/lambdarank/lambdarank.json"
        old_prod_path = (
            PROJECT_ROOT
            / f"models/lambdarank/lambdarank_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        if prod_path.exists():
            prod_path.rename(old_prod_path)
        staging_path.rename(prod_path)

        logger.info(
            f"PROMOTED: NDCG delta={ndcg_delta:.4f} >= threshold={NDCG_IMPROVEMENT_THRESHOLD}"
        )
        context["task_instance"].xcom_push(key="promoted", value=True)
        return "hot_reload_ranking_service"
    else:
        logger.info(
            f"REJECTED: NDCG delta={ndcg_delta:.4f} < threshold={NDCG_IMPROVEMENT_THRESHOLD}"
        )
        context["task_instance"].xcom_push(key="promoted", value=False)
        return "notify_completion"


def hot_reload_ranking_service(**context):
    """Tell the ranking service to reload its LambdaRank model from disk."""
    try:
        resp = requests.post(f"{RANKING_URL}/reload/lambdarank", timeout=30)
        resp.raise_for_status()
        logger.info(f"Ranking service hot-reloaded: {resp.json()}")
    except Exception as e:
        logger.error(f"Hot reload failed: {e}")
        raise


def notify_completion(**context):
    """Log final summary of the retraining run."""
    ti = context["task_instance"]
    promoted = ti.xcom_pull(key="promoted", task_ids="promote_if_better") or False
    staging_ndcg = ti.xcom_pull(key="staging_ndcg", task_ids="evaluate_new_model") or 0
    prod_ndcg = ti.xcom_pull(key="prod_ndcg", task_ids="evaluate_new_model") or 0

    status = "PROMOTED" if promoted else "REJECTED"
    logger.info(
        f"Retraining complete | Status: {status} | "
        f"Staging NDCG@10: {staging_ndcg:.4f} | "
        f"Prod NDCG@10: {prod_ndcg:.4f}"
    )


# ── DAG Definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="lambdarank_retraining",
    default_args=default_args,
    description="Automated LambdaRank retraining from click feedback",
    schedule="0 2 * * *",  # daily at 2am
    catchup=False,
    tags=["ml", "ranking", "retraining"],
) as dag:
    check_threshold = BranchPythonOperator(
        task_id="check_click_threshold",
        python_callable=check_click_threshold,
    )

    skip = EmptyOperator(task_id="skip_retraining")

    extract_features = PythonOperator(
        task_id="extract_click_features",
        python_callable=extract_click_features,
    )

    train = PythonOperator(
        task_id="train_lambdarank_with_clicks",
        python_callable=train_lambdarank_with_clicks,
    )

    evaluate = PythonOperator(
        task_id="evaluate_new_model",
        python_callable=evaluate_new_model,
    )

    promote = BranchPythonOperator(
        task_id="promote_if_better",
        python_callable=promote_if_better,
    )

    hot_reload = PythonOperator(
        task_id="hot_reload_ranking_service",
        python_callable=hot_reload_ranking_service,
    )

    notify = PythonOperator(
        task_id="notify_completion",
        python_callable=notify_completion,
        trigger_rule="none_failed_min_one_success",
    )

    # DAG topology
    check_threshold >> [extract_features, skip]
    extract_features >> train >> evaluate >> promote
    promote >> [hot_reload, notify]
    hot_reload >> notify
