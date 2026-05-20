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
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
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
    Extract training features from click logs.
    Clicks are treated as implicit positive relevance signals:
      - Clicked document at rank K → label 1
      - Non-clicked documents shown → label 0 (skipped for now, future work)

    Merges click data with BM25 + two-tower features to build
    a combined feature matrix for LambdaRank retraining.
    """
    from sqlalchemy import create_engine, text
    import torch
    import sys

    sys.path.insert(0, "/opt/airflow/project")

    from training.two_tower_model import load_two_tower

    engine = create_engine(POSTGRES_DSN)
    with engine.connect() as conn:
        click_df = pd.read_sql(
            text(
                "SELECT query_text, doc_id, rank_shown, ranker_version FROM click_logs ORDER BY created_at DESC LIMIT 50000"
            ),
            conn,
        )

    logger.info(f"Loaded {len(click_df)} click events")

    # Load artifacts
    with open(PROJECT_ROOT / "data/indexes/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(PROJECT_ROOT / "data/indexes/bm25_pid_list.pkl", "rb") as f:
        bm25_pid_list = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tt_model, tokenizer = load_two_tower(
        str(PROJECT_ROOT / "models/two_tower"), device=device
    )

    passages_df = pd.read_parquet(PROJECT_ROOT / "data/processed/passages.parquet")
    pid_to_text = dict(zip(passages_df["pid"], passages_df["text"]))
    pid_to_len = dict(zip(passages_df["pid"], passages_df["token_count"]))

    X_rows, y_rows, groups = [], [], []
    pid_to_bm25_idx = {pid: i for i, pid in enumerate(bm25_pid_list)}

    for query_text, group_df in click_df.groupby("query_text"):
        q_terms = set(query_text.lower().split())
        q_len = len(query_text.split())

        bm25_scores_all = bm25.get_scores(query_text.lower().split())

        # Build feature row for each clicked doc
        group_rows = []
        for _, row in group_df.iterrows():
            pid = int(row["doc_id"])
            doc_text = pid_to_text.get(pid, "")

            bm25_idx = pid_to_bm25_idx.get(pid, 0)
            bm25_score = float(bm25_scores_all[bm25_idx])
            doc_terms = set(doc_text.lower().split())
            overlap = len(q_terms & doc_terms) / max(len(q_terms), 1)
            doc_len = pid_to_len.get(pid, 0)

            # Two-tower score
            enc_q = tokenizer(
                query_text,
                max_length=64,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc_d = tokenizer(
                doc_text,
                max_length=180,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                q_emb = (
                    tt_model.encode_query(
                        enc_q["input_ids"].to(device),
                        enc_q["attention_mask"].to(device),
                    )
                    .cpu()
                    .numpy()
                )
                d_emb = (
                    tt_model.encode_doc(
                        enc_d["input_ids"].to(device),
                        enc_d["attention_mask"].to(device),
                    )
                    .cpu()
                    .numpy()
                )
            tt_score = float((d_emb @ q_emb.T).squeeze())

            rank_norm = float(row["rank_shown"]) / 10.0

            group_rows.append(
                {
                    "features": [
                        bm25_score,
                        tt_score,
                        min(doc_len / 200.0, 5.0),
                        overlap,
                        min(q_len / 20.0, 3.0),
                        rank_norm,
                        rank_norm,
                    ],
                    "label": 1,  # all click events are positives
                }
            )

        if group_rows:
            for r in group_rows:
                X_rows.append(r["features"])
                y_rows.append(r["label"])
            groups.append(len(group_rows))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)

    # Merge with existing MS MARCO training data for stability
    existing_X = (
        np.load("data/processed/lambdarank_train_X.npy")
        if Path("data/processed/lambdarank_train_X.npy").exists()
        else X
    )
    existing_y = (
        np.load("data/processed/lambdarank_train_y.npy")
        if Path("data/processed/lambdarank_train_y.npy").exists()
        else y
    )

    X_combined = np.vstack([existing_X, X])
    y_combined = np.concatenate([existing_y, y])

    np.save(PROJECT_ROOT / "data/processed/click_train_X.npy", X_combined)
    np.save(PROJECT_ROOT / "data/processed/click_train_y.npy", y_combined)
    with open(PROJECT_ROOT / "data/processed/click_groups.json", "w") as f:
        json.dump(groups, f)

    context["task_instance"].xcom_push(key="num_click_features", value=len(X_rows))
    logger.info(f"Extracted {len(X_rows)} click feature rows")


def train_lambdarank_with_clicks(**context):
    """Retrain LambdaRank on click-augmented feature matrix."""
    X = np.load(PROJECT_ROOT / "data/processed/click_train_X.npy")
    y = np.load(PROJECT_ROOT / "data/processed/click_train_y.npy")
    with open(PROJECT_ROOT / "data/processed/click_groups.json") as f:
        groups = json.load(f)

    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(groups if groups else [len(X)])

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "tree_method": "hist",
        "seed": 42,
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("neural-search-ranking")

    with mlflow.start_run(run_name="lambdarank_click_retrain") as run:
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=300,
            callbacks=[xgb.callback.EvaluationMonitor(period=50)],
        )

        # Save to staging path (not production yet)
        staging_path = str(PROJECT_ROOT / "models/lambdarank/lambdarank_staging.json")
        booster.save_model(staging_path)
        mlflow.log_artifact(staging_path)
        run_id = run.info.run_id

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
