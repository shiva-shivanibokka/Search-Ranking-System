"""
Click-feedback retraining — the free (GitHub Actions) replacement for the Airflow
DAG. Mirrors airflow_dags/retraining_dag.py but runs as a single CLI so it can be
scheduled on free CI instead of an always-on Airflow.

Flow:
  1. Read click_logs from Postgres (Neon via DATABASE_URL). If fewer than
     RETRAINING_CLICK_THRESHOLD clicks, exit 0 (nothing to do).
  2. Build the same 7-feature LambdaRank vectors from clicks (BM25 + two-tower +
     passage stats), grouped by query.
  3. Retrain XGBoost rank:ndcg.
  4. Save models/lambdarank/lambdarank.json and (if HF creds are set) publish it
     back to the HF Hub artifact repo so the live Space picks it up on restart.

Prereqs: serving artifacts present (run scripts/bootstrap.py first) and
DATABASE_URL pointing at the clicks DB.

Note: the full NDCG@10 promotion gate runs in the local Airflow pipeline (it needs
the dev-set eval harness). This CI job is threshold-gated retrain + publish; treat
the Airflow DAG as the gated path and this as the lightweight scheduled path.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

THRESHOLD = int(os.getenv("RETRAINING_CLICK_THRESHOLD", "1000"))


def _load_clicks() -> pd.DataFrame:
    from sqlalchemy import text

    from services.shared.database import get_engine

    with get_engine().connect() as conn:
        return pd.read_sql(
            text(
                "SELECT query_text, doc_id, rank_shown, ranker_version "
                "FROM click_logs ORDER BY created_at DESC LIMIT 50000"
            ),
            conn,
        )


def _build_features(click_df: pd.DataFrame):
    import torch

    from training.two_tower_model import load_two_tower

    data = PROJECT_ROOT / "data"
    models = PROJECT_ROOT / "models"
    with open(data / "indexes" / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(data / "indexes" / "bm25_pid_list.pkl", "rb") as f:
        bm25_pid_list = pickle.load(f)
    bm25_idx = {pid: i for i, pid in enumerate(bm25_pid_list)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tt_model, tok = load_two_tower(str(models / "two_tower"), device=device)

    pdf = pd.read_parquet(data / "processed" / "passages.parquet")
    pid_to_text = dict(zip(pdf["pid"], pdf["text"]))
    pid_to_len = dict(zip(pdf["pid"], pdf["token_count"]))

    X, y, groups = [], [], []
    for query_text, group in click_df.groupby("query_text"):
        q_terms = set(query_text.lower().split())
        q_len = len(query_text.split())
        bm25_all = bm25.get_scores(query_text.lower().split())
        rows = 0
        for _, row in group.iterrows():
            pid = int(row["doc_id"])
            doc_text = pid_to_text.get(pid, "")
            bm25_score = float(bm25_all[bm25_idx.get(pid, 0)])
            overlap = len(q_terms & set(doc_text.lower().split())) / max(len(q_terms), 1)
            doc_len = pid_to_len.get(pid, 0)

            enc_q = tok(query_text, max_length=64, truncation=True, return_tensors="pt")
            enc_d = tok(doc_text, max_length=180, truncation=True, return_tensors="pt")
            with torch.no_grad():
                q_emb = tt_model.encode_query(
                    enc_q["input_ids"].to(device), enc_q["attention_mask"].to(device)
                ).cpu().numpy()
                d_emb = tt_model.encode_doc(
                    enc_d["input_ids"].to(device), enc_d["attention_mask"].to(device)
                ).cpu().numpy()
            tt_score = float((d_emb @ q_emb.T).squeeze())
            rank_norm = float(row["rank_shown"]) / 10.0

            X.append([bm25_score, tt_score, min(doc_len / 200.0, 5.0), overlap,
                      min(q_len / 20.0, 3.0), rank_norm, rank_norm])
            y.append(1)
            rows += 1
        if rows:
            groups.append(rows)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), groups


def _train_and_save(X, y, groups) -> Path:
    import xgboost as xgb

    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_group(groups if groups else [len(X)])
    params = {
        "objective": "rank:ndcg", "eval_metric": "ndcg@10",
        "eta": 0.05, "max_depth": 6, "subsample": 0.8, "tree_method": "hist", "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=300)
    out = PROJECT_ROOT / "models" / "lambdarank" / "lambdarank.json"
    booster.save_model(str(out))
    return out


def _publish(model_path: Path) -> None:
    repo = os.getenv("HF_ARTIFACTS_REPO", "")
    token = os.getenv("HF_TOKEN")
    if not repo or repo.startswith("REPLACE_ME") or not token:
        print("HF creds not set — skipping publish (model saved locally only).")
        return
    from huggingface_hub import HfApi

    HfApi(token=token).upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="models/lambdarank/lambdarank.json",
        repo_id=repo,
        repo_type=os.getenv("HF_ARTIFACTS_REPO_TYPE", "model"),
    )
    print(f"Published updated LambdaRank model to {repo}.")


def main() -> int:
    click_df = _load_clicks()
    n = len(click_df)
    print(f"Loaded {n} click events (threshold {THRESHOLD}).")
    if n < THRESHOLD:
        print("Below threshold — nothing to retrain.")
        return 0

    X, y, groups = _build_features(click_df)
    if len(X) == 0:
        print("No usable click features — exiting.")
        return 0
    print(f"Built {len(X)} feature rows across {len(groups)} query groups.")

    model_path = _train_and_save(X, y, groups)
    print(f"Saved retrained model to {model_path}.")

    # Record what happened for the workflow log.
    summary = {"clicks": int(n), "feature_rows": int(len(X)), "groups": int(len(groups))}
    print("RETRAIN_SUMMARY " + json.dumps(summary))

    _publish(model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
