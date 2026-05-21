"""
Train a LambdaRank reranker using XGBoost.

LambdaRank is a Learning-to-Rank model that directly optimizes NDCG.
It uses XGBoost's rank:ndcg objective with hand-crafted features per
(query, document) pair.

Features:
  - bm25_score         : BM25 relevance score
  - two_tower_cosine   : cosine similarity from two-tower
  - doc_length         : passage token count (normalized)
  - query_term_overlap : fraction of query terms found in document
  - query_length       : number of query tokens (normalized)
  - bm25_rank          : rank position in BM25 top-100 (1-indexed)
  - two_tower_rank     : rank position in two-tower top-100 (1-indexed)

Training labels: binary relevance from qrels (1 = relevant, 0 = not relevant)
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
import mlflow
from rich.console import Console
import torch
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from training.two_tower_model import load_two_tower
from configs.training_config import get_training_config

console = Console()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_bm25_scores(
    bm25, pid_list: list, query_text: str, candidate_pids: list
) -> dict:
    """Get BM25 scores for a set of candidate pids."""
    tokenized = query_text.lower().split()
    all_scores = bm25.get_scores(tokenized)
    pid_to_idx = {pid: i for i, pid in enumerate(pid_list)}
    return {
        pid: float(all_scores[pid_to_idx[pid]])
        for pid in candidate_pids
        if pid in pid_to_idx
    }


def compute_two_tower_scores(
    model,
    tokenizer,
    query_text: str,
    candidate_texts: list,
    max_q_len: int = 64,
    max_d_len: int = 180,
) -> np.ndarray:
    """Compute cosine similarity between query and candidate docs."""
    enc_q = tokenizer(
        query_text,
        max_length=max_q_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        q_emb = (
            model.encode_query(
                enc_q["input_ids"].to(DEVICE),
                enc_q["attention_mask"].to(DEVICE),
            )
            .cpu()
            .numpy()
        )  # (1, D)

    enc_d = tokenizer(
        candidate_texts,
        max_length=max_d_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        d_emb = (
            model.encode_doc(
                enc_d["input_ids"].to(DEVICE),
                enc_d["attention_mask"].to(DEVICE),
            )
            .cpu()
            .numpy()
        )  # (N, D)

    scores = (d_emb @ q_emb.T).squeeze(-1)  # (N,)
    return scores


def build_feature_matrix(
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    passages_df: pd.DataFrame,
    bm25,
    bm25_pid_list: list,
    two_tower_model,
    tokenizer,
    faiss_index,
    faiss_pid_list: list,
    top_k: int = 100,
    max_queries: int = 5000,
) -> tuple:
    """
    Build feature matrix for LambdaRank training.

    Uses FAISS for fast candidate retrieval instead of per-query BM25 scan.
    FAISS ANN search returns top-K candidates in <5ms vs ~1.3s for BM25.

    Returns:
      X: (N_pairs, n_features) feature matrix
      y: (N_pairs,) binary relevance labels
      groups: list of group sizes (docs per query) — required by XGBoost rank
    """
    pid_to_text = dict(zip(passages_df["pid"], passages_df["text"]))
    pid_to_len = dict(zip(passages_df["pid"], passages_df["token_count"]))
    pos_pids_by_qid = qrels_df.groupby("qid")["pid"].apply(set).to_dict()

    # Only use queries that have at least one positive
    queries_with_pos = set(qrels_df["qid"].unique())
    queries_sample = queries_df[queries_df["qid"].isin(queries_with_pos)].head(
        max_queries
    )

    X_rows, y_rows, groups = [], [], []

    for _, row in tqdm(
        queries_sample.iterrows(), total=len(queries_sample), desc="Building features"
    ):
        qid = row["qid"]
        query_text = str(row["text"])
        gold_pids = pos_pids_by_qid.get(qid, set())

        # Fast FAISS retrieval for candidates
        enc_q = tokenizer(
            query_text,
            max_length=64,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            q_emb = (
                two_tower_model.encode_query(
                    enc_q["input_ids"].to(DEVICE),
                    enc_q["attention_mask"].to(DEVICE),
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        scores_faiss, indices = faiss_index.search(q_emb, top_k)
        candidate_pids = [faiss_pid_list[i] for i in indices[0] if i >= 0]
        tt_scores_faiss = [
            float(scores_faiss[0][r]) for r in range(len(candidate_pids))
        ]

        if not candidate_pids:
            continue

        candidate_texts = [pid_to_text.get(p, "") for p in candidate_pids]

        # BM25 scores: compute term frequency overlap manually for candidates only
        # Avoids scanning all 500K passages — just score the 100 FAISS candidates
        tokenized_q = query_text.lower().split()
        q_term_set = set(tokenized_q)
        bm25_scores = []
        for p in candidate_pids:
            doc_text = pid_to_text.get(p, "")
            doc_terms = doc_text.lower().split()
            # Simple TF-based proxy score (fast, avoids full BM25 scan)
            tf_score = sum(doc_terms.count(t) for t in q_term_set)
            bm25_scores.append(float(tf_score))

        # TT rank from FAISS order (already sorted by cosine sim)
        n = len(candidate_pids)
        tt_rank_arr = np.arange(1, n + 1)
        bm25_rank_arr = np.argsort(np.argsort([-s for s in bm25_scores])) + 1

        q_len = len(query_text.split())
        q_terms = set(query_text.lower().split())

        group_size = 0
        for rank_i, pid in enumerate(candidate_pids):
            doc_text = pid_to_text.get(pid, "")
            doc_len = pid_to_len.get(pid, 0)
            overlap = len(q_terms & set(doc_text.lower().split())) / max(
                len(q_terms), 1
            )

            features = [
                bm25_scores[rank_i],
                tt_scores_faiss[rank_i],
                min(doc_len / 200.0, 5.0),
                overlap,
                min(q_len / 20.0, 3.0),
                bm25_rank_arr[rank_i] / n,
                tt_rank_arr[rank_i] / n,
            ]
            X_rows.append(features)
            y_rows.append(1 if pid in gold_pids else 0)
            group_size += 1

        if group_size > 0:
            groups.append(group_size)

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    console.print(
        f"[green]Feature matrix: {X.shape}, positives: {int(y.sum()):,}[/green]"
    )
    return X, y, groups


def train(config_path: str = "configs/config.yaml"):
    cfg = get_training_config(config_path)
    lr_cfg = cfg.lambdarank
    mlf_cfg = cfg.mlflow

    console.print(f"[bold]Training LambdaRank on {DEVICE}[/bold]")

    # ── Load dependencies ───────────────────────────────────────────────────────
    import faiss as faiss_lib

    with open("data/indexes/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open("data/indexes/bm25_pid_list.pkl", "rb") as f:
        bm25_pid_list = pickle.load(f)

    console.print("[cyan]Loading FAISS index...[/cyan]")
    faiss_index = faiss_lib.read_index("data/indexes/faiss_ivfpq.index")
    if hasattr(faiss_index, "nprobe"):
        faiss_index.nprobe = 64
    with open("data/indexes/docid_map.pkl", "rb") as f:
        faiss_pid_list = pickle.load(f)

    two_tower_model, tokenizer = load_two_tower(
        cfg.two_tower.save_dir, device=str(DEVICE)
    )

    passages_df = pd.read_parquet("data/processed/passages.parquet")
    train_queries_df = pd.read_parquet("data/processed/train_queries.parquet")
    train_qrels_df = pd.read_parquet("data/processed/train_qrels.parquet")
    dev_queries_df = pd.read_parquet("data/processed/dev_queries.parquet")
    dev_qrels_df = pd.read_parquet("data/processed/dev_qrels.parquet")

    # ── Build feature matrices ──────────────────────────────────────────────────
    console.print("[cyan]Building train feature matrix...[/cyan]")
    X_train, y_train, groups_train = build_feature_matrix(
        train_queries_df,
        train_qrels_df,
        passages_df,
        bm25,
        bm25_pid_list,
        two_tower_model,
        tokenizer,
        faiss_index,
        faiss_pid_list,
        max_queries=5000,
    )

    console.print("[cyan]Building dev feature matrix...[/cyan]")
    X_dev, y_dev, groups_dev = build_feature_matrix(
        dev_queries_df,
        dev_qrels_df,
        passages_df,
        bm25,
        bm25_pid_list,
        two_tower_model,
        tokenizer,
        faiss_index,
        faiss_pid_list,
        max_queries=500,
    )

    # ── XGBoost LambdaRank ──────────────────────────────────────────────────────
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(groups_train)

    ddev = xgb.DMatrix(X_dev, label=y_dev)
    ddev.set_group(groups_dev)

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "eta": lr_cfg.learning_rate,
        "max_depth": lr_cfg.max_depth,
        "subsample": lr_cfg.subsample,
        "colsample_bytree": lr_cfg.colsample_bytree,
        "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
    }

    evals = [(dtrain, "train"), (ddev, "dev")]
    evals_result = {}

    mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
    mlflow.set_experiment(mlf_cfg.experiment_name)

    with mlflow.start_run(run_name="lambdarank_training"):
        mlflow.log_params(
            {
                "n_estimators": lr_cfg.n_estimators,
                "max_depth": lr_cfg.max_depth,
                "learning_rate": lr_cfg.learning_rate,
                "subsample": lr_cfg.subsample,
                "features": lr_cfg.features,
            }
        )

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=lr_cfg.n_estimators,
            evals=evals,
            evals_result=evals_result,
            callbacks=[xgb.callback.EvaluationMonitor(period=50)],
        )

        save_dir = Path(lr_cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "lambdarank.json"
        booster.save_model(str(model_path))

        feature_names_path = save_dir / "feature_names.json"
        with open(feature_names_path, "w") as f:
            json.dump({"features": lr_cfg.features}, f, indent=2)

        # Log final NDCG
        final_train_ndcg = evals_result["train"]["ndcg@10"][-1]
        final_dev_ndcg = evals_result["dev"]["ndcg@10"][-1]
        console.print(f"\n[bold]Final Train NDCG@10: {final_train_ndcg:.4f}[/bold]")
        console.print(f"[bold]Final Dev NDCG@10:   {final_dev_ndcg:.4f}[/bold]")

        mlflow.log_metric("final_train_ndcg10", final_train_ndcg)
        mlflow.log_metric("final_dev_ndcg10", final_dev_ndcg)
        mlflow.log_artifact(str(model_path))

        console.print(f"\n[bold green]LambdaRank training complete.[/bold green]")
        console.print(f"Model saved → {model_path}")
        console.print("Next step: [cyan]python training/evaluate.py[/cyan]")


if __name__ == "__main__":
    train()
