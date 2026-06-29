"""
Train a Query Difficulty Classifier for cascaded ranking.

The classifier predicts whether a query is "hard" — meaning LambdaRank and
CrossEncoder disagree significantly on the ranking, and CrossEncoder is likely
needed to get good results.

Design rationale
────────────────
Not every query benefits from expensive CrossEncoder reranking. Short, specific
queries (e.g. "python list sort") often rank correctly with LambdaRank alone.
Complex, ambiguous informational queries (e.g. "what causes inflation in
developing economies") genuinely need CrossEncoder's deep query-document
interaction modelling to surface the right results.

This classifier lets us route intelligently:
  - Easy query  → LambdaRank  (fast, ~10ms, quality ≈ CrossEncoder)
  - Hard query  → CrossEncoder (slow, ~150ms, quality >> LambdaRank)

Training signal
───────────────
A query is labelled "hard" (1) if:
  - Kendall's tau between LambdaRank and CrossEncoder top-10 rankings < threshold
    (i.e. they strongly disagree on ordering), AND
  - CrossEncoder NDCG@10 - LambdaRank NDCG@10 > min_ndcg_gain
    (i.e. CrossEncoder is actually better, not just different)

Features (all computed from query + top-100 BM25 candidates, no doc encoding)
───────────────────────────────────────────────────────────────────────────────
  query_length          : number of tokens in query
  query_entropy         : entropy of token frequency distribution (measures specificity)
  bm25_score_gap        : score(rank_1) - score(rank_2) from BM25 (margin)
  tt_score_variance     : variance of two-tower cosine scores across top-10
  tt_bm25_score_ratio   : mean(tt_scores_top10) / (mean(bm25_scores_top10) + ε)
  intent_is_informational: 1 if rule-based intent = informational, else 0
  top1_bm25_score       : raw BM25 score of top result (absolute relevance signal)
  top1_tt_score         : raw two-tower cosine of top result

The key insight: if BM25 has a strong rank-1 margin (clear winner) and TT
scores are tightly clustered (low variance), the query is probably easy.
If BM25 scores are flat and TT variance is high, the rankers will disagree.
"""

import json
import pickle
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from rich.console import Console
from rich.table import Table
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kendalltau
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.training_config import get_training_config
from training.train_cross_encoder import load_cross_encoder
from training.two_tower_model import load_two_tower

console = Console()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Feature extraction ────────────────────────────────────────────────────────


def compute_query_entropy(query_text: str) -> float:
    """
    Entropy of the query token frequency distribution.
    High entropy = many distinct tokens = complex/broad query.
    Low entropy = repeated/few tokens = specific/short query.
    """
    tokens = query_text.lower().split()
    if not tokens:
        return 0.0
    token_counts = np.array([tokens.count(t) for t in set(tokens)], dtype=np.float32)
    probs = token_counts / token_counts.sum()
    return float(scipy_entropy(probs))


def compute_features(
    query_text: str,
    bm25,
    bm25_pid_list: list,
    tt_model,
    tt_tokenizer,
    pid_to_text: dict,
    top_k: int = 100,
    max_q_len: int = 64,
    max_d_len: int = 180,
) -> np.ndarray:
    """
    Compute the 8 difficulty features for a single query.
    Uses BM25 top-K as candidates for two-tower scoring.
    No document encoding needed beyond top-K BM25 candidates.
    """
    # BM25 retrieval
    tokenized = query_text.lower().split()
    bm25_scores_all = bm25.get_scores(tokenized)
    top_indices = bm25_scores_all.argsort()[::-1][:top_k]
    bm25_top_scores = np.array([float(bm25_scores_all[i]) for i in top_indices])

    # BM25 features
    bm25_score_gap = (
        float(bm25_top_scores[0] - bm25_top_scores[1])
        if len(bm25_top_scores) >= 2
        else 0.0
    )
    top1_bm25_score = float(bm25_top_scores[0]) if len(bm25_top_scores) > 0 else 0.0

    # Candidate texts for two-tower scoring
    candidate_pids = [bm25_pid_list[i] for i in top_indices[:10]]
    candidate_texts = [pid_to_text.get(p, "") for p in candidate_pids]

    # Two-tower scores for top-10 BM25 candidates
    enc_q = tt_tokenizer(
        query_text,
        max_length=max_q_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    enc_d = tt_tokenizer(
        candidate_texts,
        max_length=max_d_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        q_emb = (
            tt_model.encode_query(
                enc_q["input_ids"].to(DEVICE),
                enc_q["attention_mask"].to(DEVICE),
            )
            .cpu()
            .numpy()
        )
        d_emb = (
            tt_model.encode_doc(
                enc_d["input_ids"].to(DEVICE),
                enc_d["attention_mask"].to(DEVICE),
            )
            .cpu()
            .numpy()
        )
    tt_scores = (d_emb @ q_emb.T).squeeze(-1)  # (10,)

    tt_score_variance = float(np.var(tt_scores)) if len(tt_scores) > 1 else 0.0
    top1_tt_score = float(tt_scores[0]) if len(tt_scores) > 0 else 0.0

    mean_bm25 = (
        float(np.mean(bm25_top_scores[:10])) if len(bm25_top_scores) > 0 else 0.0
    )
    mean_tt = float(np.mean(tt_scores)) if len(tt_scores) > 0 else 0.0
    tt_bm25_score_ratio = mean_tt / (mean_bm25 + 1e-8)

    return np.array(
        [
            float(len(query_text.split())),  # query_length
            compute_query_entropy(query_text),  # query_entropy
            bm25_score_gap,  # bm25_score_gap
            tt_score_variance,  # tt_score_variance
            tt_bm25_score_ratio,  # tt_bm25_score_ratio
            0.0,  # intent_is_informational (filled later)
            top1_bm25_score,  # top1_bm25_score
            top1_tt_score,  # top1_tt_score
        ],
        dtype=np.float32,
    )


# ── Label generation ──────────────────────────────────────────────────────────


def generate_difficulty_labels(
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    passages_df: pd.DataFrame,
    bm25,
    bm25_pid_list: list,
    tt_model,
    tt_tokenizer,
    lr_booster: xgb.Booster,
    lr_feature_names: list,
    ce_model,
    ce_tokenizer,
    kendall_tau_threshold: float = 0.6,
    min_ndcg_gain: float = 0.05,
    top_k: int = 100,
    max_queries: int = 5000,
) -> tuple:
    """
    Generate (features, labels) for difficulty classifier training.

    Label logic:
      hard (1) = CrossEncoder significantly outperforms LambdaRank AND
                 they disagree on ranking (low Kendall tau)
      easy (0) = LambdaRank is close to CrossEncoder quality

    This gives us a clean, principled signal: label what CrossEncoder
    actually helps with, not just what it scores differently.
    """
    pid_to_text = dict(zip(passages_df["pid"], passages_df["text"]))
    pid_to_len = dict(zip(passages_df["pid"], passages_df["token_count"]))
    pos_pids_by_qid = qrels_df.groupby("qid")["pid"].apply(set).to_dict()

    queries_sample = queries_df.head(max_queries)

    X_rows, y_rows = [], []
    n_hard = 0
    n_easy = 0
    n_skipped = 0

    for _, row in tqdm(
        queries_sample.iterrows(),
        total=len(queries_sample),
        desc="Generating difficulty labels",
    ):
        qid = row["qid"]
        query_text = str(row["text"])
        gold_pids = pos_pids_by_qid.get(qid, set())
        if not gold_pids:
            n_skipped += 1
            continue

        # BM25 top-100 candidates
        tokenized = query_text.lower().split()
        bm25_scores_all = bm25.get_scores(tokenized)
        top_indices = bm25_scores_all.argsort()[::-1][:top_k]
        candidate_pids = [bm25_pid_list[i] for i in top_indices]
        candidate_bm25_scores = [float(bm25_scores_all[i]) for i in top_indices]
        candidate_texts = [pid_to_text.get(p, "") for p in candidate_pids]

        # Two-tower scores for all candidates
        enc_q = tt_tokenizer(
            query_text,
            max_length=64,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc_d = tt_tokenizer(
            candidate_texts,
            max_length=180,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            q_emb = (
                tt_model.encode_query(
                    enc_q["input_ids"].to(DEVICE), enc_q["attention_mask"].to(DEVICE)
                )
                .cpu()
                .numpy()
            )
            d_emb = (
                tt_model.encode_doc(
                    enc_d["input_ids"].to(DEVICE), enc_d["attention_mask"].to(DEVICE)
                )
                .cpu()
                .numpy()
            )
        tt_scores = (d_emb @ q_emb.T).squeeze(-1)

        # Build LambdaRank feature matrix
        n = len(candidate_pids)
        tt_rank_order = np.argsort(tt_scores)[::-1]
        tt_ranks = np.empty_like(tt_rank_order)
        tt_ranks[tt_rank_order] = np.arange(1, n + 1)

        q_terms = set(query_text.lower().split())
        q_len = len(query_text.split())
        bm25_pid_to_idx = {pid: i for i, pid in enumerate(bm25_pid_list)}

        X_lr = []
        for i, pid in enumerate(candidate_pids):
            bm25_idx = bm25_pid_to_idx.get(pid, 0)
            bm25_score = (
                float(bm25_scores_all[bm25_idx])
                if bm25_idx < len(bm25_scores_all)
                else 0.0
            )
            doc_text = pid_to_text.get(pid, "")
            doc_terms = set(doc_text.lower().split())
            overlap = len(q_terms & doc_terms) / max(len(q_terms), 1)
            doc_len = pid_to_len.get(pid, 0)
            X_lr.append(
                [
                    candidate_bm25_scores[i],
                    float(tt_scores[i]),
                    min(doc_len / 200.0, 5.0),
                    overlap,
                    min(q_len / 20.0, 3.0),
                    (i + 1) / n,
                    tt_ranks[i] / n,
                ]
            )

        dm = xgb.DMatrix(np.array(X_lr, dtype=np.float32))
        lr_scores = lr_booster.predict(dm)
        lr_top10 = list(np.argsort(lr_scores)[::-1][:10])
        lr_ranked_pids = [candidate_pids[i] for i in lr_top10]

        # CrossEncoder scores for top-100 candidates
        ce_scores_list = []
        batch_size = 32
        for i in range(0, len(candidate_pids), batch_size):
            batch_pids = candidate_pids[i : i + batch_size]
            batch_texts = [pid_to_text.get(p, "") for p in batch_pids]
            enc = ce_tokenizer(
                [query_text] * len(batch_texts),
                batch_texts,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                s = (
                    ce_model.predict_score(
                        enc["input_ids"].to(DEVICE),
                        enc["attention_mask"].to(DEVICE),
                    )
                    .cpu()
                    .numpy()
                )
            ce_scores_list.extend(zip(range(i, i + len(batch_pids)), s))

        ce_scores_list.sort(key=lambda x: x[1], reverse=True)
        ce_top10 = [candidate_pids[idx] for idx, _ in ce_scores_list[:10]]

        # Compute NDCG@10 for both rankers
        def ndcg10(ranked_pids):
            dcg = sum(
                1.0 / np.log2(r + 2)
                for r, pid in enumerate(ranked_pids[:10])
                if pid in gold_pids
            )
            ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(gold_pids), 10)))
            return dcg / ideal if ideal > 0 else 0.0

        lr_ndcg = ndcg10(lr_ranked_pids)
        ce_ndcg = ndcg10(ce_top10)
        ndcg_gain = ce_ndcg - lr_ndcg

        # Kendall's tau between the two top-10 rankings
        # Use shared pids only; lower tau = more disagreement
        shared = [p for p in lr_ranked_pids if p in ce_top10]
        if len(shared) >= 3:
            lr_order = [lr_ranked_pids.index(p) for p in shared]
            ce_order = [ce_top10.index(p) for p in shared]
            tau, _ = kendalltau(lr_order, ce_order)
        else:
            tau = -1.0  # maximum disagreement when almost no overlap

        # Hard if CE significantly outperforms AND rankings disagree
        is_hard = int(ndcg_gain >= min_ndcg_gain and tau <= kendall_tau_threshold)

        if is_hard:
            n_hard += 1
        else:
            n_easy += 1

        # Extract features (intent flag left as 0 here; ranking service will use live intent)
        features = compute_features(
            query_text, bm25, bm25_pid_list, tt_model, tt_tokenizer, pid_to_text
        )
        X_rows.append(features)
        y_rows.append(float(is_hard))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)

    console.print(
        f"[green]Labels: {n_hard} hard ({100 * n_hard / max(n_hard + n_easy, 1):.1f}%), "
        f"{n_easy} easy, {n_skipped} skipped[/green]"
    )
    return X, y


# ── Training ──────────────────────────────────────────────────────────────────


def train(config_path: str = "configs/config.yaml"):
    cfg = get_training_config(config_path)
    dc_cfg = cfg.difficulty_classifier
    lr_cfg = cfg.lambdarank
    mlf_cfg = cfg.mlflow

    console.print(f"[bold]Training Query Difficulty Classifier on {DEVICE}[/bold]")
    console.print(
        f"  Features: {dc_cfg.features}\n"
        f"  Routing threshold: {dc_cfg.routing_threshold}\n"
        f"  Save dir: {dc_cfg.save_dir}"
    )

    # ── Load all dependencies ──────────────────────────────────────────────────
    console.print("[cyan]Loading BM25 index...[/cyan]")
    with open(cfg.bm25.index_path, "rb") as f:
        bm25 = pickle.load(f)
    with open("data/indexes/bm25_pid_list.pkl", "rb") as f:
        bm25_pid_list = pickle.load(f)

    console.print("[cyan]Loading two-tower model...[/cyan]")
    tt_model, tt_tokenizer = load_two_tower(cfg.two_tower.save_dir, device=str(DEVICE))

    console.print("[cyan]Loading cross-encoder...[/cyan]")
    ce_model, ce_tokenizer = load_cross_encoder(
        cfg.cross_encoder.save_dir, device=str(DEVICE)
    )

    console.print("[cyan]Loading LambdaRank booster...[/cyan]")
    lr_booster = xgb.Booster()
    lr_booster.load_model(str(Path(lr_cfg.save_dir) / "lambdarank.json"))
    with open(Path(lr_cfg.save_dir) / "feature_names.json") as f:
        lr_feature_names = json.load(f)["features"]

    console.print("[cyan]Loading data...[/cyan]")
    passages_df = pd.read_parquet("data/processed/passages.parquet")
    dev_queries_df = pd.read_parquet("data/processed/dev_queries.parquet")
    dev_qrels_df = pd.read_parquet("data/processed/dev_qrels.parquet")
    train_queries_df = pd.read_parquet("data/processed/train_queries.parquet")
    train_qrels_df = pd.read_parquet("data/processed/train_qrels.parquet")

    # ── Generate labels on dev set (ground-truth signal from both rankers) ─────
    # We use dev set for label generation because dev_qrels has reliable relevance judgments.
    # We use train set (no qrels needed for features) for training diversity.
    console.print("\n[cyan]Generating difficulty labels on dev queries...[/cyan]")
    X_dev, y_dev = generate_difficulty_labels(
        dev_queries_df,
        dev_qrels_df,
        passages_df,
        bm25,
        bm25_pid_list,
        tt_model,
        tt_tokenizer,
        lr_booster,
        lr_feature_names,
        ce_model,
        ce_tokenizer,
        max_queries=2000,  # 2K dev queries for label generation
    )

    # Train/val split (80/20 within labeled dev queries)
    n = len(X_dev)
    n_train = int(n * 0.8)
    idx = np.random.RandomState(42).permutation(n)
    X_train, y_train = X_dev[idx[:n_train]], y_dev[idx[:n_train]]
    X_val, y_val = X_dev[idx[n_train:]], y_dev[idx[n_train:]]

    console.print(
        f"\n[green]Train: {len(X_train)} samples, Val: {len(X_val)} samples[/green]"
    )
    console.print(
        f"[green]Train positive rate: {y_train.mean():.3f} | "
        f"Val positive rate: {y_val.mean():.3f}[/green]"
    )

    # ── XGBoost binary classifier ──────────────────────────────────────────────
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=dc_cfg.features)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=dc_cfg.features)

    # Use scale_pos_weight to handle class imbalance (expect ~30-40% hard queries)
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "eta": dc_cfg.learning_rate,
        "max_depth": dc_cfg.max_depth,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
    }

    evals_result: dict = {}

    mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
    mlflow.set_experiment(mlf_cfg.experiment_name)

    with mlflow.start_run(run_name="difficulty_classifier_training"):
        mlflow.log_params(
            {
                "n_estimators": dc_cfg.n_estimators,
                "max_depth": dc_cfg.max_depth,
                "learning_rate": dc_cfg.learning_rate,
                "routing_threshold": dc_cfg.routing_threshold,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "positive_rate_train": float(y_train.mean()),
                "scale_pos_weight": scale_pos_weight,
                "features": dc_cfg.features,
            }
        )

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=dc_cfg.n_estimators,
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            callbacks=[xgb.callback.EvaluationMonitor(period=20)],
        )

        # ── Evaluate at threshold ──────────────────────────────────────────────
        val_preds = booster.predict(dval)
        val_labels_pred = (val_preds >= dc_cfg.routing_threshold).astype(int)
        accuracy = float((val_labels_pred == y_val.astype(int)).mean())
        precision = float(
            (val_labels_pred & y_val.astype(int)).sum() / max(val_labels_pred.sum(), 1)
        )
        recall = float(
            (val_labels_pred & y_val.astype(int)).sum() / max(y_val.sum(), 1)
        )
        ce_routing_rate = float(val_labels_pred.mean())  # % of queries routed to CE

        final_val_auc = evals_result["val"]["auc"][-1]
        final_val_logloss = evals_result["val"]["logloss"][-1]

        console.print("\n[bold]Difficulty Classifier Results[/bold]")
        table = Table(show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_row("Val AUC", f"{final_val_auc:.4f}")
        table.add_row("Val LogLoss", f"{final_val_logloss:.4f}")
        table.add_row("Accuracy", f"{accuracy:.4f}")
        table.add_row("Precision (hard)", f"{precision:.4f}")
        table.add_row("Recall (hard)", f"{recall:.4f}")
        table.add_row(
            "CE routing rate", f"{ce_routing_rate:.3f} ({ce_routing_rate * 100:.1f}%)"
        )
        console.print(table)

        mlflow.log_metrics(
            {
                "val_auc": final_val_auc,
                "val_logloss": final_val_logloss,
                "accuracy": accuracy,
                "precision_hard": precision,
                "recall_hard": recall,
                "ce_routing_rate": ce_routing_rate,
            }
        )

        # Feature importance
        importance = booster.get_score(importance_type="gain")
        console.print("\n[cyan]Feature importances (gain):[/cyan]")
        for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {feat}: {score:.2f}")

        # ── Save ──────────────────────────────────────────────────────────────
        save_dir = Path(dc_cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / "difficulty_classifier.json"
        booster.save_model(str(model_path))

        meta = {
            "features": dc_cfg.features,
            "routing_threshold": dc_cfg.routing_threshold,
            "val_auc": final_val_auc,
            "accuracy": accuracy,
            "ce_routing_rate": ce_routing_rate,
            "scale_pos_weight": scale_pos_weight,
        }
        with open(save_dir / "classifier_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(save_dir / "classifier_meta.json"))

    console.print(
        f"\n[bold green]Difficulty classifier saved → {model_path}[/bold green]"
    )
    console.print(
        f"  CE routing rate: {ce_routing_rate * 100:.1f}% of queries "
        f"(remainder use LambdaRank)"
    )
    console.print(
        "Next step: services will hot-load this model automatically on startup."
    )


if __name__ == "__main__":
    train()
