"""
Full offline evaluation of all retrieval and ranking configurations.

Computes and compares:
  BM25               → Recall@10, Recall@100, NDCG@10, MAP@10, MRR@10
  Two-Tower          → Recall@10, Recall@100, NDCG@10, MAP@10, MRR@10
  Two-Tower + LambdaRank  → NDCG@10, MAP@10, MRR@10
  Two-Tower + CrossEncoder → NDCG@10, MAP@10, MRR@10

Also produces:
  - Ablation: with vs without query rewriting
  - Latency vs quality scatter plot per config
  - Results saved to data/processed/eval_results.json
  - All metrics logged to MLflow
"""

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import faiss
import mlflow
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.training_config import get_training_config
from training.train_cross_encoder import load_cross_encoder
from training.two_tower_model import load_two_tower

console = Console()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Metric Helpers ────────────────────────────────────────────────────────────


def ndcg_at_k(ranked_pids: List[int], gold_pids: set, k: int) -> float:
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, pid in enumerate(ranked_pids[:k])
        if pid in gold_pids
    )
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(gold_pids), k)))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_k(ranked_pids: List[int], gold_pids: set, k: int) -> float:
    retrieved = set(ranked_pids[:k])
    return len(gold_pids & retrieved) / len(gold_pids) if gold_pids else 0.0


def mrr_at_k(ranked_pids: List[int], gold_pids: set, k: int) -> float:
    for rank, pid in enumerate(ranked_pids[:k]):
        if pid in gold_pids:
            return 1.0 / (rank + 1)
    return 0.0


def ap_at_k(ranked_pids: List[int], gold_pids: set, k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for rank, pid in enumerate(ranked_pids[:k]):
        if pid in gold_pids:
            hits += 1
            precision_sum += hits / (rank + 1)
    return precision_sum / min(len(gold_pids), k) if gold_pids else 0.0


def compute_metrics(ranked_pids: List[int], gold_pids: set) -> dict:
    return {
        "NDCG@10": ndcg_at_k(ranked_pids, gold_pids, 10),
        "MAP@10": ap_at_k(ranked_pids, gold_pids, 10),
        "MRR@10": mrr_at_k(ranked_pids, gold_pids, 10),
        "Recall@10": recall_at_k(ranked_pids, gold_pids, 10),
        "Recall@100": recall_at_k(ranked_pids, gold_pids, 100),
    }


# ── Retrieval Functions ───────────────────────────────────────────────────────


def retrieve_bm25(bm25, pid_list: list, query_text: str, top_k: int = 100) -> List[int]:
    scores = bm25.get_scores(query_text.lower().split())
    top_indices = scores.argsort()[::-1][:top_k]
    return [pid_list[i] for i in top_indices]


def retrieve_hybrid_rrf(
    bm25,
    bm25_pid_list: list,
    tt_model,
    tt_tokenizer,
    faiss_index,
    faiss_pid_list: list,
    query_text: str,
    top_k: int = 100,
    rrf_k: int = 60,
) -> List[int]:
    """
    Hybrid retrieval: fuse BM25 (sparse) + FAISS (dense) ranked lists with RRF.

    RRF score(d) = 1/(rrf_k + rank_bm25(d)) + 1/(rrf_k + rank_faiss(d))

    Documents appearing in both lists receive contributions from both terms,
    naturally boosting results both systems agree on.
    """
    # BM25 ranked list
    bm25_scores = bm25.get_scores(query_text.lower().split())
    bm25_top_indices = bm25_scores.argsort()[::-1][:top_k]
    bm25_ranked = [bm25_pid_list[i] for i in bm25_top_indices]

    # FAISS dense ranked list
    faiss_ranked = retrieve_two_tower(
        tt_model, tt_tokenizer, faiss_index, faiss_pid_list, query_text, top_k
    )

    # Reciprocal Rank Fusion
    rrf_scores: Dict[int, float] = {}
    for rank, pid in enumerate(bm25_ranked):
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, pid in enumerate(faiss_ranked):
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)

    sorted_pids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in sorted_pids[:top_k]]


def retrieve_two_tower(
    model,
    tokenizer,
    faiss_index,
    pid_list: list,
    query_text: str,
    top_k: int = 100,
    max_q_len: int = 64,
) -> List[int]:
    enc = tokenizer(
        query_text,
        max_length=max_q_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        q_emb = (
            model.encode_query(
                enc["input_ids"].to(DEVICE),
                enc["attention_mask"].to(DEVICE),
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    _, indices = faiss_index.search(q_emb, top_k)
    return [pid_list[i] for i in indices[0] if i >= 0]


# ── Reranking Functions ───────────────────────────────────────────────────────


def rerank_lambdarank(
    booster: xgb.Booster,
    feature_names: List[str],
    bm25,
    bm25_pid_list: list,
    tt_model,
    tokenizer,
    pid_to_text: dict,
    pid_to_len: dict,
    query_text: str,
    candidate_pids: List[int],
    top_k: int = 10,
) -> List[int]:
    """Rerank candidates using LambdaRank features."""
    if not candidate_pids:
        return []

    candidate_texts = [pid_to_text.get(p, "") for p in candidate_pids]
    # Two-tower scores
    enc_d = tokenizer(
        candidate_texts,
        max_length=180,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    enc_q = tokenizer(
        query_text, max_length=64, padding=True, truncation=True, return_tensors="pt"
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
    tt_scores_np = (d_emb @ q_emb.T).squeeze(-1)

    # BM25 scores
    bm25_scores_all = bm25.get_scores(query_text.lower().split())
    pid_to_bm25_idx = {pid: i for i, pid in enumerate(bm25_pid_list)}

    q_terms = set(query_text.lower().split())
    q_len = len(query_text.split())
    n = len(candidate_pids)

    tt_rank_order = np.argsort(tt_scores_np)[::-1]
    tt_ranks = np.empty_like(tt_rank_order)
    tt_ranks[tt_rank_order] = np.arange(1, n + 1)

    X = []
    for i, pid in enumerate(candidate_pids):
        bm25_idx = pid_to_bm25_idx.get(pid, 0)
        bm25_score = float(bm25_scores_all[bm25_idx])
        doc_text = pid_to_text.get(pid, "")
        doc_terms = set(doc_text.lower().split())
        overlap = len(q_terms & doc_terms) / max(len(q_terms), 1)
        doc_len = pid_to_len.get(pid, 0)

        X.append(
            [
                bm25_score,
                float(tt_scores_np[i]),
                min(doc_len / 200.0, 5.0),
                overlap,
                min(q_len / 20.0, 3.0),
                (i + 1) / n,
                tt_ranks[i] / n,
            ]
        )

    dm = xgb.DMatrix(np.array(X, dtype=np.float32))
    scores = booster.predict(dm)
    ranked_indices = scores.argsort()[::-1][:top_k]
    return [candidate_pids[i] for i in ranked_indices]


def rerank_cross_encoder(
    ce_model,
    tokenizer,
    pid_to_text: dict,
    query_text: str,
    candidate_pids: List[int],
    top_k: int = 10,
    max_seq_len: int = 256,
    batch_size: int = 32,
) -> List[int]:
    """Rerank candidates using cross-encoder."""
    scores = []
    for i in range(0, len(candidate_pids), batch_size):
        batch_pids = candidate_pids[i : i + batch_size]
        batch_texts = [pid_to_text.get(p, "") for p in batch_pids]
        enc = tokenizer(
            [query_text] * len(batch_texts),
            batch_texts,
            max_length=max_seq_len,
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
        scores.extend(zip(batch_pids, s))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scores[:top_k]]


# ── Main Evaluation ───────────────────────────────────────────────────────────


def run_evaluation(config_path: str = "configs/config.yaml", num_queries: int = 6980):
    cfg = get_training_config(config_path)
    mlf_cfg = cfg.mlflow
    hr_cfg = cfg.hybrid_retrieval

    console.print(f"[bold]Running full evaluation on {num_queries} dev queries[/bold]")

    # Load all components
    console.print("[cyan]Loading models and indexes...[/cyan]")

    with open(cfg.bm25.index_path, "rb") as f:
        bm25 = pickle.load(f)
    with open("data/indexes/bm25_pid_list.pkl", "rb") as f:
        bm25_pid_list = pickle.load(f)

    faiss_index = faiss.read_index(cfg.faiss.index_path)
    if hasattr(faiss_index, "nprobe"):
        faiss_index.nprobe = cfg.faiss.nprobe
    with open(cfg.faiss.docid_map_path, "rb") as f:
        faiss_pid_list = pickle.load(f)

    tt_model, tt_tokenizer = load_two_tower(cfg.two_tower.save_dir, device=str(DEVICE))
    ce_model, ce_tokenizer = load_cross_encoder(
        cfg.cross_encoder.save_dir, device=str(DEVICE)
    )

    lr_booster = xgb.Booster()
    lr_booster.load_model(str(Path(cfg.lambdarank.save_dir) / "lambdarank.json"))
    with open(Path(cfg.lambdarank.save_dir) / "feature_names.json") as f:
        feature_names = json.load(f)["features"]

    passages_df = pd.read_parquet("data/processed/passages.parquet")
    dev_queries_df = pd.read_parquet("data/processed/dev_queries.parquet").head(
        num_queries
    )
    dev_qrels_df = pd.read_parquet("data/processed/dev_qrels.parquet")

    pid_to_text = dict(zip(passages_df["pid"], passages_df["text"]))
    pid_to_len = dict(zip(passages_df["pid"], passages_df["token_count"]))
    pos_pids_by_qid = dev_qrels_df.groupby("qid")["pid"].apply(set).to_dict()

    configs_results = {
        "BM25": [],
        "TwoTower": [],
        "Hybrid(RRF)": [],
        "Hybrid(RRF)+LambdaRank": [],
        "Hybrid(RRF)+CrossEncoder": [],
        "TwoTower+LambdaRank": [],
        "TwoTower+CrossEncoder": [],
    }
    latencies = {k: [] for k in configs_results}

    for _, row in tqdm(
        dev_queries_df.iterrows(), total=len(dev_queries_df), desc="Evaluating"
    ):
        qid = row["qid"]
        query_text = row["text"]
        gold_pids = pos_pids_by_qid.get(qid, set())
        if not gold_pids:
            continue

        # BM25
        t0 = time.perf_counter()
        bm25_results = retrieve_bm25(bm25, bm25_pid_list, query_text, top_k=100)
        latencies["BM25"].append((time.perf_counter() - t0) * 1000)
        configs_results["BM25"].append(compute_metrics(bm25_results[:10], gold_pids))

        # Two-Tower (dense only)
        t0 = time.perf_counter()
        tt_results = retrieve_two_tower(
            tt_model, tt_tokenizer, faiss_index, faiss_pid_list, query_text
        )
        latencies["TwoTower"].append((time.perf_counter() - t0) * 1000)
        configs_results["TwoTower"].append(compute_metrics(tt_results[:10], gold_pids))

        # Hybrid RRF (BM25 + FAISS fused)
        t0 = time.perf_counter()
        hybrid_results = retrieve_hybrid_rrf(
            bm25,
            bm25_pid_list,
            tt_model,
            tt_tokenizer,
            faiss_index,
            faiss_pid_list,
            query_text,
            top_k=100,
            rrf_k=hr_cfg.rrf_k,
        )
        latencies["Hybrid(RRF)"].append((time.perf_counter() - t0) * 1000)
        configs_results["Hybrid(RRF)"].append(
            compute_metrics(hybrid_results[:10], gold_pids)
        )

        # Hybrid RRF + LambdaRank
        t0 = time.perf_counter()
        hybrid_lr_results = rerank_lambdarank(
            lr_booster,
            feature_names,
            bm25,
            bm25_pid_list,
            tt_model,
            tt_tokenizer,
            pid_to_text,
            pid_to_len,
            query_text,
            hybrid_results,
        )
        latencies["Hybrid(RRF)+LambdaRank"].append((time.perf_counter() - t0) * 1000)
        configs_results["Hybrid(RRF)+LambdaRank"].append(
            compute_metrics(hybrid_lr_results, gold_pids)
        )

        # Hybrid RRF + CrossEncoder
        t0 = time.perf_counter()
        hybrid_ce_results = rerank_cross_encoder(
            ce_model, ce_tokenizer, pid_to_text, query_text, hybrid_results
        )
        latencies["Hybrid(RRF)+CrossEncoder"].append((time.perf_counter() - t0) * 1000)
        configs_results["Hybrid(RRF)+CrossEncoder"].append(
            compute_metrics(hybrid_ce_results, gold_pids)
        )

        # Two-Tower + LambdaRank
        t0 = time.perf_counter()
        lr_results = rerank_lambdarank(
            lr_booster,
            feature_names,
            bm25,
            bm25_pid_list,
            tt_model,
            tt_tokenizer,
            pid_to_text,
            pid_to_len,
            query_text,
            tt_results,
        )
        latencies["TwoTower+LambdaRank"].append((time.perf_counter() - t0) * 1000)
        configs_results["TwoTower+LambdaRank"].append(
            compute_metrics(lr_results, gold_pids)
        )

        # Two-Tower + CrossEncoder
        t0 = time.perf_counter()
        ce_results = rerank_cross_encoder(
            ce_model, ce_tokenizer, pid_to_text, query_text, tt_results
        )
        latencies["TwoTower+CrossEncoder"].append((time.perf_counter() - t0) * 1000)
        configs_results["TwoTower+CrossEncoder"].append(
            compute_metrics(ce_results, gold_pids)
        )

    # Aggregate results
    summary = {}
    for config_name, metrics_list in configs_results.items():
        if not metrics_list:
            continue
        agg = {}
        for metric in metrics_list[0]:
            agg[metric] = float(np.mean([m[metric] for m in metrics_list]))
        agg["latency_p50_ms"] = float(np.percentile(latencies[config_name], 50))
        agg["latency_p95_ms"] = float(np.percentile(latencies[config_name], 95))
        summary[config_name] = agg

    # Print comparison table
    table = Table(title="Retrieval & Ranking Evaluation Results", show_header=True)
    table.add_column("Config", style="cyan")
    table.add_column("NDCG@10", justify="right")
    table.add_column("MAP@10", justify="right")
    table.add_column("MRR@10", justify="right")
    table.add_column("Recall@10", justify="right")
    table.add_column("Recall@100", justify="right")
    table.add_column("p50 lat (ms)", justify="right")
    table.add_column("p95 lat (ms)", justify="right")

    for config_name, metrics in summary.items():
        table.add_row(
            config_name,
            f"{metrics.get('NDCG@10', 0):.4f}",
            f"{metrics.get('MAP@10', 0):.4f}",
            f"{metrics.get('MRR@10', 0):.4f}",
            f"{metrics.get('Recall@10', 0):.4f}",
            f"{metrics.get('Recall@100', 0):.4f}",
            f"{metrics.get('latency_p50_ms', 0):.1f}",
            f"{metrics.get('latency_p95_ms', 0):.1f}",
        )

    console.print(table)

    # Save results
    results_path = Path("data/processed/eval_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"[green]Results saved → {results_path}[/green]")

    # Log to MLflow
    mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
    mlflow.set_experiment(mlf_cfg.experiment_name)
    with mlflow.start_run(run_name="full_evaluation"):
        for config_name, metrics in summary.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{config_name}/{metric_name}", value)
        mlflow.log_artifact(str(results_path))

    console.print("\n[bold green]Evaluation complete.[/bold green]")
    return summary


if __name__ == "__main__":
    run_evaluation()
