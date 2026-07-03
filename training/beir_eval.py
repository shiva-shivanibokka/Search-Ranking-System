"""
BEIR zero-shot evaluation library.

Reuses the repo's trained two-tower encoder and the exact RRF fusion formula
from deploy/engine.py to measure out-of-domain generalization on small BEIR
datasets (SciFact, NFCorpus, FiQA). CPU-only: the corpora are small enough for
brute-force dense search, so no FAISS index is built.

Metric computation is delegated to training.evaluate.compute_metrics so the BEIR
numbers use the identical NDCG@10 / Recall@100 / MAP@10 / MRR@10 code path as the
in-domain MS MARCO evaluation.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from rank_bm25 import BM25Okapi

from training.evaluate import compute_metrics


def doc_to_text(doc: dict) -> str:
    """BEIR corpus docs are {'title': str, 'text': str}; concatenate them."""
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


class TwoTowerBEIRAdapter:
    """Wraps the repo's two-tower encoder in BEIR's encode_queries/encode_corpus
    interface, returning L2-normalized numpy embeddings."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        max_q_len: int = 64,
        max_d_len: int = 180,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_q_len = max_q_len
        self.max_d_len = max_d_len

    def _encode(
        self, texts: List[str], encode_fn, max_len: int, batch_size: int
    ) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                max_length=max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                emb = (
                    encode_fn(
                        enc["input_ids"].to(self.device),
                        enc["attention_mask"].to(self.device),
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            out.append(emb)
        return np.vstack(out)

    def encode_queries(
        self, queries: List[str], batch_size: int = 32
    ) -> np.ndarray:
        return self._encode(
            queries, self.model.encode_query, self.max_q_len, batch_size
        )

    def encode_corpus(
        self, corpus: List[dict], batch_size: int = 32
    ) -> np.ndarray:
        texts = [doc_to_text(d) for d in corpus]
        return self._encode(texts, self.model.encode_doc, self.max_d_len, batch_size)


def dense_rank(
    query_emb: np.ndarray, doc_emb: np.ndarray, doc_ids: List[str], top_k: int
) -> List[str]:
    """Rank doc_ids by dot-product similarity to a single query embedding.

    Embeddings are L2-normalized, so dot product == cosine similarity.
    """
    scores = doc_emb @ query_emb  # (N,)
    top = np.argsort(scores)[::-1][:top_k]
    return [doc_ids[i] for i in top]


def rrf_fuse(
    dense_ranked: List[str],
    sparse_ranked: List[str],
    rrf_k: int,
    top_k: int,
) -> List[str]:
    """Reciprocal Rank Fusion — identical formula to
    deploy/engine.SearchEngine._rrf: score(d) = Σ 1 / (rrf_k + rank(d)),
    where rank is 1-indexed within each ranked list.
    """
    scores: Dict[str, float] = {}
    for rank, pid in enumerate(dense_ranked):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, pid in enumerate(sparse_ranked):
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [pid for pid, _ in ranked]


def evaluate_beir_dataset(
    corpus: Dict[str, dict],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    adapter,
    rrf_k: int = 60,
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Evaluate BM25, TwoTower (dense), and Hybrid(RRF) on one BEIR dataset.

    Returns {config_name: {metric_name: mean_value}} using the repo's
    training.evaluate.compute_metrics for every configuration.
    """
    doc_ids = list(corpus.keys())
    corpus_texts = [doc_to_text(corpus[d]) for d in doc_ids]
    doc_emb = adapter.encode_corpus([corpus[d] for d in doc_ids])

    qids = [q for q in queries if qrels.get(q)]
    query_texts = [queries[q] for q in qids]
    query_emb = adapter.encode_queries(query_texts)

    bm25 = BM25Okapi([t.lower().split() for t in corpus_texts])

    per_config: Dict[str, List[dict]] = {
        "BM25": [],
        "TwoTower": [],
        "Hybrid(RRF)": [],
    }
    for i, qid in enumerate(qids):
        gold = {d for d, rel in qrels[qid].items() if rel > 0}
        if not gold:
            continue

        bm25_scores = bm25.get_scores(query_texts[i].lower().split())
        bm25_top = np.argsort(bm25_scores)[::-1][:top_k]
        sparse_ranked = [doc_ids[j] for j in bm25_top]

        dense_ranked = dense_rank(query_emb[i], doc_emb, doc_ids, top_k)
        fused = rrf_fuse(dense_ranked, sparse_ranked, rrf_k, top_k)

        per_config["BM25"].append(compute_metrics(sparse_ranked, gold))
        per_config["TwoTower"].append(compute_metrics(dense_ranked, gold))
        per_config["Hybrid(RRF)"].append(compute_metrics(fused, gold))

    summary: Dict[str, Dict[str, float]] = {}
    for name, rows in per_config.items():
        if not rows:
            continue
        summary[name] = {
            metric: float(np.mean([r[metric] for r in rows])) for metric in rows[0]
        }
    return summary
