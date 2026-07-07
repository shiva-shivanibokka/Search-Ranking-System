"""
Consolidated single-process search engine for the free-tier deployment.

The full system is 5 microservices; a free Hugging Face Space gives you one
process. This module runs the *same pipeline* in-process so the demo fits a free
host: query understanding (optional LLM) -> hybrid retrieve (FAISS + BM25 + RRF)
-> rerank (LambdaRank, optional CrossEncoder).

It deliberately mirrors the logic in services/retrieval/main.py and
services/ranking/main.py (RRF fusion, the 7-feature LambdaRank vector) so results
match the microservice deployment. Everything optional degrades gracefully:
  * no LLM key            -> rule-based query understanding only
  * no cross-encoder      -> LambdaRank ranking only
  * no Redis (Upstash)    -> no caching, still correct
The heavy artifacts are pulled by scripts/bootstrap.py before this loads.
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

import numpy as np
import torch

from services.shared.features import Candidate, build_lambdarank_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SearchEngine:
    def __init__(self) -> None:
        import faiss
        import pandas as pd

        from training.two_tower_model import load_two_tower

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data = PROJECT_ROOT / "data"
        models = PROJECT_ROOT / "models"

        # ── Dense: two-tower encoder + FAISS index ───────────────────────────
        self.model, self.tokenizer = load_two_tower(
            str(models / "two_tower"), device=str(self.device)
        )
        self.faiss_index = faiss.read_index(str(data / "indexes" / "faiss_ivfpq.index"))
        if hasattr(self.faiss_index, "nprobe"):
            self.faiss_index.nprobe = int(os.getenv("FAISS_NPROBE", "64"))
        with open(data / "indexes" / "docid_map.pkl", "rb") as f:
            self.faiss_pid_list = pickle.load(f)

        # ── Sparse: BM25 (hybrid retrieval + LambdaRank feature) ─────────────
        with open(data / "indexes" / "bm25_index.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(data / "indexes" / "bm25_pid_list.pkl", "rb") as f:
            self.bm25_pid_list = pickle.load(f)
        self.bm25_idx = {pid: i for i, pid in enumerate(self.bm25_pid_list)}

        # ── Passage text + lengths ───────────────────────────────────────────
        pdf = pd.read_parquet(data / "processed" / "passages.parquet")
        self.pid_to_text = dict(zip(pdf["pid"].tolist(), pdf["text"].tolist()))
        self.pid_to_len = dict(zip(pdf["pid"].tolist(), pdf["token_count"].tolist()))

        # ── LambdaRank reranker ──────────────────────────────────────────────
        import xgboost as xgb

        self.lambdarank = xgb.Booster()
        self.lambdarank.load_model(str(models / "lambdarank" / "lambdarank.json"))

        # ── Optional CrossEncoder ────────────────────────────────────────────
        self.cross_encoder = None
        self.ce_tokenizer = None
        ce_dir = models / "cross_encoder"
        if (ce_dir / "config.json").exists():
            try:
                from training.cross_encoder_model import load_cross_encoder

                self.cross_encoder, self.ce_tokenizer = load_cross_encoder(
                    str(ce_dir), device=str(self.device)
                )
            except Exception:
                self.cross_encoder = None

        self.rrf_k = int(os.getenv("RRF_K", "60"))

    # ── Retrieval ────────────────────────────────────────────────────────────
    def _embed(self, text: str) -> np.ndarray:
        enc = self.tokenizer(
            text, max_length=64, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            return (
                self.model.encode_query(
                    enc["input_ids"].to(self.device),
                    enc["attention_mask"].to(self.device),
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    def _faiss(self, text: str, top_k: int) -> list[dict]:
        scores, idx = self.faiss_index.search(self._embed(text), top_k)
        out = []
        for rank, (s, i) in enumerate(zip(scores[0], idx[0])):
            if i < 0:
                continue
            out.append({"pid": self.faiss_pid_list[i], "score": float(s), "rank": rank + 1})
        return out

    def _bm25(self, query: str, top_k: int, scores=None) -> list[dict]:
        # ``scores`` may be a precomputed full BM25 score vector (from a single
        # scan the caller already did) to avoid re-scanning ~1M docs.
        if scores is None:
            scores = self.bm25.get_scores(query.lower().split())
        top = scores.argsort()[::-1][:top_k]
        return [
            {"pid": self.bm25_pid_list[i], "score": float(scores[i]), "rank": r + 1}
            for r, i in enumerate(top)
        ]

    def _rrf(self, dense: list[dict], sparse: list[dict], top_k: int) -> list[dict]:
        scores: dict[int, float] = {}
        for item in dense:
            scores[item["pid"]] = scores.get(item["pid"], 0.0) + 1.0 / (self.rrf_k + item["rank"])
        for item in sparse:
            scores[item["pid"]] = scores.get(item["pid"], 0.0) + 1.0 / (self.rrf_k + item["rank"])
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"pid": pid, "score": s, "rank": r + 1} for r, (pid, s) in enumerate(ranked)]

    def retrieve(self, query: str, embed_text: str, top_k: int = 100) -> list[dict]:
        dense = self._faiss(embed_text, top_k)
        sparse = self._bm25(query, top_k)
        fused = self._rrf(dense, sparse, top_k)
        return [
            {
                "doc_id": item["pid"],
                "text": self.pid_to_text.get(item["pid"], ""),
                "score": item["score"],
                "retrieval_rank": item["rank"],
            }
            for item in fused
        ]

    # ── Ranking (mirrors services/ranking/main.py feature order) ─────────────
    def _rerank_lambdarank(
        self, query: str, cands: list[dict], top_k: int, bm25_scores_all=None
    ) -> list[dict]:
        if not cands:
            return []
        import xgboost as xgb

        candidates = [
            Candidate(
                doc_id=c["doc_id"],
                text=c["text"],
                score=c["score"],
                retrieval_rank=c["retrieval_rank"],
            )
            for c in cands
        ]
        # Reuse the retrieval-stage BM25 scores + the prebuilt pid index (self.bm25_idx)
        # so the feature builder does not re-scan/re-index ~1M docs.
        X = build_lambdarank_features(
            query,
            candidates,
            self.bm25,
            self.bm25_pid_list,
            self.pid_to_len,
            bm25_scores_all=bm25_scores_all,
            bm25_idx=self.bm25_idx,
        )
        preds = self.lambdarank.predict(xgb.DMatrix(X))
        top = preds.argsort()[::-1][:top_k]
        return [
            {"rank": r + 1, "doc_id": cands[i]["doc_id"], "text": cands[i]["text"],
             "score": float(preds[i]), "ranker": "lambdarank"}
            for r, i in enumerate(top)
        ]

    def _rerank_crossencoder(self, query: str, cands: list[dict], top_k: int, batch: int = 32) -> list[dict]:
        if not cands or self.cross_encoder is None:
            return self._rerank_lambdarank(query, cands, top_k)
        # Cap the number of (query, doc) pairs the cross-encoder scores: a full
        # DistilBERT forward pass over 100 candidates on CPU is very slow. The
        # fused top-N almost always contains the relevant docs. Rerank at least
        # top_k so we can always fill the requested results.
        depth = max(top_k, int(os.getenv("CE_RERANK_DEPTH", "30")))
        cands = cands[:depth]
        max_len = int(os.getenv("CE_MAX_SEQ_LEN", "256"))
        scored = []
        for i in range(0, len(cands), batch):
            chunk = cands[i : i + batch]
            enc = self.ce_tokenizer(
                [query] * len(chunk), [c["text"] for c in chunk],
                max_length=max_len, padding=True, truncation=True, return_tensors="pt",
            )
            with torch.no_grad():
                # Rank + display by the raw relevance LOGIT, not the sigmoid
                # probability: the ordering is identical (sigmoid is monotonic) but
                # the scores spread out into a meaningful range instead of all
                # saturating at ~1.000, matching how LambdaRank shows its scores.
                s = self.cross_encoder.forward(
                    enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
                ).cpu().numpy()
            scored.extend(zip(range(i, i + len(chunk)), s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {"rank": r + 1, "doc_id": cands[i]["doc_id"], "text": cands[i]["text"],
             "score": float(sc), "ranker": "crossencoder"}
            for r, (i, sc) in enumerate(scored[:top_k])
        ]

    def rank(
        self, query: str, cands: list[dict], top_k: int, ranker: str, bm25_scores_all=None
    ) -> list[dict]:
        if ranker == "crossencoder":
            return self._rerank_crossencoder(query, cands, top_k)
        return self._rerank_lambdarank(query, cands, top_k, bm25_scores_all=bm25_scores_all)


# ── Lightweight intent rules (mirror query_understanding, no LLM needed) ───────
_NAV = [r"^(how to get to|directions to|location of|address of|where is)\b",
        r"\b(homepage|website|official site|login|sign in)\b",
        r"^(github|twitter|linkedin|youtube|facebook|wikipedia)\b"]
_TXN = [r"\b(buy|purchase|order|price|cheap|discount|deal|subscribe|download|install)\b",
        r"\b(best|top|review|compare|vs\.?|versus)\b"]
_INFO = [r"^(what|why|how|when|who|which|explain|define|describe)\b",
         r"\b(meaning|definition|difference between|example of|tutorial)\b"]


def classify_intent(query: str) -> str:
    q = query.lower().strip()
    for p in _NAV:
        if re.search(p, q):
            return "navigational"
    for p in _TXN:
        if re.search(p, q):
            return "transactional"
    for p in _INFO:
        if re.search(p, q):
            return "informational"
    return "informational"
