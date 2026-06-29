"""
Retrieval Service — Port 8002

Responsibilities:
  1. Load FAISS IVF+PQ index + BM25 index + two-tower query encoder at startup
  2. On each request:
     a. Check Redis cache
     b. Cache hit → return immediately (<5ms)
     c. Cache miss → run hybrid retrieval:
        - BM25 sparse retrieval (top-K)
        - FAISS dense retrieval (top-K)
        - Fuse both ranked lists with Reciprocal Rank Fusion (RRF)
  3. Return top-100 fused candidates to gateway
  4. Expose Prometheus metrics: cache hit rate, retrieval latency by mode

RRF formula: score(d) = Σ_i  1 / (k + rank_i(d))
  where k=60 is a smoothing constant that dampens the impact of very high ranks.
  Documents appearing in both lists get additive boosts — naturally favoring
  results that both sparse and dense systems agree on.
"""

import hashlib
import json
import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import redis
import structlog
import torch
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

from services.shared.logger import bind_request_id, configure_logging

configure_logging("retrieval")
logger = structlog.get_logger()

# ── Prometheus ────────────────────────────────────────────────────────────────
RETRIEVAL_REQUESTS = Counter(
    "retrieval_requests_total", "Total retrieval requests", ["cache_status", "mode"]
)
RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_ms",
    "Retrieval latency by mode",
    ["mode"],
    buckets=[1, 5, 10, 20, 30, 50, 100, 200],
)
CACHE_HIT_RATE = Counter("retrieval_cache_hits_total", "Redis cache hits")
CACHE_MISS_RATE = Counter("retrieval_cache_misses_total", "Redis cache misses")
RRF_OVERLAP = Histogram(
    "retrieval_rrf_overlap_ratio",
    "Fraction of candidates appearing in both BM25 and FAISS lists",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Schemas ───────────────────────────────────────────────────────────────────


class RetrieveRequest(BaseModel):
    query: str
    request_id: str
    top_k: int = 100
    hyde_passage: Optional[str] = None  # if provided, embed this instead


class Candidate(BaseModel):
    doc_id: int
    text: str
    score: float
    retrieval_rank: int


class RetrieveResponse(BaseModel):
    candidates: list[Candidate]
    cache_hit: bool
    latency_ms: float
    retrieval_mode: str  # "hybrid" | "dense_only"


# ── Global state (loaded once at startup) ─────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
faiss_index = None
faiss_pid_list = None
bm25_index = None
bm25_pid_list = None
pid_to_text = None
two_tower_model = None
tt_tokenizer = None
redis_client = None

CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "100"))
HYBRID_ENABLED = os.getenv("HYBRID_RETRIEVAL_ENABLED", "true").lower() == "true"
RRF_K = int(os.getenv("RRF_K", "60"))
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "data/indexes/bm25_index.pkl")
BM25_PID_PATH = os.getenv("BM25_PID_PATH", "data/indexes/bm25_pid_list.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global \
        faiss_index, \
        faiss_pid_list, \
        bm25_index, \
        bm25_pid_list, \
        pid_to_text, \
        two_tower_model, \
        tt_tokenizer, \
        redis_client

    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from training.two_tower_model import load_two_tower

    # ── FAISS index ───────────────────────────────────────────────────────────
    index_path = os.getenv("FAISS_INDEX_PATH", "data/indexes/faiss_ivfpq.index")
    docid_map_path = os.getenv("FAISS_DOCID_MAP_PATH", "data/indexes/docid_map.pkl")
    passages_path = os.getenv("PASSAGES_PATH", "data/processed/passages.parquet")
    model_dir = os.getenv("TWO_TOWER_MODEL_DIR", "models/two_tower")

    logger.info("loading.faiss_index", path=index_path)
    faiss_index = faiss.read_index(index_path)
    nprobe = int(os.getenv("FAISS_NPROBE", "64"))
    if hasattr(faiss_index, "nprobe"):
        faiss_index.nprobe = nprobe
    logger.info("faiss.loaded", ntotal=faiss_index.ntotal, nprobe=nprobe)

    with open(docid_map_path, "rb") as f:
        faiss_pid_list = pickle.load(f)

    # ── BM25 index (for hybrid retrieval) ────────────────────────────────────
    if HYBRID_ENABLED:
        logger.info("loading.bm25_index", path=BM25_INDEX_PATH)
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_index = pickle.load(f)
        with open(BM25_PID_PATH, "rb") as f:
            bm25_pid_list = pickle.load(f)
        logger.info("bm25.loaded", num_docs=len(bm25_pid_list))
    else:
        logger.info(
            "hybrid_retrieval.disabled", reason="HYBRID_RETRIEVAL_ENABLED=false"
        )

    # ── Passages text lookup ──────────────────────────────────────────────────
    logger.info("loading.passages", path=passages_path)
    passages_df = pd.read_parquet(passages_path)
    pid_to_text = dict(zip(passages_df["pid"].tolist(), passages_df["text"].tolist()))
    logger.info("passages.loaded", count=len(pid_to_text))

    # ── Two-tower query encoder ───────────────────────────────────────────────
    logger.info("loading.two_tower", dir=model_dir)
    two_tower_model, tt_tokenizer = load_two_tower(model_dir, device=str(DEVICE))
    logger.info("two_tower.loaded")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    try:
        redis_client = redis.Redis(
            host=redis_host, port=redis_port, db=0, decode_responses=True
        )
        redis_client.ping()
        logger.info("redis.connected", host=redis_host, port=redis_port)
    except Exception as e:
        logger.warning("redis.connection_failed", error=str(e))
        redis_client = None

    yield

    logger.info("retrieval.shutdown")


app = FastAPI(title="Retrieval Service", version="1.0.0", lifespan=lifespan)


# ── Cache helpers ─────────────────────────────────────────────────────────────


def _cache_key(query: str, top_k: int, mode: str) -> str:
    """Deterministic cache key includes retrieval mode to avoid stale cross-mode hits."""
    h = hashlib.md5(f"{query.lower().strip()}:{top_k}:{mode}".encode()).hexdigest()
    return f"retrieval:{h}"


def _get_cache(query: str, top_k: int, mode: str) -> Optional[list]:
    if redis_client is None:
        return None
    try:
        cached = redis_client.get(_cache_key(query, top_k, mode))
        if cached:
            CACHE_HIT_RATE.inc()
            return json.loads(cached)
    except Exception as e:
        logger.warning("cache.get_failed", error=str(e))
    return None


def _set_cache(query: str, top_k: int, mode: str, candidates: list) -> None:
    if redis_client is None:
        return
    try:
        redis_client.setex(
            _cache_key(query, top_k, mode), CACHE_TTL, json.dumps(candidates)
        )
    except Exception as e:
        logger.warning("cache.set_failed", error=str(e))


# ── Retrieval helpers ─────────────────────────────────────────────────────────


def embed_query(text: str) -> np.ndarray:
    """Encode a single query text to a normalized L2 embedding."""
    enc = tt_tokenizer(
        text, max_length=64, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        emb = (
            two_tower_model.encode_query(
                enc["input_ids"].to(DEVICE),
                enc["attention_mask"].to(DEVICE),
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return emb  # (1, D)


def _faiss_retrieve(embed_text: str, top_k: int) -> list[dict]:
    """Dense FAISS ANN retrieval. Returns list of {pid, score, rank} dicts."""
    q_emb = embed_query(embed_text)
    scores, indices = faiss_index.search(q_emb, top_k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:
            continue
        results.append(
            {"pid": faiss_pid_list[idx], "score": float(score), "rank": rank + 1}
        )
    return results


def _bm25_retrieve(query_text: str, top_k: int) -> list[dict]:
    """Sparse BM25 retrieval. Returns list of {pid, score, rank} dicts."""
    tokenized = query_text.lower().split()
    scores = bm25_index.get_scores(tokenized)
    top_indices = scores.argsort()[::-1][:top_k]
    return [
        {
            "pid": bm25_pid_list[i],
            "score": float(scores[i]),
            "rank": rank + 1,
        }
        for rank, i in enumerate(top_indices)
    ]


def _reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    top_k: int = 100,
) -> list[dict]:
    """
    Fuse dense (FAISS) and sparse (BM25) ranked lists using Reciprocal Rank Fusion.

    RRF score for document d:
        rrf(d) = 1/(k + rank_dense(d)) + 1/(k + rank_sparse(d))

    Documents only in one list still get a score from that list alone.
    k=60 is the standard constant — reduces the impact of rank-1 dominance.

    Returns top_k candidates sorted by fused RRF score descending,
    each carrying an rrf_score for downstream use.
    """
    rrf_scores: dict[int, float] = {}

    for item in dense_results:
        pid = item["pid"]
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + item["rank"])

    for item in sparse_results:
        pid = item["pid"]
        rrf_scores[pid] = rrf_scores.get(pid, 0.0) + 1.0 / (k + item["rank"])

    # Measure how much the two systems agree (overlap in their candidate sets)
    dense_pids = {r["pid"] for r in dense_results}
    sparse_pids = {r["pid"] for r in sparse_results}
    overlap = len(dense_pids & sparse_pids) / max(len(dense_pids | sparse_pids), 1)
    RRF_OVERLAP.observe(overlap)

    sorted_pids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"pid": pid, "rrf_score": score, "rank": rank + 1}
        for rank, (pid, score) in enumerate(sorted_pids)
    ]


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "retrieval",
        "faiss_vectors": faiss_index.ntotal if faiss_index else 0,
        "bm25_loaded": bm25_index is not None,
        "hybrid_enabled": HYBRID_ENABLED,
        "redis": redis_client is not None,
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    bind_request_id(req.request_id)
    t0 = time.perf_counter()

    # Use HyDE passage for dense embedding if provided
    embed_text = req.hyde_passage if req.hyde_passage else req.query
    mode = "hybrid" if (HYBRID_ENABLED and bm25_index is not None) else "dense_only"

    # ── Cache check ───────────────────────────────────────────────────────────
    cached = _get_cache(embed_text, req.top_k, mode)
    if cached is not None:
        latency_ms = (time.perf_counter() - t0) * 1000
        RETRIEVAL_REQUESTS.labels(cache_status="hit", mode=mode).inc()
        RETRIEVAL_LATENCY.labels(mode=mode).observe(latency_ms)
        logger.info(
            "retrieval.cache_hit", query=req.query, latency_ms=round(latency_ms, 2)
        )
        return RetrieveResponse(
            candidates=[Candidate(**c) for c in cached],
            cache_hit=True,
            latency_ms=round(latency_ms, 2),
            retrieval_mode=mode,
        )

    CACHE_MISS_RATE.inc()

    if mode == "hybrid":
        # ── Hybrid: BM25 + FAISS → RRF ───────────────────────────────────────
        dense_results = _faiss_retrieve(embed_text, top_k=req.top_k)
        # BM25 always uses the raw query text (keywords, not HyDE embedding)
        sparse_results = _bm25_retrieve(req.query, top_k=req.top_k)

        fused = _reciprocal_rank_fusion(
            dense_results, sparse_results, k=RRF_K, top_k=req.top_k
        )

        candidates = [
            {
                "doc_id": item["pid"],
                "text": pid_to_text.get(item["pid"], ""),
                "score": item["rrf_score"],
                "retrieval_rank": item["rank"],
            }
            for item in fused
        ]

        logger.info(
            "retrieval.hybrid_complete",
            query=req.query,
            dense_candidates=len(dense_results),
            sparse_candidates=len(sparse_results),
            fused_candidates=len(candidates),
        )
    else:
        # ── Dense-only fallback (BM25 not loaded) ────────────────────────────
        dense_results = _faiss_retrieve(embed_text, top_k=req.top_k)
        candidates = [
            {
                "doc_id": r["pid"],
                "text": pid_to_text.get(r["pid"], ""),
                "score": r["score"],
                "retrieval_rank": r["rank"],
            }
            for r in dense_results
        ]

    # ── Cache result ──────────────────────────────────────────────────────────
    _set_cache(embed_text, req.top_k, mode, candidates)

    latency_ms = (time.perf_counter() - t0) * 1000
    RETRIEVAL_REQUESTS.labels(cache_status="miss", mode=mode).inc()
    RETRIEVAL_LATENCY.labels(mode=mode).observe(latency_ms)

    logger.info(
        "retrieval.complete",
        query=req.query,
        mode=mode,
        num_candidates=len(candidates),
        latency_ms=round(latency_ms, 2),
    )

    return RetrieveResponse(
        candidates=[Candidate(**c) for c in candidates],
        cache_hit=False,
        latency_ms=round(latency_ms, 2),
        retrieval_mode=mode,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.retrieval.main:app", host="0.0.0.0", port=8002, reload=False)
