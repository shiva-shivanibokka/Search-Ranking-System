"""
Retrieval Service — Port 8002

Responsibilities:
  1. Load FAISS IVF+PQ index + two-tower query encoder at startup
  2. On each request:
     a. Check Redis cache (key = quantized query embedding bucket)
     b. Cache hit → return cached candidates immediately (<5ms)
     c. Cache miss → encode query with query tower → FAISS search → cache result
  3. Return top-100 candidate (doc_id, text, score) tuples to gateway
  4. Expose Prometheus metrics: cache hit rate, FAISS latency, requests/sec
"""

import os
import json
import time
import pickle
import hashlib
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import faiss
import redis
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import structlog
import pandas as pd

from services.shared.logger import configure_logging, bind_request_id

configure_logging("retrieval")
logger = structlog.get_logger()

# ── Prometheus ────────────────────────────────────────────────────────────────
RETRIEVAL_REQUESTS = Counter(
    "retrieval_requests_total", "Total retrieval requests", ["cache_status"]
)
RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_ms",
    "FAISS retrieval latency",
    buckets=[1, 5, 10, 20, 30, 50, 100, 200],
)
CACHE_HIT_RATE = Counter("retrieval_cache_hits_total", "Redis cache hits")
CACHE_MISS_RATE = Counter("retrieval_cache_misses_total", "Redis cache misses")

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


# ── Global state (loaded once at startup) ─────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
faiss_index = None
pid_list = None
pid_to_text = None
two_tower_model = None
tt_tokenizer = None
redis_client = None
CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "100"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global \
        faiss_index, \
        pid_list, \
        pid_to_text, \
        two_tower_model, \
        tt_tokenizer, \
        redis_client

    # Load FAISS index
    index_path = os.getenv("FAISS_INDEX_PATH", "data/indexes/faiss_ivfpq.index")
    docid_map_path = os.getenv("FAISS_DOCID_MAP_PATH", "data/indexes/docid_map.pkl")
    passages_path = os.getenv("PASSAGES_PATH", "data/processed/passages.parquet")
    model_dir = os.getenv("TWO_TOWER_MODEL_DIR", "models/two_tower")

    logger.info("loading.faiss_index", path=index_path)
    faiss_index = faiss.read_index(index_path)
    nprobe = int(os.getenv("FAISS_NPROBE", "64"))
    faiss_index.nprobe = nprobe
    logger.info("faiss.loaded", ntotal=faiss_index.ntotal, nprobe=nprobe)

    with open(docid_map_path, "rb") as f:
        pid_list = pickle.load(f)

    logger.info("loading.passages", path=passages_path)
    passages_df = pd.read_parquet(passages_path)
    pid_to_text = dict(zip(passages_df["pid"].tolist(), passages_df["text"].tolist()))
    logger.info("passages.loaded", count=len(pid_to_text))

    # Load two-tower query encoder
    logger.info("loading.two_tower", dir=model_dir)
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from training.two_tower_model import load_two_tower

    two_tower_model, tt_tokenizer = load_two_tower(model_dir, device=str(DEVICE))
    logger.info("two_tower.loaded")

    # Redis connection
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


def _cache_key(query: str, top_k: int) -> str:
    """Deterministic cache key from query text + top_k."""
    h = hashlib.md5(f"{query.lower().strip()}:{top_k}".encode()).hexdigest()
    return f"retrieval:{h}"


def _get_cache(query: str, top_k: int) -> Optional[list]:
    if redis_client is None:
        return None
    try:
        key = _cache_key(query, top_k)
        cached = redis_client.get(key)
        if cached:
            CACHE_HIT_RATE.inc()
            return json.loads(cached)
    except Exception as e:
        logger.warning("cache.get_failed", error=str(e))
    return None


def _set_cache(query: str, top_k: int, candidates: list) -> None:
    if redis_client is None:
        return
    try:
        key = _cache_key(query, top_k)
        redis_client.setex(key, CACHE_TTL, json.dumps(candidates))
    except Exception as e:
        logger.warning("cache.set_failed", error=str(e))


# ── Embedding ─────────────────────────────────────────────────────────────────


def embed_query(text: str) -> np.ndarray:
    """Encode a single query text to a normalized embedding."""
    enc = tt_tokenizer(
        text,
        max_length=64,
        padding=True,
        truncation=True,
        return_tensors="pt",
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


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "retrieval",
        "faiss_vectors": faiss_index.ntotal if faiss_index else 0,
        "redis": redis_client is not None,
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest):
    bind_request_id(req.request_id)
    t0 = time.perf_counter()

    # Use HyDE passage for embedding if provided (improves recall for informational queries)
    embed_text = req.hyde_passage if req.hyde_passage else req.query

    # ── Cache check ───────────────────────────────────────────────────────────
    cached = _get_cache(embed_text, req.top_k)
    if cached is not None:
        latency_ms = (time.perf_counter() - t0) * 1000
        RETRIEVAL_REQUESTS.labels(cache_status="hit").inc()
        RETRIEVAL_LATENCY.observe(latency_ms)
        logger.info(
            "retrieval.cache_hit", query=req.query, latency_ms=round(latency_ms, 2)
        )
        return RetrieveResponse(
            candidates=[Candidate(**c) for c in cached],
            cache_hit=True,
            latency_ms=round(latency_ms, 2),
        )

    CACHE_MISS_RATE.inc()

    # ── FAISS ANN search ──────────────────────────────────────────────────────
    q_emb = embed_query(embed_text)
    scores, indices = faiss_index.search(q_emb, req.top_k)

    candidates = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:
            continue
        pid = pid_list[idx]
        text = pid_to_text.get(pid, "")
        candidates.append(
            {
                "doc_id": pid,
                "text": text,
                "score": float(score),
                "retrieval_rank": rank + 1,
            }
        )

    # ── Cache result ──────────────────────────────────────────────────────────
    _set_cache(embed_text, req.top_k, candidates)

    latency_ms = (time.perf_counter() - t0) * 1000
    RETRIEVAL_REQUESTS.labels(cache_status="miss").inc()
    RETRIEVAL_LATENCY.observe(latency_ms)

    logger.info(
        "retrieval.complete",
        query=req.query,
        num_candidates=len(candidates),
        latency_ms=round(latency_ms, 2),
    )

    return RetrieveResponse(
        candidates=[Candidate(**c) for c in candidates],
        cache_hit=False,
        latency_ms=round(latency_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.retrieval.main:app", host="0.0.0.0", port=8002, reload=False)
