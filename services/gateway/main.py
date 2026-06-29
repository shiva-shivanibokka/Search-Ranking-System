"""
API Gateway — single entry point for all search requests.

Responsibilities:
  1. Inject a unique request_id into every request
  2. Route to query-understanding → retrieval → ranking services
  3. Aggregate latency breakdown per stage
  4. Log query + metrics to PostgreSQL asynchronously (non-blocking)
  5. Expose Prometheus metrics (request count, latency histograms)
  6. Return ranked results to client

All inter-service calls are async (httpx.AsyncClient).
Downstream services are never awaited serially — understanding runs first,
then retrieval and ranking are pipelined.
"""

import os
import uuid
import time
import asyncio
import threading
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import structlog

from services.shared.logger import (
    configure_logging,
    bind_request_id,
    clear_request_context,
)
from services.shared.database import get_db_session, QueryLog, create_tables

configure_logging("api-gateway")
logger = structlog.get_logger()

# ── Service URLs ──────────────────────────────────────────────────────────────
QUERY_UNDERSTANDING_URL = os.getenv(
    "QUERY_UNDERSTANDING_URL", "http://query-understanding:8001"
)
RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://retrieval:8002")
RANKING_URL = os.getenv("RANKING_URL", "http://ranking:8003")

# ── Prometheus Metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "gateway_requests_total",
    "Total search requests",
    ["status"],
)
REQUEST_LATENCY = Histogram(
    "gateway_request_latency_ms",
    "End-to-end request latency in ms",
    buckets=[10, 25, 50, 100, 150, 200, 300, 500, 1000],
)
STAGE_LATENCY = Histogram(
    "gateway_stage_latency_ms",
    "Per-stage latency in ms",
    ["stage"],
    buckets=[5, 10, 20, 30, 50, 100, 150, 200, 500],
)

# ── Schemas ───────────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    ranker: Optional[str] = None  # "lambdarank" | "crossencoder" | None (A/B auto)


class SearchResult(BaseModel):
    rank: int
    doc_id: int
    text: str
    score: float
    ranker: str


class SearchResponse(BaseModel):
    request_id: str
    query: str
    rewritten_query: Optional[str]
    intent: Optional[str]
    results: list[SearchResult]
    latency: dict  # breakdown per stage


# ── App lifecycle ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        create_tables()
        logger.info("database.tables_created")
    except Exception as e:
        logger.warning("database.init_failed", error=str(e))
    yield


app = FastAPI(
    title="Neural Search Ranking — API Gateway",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Rate limiting (protects a public demo from abuse / runaway LLM cost) ────────
# Simple in-memory per-IP sliding window. Set RATE_LIMIT_PER_MINUTE=0 to disable.
# Note: in-memory state is per-process; for multi-instance deploys back this with
# Redis. For a single-process free-tier demo this is sufficient.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
_rate_lock = threading.Lock()
_rate_hits: dict[str, deque] = {}


def _client_ip(request: Request) -> str:
    # Honor X-Forwarded-For when behind a proxy (HF Spaces, Render, etc.).
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    if RATE_LIMIT_PER_MINUTE <= 0 or request.url.path != "/search":
        return await call_next(request)

    ip = _client_ip(request)
    now = time.monotonic()
    window_start = now - 60.0
    with _rate_lock:
        hits = _rate_hits.setdefault(ip, deque())
        while hits and hits[0] < window_start:
            hits.popleft()
        if len(hits) >= RATE_LIMIT_PER_MINUTE:
            retry_after = max(1, int(60 - (now - hits[0])))
            logger.warning("rate_limit.exceeded", ip=ip, limit=RATE_LIMIT_PER_MINUTE)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Slow down."},
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)
    return await call_next(request)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "service": "api-gateway"}


@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    request_id = str(uuid.uuid4())
    bind_request_id(request_id)
    t_total_start = time.perf_counter()
    latency_breakdown = {}

    logger.info("search.start", query=req.query, top_k=req.top_k)

    async with httpx.AsyncClient(timeout=10.0) as client:
        # ── Stage 1: Query Understanding ───────────────────────────────────────
        t0 = time.perf_counter()
        try:
            qu_resp = await client.post(
                f"{QUERY_UNDERSTANDING_URL}/understand",
                json={"query": req.query, "request_id": request_id},
            )
            qu_resp.raise_for_status()
            qu_data = qu_resp.json()
        except Exception as e:
            logger.warning("query_understanding.failed", error=str(e))
            qu_data = {
                "rewritten_query": req.query,
                "intent": "informational",
                "hyde_passage": None,
                "skip_rewrite": True,
            }

        qu_latency = (time.perf_counter() - t0) * 1000
        latency_breakdown["query_understanding_ms"] = round(qu_latency, 2)
        STAGE_LATENCY.labels(stage="query_understanding").observe(qu_latency)

        effective_query = qu_data.get("rewritten_query", req.query)
        intent = qu_data.get("intent", "informational")
        hyde_passage = qu_data.get("hyde_passage")  # forwarded to retrieval for dense embedding

        # ── Stage 2: Retrieval ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            ret_resp = await client.post(
                f"{RETRIEVAL_URL}/retrieve",
                json={
                    "query": effective_query,
                    "request_id": request_id,
                    "top_k": 100,
                    "hyde_passage": hyde_passage,  # HyDE: embed the hypothetical answer when present
                },
            )
            ret_resp.raise_for_status()
            ret_data = ret_resp.json()
        except Exception as e:
            logger.error("retrieval.failed", error=str(e))
            REQUEST_COUNT.labels(status="error").inc()
            raise HTTPException(status_code=503, detail="Retrieval service unavailable")

        ret_latency = (time.perf_counter() - t0) * 1000
        latency_breakdown["retrieval_ms"] = round(ret_latency, 2)
        latency_breakdown["cache_hit"] = ret_data.get("cache_hit", False)
        latency_breakdown["retrieval_mode"] = ret_data.get("retrieval_mode", "unknown")
        STAGE_LATENCY.labels(stage="retrieval").observe(ret_latency)

        # ── Stage 3: Ranking ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            rank_resp = await client.post(
                f"{RANKING_URL}/rank",
                json={
                    "query": effective_query,
                    "request_id": request_id,
                    "candidates": ret_data["candidates"],
                    "top_k": req.top_k,
                    "ranker": req.ranker,
                    "intent": intent,  # passed to difficulty classifier
                },
            )
            rank_resp.raise_for_status()
            rank_data = rank_resp.json()
        except Exception as e:
            logger.error("ranking.failed", error=str(e))
            REQUEST_COUNT.labels(status="error").inc()
            raise HTTPException(status_code=503, detail="Ranking service unavailable")

        rank_latency = (time.perf_counter() - t0) * 1000
        latency_breakdown["ranking_ms"] = round(rank_latency, 2)
        STAGE_LATENCY.labels(stage="ranking").observe(rank_latency)

    total_latency = (time.perf_counter() - t_total_start) * 1000
    latency_breakdown["total_ms"] = round(total_latency, 2)

    REQUEST_LATENCY.observe(total_latency)
    REQUEST_COUNT.labels(status="success").inc()

    logger.info(
        "search.complete",
        total_ms=round(total_latency, 2),
        num_results=len(rank_data["results"]),
        intent=intent,
        cache_hit=latency_breakdown.get("cache_hit", False),
        ranker=rank_data.get("ranker_used"),
    )

    # ── Async DB log (fire-and-forget, does not block response) ───────────────
    asyncio.create_task(
        _log_query_to_db(
            request_id=request_id,
            query_text=req.query,
            rewritten_query=effective_query if effective_query != req.query else None,
            intent=intent,
            ranker_version=rank_data.get("ranker_used"),
            ab_variant=rank_data.get("ab_variant"),
            num_results=len(rank_data["results"]),
            total_latency_ms=total_latency,
            retrieval_latency_ms=ret_latency,
            ranking_latency_ms=rank_latency,
            cache_hit=latency_breakdown.get("cache_hit", False),
        )
    )

    clear_request_context()

    return SearchResponse(
        request_id=request_id,
        query=req.query,
        rewritten_query=effective_query if effective_query != req.query else None,
        intent=intent,
        results=[SearchResult(**r) for r in rank_data["results"]],
        latency=latency_breakdown,
    )


def _sync_log_query_to_db(kwargs: dict):
    """Synchronous DB write — runs in a thread pool executor."""
    try:
        session = get_db_session()
        try:
            log = QueryLog(**kwargs)
            session.add(log)
            session.commit()
        finally:
            session.close()
    except Exception as e:
        logger.warning("db_log.failed", error=str(e))


async def _log_query_to_db(**kwargs):
    """Write query log to PostgreSQL. Offloads blocking I/O to thread pool."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _sync_log_query_to_db, kwargs)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.gateway.main:app", host="0.0.0.0", port=8000, reload=False)
