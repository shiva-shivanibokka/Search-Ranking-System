"""
Retrieval API — FastAPI over the consolidated search engine (deploy/engine.py).

This is the HTTP surface the SvelteKit frontend calls. It runs the *same*
pipeline as the microservice deployment in one process (query understanding ->
hybrid retrieve (FAISS + BM25 + RRF) -> learned rerank) and returns a full
**stage breakdown** so the UI can show how a result was produced, not just the
final list.

Design notes:
  * RAG is client-side / BYOK: the browser calls POST /search to get passages,
    then calls the chosen LLM directly with the user's key. This API never sees
    that key. The only server-side LLM use is optional HyDE query expansion,
    gated on the server's own LLM_PROVIDER env (default: none).
  * The heavy artifacts (model + FAISS/BM25 indexes + passages) are pulled by
    scripts/bootstrap.py before this loads. Loading them takes ~15-30s; on Cloud
    Run raise the startup-probe timeout accordingly (see deploy/cloudrun.md).
  * Everything optional degrades gracefully: no LLM key -> no HyDE; no
    cross-encoder -> LambdaRank only; no DATABASE_URL -> /click is a no-op.

Run locally:  uvicorn deploy.api:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Global engine handle ───────────────────────────────────────────────────────
# Populated at startup (lifespan). Tests monkeypatch this with a fake engine so
# they never load the 1.2GB artifacts.
ENGINE = None  # type: ignore[var-annotated]
LLM = None  # type: ignore[var-annotated]


def _load_engine() -> None:
    """Load the search engine + LLM provider into module globals (idempotent)."""
    global ENGINE, LLM
    if ENGINE is not None:
        return
    from deploy.engine import SearchEngine
    from services.shared.llm import get_llm_provider

    print("Loading search engine (models + indexes)...", flush=True)
    ENGINE = SearchEngine()
    LLM = get_llm_provider()
    print(f"Engine ready. LLM provider: {LLM.name} (available={LLM.available})", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Skip the heavy load when API_SKIP_ENGINE_LOAD is set (tests, or a keep-warm
    # container that only needs to answer /health cheaply).
    if os.getenv("API_SKIP_ENGINE_LOAD") != "1":
        _load_engine()
    yield


app = FastAPI(
    title="Neural Search Ranking — Retrieval API",
    version="1.0.0",
    description="Hybrid (FAISS + BM25 + RRF) retrieval with learned reranking, "
    "over ~1M MS MARCO passages. Returns a full pipeline stage breakdown.",
    lifespan=lifespan,
)

# CORS: lock to the frontend origin(s) in production via ALLOWED_ORIGINS
# (comma-separated). Default "*" is convenient for local dev.
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Rate limiting (protects a public demo from abuse) ──────────────────────────
# Per-IP sliding window, in-memory. Set RATE_LIMIT_PER_MINUTE=0 to disable.
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
_rate_lock = threading.Lock()
_rate_hits: dict[str, deque] = {}
_RATE_LIMITED_PATHS = {"/search"}


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    if RATE_LIMIT_PER_MINUTE <= 0 or request.url.path not in _RATE_LIMITED_PATHS:
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
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Slow down."},
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)
    return await call_next(request)


# ── Schemas ────────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    top_k: int = Field(10, ge=1, le=50, description="Results to return after rerank.")
    candidates: int = Field(
        100, ge=10, le=200, description="Candidates retrieved before reranking."
    )
    ranker: Literal["lambdarank", "crossencoder"] = "lambdarank"
    use_hyde: bool = Field(
        True, description="Server-side HyDE query expansion (only if the server "
        "has an LLM provider configured; otherwise ignored)."
    )


class ResultItem(BaseModel):
    rank: int
    doc_id: int
    text: str
    score: float
    ranker: str


class StageCandidate(BaseModel):
    doc_id: int
    score: float
    rank: int


class Stages(BaseModel):
    intent: str
    hyde_used: bool
    embed_text_preview: str
    dense_top: list[StageCandidate]
    sparse_top: list[StageCandidate]
    fused_count: int


class Timings(BaseModel):
    hyde_ms: float
    retrieve_ms: float
    rerank_ms: float
    total_ms: float


class SearchResponse(BaseModel):
    request_id: str
    query: str
    ranker: str
    results: list[ResultItem]
    stages: Stages
    timings: Timings


class ClickRequest(BaseModel):
    request_id: str
    query: str
    doc_id: int
    rank: int
    ranker: str = "lambdarank"


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Cheap liveness/readiness probe. Also used by the keep-warm ping."""
    engine_ready = ENGINE is not None
    info: dict = {"status": "ok", "engine_ready": engine_ready}
    if engine_ready:
        info.update(
            {
                "device": str(getattr(ENGINE, "device", "cpu")),
                "index_size": len(getattr(ENGINE, "faiss_pid_list", []) or []),
                "cross_encoder": getattr(ENGINE, "cross_encoder", None) is not None,
                "llm_provider": getattr(LLM, "name", "none"),
                "llm_available": bool(getattr(LLM, "available", False)),
            }
        )
    return info


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if ENGINE is None:
        # 503: artifacts still loading (cold start) or load was skipped.
        return JSONResponse(status_code=503, content={"detail": "Engine not ready."})

    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    intent = _classify_intent(req.query)

    # Optional server-side HyDE (never uses the client's BYOK key).
    embed_text = req.query
    hyde_used = False
    t_hyde0 = time.perf_counter()
    if req.use_hyde and getattr(LLM, "available", False) and intent == "informational":
        try:
            embed_text = LLM.complete(
                "Write a short, factual passage (2-4 sentences) that would be the "
                "ideal answer to the user's question. Be specific and factual.",
                f"Question: {req.query}",
                max_tokens=256,
            )
            hyde_used = True
        except Exception:
            embed_text = req.query
    hyde_ms = (time.perf_counter() - t_hyde0) * 1000

    # Retrieve (with per-stage candidate lists for the breakdown).
    t_ret0 = time.perf_counter()
    dense = ENGINE._faiss(embed_text, req.candidates)
    sparse = ENGINE._bm25(req.query, req.candidates)
    fused = ENGINE._rrf(dense, sparse, req.candidates)
    cands = [
        {
            "doc_id": item["pid"],
            "text": ENGINE.pid_to_text.get(item["pid"], ""),
            "score": item["score"],
            "retrieval_rank": item["rank"],
        }
        for item in fused
    ]
    retrieve_ms = (time.perf_counter() - t_ret0) * 1000

    # Rerank.
    t_rk0 = time.perf_counter()
    results = ENGINE.rank(req.query, cands, top_k=req.top_k, ranker=req.ranker)
    rerank_ms = (time.perf_counter() - t_rk0) * 1000

    total_ms = (time.perf_counter() - t0) * 1000

    return SearchResponse(
        request_id=request_id,
        query=req.query,
        ranker=req.ranker,
        results=[ResultItem(**r) for r in results],
        stages=Stages(
            intent=intent,
            hyde_used=hyde_used,
            embed_text_preview=embed_text[:200],
            dense_top=[StageCandidate(doc_id=d["pid"], score=d["score"], rank=d["rank"]) for d in dense[:10]],
            sparse_top=[StageCandidate(doc_id=s["pid"], score=s["score"], rank=s["rank"]) for s in sparse[:10]],
            fused_count=len(fused),
        ),
        timings=Timings(
            hyde_ms=round(hyde_ms, 1),
            retrieve_ms=round(retrieve_ms, 1),
            rerank_ms=round(rerank_ms, 1),
            total_ms=round(total_ms, 1),
        ),
    )


@app.post("/click")
async def click(req: ClickRequest):
    """Best-effort click logging to Postgres (Neon). No-op if DB not configured."""
    if not os.getenv("DATABASE_URL") and not os.getenv("POSTGRES_HOST"):
        return {"logged": False, "reason": "no DATABASE_URL configured"}
    try:
        from datetime import datetime

        from services.shared.database import ClickLog, create_tables, get_db_session

        create_tables()
        session = get_db_session()
        try:
            session.add(
                ClickLog(
                    request_id=req.request_id,
                    query_text=req.query,
                    doc_id=int(req.doc_id),
                    rank_shown=int(req.rank),
                    ranker_version=req.ranker,
                    clicked=True,
                    created_at=datetime.utcnow(),
                )
            )
            session.commit()
        finally:
            session.close()
        return {"logged": True, "doc_id": req.doc_id, "rank": req.rank}
    except Exception as e:  # pragma: no cover - depends on live DB
        return {"logged": False, "reason": str(e)}


def _classify_intent(query: str) -> str:
    # Imported lazily so tests can run without importing the engine module's
    # heavy deps at collection time.
    from deploy.engine import classify_intent

    return classify_intent(query)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
