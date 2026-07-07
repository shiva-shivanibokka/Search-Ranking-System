"""
Ranking Service — Port 8003

Responsibilities:
  1. Load LambdaRank and CrossEncoder models at startup
  2. Routing strategy (in priority order):
     a. If caller forces a specific ranker (req.ranker) → use that
     b. Otherwise → deterministic A/B hash-based split between LambdaRank
        and CrossEncoder
  3. Rerank top-100 candidates → top-10 results
  4. Log ranker variant + routing method for CTR tracking and analysis
  5. Support hot-reload of LambdaRank model without restart

A/B testing
───────────
A/B testing splits traffic deterministically (by request_id hash) between the
two rankers to measure aggregate CTR differences. The same request_id always
routes to the same variant, so results stay consistent across repeated queries.
"""

import hashlib
import json
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
import torch
import xgboost as xgb
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

from services.shared.features import Candidate, build_lambdarank_features
from services.shared.logger import bind_request_id, configure_logging

configure_logging("ranking")
logger = structlog.get_logger()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Prometheus ────────────────────────────────────────────────────────────────
RANKING_REQUESTS = Counter(
    "ranking_requests_total", "Total ranking requests", ["ranker", "ab_variant"]
)
RANKING_LATENCY = Histogram(
    "ranking_latency_ms",
    "Reranking latency",
    ["ranker"],
    buckets=[10, 25, 50, 75, 100, 150, 200, 300, 500],
)
AB_NDCG = Histogram(
    "ranking_ab_ndcg",
    "NDCG score per A/B variant (approximated from click signal)",
    ["variant"],
)
ROUTING_METHOD = Counter(
    "ranking_routing_method_total",
    "How routing decisions were made",
    ["method"],  # "ab_split" | "forced"
)

# ── Schemas ───────────────────────────────────────────────────────────────────


class CandidateIn(BaseModel):
    doc_id: int
    text: str
    score: float
    retrieval_rank: int


class RankRequest(BaseModel):
    query: str
    request_id: str
    candidates: list[CandidateIn]
    top_k: int = 10
    ranker: Optional[str] = None  # force specific ranker, else A/B split
    intent: Optional[str] = None  # from query understanding; used for logging


class RankedResult(BaseModel):
    rank: int
    doc_id: int
    text: str
    score: float
    ranker: str


class RankResponse(BaseModel):
    results: list[RankedResult]
    ranker_used: str
    ab_variant: str
    latency_ms: float
    routing_method: str  # "ab_split" | "forced"


# ── Global models ─────────────────────────────────────────────────────────────

lambdarank_booster: Optional[xgb.Booster] = None
lambdarank_feature_names: Optional[list] = None
cross_encoder_model = None
ce_tokenizer = None
bm25 = None
bm25_pid_list = None
pid_to_len: dict = {}

AB_CROSSENCODER_FRACTION = float(os.getenv("AB_CROSSENCODER_FRACTION", "0.5"))
LAMBDARANK_MODEL_PATH = os.getenv(
    "LAMBDARANK_MODEL_PATH", "models/lambdarank/lambdarank.json"
)
CROSS_ENCODER_DIR = os.getenv("CROSS_ENCODER_DIR", "models/cross_encoder")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global lambdarank_booster, lambdarank_feature_names
    global cross_encoder_model, ce_tokenizer
    global bm25, bm25_pid_list, pid_to_len

    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

    # Load LambdaRank
    logger.info("loading.lambdarank", path=LAMBDARANK_MODEL_PATH)
    lambdarank_booster = xgb.Booster()
    lambdarank_booster.load_model(LAMBDARANK_MODEL_PATH)
    feat_path = str(Path(LAMBDARANK_MODEL_PATH).parent / "feature_names.json")
    with open(feat_path) as f:
        lambdarank_feature_names = json.load(f)["features"]
    logger.info("lambdarank.loaded")

    # Load CrossEncoder
    logger.info("loading.cross_encoder", dir=CROSS_ENCODER_DIR)
    from training.train_cross_encoder import load_cross_encoder

    cross_encoder_model, ce_tokenizer = load_cross_encoder(
        CROSS_ENCODER_DIR, device=str(DEVICE)
    )
    logger.info("cross_encoder.loaded")

    # Load BM25 (for LambdaRank features)
    bm25_path = os.getenv("BM25_INDEX_PATH", "data/indexes/bm25_index.pkl")
    bm25_pid_path = os.getenv("BM25_PID_PATH", "data/indexes/bm25_pid_list.pkl")
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(bm25_pid_path, "rb") as f:
        bm25_pid_list = pickle.load(f)

    # Load passage lengths for LambdaRank features
    import pandas as pd

    passages_path = os.getenv("PASSAGES_PATH", "data/processed/passages.parquet")
    passages_df = pd.read_parquet(passages_path, columns=["pid", "token_count"])
    pid_to_len = dict(
        zip(passages_df["pid"].tolist(), passages_df["token_count"].tolist())
    )

    logger.info("ranking.ready")

    yield


app = FastAPI(title="Ranking Service", version="1.0.0", lifespan=lifespan)


# ── Routing ───────────────────────────────────────────────────────────────────


def _ab_variant(request_id: str) -> str:
    """
    Deterministic A/B variant assignment based on request_id hash.
    Same request_id always routes to same variant — reproducible.
    """
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    fraction = (h % 1000) / 1000.0
    return "crossencoder" if fraction < AB_CROSSENCODER_FRACTION else "lambdarank"


# ── LambdaRank reranking ──────────────────────────────────────────────────────


def _feature_matrix(query: str, candidates: list[CandidateIn]) -> np.ndarray:
    """Build the shared LambdaRank feature matrix from module-level index globals."""
    cands = [
        Candidate(doc_id=c.doc_id, text=c.text, score=c.score, retrieval_rank=c.retrieval_rank)
        for c in candidates
    ]
    return build_lambdarank_features(query, cands, bm25, bm25_pid_list, pid_to_len)


def _rerank_lambdarank(query: str, candidates: list[CandidateIn], top_k: int) -> list:
    if not candidates:
        return []

    X = _feature_matrix(query, candidates)

    dm = xgb.DMatrix(X)
    lr_scores = lambdarank_booster.predict(dm)
    ranked_indices = lr_scores.argsort()[::-1][:top_k]

    return [
        {
            "rank": rank + 1,
            "doc_id": candidates[i].doc_id,
            "text": candidates[i].text,
            "score": float(lr_scores[i]),
            "ranker": "lambdarank",
        }
        for rank, i in enumerate(ranked_indices)
    ]


# ── CrossEncoder reranking ────────────────────────────────────────────────────


def _rerank_crossencoder(
    query: str, candidates: list[CandidateIn], top_k: int, batch_size: int = 32
) -> list:
    if not candidates:
        return []

    max_seq_len = int(os.getenv("CE_MAX_SEQ_LEN", "256"))
    all_scores = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        texts = [c.text for c in batch]
        enc = ce_tokenizer(
            [query] * len(texts),
            texts,
            max_length=max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            scores = (
                cross_encoder_model.predict_score(
                    enc["input_ids"].to(DEVICE),
                    enc["attention_mask"].to(DEVICE),
                )
                .cpu()
                .numpy()
            )
        all_scores.extend(zip(range(i, i + len(batch)), scores))

    all_scores.sort(key=lambda x: x[1], reverse=True)
    top = all_scores[:top_k]

    return [
        {
            "rank": rank + 1,
            "doc_id": candidates[idx].doc_id,
            "text": candidates[idx].text,
            "score": float(score),
            "ranker": "crossencoder",
        }
        for rank, (idx, score) in enumerate(top)
    ]


# ── Hot reload ────────────────────────────────────────────────────────────────


@app.post("/reload/lambdarank")
async def reload_lambdarank():
    """
    Hot-reload the LambdaRank model and feature names from disk without restarting.
    Called by Airflow after promoting a new model version.
    """
    global lambdarank_booster, lambdarank_feature_names
    try:
        new_booster = xgb.Booster()
        new_booster.load_model(LAMBDARANK_MODEL_PATH)
        feat_path = str(Path(LAMBDARANK_MODEL_PATH).parent / "feature_names.json")
        with open(feat_path) as f:
            new_feature_names = json.load(f)["features"]
        lambdarank_booster = new_booster
        lambdarank_feature_names = new_feature_names
        logger.info("lambdarank.hot_reloaded", path=LAMBDARANK_MODEL_PATH)
        return {"status": "reloaded", "path": LAMBDARANK_MODEL_PATH}
    except Exception as e:
        logger.error("lambdarank.reload_failed", error=str(e))
        return {"status": "error", "detail": str(e)}


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "ranking",
        "lambdarank_loaded": lambdarank_booster is not None,
        "crossencoder_loaded": cross_encoder_model is not None,
        "routing_mode": "ab_split",
        "ab_ce_fraction": AB_CROSSENCODER_FRACTION,
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/rank", response_model=RankResponse)
async def rank(req: RankRequest):
    bind_request_id(req.request_id)
    t0 = time.perf_counter()

    routing_method: str

    # ── Routing decision (priority: forced > A/B split) ───────────────────────
    if req.ranker in ("lambdarank", "crossencoder"):
        # Caller explicitly forced a ranker (e.g. the frontend's ranker selector)
        ranker = req.ranker
        ab_variant = f"forced_{ranker}"
        routing_method = "forced"
        ROUTING_METHOD.labels(method="forced").inc()

    else:
        # Deterministic A/B split between LambdaRank and CrossEncoder
        ranker = _ab_variant(req.request_id)
        ab_variant = ranker
        routing_method = "ab_split"
        ROUTING_METHOD.labels(method="ab_split").inc()

    logger.info(
        "ranking.start",
        ranker=ranker,
        routing_method=routing_method,
        num_candidates=len(req.candidates),
    )

    if ranker == "crossencoder":
        results = _rerank_crossencoder(req.query, req.candidates, req.top_k)
    else:
        results = _rerank_lambdarank(req.query, req.candidates, req.top_k)

    latency_ms = (time.perf_counter() - t0) * 1000
    RANKING_REQUESTS.labels(ranker=ranker, ab_variant=ab_variant).inc()
    RANKING_LATENCY.labels(ranker=ranker).observe(latency_ms)

    logger.info(
        "ranking.complete",
        ranker=ranker,
        routing_method=routing_method,
        num_results=len(results),
        latency_ms=round(latency_ms, 2),
    )

    return RankResponse(
        results=[RankedResult(**r) for r in results],
        ranker_used=ranker,
        ab_variant=ab_variant,
        latency_ms=round(latency_ms, 2),
        routing_method=routing_method,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.ranking.main:app", host="0.0.0.0", port=8003, reload=False)
