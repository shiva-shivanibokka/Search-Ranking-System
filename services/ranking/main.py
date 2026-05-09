"""
Ranking Service — Port 8003

Responsibilities:
  1. Load LambdaRank (XGBoost) and CrossEncoder (DistilBERT) models at startup
  2. A/B test: route incoming requests to LambdaRank or CrossEncoder based on
     AB_CROSSENCODER_FRACTION env var (default 0.5 = 50/50 split)
  3. Rerank top-100 candidates → top-10 results
  4. Log which ranker variant was used (for CTR tracking per variant)
  5. Support hot-reload of LambdaRank model (picks up new model without restart)
     — used by the Airflow retraining DAG after promoting a new model

A/B variant is deterministic per request_id (hash-based split),
so the same request always routes to the same variant — reproducible.
"""

import os
import json
import time
import pickle
import hashlib
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

import numpy as np
import xgboost as xgb
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import structlog

from services.shared.logger import configure_logging, bind_request_id

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
    ranker: Optional[str] = None  # force specific ranker, else A/B


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


# ── A/B routing ───────────────────────────────────────────────────────────────


def _ab_variant(request_id: str) -> str:
    """
    Deterministic A/B variant assignment based on request_id hash.
    Same request_id always routes to same variant — reproducible.
    """
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    fraction = (h % 1000) / 1000.0
    return "crossencoder" if fraction < AB_CROSSENCODER_FRACTION else "lambdarank"


# ── LambdaRank reranking ──────────────────────────────────────────────────────


def _rerank_lambdarank(query: str, candidates: list[CandidateIn], top_k: int) -> list:
    if not candidates:
        return []

    bm25_scores_all = bm25.get_scores(query.lower().split())
    pid_to_bm25_idx = {pid: i for i, pid in enumerate(bm25_pid_list)}

    q_terms = set(query.lower().split())
    q_len = len(query.split())
    n = len(candidates)

    tt_scores = np.array([c.score for c in candidates])
    tt_rank_order = np.argsort(tt_scores)[::-1]
    tt_ranks = np.empty_like(tt_rank_order)
    tt_ranks[tt_rank_order] = np.arange(1, n + 1)

    X = []
    for i, cand in enumerate(candidates):
        bm25_idx = pid_to_bm25_idx.get(cand.doc_id, 0)
        bm25_score = float(bm25_scores_all[bm25_idx])
        doc_terms = set(cand.text.lower().split())
        overlap = len(q_terms & doc_terms) / max(len(q_terms), 1)
        doc_len = pid_to_len.get(cand.doc_id, 0)

        X.append(
            [
                bm25_score,
                float(cand.score),
                min(doc_len / 200.0, 5.0),
                overlap,
                min(q_len / 20.0, 3.0),
                cand.retrieval_rank / n,
                tt_ranks[i] / n,
            ]
        )

    dm = xgb.DMatrix(np.array(X, dtype=np.float32))
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
    Hot-reload the LambdaRank model from disk without restarting the service.
    Called by Airflow after promoting a new model version.
    """
    global lambdarank_booster
    try:
        new_booster = xgb.Booster()
        new_booster.load_model(LAMBDARANK_MODEL_PATH)
        lambdarank_booster = new_booster
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
        "ab_ce_fraction": AB_CROSSENCODER_FRACTION,
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/rank", response_model=RankResponse)
async def rank(req: RankRequest):
    bind_request_id(req.request_id)
    t0 = time.perf_counter()

    # Determine which ranker to use
    if req.ranker in ("lambdarank", "crossencoder"):
        ranker = req.ranker
        ab_variant = f"forced_{ranker}"
    else:
        ranker = _ab_variant(req.request_id)
        ab_variant = ranker

    logger.info(
        "ranking.start",
        ranker=ranker,
        ab_variant=ab_variant,
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
        num_results=len(results),
        latency_ms=round(latency_ms, 2),
    )

    return RankResponse(
        results=[RankedResult(**r) for r in results],
        ranker_used=ranker,
        ab_variant=ab_variant,
        latency_ms=round(latency_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.ranking.main:app", host="0.0.0.0", port=8003, reload=False)
