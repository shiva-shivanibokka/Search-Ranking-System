"""
Ranking Service — Port 8003

Responsibilities:
  1. Load LambdaRank, CrossEncoder, and DifficultyClassifier models at startup
  2. Routing strategy (in priority order):
     a. If caller forces a specific ranker (req.ranker) → use that
     b. If difficulty classifier is loaded → use difficulty-based routing:
          - predict query difficulty score (0=easy, 1=hard)
          - hard queries  → CrossEncoder  (slow, accurate)
          - easy queries  → LambdaRank    (fast, good enough)
     c. Fallback → A/B hash-based split (when classifier not loaded)
  3. Rerank top-100 candidates → top-10 results
  4. Log ranker variant + routing method for CTR tracking and analysis
  5. Support hot-reload of LambdaRank model without restart

Difficulty routing vs A/B testing
───────────────────────────────────
A/B testing splits traffic randomly to measure aggregate CTR differences.
Difficulty routing uses the classifier to make per-query routing decisions.

Both run simultaneously: difficulty routing determines the *default* ranker,
while A/B is used when the classifier is not available. This means once the
classifier is trained and deployed, the system shifts from random allocation
to intelligent allocation while still logging routing decisions for analysis.
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
DIFFICULTY_SCORE = Histogram(
    "ranking_difficulty_score",
    "Predicted query difficulty score distribution",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
ROUTING_METHOD = Counter(
    "ranking_routing_method_total",
    "How routing decisions were made",
    ["method"],  # "difficulty_classifier" | "ab_split" | "forced"
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
    ranker: Optional[str] = None  # force specific ranker, else auto-route
    intent: Optional[str] = None  # from query understanding; used by difficulty router


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
    routing_method: str  # "difficulty_classifier" | "ab_split" | "forced"
    difficulty_score: Optional[float] = None  # None when not using classifier


# ── Global models ─────────────────────────────────────────────────────────────

lambdarank_booster: Optional[xgb.Booster] = None
lambdarank_feature_names: Optional[list] = None
cross_encoder_model = None
ce_tokenizer = None
bm25 = None
bm25_pid_list = None
pid_to_len: dict = {}
difficulty_booster: Optional[xgb.Booster] = None
difficulty_meta: Optional[dict] = None

AB_CROSSENCODER_FRACTION = float(os.getenv("AB_CROSSENCODER_FRACTION", "0.5"))
LAMBDARANK_MODEL_PATH = os.getenv(
    "LAMBDARANK_MODEL_PATH", "models/lambdarank/lambdarank.json"
)
CROSS_ENCODER_DIR = os.getenv("CROSS_ENCODER_DIR", "models/cross_encoder")
DIFFICULTY_CLASSIFIER_DIR = os.getenv(
    "DIFFICULTY_CLASSIFIER_DIR", "models/difficulty_classifier"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global lambdarank_booster, lambdarank_feature_names
    global cross_encoder_model, ce_tokenizer
    global bm25, bm25_pid_list, pid_to_len
    global difficulty_booster, difficulty_meta

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

    # Load difficulty classifier (optional — falls back to A/B if not present)
    diff_model_path = Path(DIFFICULTY_CLASSIFIER_DIR) / "difficulty_classifier.json"
    diff_meta_path = Path(DIFFICULTY_CLASSIFIER_DIR) / "classifier_meta.json"
    if diff_model_path.exists() and diff_meta_path.exists():
        logger.info("loading.difficulty_classifier", path=str(diff_model_path))
        difficulty_booster = xgb.Booster()
        difficulty_booster.load_model(str(diff_model_path))
        with open(diff_meta_path) as f:
            difficulty_meta = json.load(f)
        logger.info(
            "difficulty_classifier.loaded",
            val_auc=difficulty_meta.get("val_auc"),
            routing_threshold=difficulty_meta.get("routing_threshold"),
            ce_routing_rate=difficulty_meta.get("ce_routing_rate"),
        )
    else:
        logger.info(
            "difficulty_classifier.not_found",
            path=str(diff_model_path),
            fallback="ab_split",
        )

    logger.info("ranking.ready")

    yield


app = FastAPI(title="Ranking Service", version="1.0.0", lifespan=lifespan)


# ── Routing ───────────────────────────────────────────────────────────────────


def _ab_variant(request_id: str) -> str:
    """
    Deterministic A/B variant assignment based on request_id hash.
    Same request_id always routes to same variant — reproducible.
    Used as fallback when difficulty classifier is not loaded.
    """
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    fraction = (h % 1000) / 1000.0
    return "crossencoder" if fraction < AB_CROSSENCODER_FRACTION else "lambdarank"


def _difficulty_route(
    query: str,
    candidates: list,
    intent: Optional[str] = None,
) -> tuple[str, float]:
    """
    Use the difficulty classifier to route query to LambdaRank or CrossEncoder.

    Computes the same 8 features used during training from live request data.
    Falls back to A/B routing if classifier not available.

    Returns:
      (ranker_name, difficulty_score)  where difficulty_score ∈ [0, 1]
    """
    if difficulty_booster is None or difficulty_meta is None:
        return _ab_variant("fallback"), 0.5

    threshold = difficulty_meta.get("routing_threshold", 0.5)

    # Feature 1: query_length
    tokens = query.split()
    query_length = float(len(tokens))

    # Feature 2: query_entropy
    import math

    token_counts = {}
    for t in tokens:
        token_counts[t] = token_counts.get(t, 0) + 1
    total = len(tokens)
    query_entropy = 0.0
    if total > 0:
        for count in token_counts.values():
            p = count / total
            if p > 0:
                query_entropy -= p * math.log2(p)

    # Features 3-5: from candidate scores (BM25 scores not available here —
    # use TT scores from candidates as proxy for ranking uncertainty)
    tt_scores = [c.score for c in candidates[:10]] if candidates else [0.0]
    tt_score_variance = float(np.var(tt_scores)) if len(tt_scores) > 1 else 0.0

    sorted_scores = sorted(tt_scores, reverse=True)
    bm25_score_gap = (
        float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0.0
    )
    top1_tt_score = float(sorted_scores[0]) if sorted_scores else 0.0

    # tt_bm25_score_ratio: approximate with normalized TT score gap
    tt_bm25_score_ratio = float(np.mean(tt_scores)) / (abs(bm25_score_gap) + 1e-8)

    # Feature 6: intent_is_informational
    intent_is_informational = 1.0 if intent == "informational" else 0.0

    # Feature 7: top1_bm25_score — not available in ranking service;
    # use retrieval rank 1 candidate's score as proxy
    top1_bm25_score = float(candidates[0].score) if candidates else 0.0

    features = np.array(
        [
            query_length,
            query_entropy,
            bm25_score_gap,
            tt_score_variance,
            tt_bm25_score_ratio,
            intent_is_informational,
            top1_bm25_score,
            top1_tt_score,
        ],
        dtype=np.float32,
    ).reshape(1, -1)

    dm = xgb.DMatrix(features, feature_names=difficulty_meta.get("features", None))
    difficulty_score = float(difficulty_booster.predict(dm)[0])
    ranker = "crossencoder" if difficulty_score >= threshold else "lambdarank"
    return ranker, difficulty_score


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
        "difficulty_classifier_loaded": difficulty_booster is not None,
        "routing_mode": "difficulty_classifier"
        if difficulty_booster is not None
        else "ab_split",
        "ab_ce_fraction": AB_CROSSENCODER_FRACTION,
        "difficulty_val_auc": difficulty_meta.get("val_auc")
        if difficulty_meta
        else None,
        "difficulty_ce_routing_rate": difficulty_meta.get("ce_routing_rate")
        if difficulty_meta
        else None,
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/rank", response_model=RankResponse)
async def rank(req: RankRequest):
    bind_request_id(req.request_id)
    t0 = time.perf_counter()

    difficulty_score: Optional[float] = None
    routing_method: str

    # ── Routing decision (priority: forced > difficulty > A/B) ────────────────
    if req.ranker in ("lambdarank", "crossencoder"):
        # Caller explicitly forced a ranker (e.g. Gradio UI comparison tab)
        ranker = req.ranker
        ab_variant = f"forced_{ranker}"
        routing_method = "forced"
        ROUTING_METHOD.labels(method="forced").inc()

    elif difficulty_booster is not None:
        # Difficulty classifier is loaded — use intelligent per-query routing
        ranker, difficulty_score = _difficulty_route(
            req.query, req.candidates, intent=req.intent
        )
        ab_variant = ranker
        routing_method = "difficulty_classifier"
        DIFFICULTY_SCORE.observe(difficulty_score)
        ROUTING_METHOD.labels(method="difficulty_classifier").inc()

    else:
        # Classifier not trained yet — fall back to deterministic A/B split
        ranker = _ab_variant(req.request_id)
        ab_variant = ranker
        routing_method = "ab_split"
        ROUTING_METHOD.labels(method="ab_split").inc()

    logger.info(
        "ranking.start",
        ranker=ranker,
        routing_method=routing_method,
        difficulty_score=round(difficulty_score, 4)
        if difficulty_score is not None
        else None,
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
        difficulty_score=difficulty_score,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.ranking.main:app", host="0.0.0.0", port=8003, reload=False)
