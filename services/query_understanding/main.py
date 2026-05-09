"""
Query Understanding Service — Port 8001

Responsibilities:
  1. Intent classification: navigational / informational / transactional
     (rule-based fast path first, LLM only if ambiguous)
  2. Query rewriting: clean up malformed / ambiguous queries using Claude
  3. HyDE (Hypothetical Document Embedding): for informational queries,
     generate a hypothetical answer and return it as an expanded query
     → this is then embedded by the retrieval service to improve recall

Uses Anthropic Claude claude-haiku-4-5 (fast + cheap) for all LLM calls.
Skips LLM if intent is clear from simple rules (saves latency + cost).
"""

import os
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import anthropic
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import structlog

from services.shared.logger import configure_logging, bind_request_id

configure_logging("query-understanding")
logger = structlog.get_logger()

# ── Schemas ───────────────────────────────────────────────────────────────────


class UnderstandRequest(BaseModel):
    query: str
    request_id: str


class UnderstandResponse(BaseModel):
    rewritten_query: str
    intent: str  # navigational | informational | transactional
    hyde_passage: Optional[str] = None  # hypothetical answer for HyDE
    rewrite_applied: bool = False
    latency_ms: float = 0.0


# ── Prometheus ────────────────────────────────────────────────────────────────
QU_REQUESTS = Counter("qu_requests_total", "Total QU requests", ["intent"])
QU_LLM_CALLS = Counter("qu_llm_calls_total", "LLM calls made", ["call_type"])
QU_LATENCY = Histogram(
    "qu_latency_ms", "QU service latency", buckets=[5, 10, 20, 30, 50, 100, 200]
)

# ── Intent classification rules ───────────────────────────────────────────────
NAVIGATIONAL_PATTERNS = [
    r"^(how to get to|directions to|location of|address of|where is)\b",
    r"\b(homepage|website|official site|login|sign in)\b",
    r"^(github|twitter|linkedin|youtube|facebook|wikipedia)\b",
]
TRANSACTIONAL_PATTERNS = [
    r"\b(buy|purchase|order|price|cheap|discount|deal|subscribe|download|install)\b",
    r"\b(best|top|review|compare|vs\.?|versus)\b",
]
INFORMATIONAL_PATTERNS = [
    r"^(what|why|how|when|who|which|explain|define|describe)\b",
    r"\b(meaning|definition|difference between|example of|tutorial)\b",
]


def classify_intent_rules(query: str) -> Optional[str]:
    """Fast rule-based intent classification. Returns None if ambiguous."""
    q_lower = query.lower().strip()
    for pattern in NAVIGATIONAL_PATTERNS:
        if re.search(pattern, q_lower):
            return "navigational"
    for pattern in TRANSACTIONAL_PATTERNS:
        if re.search(pattern, q_lower):
            return "transactional"
    for pattern in INFORMATIONAL_PATTERNS:
        if re.search(pattern, q_lower):
            return "informational"
    return None


# ── LLM client ────────────────────────────────────────────────────────────────


class QueryUnderstandingLLM:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("LLM_MODEL", "claude-haiku-4-5")

    def _call(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()

    def classify_intent(self, query: str) -> str:
        """Classify query intent using Claude when rules are ambiguous."""
        QU_LLM_CALLS.labels(call_type="intent_classification").inc()
        system = (
            "You are a search query intent classifier. "
            "Classify the user's query into exactly one of: navigational, informational, transactional. "
            "Reply with only the single word classification."
        )
        result = self._call(system, f"Query: {query}")
        result = result.lower().strip()
        if result not in ("navigational", "informational", "transactional"):
            return "informational"
        return result

    def rewrite_query(self, query: str) -> str:
        """Rewrite ambiguous or poorly formed queries into clear standalone queries."""
        QU_LLM_CALLS.labels(call_type="query_rewrite").inc()
        system = (
            "You are a search query rewriter. "
            "Rewrite the user's query to be clearer and more specific for document retrieval. "
            "Return only the rewritten query, nothing else. "
            "If the query is already clear, return it unchanged."
        )
        return self._call(system, f"Query: {query}")

    def generate_hyde_passage(self, query: str) -> str:
        """
        HyDE: Generate a hypothetical document that would answer the query.
        The embedding of this passage improves retrieval quality for
        informational queries by bridging the query-document embedding gap.
        """
        QU_LLM_CALLS.labels(call_type="hyde").inc()
        system = (
            "You are a search assistant. Write a short, factual passage (2-4 sentences) "
            "that would be the ideal answer to the user's question. "
            "Write as if you are an expert. Be specific and factual."
        )
        return self._call(system, f"Question: {query}")


# ── App ───────────────────────────────────────────────────────────────────────

llm: Optional[QueryUnderstandingLLM] = None
REWRITE_CONFIDENCE_THRESHOLD = float(os.getenv("REWRITE_THRESHOLD", "0.4"))
ENABLE_HYDE = os.getenv("ENABLE_HYDE", "true").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    try:
        llm = QueryUnderstandingLLM()
        logger.info("llm.initialized", model=llm.model)
    except Exception as e:
        logger.warning("llm.init_failed", error=str(e))
        llm = None
    yield


app = FastAPI(title="Query Understanding Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "query-understanding"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/understand", response_model=UnderstandResponse)
async def understand(req: UnderstandRequest):
    bind_request_id(req.request_id)
    t0 = time.perf_counter()

    query = req.query.strip()
    rewrite_applied = False
    hyde_passage = None

    # ── Step 1: Intent classification ─────────────────────────────────────────
    intent = classify_intent_rules(query)
    if intent is None:
        # Rules were ambiguous — call LLM
        if llm:
            intent = llm.classify_intent(query)
        else:
            intent = "informational"

    QU_REQUESTS.labels(intent=intent).inc()
    logger.info("intent.classified", intent=intent, rule_based=(intent is not None))

    # ── Step 2: Query rewriting (only for short/ambiguous queries) ─────────────
    rewritten_query = query
    words = query.split()
    is_short = len(words) <= 3
    is_ambiguous = intent == "informational" and is_short

    if is_ambiguous and llm:
        rewritten = llm.rewrite_query(query)
        if rewritten.lower() != query.lower() and len(rewritten) > len(query):
            rewritten_query = rewritten
            rewrite_applied = True
            logger.info("query.rewritten", original=query, rewritten=rewritten_query)

    # ── Step 3: HyDE for informational queries ─────────────────────────────────
    if ENABLE_HYDE and intent == "informational" and llm:
        try:
            hyde_passage = llm.generate_hyde_passage(rewritten_query)
            logger.info("hyde.generated", length=len(hyde_passage))
        except Exception as e:
            logger.warning("hyde.failed", error=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000
    QU_LATENCY.observe(latency_ms)

    return UnderstandResponse(
        rewritten_query=rewritten_query,
        intent=intent,
        hyde_passage=hyde_passage,
        rewrite_applied=rewrite_applied,
        latency_ms=round(latency_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.query_understanding.main:app", host="0.0.0.0", port=8001, reload=False
    )
