"""
Unit tests for the retrieval API (deploy/api.py).

These never load the real 1.2GB engine: they set API_SKIP_ENGINE_LOAD=1 and
monkeypatch deploy.api.ENGINE / .LLM with lightweight fakes. That keeps the
tests fast and CI-friendly while still exercising the full request/response
contract, the stage breakdown, HyDE gating, and rate limiting.
"""

from __future__ import annotations

import os

import pytest

os.environ["API_SKIP_ENGINE_LOAD"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

import deploy.api as api  # noqa: E402


class _FakeBM25:
    def get_scores(self, tokens):
        # deploy/api.py scans BM25 once and threads the vector through; the fake
        # value is unused by FakeEngine's fixed _bm25/rank, it just must not crash.
        return [0.0] * 10


class FakeEngine:
    """Minimal stand-in exposing exactly what deploy/api.py calls."""

    def __init__(self):
        self.device = "cpu"
        self.faiss_pid_list = list(range(1000))
        self.cross_encoder = None
        self.pid_to_text = {1: "alpha passage", 2: "beta passage", 3: "gamma passage"}
        self.bm25 = _FakeBM25()

    def _faiss(self, text, top_k):
        return [
            {"pid": 1, "score": 0.9, "rank": 1},
            {"pid": 2, "score": 0.8, "rank": 2},
        ][:top_k]

    def _bm25(self, query, top_k, scores=None):
        return [
            {"pid": 2, "score": 5.0, "rank": 1},
            {"pid": 3, "score": 4.0, "rank": 2},
        ][:top_k]

    def _rrf(self, dense, sparse, top_k):
        # pid 2 appears in both -> should rank first
        return [
            {"pid": 2, "score": 0.033, "rank": 1},
            {"pid": 1, "score": 0.016, "rank": 2},
            {"pid": 3, "score": 0.016, "rank": 3},
        ][:top_k]

    def rank(self, query, cands, top_k, ranker, bm25_scores_all=None):
        return [
            {"rank": i + 1, "doc_id": c["doc_id"], "text": c["text"],
             "score": 1.0 - i * 0.1, "ranker": ranker}
            for i, c in enumerate(cands[:top_k])
        ]


class FakeLLM:
    def __init__(self, available=False):
        self.name = "fake"
        self.available = available
        self.calls = 0

    def complete(self, system, user, *, max_tokens=256, temperature=0.0):
        self.calls += 1
        return "a hypothetical factual answer passage"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "ENGINE", FakeEngine())
    monkeypatch.setattr(api, "LLM", FakeLLM(available=False))
    # Disable rate limiting by default so unrelated tests don't trip it.
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MINUTE", 0)
    monkeypatch.setattr(api, "_rate_hits", {})
    with TestClient(api.app) as c:
        yield c


def test_health_ready(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine_ready"] is True
    assert body["index_size"] == 1000
    assert body["cross_encoder"] is False
    assert body["llm_provider"] == "fake"


def test_health_not_ready(monkeypatch):
    monkeypatch.setattr(api, "ENGINE", None)
    with TestClient(api.app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "engine_ready": False}


def test_search_returns_ranked_results_and_stages(client):
    r = client.post("/search", json={"query": "what causes inflation", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "what causes inflation"
    assert body["ranker"] == "lambdarank"
    assert [x["rank"] for x in body["results"]] == [1, 2, 3]
    # Stage breakdown present
    st = body["stages"]
    assert st["intent"] == "informational"
    assert st["hyde_used"] is False
    assert st["fused_count"] == 3
    assert len(st["dense_top"]) == 2
    assert len(st["sparse_top"]) == 2
    # Timings present
    assert set(body["timings"]) == {"hyde_ms", "retrieve_ms", "rerank_ms", "total_ms"}


def test_search_respects_top_k(client):
    r = client.post("/search", json={"query": "inflation", "top_k": 1})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 1


def test_search_validation_rejects_empty_query(client):
    r = client.post("/search", json={"query": "", "top_k": 3})
    assert r.status_code == 422


def test_search_validation_rejects_bad_top_k(client):
    r = client.post("/search", json={"query": "x", "top_k": 999})
    assert r.status_code == 422


def test_hyde_used_when_llm_available_and_informational(monkeypatch):
    monkeypatch.setattr(api, "ENGINE", FakeEngine())
    llm = FakeLLM(available=True)
    monkeypatch.setattr(api, "LLM", llm)
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MINUTE", 0)
    with TestClient(api.app) as c:
        r = c.post("/search", json={"query": "why is the sky blue", "use_hyde": True})
    assert r.status_code == 200
    body = r.json()
    assert body["stages"]["hyde_used"] is True
    assert llm.calls == 1
    assert "hypothetical" in body["stages"]["embed_text_preview"]


def test_hyde_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(api, "ENGINE", FakeEngine())
    llm = FakeLLM(available=True)
    monkeypatch.setattr(api, "LLM", llm)
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MINUTE", 0)
    with TestClient(api.app) as c:
        r = c.post("/search", json={"query": "why is the sky blue", "use_hyde": False})
    assert r.status_code == 200
    assert r.json()["stages"]["hyde_used"] is False
    assert llm.calls == 0


def test_search_503_when_engine_not_loaded(monkeypatch):
    monkeypatch.setattr(api, "ENGINE", None)
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MINUTE", 0)
    with TestClient(api.app) as c:
        r = c.post("/search", json={"query": "x"})
    assert r.status_code == 503


def test_rate_limit_returns_429(monkeypatch):
    monkeypatch.setattr(api, "ENGINE", FakeEngine())
    monkeypatch.setattr(api, "LLM", FakeLLM(available=False))
    monkeypatch.setattr(api, "RATE_LIMIT_PER_MINUTE", 2)
    monkeypatch.setattr(api, "_rate_hits", {})
    with TestClient(api.app) as c:
        assert c.post("/search", json={"query": "a"}).status_code == 200
        assert c.post("/search", json={"query": "b"}).status_code == 200
        r = c.post("/search", json={"query": "c"})
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_click_noop_without_db(client, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    r = client.post("/click", json={
        "request_id": "r1", "query": "q", "doc_id": 1, "rank": 1,
    })
    assert r.status_code == 200
    assert r.json()["logged"] is False
