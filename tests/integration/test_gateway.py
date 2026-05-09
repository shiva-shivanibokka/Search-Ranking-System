"""
Integration tests for the API Gateway.
These run against a live gateway instance (requires docker-compose up).
Set GATEWAY_URL env var to override the default.
"""

import os
import pytest
import httpx

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def client():
    return httpx.Client(base_url=GATEWAY_URL, timeout=30.0)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_search_returns_results(client):
    resp = client.post(
        "/search", json={"query": "what is machine learning", "top_k": 5}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "request_id" in data
    assert len(data["results"]) <= 5
    assert "latency" in data
    assert "total_ms" in data["latency"]


def test_search_results_have_required_fields(client):
    resp = client.post("/search", json={"query": "information retrieval systems"})
    assert resp.status_code == 200
    data = resp.json()
    for result in data["results"]:
        assert "rank" in result
        assert "doc_id" in result
        assert "text" in result
        assert "score" in result
        assert "ranker" in result


def test_search_ranker_forced_lambdarank(client):
    resp = client.post("/search", json={"query": "test query", "ranker": "lambdarank"})
    assert resp.status_code == 200
    data = resp.json()
    for result in data["results"]:
        assert result["ranker"] == "lambdarank"


def test_search_ranker_forced_crossencoder(client):
    resp = client.post(
        "/search", json={"query": "test query", "ranker": "crossencoder"}
    )
    assert resp.status_code == 200
    data = resp.json()
    for result in data["results"]:
        assert result["ranker"] == "crossencoder"


def test_search_latency_under_500ms(client):
    """End-to-end latency should be under 500ms (excluding first cold start)."""
    # Warm up
    client.post("/search", json={"query": "warmup query"})
    # Measure
    resp = client.post("/search", json={"query": "what causes inflation"})
    data = resp.json()
    total_ms = data["latency"]["total_ms"]
    assert total_ms < 500, f"Latency too high: {total_ms}ms"


def test_search_empty_query_handled(client):
    resp = client.post("/search", json={"query": ""})
    # Should not crash — either return empty results or 422 validation error
    assert resp.status_code in (200, 422)


def test_search_request_id_unique(client):
    r1 = client.post("/search", json={"query": "test"})
    r2 = client.post("/search", json={"query": "test"})
    assert r1.json()["request_id"] != r2.json()["request_id"]


def test_second_request_hits_cache(client):
    """Second identical query should have lower retrieval latency (cache hit)."""
    query = "unique test query for cache check 12345"
    r1 = client.post("/search", json={"query": query})
    r2 = client.post("/search", json={"query": query})

    lat1 = r1.json()["latency"]["retrieval_ms"]
    lat2 = r2.json()["latency"]["retrieval_ms"]
    cache_hit = r2.json()["latency"].get("cache_hit", False)

    assert cache_hit is True, "Second request should be a cache hit"
    assert lat2 < lat1, (
        f"Cache miss ({lat1}ms) should be slower than cache hit ({lat2}ms)"
    )


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "gateway_requests_total" in resp.text
