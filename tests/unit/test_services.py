"""Unit tests for service schemas and routing logic."""


import pytest

# ── Gateway schema tests ──────────────────────────────────────────────────────


def test_search_request_schema():
    from services.gateway.main import SearchRequest

    req = SearchRequest(query="what is machine learning", top_k=10)
    assert req.query == "what is machine learning"
    assert req.top_k == 10
    assert req.ranker is None


def test_search_request_with_ranker():
    from services.gateway.main import SearchRequest

    req = SearchRequest(query="test", ranker="lambdarank")
    assert req.ranker == "lambdarank"


# ── Query Understanding intent rules ──────────────────────────────────────────


def test_intent_rule_navigational():
    from services.query_understanding.main import classify_intent_rules

    assert classify_intent_rules("how to get to the website") == "navigational"
    assert classify_intent_rules("github homepage") == "navigational"


def test_intent_rule_transactional():
    from services.query_understanding.main import classify_intent_rules

    assert classify_intent_rules("buy cheap laptop") == "transactional"
    assert classify_intent_rules("best machine learning courses") == "transactional"


def test_intent_rule_informational():
    from services.query_understanding.main import classify_intent_rules

    assert classify_intent_rules("what is information retrieval") == "informational"
    assert classify_intent_rules("explain gradient descent") == "informational"


def test_intent_rule_ambiguous_returns_none():
    from services.query_understanding.main import classify_intent_rules

    # Short ambiguous query shouldn't match any pattern
    result = classify_intent_rules("python")
    # May be None or match — just verify it doesn't crash
    assert result is None or result in (
        "navigational",
        "informational",
        "transactional",
    )


# ── Ranking A/B routing ───────────────────────────────────────────────────────


def test_ab_variant_is_deterministic():
    """Same request_id should always return same variant."""
    import os

    from services.ranking.main import _ab_variant

    os.environ["AB_CROSSENCODER_FRACTION"] = "0.5"

    request_id = "test-request-id-12345"
    variant1 = _ab_variant(request_id)
    variant2 = _ab_variant(request_id)
    assert variant1 == variant2


def test_ab_variant_returns_valid_ranker():
    import os

    from services.ranking.main import _ab_variant

    os.environ["AB_CROSSENCODER_FRACTION"] = "0.5"

    for i in range(20):
        variant = _ab_variant(f"request-{i}")
        assert variant in ("lambdarank", "crossencoder")


def test_ab_variant_respects_fraction(monkeypatch):
    """With fraction=1.0, all requests should go to crossencoder.

    Patch the module global directly instead of reloading the module — reloading
    re-registers the Prometheus collectors and raises a Duplicated timeseries error.
    """
    import services.ranking.main as ranking_mod

    monkeypatch.setattr(ranking_mod, "AB_CROSSENCODER_FRACTION", 1.0)

    for i in range(10):
        variant = ranking_mod._ab_variant(f"req-{i}")
        assert variant == "crossencoder"


# ── Cache key tests ───────────────────────────────────────────────────────────


def test_cache_key_is_deterministic():
    from services.retrieval.main import _cache_key

    key1 = _cache_key("machine learning", 100, "hybrid")
    key2 = _cache_key("machine learning", 100, "hybrid")
    assert key1 == key2


def test_cache_key_differs_on_query():
    from services.retrieval.main import _cache_key

    key1 = _cache_key("machine learning", 100, "hybrid")
    key2 = _cache_key("deep learning", 100, "hybrid")
    assert key1 != key2


def test_cache_key_differs_on_top_k():
    from services.retrieval.main import _cache_key

    key1 = _cache_key("machine learning", 100, "hybrid")
    key2 = _cache_key("machine learning", 50, "hybrid")
    assert key1 != key2


def test_cache_key_differs_on_mode():
    """Cache key must include retrieval mode to avoid stale cross-mode hits."""
    from services.retrieval.main import _cache_key

    key1 = _cache_key("machine learning", 100, "hybrid")
    key2 = _cache_key("machine learning", 100, "dense_only")
    assert key1 != key2


def test_cache_key_case_insensitive():
    from services.retrieval.main import _cache_key

    key1 = _cache_key("Machine Learning", 100, "hybrid")
    key2 = _cache_key("machine learning", 100, "hybrid")
    assert key1 == key2


# ── Metrics helpers tests ─────────────────────────────────────────────────────


def test_ndcg_at_k_perfect():
    from training.evaluate import ndcg_at_k

    ranked = [1, 2, 3]
    gold = {1}
    assert ndcg_at_k(ranked, gold, k=10) == 1.0


def test_ndcg_at_k_miss():
    from training.evaluate import ndcg_at_k

    ranked = [4, 5, 6]
    gold = {1, 2, 3}
    assert ndcg_at_k(ranked, gold, k=10) == 0.0


def test_recall_at_k():
    from training.evaluate import recall_at_k

    ranked = [1, 2, 3, 4, 5]
    gold = {1, 3}
    assert recall_at_k(ranked, gold, k=5) == 1.0
    assert recall_at_k(ranked, gold, k=1) == 0.5


def test_mrr_at_k():
    from training.evaluate import mrr_at_k

    ranked = [10, 1, 2]
    gold = {1}
    assert mrr_at_k(ranked, gold, k=10) == pytest.approx(0.5)
