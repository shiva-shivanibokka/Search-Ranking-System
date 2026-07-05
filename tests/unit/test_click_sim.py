"""Unit tests for the click simulator (MS MARCO replay + ORCAS-weighted PBM).

These tests are pure/offline: no real model, no real qrels parquet, no
network. `simulate` is exercised against a fake engine + synthetic DataFrames
+ an in-memory SQLite session.
"""

from __future__ import annotations

import random

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def test_sample_clicks_prefers_relevant_top():
    from scripts.simulate_clicks import sample_clicks

    propensity = {1: 1.0, 10: 0.1}
    gold_pids = {100}
    rng = random.Random(0)

    shown = [
        {"doc_id": 100, "rank": 1},  # relevant, rank 1
        {"doc_id": 200, "rank": 1},  # irrelevant, rank 1
    ]

    clicks_relevant = 0
    clicks_irrelevant = 0
    for _ in range(2000):
        for doc_id, _rank, clicked in sample_clicks(shown, gold_pids, propensity, rng):
            if not clicked:
                continue
            if doc_id == 100:
                clicks_relevant += 1
            else:
                clicks_irrelevant += 1

    assert clicks_relevant > clicks_irrelevant

    # Propensity effect: same relevant doc, rank 1 vs. rank 10.
    rng2 = random.Random(1)
    clicks_rank1 = 0
    clicks_rank10 = 0
    for _ in range(2000):
        (_, _, clicked1), = sample_clicks(
            [{"doc_id": 100, "rank": 1}], gold_pids, propensity, rng2
        )
        (_, _, clicked10), = sample_clicks(
            [{"doc_id": 100, "rank": 10}], gold_pids, propensity, rng2
        )
        clicks_rank1 += int(clicked1)
        clicks_rank10 += int(clicked10)

    assert clicks_rank10 < clicks_rank1


def test_load_replay_workload_from_qrels():
    from scripts.simulate_clicks import load_replay_workload

    queries_df = pd.DataFrame(
        {"qid": ["q1", "q2", "q3"], "text": ["one", "two", "three"]}
    )
    qrels_df = pd.DataFrame(
        {"qid": ["q1", "q2", "q2"], "pid": [5, 6, 7]}
    )

    qid_to_text, qid_to_gold = load_replay_workload(queries_df, qrels_df)

    assert qid_to_gold == {"q1": {5}, "q2": {6, 7}}
    # Only qids present in qrels are kept — q3 has no qrels, so it is dropped
    # from both dicts (every workload qid is guaranteed a gold set).
    assert set(qid_to_text.keys()) == {"q1", "q2"}
    assert qid_to_text["q1"] == "one"
    assert qid_to_text["q2"] == "two"


def test_build_sampling_weights_uses_orcas_popularity():
    from scripts.simulate_clicks import build_sampling_weights

    qid_to_text = {"q1": "machine learning", "q2": "rare query"}
    query_popularity = {"machine learning": 50}

    weights, matched = build_sampling_weights(qid_to_text, query_popularity)

    assert weights["q1"] == 50.0
    assert weights["q2"] == 1.0
    assert matched == 1


class _FakeEngine:
    """Fake SearchEngine stand-in: fixed candidate list, no model/network."""

    def retrieve(self, query, embed_text, top_k):
        return [{"doc_id": 5}, {"doc_id": 6}, {"doc_id": 7}]


class _Cfg:
    queries = 1
    top_k = 10
    irrelevant_ctr = 0.02


def test_simulate_records_impressions_and_negatives():
    from scripts.simulate_clicks import simulate
    from services.shared.database import Base, ClickLog, ImpressionLog

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    qid_to_text = {"q1": "q one"}
    qid_to_gold = {"q1": {5}}
    weights = {"q1": 1.0}
    calibration = {
        "propensity": {"1": 1.0, "2": 0.7, "3": 0.5},
        "query_popularity": {},
    }

    result = simulate(
        _FakeEngine(),
        qid_to_text,
        qid_to_gold,
        weights,
        calibration,
        _Cfg(),
        session,
        seed=0,
    )

    impressions = session.query(ImpressionLog).all()
    clicks = session.query(ClickLog).all()

    assert len(impressions) == 3
    clicked_doc5 = [c for c in clicks if c.doc_id == 5]
    assert len(clicked_doc5) >= 1

    clicked_doc_ids = {c.doc_id for c in clicks}
    shown_doc_ids = {i.doc_id for i in impressions}
    real_negatives = shown_doc_ids - clicked_doc_ids
    assert real_negatives & {6, 7}, "expected at least one shown-not-clicked negative"

    assert result["impressions"] == 3
    assert result["clicks"] == len(clicks)
    assert result["negatives"] >= 1
    assert result["negatives"] == result["impressions"] - result["clicks"]
    assert result["queries_replayed"] == 1
