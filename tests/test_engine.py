"""
Unit tests for the consolidated engine's fusion + feature math, without loading
the 1.2GB artifacts. We instantiate SearchEngine via __new__ (skipping the heavy
__init__) and set only the attributes the method under test touches, and we test
the shared LambdaRank feature builder directly with a tiny fake BM25.
"""

from __future__ import annotations

import numpy as np


def test_rrf_prefers_agreement_and_ranks_1_indexed():
    from deploy.engine import SearchEngine

    eng = SearchEngine.__new__(SearchEngine)  # no __init__ / no model load
    eng.rrf_k = 60
    dense = [{"pid": 1, "score": 0.9, "rank": 1}, {"pid": 2, "score": 0.8, "rank": 2}]
    sparse = [{"pid": 2, "score": 5.0, "rank": 1}, {"pid": 3, "score": 4.0, "rank": 2}]

    fused = eng._rrf(dense, sparse, top_k=10)
    pids = [f["pid"] for f in fused]

    # pid 2 appears in BOTH lists -> highest fused score -> rank 1
    assert pids[0] == 2
    assert set(pids) == {1, 2, 3}
    assert [f["rank"] for f in fused] == [1, 2, 3]
    # fused scores strictly descending
    scores = [f["score"] for f in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_respects_top_k():
    from deploy.engine import SearchEngine

    eng = SearchEngine.__new__(SearchEngine)
    eng.rrf_k = 60
    dense = [{"pid": i, "score": 1.0, "rank": i} for i in range(1, 6)]
    sparse = [{"pid": i + 10, "score": 1.0, "rank": i} for i in range(1, 6)]
    fused = eng._rrf(dense, sparse, top_k=3)
    assert len(fused) == 3


class _FakeBM25:
    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype=np.float32)
        self.calls = 0

    def get_scores(self, tokens):
        self.calls += 1
        return self._scores


def test_features_shape_and_missing_pid_is_zero():
    from services.shared.features import Candidate, build_lambdarank_features

    bm25 = _FakeBM25([10.0, 20.0, 30.0])
    bm25_pid_list = [100, 200, 300]
    pid_to_len = {100: 50, 200: 60}
    cands = [
        Candidate(doc_id=200, text="alpha beta", score=0.5, retrieval_rank=1),
        Candidate(doc_id=999, text="gamma", score=0.4, retrieval_rank=2),  # not in index
    ]
    X = build_lambdarank_features("alpha", cands, bm25, bm25_pid_list, pid_to_len)

    assert X.shape == (2, 7)
    assert X[0, 0] == 20.0  # pid 200 -> its real BM25 score
    assert X[1, 0] == 0.0  # pid 999 missing -> 0.0 (NOT doc-0's 10.0)


def test_features_precomputed_scores_skip_rescan():
    from services.shared.features import Candidate, build_lambdarank_features

    bm25 = _FakeBM25([1.0, 2.0])
    cands = [Candidate(doc_id=0, text="a", score=0.1, retrieval_rank=1)]
    build_lambdarank_features(
        "a",
        cands,
        bm25,
        [0, 1],
        {0: 10},
        bm25_scores_all=np.array([7.0, 8.0]),
        bm25_idx={0: 0, 1: 1},
    )
    # get_scores must NOT be called when the vector is supplied
    assert bm25.calls == 0
