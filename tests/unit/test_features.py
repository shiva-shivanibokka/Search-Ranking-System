"""Unit tests for the shared LambdaRank feature builder (services/shared/features.py).

These tests anchor correctness against hand-computed values so any future
change to the formula (or to the serve wrappers that call it) is caught.
"""

import numpy as np


class _FakeBM25:
    """Fake bm25 index: ignores tokens, always returns a fixed score array."""

    def get_scores(self, tokens):
        return np.array([2.0, 0.5])


def test_builder_matches_hand_computed():
    from services.shared.features import Candidate, build_lambdarank_features

    bm25 = _FakeBM25()
    bm25_pid_list = [10, 20]
    pid_to_len = {10: 400, 20: 50}

    candidates = [
        Candidate(doc_id=10, text="machine learning models", score=0.9, retrieval_rank=1),
        Candidate(doc_id=20, text="cooking recipes", score=0.7, retrieval_rank=2),
    ]

    out = build_lambdarank_features(
        "machine learning", candidates, bm25, bm25_pid_list, pid_to_len
    )

    # Hand-computed per the spec:
    # doc 10: bm25=2.0 (idx0), tt_score=0.9, doc_len=min(400/200,5)=2.0,
    #         overlap=|{"machine","learning"} & {"machine","learning","models"}|/2=1.0,
    #         q_len=min(2/20,3)=0.1, retrieval_rank/n=1/2=0.5, tt_rank/n: order=[0,1]
    #         (0.9>0.7) -> tt_ranks=[1,2] -> tt_ranks[0]/2=0.5
    # doc 20: bm25=0.5 (idx1), tt_score=0.7, doc_len=min(50/200,5)=0.25,
    #         overlap=|{} |/2=0.0, q_len=0.1, retrieval_rank/n=2/2=1.0,
    #         tt_ranks[1]/2=2/2=1.0
    expected = np.array(
        [
            [2.0, 0.9, 2.0, 1.0, 0.1, 0.5, 0.5],
            [0.5, 0.7, 0.25, 0.0, 0.1, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    assert out.shape == (2, 7)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_serve_matrix_equals_shared_builder():
    import services.ranking.main as rk
    from services.shared.features import Candidate, build_lambdarank_features

    fake_bm25 = _FakeBM25()
    rk.bm25 = fake_bm25
    rk.bm25_pid_list = [10, 20]
    rk.pid_to_len = {10: 400, 20: 50}

    cands = [
        rk.CandidateIn(doc_id=10, text="machine learning models", score=0.9, retrieval_rank=1),
        rk.CandidateIn(doc_id=20, text="cooking recipes", score=0.7, retrieval_rank=2),
    ]

    serve_out = rk._feature_matrix("machine learning", cands)
    shared_out = build_lambdarank_features(
        "machine learning",
        [
            Candidate(10, "machine learning models", 0.9, 1),
            Candidate(20, "cooking recipes", 0.7, 2),
        ],
        fake_bm25,
        [10, 20],
        {10: 400, 20: 50},
    )

    np.testing.assert_allclose(serve_out, shared_out, rtol=1e-6)
