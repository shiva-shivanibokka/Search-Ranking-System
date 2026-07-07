"""
Shared LambdaRank feature builder — the single source of truth for the
7-feature vector fed to the LambdaRank XGBoost model.

This exists to close a train/serve skew: before this module, the same
formula was hand-copied into services/ranking/main.py and deploy/engine.py
(and needs to be reused by the offline retrain/simulation scripts). Any drift
between those copies would silently poison the ranker. Now every caller
(serve paths and future training paths) builds features by calling
`build_lambdarank_features` here.

Note the "bm25_rank" feature is a deliberately preserved quirk: it is named
bm25_rank but actually holds the candidate's *retrieval* rank (whatever
upstream fusion produced), not a BM25-only rank. This was true of the
original serve-time code and is preserved verbatim so train == serve.

KNOWN RESIDUAL SKEW (honest limitation): this module unifies the feature
*formula*, but the numeric `score` a caller passes in as `two_tower_cosine_sim`
still differs by path. At serve time (deploy/engine.py, services/ranking/main.py)
`score` is the RRF-*fused* score (~1/(k+rank) scale). In the offline
retrain path (scripts/retrain_from_clicks.py) it is the raw two-tower dot
product. Likewise `retrieval_rank` is the fusion rank at serve time but the
logged display rank at retrain time. So "train == serve" holds for the
*computation* but not yet for the *input distribution* of these two features.
This is pre-existing (the old hand-copied code had the same gap) and does not
crash; it can degrade retrained-ranker quality. Fully closing it means having
the retrain path reconstruct the fused score/rank — tracked as a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_NAMES = [
    "bm25_score",
    "two_tower_cosine_sim",
    "doc_length",
    "query_term_overlap",
    "query_length",
    "bm25_rank",
    "two_tower_rank",
]


@dataclass
class Candidate:
    doc_id: int
    text: str
    score: float
    retrieval_rank: int


def build_lambdarank_features(
    query: str,
    candidates: list[Candidate],
    bm25,
    bm25_pid_list: list[int],
    pid_to_len: dict[int, int],
    bm25_scores_all=None,
    bm25_idx: dict[int, int] | None = None,
) -> np.ndarray:
    """Build the (n, 7) LambdaRank feature matrix for a query and its candidates.

    Feature order matches FEATURE_NAMES exactly. See module docstring for the
    "bm25_rank holds retrieval_rank" quirk.

    ``bm25_scores_all`` / ``bm25_idx`` are optional precomputed inputs: pass them
    to avoid a second full ~1M-doc BM25 scan and a 1M-entry dict rebuild per query
    when the caller already has them (e.g. the retrieval stage just scored BM25).
    When omitted they are computed here (backward compatible).
    """
    if not candidates:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    if bm25_scores_all is None:
        bm25_scores_all = bm25.get_scores(query.lower().split())
    if bm25_idx is None:
        bm25_idx = {pid: i for i, pid in enumerate(bm25_pid_list)}

    q_terms = set(query.lower().split())
    q_len = len(query.split())
    n = len(candidates)

    tt_scores = np.array([c.score for c in candidates])
    order = np.argsort(tt_scores)[::-1]
    tt_ranks = np.empty_like(order)
    tt_ranks[order] = np.arange(1, n + 1)

    X = []
    for i, cand in enumerate(candidates):
        # A candidate absent from the BM25 index gets a real 0.0 (not doc-0's
        # score) — a missing passage must not inherit an arbitrary ranking signal.
        pos = bm25_idx.get(cand.doc_id)
        bm25_score = float(bm25_scores_all[pos]) if pos is not None else 0.0
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

    return np.asarray(X, dtype=np.float32)
