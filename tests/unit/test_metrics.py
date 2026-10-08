"""IR metric definitions, pinned against hand-computed values.

MAP@10 in particular: the divisor used to be ``min(|gold|, k)``, which is not the
convention published BEIR numbers use. See ``ap_at_k``'s docstring.
"""

from __future__ import annotations

import pytest

from training.evaluate import (
    ap_at_k,
    compute_metrics,
    mlflow_metric_name,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_ap_divides_by_total_relevant_not_min_with_k():
    """10 retrieved, 4 relevant, all 4 at ranks 1-4, but 20 relevant exist.

    precision_sum = 1/1 + 2/2 + 3/3 + 4/4 = 4.0
    BEIR / pytrec_eval convention: 4.0 / 20 = 0.20
    the old convention:            4.0 / min(20, 10) = 0.40
    """
    ranked = list(range(10))
    gold = set(range(4)) | set(range(100, 116))  # 4 retrieved + 16 unretrieved = 20
    assert len(gold) == 20
    assert ap_at_k(ranked, gold, 10) == pytest.approx(0.20)


def test_ap_is_unaffected_when_there_is_one_relevant_doc():
    """The two conventions agree when |gold| <= 1, which is why MS MARCO dev and
    SciFact barely moved while NFCorpus halved."""
    assert ap_at_k([5, 1, 2], {5}, 10) == pytest.approx(1.0)
    assert ap_at_k([1, 5, 2], {5}, 10) == pytest.approx(0.5)


def test_ap_cannot_reach_one_when_relevant_exceeds_k():
    """Retrieving 10 of 38 relevant documents is not a perfect result."""
    gold = set(range(38))
    assert ap_at_k(list(range(10)), gold, 10) < 1.0


def test_ap_empty_gold_is_zero():
    assert ap_at_k([1, 2, 3], set(), 10) == 0.0


def test_recall_and_mrr_and_ndcg_on_a_hand_checked_case():
    ranked = [10, 11, 12, 13]
    gold = {11, 13}
    assert recall_at_k(ranked, gold, 10) == pytest.approx(1.0)
    assert mrr_at_k(ranked, gold, 10) == pytest.approx(0.5)  # first hit at rank 2
    # DCG = 1/log2(3) + 1/log2(5); ideal = 1/log2(2) + 1/log2(3)
    expected = (1 / 1.5849625007211562 + 1 / 2.321928094887362) / (
        1 / 1.0 + 1 / 1.5849625007211562
    )
    assert ndcg_at_k(ranked, gold, 10) == pytest.approx(expected)


def test_compute_metrics_reports_the_expected_keys():
    m = compute_metrics([1, 2, 3], {2})
    assert set(m) == {"NDCG@10", "MAP@10", "MRR@10", "Recall@10", "Recall@100"}


# ---------------------------------------------------------------------------
# MLflow metric names
# ---------------------------------------------------------------------------


def test_mlflow_metric_names_are_accepted_by_mlflow():
    """Every name this evaluation can produce must be loggable.

    `mlflow.log_metric("Hybrid(RRF)/NDCG@10", ...)` raised on the first call --
    both "@" and parentheses are rejected -- so the documented "All metrics logged
    to MLflow" never happened and run_evaluation() exited non-zero after having
    already written eval_results.json.
    """
    from mlflow.utils.validation import _validate_metric_name

    configs = [
        "BM25",
        "TwoTower",
        "Hybrid(RRF)",
        "Hybrid(RRF)+LambdaRank",
        "Hybrid(RRF)+CrossEncoder",
        "TwoTower+LambdaRank",
        "TwoTower+CrossEncoder",
    ]
    for config in configs:
        for metric in compute_metrics([1], {1}):
            _validate_metric_name(mlflow_metric_name(config, metric))


def test_mlflow_metric_name_keeps_the_cutoff_readable():
    assert mlflow_metric_name("BM25", "NDCG@10") == "BM25/NDCG_at_10"
    assert mlflow_metric_name("Hybrid(RRF)", "MAP@10") == "Hybrid_RRF_/MAP_at_10"
