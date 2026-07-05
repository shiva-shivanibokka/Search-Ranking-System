"""Unit tests for scripts/preprocess.py's corpus + hard-negative mining."""

import pandas as pd


def test_build_gold_inclusive_corpus_includes_all_gold_and_hits_target(tmp_path):
    from scripts.preprocess import build_gold_inclusive_corpus

    collection_path = tmp_path / "collection.tsv"
    lines = [f"{pid}\tpassage text number {pid}" for pid in range(20)]
    collection_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    gold_pids = {3, 17}
    df = build_gold_inclusive_corpus(
        gold_pids, collection_path=collection_path, target_size=8, seed=42
    )

    assert gold_pids.issubset(set(df["pid"]))
    assert len(df) == 8
    assert len(df["pid"].unique()) == 8
    # Original MS MARCO pids are preserved verbatim, never renumbered.
    assert df[df["pid"] == 3]["text"].iloc[0] == "passage text number 3"


def test_build_gold_inclusive_corpus_never_drops_gold_even_if_target_too_small(tmp_path):
    from scripts.preprocess import build_gold_inclusive_corpus

    collection_path = tmp_path / "collection.tsv"
    lines = [f"{pid}\tpassage text number {pid}" for pid in range(20)]
    collection_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    gold_pids = set(range(10))  # 10 gold pids, but target_size below that
    df = build_gold_inclusive_corpus(
        gold_pids, collection_path=collection_path, target_size=5, seed=42
    )

    assert gold_pids.issubset(set(df["pid"]))
    assert len(df) == 10  # coverage is never sacrificed to hit the size target


def test_mine_hard_negatives_excludes_gold_and_respects_count():
    from scripts.preprocess import build_bm25s_index, mine_hard_negatives

    passages_df = pd.DataFrame(
        {
            "pid": [1, 2, 3, 4, 5, 6],
            "text": [
                "cats are small mammals that purr",
                "dogs are loyal mammals that bark",
                "cats and dogs are common household pets",
                "the stock market rose sharply today",
                "cats love to chase small mice at night",
                "quantum physics describes subatomic particles",
            ],
        }
    )
    queries_df = pd.DataFrame({"qid": [100], "text": ["what mammals are cats"]})
    qrels_df = pd.DataFrame({"qid": [100], "pid": [1], "relevance": [1]})

    retriever = build_bm25s_index(passages_df)
    result = mine_hard_negatives(
        queries_df, qrels_df, passages_df, retriever,
        top_k=5, hard_neg_per_query=2, max_queries=10,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["qid"] == 100
    assert row["pos_pid"] == 1
    hard_negs = row["hard_neg_pids"]
    assert len(hard_negs) == 2
    assert 1 not in hard_negs  # the positive pid must never appear as a hard negative
    assert len(row["hard_neg_texts"]) == 2


def test_mine_hard_negatives_skips_queries_without_qrels():
    from scripts.preprocess import build_bm25s_index, mine_hard_negatives

    passages_df = pd.DataFrame(
        {"pid": [1, 2, 3], "text": ["alpha text", "beta text", "gamma text"]}
    )
    queries_df = pd.DataFrame(
        {"qid": [1, 2], "text": ["alpha query", "no qrels for this one"]}
    )
    qrels_df = pd.DataFrame({"qid": [1], "pid": [1], "relevance": [1]})

    retriever = build_bm25s_index(passages_df)
    result = mine_hard_negatives(
        queries_df, qrels_df, passages_df, retriever,
        top_k=3, hard_neg_per_query=1, max_queries=10,
    )

    assert len(result) == 1
    assert result.iloc[0]["qid"] == 1
