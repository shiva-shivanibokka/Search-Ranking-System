"""Unit tests for scripts/preprocess.py's corpus + hard-negative mining."""


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
