"""Unit tests for the BEIR zero-shot evaluation harness."""

import json

import pytest


def test_load_beir_dataset_parses_corpus_queries_qrels(tmp_path):
    pytest.importorskip("beir")
    from scripts.download_beir import load_beir_dataset

    (tmp_path / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": "T1", "text": "body one"})
        + "\n"
        + json.dumps({"_id": "d2", "title": "", "text": "body two"})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "question one"}) + "\n",
        encoding="utf-8",
    )
    qrels_dir = tmp_path / "qrels"
    qrels_dir.mkdir()
    (qrels_dir / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8"
    )

    corpus, queries, qrels = load_beir_dataset(str(tmp_path), split="test")

    assert corpus["d1"]["text"] == "body one"
    assert queries["q1"] == "question one"
    assert qrels["q1"]["d1"] == 1
