"""Unit tests for the BEIR zero-shot evaluation harness."""

import json

import numpy as np
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


def test_doc_to_text_concatenates_title():
    from training.beir_eval import doc_to_text

    assert doc_to_text({"title": "Cats", "text": "are mammals"}) == "Cats are mammals"
    assert doc_to_text({"title": "", "text": "no title here"}) == "no title here"


def test_adapter_encode_shapes_and_l2_norm():
    from transformers import AutoTokenizer

    from training.beir_eval import TwoTowerBEIRAdapter
    from training.two_tower_model import TwoTowerModel

    model = TwoTowerModel("distilbert-base-uncased", projection_dim=256)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    adapter = TwoTowerBEIRAdapter(model, tokenizer, device="cpu")

    corpus = [
        {"title": "A", "text": "alpha document"},
        {"title": "", "text": "beta document"},
        {"title": "C", "text": "gamma document"},
    ]
    q_emb = adapter.encode_queries(["what is alpha", "beta test"])
    d_emb = adapter.encode_corpus(corpus)

    assert q_emb.shape == (2, 256)
    assert d_emb.shape == (3, 256)
    assert np.allclose(np.linalg.norm(d_emb, axis=1), 1.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(q_emb, axis=1), 1.0, atol=1e-4)
