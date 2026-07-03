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


def test_dense_rank_orders_by_similarity():
    from training.beir_eval import dense_rank

    # Doc embeddings: d0 aligned with query, d2 anti-aligned, d1 orthogonal.
    doc_emb = np.array(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    query_emb = np.array([1.0, 0.0], dtype=np.float32)
    ranked = dense_rank(query_emb, doc_emb, ["d0", "d1", "d2"], top_k=3)
    assert ranked == ["d0", "d1", "d2"]


def test_rrf_fuse_matches_hand_computed():
    from training.beir_eval import rrf_fuse

    # Dense ranks (1-indexed): A=1, B=2, C=3. Sparse ranks: B=1, D=2, A=3.
    # With rrf_k=1, score = 1/(1+rank):
    #   A = 1/2 + 1/4 = 0.75
    #   B = 1/3 + 1/2 = 0.8333...
    #   C = 1/4       = 0.25
    #   D = 1/3       = 0.3333...
    # => order B > A > D > C
    fused = rrf_fuse(["A", "B", "C"], ["B", "D", "A"], rrf_k=1, top_k=10)
    assert fused == ["B", "A", "D", "C"]
