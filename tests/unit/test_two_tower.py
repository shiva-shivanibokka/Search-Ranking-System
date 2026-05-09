"""Unit tests for the two-tower model."""

import pytest
import torch
from training.two_tower_model import TwoTowerModel, EncoderTower


def test_encoder_tower_output_shape():
    tower = EncoderTower(
        "distilbert-base-uncased", embedding_dim=768, projection_dim=256
    )
    input_ids = torch.randint(0, 1000, (4, 32))
    attention_mask = torch.ones(4, 32, dtype=torch.long)
    out = tower(input_ids, attention_mask)
    assert out.shape == (4, 256), f"Expected (4,256), got {out.shape}"


def test_encoder_tower_l2_normalized():
    tower = EncoderTower(
        "distilbert-base-uncased", embedding_dim=768, projection_dim=256
    )
    input_ids = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones(2, 32, dtype=torch.long)
    out = tower(input_ids, attention_mask)
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5), (
        "Embeddings should be L2 normalized"
    )


def test_two_tower_forward_returns_scalar_loss():
    model = TwoTowerModel(
        model_name="distilbert-base-uncased",
        projection_dim=256,
        temperature=0.05,
    )
    bsz, seq_q, seq_d = 4, 16, 32

    q_ids = torch.randint(0, 1000, (bsz, seq_q))
    q_mask = torch.ones(bsz, seq_q, dtype=torch.long)
    p_ids = torch.randint(0, 1000, (bsz, seq_d))
    p_mask = torch.ones(bsz, seq_d, dtype=torch.long)

    loss = model(q_ids, q_mask, p_ids, p_mask)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"


def test_two_tower_with_hard_negatives():
    model = TwoTowerModel(
        model_name="distilbert-base-uncased",
        projection_dim=256,
    )
    bsz, k, seq = 2, 3, 16

    q_ids = torch.randint(0, 1000, (bsz, seq))
    q_mask = torch.ones(bsz, seq, dtype=torch.long)
    p_ids = torch.randint(0, 1000, (bsz, seq))
    p_mask = torch.ones(bsz, seq, dtype=torch.long)
    hn_ids = torch.randint(0, 1000, (bsz, k, seq))
    hn_mask = torch.ones(bsz, k, seq, dtype=torch.long)

    loss = model(q_ids, q_mask, p_ids, p_mask, hn_ids, hn_mask)
    assert loss.ndim == 0 and loss.item() > 0


def test_encode_query_encode_doc_cosine():
    """Query and its positive doc should have higher cosine sim than random doc."""
    model = TwoTowerModel("distilbert-base-uncased", projection_dim=256)
    model.eval()

    q_ids = torch.randint(100, 1000, (1, 16))
    q_mask = torch.ones(1, 16, dtype=torch.long)

    pos_ids = q_ids.clone()  # same tokens → should be high similarity
    pos_mask = q_mask.clone()

    neg_ids = torch.randint(100, 1000, (1, 16))
    neg_mask = torch.ones(1, 16, dtype=torch.long)

    with torch.no_grad():
        q_emb = model.encode_query(q_ids, q_mask)
        pos_emb = model.encode_doc(pos_ids, pos_mask)
        neg_emb = model.encode_doc(neg_ids, neg_mask)

    pos_sim = (q_emb * pos_emb).sum().item()
    neg_sim = (q_emb * neg_emb).sum().item()
    assert pos_sim > neg_sim, (
        f"Positive sim ({pos_sim:.4f}) should > negative sim ({neg_sim:.4f})"
    )
