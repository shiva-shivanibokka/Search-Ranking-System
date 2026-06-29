"""Unit tests for the two-tower model."""

import torch

from training.two_tower_model import EncoderTower, TwoTowerModel


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


def test_encode_query_doc_deterministic_and_unit_norm():
    """Encoders must be deterministic and produce unit vectors.

    Note: the old version asserted "same tokens → higher cosine similarity",
    but the query/doc towers have *separate* weights, so that property only
    holds after training — it makes for a flaky test on a randomly-initialized
    model. Here we test invariants that hold regardless of training: encoding
    is deterministic in eval mode, and embeddings are L2-normalized so every
    cosine similarity is a valid value in [-1, 1].
    """
    model = TwoTowerModel("distilbert-base-uncased", projection_dim=256)
    model.eval()

    q_ids = torch.randint(100, 1000, (1, 16))
    q_mask = torch.ones(1, 16, dtype=torch.long)
    d_ids = torch.randint(100, 1000, (1, 16))
    d_mask = torch.ones(1, 16, dtype=torch.long)

    with torch.no_grad():
        q_emb1 = model.encode_query(q_ids, q_mask)
        q_emb2 = model.encode_query(q_ids, q_mask)
        d_emb = model.encode_doc(d_ids, d_mask)

    # Deterministic in eval mode.
    assert torch.allclose(q_emb1, q_emb2, atol=1e-6)
    # Unit-normalized (so dot product == cosine similarity).
    assert torch.allclose(q_emb1.norm(p=2, dim=-1), torch.ones(1), atol=1e-5)
    assert torch.allclose(d_emb.norm(p=2, dim=-1), torch.ones(1), atol=1e-5)
    # Cosine similarity between unit vectors is always within [-1, 1].
    cos = (q_emb1 * d_emb).sum().item()
    assert -1.0001 <= cos <= 1.0001
