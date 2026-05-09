"""
Two-Tower Dual Encoder Model.

Architecture:
  - Shared DistilBERT backbone for both query and document towers
  - Separate projection heads → 256-dim embedding space
  - Trained with InfoNCE contrastive loss + in-batch negatives + hard negatives
  - At inference: query encoder runs online, doc encoder runs offline to build FAISS index
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Optional


class EncoderTower(nn.Module):
    """
    Single tower: DistilBERT backbone + linear projection head.
    Both query tower and doc tower share this class but have separate weights.
    """

    def __init__(
        self,
        model_name: str,
        embedding_dim: int,
        projection_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size  # 768 for DistilBERT
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns L2-normalized embeddings of shape (batch, projection_dim).
        Mean pooling over token dimension (masked), then project + normalize.
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Mean pooling — average only over real tokens (not padding)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask  # (batch, hidden)

        projected = self.projection(pooled)  # (batch, projection_dim)
        return F.normalize(projected, p=2, dim=-1)  # L2 normalize


class TwoTowerModel(nn.Module):
    """
    Full two-tower model with separate query and document towers.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        embedding_dim: int = 768,
        projection_dim: int = 256,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.temperature = temperature
        self.projection_dim = projection_dim

        self.query_tower = EncoderTower(model_name, embedding_dim, projection_dim)
        self.doc_tower = EncoderTower(model_name, embedding_dim, projection_dim)

    def encode_query(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.query_tower(input_ids, attention_mask)

    def encode_doc(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.doc_tower(input_ids, attention_mask)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        pos_input_ids: torch.Tensor,
        pos_attention_mask: torch.Tensor,
        hard_neg_input_ids: Optional[torch.Tensor] = None,
        hard_neg_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        InfoNCE loss with in-batch negatives + optional hard negatives.

        For a batch of size B:
          - query embeddings: (B, D)
          - positive doc embeddings: (B, D)
          - In-batch negatives: the other B-1 positives serve as negatives for each query
          - Hard negatives (optional): explicitly mined hard negatives appended to negatives

        Returns scalar loss.
        """
        q_emb = self.encode_query(query_input_ids, query_attention_mask)  # (B, D)
        pos_emb = self.encode_doc(pos_input_ids, pos_attention_mask)  # (B, D)

        if hard_neg_input_ids is not None:
            # hard_neg shape: (B * num_hard_neg, seq_len) → encode → (B * num_hard_neg, D)
            bsz = q_emb.size(0)
            hard_neg_emb = self.encode_doc(
                hard_neg_input_ids.view(-1, hard_neg_input_ids.size(-1)),
                hard_neg_attention_mask.view(-1, hard_neg_attention_mask.size(-1)),
            )  # (B * num_hard_neg, D)
            # Concatenate positives + hard negatives as the doc set
            doc_emb = torch.cat([pos_emb, hard_neg_emb], dim=0)  # (B + B*K, D)
        else:
            doc_emb = pos_emb  # in-batch negatives only

        # Similarity matrix: (B, num_docs)
        sim_matrix = torch.matmul(q_emb, doc_emb.T) / self.temperature

        # Labels: each query's positive is at index i (first B columns)
        labels = torch.arange(q_emb.size(0), device=q_emb.device)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


def load_two_tower(checkpoint_dir: str, device: str = "cuda") -> tuple:
    """Load a trained TwoTowerModel from checkpoint directory."""
    import os
    import json

    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    model = TwoTowerModel(
        model_name=cfg["model_name"],
        embedding_dim=cfg["embedding_dim"],
        projection_dim=cfg["projection_dim"],
        temperature=cfg.get("temperature", 0.05),
    )
    state_dict = torch.load(
        os.path.join(checkpoint_dir, "model.pt"), map_location=device
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    return model, tokenizer
