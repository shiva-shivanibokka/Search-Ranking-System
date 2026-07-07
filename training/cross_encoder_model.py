"""
CrossEncoder model + (de)serialization — the serving-lightweight half of the
cross-encoder, split out so the serving stack (deploy/engine.py) can import
`load_cross_encoder` WITHOUT pulling in training-only deps (mlflow, rich, tqdm)
that train_cross_encoder.py imports. Mirrors training/two_tower_model.py.

Only depends on torch + transformers, both present in the serving image.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class CrossEncoderModel(nn.Module):
    """DistilBERT backbone + a single linear head over the [CLS] token.

    Input: ``[CLS] query [SEP] document [SEP]`` → scalar relevance logit.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns relevance logits of shape (batch,)."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_emb).squeeze(-1)

    def predict_score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities for inference."""
        logits = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits)


def save_cross_encoder(model: CrossEncoderModel, tokenizer, save_dir: str, config: dict) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "model.pt")
    tokenizer.save_pretrained(save_path)
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def load_cross_encoder(checkpoint_dir: str, device: str = "cuda") -> tuple:
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        cfg = json.load(f)
    model = CrossEncoderModel(model_name=cfg["model_name"])
    state_dict = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    return model, tokenizer
