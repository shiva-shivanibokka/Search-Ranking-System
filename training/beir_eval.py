"""
BEIR zero-shot evaluation library.

Reuses the repo's trained two-tower encoder and the exact RRF fusion formula
from deploy/engine.py to measure out-of-domain generalization on small BEIR
datasets (SciFact, NFCorpus, FiQA). CPU-only: the corpora are small enough for
brute-force dense search, so no FAISS index is built.

Metric computation is delegated to training.evaluate.compute_metrics so the BEIR
numbers use the identical NDCG@10 / Recall@100 / MAP@10 / MRR@10 code path as the
in-domain MS MARCO evaluation.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch


def doc_to_text(doc: dict) -> str:
    """BEIR corpus docs are {'title': str, 'text': str}; concatenate them."""
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


class TwoTowerBEIRAdapter:
    """Wraps the repo's two-tower encoder in BEIR's encode_queries/encode_corpus
    interface, returning L2-normalized numpy embeddings."""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cpu",
        max_q_len: int = 64,
        max_d_len: int = 180,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_q_len = max_q_len
        self.max_d_len = max_d_len

    def _encode(
        self, texts: List[str], encode_fn, max_len: int, batch_size: int
    ) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                max_length=max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                emb = (
                    encode_fn(
                        enc["input_ids"].to(self.device),
                        enc["attention_mask"].to(self.device),
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            out.append(emb)
        return np.vstack(out)

    def encode_queries(
        self, queries: List[str], batch_size: int = 32
    ) -> np.ndarray:
        return self._encode(
            queries, self.model.encode_query, self.max_q_len, batch_size
        )

    def encode_corpus(
        self, corpus: List[dict], batch_size: int = 32
    ) -> np.ndarray:
        texts = [doc_to_text(d) for d in corpus]
        return self._encode(texts, self.model.encode_doc, self.max_d_len, batch_size)
