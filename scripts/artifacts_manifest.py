"""
Single source of truth for the model/data artifacts the *serving* stack needs.

Large artifacts (model weights, FAISS index, BM25 index, passages) are NOT
committed to git — they are published to a Hugging Face Hub dataset/model repo
and pulled at boot by ``scripts/bootstrap.py``. This replaces the missing DVC
remote and makes a fresh clone runnable.

Why only a subset of ``models/`` is listed: training produces several two-tower
checkpoints (``model_epoch1.pt`` … ``model_final.pt``, ~513MB each). Serving only
needs ``model_best.pt``. Publishing just the serving set keeps the artifact repo
small and the deploy boot fast.

Env:
  HF_ARTIFACTS_REPO   Hugging Face repo id that stores the artifacts,
                      e.g. "your-username/search-ranking-artifacts".
"""

from __future__ import annotations

import os

# Hugging Face repo that holds the published artifacts. Override via env.
# Format: "<hf-username>/<repo-name>". Create it once with scripts/publish_artifacts.py.
HF_ARTIFACTS_REPO = os.getenv("HF_ARTIFACTS_REPO", "shiva-1993/search-ranking-system")
HF_ARTIFACTS_REVISION = os.getenv("HF_ARTIFACTS_REVISION", "main")

# Repo-relative paths the serving stack must have present locally to start.
# Each is downloaded from HF Hub to the same relative path in the project root.
SERVING_ARTIFACTS: list[str] = [
    # Two-tower query encoder (only the best checkpoint is needed to serve)
    "models/two_tower/model_best.pt",
    "models/two_tower/config.json",
    "models/two_tower/tokenizer.json",
    "models/two_tower/tokenizer_config.json",
    "models/two_tower/vocab.txt",
    "models/two_tower/special_tokens_map.json",
    # LambdaRank reranker
    "models/lambdarank/lambdarank.json",
    "models/lambdarank/feature_names.json",
    # FAISS dense index + id map
    "data/indexes/faiss_ivfpq.index",
    "data/indexes/docid_map.pkl",
    # BM25 sparse index (hybrid retrieval + lambdarank features)
    "data/indexes/bm25_index.pkl",
    "data/indexes/bm25_pid_list.pkl",
    # Passage text + lengths
    "data/processed/passages.parquet",
]

# Optional artifacts — present them if available, but the stack degrades
# gracefully when they are missing (see services that load them).
OPTIONAL_ARTIFACTS: list[str] = [
    "models/cross_encoder/model_best.pt",
    "models/cross_encoder/config.json",
    "models/cross_encoder/tokenizer.json",
    "models/cross_encoder/tokenizer_config.json",
    "models/cross_encoder/vocab.txt",
    "models/cross_encoder/special_tokens_map.json",
    "models/difficulty_classifier/difficulty_classifier.json",
    "models/difficulty_classifier/classifier_meta.json",
]
