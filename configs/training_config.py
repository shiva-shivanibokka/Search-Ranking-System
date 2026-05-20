"""
Typed configuration dataclasses for training jobs.
Loaded from config.yaml and validated at startup.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import os


@dataclass
class TwoTowerConfig:
    model_name: str = "distilbert-base-uncased"
    embedding_dim: int = 768
    projection_dim: int = 256
    temperature: float = 0.05
    hard_negatives_per_query: int = 5
    batch_size: int = 64
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_steps: int = 1000
    max_seq_len_query: int = 64
    max_seq_len_doc: int = 180
    save_dir: str = "models/two_tower"


@dataclass
class CrossEncoderConfig:
    model_name: str = "distilbert-base-uncased"
    max_seq_len: int = 256
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    epochs: int = 2
    save_dir: str = "models/cross_encoder"
    top_k_rerank: int = 100


@dataclass
class LambdaRankConfig:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    ndcg_k: int = 10
    save_dir: str = "models/lambdarank"
    features: List[str] = field(
        default_factory=lambda: [
            "bm25_score",
            "two_tower_cosine_sim",
            "doc_length",
            "query_term_overlap",
            "query_length",
            "bm25_rank",
            "two_tower_rank",
        ]
    )


@dataclass
class FAISSConfig:
    index_type: str = "IVF1024,PQ32"
    nprobe: int = 64
    embedding_dim: int = 256
    index_path: str = "data/indexes/faiss_ivfpq.index"
    docid_map_path: str = "data/indexes/docid_map.pkl"


@dataclass
class BM25Config:
    k1: float = 0.9
    b: float = 0.4
    index_path: str = "data/indexes/bm25_index.pkl"


@dataclass
class MLflowConfig:
    tracking_uri: str = "http://localhost:5001"
    experiment_name: str = "neural-search-ranking"


@dataclass
class EvaluationConfig:
    recall_at_k: List[int] = field(default_factory=lambda: [10, 100])
    ndcg_at_k: List[int] = field(default_factory=lambda: [10])
    map_at_k: List[int] = field(default_factory=lambda: [10])
    mrr_at_k: List[int] = field(default_factory=lambda: [10])
    dev_queries: int = 6980


@dataclass
class TrainingConfig:
    two_tower: TwoTowerConfig = field(default_factory=TwoTowerConfig)
    cross_encoder: CrossEncoderConfig = field(default_factory=CrossEncoderConfig)
    lambdarank: LambdaRankConfig = field(default_factory=LambdaRankConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_training_config(config_path: str = "configs/config.yaml") -> TrainingConfig:
    raw = load_config(config_path)
    cfg = TrainingConfig()

    def _filter(cls, d: dict) -> dict:
        valid = {f.name for f in dataclasses.fields(cls)}
        return {k: v for k, v in d.items() if k in valid}

    tt = raw.get("two_tower", {})
    cfg.two_tower = TwoTowerConfig(**_filter(TwoTowerConfig, tt))

    ce = raw.get("cross_encoder", {})
    cfg.cross_encoder = CrossEncoderConfig(**_filter(CrossEncoderConfig, ce))

    lr = raw.get("lambdarank", {})
    cfg.lambdarank = LambdaRankConfig(**_filter(LambdaRankConfig, lr))

    fa = raw.get("faiss", {})
    cfg.faiss = FAISSConfig(**_filter(FAISSConfig, fa))

    bm = raw.get("bm25", {})
    cfg.bm25 = BM25Config(**_filter(BM25Config, bm))

    ml = raw.get("mlflow", {})
    cfg.mlflow = MLflowConfig(**_filter(MLflowConfig, ml))

    ev = raw.get("evaluation", {})
    cfg.evaluation = EvaluationConfig(**_filter(EvaluationConfig, ev))

    return cfg
