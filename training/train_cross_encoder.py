"""
Fine-tune DistilBERT as a Cross-Encoder reranker on MS MARCO.

The cross-encoder concatenates query + document as a single sequence:
  [CLS] query [SEP] document [SEP]

And outputs a single relevance score via a classification head.
This is slower than the two-tower at query time (can't pre-compute doc embeddings)
but achieves much higher accuracy — used only for reranking top-100 candidates.

Training data: MS MARCO training triples (query, pos, neg)
  - Positive pair: (query, pos_doc) → label 1
  - Negative pair: (query, neg_doc) → label 0
"""

import json
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rich.console import Console
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.training_config import get_training_config

console = Console()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ─────────────────────────────────────────────────────────────────────


class CrossEncoderModel(nn.Module):
    """
    Cross-encoder: DistilBERT backbone + single linear classification head.
    Input: [CLS] query [SEP] document [SEP]
    Output: scalar relevance logit
    """

    def __init__(
        self, model_name: str = "distilbert-base-uncased", dropout: float = 0.1
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Returns relevance logits of shape (batch,)."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_emb).squeeze(-1)

    def predict_score(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Returns sigmoid probabilities for inference."""
        logits = self.forward(input_ids, attention_mask)
        return torch.sigmoid(logits)


def save_cross_encoder(
    model: CrossEncoderModel, tokenizer, save_dir: str, config: dict
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path / "model.pt")
    tokenizer.save_pretrained(save_path)
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def load_cross_encoder(checkpoint_dir: str, device: str = "cuda") -> tuple:
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    model = CrossEncoderModel(model_name=cfg["model_name"])
    state_dict = torch.load(
        os.path.join(checkpoint_dir, "model.pt"), map_location=device
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    return model, tokenizer


# ── Dataset ───────────────────────────────────────────────────────────────────


class CrossEncoderDataset(Dataset):
    """
    Each item: (query, document, label)
    Built from MS MARCO triples: each triple gives 1 positive + 1 negative pair.
    """

    def __init__(
        self,
        triples_path: str,
        tokenizer,
        max_seq_len: int = 256,
        max_samples: int = -1,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        df = pd.read_parquet(triples_path)
        if max_samples > 0:
            df = df.head(max_samples)

        self.pairs = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building CE dataset"):
            self.pairs.append((row["query"], row["pos_text"], 1))
            self.pairs.append((row["query"], row["neg_text"], 0))

        console.print(f"[green]CrossEncoder dataset: {len(self.pairs):,} pairs[/green]")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, doc, label = self.pairs[idx]
        enc = self.tokenizer(
            query,
            doc,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate_ndcg(
    model: CrossEncoderModel,
    tokenizer,
    passages_df: pd.DataFrame,
    dev_queries_df: pd.DataFrame,
    dev_qrels_df: pd.DataFrame,
    bm25_candidates: dict,  # qid → [pid1, pid2, ...]  (top-100 from BM25)
    k: int = 10,
    sample_size: int = 500,
    max_seq_len: int = 256,
    batch_size: int = 32,
) -> float:
    """Compute NDCG@k by reranking BM25 top-100 with the cross-encoder."""
    model.eval()
    pid_to_text = dict(zip(passages_df["pid"], passages_df["text"]))
    pos_pids = dev_qrels_df.groupby("qid")["pid"].apply(set).to_dict()

    dev_sample = dev_queries_df.sample(
        min(sample_size, len(dev_queries_df)), random_state=42
    )
    ndcg_scores = []

    for _, row in tqdm(dev_sample.iterrows(), total=len(dev_sample), desc="NDCG eval"):
        qid = row["qid"]
        query_text = row["text"]
        gold_pids = pos_pids.get(qid, set())
        candidates = bm25_candidates.get(qid, [])[:100]

        if not gold_pids or not candidates:
            continue

        # Score all (query, candidate) pairs
        scores = []
        for i in range(0, len(candidates), batch_size):
            batch_pids = candidates[i : i + batch_size]
            batch_texts = [pid_to_text.get(p, "") for p in batch_pids]
            enc = tokenizer(
                [query_text] * len(batch_texts),
                batch_texts,
                max_length=max_seq_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                s = (
                    model.predict_score(
                        enc["input_ids"].to(DEVICE),
                        enc["attention_mask"].to(DEVICE),
                    )
                    .cpu()
                    .numpy()
                )
            scores.extend(zip(batch_pids, s))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        ranked_pids = [p for p, _ in scores[:k]]

        # NDCG@k
        dcg = sum(
            (1 / np.log2(rank + 2))
            for rank, pid in enumerate(ranked_pids)
            if pid in gold_pids
        )
        ideal_dcg = sum(1 / np.log2(rank + 2) for rank in range(min(len(gold_pids), k)))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


# ── Training ──────────────────────────────────────────────────────────────────


def train(config_path: str = "configs/config.yaml"):
    cfg = get_training_config(config_path)
    ce_cfg = cfg.cross_encoder
    mlf_cfg = cfg.mlflow

    console.print(f"[bold]Training Cross-Encoder on {DEVICE}[/bold]")

    tokenizer = AutoTokenizer.from_pretrained(ce_cfg.model_name)
    model = CrossEncoderModel(model_name=ce_cfg.model_name).to(DEVICE)

    dataset = CrossEncoderDataset(
        "data/processed/train_triples.parquet",
        tokenizer,
        max_seq_len=ce_cfg.max_seq_len,
        max_samples=200000,  # 200K triples = 400K pairs, sufficient for finetuning
    )
    loader = DataLoader(
        dataset,
        batch_size=ce_cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=ce_cfg.learning_rate, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    save_dir = ce_cfg.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
    mlflow.set_experiment(mlf_cfg.experiment_name)

    with mlflow.start_run(run_name="cross_encoder_training"):
        mlflow.log_params(
            {
                "model_name": ce_cfg.model_name,
                "max_seq_len": ce_cfg.max_seq_len,
                "batch_size": ce_cfg.batch_size,
                "gradient_accumulation_steps": ce_cfg.gradient_accumulation_steps,
                "learning_rate": ce_cfg.learning_rate,
                "epochs": ce_cfg.epochs,
            }
        )

        global_step = 0
        accum_steps = ce_cfg.gradient_accumulation_steps

        for epoch in range(ce_cfg.epochs):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{ce_cfg.epochs}")
            for step, batch in enumerate(pbar):
                logits = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                )
                loss = criterion(logits, batch["label"].to(DEVICE))
                loss = loss / accum_steps
                loss.backward()

                if (step + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss.item() * accum_steps
                pbar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}"})

                if global_step % 500 == 0:
                    mlflow.log_metric(
                        "train_loss_ce", loss.item() * accum_steps, step=global_step
                    )

            avg_loss = epoch_loss / len(loader)
            console.print(f"\n[bold]Epoch {epoch + 1} avg loss: {avg_loss:.4f}[/bold]")
            mlflow.log_metric("epoch_avg_loss_ce", avg_loss, step=epoch)

        # Save model
        model_config = {
            "model_name": ce_cfg.model_name,
            "max_seq_len": ce_cfg.max_seq_len,
        }
        save_cross_encoder(model, tokenizer, save_dir, model_config)
        mlflow.log_artifacts(save_dir, artifact_path="cross_encoder_model")

        console.print("\n[bold green]Cross-encoder training complete.[/bold green]")
        console.print(f"Model saved → {save_dir}")
        console.print("Next step: [cyan]python training/train_lambdarank.py[/cyan]")


if __name__ == "__main__":
    train()
