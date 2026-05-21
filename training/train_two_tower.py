"""
Train the Two-Tower Dual Encoder on MS MARCO.

Training strategy:
  - In-batch negatives (standard) + hard negatives (mined by BM25)
  - InfoNCE contrastive loss
  - Linear warmup → cosine decay LR schedule
  - Evaluate Recall@10 and Recall@100 on dev set every epoch
  - All experiments tracked in MLflow
"""

import os
import sys
import json
import math
import pickle
import logging
import warnings
from pathlib import Path

# Suppress HuggingFace connectivity warnings — model is already cached locally
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*huggingface_hub.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from rich.console import Console

sys.path.append(str(Path(__file__).resolve().parents[1]))
from training.two_tower_model import TwoTowerModel
from configs.training_config import get_training_config

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ──────────────────────────────────────────────────────────────────


class TwoTowerDataset(Dataset):
    """
    Each sample: (query, positive_doc, [hard_neg_1, ..., hard_neg_K])
    Loaded from hard_negatives.parquet which was built by preprocess.py.
    Falls back to train_triples.parquet if hard negatives are not available.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_q_len: int = 64,
        max_d_len: int = 180,
        num_hard_neg: int = 5,
    ):
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_d_len = max_d_len
        self.num_hard_neg = num_hard_neg

        hard_neg_path = Path(data_dir) / "hard_negatives.parquet"
        triples_path = Path(data_dir) / "train_triples.parquet"

        if hard_neg_path.exists():
            console.print("[cyan]Loading hard negatives dataset...[/cyan]")
            df = pd.read_parquet(hard_neg_path)
            self.queries = df["query"].tolist()
            self.pos_texts = df["pos_text"].tolist()
            self.hard_neg_texts = df["hard_neg_texts"].tolist()
            self.use_hard_neg = True
        else:
            console.print(
                "[yellow]hard_negatives.parquet not found — falling back to triples[/yellow]"
            )
            df = pd.read_parquet(triples_path)
            self.queries = df["query"].tolist()
            self.pos_texts = df["pos_text"].tolist()
            self.hard_neg_texts = df["neg_text"].apply(lambda x: [x]).tolist()
            self.use_hard_neg = True

        console.print(f"[green]Dataset: {len(self.queries):,} training samples[/green]")

    def __len__(self):
        return len(self.queries)

    def _encode(self, text: str, max_len: int) -> dict:
        return self.tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        q_enc = self._encode(self.queries[idx], self.max_q_len)
        pos_enc = self._encode(self.pos_texts[idx], self.max_d_len)

        hard_negs = self.hard_neg_texts[idx][: self.num_hard_neg]
        # Pad to num_hard_neg if fewer available
        while len(hard_negs) < self.num_hard_neg:
            hard_negs.append(hard_negs[0] if hard_negs else self.pos_texts[idx])

        neg_encs = [self._encode(t, self.max_d_len) for t in hard_negs]

        return {
            "query_input_ids": q_enc["input_ids"].squeeze(0),
            "query_attention_mask": q_enc["attention_mask"].squeeze(0),
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "hard_neg_input_ids": torch.stack(
                [e["input_ids"].squeeze(0) for e in neg_encs]
            ),
            "hard_neg_attention_mask": torch.stack(
                [e["attention_mask"].squeeze(0) for e in neg_encs]
            ),
        }


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_recall(
    model: TwoTowerModel,
    tokenizer,
    passages_df: pd.DataFrame,
    dev_queries_df: pd.DataFrame,
    dev_qrels_df: pd.DataFrame,
    batch_size: int = 256,
    max_seq_len: int = 180,
    k_values: list = [10, 100],
    sample_size: int = 1000,
) -> dict:
    """
    Compute Recall@K by brute-force cosine search over passage embeddings.
    Uses a sample_size subset of dev queries for speed during training.
    Full evaluation runs in evaluate_retrieval.py.
    """
    model.eval()
    dev_sample = dev_queries_df.sample(
        min(sample_size, len(dev_queries_df)), random_state=42
    )
    pos_pids = dev_qrels_df.groupby("qid")["pid"].apply(set).to_dict()

    # Embed all passages
    console.print(f"  [cyan]Embedding {len(passages_df):,} passages for eval...[/cyan]")
    all_doc_embs = []
    pid_list = passages_df["pid"].tolist()

    for i in tqdm(range(0, len(passages_df), batch_size), desc="Doc embeddings"):
        batch_texts = passages_df["text"].iloc[i : i + batch_size].tolist()
        enc = tokenizer(
            batch_texts,
            max_length=max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            emb = model.encode_doc(
                enc["input_ids"].to(DEVICE),
                enc["attention_mask"].to(DEVICE),
            )
        all_doc_embs.append(emb.cpu().numpy())

    doc_matrix = np.vstack(all_doc_embs)  # (N_passages, D)

    # Embed queries and compute recall
    recalls = {k: [] for k in k_values}
    for _, row in tqdm(dev_sample.iterrows(), total=len(dev_sample), desc="Query eval"):
        qid = row["qid"]
        enc = tokenizer(
            row["text"],
            max_length=64,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            q_emb = (
                model.encode_query(
                    enc["input_ids"].to(DEVICE),
                    enc["attention_mask"].to(DEVICE),
                )
                .cpu()
                .numpy()
            )  # (1, D)

        scores = (doc_matrix @ q_emb.T).squeeze(-1)  # (N_passages,)
        gold_pids = pos_pids.get(qid, set())
        if not gold_pids:
            continue

        for k in k_values:
            top_k_indices = scores.argsort()[::-1][:k]
            retrieved_pids = set(pid_list[j] for j in top_k_indices if j >= 0)
            recall = len(gold_pids & retrieved_pids) / len(gold_pids)
            recalls[k].append(recall)

    return {f"Recall@{k}": np.mean(v) for k, v in recalls.items() if v}


# ── LR Scheduler ─────────────────────────────────────────────────────────────


def get_linear_warmup_cosine_decay(
    optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ── Main Training Loop ───────────────────────────────────────────────────────


def train(config_path: str = "configs/config.yaml"):
    cfg = get_training_config(config_path)
    tt_cfg = cfg.two_tower
    mlf_cfg = cfg.mlflow

    console.print(f"[bold]Training Two-Tower on device: {DEVICE}[/bold]")
    console.print(f"  Model: {tt_cfg.model_name}")
    console.print(f"  Projection dim: {tt_cfg.projection_dim}")
    console.print(f"  Batch size: {tt_cfg.batch_size}")
    console.print(f"  Epochs: {tt_cfg.epochs}")

    # ── Setup ──────────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(tt_cfg.model_name)
    model = TwoTowerModel(
        model_name=tt_cfg.model_name,
        embedding_dim=tt_cfg.embedding_dim,
        projection_dim=tt_cfg.projection_dim,
        temperature=tt_cfg.temperature,
    ).to(DEVICE)

    dataset = TwoTowerDataset(
        "data/processed",
        tokenizer,
        max_q_len=tt_cfg.max_seq_len_query,
        max_d_len=tt_cfg.max_seq_len_doc,
        num_hard_neg=tt_cfg.hard_negatives_per_query,
    )
    loader = DataLoader(
        dataset,
        batch_size=tt_cfg.batch_size,
        shuffle=True,
        num_workers=0,  # 0 = main process only (multiprocessing broken on Windows)
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=tt_cfg.learning_rate, weight_decay=0.01)
    total_steps = len(loader) * tt_cfg.epochs
    scheduler = get_linear_warmup_cosine_decay(
        optimizer, tt_cfg.warmup_steps, total_steps
    )

    # Dev data for evaluation
    passages_df = pd.read_parquet("data/processed/passages.parquet")
    dev_queries_df = pd.read_parquet("data/processed/dev_queries.parquet")
    dev_qrels_df = pd.read_parquet("data/processed/dev_qrels.parquet")

    save_dir = Path(tt_cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
    mlflow.set_experiment(mlf_cfg.experiment_name)

    with mlflow.start_run(run_name="two_tower_training"):
        mlflow.log_params(
            {
                "model_name": tt_cfg.model_name,
                "projection_dim": tt_cfg.projection_dim,
                "temperature": tt_cfg.temperature,
                "batch_size": tt_cfg.batch_size,
                "learning_rate": tt_cfg.learning_rate,
                "epochs": tt_cfg.epochs,
                "hard_negatives_per_query": tt_cfg.hard_negatives_per_query,
                "train_samples": len(dataset),
            }
        )

        global_step = 0
        best_recall_at_10 = 0.0

        for epoch in range(tt_cfg.epochs):
            model.train()
            epoch_loss = 0.0

            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{tt_cfg.epochs}")
            for batch in pbar:
                optimizer.zero_grad()

                loss = model(
                    query_input_ids=batch["query_input_ids"].to(DEVICE),
                    query_attention_mask=batch["query_attention_mask"].to(DEVICE),
                    pos_input_ids=batch["pos_input_ids"].to(DEVICE),
                    pos_attention_mask=batch["pos_attention_mask"].to(DEVICE),
                    hard_neg_input_ids=batch["hard_neg_input_ids"].to(DEVICE),
                    hard_neg_attention_mask=batch["hard_neg_attention_mask"].to(DEVICE),
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

                if global_step % 500 == 0:
                    mlflow.log_metric("train_loss", loss.item(), step=global_step)

            avg_loss = epoch_loss / len(loader)
            console.print(f"\n[bold]Epoch {epoch + 1} avg loss: {avg_loss:.4f}[/bold]")
            mlflow.log_metric("epoch_avg_loss", avg_loss, step=epoch)

            # Evaluate on dev sample
            recall_metrics = evaluate_recall(
                model, tokenizer, passages_df, dev_queries_df, dev_qrels_df
            )
            for k, v in recall_metrics.items():
                console.print(f"  {k}: {v:.4f}")
                mlflow.log_metric(k, v, step=epoch)

            # Save best model
            if recall_metrics.get("Recall@10", 0) > best_recall_at_10:
                best_recall_at_10 = recall_metrics["Recall@10"]
                torch.save(model.state_dict(), save_dir / "model_best.pt")
                console.print(
                    f"  [green]New best Recall@10: {best_recall_at_10:.4f} — saved[/green]"
                )

        # Save final model + tokenizer + config
        torch.save(model.state_dict(), save_dir / "model_final.pt")
        tokenizer.save_pretrained(save_dir)

        config_dict = {
            "model_name": tt_cfg.model_name,
            "embedding_dim": tt_cfg.embedding_dim,
            "projection_dim": tt_cfg.projection_dim,
            "temperature": tt_cfg.temperature,
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        mlflow.log_artifacts(str(save_dir), artifact_path="two_tower_model")
        mlflow.log_metric("best_recall_at_10", best_recall_at_10)

        console.print(
            f"\n[bold green]Training complete. Best Recall@10: {best_recall_at_10:.4f}[/bold green]"
        )
        console.print(f"Model saved → {save_dir}")
        console.print("Next step: [cyan]python training/build_faiss_index.py[/cyan]")


if __name__ == "__main__":
    train()
