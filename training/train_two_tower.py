"""
Train the Two-Tower Dual Encoder on MS MARCO.

Training strategy:
  - In-batch negatives (standard) + randomly sampled negatives per query
    (see scripts/preprocess.py::mine_hard_negatives — despite the module's
    naming, negatives are sampled randomly, not BM25-mined)
  - InfoNCE contrastive loss
  - Linear warmup -> cosine decay LR schedule
  - Checkpoint selection is by per-epoch Recall@10 on a small, fixed eval
    index (all dev qrels gold passages + a capped random distractor sample —
    see build_eval_corpus), not by training loss. Re-embedding the full ~1M
    passage collection every epoch would be too slow; the small eval index
    makes per-epoch recall evaluation cheap. Full Recall@10 / Recall@100
    over the complete collection are still measured separately, after
    training, by scripts/eval_recall.py against the committed FAISS index.
  - All experiments tracked in MLflow
"""

import json
import logging
import math
import os
import sys
import warnings
from pathlib import Path

# Suppress HuggingFace connectivity warnings — model is already cached locally
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*huggingface_hub.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.training_config import get_training_config
from training.two_tower_model import TwoTowerModel

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
        self.num_hard_neg = num_hard_neg

        hard_neg_path = Path(data_dir) / "hard_negatives.parquet"
        triples_path = Path(data_dir) / "train_triples.parquet"

        if hard_neg_path.exists():
            console.print("[cyan]Loading hard negatives dataset...[/cyan]")
            df = pd.read_parquet(hard_neg_path)
            queries = df["query"].tolist()
            pos_texts = df["pos_text"].tolist()
            hard_neg_texts = df["hard_neg_texts"].tolist()
        else:
            console.print(
                "[yellow]hard_negatives.parquet not found — falling back to triples[/yellow]"
            )
            df = pd.read_parquet(triples_path)
            queries = df["query"].tolist()
            pos_texts = df["pos_text"].tolist()
            hard_neg_texts = df["neg_text"].apply(lambda x: [x]).tolist()

        console.print(f"[green]Dataset: {len(queries):,} training samples[/green]")

        # ── Pre-tokenize everything once at load time ─────────────────────────
        # This moves all CPU tokenization out of the training loop entirely.
        # Takes ~2-3 min at startup but reduces per-batch time from ~9s to ~0.1s.
        console.print("[cyan]Pre-tokenizing queries...[/cyan]")
        q_enc = tokenizer(
            queries,
            max_length=max_q_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.query_input_ids = q_enc["input_ids"]
        self.query_attention_mask = q_enc["attention_mask"]

        console.print("[cyan]Pre-tokenizing positive docs...[/cyan]")
        pos_enc = tokenizer(
            pos_texts,
            max_length=max_d_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.pos_input_ids = pos_enc["input_ids"]
        self.pos_attention_mask = pos_enc["attention_mask"]

        console.print("[cyan]Pre-tokenizing hard negatives...[/cyan]")
        # Flatten all hard negs, tokenize in one shot, reshape
        flat_negs = []
        for negs in hard_neg_texts:
            padded = (negs + [negs[0]] * num_hard_neg)[:num_hard_neg]
            flat_negs.extend(padded)

        neg_enc = tokenizer(
            flat_negs,
            max_length=max_d_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        n = len(queries)
        self.neg_input_ids = neg_enc["input_ids"].view(n, num_hard_neg, -1)
        self.neg_attention_mask = neg_enc["attention_mask"].view(n, num_hard_neg, -1)

        console.print(
            "[green]Pre-tokenization complete — training will be fast.[/green]"
        )

    def __len__(self):
        return self.query_input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "query_input_ids": self.query_input_ids[idx],
            "query_attention_mask": self.query_attention_mask[idx],
            "pos_input_ids": self.pos_input_ids[idx],
            "pos_attention_mask": self.pos_attention_mask[idx],
            "hard_neg_input_ids": self.neg_input_ids[idx],
            "hard_neg_attention_mask": self.neg_attention_mask[idx],
        }


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_recall(
    model: TwoTowerModel,
    tokenizer,
    passages_df: pd.DataFrame,
    dev_queries_df: pd.DataFrame,
    dev_qrels_df: pd.DataFrame,
    batch_size: int = 128,
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
    recalls: dict = {k: [] for k in k_values}
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

    # Use underscore instead of @ — MLflow rejects @ in metric names
    return {f"Recall_at_{k}": np.mean(v) for k, v in recalls.items() if v}


def build_eval_corpus(
    passages_df: pd.DataFrame,
    dev_qrels_df: pd.DataFrame,
    max_distractors: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Small, fixed eval corpus for per-epoch recall checkpointing: every dev
    qrels gold passage, plus a capped random distractor sample. Built ONCE
    before training starts (the doc tower changes every epoch, so re-embedding
    the full ~1M corpus per epoch would be too slow — re-embedding this small,
    fixed corpus is cheap, ~1-3 min per epoch on an RTX 4060).
    """
    gold_pids = set(dev_qrels_df["pid"])
    gold_df = passages_df[passages_df["pid"].isin(gold_pids)]
    distractor_pool = passages_df[~passages_df["pid"].isin(gold_pids)]
    n = min(max_distractors, len(distractor_pool))
    distractor_df = distractor_pool.sample(n=n, random_state=seed)
    return pd.concat([gold_df, distractor_df], ignore_index=True)


def select_best_epoch(recall_at_10_by_epoch: dict) -> int:
    """
    Return the epoch (1-indexed) with the highest eval Recall@10. Ties break
    to the earliest epoch (a smaller/earlier-converged model, all else equal).
    """
    return max(recall_at_10_by_epoch, key=lambda e: (recall_at_10_by_epoch[e], -e))


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
        # Honest training-loss tracking (kept for logging/debugging), but
        # checkpoint SELECTION is now by Recall@10 on the small, fixed eval
        # index built below — not by training loss. Re-embedding the full
        # ~1M corpus every epoch would be too slow, which is exactly why the
        # original code skipped mid-training eval; the small eval index makes
        # per-epoch recall evaluation cheap (~1-3 min/epoch).
        best_train_loss = 0.0
        recall_history: dict = {}
        # Mixed precision (AMP): ~2x faster + lower VRAM on CUDA; no-op on CPU.
        use_amp = DEVICE.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        eval_corpus_df = build_eval_corpus(
            passages_df, dev_qrels_df, max_distractors=tt_cfg.eval_max_distractors
        )
        console.print(
            f"[cyan]Recall-checkpoint eval index: {len(eval_corpus_df):,} passages "
            f"(all dev gold + up to {tt_cfg.eval_max_distractors:,} distractors)[/cyan]"
        )

        for epoch in range(tt_cfg.epochs):
            model.train()
            epoch_loss = 0.0

            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{tt_cfg.epochs}")
            for batch in pbar:
                optimizer.zero_grad()

                with torch.autocast("cuda", enabled=use_amp):
                    loss = model(
                        query_input_ids=batch["query_input_ids"].to(DEVICE),
                        query_attention_mask=batch["query_attention_mask"].to(DEVICE),
                        pos_input_ids=batch["pos_input_ids"].to(DEVICE),
                        pos_attention_mask=batch["pos_attention_mask"].to(DEVICE),
                        hard_neg_input_ids=batch["hard_neg_input_ids"].to(DEVICE),
                        hard_neg_attention_mask=batch["hard_neg_attention_mask"].to(DEVICE),
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
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
            if best_train_loss == 0.0 or avg_loss < best_train_loss:
                best_train_loss = avg_loss

            torch.save(model.state_dict(), save_dir / f"model_epoch{epoch + 1}.pt")
            console.print(
                f"  [green]Checkpoint saved -> model_epoch{epoch + 1}.pt[/green]"
            )

            recall_metrics = evaluate_recall(
                model, tokenizer, eval_corpus_df, dev_queries_df, dev_qrels_df,
                k_values=[10, 100],
            )
            epoch_recall_at_10 = recall_metrics.get("Recall_at_10", 0.0)
            recall_history[epoch + 1] = epoch_recall_at_10
            mlflow.log_metric("eval_recall_at_10", epoch_recall_at_10, step=epoch)
            mlflow.log_metric(
                "eval_recall_at_100", recall_metrics.get("Recall_at_100", 0.0), step=epoch
            )
            console.print(
                f"  [bold]Epoch {epoch + 1} eval Recall@10 (small index): "
                f"{epoch_recall_at_10:.4f}[/bold]"
            )

            if select_best_epoch(recall_history) == epoch + 1:
                torch.save(model.state_dict(), save_dir / "model_best.pt")
                console.print(
                    f"  [green]New best Recall@10: {epoch_recall_at_10:.4f} "
                    f"— saved as model_best.pt[/green]"
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
        # Training-loss metric kept for honesty/debugging — NOT what selects
        # model_best.pt. Selection is by eval_recall_at_10 (see epoch loop).
        mlflow.log_metric("best_train_loss", best_train_loss)
        best_epoch = select_best_epoch(recall_history) if recall_history else None
        best_recall_at_10 = recall_history.get(best_epoch, 0.0) if best_epoch else 0.0
        mlflow.log_metric("best_recall_at_10", best_recall_at_10)

        console.print(
            f"\n[bold green]Training complete. Best train loss: {best_train_loss:.4f} | "
            f"Best eval Recall@10: {best_recall_at_10:.4f} (epoch {best_epoch})[/bold green]"
        )
        console.print(f"Model saved -> {save_dir}")
        console.print("Next step: [cyan]python training/build_faiss_index.py[/cyan]")


if __name__ == "__main__":
    train()
