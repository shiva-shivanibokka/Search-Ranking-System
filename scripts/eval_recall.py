"""
Measure REAL Recall@10 / Recall@100 for the deployed two-tower model.

This script exists because training/train_two_tower.py does NOT run recall
evaluation during training (it tracks/selects checkpoints by training loss —
see best_train_loss in that file). This script produces the actual,
measured retrieval quality of models/two_tower/model_best.pt against the
FAISS index that is committed to the repo, so the query encoder and the
document index are guaranteed to match.

Usage:
    python scripts/eval_recall.py

Outputs:
    - Console summary (Recall@10, Recall@100, device used)
    - data/processed/two_tower_recall.json
"""

import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.training_config import get_training_config
from training.evaluate import recall_at_k
from training.two_tower_model import load_two_tower

console = Console()


def main(config_path: str = "configs/config.yaml", batch_size: int = 64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold]Evaluating two-tower recall on device: {device}[/bold]")

    cfg = get_training_config(config_path)

    console.print("[cyan]Loading two-tower model (model_best.pt)...[/cyan]")
    model, tokenizer = load_two_tower(cfg.two_tower.save_dir, device=device)
    model.eval()

    console.print("[cyan]Loading FAISS index...[/cyan]")
    faiss_index = faiss.read_index(cfg.faiss.index_path)
    if hasattr(faiss_index, "nprobe"):
        faiss_index.nprobe = cfg.faiss.nprobe

    with open(cfg.faiss.docid_map_path, "rb") as f:
        pid_list = pickle.load(f)

    console.print("[cyan]Loading dev queries and qrels...[/cyan]")
    dev_queries_df = pd.read_parquet("data/processed/dev_queries.parquet")
    dev_qrels_df = pd.read_parquet("data/processed/dev_qrels.parquet")

    pos_pids_by_qid = (
        dev_qrels_df[dev_qrels_df["relevance"] > 0]
        .groupby("qid")["pid"]
        .apply(set)
        .to_dict()
    )

    # Only evaluate queries that actually have a gold passage in the qrels.
    dev_queries_df = dev_queries_df[dev_queries_df["qid"].isin(pos_pids_by_qid)]
    console.print(
        f"[green]Evaluating on {len(dev_queries_df):,} dev queries with gold pids[/green]"
    )

    recalls_10 = []
    recalls_100 = []

    qids = dev_queries_df["qid"].tolist()
    texts = dev_queries_df["text"].tolist()

    for i in tqdm(range(0, len(texts), batch_size), desc="Query batches"):
        batch_qids = qids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]

        enc = tokenizer(
            batch_texts,
            max_length=cfg.two_tower.max_seq_len_query,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            q_emb = (
                model.encode_query(
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        _, indices = faiss_index.search(q_emb, 100)

        for row_idx, qid in enumerate(batch_qids):
            gold_pids = pos_pids_by_qid.get(qid)
            if not gold_pids:
                continue
            retrieved_indices = indices[row_idx]
            ranked_pids = [
                pid_list[j] for j in retrieved_indices if j >= 0 and j < len(pid_list)
            ]
            recalls_10.append(recall_at_k(ranked_pids, gold_pids, 10))
            recalls_100.append(recall_at_k(ranked_pids, gold_pids, 100))

    recall_at_10 = float(np.mean(recalls_10)) if recalls_10 else 0.0
    recall_at_100 = float(np.mean(recalls_100)) if recalls_100 else 0.0

    console.print("\n[bold green]Real measured Two-Tower recall (dev set):[/bold green]")
    console.print(f"  Dev queries evaluated: {len(recalls_10)}")
    console.print(f"  Recall@10:  {recall_at_10:.4f}")
    console.print(f"  Recall@100: {recall_at_100:.4f}")
    console.print(f"  Device: {device}")

    result = {
        "model": "model_best.pt",
        "dev_queries": len(recalls_10),
        "recall_at_10": recall_at_10,
        "recall_at_100": recall_at_100,
        "device": device,
    }

    out_path = Path("data/processed/two_tower_recall.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    console.print(f"[green]Saved -> {out_path}[/green]")


if __name__ == "__main__":
    main()
