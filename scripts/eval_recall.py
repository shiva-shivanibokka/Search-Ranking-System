"""
Measure REAL Recall@10 / Recall@100 for the deployed two-tower model — honestly.

Two numbers matter here, and reporting only the first is misleading:

1. **Index coverage.** The demo indexes a 500K-passage subset (pids 0..499,999)
   of MS MARCO's full ~8.8M collection (see `max_passages` in configs/config.yaml).
   The dev qrels reference gold passages from the *full* collection, so only a
   small fraction of dev queries even have their gold passage present in the
   index. A query whose gold passage was never indexed is *unanswerable* — no
   retriever can score it, so counting it drags "recall" toward zero for a
   reason that has nothing to do with model quality.

2. **Answerable-only recall.** Restricting to dev queries whose gold passage IS
   in the indexed subset measures the model's actual retrieval quality.

This script reports coverage, the naive all-query recall, and the answerable-only
recall, using EXACT dot-product search over the committed doc embeddings
(data/embeddings/doc_embeddings.npy, aligned with data/indexes/docid_map.pkl).
Exact search isolates model quality from the deployed FAISS IVF+PQ index's
approximation error.

Usage:
    python scripts/eval_recall.py

Outputs:
    - Console summary
    - data/processed/two_tower_recall.json
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console

sys.path.append(str(Path(__file__).resolve().parents[1]))
from training.evaluate import recall_at_k
from training.two_tower_model import load_two_tower

console = Console()

DOC_EMB_PATH = "data/embeddings/doc_embeddings.npy"
DOCID_MAP_PATH = "data/indexes/docid_map.pkl"
MODEL_DIR = "models/two_tower"


def main(batch_size: int = 64) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold]Two-tower recall eval on device: {device}[/bold]")

    doc_emb = np.load(DOC_EMB_PATH).astype(np.float32)
    with open(DOCID_MAP_PATH, "rb") as f:
        pid_list = pickle.load(f)
    pid_arr = np.asarray(pid_list)
    indexed = set(pid_list)
    console.print(f"Indexed passages: {len(pid_list):,} (a subset of MS MARCO's ~8.8M)")

    qrels = pd.read_parquet("data/processed/dev_qrels.parquet")
    qrels = qrels[qrels["relevance"] > 0]
    gold_by_qid = qrels.groupby("qid")["pid"].apply(set).to_dict()
    queries = pd.read_parquet("data/processed/dev_queries.parquet")
    qid2text = dict(zip(queries["qid"], queries["text"]))

    all_gold = set(qrels["pid"])
    covered = len(all_gold & indexed)
    coverage = covered / len(all_gold) if all_gold else 0.0
    console.print(
        f"Dev gold passages in the index: {covered:,}/{len(all_gold):,} "
        f"({100 * coverage:.1f}%) -> the rest are unanswerable on this subset"
    )

    answerable = [
        (qid, g) for qid, g in gold_by_qid.items() if (g & indexed) and qid in qid2text
    ]
    console.print(f"Answerable dev queries (gold in index): {len(answerable):,}")

    model, tokenizer = load_two_tower(MODEL_DIR, device=device)
    model.eval()

    def encode(texts: list[str]) -> np.ndarray:
        enc = tokenizer(
            texts, max_length=64, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            v = model.encode_query(
                enc["input_ids"].to(device), enc["attention_mask"].to(device)
            )
        return v.cpu().numpy().astype(np.float32)

    r10, r100 = [], []
    for i in range(0, len(answerable), batch_size):
        chunk = answerable[i : i + batch_size]
        q_emb = encode([qid2text[qid] for qid, _ in chunk])
        sims = q_emb @ doc_emb.T
        top = np.argpartition(-sims, 100, axis=1)[:, :100]
        for row, (_, gold) in enumerate(chunk):
            order = top[row][np.argsort(-sims[row, top[row]])]
            ranked = [int(pid_arr[j]) for j in order]
            r10.append(recall_at_k(ranked, gold, 10))
            r100.append(recall_at_k(ranked, gold, 100))

    ans_r10 = float(np.mean(r10)) if r10 else 0.0
    ans_r100 = float(np.mean(r100)) if r100 else 0.0
    # Naive all-query recall is bounded above by coverage; report it so the
    # gap between it and answerable-only recall is explicit, not hidden.
    naive_r100 = ans_r100 * coverage

    console.print("\n[bold green]Measured two-tower recall (dev set):[/bold green]")
    console.print(f"  Index coverage of dev gold passages: {100 * coverage:.1f}%")
    console.print(f"  Answerable queries evaluated:        {len(r10)}")
    console.print(f"  Recall@10  (answerable-only):        {ans_r10:.4f}")
    console.print(f"  Recall@100 (answerable-only):        {ans_r100:.4f}")
    console.print(f"  Recall@100 (naive, all dev queries): ~{naive_r100:.4f}")

    result = {
        "model": "model_best.pt",
        "method": "exact dot-product over committed doc embeddings",
        "index_size": len(pid_list),
        "dev_gold_coverage": round(coverage, 4),
        "answerable_queries": len(r10),
        "recall_at_10_answerable": round(ans_r10, 4),
        "recall_at_100_answerable": round(ans_r100, 4),
        "recall_at_100_naive_all_queries": round(naive_r100, 4),
        "device": device,
        "note": (
            "Naive recall is low only because the demo index holds a 500K subset "
            "of ~8.8M passages; ~98% of dev gold passages are absent. "
            "Answerable-only recall is the model's real retrieval quality."
        ),
    }
    out_path = Path("data/processed/two_tower_recall.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    console.print(f"[green]Saved -> {out_path}[/green]")


if __name__ == "__main__":
    main()
