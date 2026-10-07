"""Measure what the FAISS IVF+PQ index actually costs against exact search.

WHY THIS SCRIPT EXISTS
----------------------
The README claimed the approximate index retains "~95% recall@100 compared to
exact search". Nothing in the repo measured that -- it was a plausible figure
borrowed from the FAISS literature for a generic IVF+PQ setup, not a measurement
of THIS index with THIS nlist/nprobe/m configuration over THESE embeddings. An
unmeasured number next to a column of measured ones is the most misleading kind,
because a reader cannot tell them apart.

So this measures it, and reports the two different things "recall vs exact" can
mean -- which differ a lot, and only the second one matters to a user:

  1. **Set agreement @k** -- how much of exact search's top-k the approximate
     index returns. This is the quantity the FAISS literature usually means by
     "recall@k" of an ANN index. It is a measure of index fidelity.

  2. **Gold recall retained** -- the end-to-end number: Recall@k against the
     human relevance judgements under FAISS, divided by the same under exact
     search. Approximate search can drop documents exact search found while
     keeping the *relevant* ones, so this is usually much kinder than (1).

Usage:
    python scripts/eval_faiss_recall.py [--queries 1000]

Outputs:
    - Console summary
    - data/processed/faiss_recall.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console

sys.path.append(str(Path(__file__).resolve().parents[1]))
from training.two_tower_model import load_two_tower  # noqa: E402

console = Console()

DOC_EMB_PATH = "data/embeddings/doc_embeddings.npy"
DOCID_MAP_PATH = "data/indexes/docid_map.pkl"
FAISS_INDEX_PATH = "data/indexes/faiss_ivfpq.index"
MODEL_DIR = "models/two_tower"
OUT_PATH = Path("data/processed/faiss_recall.json")


def encode_queries(texts: list[str], model, tokenizer, device: str, bs: int = 64):
    out = []
    for i in range(0, len(texts), bs):
        enc = tokenizer(
            texts[i : i + bs],
            max_length=64,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            emb = (
                model.encode_query(
                    enc["input_ids"].to(device), enc["attention_mask"].to(device)
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        out.append(emb)
    return np.vstack(out)


def main(n_queries: int, k: int = 100, seed: int = 0) -> None:
    import faiss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[bold]FAISS vs exact, device={device}[/bold]")

    doc_emb = np.load(DOC_EMB_PATH).astype(np.float32)
    with open(DOCID_MAP_PATH, "rb") as f:
        pid_list = pickle.load(f)
    pid_arr = np.asarray(pid_list)
    index = faiss.read_index(FAISS_INDEX_PATH)
    console.print(
        f"docs={len(pid_arr):,}  dim={doc_emb.shape[1]}  "
        f"nprobe={getattr(index, 'nprobe', 'n/a')}"
    )

    qrels = pd.read_parquet("data/processed/dev_qrels.parquet")
    qrels = qrels[qrels["relevance"] > 0]
    gold_by_qid = qrels.groupby("qid")["pid"].apply(set).to_dict()
    queries = pd.read_parquet("data/processed/dev_queries.parquet")
    qid2text = dict(zip(queries["qid"], queries["text"]))

    indexed = set(pid_list)
    # Only answerable queries: a query whose gold passage was never indexed tells
    # us nothing about approximation error.
    qids = [q for q, g in gold_by_qid.items() if q in qid2text and (g & indexed)]
    rng = np.random.default_rng(seed)
    if n_queries and n_queries < len(qids):
        qids = list(rng.choice(np.asarray(qids), size=n_queries, replace=False))
    console.print(f"Evaluating {len(qids):,} answerable dev queries")

    model, tokenizer = load_two_tower(MODEL_DIR, device=device)
    q_emb = encode_queries([qid2text[q] for q in qids], model, tokenizer, device)

    # FAISS top-k
    _, faiss_idx = index.search(q_emb, k)

    agree = []
    rec_exact = []
    rec_faiss = []
    # Exact top-k in query chunks to bound memory (1M x dim is already ~1GB).
    chunk = 64
    for start in range(0, len(qids), chunk):
        qe = q_emb[start : start + chunk]
        sims = doc_emb @ qe.T  # (N_docs, chunk)
        part = np.argpartition(-sims, kth=k - 1, axis=0)[:k]  # (k, chunk)
        for j in range(qe.shape[0]):
            col = part[:, j]
            col = col[np.argsort(-sims[col, j])]
            exact_pids = set(pid_arr[col].tolist())
            fp = faiss_idx[start + j]
            faiss_pids = set(pid_arr[fp[fp >= 0]].tolist())
            gold = gold_by_qid[qids[start + j]] & indexed

            agree.append(len(exact_pids & faiss_pids) / k)
            rec_exact.append(len(gold & exact_pids) / len(gold))
            rec_faiss.append(len(gold & faiss_pids) / len(gold))

    agreement = float(np.mean(agree))
    r_exact = float(np.mean(rec_exact))
    r_faiss = float(np.mean(rec_faiss))
    retained = (r_faiss / r_exact) if r_exact > 0 else 0.0

    result = {
        "num_queries": len(qids),
        "k": k,
        "index_size": int(len(pid_arr)),
        "nprobe": int(getattr(index, "nprobe", 0)),
        "seed": seed,
        "device": device,
        f"set_agreement_at_{k}": round(agreement, 4),
        f"gold_recall_at_{k}_exact": round(r_exact, 4),
        f"gold_recall_at_{k}_faiss": round(r_faiss, 4),
        "gold_recall_retained_fraction": round(retained, 4),
        "note": (
            "set_agreement is the fraction of exact search's top-k that the IVF+PQ "
            "index also returns (index fidelity). gold_recall_retained_fraction is "
            "FAISS Recall@k divided by exact Recall@k against the dev qrels "
            "(end-to-end cost to the user). They are different quantities and the "
            "second is the one a search quality claim should cite."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    console.print(f"[bold]set agreement@{k}      [/bold] {agreement:.4f}")
    console.print(f"[bold]gold recall@{k} exact  [/bold] {r_exact:.4f}")
    console.print(f"[bold]gold recall@{k} faiss  [/bold] {r_faiss:.4f}")
    console.print(f"[bold]fraction retained      [/bold] {retained:.4f}")
    console.print(f"[green]saved -> {OUT_PATH}[/green]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()
    main(args.queries, args.k)
