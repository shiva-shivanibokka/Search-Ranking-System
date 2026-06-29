"""
Build a FAISS IVF+PQ index from two-tower document embeddings.

Steps:
  1. Load trained doc encoder from models/two_tower/
  2. Embed all 500K passages in batches
  3. Save embeddings to data/embeddings/doc_embeddings.npy (versioned by DVC)
  4. Train FAISS IVF1024,PQ32 index
  5. Add all embeddings to the index
  6. Save index to data/indexes/faiss_ivfpq.index
  7. Save pid→index mapping to data/indexes/docid_map.pkl

At query time, the retrieval service:
  - Encodes the query with query_encoder
  - Calls index.search(q_emb, top_k=100)
  - Maps returned indices back to pids using docid_map
"""

import pickle
import sys
from pathlib import Path

import faiss
import mlflow
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.training_config import get_training_config
from training.two_tower_model import TwoTowerModel, load_two_tower

console = Console()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embed_passages(
    model: TwoTowerModel,
    tokenizer,
    passages_df: pd.DataFrame,
    batch_size: int = 512,
    max_seq_len: int = 180,
) -> np.ndarray:
    """Embed all passages with the doc tower. Returns (N, D) float32 array."""
    model.eval()
    all_embs = []

    for i in tqdm(range(0, len(passages_df), batch_size), desc="Embedding passages"):
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
        all_embs.append(emb.cpu().numpy().astype(np.float32))

    return np.vstack(all_embs)


def build_ivfpq_index(
    embeddings: np.ndarray,
    index_type: str = "IVF1024,PQ32",
    nprobe: int = 64,
    use_gpu: bool = False,
) -> faiss.Index:
    """
    Build and train a FAISS IVF+PQ index.

    IVF (Inverted File Index): divides the space into nlist Voronoi cells.
      At search time, only nprobe cells are visited → faster than flat search.
    PQ (Product Quantization): compresses vectors by splitting into subvectors
      and quantizing each. Dramatically reduces memory (from 256*4 bytes = 1KB
      per vector to ~32 bytes with PQ32).

    For 500K vectors at 256-dim:
      - Flat (exact): 500K * 256 * 4 bytes = ~512 MB, exact results
      - IVF1024,PQ32: ~16 MB, ~95% recall@10 vs exact search
    """
    dim = embeddings.shape[1]
    console.print(
        f"[cyan]Building FAISS {index_type} index (dim={dim}, N={len(embeddings):,})[/cyan]"
    )

    # FAISS requires training data to be contiguous float32
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    index = faiss.index_factory(dim, index_type, faiss.METRIC_INNER_PRODUCT)

    if use_gpu and faiss.get_num_gpus() > 0:
        console.print("[cyan]Using GPU for FAISS index training[/cyan]")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Train the index (learns cluster centroids and PQ codebooks)
    console.print("[cyan]Training FAISS index...[/cyan]")
    index.train(embeddings)
    console.print("[green]Index trained.[/green]")

    # Add all vectors
    console.print("[cyan]Adding vectors to index...[/cyan]")
    index.add(embeddings)
    console.print(f"[green]Added {index.ntotal:,} vectors.[/green]")

    # Set nprobe for search time accuracy/speed tradeoff
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
    else:
        # For GPU index, need to set via IndexIVF
        faiss.downcast_index(index).nprobe = nprobe

    # Move back to CPU if we trained on GPU
    if use_gpu and faiss.get_num_gpus() > 0:
        index = faiss.index_gpu_to_cpu(index)

    return index


def main(config_path: str = "configs/config.yaml"):
    cfg = get_training_config(config_path)
    faiss_cfg = cfg.faiss
    mlf_cfg = cfg.mlflow

    emb_dir = Path("data/embeddings")
    idx_dir = Path("data/indexes")
    emb_dir.mkdir(parents=True, exist_ok=True)
    idx_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────────
    checkpoint_dir = cfg.two_tower.save_dir
    console.print(f"[cyan]Loading two-tower model from {checkpoint_dir}[/cyan]")
    model, tokenizer = load_two_tower(checkpoint_dir, device=str(DEVICE))

    # ── Load passages ───────────────────────────────────────────────────────────
    console.print("[cyan]Loading passages...[/cyan]")
    passages_df = pd.read_parquet("data/processed/passages.parquet")
    pid_list = passages_df["pid"].tolist()

    # ── Embed ───────────────────────────────────────────────────────────────────
    emb_path = emb_dir / "doc_embeddings.npy"
    if emb_path.exists():
        console.print(f"[yellow]Loading cached embeddings from {emb_path}[/yellow]")
        doc_embeddings = np.load(emb_path)
    else:
        doc_embeddings = embed_passages(model, tokenizer, passages_df)
        np.save(emb_path, doc_embeddings)
        console.print(
            f"[green]Embeddings saved → {emb_path} ({doc_embeddings.shape})[/green]"
        )

    # ── Build FAISS index ───────────────────────────────────────────────────────
    index_path = Path(faiss_cfg.index_path)
    if index_path.exists():
        console.print(f"[yellow]FAISS index already exists at {index_path}[/yellow]")
        index = faiss.read_index(str(index_path))
    else:
        use_gpu = torch.cuda.is_available()
        index = build_ivfpq_index(
            doc_embeddings,
            index_type=faiss_cfg.index_type,
            nprobe=faiss_cfg.nprobe,
            use_gpu=use_gpu,
        )
        faiss.write_index(index, str(index_path))
        console.print(f"[green]FAISS index saved → {index_path}[/green]")

    # ── Save pid map ─────────────────────────────────────────────────────────────
    docid_map_path = Path(faiss_cfg.docid_map_path)
    with open(docid_map_path, "wb") as f:
        pickle.dump(pid_list, f)
    console.print(f"[green]PID map saved → {docid_map_path}[/green]")

    # ── Sanity check: search a test query ────────────────────────────────────────
    console.print("[cyan]Sanity check: searching test query...[/cyan]")
    test_query = "what is information retrieval"
    enc = tokenizer(
        test_query, max_length=64, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        q_emb = (
            model.encode_query(
                enc["input_ids"].to(DEVICE),
                enc["attention_mask"].to(DEVICE),
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    scores, indices = index.search(q_emb, 5)
    console.print(f"\nQuery: '{test_query}'")
    console.print("Top-5 results:")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        pid = pid_list[idx]
        text_snippet = passages_df[passages_df["pid"] == pid]["text"].values
        snippet = text_snippet[0][:100] if len(text_snippet) > 0 else "N/A"
        console.print(f"  {rank + 1}. [pid={pid}, score={score:.4f}] {snippet}...")

    # ── Log to MLflow ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(mlf_cfg.tracking_uri)
    mlflow.set_experiment(mlf_cfg.experiment_name)
    with mlflow.start_run(run_name="build_faiss_index"):
        mlflow.log_params(
            {
                "index_type": faiss_cfg.index_type,
                "nprobe": faiss_cfg.nprobe,
                "embedding_dim": faiss_cfg.embedding_dim,
                "num_vectors": index.ntotal,
            }
        )
        mlflow.log_artifact(str(index_path))
        mlflow.log_artifact(str(docid_map_path))

    console.print("\n[bold green]FAISS index build complete.[/bold green]")
    console.print("Next step: [cyan]python training/train_lambdarank.py[/cyan]")


if __name__ == "__main__":
    main()
