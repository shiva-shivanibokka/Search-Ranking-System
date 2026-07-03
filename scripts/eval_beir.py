"""
Run BEIR zero-shot evaluation over small out-of-domain datasets using the repo's
trained two-tower retriever + BM25 + RRF hybrid, and write committed results to
data/processed/beir_results.json.

CPU-only; minutes on the small corpora. Requires models/two_tower/ on disk.

    python scripts/eval_beir.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.training_config import get_training_config  # noqa: E402
from scripts.download_beir import (  # noqa: E402
    DEFAULT_DATASETS,
    download_beir_dataset,
    load_beir_dataset,
)
from training.beir_eval import (  # noqa: E402
    TwoTowerBEIRAdapter,
    evaluate_beir_dataset,
)
from training.two_tower_model import load_two_tower  # noqa: E402

console = Console()
RESULTS_PATH = Path("data/processed/beir_results.json")


def _default_loader(name: str) -> tuple[dict, dict, dict]:
    path = download_beir_dataset(name)
    return load_beir_dataset(path)


def run_beir_eval(
    datasets=DEFAULT_DATASETS,
    config_path: str = "configs/config.yaml",
    out_path: Path = RESULTS_PATH,
    adapter=None,
    loader=None,
) -> dict:
    """Evaluate the retriever on each dataset and write beir_results.json.

    adapter/loader are injectable for testing; defaults build the real
    two-tower adapter and download the real BEIR datasets.
    """
    cfg = get_training_config(config_path)

    if adapter is None:
        model, tokenizer = load_two_tower(cfg.two_tower.save_dir, device="cpu")
        adapter = TwoTowerBEIRAdapter(
            model,
            tokenizer,
            device="cpu",
            max_q_len=cfg.two_tower.max_seq_len_query,
            max_d_len=cfg.two_tower.max_seq_len_doc,
        )
    if loader is None:
        loader = _default_loader

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": cfg.two_tower.save_dir,
        "rrf_k": cfg.hybrid_retrieval.rrf_k,
        "datasets": {},
    }
    for name in datasets:
        console.print(f"[bold cyan]Evaluating {name}...[/bold cyan]")
        corpus, queries, qrels = loader(name)
        metrics = evaluate_beir_dataset(
            corpus, queries, qrels, adapter, rrf_k=cfg.hybrid_retrieval.rrf_k
        )
        results["datasets"][name] = {
            "corpus_size": len(corpus),
            "num_queries": sum(1 for q in queries if qrels.get(q)),
            "configs": metrics,
        }
        console.print(metrics)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    console.print(f"[green]Results saved → {out_path}[/green]")
    return results


if __name__ == "__main__":
    run_beir_eval()
