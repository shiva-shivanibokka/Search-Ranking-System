"""
Download small BEIR datasets and load them as (corpus, queries, qrels).

Uses the official BEIR package: beir.util.download_and_unzip fetches the public
TU-Darmstadt mirror, and beir.datasets.data_loader.GenericDataLoader parses
corpus.jsonl / queries.jsonl / qrels/<split>.tsv into plain dicts.

Datasets (small, CPU-friendly zero-shot benchmarks):
  scifact  (~5K docs)          nfcorpus (~3.6K docs, biomedical)   fiqa (financial QA)
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()

BEIR_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
)
DEFAULT_DATASETS = ["scifact", "nfcorpus", "fiqa"]
BEIR_ROOT = Path("data/beir")


def download_beir_dataset(name: str, out_root: Path = BEIR_ROOT) -> str:
    """Download and unzip a BEIR dataset; return the extracted dataset directory.

    Idempotent: skips the network if corpus.jsonl already exists on disk.
    """
    from beir import util

    out_root.mkdir(parents=True, exist_ok=True)
    data_path = out_root / name
    if (data_path / "corpus.jsonl").exists():
        console.print(f"[yellow]Skipping[/yellow] {name} — already downloaded")
        return str(data_path)
    url = BEIR_URL.format(name=name)
    console.print(f"[cyan]Downloading BEIR/{name}...[/cyan]")
    return util.download_and_unzip(url, str(out_root))


def load_beir_dataset(data_path: str, split: str = "test") -> tuple[dict, dict, dict]:
    """Parse a BEIR-format dataset directory into (corpus, queries, qrels).

    corpus:  {doc_id: {"title": str, "text": str}}
    queries: {query_id: str}
    qrels:   {query_id: {doc_id: relevance_int}}
    """
    from beir.datasets.data_loader import GenericDataLoader

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    return corpus, queries, qrels


def main() -> None:
    for name in DEFAULT_DATASETS:
        path = download_beir_dataset(name)
        corpus, queries, qrels = load_beir_dataset(path)
        console.print(
            f"[green]{name}[/green]: {len(corpus):,} docs, "
            f"{len(queries):,} queries, {len(qrels):,} qrels"
        )


if __name__ == "__main__":
    main()
