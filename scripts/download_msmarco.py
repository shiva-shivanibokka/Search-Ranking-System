"""
Download MS MARCO Passage Ranking dataset via Hugging Face datasets library.

Microsoft's blob storage is no longer publicly accessible (HTTP 409).
All files are now sourced from the official HuggingFace mirror:
  https://huggingface.co/datasets/microsoft/ms_marco

Outputs written to data/raw/:
  - collection.tsv         : passages  (pid \\t passage_text)
  - queries.train.tsv      : train queries (qid \\t query_text)
  - queries.dev.tsv        : dev queries   (qid \\t query_text)
  - qrels.train.tsv        : train relevance labels (qid 0 pid 1)
  - qrels.dev.small.tsv    : dev relevance labels
  - triples.train.small.tsv: (query, positive, negative) triples
"""

import sys
from pathlib import Path
from collections import defaultdict

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
)

console = Console()
RAW_DIR = Path("data/raw")

EXPECTED_FILES = [
    "collection.tsv",
    "queries.train.tsv",
    "queries.dev.tsv",
    "qrels.train.tsv",
    "qrels.dev.small.tsv",
    "triples.train.small.tsv",
]

# Limits — set to -1 to use full dataset (very large)
MAX_PASSAGES = 500_000
MAX_TRAIN_Q = 400_000
MAX_DEV_Q = 7_000
MAX_TRAIN_QRELS = -1  # take all
MAX_DEV_QRELS = -1  # take all
MAX_TRIPLES = 500_000


def already_done(path: Path) -> bool:
    if path.exists() and path.stat().st_size > 0:
        size_mb = path.stat().st_size / 1e6
        console.print(
            f"[yellow]Skipping[/yellow] {path.name} — already exists ({size_mb:.1f} MB)"
        )
        return True
    return False


def build_collection(ds_passages) -> None:
    out = RAW_DIR / "collection.tsv"
    if already_done(out):
        return
    console.print(
        f"[cyan]Writing collection.tsv (up to {MAX_PASSAGES:,} passages)...[/cyan]"
    )
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds_passages:
            if MAX_PASSAGES > 0 and written >= MAX_PASSAGES:
                break
            pid = row["id"]
            # HF ms_marco v2.1 passages field is a dict with "passage_text" list
            passages = row.get("passages", {}).get("passage_text", [])
            for text in passages:
                text = text.replace("\t", " ").replace("\n", " ").strip()
                if text:
                    f.write(f"{pid}\t{text}\n")
                    written += 1
                    if MAX_PASSAGES > 0 and written >= MAX_PASSAGES:
                        break
    console.print(f"[green]Done[/green] → {out} ({written:,} passages)")


def build_queries(ds, split: str, out_name: str, max_q: int) -> None:
    out = RAW_DIR / out_name
    if already_done(out):
        return
    console.print(f"[cyan]Writing {out_name} (up to {max_q:,} queries)...[/cyan]")
    written = 0
    seen = set()
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            if max_q > 0 and written >= max_q:
                break
            qid = row["query_id"]
            if qid in seen:
                continue
            seen.add(qid)
            query = row["query"].replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{qid}\t{query}\n")
            written += 1
    console.print(f"[green]Done[/green] → {out} ({written:,} queries)")


def build_qrels(ds, out_name: str, max_rows: int) -> None:
    out = RAW_DIR / out_name
    if already_done(out):
        return
    console.print(f"[cyan]Writing {out_name}...[/cyan]")
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            if max_rows > 0 and written >= max_rows:
                break
            qid = row["query_id"]
            passages = row.get("passages", {})
            pids = passages.get("pid", [])
            is_selected = passages.get("is_selected", [])
            for pid, sel in zip(pids, is_selected):
                if sel == 1:
                    f.write(f"{qid}\t0\t{pid}\t1\n")
                    written += 1
    console.print(f"[green]Done[/green] → {out} ({written:,} qrels)")


def build_triples(ds_train, max_triples: int) -> None:
    out = RAW_DIR / "triples.train.small.tsv"
    if already_done(out):
        return
    console.print(
        f"[cyan]Writing triples.train.small.tsv (up to {max_triples:,} triples)...[/cyan]"
    )
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in ds_train:
            if max_triples > 0 and written >= max_triples:
                break
            query = row["query"].replace("\t", " ").replace("\n", " ").strip()
            passages = row.get("passages", {})
            texts = passages.get("passage_text", [])
            is_selected = passages.get("is_selected", [])

            pos_texts = [t for t, s in zip(texts, is_selected) if s == 1]
            neg_texts = [t for t, s in zip(texts, is_selected) if s == 0]

            if not pos_texts or not neg_texts:
                continue

            pos = pos_texts[0].replace("\t", " ").replace("\n", " ").strip()
            neg = neg_texts[0].replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{query}\t{pos}\t{neg}\n")
            written += 1

    console.print(f"[green]Done[/green] → {out} ({written:,} triples)")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Downloading MS MARCO via Hugging Face datasets...[/bold]")
    console.print(
        "(First run downloads ~3 GB to HF cache, subsequent runs are instant)\n"
    )

    try:
        from datasets import load_dataset
    except ImportError:
        console.print(
            "[red]ERROR: 'datasets' package not found. Run: pip install datasets[/red]"
        )
        sys.exit(1)

    # Load splits — HF caches them locally after first download
    console.print(
        "[cyan]Loading train split from HuggingFace (this may take a while)...[/cyan]"
    )
    ds_train = load_dataset(
        "microsoft/ms_marco", "v2.1", split="train", trust_remote_code=True
    )

    console.print("[cyan]Loading validation split from HuggingFace...[/cyan]")
    ds_dev = load_dataset(
        "microsoft/ms_marco", "v2.1", split="validation", trust_remote_code=True
    )

    # ── collection.tsv ────────────────────────────────────────────────────────
    # Use train split for passages (it covers the full corpus)
    build_collection(ds_train)

    # ── queries ───────────────────────────────────────────────────────────────
    build_queries(ds_train, "train", "queries.train.tsv", MAX_TRAIN_Q)
    build_queries(ds_dev, "dev", "queries.dev.tsv", MAX_DEV_Q)

    # ── qrels ─────────────────────────────────────────────────────────────────
    build_qrels(ds_train, "qrels.train.tsv", MAX_TRAIN_QRELS)
    build_qrels(ds_dev, "qrels.dev.small.tsv", MAX_DEV_QRELS)

    # ── triples ───────────────────────────────────────────────────────────────
    build_triples(ds_train, MAX_TRIPLES)

    # ── Verify ────────────────────────────────────────────────────────────────
    console.print("\n[bold]File check:[/bold]")
    all_ok = True
    for filename in EXPECTED_FILES:
        path = RAW_DIR / filename
        if path.exists() and path.stat().st_size > 0:
            size_mb = path.stat().st_size / 1e6
            console.print(f"  [green]✓[/green] {filename} ({size_mb:.1f} MB)")
        else:
            console.print(f"  [red]✗[/red] {filename} — MISSING")
            all_ok = False

    if all_ok:
        console.print("\n[bold green]All MS MARCO files ready.[/bold green]")
        console.print("Next step: [cyan]python scripts/preprocess.py[/cyan]")
    else:
        console.print(
            "\n[bold red]Some files are missing. Re-run this script.[/bold red]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
