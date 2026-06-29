"""
Download MS MARCO Passage Ranking dataset via ir_datasets.

Microsoft blob storage returns HTTP 409 (public access disabled).
HuggingFace is blocked on some networks.
ir_datasets is a purpose-built IR benchmark library that manages its own
download mirrors and caching.

Install: pip install ir_datasets  (already in requirements.txt)

Outputs written to data/raw/:
  - collection.tsv          : passages  (pid \\t passage_text)
  - queries.train.tsv       : train queries (qid \\t query_text)
  - queries.dev.tsv         : dev queries   (qid \\t query_text)
  - qrels.train.tsv         : train relevance labels (TREC format)
  - qrels.dev.small.tsv     : dev relevance labels
  - triples.train.small.tsv : (query, positive, negative) triples
"""

import random
import sys
from pathlib import Path

from rich.console import Console

console = Console()
RAW_DIR = Path("data/raw")

# Limits — controls how much data we write to disk
MAX_PASSAGES = 500_000
MAX_TRAIN_Q = 400_000
MAX_DEV_Q = 7_000
MAX_TRAIN_QRELS = 500_000
MAX_DEV_QRELS = -1  # take all (~7K)
MAX_TRIPLES = 500_000

EXPECTED_FILES = [
    "collection.tsv",
    "queries.train.tsv",
    "queries.dev.tsv",
    "qrels.train.tsv",
    "qrels.dev.small.tsv",
    "triples.train.small.tsv",
]


def already_done(path: Path) -> bool:
    if path.exists() and path.stat().st_size > 0:
        size_mb = path.stat().st_size / 1e6
        console.print(
            f"[yellow]Skipping[/yellow] {path.name} — already exists ({size_mb:.1f} MB)"
        )
        return True
    return False


def build_collection() -> None:
    out = RAW_DIR / "collection.tsv"
    if already_done(out):
        return
    import ir_datasets

    console.print(
        f"[cyan]Writing collection.tsv (up to {MAX_PASSAGES:,} passages)...[/cyan]"
    )
    console.print(
        "[dim]ir_datasets will download the corpus on first run (~3 GB)[/dim]"
    )
    ds = ir_datasets.load("msmarco-passage")
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for doc in ds.docs_iter():
            if MAX_PASSAGES > 0 and written >= MAX_PASSAGES:
                break
            text = doc.text.replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{doc.doc_id}\t{text}\n")
            written += 1
            if written % 50_000 == 0:
                console.print(f"  [dim]{written:,} passages written...[/dim]")
    console.print(f"[green]Done[/green] → {out} ({written:,} passages)")


def build_queries_train() -> None:
    out = RAW_DIR / "queries.train.tsv"
    if already_done(out):
        return
    import ir_datasets

    console.print(
        f"[cyan]Writing queries.train.tsv (up to {MAX_TRAIN_Q:,} queries)...[/cyan]"
    )
    ds = ir_datasets.load("msmarco-passage/train")
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for q in ds.queries_iter():
            if MAX_TRAIN_Q > 0 and written >= MAX_TRAIN_Q:
                break
            text = q.text.replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{q.query_id}\t{text}\n")
            written += 1
    console.print(f"[green]Done[/green] → {out} ({written:,} queries)")


def build_queries_dev() -> None:
    out = RAW_DIR / "queries.dev.tsv"
    if already_done(out):
        return
    import ir_datasets

    console.print(
        f"[cyan]Writing queries.dev.tsv (up to {MAX_DEV_Q:,} queries)...[/cyan]"
    )
    ds = ir_datasets.load("msmarco-passage/dev/small")
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for q in ds.queries_iter():
            if MAX_DEV_Q > 0 and written >= MAX_DEV_Q:
                break
            text = q.text.replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{q.query_id}\t{text}\n")
            written += 1
    console.print(f"[green]Done[/green] → {out} ({written:,} queries)")


def build_qrels_train() -> None:
    out = RAW_DIR / "qrels.train.tsv"
    if already_done(out):
        return
    import ir_datasets

    console.print(
        f"[cyan]Writing qrels.train.tsv (up to {MAX_TRAIN_QRELS:,} qrels)...[/cyan]"
    )
    ds = ir_datasets.load("msmarco-passage/train")
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for qrel in ds.qrels_iter():
            if MAX_TRAIN_QRELS > 0 and written >= MAX_TRAIN_QRELS:
                break
            f.write(f"{qrel.query_id}\t0\t{qrel.doc_id}\t{qrel.relevance}\n")
            written += 1
    console.print(f"[green]Done[/green] → {out} ({written:,} qrels)")


def build_qrels_dev() -> None:
    out = RAW_DIR / "qrels.dev.small.tsv"
    if already_done(out):
        return
    import ir_datasets

    console.print("[cyan]Writing qrels.dev.small.tsv...[/cyan]")
    ds = ir_datasets.load("msmarco-passage/dev/small")
    written = 0
    with open(out, "w", encoding="utf-8") as f:
        for qrel in ds.qrels_iter():
            f.write(f"{qrel.query_id}\t0\t{qrel.doc_id}\t{qrel.relevance}\n")
            written += 1
    console.print(f"[green]Done[/green] → {out} ({written:,} qrels)")


def build_triples() -> None:
    """
    Build triples by reading collection.tsv, qrels.train.tsv, and
    queries.train.tsv directly from disk with explicit UTF-8 encoding.
    This avoids the Windows cp1252 crash that occurs when ir_datasets
    streams the corpus through Python's default locale encoding.
    """
    out = RAW_DIR / "triples.train.small.tsv"
    if already_done(out):
        return

    console.print(
        f"[cyan]Writing triples.train.small.tsv (up to {MAX_TRIPLES:,} triples)...[/cyan]"
    )
    console.print("[dim]Reading from disk files with UTF-8 encoding[/dim]")

    collection_path = RAW_DIR / "collection.tsv"
    qrels_path = RAW_DIR / "qrels.train.tsv"
    queries_path = RAW_DIR / "queries.train.tsv"

    # Step 1: load doc texts from collection.tsv (already on disk, UTF-8 safe)
    console.print("[dim]Loading doc texts from collection.tsv...[/dim]")
    doc_texts: dict = {}
    with open(collection_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                doc_texts[parts[0]] = parts[1].strip()
            if MAX_PASSAGES > 0 and len(doc_texts) >= MAX_PASSAGES:
                break
    console.print(f"[dim]Loaded {len(doc_texts):,} doc texts[/dim]")

    # Step 2: build qid -> [positive pid] map from qrels.train.tsv
    from collections import defaultdict

    pos_by_qid: dict = defaultdict(list)
    with open(qrels_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and int(parts[3]) > 0:
                pos_by_qid[parts[0]].append(parts[2])

    # Step 3: stream queries and write triples
    all_doc_ids = list(doc_texts.keys())
    written = 0
    skipped = 0

    with (
        open(queries_path, "r", encoding="utf-8", errors="replace") as qf,
        open(out, "w", encoding="utf-8") as outf,
    ):
        for line in qf:
            if MAX_TRIPLES > 0 and written >= MAX_TRIPLES:
                break
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            qid, query = parts[0], parts[1].strip()
            pos_ids = pos_by_qid.get(qid, [])
            if not pos_ids:
                skipped += 1
                continue
            pos_text = doc_texts.get(pos_ids[0], "")
            if not pos_text:
                skipped += 1
                continue
            # Random negative not in positives
            pos_set = set(pos_ids)
            neg_id = random.choice(all_doc_ids)
            attempts = 0
            while neg_id in pos_set and attempts < 10:
                neg_id = random.choice(all_doc_ids)
                attempts += 1
            neg_text = doc_texts.get(neg_id, "")
            if not neg_text:
                skipped += 1
                continue
            outf.write(f"{query}\t{pos_text}\t{neg_text}\n")
            written += 1
            if written % 50_000 == 0:
                console.print(f"  [dim]{written:,} triples written...[/dim]")

    console.print(
        f"[green]Done[/green] → {out} ({written:,} triples, {skipped} skipped)"
    )


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import ir_datasets
    except ImportError:
        console.print(
            "[red]ERROR: ir_datasets not installed. Run: pip install ir_datasets[/red]"
        )
        sys.exit(1)

    console.print("[bold]Downloading MS MARCO via ir_datasets...[/bold]")
    console.print(
        "[dim]Downloads are cached in ~/.ir_datasets/ after first run[/dim]\n"
    )

    build_collection()
    build_queries_train()
    build_queries_dev()
    build_qrels_train()
    build_qrels_dev()
    build_triples()

    # ── Final check ───────────────────────────────────────────────────────────
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
