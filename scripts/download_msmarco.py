"""
Download MS MARCO Passage Ranking dataset.

Files downloaded:
  - collection.tsv         : 500K passages (pid \t passage_text)
  - queries.train.tsv      : 400K training queries (qid \t query_text)
  - queries.dev.tsv        : dev queries (qid \t query_text)
  - qrels.train.tsv        : train relevance labels (qid 0 pid 1)
  - qrels.dev.small.tsv    : dev relevance labels
  - triples.train.small    : (query, positive, negative) triples for training
"""

import os
import sys
import gzip
import tarfile
import urllib.request
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import (
    Progress,
    DownloadColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)

console = Console()

URLS = {
    "collection.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
    "queries.train.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.train.tsv",
    "queries.dev.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.dev.tsv",
    "qrels.train.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv",
    "qrels.dev.small.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv",
    "triples.train.small.tsv": "https://msmarco.z22.web.core.windows.net/msmarcoranking/triples.train.small.tar.gz",
}

RAW_DIR = Path("data/raw")


def download_file(url: str, dest: Path) -> None:
    """Download a file with a rich progress bar."""
    if dest.exists():
        console.print(f"[yellow]Skipping[/yellow] {dest.name} — already exists")
        return

    console.print(f"[cyan]Downloading[/cyan] {dest.name} from {url}")
    with Progress(
        "[progress.description]{task.description}",
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(dest.name, total=None)

        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                progress.update(
                    task, total=total_size, completed=block_num * block_size
                )

        urllib.request.urlretrieve(url, dest, reporthook=reporthook)

    console.print(f"[green]Done[/green] → {dest}")


def extract_tar(archive: Path, dest_dir: Path) -> None:
    """Extract .tar.gz archive."""
    console.print(f"[cyan]Extracting[/cyan] {archive.name}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest_dir)
    archive.unlink()
    console.print(f"[green]Extracted[/green] → {dest_dir}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in URLS.items():
        is_archive = url.endswith(".tar.gz")
        final_path = RAW_DIR / filename

        if is_archive:
            archive_path = RAW_DIR / (filename + ".tar.gz")
            if not final_path.exists():
                download_file(url, archive_path)
                extract_tar(archive_path, RAW_DIR)
            else:
                console.print(
                    f"[yellow]Skipping[/yellow] {filename} — already extracted"
                )
        else:
            download_file(url, final_path)

    # Verify all files exist
    console.print("\n[bold]File check:[/bold]")
    all_ok = True
    for filename in URLS:
        path = RAW_DIR / filename
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            console.print(f"  [green]✓[/green] {filename} ({size_mb:.1f} MB)")
        else:
            console.print(f"  [red]✗[/red] {filename} — MISSING")
            all_ok = False

    if all_ok:
        console.print(
            "\n[bold green]All MS MARCO files downloaded successfully.[/bold green]"
        )
    else:
        console.print(
            "\n[bold red]Some files are missing. Re-run this script.[/bold red]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
