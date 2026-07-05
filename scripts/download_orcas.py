"""
Download and reservoir-sample the ORCAS click-log dataset.

ORCAS (Open Resource for Click Analysis in Search) is a Microsoft dataset that
maps real Bing search queries to the MS MARCO documents users clicked. It is
used here ONLY to calibrate a synthetic click model (position bias / relevance
noise) for the feedback-loop simulation — never served in production and never
used commercially.

Format: tab-separated, 4 columns: qid, query, did, url
  - qid: numeric query id
  - query: raw user query text
  - did: MS MARCO document id (D-prefixed) the user clicked
  - url: the URL of the clicked document

Source: ~18.8M rows, ~329.7 MB gzipped.

IMPORTANT — license: ORCAS is released by Microsoft under a NON-COMMERCIAL
research-only license. Downloading it requires accepting those terms. This
script will not download anything unless the caller explicitly opts in via
--accept-noncommercial-license.
"""

from __future__ import annotations

import argparse
import gzip
import random
import sys
from collections.abc import Iterable
from pathlib import Path

ORCAS_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/orcas.tsv.gz"

ORCAS_LICENSE_NOTICE = (
    "ORCAS is released by Microsoft under a NON-COMMERCIAL RESEARCH-ONLY "
    "license. By downloading ORCAS data you must accept those terms: it may "
    "only be used for non-commercial research purposes. This project uses "
    "ORCAS solely to calibrate a click model (position bias / relevance "
    "noise) for a search-ranking research simulation — it is NOT used "
    "commercially and clicked documents/urls are not redistributed. Pass "
    "--accept-noncommercial-license to confirm you accept these terms "
    "before any data is downloaded."
)

RAW_DIR = Path("data/raw")
DEFAULT_OUT = RAW_DIR / "orcas_sample.tsv"
DEFAULT_SAMPLE_SIZE = 200_000
DEFAULT_SEED = 42


def parse_orcas_line(line: str) -> dict | None:
    """Parse one ORCAS TSV line into {qid, query, did, url}.

    Returns None if the line does not have exactly 4 tab-separated fields.
    """
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 4:
        return None
    qid, query, did, url = fields
    return {"qid": qid, "query": query, "did": did, "url": url}


def sample_orcas(
    src_lines: Iterable[str], n: int, seed: int = DEFAULT_SEED
) -> list[dict]:
    """Reservoir-sample up to n parsed ORCAS rows from a (potentially huge) line stream.

    Deterministic given the same seed and input, and memory-bounded: only n
    rows are ever held in memory regardless of the size of src_lines.
    """
    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    for line in src_lines:
        row = parse_orcas_line(line)
        if row is None:
            continue
        if len(reservoir) < n:
            reservoir.append(row)
        else:
            j = rng.randint(0, seen)
            if j < n:
                reservoir[j] = row
        seen += 1
    return reservoir


def download_and_sample(
    url: str, n: int, seed: int = DEFAULT_SEED
) -> list[dict]:
    """Stream-download the gzipped ORCAS TSV and reservoir-sample it.

    Lazy-imports requests so the module (and the pure-logic tests) never
    need it installed / never touch the network unless this is actually
    called.
    """
    import requests

    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with gzip.GzipFile(fileobj=resp.raw) as gz:
            lines = (raw_line.decode("utf-8", errors="replace") for raw_line in gz)
            return sample_orcas(lines, n, seed=seed)


def write_sample_tsv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['qid']}\t{row['query']}\t{row['did']}\t{row['url']}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and reservoir-sample the ORCAS click-log dataset."
    )
    parser.add_argument(
        "--accept-noncommercial-license",
        action="store_true",
        help="Confirm you accept ORCAS's non-commercial research-only license.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of rows to reservoir-sample (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reservoir sampling (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output TSV path (default: {DEFAULT_OUT}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    print(ORCAS_LICENSE_NOTICE)

    if not args.accept_noncommercial_license:
        print(
            "\nRefusing to download: pass --accept-noncommercial-license "
            "to confirm you accept the non-commercial research-only terms above."
        )
        return 1

    rows = download_and_sample(ORCAS_URL, args.sample_size, seed=args.seed)
    write_sample_tsv(rows, args.out)
    print(f"Wrote {len(rows):,} sampled rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
