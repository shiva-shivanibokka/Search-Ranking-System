"""Load the persisted BM25 index, refusing one built with a different tokenizer.

WHY
---
A BM25 index is a term -> posting-list map. If the index is built with one
tokenization and queried with another, lookups miss and retrieval degrades
silently -- no exception, no log line, just worse results. That is not a
hypothetical: this repo changed its tokenizer (see services/shared/text.py), and
two routes can still hand a process a stale index --

  * ``data/indexes/bm25_index.pkl`` left over from an earlier build, which
    ``scripts/preprocess.py`` used to reuse unconditionally; and
  * the prebuilt artifacts ``scripts/bootstrap.py`` pulls from Hugging Face Hub,
    which were built with the old tokenizer.

So the index is written with a sidecar recording the tokenizer that built it, and
loading verifies it. A mismatch raises; a missing sidecar means a pre-fingerprint
index and also raises, because such an index is by definition the old
tokenization. Both tell the caller to rebuild.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from services.shared.text import TOKENIZER_VERSION

SIDECAR_SUFFIX = ".tokenizer.json"


class StaleIndexError(RuntimeError):
    """The index on disk was built with a different tokenizer than this code."""


def sidecar_path(index_path: str | Path) -> Path:
    return Path(str(index_path) + SIDECAR_SUFFIX)


def write_fingerprint(index_path: str | Path) -> Path:
    """Record which tokenizer built the index next to it."""
    p = sidecar_path(index_path)
    p.write_text(
        json.dumps({"tokenizer_version": TOKENIZER_VERSION}, indent=2),
        encoding="utf-8",
    )
    return p


def _hint(index_path: str | Path, found: str) -> str:
    return (
        f"BM25 index at {index_path} was built with tokenizer {found!r}, but this "
        f"code tokenizes with {TOKENIZER_VERSION!r}. Querying an index through a "
        "different tokenizer silently returns near-garbage rather than failing, so "
        "this is refused.\n"
        "Rebuild it:  rm data/indexes/bm25_index.pkl*  &&  python scripts/preprocess.py\n"
        "(If the index came from scripts/bootstrap.py, the published artifact "
        "predates the tokenizer fix and must be rebuilt locally.)"
    )


def load_bm25_index(index_path: str | Path) -> Any:
    """Unpickle the BM25 index after checking the tokenizer fingerprint."""
    index_path = Path(index_path)
    side = sidecar_path(index_path)
    if not side.exists():
        raise StaleIndexError(_hint(index_path, "unknown (no fingerprint sidecar)"))
    found = json.loads(side.read_text(encoding="utf-8")).get("tokenizer_version")
    if found != TOKENIZER_VERSION:
        raise StaleIndexError(_hint(index_path, found))
    with open(index_path, "rb") as f:
        return pickle.load(f)
