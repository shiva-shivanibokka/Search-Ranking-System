"""The BM25 tokenizer, and the invariant that every call site shares it.

These tests exist because the repo previously tokenized BM25 with
``text.lower().split()`` at eight independent call sites, including the index
build and the serving query path. See services/shared/text.py for what that
cost on BEIR.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from rank_bm25 import BM25Okapi

from services.shared.text import STOPWORDS_EN, tokenize, tokenize_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_punctuation_is_stripped_so_a_term_matches_at_a_sentence_end():
    """The actual defect: ``lower().split()`` keeps the comma, so a query for
    "cells" cannot match a document that writes "cells,"."""
    assert tokenize("The cells, and their nuclei.") == ["cells", "nuclei"]
    # The old behaviour, shown explicitly for contrast.
    assert "cells," in "The cells, and their nuclei.".lower().split()


def test_stopwords_are_removed():
    assert tokenize("the of and a") == []
    assert "the" in STOPWORDS_EN


def test_case_is_folded():
    assert tokenize("BM25 Okapi") == tokenize("bm25 okapi")


def test_digits_survive():
    """BM25 and covid19 are real query terms; do not drop them."""
    assert tokenize("BM25 covid19") == ["bm25", "covid19"]


def test_hyphens_and_slashes_split():
    assert tokenize("state-of-the-art and/or") == ["state", "art"]


def test_tokenize_corpus_matches_tokenize_per_document():
    docs = ["Hello, world!", "The second doc."]
    assert tokenize_corpus(docs) == [tokenize(d) for d in docs]


def test_punctuation_actually_changes_bm25_ranking():
    """End-to-end: with whitespace splitting the right document is unreachable.

    Document 0 is the only one about mitochondria, but it writes the word with a
    trailing full stop. Under ``lower().split()`` the query term "mitochondria"
    does not match "mitochondria." and the document scores zero.
    """
    corpus = [
        "The organelle studied was mitochondria.",
        "Ribosomes translate messenger RNA into protein chains.",
        "Chloroplasts perform photosynthesis in plant leaves.",
    ]
    query = "mitochondria"

    naive = BM25Okapi([c.lower().split() for c in corpus])
    assert naive.get_scores(query.lower().split())[0] == pytest.approx(0.0), (
        "expected the old tokenization to score the only relevant document at zero"
    )

    fixed = BM25Okapi(tokenize_corpus(corpus))
    scores = fixed.get_scores(tokenize(query))
    assert scores[0] > 0.0
    assert scores.argmax() == 0


# ---------------------------------------------------------------------------
# The invariant: no call site may reintroduce its own tokenizer.
# ---------------------------------------------------------------------------

_BANNED = re.compile(r"\.lower\(\)\s*\.split\(\)")

_SEARCH_DIRS = ("services", "training", "deploy", "scripts", "gradio_app")


# text.py is the module that DEFINES the shared tokenizer; its docstring quotes
# the old pattern on purpose to explain what was replaced.
_EXEMPT = {Path("services") / "shared" / "text.py"}


def _python_files():
    for d in _SEARCH_DIRS:
        root = REPO_ROOT / d
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if path.relative_to(REPO_ROOT) in _EXEMPT:
                continue
            yield path


def test_no_call_site_tokenizes_with_lower_split():
    """A BM25 index built with one tokenization and queried with another returns
    near-garbage, so the tokenizer must not be re-inlined anywhere.

    If this fails, import ``services.shared.text.tokenize`` at the offending
    site instead of writing a new tokenizer.
    """
    offenders = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _BANNED.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
    assert not offenders, (
        "these sites tokenize BM25 input themselves instead of using "
        f"services.shared.text.tokenize: {offenders}"
    )
