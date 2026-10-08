"""The single BM25 tokenizer for this repository.

WHY THIS MODULE EXISTS
----------------------
Every BM25 call site in this repo used to tokenize with ``text.lower().split()``
-- the index build in ``scripts/preprocess.py``, the serving path in
``deploy/engine.py`` and ``services/retrieval/main.py``, the LambdaRank feature
builder, and both evaluation harnesses. Splitting on whitespace alone leaves
punctuation glued to words, so ``"cells,"`` and ``"cells"`` are different terms
and a query term never matches a document term that happens to end a sentence.

That under-scored BM25 everywhere, and because BM25 is also the lexical half of
the hybrid retriever and one of the LambdaRank features, it under-scored the
whole system. On the BEIR datasets the cost was large:

    nDCG@10        lower().split()   this module   published BEIR BM25
    SciFact             0.5597          0.6637            ~0.665
    NFCorpus            0.2668          0.3087            ~0.325
    FiQA                0.1591          0.2303            ~0.236

The third column is the point. Landing within ~0.002-0.016 of the published
BM25 baseline on all three datasets is what tells us this tokenization is
*correct* rather than merely higher-scoring -- the previous numbers were not a
BM25 baseline at all, they were a handicapped one, which made every comparison
against it flattering and meaningless.

WHAT IT DOES
------------
Case-folds, splits on word boundaries (dropping punctuation), and removes the
classic 33-word Lucene/Snowball English stopword list. No stemming: stemming
would help a little more, but it is a modelling choice rather than a correctness
fix, and leaving it out keeps this function cheap enough for the serving path.

The stopword list is written out literally rather than imported from ``bm25s``
so that the published numbers stay reproducible even if that library changes its
list. ``bm25s`` remains a dependency of the separate ``bm25s`` index artifact in
``scripts/preprocess.py``; this module deliberately does not import it.

INVARIANT
---------
Index build and query must use this same function. A BM25 index built with one
tokenization and queried with another silently returns near-garbage, which is
exactly the class of bug this module replaced.
"""

from __future__ import annotations

import re
from typing import List

# Bump this whenever `tokenize` changes its output. A persisted BM25 index records
# it (services/shared/bm25_index.py) so an index built by one tokenizer is never
# queried through another -- that mismatch degrades retrieval silently.
#   v1 = the original `text.lower().split()`, never fingerprinted
#   v2 = word-boundary split + the 33-word English stopword list (current)
TOKENIZER_VERSION = "v2-word-stop"

# The classic Lucene / Snowball English stopword list (33 words). Written out
# explicitly: see the module docstring.
STOPWORDS_EN = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "if",
        "in",
        "into",
        "is",
        "it",
        "no",
        "not",
        "of",
        "on",
        "or",
        "such",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "will",
        "with",
    }
)

# Word boundaries: runs of ASCII letters and digits. Everything else (commas,
# full stops, hyphens, brackets, quotes) is a separator.
_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Tokenize one string for BM25: case-fold, split on word boundaries, drop
    stopwords.

    Used by the index build, every query path, and every evaluation harness.
    Keeping them identical is the whole point -- do not inline a variant.

        >>> tokenize("The cells, and their nuclei.")
        ['cells', 'nuclei']
    """
    return [w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS_EN]


def tokenize_corpus(texts) -> List[List[str]]:
    """Tokenize an iterable of documents for building a BM25 index."""
    return [tokenize(t) for t in texts]
