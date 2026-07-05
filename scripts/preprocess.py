"""
Preprocess MS MARCO into clean formats used by all downstream components.

Outputs (data/processed/):
  - passages.parquet        : pid, text, token_count
  - train_queries.parquet   : qid, text
  - dev_queries.parquet     : qid, text
  - train_qrels.parquet     : qid, pid, relevance
  - dev_qrels.parquet       : qid, pid, relevance
  - train_triples.parquet   : qid, query, pos_pid, neg_pid, pos_text, neg_text
  - hard_negatives.parquet  : qid, query, pos_pid, hard_neg_pids (from BM25 top-100)
"""

import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from rich.console import Console
from tqdm import tqdm

console = Console()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/indexes")


def load_collection(max_passages: int = -1) -> pd.DataFrame:
    """Load passage collection into a DataFrame."""
    console.print("[cyan]Loading passage collection...[/cyan]")
    path = RAW_DIR / "collection.tsv"

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Passages")):
            if max_passages > 0 and i >= max_passages:
                break
            parts = line.strip().split("\t")
            if len(parts) == 2:
                pid, text = parts
                rows.append(
                    {"pid": int(pid), "text": text, "token_count": len(text.split())}
                )

    df = pd.DataFrame(rows)
    console.print(f"[green]Loaded {len(df):,} passages[/green]")
    return df


def load_queries(split: str, max_queries: int = -1) -> pd.DataFrame:
    """Load queries for train or dev split."""
    path = RAW_DIR / f"queries.{split}.tsv"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc=f"{split} queries")):
            if max_queries > 0 and i >= max_queries:
                break
            parts = line.strip().split("\t")
            if len(parts) == 2:
                qid, text = parts
                rows.append({"qid": int(qid), "text": text})
    return pd.DataFrame(rows)


def load_qrels(split: str) -> pd.DataFrame:
    """Load relevance judgments (qid, pid, relevance=1)."""
    fname = "qrels.dev.small.tsv" if split == "dev" else "qrels.train.tsv"
    path = RAW_DIR / fname
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, pid, rel = parts[0], parts[1], parts[2], parts[3]
                rows.append({"qid": int(qid), "pid": int(pid), "relevance": int(rel)})
    return pd.DataFrame(rows)


def build_gold_inclusive_corpus(
    gold_pids: set,
    collection_path: Path = RAW_DIR / "collection.tsv",
    target_size: int = 1_000_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build the working passage corpus so every train+dev qrels gold passage is
    present, plus a reservoir-sampled distractor sample up to `target_size`
    passages total. Streams collection.tsv exactly once (Algorithm R reservoir
    sampling over the non-gold rows) so it scales to the full ~8.8M collection
    without loading it all into memory. Original MS MARCO pids are preserved
    (never renumbered), so existing qrels/triples still map correctly.

    If len(gold_pids) >= target_size, every gold pid is still kept (the
    resulting corpus may exceed target_size) — coverage is never sacrificed
    to hit the size target.
    """
    rng = np.random.default_rng(seed)
    n_distractor_slots = max(target_size - len(gold_pids), 0)

    gold_rows: dict = {}
    reservoir: list = []
    distractor_count_seen = 0

    with open(collection_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Streaming collection (gold-inclusive)"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            pid_str, text = parts
            pid = int(pid_str)

            if pid in gold_pids:
                gold_rows[pid] = text
                continue

            i = distractor_count_seen
            distractor_count_seen += 1
            if i < n_distractor_slots:
                reservoir.append((pid, text))
            else:
                j = int(rng.integers(0, i + 1))
                if j < n_distractor_slots:
                    reservoir[j] = (pid, text)

    rows = [
        {"pid": pid, "text": text, "token_count": len(text.split())}
        for pid, text in gold_rows.items()
    ] + [
        {"pid": pid, "text": text, "token_count": len(text.split())}
        for pid, text in reservoir
    ]
    df = pd.DataFrame(rows).sort_values("pid").reset_index(drop=True)
    console.print(
        f"[green]Gold-inclusive corpus: {len(gold_rows):,} gold + "
        f"{len(reservoir):,} distractors = {len(df):,} passages[/green]"
    )
    return df


def load_triples(passages_df: pd.DataFrame, max_triples: int = 500000) -> pd.DataFrame:
    """
    Load training triples (query, positive, negative).
    Triples file format: query_text\tpos_text\tneg_text (tab-separated)
    """
    console.print("[cyan]Loading training triples...[/cyan]")
    path = RAW_DIR / "triples.train.small.tsv"

    # Build pid lookup for fast joins
    pid_to_text = dict(zip(passages_df["pid"], passages_df["text"]))

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Triples", total=max_triples)):
            if i >= max_triples:
                break
            parts = line.strip().split("\t")
            if len(parts) == 3:
                query_text, pos_text, neg_text = parts
                rows.append(
                    {
                        "query": query_text,
                        "pos_text": pos_text,
                        "neg_text": neg_text,
                    }
                )

    df = pd.DataFrame(rows)
    console.print(f"[green]Loaded {len(df):,} training triples[/green]")
    return df


def build_bm25_index(passages_df: pd.DataFrame) -> BM25Okapi:
    """Build and persist a BM25 index over the passage collection."""
    console.print("[cyan]Building BM25 index...[/cyan]")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / "bm25_index.pkl"

    if index_path.exists():
        console.print("[yellow]BM25 index already exists — loading from disk[/yellow]")
        with open(index_path, "rb") as f:
            return pickle.load(f)

    tokenized_corpus = [
        text.lower().split() for text in tqdm(passages_df["text"], desc="Tokenizing")
    ]
    bm25 = BM25Okapi(tokenized_corpus, k1=0.9, b=0.4)

    with open(index_path, "wb") as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Also save pid list so we can map BM25 rank → pid
    pid_list = passages_df["pid"].tolist()
    with open(INDEX_DIR / "bm25_pid_list.pkl", "wb") as f:
        pickle.dump(pid_list, f)

    console.print(f"[green]BM25 index saved → {index_path}[/green]")
    return bm25


def build_bm25s_index(passages_df: pd.DataFrame):
    """
    Build an in-memory bm25s retriever over the passage collection, used only
    for fast hard-negative mining. bm25s is ~100-500x faster per query than
    rank-bm25 (BM25Okapi) — the format used for the separately-persisted
    *serving* BM25 index (see build_bm25_index / data/indexes/bm25_index.pkl,
    unchanged, still rank-bm25 because evaluate.py's retrieve_bm25 relies on
    its .get_scores() API).
    """
    import bm25s

    console.print("[cyan]Building bm25s index for hard-negative mining...[/cyan]")
    corpus_texts = passages_df["text"].tolist()
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=True)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)
    console.print("[green]bm25s mining index built.[/green]")
    return retriever


def mine_hard_negatives(
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    passages_df: pd.DataFrame,
    retriever,
    top_k: int = 100,
    hard_neg_per_query: int = 5,
    max_queries: int = 150000,
) -> pd.DataFrame:
    """
    Mine REAL BM25 hard negatives via bm25s: for each query, retrieve the
    top-`top_k` BM25 matches over the full corpus, drop any pid that is a
    qrels-relevant passage for that query, and keep the first
    `hard_neg_per_query` survivors. This replaces the old random-sampling
    placeholder — the model now learns relevant-vs-plausible, not
    relevant-vs-random.
    """
    import bm25s

    console.print(
        f"[cyan]Mining BM25 hard negatives for "
        f"{min(max_queries, len(queries_df)):,} queries (bm25s)...[/cyan]"
    )

    queries_with_pos = set(qrels_df["qid"].unique())
    eligible = queries_df[queries_df["qid"].isin(queries_with_pos)].head(max_queries)

    pos_pids_by_qid = qrels_df.groupby("qid")["pid"].apply(set).to_dict()
    pid_to_text = dict(zip(passages_df["pid"], passages_df["text"]))
    pid_list = passages_df["pid"].tolist()

    query_texts = eligible["text"].tolist()
    query_tokens = bm25s.tokenize(query_texts, stopwords="en", show_progress=True)
    results, _scores = retriever.retrieve(query_tokens, k=top_k, show_progress=True)

    rows = []
    for row_i, (_, row) in enumerate(
        tqdm(eligible.iterrows(), total=len(eligible), desc="Hard negs")
    ):
        qid = row["qid"]
        query_text = row["text"]
        pos_pids = pos_pids_by_qid.get(qid, set())
        if not pos_pids:
            continue
        pos_pid = next(iter(pos_pids))

        hard_negs = []
        for idx in results[row_i]:
            pid = pid_list[int(idx)]
            if pid not in pos_pids:
                hard_negs.append(pid)
            if len(hard_negs) >= hard_neg_per_query:
                break
        if not hard_negs:
            continue

        rows.append(
            {
                "qid": qid,
                "query": query_text,
                "pos_pid": pos_pid,
                "pos_text": pid_to_text.get(pos_pid, ""),
                "hard_neg_pids": hard_negs,
                "hard_neg_texts": [pid_to_text.get(p, "") for p in hard_negs],
            }
        )

    df = pd.DataFrame(rows)
    console.print(f"[green]Mined hard negatives for {len(df):,} queries[/green]")
    return df


@click.command()
@click.option(
    "--max-passages", default=500000, help="Max passages to load (-1 for all)"
)
@click.option("--max-train-queries", default=400000, help="Max train queries")
@click.option("--max-dev-queries", default=6980, help="Max dev queries")
@click.option("--max-triples", default=500000, help="Max training triples")
@click.option(
    "--skip-hard-negatives",
    is_flag=True,
    default=False,
    help="Skip hard negative mining (slow)",
)
def main(
    max_passages, max_train_queries, max_dev_queries, max_triples, skip_hard_negatives
):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # ── Passages ────────────────────────────────────────────────────────────────
    passages_path = PROCESSED_DIR / "passages.parquet"
    if passages_path.exists():
        console.print("[yellow]passages.parquet exists — loading[/yellow]")
        passages_df = pd.read_parquet(passages_path)
    else:
        passages_df = load_collection(max_passages)
        passages_df.to_parquet(passages_path, index=False)
        console.print(f"[green]Saved → {passages_path}[/green]")

    # ── Queries ─────────────────────────────────────────────────────────────────
    for split, max_q in [("train", max_train_queries), ("dev", max_dev_queries)]:
        out_path = PROCESSED_DIR / f"{split}_queries.parquet"
        if out_path.exists():
            console.print(f"[yellow]{out_path.name} exists — skipping[/yellow]")
        else:
            df = load_queries(split, max_q)
            df.to_parquet(out_path, index=False)
            console.print(f"[green]Saved → {out_path}[/green]")

    # ── QRels ───────────────────────────────────────────────────────────────────
    for split in ["train", "dev"]:
        out_path = PROCESSED_DIR / f"{split}_qrels.parquet"
        if out_path.exists():
            console.print(f"[yellow]{out_path.name} exists — skipping[/yellow]")
        else:
            df = load_qrels(split)
            df.to_parquet(out_path, index=False)
            console.print(f"[green]Saved → {out_path}[/green]")

    # ── Training Triples ────────────────────────────────────────────────────────
    triples_path = PROCESSED_DIR / "train_triples.parquet"
    if triples_path.exists():
        console.print("[yellow]train_triples.parquet exists — skipping[/yellow]")
    else:
        triples_df = load_triples(passages_df, max_triples)
        triples_df.to_parquet(triples_path, index=False)
        console.print(f"[green]Saved → {triples_path}[/green]")

    # ── BM25 Index ──────────────────────────────────────────────────────────────
    bm25 = build_bm25_index(passages_df)

    with open(INDEX_DIR / "bm25_pid_list.pkl", "rb") as f:
        pid_list = pickle.load(f)

    # ── Hard Negatives ──────────────────────────────────────────────────────────
    # NOTE: main()'s call below still uses the pre-bm25s signature and will be
    # rewired in Task 3 to build a bm25s retriever and call mine_hard_negatives
    # with the new (retriever, top_k, ...) signature.
    if not skip_hard_negatives:
        hard_neg_path = PROCESSED_DIR / "hard_negatives.parquet"
        if hard_neg_path.exists():
            console.print("[yellow]hard_negatives.parquet exists — skipping[/yellow]")
        else:
            train_queries_df = pd.read_parquet(PROCESSED_DIR / "train_queries.parquet")
            train_qrels_df = pd.read_parquet(PROCESSED_DIR / "train_qrels.parquet")
            hard_neg_df = mine_hard_negatives(
                train_queries_df, train_qrels_df, passages_df, bm25, pid_list
            )
            hard_neg_df.to_parquet(hard_neg_path, index=False)
            console.print(f"[green]Saved → {hard_neg_path}[/green]")

    console.print("\n[bold green]Preprocessing complete.[/bold green]")
    console.print("Next step: [cyan]python training/train_two_tower.py[/cyan]")


if __name__ == "__main__":
    main()
