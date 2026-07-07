# SP-1: Two-Tower Model Quality Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the two-tower retriever from "functional but lightly trained" (answerable-only Recall@100 ≈ 0.32 on a corpus missing 98% of dev gold passages) into a retriever with real, defensible in-domain recall (target Recall@100 ≈ 0.7–0.85) on a gold-inclusive ~1M-passage corpus, and update every committed number to match.

**Architecture:** Rebuild `data/processed/passages.parquet` so it contains every train+dev qrels gold passage plus a reservoir-sampled distractor pool (~1M total, original MS MARCO pids preserved); mine real BM25 hard negatives with `bm25s` over that corpus; retrain the existing `TwoTowerModel` (unchanged architecture) with a small fixed eval index so recall-based checkpointing is cheap per epoch; rebuild the FAISS index on the new corpus; re-measure and re-publish honestly.

**Tech Stack:** PyTorch, HuggingFace `transformers`, `bm25s` (new), FAISS (`faiss-cpu`), MLflow, pandas/pyarrow, pytest, HF Hub.

## Global Constraints

- **Preserve original MS MARCO pids.** The corpus rebuild must never renumber passages — existing qrels/triples reference original pids and must keep resolving.
- **Environment instability (this Windows/anaconda machine):** torch + tensorflow + faiss + large numpy arrays loaded together can SEGFAULT/HANG. Every python invocation in this plan that imports torch/transformers (tests, training, embedding, eval) MUST set env `USE_TF=0 TRANSFORMERS_NO_TF=1` first, to stop `transformers` from importing TensorFlow. Long-running steps (corpus rebuild, hard-neg mining at scale, training, FAISS embedding, eval_recall, eval_beir) MUST run as **background jobs**, not a single blocking foreground call — poll/monitor instead of blocking the session.
- **Device selection is not fully trustworthy.** `torch.cuda.is_available()` can report `True` while the device is actually unusable in this environment. If a long-running job fails at CUDA init despite `is_available()` returning `True`, the documented fallback is to force CPU for a smoke test via `CUDA_VISIBLE_DEVICES=` (empty) before re-diagnosing GPU state — do not silently retry in a loop.
- **VRAM budget: RTX 4060 Laptop, 8GB.** `batch_size: 64` with `hard_negatives_per_query: 5` may OOM. Batch size must stay configurable via `configs/config.yaml`; the documented fallback is to drop `batch_size` to `32` and rerun (accepting fewer effective in-batch negatives) — do this before attempting anything more invasive.
- **Recall-based checkpointing must use a small eval index**, not the full ~1M corpus, because the doc tower changes every epoch and re-embedding ~1M passages per epoch is too slow. The small index = all dev qrels gold passages ∪ a capped random distractor sample (≤ ~100K), rebuilt once before training starts.
- **Honesty gate:** every number written into `README.md` after this work must be copied verbatim from a committed JSON file (`data/processed/two_tower_recall.json`, `data/processed/beir_results.json`) produced by an actual run in this environment — never hand-typed or estimated.
- **Lint:** `ruff check` (rules `E`, `F`, `I`) must be clean on every file this plan touches, respecting the existing `pyproject.toml` per-file-ignores (`scripts/preprocess.py` and `training/train_two_tower.py` already ignore `F841`).
- **Commits go to `main`. No AI attribution in any commit message** (no "Co-Authored-By: Claude", no mention of Claude/Anthropic) — this is a portfolio repo.
- Unit tests must never require the real 1M corpus, real training, or a GPU. Use tiny synthetic data and small/fake models only. The real artifacts are produced by the RUN tasks.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `scripts/preprocess.py` | Modify | Add `build_gold_inclusive_corpus` (gold-inclusive ~1M corpus, reservoir-sampled distractors); add `build_bm25s_index` + rewrite `mine_hard_negatives` to do real BM25 mining via `bm25s`; reorder `main()` so qrels load before passages and both new functions are wired in with config-driven CLI defaults. |
| `requirements.txt` | Modify | Add `bm25s` (fast BM25 mining backend). |
| `tests/unit/test_preprocess.py` | Create | Unit tests for `build_gold_inclusive_corpus` and `mine_hard_negatives`/`build_bm25s_index` on tiny synthetic data. |
| `configs/config.yaml` | Modify | `data:` section gets `target_corpus_size`, `hard_neg_max_queries`, `hard_negatives_top_k`, `max_triples`; `two_tower:` section gets `hard_negatives_per_query: 5`, `batch_size: 64`, `epochs: 4`, `eval_max_distractors: 100000`. |
| `configs/training_config.py` | Modify | New `DataConfig` dataclass; `TwoTowerConfig` gets `eval_max_distractors` field; `get_training_config` wires `data:` section into `TrainingConfig.data`. |
| `tests/unit/test_training_config.py` | Create | Unit test asserting `get_training_config()` returns the new config values. |
| `training/train_two_tower.py` | Modify | Add `build_eval_corpus` (dev gold ∪ capped distractors) and `select_best_epoch` (pure function); wire both into `train()` so `model_best.pt` is selected by per-epoch Recall@10 on the small eval index, not training loss. |
| `tests/unit/test_two_tower.py` | Modify | Add tests for `build_eval_corpus` and `select_best_epoch` (stubbed data, no real training). |
| `training/build_faiss_index.py` | No code change | Rerun as-is against the new corpus + retrained model (RUN task). |
| `scripts/eval_recall.py`, `scripts/eval_beir.py` | No code change | Rerun as-is to regenerate committed JSON (RUN task). |
| `data/processed/passages.parquet`, `hard_negatives.parquet`, `two_tower_recall.json`, `beir_results.json` | Regenerated | New gold-inclusive corpus, real hard negatives, real measured recall/BEIR numbers (gitignored parquet/artifacts; JSON committed). |
| `models/two_tower/*`, `data/embeddings/*`, `data/indexes/*` | Regenerated | Retrained model, doc embeddings, FAISS index, rebuilt BM25 serving index (gitignored; distributed via HF Hub). |
| `README.md` | Modify | §11 (MLflow tracking), §14 (evaluation results) and the BEIR section updated to the new real measurements. |

---

## Task 1: Gold-inclusive corpus builder

**Files:**
- Modify: `scripts/preprocess.py`
- Test: `tests/unit/test_preprocess.py` (create)

**Interfaces:**
- Consumes: nothing new (stdlib, `numpy`, `pandas`, `tqdm`, `rich.console` already imported in `scripts/preprocess.py`).
- Produces: `build_gold_inclusive_corpus(gold_pids: set[int], collection_path: Path = RAW_DIR / "collection.tsv", target_size: int = 1_000_000, seed: int = 42) -> pd.DataFrame` with columns `pid, text, token_count`. Later tasks (Task 3's `main()` rewiring) call this with `gold_pids = set(train_qrels.pid) | set(dev_qrels.pid)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_preprocess.py`:

```python
"""Unit tests for scripts/preprocess.py's corpus + hard-negative mining."""

import pandas as pd


def test_build_gold_inclusive_corpus_includes_all_gold_and_hits_target(tmp_path):
    from scripts.preprocess import build_gold_inclusive_corpus

    collection_path = tmp_path / "collection.tsv"
    lines = [f"{pid}\tpassage text number {pid}" for pid in range(20)]
    collection_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    gold_pids = {3, 17}
    df = build_gold_inclusive_corpus(
        gold_pids, collection_path=collection_path, target_size=8, seed=42
    )

    assert gold_pids.issubset(set(df["pid"]))
    assert len(df) == 8
    assert len(df["pid"].unique()) == 8
    # Original MS MARCO pids are preserved verbatim, never renumbered.
    assert df[df["pid"] == 3]["text"].iloc[0] == "passage text number 3"


def test_build_gold_inclusive_corpus_never_drops_gold_even_if_target_too_small(tmp_path):
    from scripts.preprocess import build_gold_inclusive_corpus

    collection_path = tmp_path / "collection.tsv"
    lines = [f"{pid}\tpassage text number {pid}" for pid in range(20)]
    collection_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    gold_pids = set(range(10))  # 10 gold pids, but target_size below that
    df = build_gold_inclusive_corpus(
        gold_pids, collection_path=collection_path, target_size=5, seed=42
    )

    assert gold_pids.issubset(set(df["pid"]))
    assert len(df) == 10  # coverage is never sacrificed to hit the size target
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `USE_TF=0 TRANSFORMERS_NO_TF=1 python -m pytest tests/unit/test_preprocess.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_gold_inclusive_corpus'`

- [ ] **Step 3: Implement `build_gold_inclusive_corpus`**

In `scripts/preprocess.py`, add this function directly after `load_qrels` (after line 79):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `USE_TF=0 TRANSFORMERS_NO_TF=1 python -m pytest tests/unit/test_preprocess.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Lint and commit**

Run: `ruff check scripts/preprocess.py tests/unit/test_preprocess.py`
Expected: no errors

```bash
git add scripts/preprocess.py tests/unit/test_preprocess.py
git commit -m "preprocess: add gold-inclusive corpus builder"
```

---

## Task 2: Fast BM25 hard-negative mining via bm25s

**Files:**
- Modify: `scripts/preprocess.py`, `requirements.txt`
- Test: `tests/unit/test_preprocess.py` (extend)

**Interfaces:**
- Consumes: `bm25s` (new dependency).
- Produces: `build_bm25s_index(passages_df: pd.DataFrame) -> bm25s.BM25` (in-memory retriever, not persisted to disk — this is separate from the existing rank-bm25-based serving index at `data/indexes/bm25_index.pkl`, built by the unchanged `build_bm25_index`); rewritten `mine_hard_negatives(queries_df: pd.DataFrame, qrels_df: pd.DataFrame, passages_df: pd.DataFrame, retriever, top_k: int = 100, hard_neg_per_query: int = 5, max_queries: int = 150000) -> pd.DataFrame` with columns `qid, query, pos_pid, pos_text, hard_neg_pids, hard_neg_texts` — same output schema as before, real BM25 hard negatives instead of random ones. Task 3's `main()` rewiring calls both.

- [ ] **Step 1: Add the dependency**

In `requirements.txt`, under the `# ── Search & Ranking ──` section (after the `rank-bm25==0.2.2` line), add:

```
bm25s==0.2.6                   # fast vectorized BM25 for hard-neg mining (~100-500x faster than rank-bm25 per query)
```

Run: `pip install bm25s==0.2.6`
Expected: installs cleanly (pulls in `scipy`, already pinned above; `numpy`, already pinned).

- [ ] **Step 2: Write the failing test**

Append to `tests/unit/test_preprocess.py`:

```python
def test_mine_hard_negatives_excludes_gold_and_respects_count():
    from scripts.preprocess import build_bm25s_index, mine_hard_negatives

    passages_df = pd.DataFrame(
        {
            "pid": [1, 2, 3, 4, 5, 6],
            "text": [
                "cats are small mammals that purr",
                "dogs are loyal mammals that bark",
                "cats and dogs are common household pets",
                "the stock market rose sharply today",
                "cats love to chase small mice at night",
                "quantum physics describes subatomic particles",
            ],
        }
    )
    queries_df = pd.DataFrame({"qid": [100], "text": ["what mammals are cats"]})
    qrels_df = pd.DataFrame({"qid": [100], "pid": [1], "relevance": [1]})

    retriever = build_bm25s_index(passages_df)
    result = mine_hard_negatives(
        queries_df, qrels_df, passages_df, retriever,
        top_k=5, hard_neg_per_query=2, max_queries=10,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["qid"] == 100
    assert row["pos_pid"] == 1
    hard_negs = row["hard_neg_pids"]
    assert len(hard_negs) == 2
    assert 1 not in hard_negs  # the positive pid must never appear as a hard negative
    assert len(row["hard_neg_texts"]) == 2


def test_mine_hard_negatives_skips_queries_without_qrels():
    from scripts.preprocess import build_bm25s_index, mine_hard_negatives

    passages_df = pd.DataFrame(
        {"pid": [1, 2, 3], "text": ["alpha text", "beta text", "gamma text"]}
    )
    queries_df = pd.DataFrame(
        {"qid": [1, 2], "text": ["alpha query", "no qrels for this one"]}
    )
    qrels_df = pd.DataFrame({"qid": [1], "pid": [1], "relevance": [1]})

    retriever = build_bm25s_index(passages_df)
    result = mine_hard_negatives(
        queries_df, qrels_df, passages_df, retriever,
        top_k=3, hard_neg_per_query=1, max_queries=10,
    )

    assert len(result) == 1
    assert result.iloc[0]["qid"] == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `USE_TF=0 TRANSFORMERS_NO_TF=1 python -m pytest tests/unit/test_preprocess.py -v -k mine_hard_negatives`
Expected: FAIL with `ImportError: cannot import name 'build_bm25s_index'`

- [ ] **Step 4: Implement `build_bm25s_index` and rewrite `mine_hard_negatives`**

In `scripts/preprocess.py`, replace the entire existing `mine_hard_negatives` function (original lines 142–213) with:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `USE_TF=0 TRANSFORMERS_NO_TF=1 python -m pytest tests/unit/test_preprocess.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Lint and commit**

Run: `ruff check scripts/preprocess.py tests/unit/test_preprocess.py requirements.txt`
Expected: no errors (note: `scripts/preprocess.py` has an existing `F841` per-file-ignore in `pyproject.toml`)

```bash
git add scripts/preprocess.py tests/unit/test_preprocess.py requirements.txt
git commit -m "preprocess: mine real BM25 hard negatives with bm25s"
```

---

## Task 3: Config updates + main() rewiring

**Files:**
- Modify: `configs/config.yaml`, `configs/training_config.py`, `scripts/preprocess.py`
- Test: `tests/unit/test_training_config.py` (create)

**Interfaces:**
- Consumes: `TwoTowerConfig`, `TrainingConfig`, `get_training_config` (existing, `configs/training_config.py`); `build_gold_inclusive_corpus` (Task 1), `build_bm25s_index`/`mine_hard_negatives` (Task 2).
- Produces: `DataConfig` dataclass with fields `target_corpus_size: int = 1_000_000`, `max_train_queries: int = 400_000`, `max_dev_queries: int = 6980`, `max_triples: int = 500_000`, `hard_neg_max_queries: int = 150_000`, `hard_negatives_top_k: int = 100`; `TrainingConfig.data: DataConfig`; `TwoTowerConfig.eval_max_distractors: int = 100_000` — consumed by Task 4's `build_eval_corpus` wiring in `training/train_two_tower.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_training_config.py`:

```python
"""Unit tests for configs/training_config.py's new SP-1 fields."""

from configs.training_config import DataConfig, get_training_config


def test_get_training_config_reads_data_section():
    cfg = get_training_config("configs/config.yaml")
    assert cfg.data.target_corpus_size == 1_000_000
    assert cfg.data.hard_neg_max_queries == 150_000
    assert cfg.data.hard_negatives_top_k == 100
    assert cfg.data.max_train_queries == 400_000
    assert cfg.data.max_dev_queries == 6980


def test_get_training_config_reads_updated_two_tower_section():
    cfg = get_training_config("configs/config.yaml")
    assert cfg.two_tower.hard_negatives_per_query == 5
    assert cfg.two_tower.batch_size == 64
    assert cfg.two_tower.epochs == 4
    assert cfg.two_tower.eval_max_distractors == 100_000


def test_data_config_defaults_without_yaml():
    cfg = DataConfig()
    assert cfg.target_corpus_size == 1_000_000
    assert cfg.hard_neg_max_queries == 150_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `USE_TF=0 TRANSFORMERS_NO_TF=1 python -m pytest tests/unit/test_training_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'DataConfig'`

- [ ] **Step 3: Add `DataConfig` and wire it into `TrainingConfig`**

In `configs/training_config.py`, add this dataclass directly after the `TwoTowerConfig` class (after line 26, before `CrossEncoderConfig`):

```python
@dataclass
class DataConfig:
    target_corpus_size: int = 1_000_000
    max_train_queries: int = 400_000
    max_dev_queries: int = 6980
    max_triples: int = 500_000
    hard_neg_max_queries: int = 150_000
    hard_negatives_top_k: int = 100
```

Add `eval_max_distractors: int = 100_000` as the last field of `TwoTowerConfig` (after `save_dir: str = "models/two_tower"`, line 26):

```python
@dataclass
class TwoTowerConfig:
    model_name: str = "distilbert-base-uncased"
    embedding_dim: int = 768
    projection_dim: int = 256
    temperature: float = 0.05
    hard_negatives_per_query: int = 5
    batch_size: int = 64
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_steps: int = 1000
    max_seq_len_query: int = 64
    max_seq_len_doc: int = 180
    save_dir: str = "models/two_tower"
    eval_max_distractors: int = 100_000
```

Add `data: DataConfig = field(default_factory=DataConfig)` as a field of `TrainingConfig` (in the `TrainingConfig` dataclass, alongside `two_tower`):

```python
@dataclass
class TrainingConfig:
    two_tower: TwoTowerConfig = field(default_factory=TwoTowerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cross_encoder: CrossEncoderConfig = field(default_factory=CrossEncoderConfig)
    lambdarank: LambdaRankConfig = field(default_factory=LambdaRankConfig)
    faiss: FAISSConfig = field(default_factory=FAISSConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hybrid_retrieval: HybridRetrievalConfig = field(
        default_factory=HybridRetrievalConfig
    )
    difficulty_classifier: DifficultyClassifierConfig = field(
        default_factory=DifficultyClassifierConfig
    )
```

In `get_training_config`, add the `data:` section wiring (after the `tt = raw.get("two_tower", {})` block, before `ce = raw.get(...)`):

```python
    tt = raw.get("two_tower", {})
    cfg.two_tower = TwoTowerConfig(**_filter(TwoTowerConfig, tt))

    dt = raw.get("data", {})
    cfg.data = DataConfig(**_filter(DataConfig, dt))

    ce = raw.get("cross_encoder", {})
```

- [ ] **Step 4: Update `configs/config.yaml`**

Replace the `data:` block's subsample comment and fields (original lines 17–20):

```yaml
  # Gold-inclusive corpus target size (~1M): every train+dev qrels gold pid is
  # kept, plus a random distractor sample up to this many passages total.
  # See scripts/preprocess.py::build_gold_inclusive_corpus.
  target_corpus_size: 1000000
  max_train_queries: 400000
  max_dev_queries: 6980
  max_triples: 500000
  # BM25 (bm25s) hard-negative mining, over the full target_corpus_size corpus.
  hard_neg_max_queries: 150000
  hard_negatives_top_k: 100
```

Replace the `two_tower:` block (original lines 22–34):

```yaml
two_tower:
  model_name: "distilbert-base-uncased"
  embedding_dim: 768
  projection_dim: 256
  temperature: 0.05           # InfoNCE temperature
  hard_negatives_per_query: 5  # real BM25 hard negs now (bm25s) — was 1 (random)
  batch_size: 64               # was 32; if CUDA OOM on 8GB VRAM, drop to 32 and rerun
  learning_rate: 2.0e-5
  epochs: 4                    # was 3; target range 3-5
  warmup_steps: 1000
  max_seq_len_query: 64
  max_seq_len_doc: 180
  save_dir: models/two_tower
  eval_max_distractors: 100000  # per-epoch recall-checkpoint eval index cap
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `USE_TF=0 TRANSFORMERS_NO_TF=1 python -m pytest tests/unit/test_training_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Rewire `scripts/preprocess.py`'s `main()` to use the new builders and config-driven defaults**

Add these imports near the top of `scripts/preprocess.py` (after the existing imports, before `console = Console()`):

```python
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.training_config import get_training_config  # noqa: E402

_data_cfg = get_training_config().data
```

Replace the entire `@click.command()` decorator block and `main()` function (original lines 216–299) with:

```python
@click.command()
@click.option(
    "--target-corpus-size",
    default=_data_cfg.target_corpus_size,
    help="Gold-inclusive corpus target size (-1 for the full collection)",
)
@click.option(
    "--max-train-queries", default=_data_cfg.max_train_queries, help="Max train queries"
)
@click.option(
    "--max-dev-queries", default=_data_cfg.max_dev_queries, help="Max dev queries"
)
@click.option("--max-triples", default=_data_cfg.max_triples, help="Max training triples")
@click.option(
    "--hard-neg-max-queries",
    default=_data_cfg.hard_neg_max_queries,
    help="Max queries to mine BM25 hard negatives for",
)
@click.option(
    "--hard-neg-top-k",
    default=_data_cfg.hard_negatives_top_k,
    help="BM25 top-K candidates per query before hard-negative filtering",
)
@click.option(
    "--skip-hard-negatives",
    is_flag=True,
    default=False,
    help="Skip hard negative mining (slow)",
)
def main(
    target_corpus_size,
    max_train_queries,
    max_dev_queries,
    max_triples,
    hard_neg_max_queries,
    hard_neg_top_k,
    skip_hard_negatives,
):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # ── QRels (loaded first: gold pids drive gold-inclusive corpus selection) ──
    for split in ["train", "dev"]:
        out_path = PROCESSED_DIR / f"{split}_qrels.parquet"
        if out_path.exists():
            console.print(f"[yellow]{out_path.name} exists — skipping[/yellow]")
        else:
            df = load_qrels(split)
            df.to_parquet(out_path, index=False)
            console.print(f"[green]Saved → {out_path}[/green]")

    train_qrels_df = pd.read_parquet(PROCESSED_DIR / "train_qrels.parquet")
    dev_qrels_df = pd.read_parquet(PROCESSED_DIR / "dev_qrels.parquet")
    gold_pids = set(train_qrels_df["pid"]) | set(dev_qrels_df["pid"])

    # ── Passages (gold-inclusive) ───────────────────────────────────────────────
    passages_path = PROCESSED_DIR / "passages.parquet"
    if passages_path.exists():
        console.print("[yellow]passages.parquet exists — loading[/yellow]")
        passages_df = pd.read_parquet(passages_path)
    else:
        passages_df = build_gold_inclusive_corpus(
            gold_pids, target_size=target_corpus_size
        )
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

    # ── Training Triples (fallback dataset if hard negatives are skipped) ───────
    triples_path = PROCESSED_DIR / "train_triples.parquet"
    if triples_path.exists():
        console.print("[yellow]train_triples.parquet exists — skipping[/yellow]")
    else:
        triples_df = load_triples(passages_df, max_triples)
        triples_df.to_parquet(triples_path, index=False)
        console.print(f"[green]Saved → {triples_path}[/green]")

    # ── BM25 serving index (rank-bm25 — unchanged format/consumers) ────────────
    bm25 = build_bm25_index(passages_df)

    # ── Hard Negatives (bm25s mining index — separate, in-memory only) ─────────
    if not skip_hard_negatives:
        hard_neg_path = PROCESSED_DIR / "hard_negatives.parquet"
        if hard_neg_path.exists():
            console.print("[yellow]hard_negatives.parquet exists — skipping[/yellow]")
        else:
            train_queries_df = pd.read_parquet(PROCESSED_DIR / "train_queries.parquet")
            mining_index = build_bm25s_index(passages_df)
            hard_neg_df = mine_hard_negatives(
                train_queries_df,
                train_qrels_df,
                passages_df,
                mining_index,
                top_k=hard_neg_top_k,
                max_queries=hard_neg_max_queries,
            )
            hard_neg_df.to_parquet(hard_neg_path, index=False)
            console.print(f"[green]Saved → {hard_neg_path}[/green]")

    console.print("\n[bold green]Preprocessing complete.[/bold green]")
    console.print("Next step: [cyan]python training/train_two_tower.py[/cyan]")


if __name__ == "__main__":
    main()
```

Note: this drops the old `bm25_pid_list` re-load block (`with open(INDEX_DIR / "bm25_pid_list.pkl", "rb") as f: pid_list = pickle.load(f)`) since `mine_hard_negatives` no longer needs a separate pid list — it derives `pid_list` from `passages_df` internally (Task 2). `build_bm25_index` still writes `bm25_pid_list.pkl` to disk for the serving stack; that is unaffected.

- [ ] **Step 7: Run the full preprocess test file plus a quick import sanity check**

Run: `USE_TF=0 TRANSFORMERS_NO_TF=1 python -m pytest tests/unit/test_preprocess.py tests/unit/test_training_config.py -v`
Expected: PASS (7 tests total)

Run: `USE_TF=0 TRANSFORMERS_NO_TF=1 python -c "import scripts.preprocess"`
Expected: no exceptions (confirms `main()`'s click decorators and config import resolve)

- [ ] **Step 8: Lint and commit**

Run: `ruff check scripts/preprocess.py configs/training_config.py configs/config.yaml tests/unit/test_training_config.py`
Expected: no errors

```bash
git add scripts/preprocess.py configs/training_config.py configs/config.yaml tests/unit/test_training_config.py
git commit -m "config: gold-inclusive corpus + hard-neg mining settings, wire into preprocess.py"
```

---

## Task 4: Recall-based checkpointing with a small eval index

**Files:**
- Modify: `training/train_two_tower.py`
- Test: `tests/unit/test_two_tower.py` (extend)

**Interfaces:**
- Consumes: `evaluate_recall` (existing, `training/train_two_tower.py`, unchanged signature); `TwoTowerConfig.eval_max_distractors` (Task 3).
- Produces: `build_eval_corpus(passages_df: pd.DataFrame, dev_qrels_df: pd.DataFrame, max_distractors: int = 100_000, seed: int = 42) -> pd.DataFrame`; `select_best_epoch(recall_at_10_by_epoch: dict[int, float]) -> int`. Both are called from `train()`'s epoch loop — no other file consumes them.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_two_tower.py`:

```python
import pandas as pd


def test_build_eval_corpus_includes_all_dev_gold_and_caps_distractors():
    from training.train_two_tower import build_eval_corpus

    passages_df = pd.DataFrame(
        {"pid": list(range(20)), "text": [f"passage {i}" for i in range(20)]}
    )
    dev_qrels_df = pd.DataFrame(
        {"qid": [1, 1, 2], "pid": [3, 7, 15], "relevance": [1, 1, 1]}
    )

    eval_df = build_eval_corpus(passages_df, dev_qrels_df, max_distractors=5, seed=42)

    assert {3, 7, 15}.issubset(set(eval_df["pid"]))
    assert len(eval_df) == 3 + 5
    assert len(eval_df["pid"].unique()) == len(eval_df)


def test_build_eval_corpus_caps_at_available_passages_if_fewer_than_max():
    from training.train_two_tower import build_eval_corpus

    passages_df = pd.DataFrame({"pid": list(range(6)), "text": [f"p{i}" for i in range(6)]})
    dev_qrels_df = pd.DataFrame({"qid": [1], "pid": [0], "relevance": [1]})

    eval_df = build_eval_corpus(passages_df, dev_qrels_df, max_distractors=100, seed=42)

    assert len(eval_df) == 6  # only 6 passages exist total, can't exceed that


def test_select_best_epoch_picks_highest_recall():
    from training.train_two_tower import select_best_epoch

    assert select_best_epoch({1: 0.10, 2: 0.35, 3: 0.28}) == 2


def test_select_best_epoch_tie_break_is_earliest_epoch():
    from training.train_two_tower import select_best_epoch

    assert select_best_epoch({1: 0.30, 2: 0.30}) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `USE_TF=0 TRANSFORMERS_NO_TF=1 python -m pytest tests/unit/test_two_tower.py -v -k "eval_corpus or select_best_epoch"`
Expected: FAIL with `ImportError: cannot import name 'build_eval_corpus'`

- [ ] **Step 3: Implement `build_eval_corpus` and `select_best_epoch`**

In `training/train_two_tower.py`, add both functions directly after the `evaluate_recall` function (after line 236, before the `# ── LR Scheduler ──` section):

```python
def build_eval_corpus(
    passages_df: pd.DataFrame,
    dev_qrels_df: pd.DataFrame,
    max_distractors: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Small, fixed eval corpus for per-epoch recall checkpointing: every dev
    qrels gold passage, plus a capped random distractor sample. Built ONCE
    before training starts (the doc tower changes every epoch, so re-embedding
    the full ~1M corpus per epoch would be too slow — re-embedding this small,
    fixed corpus is cheap, ~1-3 min per epoch on an RTX 4060).
    """
    gold_pids = set(dev_qrels_df["pid"])
    gold_df = passages_df[passages_df["pid"].isin(gold_pids)]
    distractor_pool = passages_df[~passages_df["pid"].isin(gold_pids)]
    n = min(max_distractors, len(distractor_pool))
    distractor_df = distractor_pool.sample(n=n, random_state=seed)
    return pd.concat([gold_df, distractor_df], ignore_index=True)


def select_best_epoch(recall_at_10_by_epoch: dict) -> int:
    """
    Return the epoch (1-indexed) with the highest eval Recall@10. Ties break
    to the earliest epoch (a smaller/earlier-converged model, all else equal).
    """
    return max(recall_at_10_by_epoch, key=lambda e: (recall_at_10_by_epoch[e], -e))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `USE_TF=0 TRANSFORMERS_NO_TF=1 python -m pytest tests/unit/test_two_tower.py -v`
Expected: PASS (all tests, including the 4 new ones plus the 5 pre-existing)

- [ ] **Step 5: Wire both functions into `train()`'s epoch loop**

In `training/train_two_tower.py`, replace the block from `global_step = 0` through the end of the epoch `for` loop (original lines 326–389) with:

```python
        global_step = 0
        # Honest training-loss tracking (kept for logging/debugging), but
        # checkpoint SELECTION is now by Recall@10 on the small, fixed eval
        # index built below — not by training loss. Re-embedding the full
        # ~1M corpus every epoch would be too slow, which is exactly why the
        # original code skipped mid-training eval; the small eval index makes
        # per-epoch recall evaluation cheap (~1-3 min/epoch).
        best_train_loss = 0.0
        recall_history: dict = {}

        eval_corpus_df = build_eval_corpus(
            passages_df, dev_qrels_df, max_distractors=tt_cfg.eval_max_distractors
        )
        console.print(
            f"[cyan]Recall-checkpoint eval index: {len(eval_corpus_df):,} passages "
            f"(all dev gold + up to {tt_cfg.eval_max_distractors:,} distractors)[/cyan]"
        )

        for epoch in range(tt_cfg.epochs):
            model.train()
            epoch_loss = 0.0

            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{tt_cfg.epochs}")
            for batch in pbar:
                optimizer.zero_grad()

                loss = model(
                    query_input_ids=batch["query_input_ids"].to(DEVICE),
                    query_attention_mask=batch["query_attention_mask"].to(DEVICE),
                    pos_input_ids=batch["pos_input_ids"].to(DEVICE),
                    pos_attention_mask=batch["pos_attention_mask"].to(DEVICE),
                    hard_neg_input_ids=batch["hard_neg_input_ids"].to(DEVICE),
                    hard_neg_attention_mask=batch["hard_neg_attention_mask"].to(DEVICE),
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

                if global_step % 500 == 0:
                    mlflow.log_metric("train_loss", loss.item(), step=global_step)

            avg_loss = epoch_loss / len(loader)
            console.print(f"\n[bold]Epoch {epoch + 1} avg loss: {avg_loss:.4f}[/bold]")
            mlflow.log_metric("epoch_avg_loss", avg_loss, step=epoch)
            if best_train_loss == 0.0 or avg_loss < best_train_loss:
                best_train_loss = avg_loss

            torch.save(model.state_dict(), save_dir / f"model_epoch{epoch + 1}.pt")
            console.print(
                f"  [green]Checkpoint saved → model_epoch{epoch + 1}.pt[/green]"
            )

            recall_metrics = evaluate_recall(
                model, tokenizer, eval_corpus_df, dev_queries_df, dev_qrels_df,
                k_values=[10, 100],
            )
            epoch_recall_at_10 = recall_metrics.get("Recall_at_10", 0.0)
            recall_history[epoch + 1] = epoch_recall_at_10
            mlflow.log_metric("eval_recall_at_10", epoch_recall_at_10, step=epoch)
            mlflow.log_metric(
                "eval_recall_at_100", recall_metrics.get("Recall_at_100", 0.0), step=epoch
            )
            console.print(
                f"  [bold]Epoch {epoch + 1} eval Recall@10 (small index): "
                f"{epoch_recall_at_10:.4f}[/bold]"
            )

            if select_best_epoch(recall_history) == epoch + 1:
                torch.save(model.state_dict(), save_dir / "model_best.pt")
                console.print(
                    f"  [green]New best Recall@10: {epoch_recall_at_10:.4f} "
                    f"— saved as model_best.pt[/green]"
                )
```

Then replace the final `mlflow.log_metric("best_train_loss", best_train_loss)` line and the closing console print (original lines 404–412) with:

```python
        mlflow.log_artifacts(str(save_dir), artifact_path="two_tower_model")
        # Training-loss metric kept for honesty/debugging — NOT what selects
        # model_best.pt. Selection is by eval_recall_at_10 (see epoch loop).
        mlflow.log_metric("best_train_loss", best_train_loss)
        best_epoch = select_best_epoch(recall_history) if recall_history else None
        best_recall_at_10 = recall_history.get(best_epoch, 0.0) if best_epoch else 0.0
        mlflow.log_metric("best_recall_at_10", best_recall_at_10)

        console.print(
            f"\n[bold green]Training complete. Best train loss: {best_train_loss:.4f} | "
            f"Best eval Recall@10: {best_recall_at_10:.4f} (epoch {best_epoch})[/bold green]"
        )
        console.print(f"Model saved → {save_dir}")
        console.print("Next step: [cyan]python training/build_faiss_index.py[/cyan]")
```

- [ ] **Step 6: Re-run the full unit test suite for this file to confirm nothing broke**

Run: `USE_TF=0 TRANSFORMERS_NO_TF=1 python -m pytest tests/unit/test_two_tower.py -v`
Expected: PASS (all tests — the epoch-loop rewrite is not itself unit tested, matching the existing convention that `train()`/`main()` functions in this repo are integration-level, not unit-tested; `build_eval_corpus` and `select_best_epoch` carry the real test coverage)

- [ ] **Step 7: Lint and commit**

Run: `ruff check training/train_two_tower.py tests/unit/test_two_tower.py`
Expected: no errors (note: `training/train_two_tower.py` has an existing `E402, F841` per-file-ignore in `pyproject.toml`)

```bash
git add training/train_two_tower.py tests/unit/test_two_tower.py
git commit -m "train_two_tower: recall-based checkpointing via a small eval index"
```

---

## Task 5 (RUN): Rebuild the ~1M gold-inclusive corpus + mine hard negatives at scale

This is a long-running, non-unit-tested execution step. It produces the real artifacts the CODE tasks above only prepared the machinery for.

**Pre-conditions:** Tasks 1–4 committed. `data/raw/collection.tsv` (8.8M lines), `data/raw/qrels.train.tsv`, `data/raw/qrels.dev.small.tsv`, `data/raw/queries.train.tsv`, `data/raw/queries.dev.tsv`, `data/raw/triples.train.small.tsv` all present (already true per repo state).

- [ ] **Step 1: Remove stale artifacts from the old 500K-subset run**

The old `passages.parquet` (pids 0–499,999) and the old random-negative `hard_negatives.parquet` must be deleted — `preprocess.py`'s `main()` skips any step whose output file already exists.

Run:
```powershell
Remove-Item data/processed/passages.parquet -Force -ErrorAction SilentlyContinue
Remove-Item data/processed/hard_negatives.parquet -Force -ErrorAction SilentlyContinue
Remove-Item data/processed/train_triples.parquet -Force -ErrorAction SilentlyContinue
Remove-Item data/indexes/bm25_index.pkl -Force -ErrorAction SilentlyContinue
Remove-Item data/indexes/bm25_pid_list.pkl -Force -ErrorAction SilentlyContinue
```
Expected: files removed (or already absent). Do **not** delete `train_qrels.parquet` / `dev_qrels.parquet` / `train_queries.parquet` / `dev_queries.parquet` — they're unaffected by the corpus change and are needed as-is for gold-pid computation.

- [ ] **Step 2: Kick off the corpus rebuild + hard-negative mining as a background job**

Run (background — this streams 8.8M lines then mines hard negatives over ~150K queries; expect on the order of 30–90 minutes):
```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python scripts/preprocess.py > logs/preprocess_sp1.log 2>&1
```
Launch this via the Bash tool with `run_in_background: true` (or `Start-Process` if using PowerShell) rather than blocking the session. Monitor `logs/preprocess_sp1.log` for progress (`tqdm` bars print periodically) and for the final line `Preprocessing complete.`

- [ ] **Step 3: Verify artifacts and acceptance criteria**

Once the job finishes, run:
```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python -c "
import pandas as pd
passages = pd.read_parquet('data/processed/passages.parquet')
dev_qrels = pd.read_parquet('data/processed/dev_qrels.parquet')
train_qrels = pd.read_parquet('data/processed/train_qrels.parquet')
gold = set(dev_qrels['pid']) | set(train_qrels['pid'])
indexed = set(passages['pid'])
coverage = len(gold & indexed) / len(gold)
print(f'corpus size: {len(passages):,}')
print(f'gold coverage: {coverage:.4f}')
hard_negs = pd.read_parquet('data/processed/hard_negatives.parquet')
print(f'hard_negatives rows: {len(hard_negs):,}')
print(f'sample hard_neg_pids: {hard_negs.iloc[0][\"hard_neg_pids\"]}')
"
```
Expected / acceptance:
- `corpus size` ≈ 1,000,000 (or `len(gold)` if that exceeds the target — see Task 1's edge case).
- `gold coverage` == `1.0000` (100% — up from 2.0%). This is the headline fix; if it is not ~1.0, stop and debug before proceeding (check that `qrels.train.tsv`/`qrels.dev.small.tsv` pids were unioned correctly in Task 3's `main()`).
- `hard_negatives rows` ≈ 100,000–150,000 (bounded by `hard_neg_max_queries` and how many train queries have qrels).
- Confirm `data/indexes/bm25_index.pkl` and `data/indexes/bm25_pid_list.pkl` were rebuilt (check file mtimes are newer than the job start time) — this is the serving BM25 index, rebuilt automatically over the new corpus by the unchanged `build_bm25_index` call.

- [ ] **Step 4: Commit config/log-adjacent changes only (no large artifacts)**

The parquet/pkl artifacts are gitignored (per `artifacts_manifest.py`'s "distributed via HF Hub, not committed" design) — nothing to `git add` from this step beyond an optional trimmed log excerpt if you want a record. No commit required here unless you choose to note the run in a changelog.

---

## Task 6 (RUN): Retrain the two-tower model

**Pre-conditions:** Task 5 complete and verified (gold coverage ~100%).

- [ ] **Step 1: Quick smoke test before committing to a multi-hour run**

Run a short foreground check that the model, tokenizer, and one training batch work end-to-end without crashing (validates config wiring + VRAM headroom at `batch_size: 64` before launching the full job):
```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python -c "
import torch
from transformers import AutoTokenizer
from configs.training_config import get_training_config
from training.two_tower_model import TwoTowerModel

cfg = get_training_config().two_tower
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
model = TwoTowerModel(cfg.model_name, cfg.embedding_dim, cfg.projection_dim, cfg.temperature).to(device)
q = tokenizer(['test query'] * cfg.batch_size, max_length=64, padding='max_length', truncation=True, return_tensors='pt')
d = tokenizer(['test document'] * cfg.batch_size, max_length=180, padding='max_length', truncation=True, return_tensors='pt')
loss = model(q['input_ids'].to(device), q['attention_mask'].to(device), d['input_ids'].to(device), d['attention_mask'].to(device))
loss.backward()
print('OK, device:', device, 'loss:', loss.item())
"
```
Expected: prints `OK, device: cuda loss: <some positive float>` with no `CUDA out of memory` error.

**If this OOMs:** edit `configs/config.yaml`'s `two_tower.batch_size` from `64` to `32`, rerun this smoke test, and confirm it passes before continuing. This is the documented VRAM fallback (accepting fewer effective in-batch negatives), not a code change.

**If CUDA init itself fails** despite `torch.cuda.is_available()` returning `True` (the known environment quirk): rerun with `CUDA_VISIBLE_DEVICES=` set to empty to force CPU and confirm the smoke test still runs (much slower, but validates correctness); then investigate the GPU/driver state separately before attempting the full GPU training run.

- [ ] **Step 2: Launch full training as a background job**

Run (background — with the new ~1M corpus, ~100-150K hard-neg training samples, batch 64, 4 epochs, expect several hours on an RTX 4060):
```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python training/train_two_tower.py > logs/train_two_tower_sp1.log 2>&1
```
Launch via the Bash tool with `run_in_background: true`. Do not block the session waiting on it.

- [ ] **Step 3: Monitor progress**

Periodically check `logs/train_two_tower_sp1.log` (or `Get-Content -Tail 50 -Wait`) for:
- Per-epoch `avg loss` decreasing across epochs.
- Per-epoch `eval Recall@10 (small index)` printed at the end of each epoch — this should generally trend upward; the epoch with the highest value gets saved as `model_best.pt` (confirmed by the `New best Recall@10 ... saved as model_best.pt` log line).
- No `CUDA out of memory` or traceback. If one appears, stop the job, apply the batch-size-32 fallback from Step 1, and relaunch.

- [ ] **Step 4: Verify acceptance criteria once training completes**

Check:
```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python -c "
import json
from pathlib import Path
print('model_best.pt exists:', Path('models/two_tower/model_best.pt').exists())
print('model_final.pt exists:', Path('models/two_tower/model_final.pt').exists())
print('config.json:', json.loads(Path('models/two_tower/config.json').read_text()))
"
```
Expected / acceptance:
- `models/two_tower/model_best.pt` exists and was written during this run (check mtime).
- The training log shows `eval_recall_at_10` was logged every epoch and increased at least once beyond epoch 1 (evidence the recall-checkpointing loop actually ran, not just training-loss tracking).
- MLflow run under experiment `neural-search-ranking` (`mlflow ui --backend-store-uri mlruns` to inspect) shows `eval_recall_at_10`, `eval_recall_at_100`, `best_recall_at_10`, and `best_train_loss` metrics all logged.

No commit needed (model weights are gitignored, distributed via HF Hub in Task 9).

---

## Task 7 (RUN): Rebuild the FAISS index on the new corpus

**Pre-conditions:** Task 6 complete (`models/two_tower/model_best.pt` is the newly retrained model).

- [ ] **Step 1: Remove stale embeddings/index from the old 500K run**

```powershell
Remove-Item data/embeddings/doc_embeddings.npy -Force -ErrorAction SilentlyContinue
Remove-Item data/indexes/faiss_ivfpq.index -Force -ErrorAction SilentlyContinue
Remove-Item data/indexes/docid_map.pkl -Force -ErrorAction SilentlyContinue
```
(`training/build_faiss_index.py` skips re-embedding/re-training if these already exist.)

- [ ] **Step 2: Run the rebuild as a background job**

```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python training/build_faiss_index.py > logs/build_faiss_index_sp1.log 2>&1
```
Launch via the Bash tool with `run_in_background: true`. Expect ~20–40 minutes to embed ~1M passages on the RTX 4060, plus IVF+PQ training/indexing time.

- [ ] **Step 3: Verify acceptance criteria**

```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python -c "
import pickle
import numpy as np
emb = np.load('data/embeddings/doc_embeddings.npy')
with open('data/indexes/docid_map.pkl', 'rb') as f:
    pid_list = pickle.load(f)
import pandas as pd
dev_qrels = pd.read_parquet('data/processed/dev_qrels.parquet')
gold = set(dev_qrels['pid'])
coverage = len(gold & set(pid_list)) / len(gold)
print('embeddings shape:', emb.shape)
print('docid_map size:', len(pid_list))
print('dev gold coverage in new index:', coverage)
"
```
Expected / acceptance:
- `embeddings shape` == `(N, 256)` where `N` ≈ corpus size from Task 5.
- `docid_map size` == `N`, matching embeddings.
- `dev gold coverage in new index` == `1.0` (this is the number that must jump from the old 2.0%).
- The build log's "Sanity check" section (test query `"what is information retrieval"`) prints 5 plausible results with FAISS scores — a quick eyeball check the index isn't degenerate.

No commit needed (index/embeddings are gitignored, distributed via HF Hub in Task 9).

---

## Task 8 (RUN): Re-measure recall and BEIR honestly; regenerate committed JSON

**Pre-conditions:** Task 7 complete.

- [ ] **Step 1: Re-run `scripts/eval_recall.py`**

```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python scripts/eval_recall.py > logs/eval_recall_sp1.log 2>&1
```
Launch via the Bash tool with `run_in_background: true` if it runs long; this evaluates 6,980 dev queries by exact dot-product search over the ~1M committed doc embeddings, so allow a few minutes.

Verify:
```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python -c "
import json
r = json.loads(open('data/processed/two_tower_recall.json').read())
print(json.dumps(r, indent=2))
"
```
Expected / acceptance (per the design spec's success target):
- `dev_gold_coverage` == `1.0` (or very close — some gold pids can legitimately be duplicates/absent from the raw collection; must not resemble the old `0.02`).
- `answerable_queries` ≈ 6,980 (up from 146) since coverage is now ~100%.
- `recall_at_100_answerable` materially above the old `0.321` — target **0.6–0.85**; anything ≥ ~0.6 counts as a real win per the spec. If it lands well below 0.6, do not fabricate a better number — report the real one and flag the dense self-mining follow-up (BEIR/design spec's noted next lever) as future work.
- `recall_at_100_naive_all_queries` should now be close to `recall_at_100_answerable` (since coverage ≈ 1.0, the naive/answerable gap that existed at 2% coverage collapses).

- [ ] **Step 2: Re-run `scripts/eval_beir.py`**

```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python scripts/eval_beir.py > logs/eval_beir_sp1.log 2>&1
```
CPU-only, small corpora (SciFact/NFCorpus/FiQA-2018) — should complete in minutes; still launch with `run_in_background: true` out of caution per the Global Constraints.

Verify:
```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python -c "
import json
r = json.loads(open('data/processed/beir_results.json').read())
for name, ds in r['datasets'].items():
    print(name, ds['configs']['TwoTower'])
"
```
Expected: new nDCG@10/Recall@100 numbers per dataset for the `TwoTower` and `Hybrid(RRF)` configs — the better in-domain encoder is expected to lift these off the old ~0.01–0.04 floor, though BEIR is still zero-shot/out-of-domain so don't expect BM25-beating numbers necessarily; report whatever is actually measured.

- [ ] **Step 3: Commit the regenerated committed JSON files**

```bash
git add data/processed/two_tower_recall.json data/processed/beir_results.json
git commit -m "eval: re-measure two-tower recall and BEIR on the gold-inclusive corpus"
```

---

## Task 9 (RUN): Update README from regenerated JSON; republish artifacts to HF Hub

**Pre-conditions:** Task 8 complete and committed.

- [ ] **Step 1: Update README §11 (MLflow tracking section)**

Open `README.md` around line 911 (`**Important — checkpoint selection is by loss, not recall:**`). Replace that paragraph and the table at lines 913–920 with text reflecting:
- Checkpoint selection is now by Recall@10 on the small eval index (not training loss) — cite `training/train_two_tower.py`'s `select_best_epoch`/`build_eval_corpus`.
- The corpus is now gold-inclusive (~1M passages, coverage ~100%) instead of the old 500K/2.0%-coverage subset.
- Pull the actual `dev_gold_coverage`, `recall_at_10_answerable`, `recall_at_100_answerable`, `recall_at_100_naive_all_queries` values verbatim from the just-regenerated `data/processed/two_tower_recall.json` (Task 8, Step 1) — do not estimate or round beyond what's in the JSON.

- [ ] **Step 2: Update README §14 (Evaluation results table + BEIR section)**

Around line 1015–1022, update the `Two-Tower (neural retrieval)` row's `Recall@10`/`Recall@100` cells (and the accompanying footnote text describing "146 answerable queries" / "~0.005 naive") to the new measured values and new answerable-query count from Task 8's JSON. Around lines 1052–1069, replace the BEIR results table with the values from the regenerated `data/processed/beir_results.json`, and rewrite the "Honest interpretation" paragraph to match whatever was actually measured (do not keep the old "collapses to ~0.01–0.04" language if the new numbers differ).

- [ ] **Step 3: Verify every changed number traces to committed JSON**

```bash
USE_TF=0 TRANSFORMERS_NO_TF=1 python -c "
import json, re
recall = json.loads(open('data/processed/two_tower_recall.json').read())
beir = json.loads(open('data/processed/beir_results.json').read())
readme = open('README.md', encoding='utf-8').read()
# Spot-check: the recall_at_100_answerable value (to 4 decimals) must appear in README.
val = f'{recall[\"recall_at_100_answerable\"]:.3f}'
print('recall value in README:', val in readme)
"
```
Expected: `True`. This is the honesty-gate check — if a number in the README doesn't trace back to the JSON, fix the README before committing.

- [ ] **Step 4: Republish artifacts to HF Hub**

```bash
export HF_ARTIFACTS_REPO=<your-hf-username>/search-ranking-artifacts   # match the repo already in use
USE_TF=0 TRANSFORMERS_NO_TF=1 python scripts/publish_artifacts.py
```
Expected output: `Uploaded 12, skipped 0.` (all `SERVING_ARTIFACTS` from `scripts/artifacts_manifest.py` present locally after Tasks 5–7: `models/two_tower/model_best.pt` + tokenizer files, `models/lambdarank/*` (untouched by this plan, republished as-is), `data/indexes/faiss_ivfpq.index`, `data/indexes/docid_map.pkl`, `data/indexes/bm25_index.pkl`, `data/indexes/bm25_pid_list.pkl`, `data/processed/passages.parquet`).

- [ ] **Step 5: Commit the README update**

```bash
git add README.md
git commit -m "readme: update recall and BEIR numbers to the gold-inclusive corpus results"
```

---

## Self-Review

**1. Spec coverage.**
- Root cause 1 (corpus coverage bug) → Task 1 (`build_gold_inclusive_corpus`) + Task 5 (RUN rebuild) + Task 8's coverage acceptance check.
- Root cause 2 (random negatives, not hard) → Task 2 (`bm25s` mining) + Task 5 (RUN mining at scale).
- Root cause 3 (no recall-based checkpointing) → Task 4 (`build_eval_corpus`/`select_best_epoch` wired into `train()`) + Task 6 (RUN retrain, verified via log/MLflow).
- Config changes (`hard_negatives_per_query: 5`, `batch_size: 64`, `epochs: 3–5`, `max_train_queries`) → Task 3.
- FAISS rebuild → Task 7. Re-measure + re-report honestly → Task 8 (JSON) + Task 9 (README). Republish to HF Hub → Task 9, Step 4.
- VRAM/OOM fallback → Global Constraints + Task 6, Step 1. Environment instability (`USE_TF=0`, background jobs, device-selection caveat) → Global Constraints + every RUN task's commands. Dense self-mining (explicitly deferred in the spec) → correctly not implemented; referenced only as a documented follow-up in Task 8's acceptance note.
- Out-of-scope items (CrossEncoder training, serving API/SP-2, frontend/SP-3, full 8.8M indexing) → correctly untouched by every task above.

**2. Placeholder scan.** Every CODE step (Tasks 1–4) has real, complete function bodies and real assertions — no `TODO`/"add appropriate handling"/"similar to Task N" placeholders. Every RUN step (Tasks 5–9) has exact shell commands, exact expected output/acceptance values, and explicit failure-handling instructions (OOM fallback, coverage-below-100% debug pointer, honesty-gate check).

**3. Type/name consistency.** `build_gold_inclusive_corpus(gold_pids, collection_path, target_size, seed) -> pd.DataFrame` (Task 1) is called identically in Task 3's `main()` rewiring. `build_bm25s_index(passages_df) -> retriever` and `mine_hard_negatives(queries_df, qrels_df, passages_df, retriever, top_k, hard_neg_per_query, max_queries) -> pd.DataFrame` (Task 2) match their Task 3 call sites exactly (`mining_index = build_bm25s_index(passages_df)`; `mine_hard_negatives(train_queries_df, train_qrels_df, passages_df, mining_index, top_k=hard_neg_top_k, max_queries=hard_neg_max_queries)`). `DataConfig` fields (`target_corpus_size`, `max_train_queries`, `max_dev_queries`, `max_triples`, `hard_neg_max_queries`, `hard_negatives_top_k`) match the CLI option defaults wired in Task 3, Step 6, and the yaml keys in Task 3, Step 4. `TwoTowerConfig.eval_max_distractors` (Task 3) matches the `tt_cfg.eval_max_distractors` reference in Task 4's `train()` rewiring. `build_eval_corpus(passages_df, dev_qrels_df, max_distractors, seed) -> pd.DataFrame` and `select_best_epoch(recall_at_10_by_epoch: dict) -> int` (Task 4) are used with matching names/types in the `train()` epoch loop rewrite in the same task.
