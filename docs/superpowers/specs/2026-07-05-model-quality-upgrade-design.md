# SP-1 — Model Quality Upgrade (Two-Tower) — Design

**Status:** approved direction; spec for review before implementation planning.
**Scope of this sub-project:** retrain the two-tower dense retriever so its
measured retrieval quality is genuinely good and honestly reportable. The
CrossEncoder reranker, the serving API/hosting (SP-2), and the SvelteKit
frontend + RAG (SP-3) are **out of scope here** and get their own specs.

## Goal

Turn the two-tower retriever from "demonstrably functional but lightly trained"
(answerable-only Recall@100 ≈ 0.32; naive dev recall ≈ 0.005 due to corpus
coverage) into a retriever with **real, defensible in-domain recall
(target Recall@100 ≈ 0.7–0.85)** on a corpus that actually contains the
evaluation's relevant passages — and update every committed number to match.

## Background / root causes (verified)

1. **The working corpus does not contain the gold passages.** `scripts/preprocess.py`
   selects the first 500K passages by pid (0–499,999) as `passages.parquet`, but
   MS MARCO dev qrels reference passages spread across the full ~8.8M collection.
   Only **149 / 7,433 (2.0%)** of dev gold passages are in the subset, so ~98% of
   dev queries are *unanswerable* and the model was never trained on most passages
   it is evaluated against. (Confirmed: `data/indexes/docid_map.pkl` holds pids
   0–499,999; `dev_qrels ∩ indexed = 149`.)
2. **Random negatives, not hard negatives.** `mine_hard_negatives` uses
   `rng.choice` (random) — the model only learned relevant-vs-random, never
   relevant-vs-plausible. The honest reason it was skipped: per-query BM25 over
   500K passages with `rank-bm25` (pure Python) was ~12h for 50K queries.
3. **No recall-based checkpointing.** `training/train_two_tower.py` never calls its
   own `evaluate_recall`; `model_best.pt` is just the lowest-training-loss epoch.

The model itself works: a query embeds **0.67** cosine to its relevant passage vs
**0.01** to an irrelevant one. The problem is corpus construction + training
regime, not the architecture.

## Approach

### A. Gold-inclusive corpus (~1M passages)

Rebuild the working corpus so every train+dev qrels gold passage is present:

- `corpus_pids = set(train_qrels.pid) ∪ set(dev_qrels.pid) ∪ random_sample(all
  collection pids, to reach ~1,000,000 total)`.
- Stream `data/raw/collection.tsv` (8.8M rows), keep rows whose pid ∈ `corpus_pids`,
  write the new `data/processed/passages.parquet`. **Keep original MS MARCO pids**
  (so existing qrels/triples still map; nothing is re-numbered).
- Size rationale: ~1M passages embeds in ~20–40 min on the RTX 4060 and the
  FAISS IVF+PQ index + in-RAM text stay within a Cloud-Run-friendly footprint.
  Full 8.8M is deferred (overnight-scale, larger serving RAM) and noted as a
  future option, not this sub-project.

### B. Real BM25 hard-negative mining (made feasible)

- Add **`bm25s`** (a fast, pure-Python-but-vectorized BM25; ~100–500× faster than
  `rank-bm25`) as the mining backend. Build a BM25 index over the ~1M corpus once.
- For each of ~100–200K train queries: BM25 top-K (e.g. K=100), drop any pid in
  that query's qrels, take the top `hard_negatives_per_query` (default 5) as hard
  negatives. Write `data/processed/hard_negatives.parquet` (real hard negs this
  time). This makes README §5's "BM25 hard negatives" description *true*.
- (Optional, flagged for the plan, not required: a second pass of **dense
  self-mined** negatives using the freshly trained model, à la ANCE. Deferred
  unless the BM25-only result underperforms.)

### C. Retrain the two-tower

- Config changes (`configs/config.yaml` / `configs/training_config.py`):
  `hard_negatives_per_query: 5`, `batch_size: 64` (tune down if 8 GB VRAM OOMs —
  effective in-batch negatives scale with batch size; gradient accumulation is the
  fallback), `max_train_queries: ~100–200K`, `epochs: 3–5`.
- **Wire up recall-based checkpointing (with a small eval index to avoid a
  chicken-and-egg):** the doc tower changes every epoch, so re-embedding the full
  ~1M corpus per epoch is too slow — which is exactly why the original code skipped
  mid-training eval. Instead, build a **small fixed eval corpus once** =
  *all dev gold passages ∪ a capped distractor sample (≤ ~100K passages)*, and each
  epoch re-embed only that (~1–3 min on the 4060), brute-force search it, and
  compute Recall@10 over the dev queries. Select `model_best.pt` on best eval
  Recall@10. This gives a real, non-misleading recall signal cheaply. Keep the
  honest `best_train_loss` logging too, but selection is by recall now. The full
  ~1M FAISS index is built **once at the end** (step below) for the final,
  headline evaluation.
- Rebuild doc embeddings + FAISS IVF+PQ index (`training/build_faiss_index.py`)
  on the new corpus with the new doc tower.

### D. Re-measure and re-report honestly

- Re-run `scripts/eval_recall.py` (now meaningful — gold passages are in the
  index) → real Recall@10/Recall@100; commit to `two_tower_recall.json`.
- Re-run BEIR (`scripts/eval_beir.py`) — the better encoder should lift zero-shot
  numbers too; re-commit `beir_results.json`.
- Update the README numbers (§11, §14, and the BEIR section) to the new measured
  values, replacing the coverage-artifact framing with the real results (keeping
  the honest methodology notes).
- Republish artifacts to HF Hub (`scripts/publish_artifacts.py`) so the demo/API
  and any fresh clone pull the improved model + index.

## Components / files touched

| File | Change |
|---|---|
| `scripts/preprocess.py` | Gold-inclusive corpus selection; real BM25 hard-neg mining via `bm25s`. |
| `requirements.txt` | Add `bm25s` (+ its `scipy` dep already present). |
| `configs/config.yaml`, `configs/training_config.py` | `max_passages`→gold-inclusive logic; `hard_negatives_per_query`, `batch_size`, `max_train_queries`, `epochs`. |
| `training/train_two_tower.py` | Call `evaluate_recall` per epoch; select `model_best.pt` by Recall@10. |
| `training/build_faiss_index.py` | Rebuild on the new corpus (no logic change expected). |
| `scripts/eval_recall.py`, `scripts/eval_beir.py` | Re-run to regenerate committed JSON (no code change expected). |
| `README.md` | Replace recall/BEIR numbers with the new real measurements. |
| `scripts/publish_artifacts.py` | Republish improved artifacts to HF Hub. |
| `data/processed/*`, `data/indexes/*`, `data/embeddings/*`, `models/two_tower/*` | Regenerated (gitignored; distributed via HF Hub). |

## Data flow

`collection.tsv + qrels → gold-inclusive passages.parquet → bm25s index →
hard_negatives.parquet → two-tower training (InfoNCE, in-batch + hard negs,
recall-checkpointed) → model_best.pt → doc embeddings → FAISS index →
eval_recall + BEIR → committed JSON + README + HF Hub`.

## Validation / testing

- **Unit:** the corpus builder includes 100% of qrels gold pids (assert on a
  synthetic mini-collection); hard-neg miner excludes qrels-relevant pids and
  returns the requested count; recall-checkpoint selection picks the higher-recall
  epoch given stubbed scores.
- **Integration / acceptance:** after retraining, `eval_recall.py` reports
  answerable Recall@100 materially above the current 0.32 (success target ≈
  0.7–0.85; anything ≥ ~0.6 is a real win). Gold coverage in the new index is
  ~100% of dev qrels. The probe (query-vs-gold ≫ query-vs-random) still holds.
- **Honesty gate:** every README number after this work is regenerated from a
  committed JSON produced by a real run — no hand-edited values.

## Risks / tradeoffs

- **VRAM (8 GB):** batch 64 with 5 hard negs may OOM; fall back to batch 32 +
  gradient accumulation. The plan will make batch size configurable and test small.
- **Environment instability:** this machine segfaults/hangs when torch + TF + faiss
  + large arrays mix; training runs must set `USE_TF=0` and avoid loading faiss and
  the model in the same throwaway process where possible.
- **Recall target not hit:** if BM25 hard negs alone don't reach ~0.7, the dense
  self-mining pass (deferred in B) is the next lever; the spec does not *promise*
  a specific number, only a real, measured, materially-improved one.
- **Time:** ~an afternoon/overnight of mostly-unattended GPU compute.

## Out of scope (explicit)

CrossEncoder training (follow-up sub-project), the serving API + Cloud Run hosting
(SP-2), the SvelteKit frontend + client-side RAG/BYOK (SP-3), and indexing the
full 8.8M collection.

## Deliverables

A retrained, recall-checkpointed `model_best.pt`; a gold-inclusive ~1M corpus +
rebuilt FAISS index + doc embeddings; real committed `two_tower_recall.json` and
`beir_results.json`; README numbers updated to the real measurements; artifacts
republished to HF Hub.
