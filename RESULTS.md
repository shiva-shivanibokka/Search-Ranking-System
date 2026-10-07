# Independent review and fixes (branch `sop-eval`)

An adversarial review of this repository found that **the BM25 baseline every
headline comparison is measured against was broken**, and that the README's
zero-shot conclusion was the opposite of what the repo's own committed results
file said. Every defect below was reproduced before being fixed, and every number
here was measured on this machine: Windows 11, Python 3.12, `torch==2.5.1+cu121`,
one CUDA GPU.

---

## 1. Every BM25 call site tokenized on whitespace

All eight BM25 call sites tokenized with `text.lower().split()` — the 1M-passage
index build in `scripts/preprocess.py`, the serving query paths in `deploy/` and
`services/`, the LambdaRank feature builder, and both evaluation harnesses.

Splitting on whitespace alone leaves punctuation attached to words, so `"cells,"`
and `"cells"` are different terms and a query term cannot match a document term
that happens to end a sentence. A direct demonstration, now a regression test:

```
corpus[0] = "The organelle studied was mitochondria."
query     = "mitochondria"
lower().split()  -> score 0.0   (the only relevant document is unreachable)
fixed            -> score > 0, ranked first
```

Because BM25 is also the lexical half of the hybrid retriever and feature 0 of the
LambdaRank ranker, this under-scored the whole system, not just the baseline.

**The fix is validated against an external reference, not against itself.**
Replacing the tokenizer with word-boundary splitting plus the standard 33-word
English stopword list moves BM25 to within **0.0001–0.020 of the published BEIR
BM25 baseline** on all three datasets — FiQA lands at 0.2361 against a published
0.236:

| nDCG@10 | `lower().split()` | fixed | published BEIR BM25 |
|---|---|---|---|
| SciFact | 0.5597 | **0.6618** | ~0.665 |
| NFCorpus | 0.2668 | **0.3049** | ~0.325 |
| FiQA-2018 | 0.1591 | **0.2361** | ~0.236 |

The reference column is BM25 (Anserini) nDCG@10 as reported in the BEIR paper —
Thakur, Reimers, Rücklé, Srivastava & Gurevych, *BEIR: A Heterogenous Benchmark
for Zero-shot Evaluation of Information Retrieval Models*, NeurIPS 2021 Datasets
and Benchmarks ([arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663)).
An exact match is not expected: that baseline uses Anserini/Lucene with stemming,
while this one is `rank-bm25` with `k1`/`b` left at library defaults.

That third column is the whole argument. Without it, raising a baseline's score
and then declaring the higher number correct would be indistinguishable from
result-shopping. With it, the conclusion is forced: the old figures were not a
BM25 baseline, they were a handicapped one, which made every comparison against
them flattering.

**Fix.** `services/shared/text.py` is now the single tokenizer; all call sites
import it. The stopword list is written out literally rather than imported from
`bm25s`, so the published numbers stay reproducible if that library changes its
list.

Two guard tests enforce the invariant repo-wide, because the failure mode is
silent — an index built with one tokenization and queried with another returns
near-garbage without raising anything:

- `test_no_call_site_tokenizes_with_lower_split` fails if any file under
  `services/`, `training/`, `deploy/`, `scripts/` or `gradio_app/` re-inlines a
  tokenizer. Against the pre-fix tree it names all 14 offending lines.
- `test_no_call_site_unpickles_the_bm25_index_directly` fails if the index is
  loaded without the fingerprint check below.

## 2. A stale index could be loaded silently, including from the published artifacts

Changing the tokenizer created a second-order hazard worth more than the first.
Two routes could hand a process an index built by the *old* tokenizer:

- `scripts/preprocess.py` reused any existing `bm25_index.pkl` unconditionally —
  it checked only that the file existed; and
- `scripts/bootstrap.py` pulls prebuilt artifacts from Hugging Face Hub, and
  those were built before this fix.

Either way the quickstart would produce a system quietly worse than before the
fix, with no error anywhere.

**Fix.** `services/shared/bm25_index.py` writes a tokenizer fingerprint beside the
index and verifies it on load. A mismatch raises `StaleIndexError` with rebuild
instructions; a missing sidecar also raises, because an unfingerprinted index is
by definition the old tokenization. All six load sites go through it.

## 3. MAP@10 used a divisor no published BEIR number uses

`ap_at_k` divided by `min(|gold|, k)`. `pytrec_eval` — and therefore every BEIR
MAP@10 in a paper or on the leaderboard — divides by `|gold|`. Both conventions
exist, but they are not interchangeable, and these numbers are presented next to
published ones.

Invisible where queries have ~1 relevant document (MS MARCO dev, SciFact, FiQA),
large where they do not. On NFCorpus, which averages **38.2** relevant documents
per query, with the tokenizer and BM25 parameters held fixed so that only the
divisor differs:

```
MAP@10 / min(|gold|, 10)  = 0.2200   <- old
MAP@10 / |gold|           = 0.1178   <- pytrec_eval / BEIR
```

an **87% overstatement** against any published figure.

## 4. The zero-shot conclusion was inverted — and was already contradicted by the repo's own file

The README's TL;DR claimed *"hybrid retrieval beats BM25 zero-shot on BEIR"*, and
§15 claimed *"the strongest zero-shot configuration in every case is Hybrid(RRF):
it beats BM25's Recall@100 on all three datasets"*.

Regenerated with the fixed tokenizer:

| | BM25 nDCG@10 | Hybrid nDCG@10 | BM25 R@100 | Hybrid R@100 |
|---|---|---|---|---|
| SciFact | **0.6618** | 0.3421 | 0.8859 | **0.9024** |
| NFCorpus | **0.3049** | 0.2517 | 0.2384 | **0.2468** |
| FiQA-2018 | **0.2361** | 0.1583 | **0.4891** | 0.4824 |

- On **nDCG@10**, which this section itself names as the headline BEIR metric,
  **BM25 wins all three**, by a wide margin.
- On **Recall@100**, Hybrid wins **two of three** and loses FiQA.

So the "every case" claim is wrong on both metrics. Note what this is *not*: the
tokenizer bug did not create the error. In the **old** numbers BM25 already led
SciFact on nDCG@10 **0.5597 vs 0.3018**. The claim was contradicted by the
repository's own committed results file, two paragraphs above an "Honest
interpretation" section that correctly stated the hybrid trails BM25 on nDCG@10.
Both could not be true.

The corrected section reports it as what it is: a negative result. Fusing a dense
retriever that is near-useless out of domain (nDCG@10 0.03–0.11) costs a lot of
precision to buy a little deep recall, and on FiQA it does not buy even that.

The dense retriever's zero-shot range was separately misstated as "~0.01-0.04";
measured it is 0.03–0.11.

## 5. The in-domain headline conclusion did not survive fixing the baseline

With the tokenizer fixed and LambdaRank retrained on the corrected features, the
§14 table was regenerated on the same 200-query dev sample (changing the sample
size as well would have confounded the comparison):

| Configuration | NDCG@10 before | NDCG@10 now |
|---|---|---|
| BM25 (keyword baseline) | 0.337 | **0.446** |
| Two-Tower (dense) | 0.327 | 0.327 |
| Hybrid — BM25 + dense, RRF | 0.463 | **0.518** |
| Hybrid + LambdaRank | 0.400 | **0.506** |
| Hybrid + CrossEncoder | 0.505 | 0.509 |

Three claims change:

1. **"~50% over the BM25 keyword baseline" becomes +16%.** The best configuration
   now beats BM25 0.446 → 0.518. Almost the entire previously-claimed margin was
   the handicapped baseline rather than the pipeline.
2. **The best configuration is no longer the CrossEncoder — it is the
   un-reranked hybrid** (0.518 vs 0.509). Both rerankers now land marginally
   *below* plain RRF fusion on NDCG@10. The CrossEncoder keeps a real advantage on
   Recall@10 (0.715 vs 0.697), which is the metric a user feels, and it is the
   cheaper reranker here (p50 286 ms vs LambdaRank's 1,571 ms).
3. **"LambdaRank underperforms the hybrid" was a feature bug, not a modelling
   result.** It scored 0.400 against the hybrid's 0.463, and the README documented
   that gap as an honest limitation of GBDT reranking with binary labels.
   Retrained on the corrected `bm25_score` and term-overlap features it scores
   **0.506** — level with the cross-encoder.

Retraining LambdaRank was not optional. Features 0 and 3 both derive from the
tokenizer, so scoring the new features with the old model would have produced
exactly the train/serve skew this repository advertises having found and fixed.
Retrained: train nDCG@10 0.8333, dev nDCG@10 0.6814.

## 6. MLflow logging had never worked, while the module claimed it did

`training/evaluate.py` documents "All metrics logged to MLflow". It logged none.
MLflow permits only alphanumerics, `_`, `-`, `.`, space and `/` in a metric name,
and **both halves** of every name this module builds are invalid — `NDCG@10` has an
`@`, `Hybrid(RRF)` has parentheses. So `mlflow.log_metric("Hybrid(RRF)/NDCG@10",
...)` raised `MlflowException` on the very first call.

The failure mode is what made it survive: the exception escaped `run_evaluation()`
*after* `eval_results.json` had already been written a few lines earlier. The
script therefore produced correct, complete results and then exited non-zero,
which reads like a crashed evaluation rather than a logging bug — and the results
file looked fine, so nobody chased it.

**Fix.** `mlflow_metric_name()` encodes `@` as `_at_` and replaces anything else
outside MLflow's allowed set. Verified by logging all 49 metrics from the real
summary and reading them back (`BM25/NDCG_at_10 = 0.4460656…`). Two tests pin it,
one asserting every name the evaluation can produce passes MLflow's own validator.

## 7. Latency was overstated ~100×, and the stated cause was wrong

The README said the pipeline *"returns results in tens of milliseconds on GPU"*,
and the limitations section attributed the real-world slowness to Cloud Run being
CPU-only.

The repo's own committed `eval_results.json` is produced on the training machine
**with CUDA available**, and recorded — before any change of mine — Two-Tower p50
**18.3 ms**, BM25 **2,628.9 ms**, Hybrid **2,616.9 ms**. So the claim was already
contradicted by the committed file, and CPU-only hosting was never the explanation.

After the tokenizer fix, re-measured on the same machine:

| configuration | p50 before | p50 now |
|---|---|---|
| Two-Tower (dense only) | 18.3 ms | **6 ms** |
| BM25 | 2,628.9 ms | **1,036 ms** |
| Hybrid(RRF) | 2,616.9 ms | **1,031 ms** |

The BM25 halving is a side effect, not an optimisation: dropping stopwords from
the index shrank it from 460 MB to 349 MB, so there is less to scan.

The dense arm genuinely is single-digit milliseconds. Everything involving BM25 is
~1 second **on the GPU machine**. The bottleneck is `rank-bm25`, a pure-Python
implementation that scores all ~1M documents per query, and no GPU accelerates
that.

The roadmap entry now points at the real fix: **`bm25s`, which this repository
already depends on and already uses for hard-negative mining**
(`scripts/preprocess.py`), where it is 100–500× faster per query. It would fix the
latency and the memory bullet together.

## 8. The live demo is down, and the frontend said "waking up" forever

| claim | measured |
|---|---|
| "🔗 Live demo", "deployed live", "both rerankers serving live" | the Cloud Run API returns **503** on `/`, `/health` and `/docs` |

Each 503 arrives in ~0.2 s. That is an unavailable service, not a cold start — a
cold start hangs and then succeeds. The Vercel frontend itself still loads (200).

Worse, the frontend could not tell the difference. `hstatus` was set to `null`
both when the health check had not run yet and when it had failed, so the status
strip rendered *"API waking up — first call cold-starts (~1–2 min)"* in both
cases. Against a backend that is actually down, that message never stops being
shown, and it tells the visitor to keep waiting for something that is not coming.

**Fix.** The check is now tri-state (`pending` / `ok` / `failed`) and a failed
check renders **"search backend offline"** with a local-run hint and a link to the
committed results. Verified by rendering the page against an absent API.

## 9. "~95% recall@100 vs exact search" was never measured

The FAISS IVF+PQ section claimed the approximate index retains "~95% recall@100
compared to exact search". Nothing in the repository measured it; it was a
plausible figure for a generic IVF+PQ configuration, printed next to a column of
genuinely measured numbers where a reader cannot tell them apart.

`scripts/eval_faiss_recall.py` is new and measures it over 1,000 answerable dev
queries at `nprobe=64`:

| | measured |
|---|---|
| Gold Recall@100, exact search | 0.7383 |
| Gold Recall@100, FAISS IVF+PQ | 0.6560 |
| **Fraction of gold recall retained** | **88.9%** |
| Set agreement@100 with exact search | **65.3%** |

Neither of the two things "~95% recall@100" could have meant is ~95%. The two rows
answer different questions and the distinction is worth keeping: set agreement is
index *fidelity*, while gold recall retained is the end-to-end cost to a user.
Approximate search drops about a third of exact search's list but keeps most of the
*relevant* documents in it.

This also surfaced a related overstatement. The headline **"in-domain Recall@100 ≈
0.74"** is `scripts/eval_recall.py`'s **exact** dot-product measurement — the
committed `two_tower_recall.json` says so in its `method` field, but the README
quoted the number without the qualifier. Through the FAISS index the system
actually serves, the same measurement gives **0.656**. Both are now stated, with
0.743 labelled a model-quality number and 0.656 a served-system number.

(The exact figure here, 0.7383 over 1,000 queries, independently reproduces the
committed 0.7428 over all 6,980 — a useful consistency check on the new script.)

## 10. The BEIR harness scored a different BM25 than the one this repo ships

`scripts/preprocess.py` has always built the 1M-passage serving index with
`k1=0.9, b=0.4`. `training/beir_eval.py` constructed its own `BM25Okapi` without
passing them, so it silently used `rank-bm25`'s library defaults, `k1=1.5,
b=0.75`.

The BEIR table therefore reported a BM25 that was neither the one this repository
serves nor the one it was being compared against — `k1=0.9, b=0.4` is also
Anserini's default, which is what the published BEIR BM25 baseline uses. Both
parameters now come from one place (`services/shared/bm25_index.py`).

## 11. The artifact manifest omitted a now-required file

`scripts/artifacts_manifest.py` lists what `publish_artifacts.py` uploads and
`bootstrap.py` downloads. The tokenizer-fingerprint sidecar added in §2 is
required for the index to load at all, so publishing the index without it would
make a fresh `bootstrap.py` + `docker-compose up` refuse to start. Added to the
manifest.

> **Action required outside this branch:** the artifacts currently published on
> Hugging Face Hub were built with the old tokenizer, and the LambdaRank model was
> trained on the old features. Both must be re-published with
> `python scripts/publish_artifacts.py` before the documented quickstart works
> again. Until then it fails loudly with `StaleIndexError` and rebuild
> instructions, which is the intended behaviour — the alternative was a silently
> degraded system.

## 12. No LICENSE file

The README has carried an MIT badge since the first commit with no `LICENSE` file
in the repository. Added, matching the sibling repositories.

## Known limitations, stated rather than fixed

- **The serving BM25 is still `rank-bm25`.** Switching to `bm25s` is the right
  fix for both latency and memory, but it changes a serving artifact format and
  belongs in its own change with its own before/after measurement.
- **`python -m training.train_lambdarank > log.txt` crashes on Windows.** `rich`
  writes a `→` that cp1252 cannot encode, and the run dies a few queries in with
  `UnicodeEncodeError`. Worked around here with `PYTHONIOENCODING=utf-8`; the
  documented command still needs it.
- **The 10 integration tests are deselected by default** (`-m 'not integration'`).
  This is deliberate and documented in `pyproject.toml` — they need a live
  `docker-compose` stack — but it does mean the advertised test count is a
  unit-test count.
- **The Cloud Run service is not redeployed.** Free-tier hosting was not kept
  running; the README now says so rather than implying a cold start.

## Verification

```
pytest       — 113 passed, 10 deselected (integration, needs a live stack)
ruff check   — All checks passed!
svelte-check — 0 errors, 0 warnings, 150 files
BM25 index   — rebuilt over 1,000,000 passages in 44s, fingerprint written,
               349 MB (was 460 MB: stopwords are no longer indexed)
LambdaRank   — retrained on corrected features, train nDCG@10 0.8333 / dev 0.6814
MLflow       — 49 metrics logged and read back (was: raised on the first call)
BEIR         — regenerated, BM25 within 0.0001–0.020 of the published baseline
FAISS recall — measured over 1,000 answerable dev queries at nprobe=64
frontend     — offline state rendered and checked against an absent API
```

Regenerated result files committed on this branch:
`data/processed/beir_results.json`, `eval_results.json`, and the new
`faiss_recall.json`.
