# Production-Readiness Audit — Search-Ranking-System

_Audit date: 2026-07-03. Scope: full repo. No code was changed to produce this report._

## TL;DR

This is an ambitious, largely well-built system (5 FastAPI microservices, two-tower +
FAISS + BM25 + RRF + LambdaRank + CrossEncoder, consolidated demo, Alembic, CI, ADRs,
MLflow/Airflow/Prometheus/Grafana). The core ML math that matters (InfoNCE loss, RRF
fusion, the offline eval harness) is genuinely correct.

The problems are three distinct things — only one is "missing features":

1. **README-vs-code honesty gaps** — biggest risk for an interview piece.
2. **Correctness bugs** — would produce wrong results in production.
3. **A dead deployment path** — DEPLOY.md targets Hugging Face Spaces (exhausted).

Two claims from the initial pass were **disproven on verification** and are NOT problems:
- `faiss-cpu==1.13.2` is a real, installable PyPI version (not a build blocker).
- Models/data **do** exist on disk (2.6 GB models + 3.8 GB data): training genuinely ran.
  They're gitignored (correct) and distributed via HF Hub.

---

## 🔴 Critical

### Interview-honesty gaps (README claims the code contradicts)
- **Fabricated recall numbers.** README shows per-epoch `Recall@10/100` tables and
  `best_recall_at_10=0.48`. `train_two_tower.py:371` sets `best_recall_at_10 = avg_loss`
  (tracks *loss*), logs it to MLflow as `best_Recall_at_10` (`:392`), and the epoch loop
  skips recall eval entirely (`:362-364`). The numbers cannot have come from this code.
- **"BM25 hard-negative mining" is random sampling.** README §5 calls it "the most
  important preprocessing step" with a worked BM25 example. `preprocess.py:153` docstring
  literally says "using random sampling"; `:160-163` admits in-batch/random negatives.
  The *code comment is honest* — the README is the liability.
- **No evidence for the headline eval table.** No `eval_results.json` / `mlruns/` exists
  anywhere on disk. The `NDCG 0.184 → 0.312` table (`README.md:971-976`) has no saved
  backing artifact; it reads as literature targets presented as measured results.

### Correctness bugs
- **LambdaRank train/serve feature skew.** `bm25_score` is a hand-rolled term-frequency
  proxy at train time (`train_lambdarank.py:174-184`) but real `bm25.get_scores()` at serve
  (`evaluate.py:218-232`); `bm25_rank` means three different things across train/eval/retrain;
  `two_tower_cosine_sim` is FAISS-approximate in training but exact at eval. The model is fed
  inputs it was not trained on.
- **Degenerate click-retraining.** `retrain_from_clicks.py:102` and
  `retraining_dag.py:214` set all labels to `1`. With `rank:ndcg` and no intra-group label
  variation there is no ranking signal. The free-tier path `_publish()` uploads to HF Hub
  with **no promotion gate** (`retrain_from_clicks.py:124`) — a scheduled run can silently
  ship a worthless model.
- **Event-loop blocking in every service.** All routes are `async def` but do synchronous
  torch/FAISS/BM25/redis/LLM work directly on the loop (retrieval, ranking, query
  understanding especially). Under concurrency, requests serialize and health checks stall.
- **Deploy crashes on any artifact hiccup.** `space_app.py:27` builds `SearchEngine()` at
  import time with no guard — one missing/corrupt artifact → raw stack trace, no UI, no
  health surface.

## 🟠 Important
- **Reproducibility, partially resolved.** `HF_ARTIFACTS_REPO` now defaults to the published
  `shiva-1993/search-ranking-system` (`artifacts_manifest.py:25`), so a fresh clone can
  `python scripts/bootstrap.py` and pull the real model + indexes. Remaining gap: no
  `torch.manual_seed`/numpy/random seeds in the neural training, so regenerating artifacts
  from scratch is not bit-reproducible (retrieval quality is stable, exact weights are not).
- **Promotion gate compares incomparable numbers.** `retraining_dag.py:322-335` compares a
  fresh eval-harness NDCG against a *stale* MLflow metric computed on a different query set /
  methodology; `experiment_ids=["1"]` hardcoded (→ 0.0 → everything "improves" if id differs).
- **No auth + wildcard CORS on the public gateway** (`gateway/main.py:113-118`); `/metrics`
  exposed unauthenticated; `top_k`/`query`/`candidates` unbounded (DoS/OOM); public `/click`
  has no auth/bounds (training-data poisoning vector).
- **Config fails silently, not loudly.** DB defaults to `searchuser/searchpass`
  (`database.py:104-105`); `create_tables()` only warns; `LLM_PROVIDER` typos degrade
  silently (`llm.py:198-205`); `training_config` does no validation despite claiming to.
- **CI has thin real coverage.** RRF, the LambdaRank feature vector, and the promotion gate
  are untested; integration tests deselected in CI; no `--cov-fail-under`; only the trivial
  gateway image is build-tested (heavy torch/faiss images that actually break aren't).
- **`ClickLog` duplicate index** (`database.py:56` auto-index + `:66` explicit same name) —
  risks a `create_tables()` error and shows as Alembic drift.
- **Demo "auto" ranker doesn't reproduce microservice routing** (no difficulty classifier /
  A-B split; query-rewrite stage dropped in `space_app.py:65-80`) — DEPLOY.md's "results
  match the microservice path" is overstated.

## 🟡 Nice-to-have
Pickle-from-Hub RCE surface; no `HEALTHCHECK`/readiness probes; `datetime.utcnow()`
deprecation; inconsistent compose memory limits; acknowledged dead code (`F841` grandfathered
in `pyproject.toml:15-26`); Grafana dashboard won't auto-provision; dead MS MARCO URLs in
`config.yaml:11-16`; moving-branch artifact revision default.

---

## Deployment reality (HF Spaces exhausted)

The code is **barely coupled to HF Spaces**. Artifact hosting is on HF *Hub* (separate
product, still usable). The only code change to switch hosts is reading `$PORT` instead of
hardcoded `7860` (`space_app.py:133`). **The binding constraint is RAM** — the consolidated
engine holds two-tower + FAISS + BM25 + all 500K passage texts in memory (~2–4 GB), ruling
out 256–512 MB free tiers.

$0 targets that actually fit:
- **Oracle Cloud "Always Free" (Ampere A1, up to 24 GB / 4 ARM vCPU)** — perpetual free,
  best fit. Caveat: ARM wheels for torch/faiss.
- **Google Cloud Run** (scale-to-zero, free monthly grant) — fits low-traffic demo; manage
  cold-start artifact re-download.
- **Shrink the footprint** (serve passage text from SQLite/Neon, mmap FAISS, drop
  cross-encoder) to fit a 512 MB tier — a real refactor.

---

## Proposed fix plan (for approval — nothing done yet)

**Workstream A — Honesty + correctness (highest leverage, lowest risk, do first):**
1. Run `evaluate.py` on the existing on-disk models; commit `eval_results.json`; make the
   README table cite measured numbers (or clearly relabel any unbacked ones as targets).
2. Rewrite README §5/§6 recall/hard-negative claims to match the code (or change the code
   to genuinely mine BM25 negatives + compute real recall — decide per cost).
3. Fix LambdaRank feature parity: one shared feature-builder used by train, eval, and
   retrain so train == serve.
4. Fix click-retraining labels (need negatives: unclicked-but-shown docs) and add the
   promotion gate to the free-tier path.
5. Fix the promotion-gate comparison (re-evaluate current prod with the same harness).

**Workstream B — Service hardening:**
6. Offload blocking work to threadpools (`run_in_threadpool`) or async clients.
7. Startup config validation (fail loudly on missing DB/secrets; validate `LLM_PROVIDER`).
8. Input bounds (`Field(ge=1, le=100)` on `top_k`, max lengths, `max_items` on candidates).
9. Real readiness probes; LLM client timeouts + bounded retries.
10. Minimal auth on gateway + `/click`; tighten CORS; fix duplicate index.

**Workstream C — Tests/CI:**
11. Unit-test RRF, the shared feature-builder, and the promotion gate; add `--cov-fail-under`;
    build the heavy images in CI.

**Workstream D — Revive deployment (needs host decision):**
12. Read `$PORT`, guard engine startup with a degraded landing page, pick host
    (Oracle Always Free recommended for the RAM headroom), handle cold-start artifact pulls.

Recommended sequence: **A → B → C → D**. A and B are the interview-defensibility core.
