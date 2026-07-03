# ORCAS-Grounded Real Feedback Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the degenerate all-clicks-are-positive feedback loop with an honest, ORCAS-calibrated click-simulation pipeline that logs impressions (real negatives), retrains LambdaRank on propensity-weighted labels using a single shared train/serve feature builder, and promotes only when nDCG@10 genuinely improves under one evaluation harness.

**Architecture:** A shared feature builder (`services/shared/features.py`) produces the exact 7-feature LambdaRank vector used by the live serve path, and is imported by serve, the click simulator, and retraining so features never skew. The serve path logs every shown passage to a new `impression_logs` table; ORCAS's real query stream is replayed through the retrieval engine, relevance is grounded in MS MARCO passage qrels, and clicks are sampled by a position-based click model whose popularity/click-volume statistics are calibrated from ORCAS (position exponent is a documented literature-based assumption). Retraining derives positives (clicked) and negatives (shown-not-clicked) from impressions, IPS-weights them by propensity, and a single gate re-evaluates prod vs staging with `training/evaluate.py` before promotion.

**Tech Stack:** Python 3.11, SQLAlchemy + Alembic (Postgres/Neon; SQLite for tests), XGBoost `rank:ndcg`, NumPy/Pandas, FastAPI, structlog, pytest, ruff.

## Design decision & honesty note

ORCAS ([orcas.tsv.gz, 18.8M rows, 329.7 MB gz](https://msmarco.z22.web.core.windows.net/msmarcoranking/orcas.tsv.gz)) is a **document-level** click log with columns `qid, query, did, url`, where `did` are **D-prefixed MS MARCO *document* IDs**. It is released under a **non-commercial research license only**. This repository is **passage-level** (`pid` in `0..~8.8M`). There is **no clean public document→passage mapping** and **no public passage-level human click dataset** exists. We therefore **do not claim raw human passage clicks**, and we do not fabricate them.

The honest design (standard in unbiased learning-to-rank research) is:

> Replay a **MS MARCO passage query stream that already has qrels** through the passage retrieval system; **weight** how often each query is replayed by **ORCAS's real query popularity** (ORCAS-matched queries sampled proportionally to their ORCAS frequency, unmatched queries replayed at a baseline weight); generate click events using a **position-based click model (PBM)** calibrated from ORCAS click realism; **ground per-passage relevance in the MS MARCO passage qrels**; and **log impressions** (the full shown result set) so shown-but-not-clicked passages become **real negatives**.

Why the replay stream is MS MARCO (not ORCAS) queries: ORCAS query text rarely maps cleanly to a MS MARCO qid, so a "ORCAS∩qrels" intersection could be tiny and **starve** the simulator. Instead the replay stream is **MS MARCO queries that have qrels** (from `train_queries.parquet` + `train_qrels.parquet`, ~500K judgments; fall back to `dev_queries`/`dev_qrels`). This **guarantees every replayed query has relevance labels** — coverage is never a blocker. ORCAS injects realism (popularity + click volume), while relevance **always** comes from qrels, **never** from ORCAS.

What is genuinely calibrated from ORCAS vs. assumed, stated plainly (no overclaiming):

- **Calibrated from ORCAS real statistics:** query **popularity** (MS MARCO queries whose text matches an ORCAS query are replayed proportionally to ORCAS frequency; non-matching queries replay at a baseline weight), and mean **clicks-per-query** (click volume the simulator reproduces).
- **Literature-based modelling assumption (ORCAS has no rank/position column):** the PBM **position-bias exponent** `eta` in `propensity(rank) = (1/rank)**eta` (default `eta≈1.0`, Joachims et al.). We document this explicitly; we never claim ORCAS provides positions.
- **Ground truth for relevance:** MS MARCO passage qrels (`data/processed/*_qrels.parquet`), not ORCAS document clicks. Every replayed query carries qrels by construction.

This fixes the current degenerate bug (every click label is `1` → no ranking signal → nothing to learn) and every README/docs sentence must reflect this framing. The ORCAS download step must print the non-commercial license notice and require an explicit opt-in flag.

## Global Constraints

- **Python 3.11**; use the repo's pinned deps (no new heavy dependencies — reuse XGBoost, SQLAlchemy, Alembic, pandas, numpy already in `requirements.txt`).
- **CPU must work** end-to-end (free-tier: HF Space + Neon + Upstash). No GPU assumptions; all new code runs on `cpu`.
- **No fabricated or overclaimed data.** Labels are NEVER all-`1`. Positives = clicked; negatives = shown-not-clicked (from impressions). Relevance is from qrels; PBM assumptions are documented.
- **Propensity-weighted labels, not all-1:** IPS weights `1/propensity(rank)` applied to clicked rows in the DMatrix.
- **Every new table gets an Alembic migration** matching the ORM in `services/shared/database.py` exactly (`migrations/env.py` autogenerate must report "no changes" after).
- **`ruff check --select E,F,I services/` must be clean** for touched service files (existing repo lint config).
- **DRY / single feature builder:** the SAME `services/shared/features.py` builder is imported by serve (`services/ranking/main.py`, `deploy/engine.py`), retraining (`scripts/retrain_from_clicks.py`, `airflow_dags/retraining_dag.py`), and the simulator (`scripts/simulate_clicks.py`). The SAME `training/evaluate.py:run_evaluation` is used on both sides of the promotion gate.
- Follow existing **structlog** logging and **pytest** test patterns (see `tests/unit/test_services.py`); DB tests use in-memory SQLite via `Base.metadata.create_all`.
- **Commits go to `main`, no AI attribution** (repo convention; portfolio repo).

---

## File Structure

Files created / modified and their responsibility:

| Path | Action | Responsibility |
|---|---|---|
| `services/shared/features.py` | **Create** | `FEATURE_NAMES`, `Candidate` dataclass, `build_lambdarank_features(...)` — the single source of truth for the 7-feature serve vector. |
| `services/shared/database.py` | Modify (L52–67) | Add `ImpressionLog` ORM model; fix `ClickLog` duplicate `request_id` index. |
| `services/shared/impressions.py` | **Create** | `build_impression_rows(...)` pure helper; `insert_impressions(session, rows)` writer. |
| `migrations/versions/0002_impression_logs.py` | **Create** | Alembic migration: create `impression_logs`, drop the redundant click index if present. |
| `services/gateway/main.py` | Modify (L281–322) | Log one `impression_logs` row per served result via `BackgroundTasks`. |
| `services/ranking/main.py` | Modify (L308–357) | `_rerank_lambdarank` calls the shared builder (removes duplicated vector code). |
| `deploy/engine.py` | Modify (L144–178) | `_rerank_lambdarank` calls the shared builder. |
| `scripts/download_orcas.py` | **Create** | Download/sample ORCAS with license notice + opt-in flag; parse `qid,query,did,url`. |
| `scripts/calibrate_orcas.py` | **Create** | Estimate query popularity, mean clicks/query, and `propensity(rank)`; write `data/processed/orcas_calibration.json`. |
| `scripts/simulate_clicks.py` | **Create** | Replay ORCAS queries → retrieve → qrels relevance → PBM click sampling → write `click_logs` + `impression_logs`. |
| `scripts/retrain_from_clicks.py` | Rewrite (whole file) | Shared builder + propensity-weighted positives/negatives from impressions; calls the shared gate. |
| `scripts/promote.py` | **Create** | `evaluate_and_gate(...)` — one-harness prod-vs-staging nDCG@10 comparison + decision. |
| `airflow_dags/retraining_dag.py` | Modify (L102–376) | `extract_click_features`/`evaluate_new_model`/`promote_if_better` delegate to the shared retrain + gate helpers. |
| `configs/config.yaml` | Modify (L159–162, new blocks) | Add `orcas:` and `click_sim:` config; keep `retraining:` keys. |
| `tests/unit/test_features.py` | **Create** | Feature-builder equals hand-computed vector AND equals live serve vector. |
| `tests/unit/test_impressions.py` | **Create** | ORM + migration parity; impression row build/insert; ClickLog single index. |
| `tests/unit/test_orcas.py` | **Create** | ORCAS parse/sample; calibration monotonicity. |
| `tests/unit/test_click_sim.py` | **Create** | Deterministic-seed clicks: relevant-top clicked > irrelevant; negatives recorded. |
| `tests/unit/test_retrain.py` | **Create** | DMatrix has >1 distinct label per group; IPS weights applied. |
| `tests/unit/test_promote.py` | **Create** | Gate rejects no-improvement, accepts improvement (stubbed eval). |
| `README.md` | Modify (feedback-loop section) | Honest rewrite + non-commercial license note. |
| `docs/adr/` (existing) | Optionally add ADR | Record the ORCAS-calibration decision (fold into Task 9). |

**Live 7-feature serve vector (ground truth — reproduce EXACTLY).** From `services/ranking/main.py:_rerank_lambdarank` (L324–342), mirrored in `deploy/engine.py` (L163–171) and `training/evaluate.py:rerank_lambdarank` (L238–248). `feature_names.json` order:
`["bm25_score","two_tower_cosine_sim","doc_length","query_term_overlap","query_length","bm25_rank","two_tower_rank"]`

```
row = [
  bm25_score,                     # bm25.get_scores(query.lower().split())[bm25_idx[doc_id]]
  float(cand.score),              # two_tower / fusion score, as-is
  min(doc_len / 200.0, 5.0),      # doc_length; doc_len = pid_to_len.get(doc_id, 0)
  overlap,                        # |q_terms ∩ doc_terms| / max(|q_terms|,1); *_terms = set(text.lower().split())
  min(q_len / 20.0, 3.0),         # q_len = len(query.split())
  cand.retrieval_rank / n,        # feature named "bm25_rank" but holds RETRIEVAL rank (documented quirk; preserved for train==serve)
  tt_ranks[i] / n,                # "two_tower_rank": rank of cand.score within the set, 1-indexed desc, /n
]
```
`tt_ranks`: `order = argsort([c.score])[::-1]; tt_ranks[order] = arange(1, n+1)`. Output dtype `float32`, shape `(n, 7)`.

---

## Task 1 — Shared feature builder (closes train/serve skew)

**Files:**
- Create: `services/shared/features.py`
- Test: `tests/unit/test_features.py`
- Modify (after builder is green): `services/ranking/main.py` (L308–357), `deploy/engine.py` (L144–178)

**Interfaces:**
- Produces:
  - `FEATURE_NAMES: list[str]` == the 7 names above.
  - `@dataclass class Candidate: doc_id:int; text:str; score:float; retrieval_rank:int`
  - `build_lambdarank_features(query:str, candidates:list[Candidate], bm25, bm25_pid_list:list[int], pid_to_len:dict[int,int]) -> np.ndarray` (shape `(len(candidates),7)`, `float32`).
  - `services/ranking/main.py:_feature_matrix(query:str, candidates:list[CandidateIn]) -> np.ndarray` (builds `Candidate` objects from the request `CandidateIn` list and returns `build_lambdarank_features(...)` using the module globals `bm25`, `bm25_pid_list`, `pid_to_len`).
- Consumes (downstream): `services/ranking/main.py`, `deploy/engine.py`, `scripts/simulate_clicks.py`, `scripts/retrain_from_clicks.py`.

Steps:
- [ ] Write failing test `tests/unit/test_features.py::test_builder_matches_hand_computed`: build a fake bm25 with `get_scores(tokens)->np.array([2.0, 0.5])`, `bm25_pid_list=[10,20]`, `pid_to_len={10:400,20:50}`, two `Candidate`s (`doc_id=10,score=0.9,retrieval_rank=1,text="machine learning models"`; `doc_id=20,score=0.7,retrieval_rank=2,text="cooking recipes"`), query `"machine learning"`. Assert the returned matrix equals the hand-computed 2×7 array (compute the 7 values by hand per the spec above) via `np.testing.assert_allclose(out, expected, rtol=1e-6)`.
- [ ] Run: `python -m pytest tests/unit/test_features.py -q` → expect **FAIL** with `ModuleNotFoundError: No module named 'services.shared.features'`.
- [ ] Implement `services/shared/features.py`: `FEATURE_NAMES`, `Candidate`, and `build_lambdarank_features` using the exact formulas above (`bm25_idx = {pid:i for i,pid in enumerate(bm25_pid_list)}`, `.get(doc_id,0)` fallback; return `np.asarray(X, dtype=np.float32)`).
- [ ] Run: `python -m pytest tests/unit/test_features.py -q` → expect **PASS** (1 passed).
- [ ] Write failing test `tests/unit/test_features.py::test_serve_matrix_equals_shared_builder`: `import services.ranking.main as rk`; set module globals `rk.bm25 = fake_bm25`, `rk.bm25_pid_list = [10,20]`, `rk.pid_to_len = {10:400,20:50}`; build the same two candidates as `rk.CandidateIn(doc_id=10, text="machine learning models", score=0.9, retrieval_rank=1)` and `CandidateIn(doc_id=20, text="cooking recipes", score=0.7, retrieval_rank=2)`; assert `np.testing.assert_allclose(rk._feature_matrix("machine learning", cands), build_lambdarank_features("machine learning", [Candidate(10,"machine learning models",0.9,1), Candidate(20,"cooking recipes",0.7,2)], fake_bm25, [10,20], {10:400,20:50}), rtol=1e-6)`.
- [ ] Run: `python -m pytest tests/unit/test_features.py -q` → expect **FAIL** with `AttributeError: module 'services.ranking.main' has no attribute '_feature_matrix'`.
- [ ] Add `_feature_matrix(query, candidates)` to `services/ranking/main.py` that does `cands = [Candidate(doc_id=c.doc_id, text=c.text, score=c.score, retrieval_rank=c.retrieval_rank) for c in candidates]; return build_lambdarank_features(query, cands, bm25, bm25_pid_list, pid_to_len)`; add `from services.shared.features import Candidate, build_lambdarank_features`.
- [ ] Refactor `services/ranking/main.py:_rerank_lambdarank` (L324–345) to replace its inline `X` construction with `X = _feature_matrix(query, candidates)`; keep `xgb.DMatrix(X)` / `predict` / argsort / return identical.
- [ ] Run: `python -m pytest tests/unit/test_features.py -q` → expect **PASS** (2 passed).
- [ ] Refactor `deploy/engine.py:_rerank_lambdarank` (L158–172) to build `Candidate(doc_id=c["doc_id"], text=c["text"], score=c["score"], retrieval_rank=c["retrieval_rank"])` and call `build_lambdarank_features(query, cands, self.bm25, self.bm25_pid_list, self.pid_to_len)` in place of the inline `X` loop; import `from services.shared.features import Candidate, build_lambdarank_features`.
- [ ] Run: `python -m pytest tests/unit/test_features.py -q` → expect **PASS** (2 passed).
- [ ] Run: `ruff check --select E,F,I services/shared/features.py services/ranking/main.py deploy/engine.py` → expect clean.
- [ ] Commit: `git add services/shared/features.py services/ranking/main.py deploy/engine.py tests/unit/test_features.py && git commit -m "Shared LambdaRank feature builder; serve paths reuse it (fix train/serve skew)"`

## Task 2 — Impression table + ORM + migration; fix ClickLog duplicate index

**Files:**
- Modify: `services/shared/database.py` (L52–67 ClickLog; add `ImpressionLog` after it)
- Create: `migrations/versions/0002_impression_logs.py`
- Test: `tests/unit/test_impressions.py`

**Interfaces:**
- Produces: `ImpressionLog(Base)` with columns `id, request_id(str64,index), query_text(Text), doc_id(int), rank_shown(int), ranker_version(str32,nullable), created_at(DateTime)`; table `impression_logs`; indexes `ix_impression_logs_request_id`, `ix_impression_logs_created_at`.
- Consumes: nothing new; reuses `Base`, `create_tables`.

Steps:
- [ ] Write failing test `tests/unit/test_impressions.py::test_impressionlog_table_roundtrip`: create in-memory engine `create_engine("sqlite://")`, `from services.shared.database import Base, ImpressionLog`; `Base.metadata.create_all(engine)`; insert one row via a `Session`, query it back, assert fields.
- [ ] Run: `python -m pytest tests/unit/test_impressions.py -q` → expect **FAIL** (`ImportError: cannot import name 'ImpressionLog'`).
- [ ] Add `ImpressionLog` model to `services/shared/database.py` with `__table_args__ = (Index("ix_impression_logs_created_at","created_at"), Index("ix_impression_logs_request_id","request_id"))` and `request_id` column **without** `index=True` (explicit index only, matching the migration).
- [ ] Fix ClickLog duplicate index: in `ClickLog` remove `index=True` on `request_id` (L56) so the only `request_id` index is the explicit `ix_click_logs_request_id` (L66) — eliminates the duplicate index while keeping the migration's index name.
- [ ] Add failing test `test_clicklog_has_single_request_id_index`: reflect `ClickLog.__table__.indexes`; assert exactly one index covers `request_id` and its name is `ix_click_logs_request_id`.
- [ ] Run: `python -m pytest tests/unit/test_impressions.py -q` → expect **PASS** (both tests pass).
- [ ] Create migration `migrations/versions/0002_impression_logs.py` (`revision="0002_impression_logs"`, `down_revision="0001_initial"`): `upgrade()` creates `impression_logs` (columns matching the ORM) + the two indexes; `downgrade()` drops them. (Do not attempt to drop the click index in SQL — it was never created separately in 0001; ORM fix alone suffices.)
- [ ] Add test `test_migration_matches_orm`: run Alembic autogenerate against a temp SQLite URL (`alembic -c alembic.ini upgrade head` then autogenerate diff) OR assert column parity by comparing `ImpressionLog.__table__.columns.keys()` to the column list declared in `0002_impression_logs.py` (import the migration module and read a module-level `COLUMNS` list you define). Choose the column-parity assertion for a hermetic CPU test.
- [ ] Run: `python -m pytest tests/unit/test_impressions.py -q` → expect **PASS**.
- [ ] Run: `ruff check --select E,F,I services/shared/database.py` → expect clean.
- [ ] Commit: `git add services/shared/database.py migrations/versions/0002_impression_logs.py tests/unit/test_impressions.py && git commit -m "Add impression_logs table + migration; fix ClickLog duplicate request_id index"`

## Task 3 — Emit impression logs from the serve path

**Files:**
- Create: `services/shared/impressions.py`
- Modify: `services/gateway/main.py` (L281–322: add impression background task)
- Test: `tests/unit/test_impressions.py` (extend)

**Interfaces:**
- Produces:
  - `build_impression_rows(request_id:str, query_text:str, ranker_version:str|None, results:list[dict]) -> list[dict]` (one dict per result: `request_id, query_text, doc_id, rank_shown, ranker_version`; `rank_shown` = `result["rank"]`, `doc_id` = `result["doc_id"]`).
  - `insert_impressions(session, rows:list[dict]) -> int` (bulk-inserts `ImpressionLog`, returns count).
- Consumes: `services/gateway/main.py` background task; `scripts/simulate_clicks.py` (Task 6).

Steps:
- [ ] Write failing test `test_build_impression_rows`: `results=[{"rank":1,"doc_id":10,"text":"a","score":0.9,"ranker":"lambdarank"},{"rank":2,"doc_id":20,...}]`; assert `build_impression_rows("req1","q","lambdarank",results)` returns 2 dicts with correct `rank_shown`/`doc_id`.
- [ ] Run: `python -m pytest tests/unit/test_impressions.py::test_build_impression_rows -q` → expect **FAIL** (module missing).
- [ ] Implement `services/shared/impressions.py` (`build_impression_rows`, `insert_impressions` using `ImpressionLog`).
- [ ] Run same test → expect **PASS**.
- [ ] Add failing test `test_insert_impressions_writes_rows`: in-memory SQLite, `create_all`, insert via `insert_impressions(session, rows)`, query `ImpressionLog` count == 2 and a row's `rank_shown` matches.
- [ ] Run → expect **FAIL** then implement/verify → **PASS**.
- [ ] Add failing test `test_gateway_logs_impressions`: import `services.gateway.main as gw`; monkeypatch `gw.get_db_session` to return a SQLite-backed session (fixture) and call `gw._sync_log_impressions_to_db({...})` with a shaped payload; assert rows land in `impression_logs`.
- [ ] Run → expect **FAIL** (`_sync_log_impressions_to_db` missing).
- [ ] Modify `services/gateway/main.py`: add `_sync_log_impressions_to_db(payload:dict)` (mirrors `_sync_log_query_to_db` L310–321, uses `insert_impressions`); in `search()` after the QueryLog background task (L296) add `background_tasks.add_task(_sync_log_impressions_to_db, dict(request_id=request_id, query_text=req.query, ranker_version=rank_data.get("ranker_used"), results=rank_data["results"]))`. Import `build_impression_rows, insert_impressions` and `ImpressionLog`.
- [ ] Run: `python -m pytest tests/unit/test_impressions.py -q` → expect **PASS** (all).
- [ ] Run: `ruff check --select E,F,I services/gateway/main.py services/shared/impressions.py` → expect clean.
- [ ] Commit: `git add services/shared/impressions.py services/gateway/main.py tests/unit/test_impressions.py && git commit -m "Log impressions (shown passages) from the serve path to derive real negatives"`

## Task 4 — ORCAS downloader + sampler (with license notice)

**Files:**
- Create: `scripts/download_orcas.py`
- Modify: `configs/config.yaml` (add `orcas:` block)
- Test: `tests/unit/test_orcas.py`

**Interfaces:**
- Produces:
  - `ORCAS_URL:str`, `ORCAS_LICENSE_NOTICE:str`.
  - `parse_orcas_line(line:str) -> dict|None` (`{"qid","query","did","url"}`; returns `None` on malformed lines).
  - `sample_orcas(src_lines:Iterable[str], n:int, seed:int=42) -> list[dict]` (reservoir sample of parsed rows).
  - `main(argv)` CLI requiring `--accept-noncommercial-license`; writes `data/raw/orcas_sample.tsv`.
- Consumes (downstream): `scripts/calibrate_orcas.py`, `scripts/simulate_clicks.py`.

Steps:
- [ ] Write failing test `tests/unit/test_orcas.py::test_parse_orcas_line`: `parse_orcas_line("1\tweather today\tD123\thttp://x")` == `{"qid":"1","query":"weather today","did":"D123","url":"http://x"}`; malformed `"bad"` → `None`.
- [ ] Run: `python -m pytest tests/unit/test_orcas.py -q` → expect **FAIL** (module missing).
- [ ] Implement `scripts/download_orcas.py`: `ORCAS_URL`, `ORCAS_LICENSE_NOTICE` (states non-commercial research-only), `parse_orcas_line` (tab-split, exactly 4 fields).
- [ ] Run → expect **PASS**.
- [ ] Add failing test `test_sample_orcas_is_deterministic_and_sized`: given 100 synthetic lines, `sample_orcas(lines,10,seed=1)` returns 10 parsed dicts and is identical across two calls with same seed.
- [ ] Run → **FAIL** then implement reservoir sampler with `random.Random(seed)` → **PASS**.
- [ ] Add failing test `test_main_requires_license_optin`: call `main([])` (no flag) → returns non-zero and prints the license notice; does not attempt any network download (monkeypatch the downloader function and assert it was NOT called).
- [ ] Run → **FAIL** then implement `main` (argparse: `--accept-noncommercial-license`, `--sample-size`, `--seed`, `--out`; prints `ORCAS_LICENSE_NOTICE` always; only downloads/streams when the flag is set; streams gz via `requests` + `gzip` line iteration into `sample_orcas`).
- [ ] Run: `python -m pytest tests/unit/test_orcas.py -q` → expect **PASS**.
- [ ] Add `orcas:` block to `configs/config.yaml` (`url`, `license: "non-commercial research only"`, `sample_size: 200000`, `raw_path: data/raw/orcas_sample.tsv`, `seed: 42`).
- [ ] Run: `ruff check --select E,F,I scripts/download_orcas.py` → expect clean.
- [ ] Commit: `git add scripts/download_orcas.py configs/config.yaml tests/unit/test_orcas.py && git commit -m "ORCAS downloader/sampler with non-commercial license opt-in"`

## Task 5 — Position-bias / propensity calibration from ORCAS

**Files:**
- Create: `scripts/calibrate_orcas.py`
- Test: `tests/unit/test_orcas.py` (extend)

**Interfaces:**
- Produces:
  - `query_popularity(rows:list[dict]) -> dict[str,int]` (normalized-query-text → ORCAS frequency; normalize = `strip().lower()`). This map is later used by the simulator to **weight** how often each MS MARCO replay query is sampled (matched by the same normalized text); it never supplies relevance.
  - `mean_clicks_per_query(rows:list[dict]) -> float` (mean distinct `did` per `qid`).
  - `propensity_curve(max_rank:int, eta:float) -> dict[int,float]` (`{rank: (1/rank)**eta}`, ranks `1..max_rank`).
  - `calibrate(rows:list[dict], max_rank:int=10, eta:float=1.0) -> dict` → JSON-serializable `{"eta","propensity","mean_clicks_per_query","query_popularity","source":"ORCAS", "notes": "<what is calibrated vs assumed>"}`; `main()` writes `data/processed/orcas_calibration.json`.
- Consumes: output of Task 4; used by `scripts/simulate_clicks.py`.

Steps:
- [ ] Write failing test `test_propensity_is_monotonic_decreasing`: `p = propensity_curve(10, eta=1.0)`; assert `p[1] > p[2] > ... > p[10] > 0` (loop over consecutive ranks).
- [ ] Run: `python -m pytest tests/unit/test_orcas.py::test_propensity_is_monotonic_decreasing -q` → expect **FAIL** (module missing).
- [ ] Implement `scripts/calibrate_orcas.py:propensity_curve` → **PASS**.
- [ ] Add failing test `test_query_popularity_and_click_volume`: rows with `qid` `A` clicked (2 distinct dids), `B` (1 did), query text repeated → assert `query_popularity` counts by normalized text and `mean_clicks_per_query == 1.5`.
- [ ] Run → **FAIL** then implement `query_popularity`, `mean_clicks_per_query` → **PASS**.
- [ ] Add failing test `test_calibrate_output_shape`: `calibrate(rows)` returns dict with keys `eta, propensity, mean_clicks_per_query, query_popularity, source, notes`; `propensity` keys are ints `1..10`; `notes` mentions that `eta` is a literature assumption and ORCAS has no position column.
- [ ] Run → **FAIL** then implement `calibrate` + `main` (json.dump; make propensity keys str for JSON, doc the round-trip) → **PASS**.
- [ ] Run: `ruff check --select E,F,I scripts/calibrate_orcas.py` → expect clean.
- [ ] Commit: `git add scripts/calibrate_orcas.py tests/unit/test_orcas.py && git commit -m "ORCAS calibration: query popularity, click volume, documented PBM propensity"`

## Task 6 — Click simulator (MS MARCO replay stream, ORCAS-weighted → retrieve → qrels → PBM → logs with negatives)

**Files:**
- Create: `scripts/simulate_clicks.py`
- Modify: `configs/config.yaml` (add `click_sim:` block)
- Test: `tests/unit/test_click_sim.py`

**Design note:** The replay stream is **MS MARCO queries that have qrels** — so every replayed query is guaranteed to carry relevance labels (no coverage/starvation failure mode). ORCAS only **weights** the sampling of that stream (queries whose normalized text matches an ORCAS query are drawn proportionally to ORCAS popularity; unmatched queries keep a baseline weight) and calibrates click volume/propensity. Relevance always comes from qrels.

**Interfaces:**
- Produces:
  - `load_replay_workload(queries_df, qrels_df) -> tuple[dict[str,str], dict[str,set[int]]]` → `(qid_to_text, qid_to_gold_pids)`, built from `train_queries`/`train_qrels` (fallback `dev_queries`/`dev_qrels`). Mirrors `training/evaluate.py` loading: `qid_to_gold = qrels_df.groupby("qid")["pid"].apply(set).to_dict()`, `qid_to_text = dict(zip(queries_df["qid"], queries_df["text"]))`. Only qids present in qrels are kept.
  - `build_sampling_weights(qid_to_text:dict[str,str], query_popularity:dict[str,int], baseline:float=1.0) -> dict[str,float]` → per-qid replay weight = `float(query_popularity[norm(text)])` if the normalized text matches an ORCAS query else `baseline`. Also returns/counts how many qids matched ORCAS.
  - `sample_clicks(shown:list[dict], gold_pids:set[int], propensity:dict[int,float], rng:random.Random, irrelevant_ctr:float=0.02) -> list[tuple[int,int,bool]]` → per shown passage `(doc_id, rank_shown, clicked)`; `clicked ~ Bernoulli(propensity[rank] * (1.0 if doc_id in gold_pids else irrelevant_ctr))`.
  - `simulate(engine, qid_to_text, qid_to_gold, weights, calibration, cfg, session, seed) -> dict` (samples `cfg.queries` qids proportional to `weights`, retrieves, writes `impression_logs` for the full shown set and `click_logs` for `clicked=True`; returns counts `{impressions, clicks, negatives, queries_replayed, queries_matched_to_orcas}`).
- Consumes: `services/shared/features.py` indirectly (engine reuses it), `services/shared/impressions.py:insert_impressions`, `ImpressionLog`, `ClickLog`, Task 5 calibration + `query_popularity`.

Steps:
- [ ] Write failing test `test_sample_clicks_prefers_relevant_top`: run 2000 trials with fixed `Random(0)`; a rank-1 passage that IS in `gold_pids` is clicked far more often than an identical-rank passage NOT in gold — assert `clicks_relevant > clicks_irrelevant`. Also assert a rank-10 relevant passage is clicked less often than a rank-1 relevant passage (propensity effect).
- [ ] Run: `python -m pytest tests/unit/test_click_sim.py -q` → expect **FAIL** with `ModuleNotFoundError: No module named 'scripts.simulate_clicks'` (or `AttributeError` for the missing function).
- [ ] Implement `scripts/simulate_clicks.py:sample_clicks` using `propensity` + `irrelevant_ctr` → **PASS**.
- [ ] Add failing test `test_load_replay_workload_from_qrels`: build tiny `queries_df` (`qid`,`text` for `q1,q2,q3`) and `qrels_df` (`qid`,`pid` for `q1→{5}`, `q2→{6,7}`); assert `load_replay_workload(...)` returns `qid_to_text` with all three and `qid_to_gold == {"q1":{5},"q2":{6,7}}` (q3 dropped — no qrels). Confirms every workload qid has gold pids.
- [ ] Run → **FAIL** then implement `load_replay_workload` → **PASS**.
- [ ] Add failing test `test_build_sampling_weights_uses_orcas_popularity`: `qid_to_text={"q1":"machine learning","q2":"rare query"}`, `query_popularity={"machine learning":50}`; assert weight for `q1==50.0`, `q2==1.0` (baseline), and the matched-count is `1`.
- [ ] Run → **FAIL** then implement `build_sampling_weights` → **PASS**.
- [ ] Add failing test `test_simulate_records_impressions_and_negatives`: build a **fake engine** with `retrieve(query, embed_text, top_k)` returning a fixed candidate list (doc_ids `[5,6,7]`), in-memory SQLite session; `qid_to_text={"q1":"q one"}`, `qid_to_gold={"q1":{5}}`, `weights={"q1":1.0}`; run `simulate(...)` with `Random(0)` for `cfg.queries=1`; assert `impression_logs` has 3 rows, `click_logs` has ≥1 row for doc 5, and at least one shown-not-clicked passage (6 or 7) is in impressions but NOT in clicks (a real negative). Assert returned `negatives >= 1` and `queries_replayed == 1`.
- [ ] Run → **FAIL** then implement `simulate` (weighted-sample qids → `engine.retrieve` → `build_impression_rows`/`insert_impressions` for the shown set; `sample_clicks`; write `ClickLog(clicked=True)` for clicked; `negatives = impressions − clicks`) → **PASS**.
- [ ] Add `click_sim:` block to `configs/config.yaml` (`top_k: 10`, `irrelevant_ctr: 0.02`, `queries: 5000`, `seed: 42`, `calibration_path: data/processed/orcas_calibration.json`, `workload: train` with dev fallback).
- [ ] Add a thin `main()` that wires config + `SearchEngine` + DB session + calibration + `query_popularity` + `load_replay_workload` + `build_sampling_weights` and calls `simulate` (guarded by `if __name__ == "__main__"`; not unit-tested end-to-end).
- [ ] Run: `python -m pytest tests/unit/test_click_sim.py -q` → expect **PASS**; `ruff check --select E,F,I scripts/simulate_clicks.py` → clean.
- [ ] Commit: `git add scripts/simulate_clicks.py configs/config.yaml tests/unit/test_click_sim.py && git commit -m "Click simulator: ORCAS replay + qrels relevance + PBM clicks + impression negatives"`

## Task 7 — Corrected retraining (propensity-weighted, non-degenerate labels)

**Files:**
- Rewrite: `scripts/retrain_from_clicks.py`
- Modify: `airflow_dags/retraining_dag.py` (L102–290: `extract_click_features`, `train_lambdarank_with_clicks` delegate to shared helpers)
- Test: `tests/unit/test_retrain.py`

**Interfaces:**
- Produces (in `scripts/retrain_from_clicks.py`):
  - `load_labeled_impressions(engine) -> pd.DataFrame` (join `impression_logs` LEFT JOIN `click_logs` on `(request_id, doc_id)`; columns `request_id, query_text, doc_id, rank_shown, clicked(bool)`).
  - `build_training_matrix(labeled:pd.DataFrame, retriever, bm25, bm25_pid_list, pid_to_len, propensity:dict[int,float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]` → `(X, y, weights, groups)` where `y=1.0` clicked else `0.0`, `weights=1/propensity[rank]` for clicked else `1.0`, `groups` = shown-set size per `request_id`; features from `build_lambdarank_features`.
  - `train_and_save(X, y, weights, groups) -> Path`.
- Consumes: `services/shared/features.py`, calibration JSON, `ImpressionLog`/`ClickLog`.

Steps:
- [ ] Write failing test `tests/unit/test_retrain.py::test_matrix_has_multiple_labels_per_group`: build a small `labeled` DataFrame for ONE `request_id` with 3 shown passages, 1 clicked; stub `retriever`/`build_lambdarank_features` (monkeypatch to return deterministic rows) so no models load; call `build_training_matrix(...)`; assert `groups==[3]`, `set(y)=={0.0,1.0}` (i.e. `len(np.unique(y[:3]))>1`), and clicked row's `weight == 1/propensity[rank]` while non-clicked weight `==1.0`.
- [ ] Run: `python -m pytest tests/unit/test_retrain.py -q` → expect **FAIL** (function missing / current file has no such API).
- [ ] Rewrite `scripts/retrain_from_clicks.py`: keep `THRESHOLD`/`_publish`; replace `_load_clicks`/`_build_features`/`_train_and_save` with `load_labeled_impressions`, `build_training_matrix` (imports and calls `build_lambdarank_features`; **no `y.append(1)`**), `train_and_save` (passes `weight=weights` to `xgb.DMatrix`). `main()` gates on impression count, builds matrix, guards `if len(np.unique(y)) < 2: print("Degenerate labels — abort"); return 0`, trains, publishes.
- [ ] Run → expect **PASS**.
- [ ] Add failing test `test_main_aborts_on_degenerate_labels`: feed a `labeled` frame where every shown row is clicked (all-1) → `main`-level guard returns 0 and prints the degenerate message (test the guard helper `is_degenerate(y)->bool`).
- [ ] Run → **FAIL** then add `is_degenerate` + wire the guard → **PASS**.
- [ ] Modify `airflow_dags/retraining_dag.py`: `extract_click_features` and `train_lambdarank_with_clicks` import from `scripts.retrain_from_clicks` (`load_labeled_impressions`, `build_training_matrix`, `train_and_save`) so the all-`1` label block (L203–216) is deleted and the DAG uses propensity-weighted labels. Keep MLflow logging around `train_and_save`.
- [ ] Run: `python -m pytest tests/unit/test_retrain.py -q` → expect **PASS**; `ruff check --select E,F,I scripts/retrain_from_clicks.py` → clean.
- [ ] Commit: `git add scripts/retrain_from_clicks.py airflow_dags/retraining_dag.py tests/unit/test_retrain.py && git commit -m "Retrain on propensity-weighted positives/negatives via shared builder (kill all-1 labels)"`

## Task 8 — Real promotion gate (one harness, prod vs staging)

**Files:**
- Create: `scripts/promote.py`
- Modify: `airflow_dags/retraining_dag.py` (L292–376: `evaluate_new_model`, `promote_if_better` delegate to gate)
- Test: `tests/unit/test_promote.py`

**Interfaces:**
- Produces:
  - `evaluate_model_ndcg(model_path:Path, num_queries:int, config_key:str="Hybrid(RRF)+LambdaRank", eval_fn=run_evaluation) -> float` (swaps `model_path` into the prod slot, runs `eval_fn`, returns `NDCG@10` for `config_key`, always restores the original model).
  - `evaluate_and_gate(prod_path:Path, staging_path:Path, margin:float, num_queries:int, eval_fn=run_evaluation) -> dict` → `{"prod_ndcg","staging_ndcg","delta","promote":bool}`; `promote = delta >= margin`. **Both** numbers come from the SAME `eval_fn` (fixes apples-to-oranges).
- Consumes: `training/evaluate.py:run_evaluation`; used by the DAG and (optionally) the CI retrain job.

Steps:
- [ ] Write failing test `tests/unit/test_promote.py::test_gate_rejects_no_improvement`: pass a stub `eval_fn` that returns `{config_key:{"NDCG@10":0.50}}` for prod and `{config_key:{"NDCG@10":0.505}}` for staging via a side-effect counter; `margin=0.01` → `promote is False`, `delta==pytest.approx(0.005)`.
- [ ] Run: `python -m pytest tests/unit/test_promote.py -q` → expect **FAIL** (module missing).
- [ ] Implement `scripts/promote.py:evaluate_model_ndcg` and `evaluate_and_gate` (accept injectable `eval_fn`; do file swap in `evaluate_model_ndcg` with try/finally restore like DAG L303–320 but for BOTH models).
- [ ] Run → expect **PASS**.
- [ ] Add failing test `test_gate_accepts_improvement`: stub returns prod `0.50`, staging `0.53`, `margin=0.01` → `promote is True`, `delta==pytest.approx(0.03)`.
- [ ] Run → **FAIL** (if logic wrong) then verify → **PASS**.
- [ ] Add test `test_evaluate_model_restores_original`: monkeypatch `eval_fn` and filesystem with tmp paths; assert after `evaluate_model_ndcg` the original prod file is back in place even if `eval_fn` raises (wrap in `pytest.raises` for the raising variant).
- [ ] Run → **FAIL** then harden try/finally → **PASS**.
- [ ] Modify `airflow_dags/retraining_dag.py`: replace `evaluate_new_model` body (delete the stale-MLflow `prod_ndcg` pull L322–333) and `promote_if_better` gate with calls to `evaluate_and_gate(prod_path, staging_path, NDCG_IMPROVEMENT_THRESHOLD, 1000)`; push `result["delta"]`/`result["promote"]` to XCom; branch on `result["promote"]`.
- [ ] Run: `python -m pytest tests/unit/test_promote.py -q` → expect **PASS**; `ruff check --select E,F,I scripts/promote.py` → clean.
- [ ] Commit: `git add scripts/promote.py airflow_dags/retraining_dag.py tests/unit/test_promote.py && git commit -m "Promotion gate: prod vs staging under one eval harness (fix apples-to-oranges)"`

## Task 9 — Honest README / docs rewrite + non-commercial license note

**Files:**
- Modify: `README.md` (feedback-loop / retraining section)
- Optionally create: `docs/adr/000X-orcas-calibrated-feedback.md`
- Test: `tests/unit/test_orcas.py` (add a docs-honesty guard)

**Interfaces:**
- Produces: updated docs prose; no code API.
- Consumes: nothing.

Steps:
- [ ] Write failing test `tests/unit/test_orcas.py::test_readme_states_simulation_and_license`: read `README.md`; assert it contains all of: `"ORCAS"`, `"simulat"` (case-insensitive), `"non-commercial"`, `"impression"`, and does **NOT** contain an overclaiming phrase like `"human passage clicks"` (assert that exact phrase is absent).
- [ ] Run: `python -m pytest tests/unit/test_orcas.py::test_readme_states_simulation_and_license -q` → expect **FAIL** (current README lacks this framing).
- [ ] Rewrite the README feedback-loop section: explain (1) impressions logged → real negatives; (2) ORCAS real query stream replayed, popularity + click-volume calibrated from ORCAS, position-bias `eta` a documented literature assumption (ORCAS has no positions); (3) relevance grounded in MS MARCO passage qrels; (4) propensity-weighted retraining; (5) promotion only on real nDCG@10 gain under one harness; (6) ORCAS is **non-commercial research license only**. No claim of raw human passage clicks.
- [ ] Run → expect **PASS**.
- [ ] (Optional) Add an ADR mirroring the "Design decision & honesty note" section; keep dates consistent (2026-07-03).
- [ ] Run full suite: `python -m pytest tests/unit -q` → expect all green.
- [ ] Commit: `git add README.md docs/ tests/unit/test_orcas.py && git commit -m "Docs: honest ORCAS-calibrated feedback-loop framing + non-commercial license note"`

---

## Self-Review

**Spec coverage (every required fix mapped to a task):**
- Degenerate all-`1` labels → **Task 7** (`build_training_matrix` with `y=clicked?1:0`, `is_degenerate` guard) and **Task 7** deletes airflow L203–216 all-`1` block.
- Train/serve feature skew → **Task 1** (single `build_lambdarank_features` imported by serve `ranking/main.py` + `engine.py`, and by retrain/simulate via the engine). The prior skew (`rank_norm=rank_shown/10.0` used for BOTH rank features in the old `retrain_from_clicks.py` L98–101) is eliminated because retrain now reconstructs features through the shared builder.
- Apples-to-oranges promotion gate → **Task 8** (`evaluate_and_gate` runs `run_evaluation` for BOTH prod and staging; deletes stale-MLflow pull at DAG L322–333).
- ClickLog duplicate index → **Task 2** (remove `index=True` on `ClickLog.request_id`, keep explicit `ix_click_logs_request_id`).
- Impression logging added → **Tasks 2 (table)** + **3 (serve emits rows)** + used as negatives in **6/7**.
- ORCAS ingestion/calibration + real query workload → **Tasks 4 + 5 + 6** (download+sample ORCAS; calibrate popularity/volume/propensity; replay stream = MS MARCO queries that have qrels, ORCAS-weighted so coverage is guaranteed and never starves).
- Alembic migration matching ORM → **Task 2** (`0002_impression_logs.py`, column-parity test).
- Non-commercial license honesty → **Tasks 4 (opt-in + notice)** + **9 (README)**.

**Placeholder scan:** No `TODO`/`pass`/`...` placeholders. Every implementation step names concrete functions with signatures; every test step states exact `python -m pytest` command and expected FAIL/PASS. The only intentionally-not-unit-tested code is the `if __name__ == "__main__"` CLI wiring in `download_orcas.py`, `calibrate_orcas.py`, `simulate_clicks.py`, `retrain_from_clicks.py` (thin glue over unit-tested helpers) — this is standard and each still has a tested core.

**Type/name consistency (Produces == next Consumes):**
- `Candidate` / `build_lambdarank_features(query, candidates, bm25, bm25_pid_list, pid_to_len) -> np.ndarray[float32,(n,7)]` — produced Task 1, consumed by serve (Task 1 via `_feature_matrix`), simulator (Task 6 via engine), retrain (Task 7). Signature identical everywhere; the Task 1 parity test asserts `ranking.main._feature_matrix(...)` equals `build_lambdarank_features(...)` for the same inputs (no monkeypatching/booster capture).
- Replay workload: `load_replay_workload(queries_df, qrels_df) -> (qid_to_text, qid_to_gold)` (Task 6) mirrors `training/evaluate.py` loading (`groupby("qid")["pid"].apply(set)`, `dict(zip(qid,text))`) and keeps only qids that have qrels, so every replayed query has gold pids. `build_sampling_weights(qid_to_text, query_popularity)` (Task 6) consumes Task 5's `query_popularity` to weight (never label) the stream.
- `build_impression_rows(...) -> list[dict]` / `insert_impressions(session, rows) -> int` — produced Task 3, consumed by gateway (Task 3) and simulator (Task 6).
- `ImpressionLog` ORM (Task 2) columns == migration columns (Task 2 parity test) == rows written by `insert_impressions` (Task 3) == joined by `load_labeled_impressions` (Task 7).
- `calibrate(...) -> {"eta","propensity","mean_clicks_per_query","query_popularity",...}` (Task 5) → `propensity: dict[int,float]` consumed by `sample_clicks` (Task 6) and `build_training_matrix` (Task 7). Note: JSON serializes propensity keys as strings — Task 5 documents the int⇄str round-trip; consumers cast keys to `int` on load (call this out in Task 6/7 impl).
- `evaluate_and_gate(prod_path, staging_path, margin, num_queries, eval_fn) -> {"prod_ndcg","staging_ndcg","delta","promote"}` (Task 8) consumed by the DAG.

**Consistency fix applied inline:** propensity dict key type (int in memory, str in JSON) is a cross-task hazard — resolved by requiring consumers in Tasks 6 and 7 to `int(k)`-cast propensity keys immediately after `json.load`, and Task 5's `calibrate` test asserts the in-memory keys are ints. Also: the feature named `bm25_rank` actually carries the retrieval rank at serve time — preserved deliberately (train==serve wins over the misleading name); documented in the File Structure section and reproduced by the shared builder.
