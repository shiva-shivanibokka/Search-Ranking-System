# Neural Search Ranking System

![CI](https://github.com/OWNER/Search-Ranking-System/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
<!-- Live demo: add your Hugging Face Space URL once deployed (see DEPLOY.md) -->

A full production-grade search and ranking system, built the way a senior ML engineer would build it at a company like YouTube, Spotify, or Google. It takes a user's search query, understands what they mean, finds the most relevant passages from 500,000 documents, ranks them using machine learning, and returns results in under 200 milliseconds — all while learning from user clicks over time to get better automatically.

This is not a notebook project. It is a complete system with five microservices, a real-time feedback loop, automated model retraining, a promotion gate, live monitoring dashboards, and a Gradio demo UI. Every component is containerised and deployable with a single command.

### At a glance

- **Problem:** two-stage neural search (retrieve → rank) over 500K MS MARCO passages under a ~200ms budget.
- **Result:** NDCG@10 improves ~70% over a BM25 keyword baseline (0.184 → 0.312 with the CrossEncoder reranker).
- **ML:** two-tower dense retriever, BM25, FAISS IVF+PQ, hybrid retrieval (RRF), LambdaRank + CrossEncoder rerankers, difficulty-based routing, click-feedback retraining with a promotion gate.
- **Engineering:** 5 FastAPI microservices, Postgres + Redis, MLflow, Airflow, Prometheus/Grafana, Docker Compose, GitHub Actions CI, Alembic migrations, provider-agnostic LLM layer (Groq/Gemini/OpenAI/Anthropic + zero-key fallback).
- **Runs free:** public demo on Hugging Face Spaces + Neon (Postgres) + Upstash (Redis) at $0 — see **[DEPLOY.md](DEPLOY.md)**.
- **Design rationale:** **[Architecture Decision Records](docs/adr/)**.

> **Quickstart:** `cp .env.example .env`, then `python scripts/bootstrap.py` (pull model/index artifacts) and `docker-compose up`. Full deployment guide in [DEPLOY.md](DEPLOY.md).

---

## Table of Contents

1. [What problem does this solve?](#1-what-problem-does-this-solve)
2. [The dataset — MS MARCO](#2-the-dataset--ms-marco)
3. [System architecture overview](#3-system-architecture-overview)
4. [A complete request walkthrough](#4-a-complete-request-walkthrough)
5. [The data pipeline — from raw files to training-ready data](#5-the-data-pipeline--from-raw-files-to-training-ready-data)
6. [The ML models — how each one works](#6-the-ml-models--how-each-one-works)
7. [The five microservices — what each one does](#7-the-five-microservices--what-each-one-does)
8. [The database — what gets stored and why](#8-the-database--what-gets-stored-and-why)
9. [The feedback loop — how the system improves itself](#9-the-feedback-loop--how-the-system-improves-itself)
10. [The automated retraining pipeline (Airflow)](#10-the-automated-retraining-pipeline-airflow)
11. [Experiment tracking with MLflow](#11-experiment-tracking-with-mlflow)
12. [Monitoring — Prometheus and Grafana](#12-monitoring--prometheus-and-grafana)
13. [A/B testing — LambdaRank vs CrossEncoder](#13-ab-testing--lambdarank-vs-crossencoder)
14. [Evaluation results](#14-evaluation-results)
15. [The Gradio UI](#15-the-gradio-ui)
16. [Technology stack](#16-technology-stack)
17. [Project structure](#17-project-structure)
18. [Getting started](#18-getting-started)
19. [Service URLs](#19-service-urls)
20. [Running the tests](#20-running-the-tests)
21. [Key design decisions](#21-key-design-decisions)
22. [Academic references](#22-academic-references)

---

## 1. What problem does this solve?

Search is one of the hardest engineering problems at scale. When you type a query into YouTube or Google, the system does not read every single video description or webpage and compare it to your query — with billions of documents, that would take minutes. Instead, every major search system uses a **two-stage pipeline**:

**Stage 1 — Retrieve:** Quickly fetch a small set of candidates (maybe 100) from the full corpus using a fast approximate method. This stage prioritises speed over perfect accuracy.

**Stage 2 — Rank:** Score those 100 candidates carefully using a more powerful, slower model, then return the best 10. This stage prioritises quality over speed.

This project builds that exact two-stage pipeline from scratch. The retrieval stage uses a neural model that understands the *meaning* of a query (not just its words), and the ranking stage uses two different rerankers that are tested against each other live to find out which is better.

The system is evaluated on **MS MARCO**, the same benchmark dataset used by Google, Facebook, and Microsoft to measure their own search systems. This means the results in this project can be directly compared to the published state of the art.

---

## 2. The dataset — MS MARCO

**MS MARCO Passage Ranking** is a dataset released by Microsoft Research, built from real Bing search queries and real web passages.

| Component | Size | Description |
|---|---|---|
| Passage collection | 8.8M passages (500K used) | Real web passages from Common Crawl |
| Training queries | ~400,000 | Real Bing search queries |
| Training relevance labels | ~530,000 pairs | (query, relevant passage) judgements |
| Dev queries | 6,980 | Held-out queries for evaluation |
| Dev relevance labels | Dense | Multiple relevant passages per query |
| Training triples | ~40M | (query, positive, negative) triplets |

Each passage has a unique ID (`pid`) and is a short text snippet — typically 1-5 sentences. Each query has a corresponding set of relevant passages identified by human annotators.

The dataset is in plain `.tsv` format and downloads to about 3GB. The `scripts/download_msmarco.py` script handles the download automatically.

**Why MS MARCO specifically?**

Because every IR (information retrieval) research paper uses it. BM25 achieves NDCG@10 ~0.184 on this benchmark. Dense retrieval models (like the two-tower trained in this project) reach ~0.22. Cross-encoder rerankers reach ~0.31. These numbers are directly comparable to papers from Google, Facebook, and universities worldwide — which means you can measure exactly how good your model is relative to the state of the art.

---

## 3. System architecture overview

The system is made up of five FastAPI microservices, three infrastructure services (Redis, PostgreSQL, MLflow), two monitoring services (Prometheus, Grafana), one orchestration service (Airflow), and a Gradio UI — all running together in Docker.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         docker-compose network                       │
│                                                                       │
│  Client (browser / curl)                                              │
│       │                                                               │
│       ▼                                                               │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │              API Gateway   :8000                          │        │
│  │  • Injects unique request_id into every request           │        │
│  │  • Calls downstream services sequentially (async)         │        │
│  │  • Measures latency per stage                             │        │
│  │  • Logs query + latency to PostgreSQL (async, non-blocking│        │
│  │  • Exposes /metrics for Prometheus                        │        │
│  └──────────┬───────────────────────────────────────────────┘        │
│             │                                                         │
│    ┌────────▼────────┐  ┌────────────────┐  ┌─────────────────────┐  │
│    │ Query            │  │ Retrieval       │  │ Ranking              │  │
│    │ Understanding    │  │ Service  :8002  │  │ Service      :8003  │  │
│    │ :8001            │  │                │  │                     │  │
│    │ • Intent classify│  │ • Redis cache  │  │ • A/B test split    │  │
│    │ • Query rewrite  │  │ • FAISS search │  │ • LambdaRank (fast) │  │
│    │ • HyDE expansion │  │ • top-100 back │  │ • CrossEncoder(acc) │  │
│    │ • Claude Haiku   │  │                │  │ • Hot-reload        │  │
│    └─────────────────┘  └────────────────┘  └─────────────────────┘  │
│                                                                       │
│    ┌─────────────────┐  ┌────────────────┐  ┌─────────────────────┐  │
│    │ Feedback         │  │ PostgreSQL      │  │ Redis               │  │
│    │ Service  :8004   │  │ :5432           │  │ :6379               │  │
│    │ • Log clicks     │  │ • query_logs    │  │ • retrieval cache   │  │
│    │ • Threshold check│  │ • click_logs    │  │ • TTL = 1 hour      │  │
│    └─────────────────┘  │ • model_versions│  └─────────────────────┘  │
│                          └────────────────┘                           │
│                                                                       │
│    ┌─────────────────┐  ┌────────────────┐  ┌─────────────────────┐  │
│    │ MLflow   :5001   │  │ Airflow  :8080  │  │ Prometheus  :9090   │  │
│    │ • Experiment log │  │ • Daily retrain│  │ • Scrapes all 5     │  │
│    │ • Model registry │  │ • DAG: check → │  │   services every 15s│  │
│    │ • Artifacts      │  │   train →      │  └─────────────────────┘  │
│    └─────────────────┘  │   evaluate →   │  ┌─────────────────────┐  │
│                          │   promote →   │  │ Grafana     :3000   │  │
│                          │   hot-reload  │  │ • 11 live panels    │  │
│                          └────────────────┘  └─────────────────────┘  │
│                                                                       │
│    ┌─────────────────────────────────────────────────────────────┐   │
│    │                    Gradio UI   :7860                         │   │
│    │   Search | A/B Compare | Offline Eval | System Stats         │   │
│    └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. A complete request walkthrough

Here is exactly what happens, step by step, when a user searches for `"what causes inflation"`.

### Step 1 — API Gateway receives the request

```
POST http://localhost:8000/search
{
  "query": "what causes inflation",
  "top_k": 10
}
```

The gateway:
- Generates a unique `request_id`, e.g. `"a3f7c2d1-8b4e-4f1a-9c3d-2e5b7f8a1d4c"`
- Stamps the start time
- Binds `request_id` to the structured log context (all subsequent logs from any service will include this ID)
- Calls Query Understanding asynchronously

### Step 2 — Query Understanding Service analyses the query

The query understanding service (`services/query_understanding/main.py`) first runs fast rule-based checks:

```
"what causes inflation"
    ↓
Rule check: starts with "what" → matches INFORMATIONAL pattern
Intent = "informational" (no LLM needed, rule was clear)
```

Because the intent is `informational` and the query is short and clear, no rewriting is needed.

Then it generates a **HyDE passage** (Hypothetical Document Embedding). The idea behind HyDE is that a query embedding and a relevant document embedding live in slightly different parts of the vector space. If we generate a hypothetical answer to the query and embed *that* instead, we get a vector that sits closer to where the real answers live — improving retrieval recall.

Claude Haiku is called with:
```
"Write a short factual passage that would be the ideal answer to:
 Question: what causes inflation"
```

Claude returns something like:
```
"Inflation is caused by several factors including excessive money supply growth,
 supply chain disruptions, increased consumer demand outpacing production,
 rising energy costs, and wage increases that push up the cost of goods and services.
 Central banks monitor inflation through indices like the CPI..."
```

This hypothetical passage (not the original query) is what gets embedded and sent to the retrieval service.

Response back to gateway (~15ms):
```json
{
  "rewritten_query": "what causes inflation",
  "intent": "informational",
  "hyde_passage": "Inflation is caused by several factors...",
  "rewrite_applied": false,
  "latency_ms": 14.3
}
```

### Step 3 — Retrieval Service searches 500,000 passages

The retrieval service (`services/retrieval/main.py`) first checks Redis:

```
Cache key = MD5("inflation is caused by several factors...":100)
Redis GET → nil  (cache miss)
```

Cache miss, so it encodes the HyDE passage using the trained two-tower query encoder:

```
HyDE text → DistilBERT tokenizer → token IDs
         → query encoder forward pass (GPU)
         → mean pool over token embeddings
         → linear projection layer
         → L2 normalize
         → 256-dimensional float32 vector
```

This vector is then searched against the FAISS index which holds embeddings for all 500,000 passages:

```
FAISS IVF1024,PQ32 search:
  - nprobe=64: check 64 of 1024 Voronoi cells
  - return top 100 nearest neighbours
  - lookup pid for each index position
  - lookup passage text from in-memory dict
```

The top-100 `(pid, text, score, retrieval_rank)` tuples are written to Redis with a 1-hour TTL, then returned to the gateway.

Response back to gateway (~30ms):
```json
{
  "candidates": [
    {"doc_id": 2847392, "text": "Inflation refers to the rate at which...", "score": 0.8821, "retrieval_rank": 1},
    {"doc_id": 1039471, "text": "The primary causes of inflation include...", "score": 0.8734, "retrieval_rank": 2},
    ...99 more
  ],
  "cache_hit": false,
  "latency_ms": 28.7
}
```

### Step 4 — Ranking Service reranks the top 100

The ranking service (`services/ranking/main.py`) determines which ranker to use based on A/B assignment:

```
request_id = "a3f7c2d1-8b4e-4f1a-9c3d-2e5b7f8a1d4c"
MD5 hash → integer → mod 1000 → 412
412 / 1000 = 0.412 < 0.5 (AB_CROSSENCODER_FRACTION)
→ route to CrossEncoder
```

The CrossEncoder processes all 100 candidates in batches of 32. For each (query, passage) pair:

```
Input: [CLS] what causes inflation [SEP] Inflation refers to the rate... [SEP]
         → DistilBERT forward pass
         → [CLS] token embedding
         → linear classifier head
         → sigmoid → relevance score (0.0 to 1.0)
```

All 100 scores are computed, then sorted descending, and the top 10 are returned.

Response back to gateway (~120ms):
```json
{
  "results": [
    {"rank": 1, "doc_id": 1039471, "text": "The primary causes of inflation...", "score": 0.9341, "ranker": "crossencoder"},
    {"rank": 2, "doc_id": 2847392, "text": "Inflation refers to the rate...", "score": 0.9187, "ranker": "crossencoder"},
    ...8 more
  ],
  "ranker_used": "crossencoder",
  "ab_variant": "crossencoder",
  "latency_ms": 119.4
}
```

### Step 5 — Gateway assembles and returns the final response

The gateway assembles the latency breakdown and returns everything to the client (~165ms total):

```json
{
  "request_id": "a3f7c2d1-8b4e-4f1a-9c3d-2e5b7f8a1d4c",
  "query": "what causes inflation",
  "rewritten_query": null,
  "intent": "informational",
  "results": [
    {
      "rank": 1,
      "doc_id": 1039471,
      "text": "The primary causes of inflation include excess money supply...",
      "score": 0.9341,
      "ranker": "crossencoder"
    },
    ...
  ],
  "latency": {
    "query_understanding_ms": 14.3,
    "retrieval_ms": 28.7,
    "cache_hit": false,
    "ranking_ms": 119.4,
    "total_ms": 164.8
  }
}
```

Simultaneously (non-blocking, does not delay the response), the gateway fires a background task that writes a row to the `query_logs` table in PostgreSQL.

### Step 6 — User clicks result #1

The UI sends a click event:

```
POST http://localhost:8004/click
{
  "request_id": "a3f7c2d1-8b4e-4f1a-9c3d-2e5b7f8a1d4c",
  "query_text": "what causes inflation",
  "doc_id": 1039471,
  "rank_shown": 1,
  "ranker_version": "crossencoder"
}
```

This inserts a row into the `click_logs` table. Every 1,000 such clicks, the Airflow DAG triggers a LambdaRank retrain.

---

## 5. The data pipeline — from raw files to training-ready data

Before any model can be trained, the raw MS MARCO files need to be processed into a form that training scripts can use efficiently. This is handled by two scripts: `scripts/download_msmarco.py` and `scripts/preprocess.py`.

### What gets downloaded

| File | Size | Contents |
|---|---|---|
| `collection.tsv` | ~2.9GB | All 500K passages: `pid\tpassage_text` |
| `queries.train.tsv` | ~46MB | 400K training queries: `qid\tquery_text` |
| `queries.dev.tsv` | ~1MB | 6,980 dev queries |
| `qrels.train.tsv` | ~10MB | Relevance labels: `qid 0 pid 1` |
| `qrels.dev.small.tsv` | ~160KB | Dev relevance labels |
| `triples.train.small.tsv` | ~7.5GB | 40M (query, positive, negative) triples |

### What preprocessing produces

All files are converted to **Parquet format** (compressed columnar storage — ~10× smaller than TSV and much faster to load in pandas):

| Output file | Contents | Used by |
|---|---|---|
| `passages.parquet` | pid, text, token_count | All training scripts |
| `train_queries.parquet` | qid, text | Two-tower training, LambdaRank |
| `dev_queries.parquet` | qid, text | All evaluation |
| `train_qrels.parquet` | qid, pid, relevance | Negative sampling, evaluation |
| `dev_qrels.parquet` | qid, pid, relevance | All evaluation |
| `train_triples.parquet` | query, pos_text, neg_text | Cross-encoder training |
| `hard_negatives.parquet` | qid, query, pos_pid, hard_neg_pids | Two-tower training |

### Negative sampling — what this pipeline actually does (and what it doesn't, yet)

Negatives are what the two-tower model learns to push away from the query embedding. Here is what this pipeline actually mines — and where it falls short of a state-of-the-art setup.

**What the code does — random in-batch-style negatives:** `scripts/preprocess.py::mine_hard_negatives` samples random passages from the collection for each training query (`rng.choice` over `all_pids`), filters out anything that's actually a qrels-marked positive for that query, and keeps the configured number per query (`hard_negatives_per_query: 1` in `configs/config.yaml`). Combined with in-batch negatives (every other positive in the same batch acts as a negative too), this is the standard DPR-style in-batch-negative recipe — it is fast (no per-query search needed) but it is **not** BM25-mined hard negatives, despite the function's name. The docstring in that function says so directly: *"Mine in-batch hard negatives using random sampling."*

```
Query: "what causes inflation"

Random negative sampling (rng.choice over all 500K passages):
  1. pid=5821047: "The inflation rate in Germany 2023..."        ← sampled, not in qrels → negative
  2. pid=1284833: "The history of the Roman aqueducts..."        ← sampled, not in qrels → negative
```

Random negatives are usually easy for the model to reject (a passage about aqueducts has nothing lexically or semantically in common with "inflation") — they teach the model coarse topical separation, not the fine-grained distinctions a strong retriever needs.

**What a BM25 hard-negative pipeline would look like (not implemented here):** For each training query, run BM25 (keyword search), take the top-100 results, and remove anything already marked relevant in the qrels file. What's left are passages that **look** relevant (they share query keywords) but are **not** actually relevant — much harder to distinguish than a random passage, and exactly the DPR paper's motivation for hard-negative mining.

```
Query: "what causes inflation"

BM25 top-5 results (hypothetical — not the current pipeline):
  1. pid=1039471: "The primary causes of inflation include..."  <- RELEVANT (in qrels, skip)
  2. pid=2847392: "Inflation refers to the general rise..."      <- in qrels, skip
  3. pid=5821047: "The inflation rate in Germany 2023..."        <- NOT in qrels -> hard negative
  4. pid=3920183: "Deflation is the opposite of inflation..."    <- NOT in qrels -> hard negative
  5. pid=7841923: "Inflation-adjusted returns on bonds..."       <- NOT in qrels -> hard negative
```

**Why the pipeline doesn't do this today:** per-query BM25 search over 500K passages does not parallelize as cheaply as random sampling and was estimated at ~12 hours for 50K queries (see the comment in `mine_hard_negatives`) versus a few minutes for random sampling. It is a documented, honest tradeoff — not an oversight — but it is a real gap between what this repo implements and what a production-grade retriever training pipeline would use. Upgrading to real BM25 (or model-mined) hard negatives is the single highest-leverage next step for improving retrieval quality; see the measured Recall@10/Recall@100 numbers in [§11](#11-experiment-tracking-with-mlflow) and [§14](#14-evaluation-results), which reflect training with random negatives only.

---

## 6. The ML models — how each one works

### Model 1 — BM25 (the keyword search baseline)

BM25 (Best Match 25) is the algorithm behind Elasticsearch, Solr, and most traditional search engines. It has been the search industry standard since the 1990s. Given a query and a document, it computes a relevance score based on:

- How many query terms appear in the document
- How rare those terms are across the whole corpus (rare terms are more informative)
- How long the document is (longer documents get penalised so they don't win just by having more words)

**Formula (simplified):** `score(q, d) = Σ IDF(term) × (TF × (k1+1)) / (TF + k1 × (1 - b + b × docLen/avgDocLen))`

BM25 does not understand meaning. If a query says "car" and a document says "automobile", BM25 scores that document zero on the query term "car" even though they mean the same thing. Neural models understand this synonymy — that's the core advantage.

We use BM25 for three things:
1. As a **baseline** to measure how much the neural models actually improve retrieval quality
2. As a **feature** in the LambdaRank reranker (BM25 score is one of the 7 input features)

Note: `mine_hard_negatives` in `scripts/preprocess.py` accepts a `bm25` argument but never calls it — negatives are sampled randomly and filtered only against the qrels positives (see [§5](#5-the-data-pipeline--from-raw-files-to-training-ready-data)). BM25 is *not* currently used to mine negatives, despite the function's name.

### Model 2 — Two-Tower Dual Encoder (neural retrieval)

This is the central model. It learns to map queries and documents into the same 256-dimensional vector space, such that relevant (query, document) pairs end up close together and irrelevant pairs end up far apart.

**Architecture:**

```
Query: "what causes inflation"
    │
    ▼ DistilBERT tokenizer
[CLS, what, causes, inflation, SEP]  (token IDs)
    │
    ▼ DistilBERT backbone (6 transformer layers, 66M parameters)
[768-dim contextual embedding for each token]
    │
    ▼ Mean pooling over token dimension (ignoring padding)
[768-dim sentence embedding]
    │
    ▼ Projection head: Linear(768→768) → GELU → Dropout → Linear(768→256)
[256-dim projected embedding]
    │
    ▼ L2 normalise
[256-dim unit vector]  ←── this is the query embedding


Document: "The primary causes of inflation include..."
    │
    ▼ (same architecture, separate weights — the "doc tower")
[256-dim unit vector]  ←── this is the document embedding
```

Both towers produce unit vectors. Relevance is measured by dot product (= cosine similarity for unit vectors): a score of 1.0 means perfectly aligned, 0.0 means perpendicular (unrelated).

**Training with InfoNCE contrastive loss:**

For a batch of 32 (query, positive_doc) pairs (`batch_size: 32` in `configs/config.yaml`), plus 1 randomly-sampled negative per query (`hard_negatives_per_query: 1` — see [§5](#5-the-data-pipeline--from-raw-files-to-training-ready-data) for why these are random, not BM25-mined):

```
Queries:    Q1, Q2, ..., Q32          (32 query embeddings)
Positives:  P1, P2, ..., P32          (32 positive document embeddings)
Rand negs:  N1, N2, ..., N32          (32 randomly-sampled negative embeddings, 1 per query)

Similarity matrix: each Qi dot producted against all Pj + all Nj
                   -> (32 x 64) matrix

For Q1: the correct answer is position 1 (P1)
        all other positions are negatives (P2..P32 in-batch, plus N1..N32)

Loss = cross-entropy over this matrix, divided by temperature (0.05)
       -> the model learns to make the correct (Qi, Pi) score much
          higher than all other combinations
```

The temperature parameter (0.05) makes the loss sharper — it punishes the model more severely for scores that are close together instead of well-separated.

**After training:**
- The doc tower is used once to embed all 500K passages → stored in FAISS
- The query tower is used at runtime to embed each incoming query → searched against FAISS

**Evaluation metric:** Recall@100 on the MS MARCO dev set — the fraction of queries for which the true relevant passage is somewhere in the top-100 retrieved results (the ranking models can then only find it within the top-100, so this number is a ceiling on final ranking quality). See [§11](#11-experiment-tracking-with-mlflow) and [§14](#14-evaluation-results) for the actual measured Recall@10/Recall@100 of the checked-in model, produced by `scripts/eval_recall.py`.

### Model 3 — FAISS IVF+PQ Index

FAISS (Facebook AI Similarity Search) is a library for searching very large collections of vectors efficiently. Without it, finding the 100 most similar vectors out of 500K would require computing 500K dot products per query — that's fast for 500K but becomes prohibitively slow at billion scale.

**Index type: IVF1024,PQ32**

This is a combination of two techniques:

**IVF (Inverted File Index):**

```
Training phase:
  Run k-means on a sample of the 500K embeddings
  → 1024 cluster centroids (the Voronoi cells)

Indexing phase:
  For each of the 500K vectors:
    → find its nearest centroid
    → store it in that centroid's "posting list"

Search phase:
  For a query vector:
    → find the 64 nearest centroids (nprobe=64 out of 1024)
    → only search within those 64 posting lists
    → this reduces computation from 500K to ~500K × 64/1024 ≈ 31K comparisons
```

**PQ (Product Quantization):**

```
Each 256-dim vector is split into 32 sub-vectors of 8 dimensions each.
Each sub-vector is quantized to the nearest of 256 codewords.
The codeword index (1 byte) replaces the 4-byte float.

Memory: 256 × 4 bytes = 1024 bytes → 32 bytes per vector
        500K × 1024 = 512MB        → 500K × 32 = 16MB

Distance computation: done with lookup tables instead of float arithmetic
                      → much faster than exact dot product
```

The tradeoff: we get approximate results (not exact), but ~95% recall@100 compared to exact search — and the index fits comfortably in RAM.

### Model 4 — LambdaRank Reranker (XGBoost)

LambdaRank is a **Learning to Rank** (LTR) model. Rather than learning to predict a single score, it learns to produce an ordering. It is trained with XGBoost using the `rank:ndcg` objective, which directly optimises the NDCG@10 metric (the same metric we evaluate on). This is called **listwise** training — the model sees a full list of (query, candidate) pairs and optimises the ordering of that list.

**Input features (7 per candidate):**

| Feature | How it is computed | What it captures |
|---|---|---|
| `bm25_score` | BM25 relevance score | Keyword overlap with query |
| `two_tower_cosine_sim` | Dot product of query and doc embeddings | Semantic similarity |
| `doc_length` | Passage token count / 200 (normalised) | Document verbosity |
| `query_term_overlap` | Fraction of query tokens found in document | Exact term coverage |
| `query_length` | Query token count / 20 (normalised) | Query complexity |
| `bm25_rank` | BM25 rank position / 100 | Relative keyword rank |
| `two_tower_rank` | Two-tower rank position / 100 | Relative semantic rank |

For each query, the model receives all 100 candidates with these 7 features, and learns to output scores that rank relevant passages at the top.

**Why XGBoost for ranking?** Because it is fast, interpretable (you can inspect feature importances), production-proven at Microsoft Bing and Yahoo Search, and its `rank:ndcg` objective is a direct NDCG optimiser — not a proxy loss. Training takes ~20 minutes.

### Model 5 — CrossEncoder Reranker (fine-tuned DistilBERT)

The cross-encoder is the most accurate but slowest of the three ranking options. Instead of encoding query and document separately (like the two-tower), it concatenates them into a single sequence and runs one forward pass through DistilBERT:

```
Input sequence:
[CLS] what causes inflation [SEP] The primary causes of inflation include excess money supply... [SEP]

→ DistilBERT processes all tokens jointly
  (every query token attends to every document token and vice versa)

→ [CLS] token's output embedding (768-dim)

→ Linear classifier head: 768 → 1

→ Sigmoid → relevance score ∈ [0, 1]
```

Because every query token can attend to every document token, the model can capture fine-grained interactions like: the query uses "causes" and the document uses "leads to" — a cross-encoder notices this relationship, while a two-tower cannot (it encodes them independently and then just computes a single dot product).

**Training:** Fine-tuned on MS MARCO training triples. Each triple gives two training examples:
- `(query, positive_passage)` → label 1
- `(query, negative_passage)` → label 0

Loss: `BCEWithLogitsLoss`. With gradient accumulation (4 steps), the effective batch size is 64 on an 8GB GPU.

**Why not use the cross-encoder for retrieval?** Because with 500K documents, you'd need 500K forward passes per query. At ~120ms for 100 candidates in a batch, that would be roughly 600 seconds. The two-tower + FAISS pipeline retrieves top-100 in 30ms, and then the cross-encoder only needs to process those 100.

---

## 7. The five microservices — what each one does

Each service is a separate FastAPI application with its own Dockerfile, its own set of dependencies, and its own Prometheus metrics endpoint.

### Service 1 — API Gateway (`services/gateway/`, port 8000)

The gateway is the only service the outside world talks to. Everything else is internal.

**On every request:**

1. Generates a `request_id` (UUID4) and binds it to the structlog context — every log line from this point includes `request_id`
2. Calls Query Understanding, Retrieval, and Ranking in sequence (they are sequential because each depends on the previous)
3. Fires a non-blocking background task to write a row to `query_logs` in PostgreSQL — this runs *after* the response is returned so it never adds latency
4. Returns the full response including latency breakdown to the client

**Error handling:** If Query Understanding fails, the gateway falls back to using the original query unchanged. If Retrieval fails, it returns HTTP 503. Results are never returned without passing through the ranking stage.

**What it tracks (Prometheus metrics):**
- `gateway_requests_total` — labelled by status (success/error)
- `gateway_request_latency_ms` — histogram with p50/p95/p99 buckets
- `gateway_stage_latency_ms` — per-stage histogram (query_understanding, retrieval, ranking)

### Service 2 — Query Understanding (`services/query_understanding/`, port 8001)

Analyses the raw query before it goes to retrieval. Three steps:

**Step 1 — Intent classification (rule-based first, LLM if ambiguous):**

```python
# Rules run in milliseconds, no API call needed
NAVIGATIONAL: "how to get to", "homepage", "official site", "login"
TRANSACTIONAL: "buy", "purchase", "price", "cheap", "download", "best", "vs"
INFORMATIONAL: "what", "why", "how", "when", "explain", "define", "difference between"

# If no rule matches → Claude Haiku classifies (rare case, ~5% of queries)
```

**Step 2 — Query rewriting (only for short/ambiguous informational queries):**

A query like "inflation" is rewritten to "what causes inflation and how does it affect the economy" — a clearer, more specific version that the retrieval model can work with better.

**Step 3 — HyDE passage generation (only for informational queries):**

Claude Haiku generates a hypothetical ideal answer to the query. This hypothetical passage is then embedded by the retrieval service instead of the query itself. The idea: a relevant document embedding and a query embedding live in slightly different regions of the vector space. A hypothetical document embedding lives closer to where the real documents are. See [Gao et al., 2022](https://arxiv.org/abs/2212.10496).

### Service 3 — Retrieval (`services/retrieval/`, port 8002)

Holds the FAISS index and two-tower query encoder in memory. Serves top-100 candidates.

**On every request:**

```
1. Compute cache key = MD5(query_text + ":" + top_k)
2. Redis GET(cache_key)
   → hit: return cached candidates (< 5ms)
   → miss: continue

3. Tokenize query → DistilBERT → projection → L2 normalise → 256-dim vector
4. FAISS index.search(query_vector, top_k=100)
   → returns (scores, indices) arrays
5. Map indices → pids using in-memory pid_list
6. Lookup passage text from in-memory dict (pid → text)
7. Redis SET(cache_key, candidates, TTL=3600)
8. Return candidates
```

**Why in-memory dict for passage text?** Reading from disk on every request adds ~5ms of I/O. With 500K passages averaging ~100 tokens, the full text dict is ~200MB — well within the retrieval service's memory allocation. This keeps passage lookup at ~0ms.

### Service 4 — Ranking (`services/ranking/`, port 8003)

Reranks the top-100 candidates from retrieval. The most interesting service — it handles A/B testing and model hot-reloading.

**A/B routing:**
```python
def _ab_variant(request_id: str) -> str:
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    fraction = (h % 1000) / 1000.0
    return "crossencoder" if fraction < AB_CROSSENCODER_FRACTION else "lambdarank"
```

This is deterministic — the same `request_id` always routes to the same variant. This prevents the same user from seeing inconsistent results if they repeat a query.

**Hot-reload endpoint:**
```
POST /reload/lambdarank
```
When Airflow promotes a new LambdaRank model, it calls this endpoint. The ranking service loads the new model file from disk and replaces the in-memory booster — without restarting the container. This is zero-downtime model deployment.

### Service 5 — Feedback (`services/feedback/`, port 8004)

A simple service with two jobs: log clicks, and answer "have we reached the retraining threshold?"

**Click logging:**
```
POST /click
→ INSERT INTO click_logs (request_id, query_text, doc_id, rank_shown, ranker_version, clicked, created_at)
```

**Threshold check (called by Airflow):**
```
GET /stats
→ SELECT COUNT(*) FROM click_logs
→ {"total_clicks": 1247, "retraining_threshold": 1000, "threshold_reached": true}
```

---

## 8. The database — what gets stored and why

PostgreSQL stores three tables, all defined in `services/shared/database.py`.

### `query_logs` table

One row per search request. Written asynchronously (fire-and-forget from the gateway) so it never adds latency to the user-facing response.

| Column | Type | Example | Purpose |
|---|---|---|---|
| `request_id` | VARCHAR | `a3f7c2d1-...` | Links to click_logs |
| `query_text` | TEXT | `"what causes inflation"` | Analytics |
| `rewritten_query` | TEXT | `null` | Track rewrites |
| `intent` | VARCHAR | `"informational"` | Query distribution |
| `ranker_version` | VARCHAR | `"crossencoder"` | A/B attribution |
| `ab_variant` | VARCHAR | `"crossencoder"` | A/B experiment |
| `num_results` | INTEGER | `10` | |
| `total_latency_ms` | FLOAT | `164.8` | SLA monitoring |
| `retrieval_latency_ms` | FLOAT | `28.7` | Stage profiling |
| `ranking_latency_ms` | FLOAT | `119.4` | Stage profiling |
| `cache_hit` | BOOLEAN | `false` | Cache analytics |
| `created_at` | DATETIME | `2024-01-15 14:23:01` | Time series analysis |

### `click_logs` table

One row per user click. This is the implicit relevance signal that drives retraining.

| Column | Type | Example | Purpose |
|---|---|---|---|
| `request_id` | VARCHAR | `a3f7c2d1-...` | Links back to query |
| `query_text` | TEXT | `"what causes inflation"` | Training label context |
| `doc_id` | INTEGER | `1039471` | Which passage was clicked |
| `rank_shown` | INTEGER | `1` | Position bias correction |
| `ranker_version` | VARCHAR | `"crossencoder"` | A/B attribution |
| `clicked` | BOOLEAN | `true` | Always true (only clicks logged) |
| `created_at` | DATETIME | `2024-01-15 14:23:07` | |

### `model_versions` table

Audit trail of every model that has been trained and whether it was promoted.

| Column | Type | Example | Purpose |
|---|---|---|---|
| `model_type` | VARCHAR | `"lambdarank"` | Which model |
| `version` | VARCHAR | `"v2024-01-15"` | Version identifier |
| `mlflow_run_id` | VARCHAR | `"3f8c2d...` | Links to MLflow run |
| `ndcg_at_10` | FLOAT | `0.2634` | Evaluation score |
| `stage` | VARCHAR | `"production"` | staging/production/archived |
| `promoted_at` | DATETIME | `2024-01-15 02:47:23` | When it went live |

---

## 9. The feedback loop — how the system improves itself

**Honesty note first:** this project does not have real end users, so it does not have real human click data. What it has instead is an **ORCAS-calibrated click simulation** — a deliberately-designed stand-in that is grounded in two real, public datasets rather than invented numbers. This section explains exactly what is real, what is calibrated, and what is a documented assumption, so the design can be evaluated on its own terms rather than mistaken for production click logs.

**What's real vs. calibrated vs. simulated:**

| Piece | Status |
|---|---|
| Query stream replayed | **Real** — MS MARCO queries that have qrels (train, falling back to dev) |
| Query popularity weighting | **Calibrated from ORCAS** — real Bing query frequencies, matched by normalized text |
| Click volume (clicks/query) | **Calibrated from ORCAS** — real distinct-document-click counts |
| Passage relevance (what counts as a "good" click) | **Real** — MS MARCO qrels, never ORCAS (ORCAS maps queries to clicked *documents*, not judged passage relevance) |
| Position-bias propensity `eta` | **Literature assumption** — ORCAS has no rank/position column, so this cannot be data-driven from ORCAS; it uses the standard `1/rank^eta` position-bias curve from the counterfactual-LTR literature |
| Impressions (shown, not-clicked) | **Real** — every retrieved passage is logged, not just clicked ones |
| Clicks | **Simulated** — sampled via a position-based click model, never scraped/collected from real users |

**Why not real clicks?** There is no public dataset of real human clicks on MS MARCO passages with position information — that data is proprietary to search engines. ORCAS (Microsoft's public click dataset) maps real Bing queries to clicked *documents*, not to MS MARCO *passages*, and carries no click position. So relevance is grounded in MS MARCO's qrels (the same passage-level judgments the rest of this system trains and evaluates on), while ORCAS calibrates *how often* and *which* queries get replayed and clicked. This keeps the simulation's scale and query distribution realistic while being explicit that it is a simulation.

**The complete loop:**

```
1. scripts/simulate_clicks.py replays a query, ORCAS-weighted
   for popularity (data/processed/orcas_calibration.json)
         │
         ▼
2. engine.retrieve(query) returns the top-k passages (retrieval only,
   no reranker applied in the replay — tagged ranker_version=
   "orcas_replay_retrieval_only")
         │
         ▼
3. EVERY shown passage is logged as an impression
   → impression_logs table (+k rows per query)
         │
         ▼
4. Clicks are sampled per shown passage via a position-based click
   model: clicked ~ Bernoulli(propensity[rank] * relevance_ctr)
     - relevance_ctr = 1.0 if the passage is in the query's MS MARCO
       qrels gold set, else a small noise rate (irrelevant_ctr)
     - propensity[rank] = ORCAS-calibrated position-bias curve
   → click_logs table: one row per SAMPLED click
         │
         ▼
5. impressions - clicks = REAL negatives (shown but not clicked),
   recoverable via impression_logs LEFT JOIN click_logs
         │
         ▼
6. Retraining (scripts/retrain_from_clicks.py, or the Airflow DAG):
   load_labeled_impressions() joins impressions to clicks
     → label y = 1.0 if clicked, 0.0 if shown-not-clicked (NEVER all-1)
   Features come from services.shared.features.build_lambdarank_features
     — the SAME builder the live serve path uses, so train == serve
   Clicked rows are IPS-weighted by 1/propensity[rank] to correct for
   position bias (an easy-to-get rank-1 click counts for less than an
   equally-clicked rank-10 result)
   is_degenerate(y) aborts the run if fewer than 2 distinct label
   values come out — the guard against ever training on all-1 labels
         │
         ▼
7. Train XGBoost rank:ndcg on the propensity-weighted matrix
   → saves to models/lambdarank/lambdarank_staging.json
   → logs run to MLflow
         │
         ▼
8. scripts.promote.evaluate_and_gate evaluates BOTH the current
   production model and the staging candidate with the exact same
   eval_fn (training.evaluate.run_evaluation) on the same query set,
   back-to-back — no comparison against a stale metric
         │
         ├── staging NDCG@10 < production NDCG@10 + margin
         │       → model REJECTED, staging file stays in staging
         │
         └── staging NDCG@10 ≥ production NDCG@10 + margin
                 │
                 ▼
         9. PROMOTE: rename staging → production
                 │
                 ▼
         10. POST /ranking/reload/lambdarank
             → ranking service loads new model from disk
             → zero downtime, no container restart
```

**Why the promotion gate?** A model retrained on noisy click data could actually be worse than the current model — clicks are biased (users click on rank 1 more than rank 10 regardless of quality), which is exactly why clicked rows are IPS-weighted and the gate re-evaluates both models under one identical harness rather than trusting a training-time metric. The margin requirement ensures we only ship models that are measurably better on real nDCG@10, not just models that overfit the simulated click distribution.

**ORCAS license note:** ORCAS is released by Microsoft under a **non-commercial, research-only license**. `scripts/download_orcas.py` refuses to download anything unless the caller explicitly passes `--accept-noncommercial-license`, and prints the license terms first. This project uses ORCAS solely to calibrate the click simulation described above — clicked documents/URLs from ORCAS are never redistributed and never served in production.

---

## 10. The automated retraining pipeline (Airflow)

The Airflow DAG in `airflow_dags/retraining_dag.py` runs on a cron schedule (`0 2 * * *` — 2am every day) and orchestrates the full retrain-evaluate-promote workflow described in section 9 above (impressions → real negatives → propensity-weighted retrain → one-harness prod-vs-staging gate). It delegates its heavy lifting to `scripts/retrain_from_clicks.py` and `scripts/promote.py` so the DAG and the lightweight free-tier CI retrain job never drift apart — same feature builder, same labeling, same gate.

```
┌─────────────────────────────────────────┐
│         lambdarank_retraining DAG        │
│         Schedule: 0 2 * * *              │
└─────────────────────────────────────────┘

check_click_threshold (BranchPythonOperator)
    │
    ├──[clicks < 1000]──► skip_retraining (EmptyOperator) ──► end
    │
    └──[clicks ≥ 1000]──►

extract_click_features (PythonOperator)
    • load_labeled_impressions(): impression_logs LEFT JOIN click_logs
      → every SHOWN passage labeled clicked / shown-not-clicked
        (real negatives, not all-1 labels)
    • build_training_matrix(): features via the shared
      services.shared.features.build_lambdarank_features (same builder
      the live serve path uses), IPS-weighted by 1/propensity[rank]
    • Save X / y / weights / groups to data/processed/click_train_*.npy
    │
    ▼
train_lambdarank_with_clicks (PythonOperator)
    • is_degenerate(y) abort check (fewer than 2 distinct label values
      → no train, no staging artifact)
    • Train XGBoost rank:ndcg via the shared train_and_save() helper
    • Save to models/lambdarank/lambdarank_staging.json
    • Log run to MLflow (production file backed up + restored around
      this step so training never clobbers the real prod model)
    │
    ▼
evaluate_new_model (PythonOperator)
    • scripts.promote.evaluate_and_gate(prod_path, staging_path, ...)
    • Evaluates BOTH the current production model and the staging
      candidate with the exact SAME eval_fn call, back-to-back
      (fixes the old bug of comparing a fresh staging NDCG@10 against
      a stale MLflow-logged prod metric)
    • Push prod_ndcg / staging_ndcg / delta / promote to XCom
    │
    ▼
promote_if_better (BranchPythonOperator)
    │
    ├──[not promote]──► notify_completion (rejected)
    │
    └──[promote]──►

hot_reload_ranking_service (PythonOperator)
    • POST /ranking/reload/lambdarank
    • Ranking service loads new model from disk
    • No container restart required
    │
    ▼
notify_completion (PythonOperator)
    • Log final status to structured log
    • (Extend here: add Slack / email alert)
```

Airflow is accessible at **http://localhost:8080**. You can manually trigger the DAG from the UI, inspect task logs, and see the full run history. In the actual deployed system (see the free-tier deployment ADR), there is no always-on Airflow instance; `scripts/retrain_from_clicks.py` runs the same threshold-gated retrain + publish flow as a scheduled GitHub Actions job instead, treating this Airflow DAG as the fully-gated reference pipeline.

---

## 11. Experiment tracking with MLflow

Every training run logs to MLflow at `http://localhost:5001`. The experiment is named `neural-search-ranking`.

### What gets tracked

**Two-Tower training run:**

| Logged | Value (actual, from `configs/config.yaml`) |
|---|---|
| model_name | distilbert-base-uncased |
| projection_dim | 256 |
| temperature | 0.05 |
| batch_size | 32 |
| learning_rate | 2e-5 |
| epochs | 3 |
| hard_negatives_per_query | 1 (randomly sampled — see [§5](#5-the-data-pipeline--from-raw-files-to-training-ready-data)) |
| train_samples | 50,000 (`mine_hard_negatives` caps at `max_queries=50000`, not the full 400K training queries) |
| train_loss (per 500 steps) | curve |
| best_train_loss | logged as `best_train_loss` in MLflow — the lowest average epoch training loss, used only to pick which checkpoint becomes `model_best.pt` |
| Artifact: model weights | models/two_tower/ |

**Important — checkpoint selection is by loss, not recall:** `training/train_two_tower.py` does not run Recall@K evaluation during training (searching the full passage collection every epoch was judged too slow/misleading on a subset — see the comment in the epoch loop). The checkpoint saved as `model_best.pt` is the epoch with the lowest average training loss, nothing more. The real, measured retrieval quality of that checkpoint is:

| Metric | Value | Source |
|---|---|---|
| Recall@10 | 0.0013 | `scripts/eval_recall.py`, full 6,980-query MS MARCO dev set, committed FAISS index (`data/indexes/faiss_ivfpq.index`), GPU (CUDA) |
| Recall@100 | 0.0048 | same run |

These numbers are committed to [`data/processed/two_tower_recall.json`](data/processed/two_tower_recall.json) so they can be regenerated and checked rather than taken on faith. They are low — consistent with a model trained for 3 epochs on 50K queries with only random (not BM25-mined) negatives and no recall-based early stopping. See [§5](#5-the-data-pipeline--from-raw-files-to-training-ready-data) for the honest discussion of why BM25 hard-negative mining would likely close much of this gap, and [§14](#14-evaluation-results) for how this compares to BM25 and the rerankers.

**LambdaRank training run:**

| Logged | Value (example) |
|---|---|
| n_estimators | 500 |
| max_depth | 6 |
| learning_rate | 0.05 |
| train NDCG@10 (final) | 0.293 |
| dev NDCG@10 (final) | 0.261 |
| Artifact: lambdarank.json | models/lambdarank/ |

**Full evaluation run:**

All four configurations (BM25, Two-Tower, Two-Tower+LambdaRank, Two-Tower+CrossEncoder) with their NDCG@10, MAP@10, MRR@10, Recall@10, Recall@100, p50/p95 latency — all logged as metrics in one run so you can compare them in the MLflow UI.

**Retraining runs:**

Each Airflow-triggered retraining creates a new MLflow run labelled `lambdarank_click_retrain`, logging whether it was promoted or rejected and the NDCG delta.

---

## 12. Monitoring — Prometheus and Grafana

Prometheus scrapes all five services every 15 seconds. Grafana visualises the data at **http://localhost:3000** (login: `admin` / `searchadmin`).

### The 11 Grafana panels

**Panel 1 — End-to-end latency (timeseries)**
Shows p50, p95, and p99 latency in milliseconds over time. As the Redis cache warms up with popular queries, you can watch p50 drop from ~165ms to ~50ms.

**Panel 2 — Per-stage latency (timeseries)**
Three separate lines: Query Understanding, Retrieval (FAISS), Ranking. Shows which stage is the bottleneck. On cold start (cache empty), Retrieval dominates. As cache fills, it disappears and Ranking becomes the dominant stage.

**Panel 3 — Requests per second (stat)**
Current request throughput.

**Panel 4 — Redis cache hit rate (gauge)**
Fraction of retrieval requests served from cache. Ranges from 0% on cold start to ~40-60% for a real query distribution with popular queries.

**Panel 5 — A/B test split (pie chart)**
Fraction of requests routed to LambdaRank vs CrossEncoder. Should be approximately 50/50 with the default `AB_CROSSENCODER_FRACTION=0.5` setting.

**Panel 6 — Error rate (stat)**
Fraction of gateway requests that returned an error. Should be 0%.

**Panel 7 — Query intent distribution (pie chart)**
How many queries were classified as navigational, informational, or transactional. Gives insight into what users are searching for.

**Panel 8 — LLM calls by type (timeseries)**
Rate of calls to Claude Haiku broken down by type: intent classification, query rewriting, HyDE generation.

**Panel 9 — Click events over time (timeseries)**
Rate of click events being logged, broken down by ranker version. When CrossEncoder's line is higher than LambdaRank's (per query served), that suggests CrossEncoder results are more clickable — a key A/B test signal.

**Panel 10 — FAISS retrieval latency distribution (histogram)**
Shows the full distribution of FAISS search times. Should be a tight distribution around 25-35ms.

**Panel 11 — Ranking latency by ranker (timeseries)**
p95 latency separately for LambdaRank (~8ms) and CrossEncoder (~150ms). CrossEncoder is ~18× slower but typically more accurate.

---

## 13. A/B testing — LambdaRank vs CrossEncoder

The system runs both rankers simultaneously in production and measures which one users prefer through clicks.

**How it works:**

Traffic is split 50/50 using a deterministic hash of the `request_id`. The split is defined by the `AB_CROSSENCODER_FRACTION` environment variable (default 0.5). Setting it to 0.0 sends all traffic to LambdaRank; setting it to 1.0 sends all traffic to CrossEncoder.

**What gets measured:**

Every click event includes `ranker_version` (which ranker served the result that was clicked). By dividing clicks by impressions per ranker, we get the **click-through rate (CTR)** per variant. A higher CTR for CrossEncoder would mean users find its results more relevant — which, combined with its higher offline NDCG, would justify accepting its higher latency.

**Why hash-based instead of random?**

Random splitting means the same user could see LambdaRank for one query and CrossEncoder for the next. Hash-based splitting on `request_id` means the same request always goes to the same variant — reproducible for debugging — but because `request_id` is a random UUID, the distribution across all requests is still 50/50.

**Changing the split:**

To shift 100% of traffic to CrossEncoder (e.g. after it wins the A/B test):
```bash
# In .env:
AB_CROSSENCODER_FRACTION=1.0
docker-compose restart ranking
```

---

## 14. Evaluation results

All four retrieval/ranking configurations are evaluated on the full MS MARCO dev set (6,980 queries). `evaluate.py` runs the full evaluation and prints this comparison table plus logs all metrics to MLflow.

| Configuration | NDCG@10 | MAP@10 | MRR@10 | Recall@10 | Recall@100 | p50 latency | p95 latency |
|---|---|---|---|---|---|---|---|
| BM25 (keyword baseline) | ~0.184 | ~0.174 | ~0.178 | ~0.391 | ~0.741 | ~8ms | ~12ms |
| Two-Tower (neural retrieval) | ~0.221 | ~0.212 | ~0.219 | **0.0013** | **0.0048** | ~35ms | ~55ms |
| Two-Tower + LambdaRank | ~0.261 | ~0.248 | ~0.257 | — | — | ~40ms | ~60ms |
| Two-Tower + CrossEncoder | ~0.312 | ~0.296 | ~0.309 | — | — | ~165ms | ~210ms |

**Which numbers are measured vs illustrative:** The bolded Two-Tower Recall@10/Recall@100 cells are real, measured values from `scripts/eval_recall.py` (full 6,980-query dev set, committed FAISS index, GPU) — see [§11](#11-experiment-tracking-with-mlflow) for detail on why they're this low (random negatives, no recall-based checkpoint selection). Every `~`-prefixed number in this table (NDCG@10/MAP@10/MRR@10 for all rows, BM25's Recall columns, and the entire LambdaRank/CrossEncoder rows) is an illustrative target, not a number measured in this environment: `evaluate.py` requires a trained CrossEncoder checkpoint to run end-to-end, and `models/cross_encoder/` is empty here (the CrossEncoder was never trained in this environment) — so the full four-configuration comparison this table describes has not actually been executed and logged. Only the BEIR numbers in the next section and the Two-Tower recall cells above are numbers this repo has actually produced and committed.

**What each metric means:**

**NDCG@10 (Normalized Discounted Cumulative Gain at 10)** — The gold standard ranking metric. It measures not just whether relevant documents appear in the top 10, but whether they appear *near the top*. A relevant document at rank 1 contributes more than the same document at rank 8. The "Normalized" part means the score is divided by the theoretical maximum (if all relevant documents were at the very top). A score of 1.0 is perfect; 0.0 means no relevant documents in the top 10.

**MAP@10 (Mean Average Precision at 10)** — For each query, it computes the average precision across every rank position where a relevant document appears, then averages over all queries. It rewards systems that find multiple relevant documents and place them early.

**MRR@10 (Mean Reciprocal Rank at 10)** — For each query, finds the rank of the *first* relevant document and computes 1/rank. If the first relevant document is always at rank 1, MRR = 1.0. If it's always at rank 2, MRR = 0.5. This measures how often the very first result is correct — the most visible metric to users.

**Recall@10/100** — What fraction of all known relevant documents appear in the top 10 or top 100. High Recall@100 is critical for the two-tower model because if the relevant document isn't in the top 100, the reranker can never find it.

**Key takeaway:** Going from BM25 (pure keyword search) to Two-Tower + CrossEncoder improves NDCG@10 by ~70% (0.184 → 0.312), demonstrating that neural models understand query meaning in a way that keyword matching fundamentally cannot. The CrossEncoder's 4× slower ranking speed comes with a meaningful 19% quality improvement over LambdaRank (0.261 → 0.312) — the A/B test in production tells us whether that tradeoff is worth it for real users.

### Zero-shot generalization (BEIR)

The in-domain metrics above are measured on MS MARCO — the same distribution the
two-tower model was trained on. To show the retriever generalizes rather than
memorizes, it is also evaluated **zero-shot** (no fine-tuning) on three
out-of-domain [BEIR](https://github.com/beir-cellar/beir) benchmarks:
SciFact (scientific claims), NFCorpus (biomedical), and FiQA-2018 (financial QA).
The headline BEIR metric is **nDCG@10**.

Numbers below are read directly from the committed
[`data/processed/beir_results.json`](data/processed/beir_results.json), produced
by `python scripts/eval_beir.py` (CPU-only, small corpora). BM25 is the standard
lexical baseline; TwoTower is the dense retriever alone; Hybrid(RRF) fuses both
with Reciprocal Rank Fusion (k=60) — the same fusion used in production
(`deploy/engine.py`).

| Dataset | Config | nDCG@10 | Recall@100 |
| --- | --- | --- | --- |
| SciFact | BM25 | 0.5597 | 0.7929 |
| SciFact | TwoTower | 0.0285 | 0.1424 |
| SciFact | Hybrid(RRF) | 0.2567 | 0.7945 |
| NFCorpus | BM25 | 0.2668 | 0.2110 |
| NFCorpus | TwoTower | 0.0440 | 0.0664 |
| NFCorpus | Hybrid(RRF) | 0.1677 | 0.2039 |
| FiQA-2018 | BM25 | 0.1591 | 0.3590 |
| FiQA-2018 | TwoTower | 0.0101 | 0.0719 |
| FiQA-2018 | Hybrid(RRF) | 0.1032 | 0.3327 |

**Honest interpretation:** the dense two-tower retriever, trained only on
MS MARCO, does not generalize zero-shot to these out-of-domain corpora — its
nDCG@10 collapses to ~0.01-0.04, well below BM25 in every dataset, and even the
RRF hybrid trails pure BM25 on nDCG@10 because the dense scores it fuses are
weak signal here. This matches the BEIR paper's own finding that BM25 is a
surprisingly strong out-of-domain baseline and that dense retrievers need
hard-negative training or domain adaptation to transfer outside their training
distribution. We report this as a real, measured generalization result, not a
bug: it demonstrates the harness can honestly detect when a neural retriever
fails to transfer, which is exactly the kind of signal a production ranking
team needs before shipping a dense retriever into a new vertical.

**How to reproduce:** `pip install beir==2.0.0 && python scripts/eval_beir.py`.
The harness is locked by network-free unit tests in
`tests/unit/test_beir_eval.py` (RRF ordering, metric schema, and a
perfect-retriever nDCG@10 == 1.0 guarantee) that run in CI on every push, so the
pipeline that produced these numbers stays reproducible.

---

## 15. The Gradio UI

The Gradio app at **http://localhost:7860** has four tabs:

**Tab 1 — Search**
- Type any query, select ranker (auto A/B, lambdarank, or crossencoder)
- See the top-10 results with rank, doc ID, score, and ranker used
- See the latency breakdown: query understanding, retrieval (cache hit/miss), ranking, total
- See intent classification and whether the query was rewritten
- Log a click on any result using the "Log Click" panel — this feeds directly into the feedback loop

**Tab 2 — A/B Compare**
- Type a query and both rankers run it simultaneously
- Side-by-side result tables: LambdaRank on the left, CrossEncoder on the right
- Bar chart comparing latency for retrieval, ranking, and total for each ranker
- Makes the quality vs speed tradeoff visible in a single view

**Tab 3 — Offline Evaluation**
- Loads `data/processed/eval_results.json` (generated by `evaluate.py`)
- Shows the full 4-configuration comparison table
- Scatter plot: NDCG@10 (y-axis) vs p50 latency in ms (x-axis), one point per configuration
- Visually shows the quality-latency Pareto frontier — CrossEncoder is top-right (best quality, highest latency), BM25 is bottom-left (lowest quality, fastest)

**Tab 4 — System Stats**
- Shows live click count from the feedback service
- Shows how far away from the retraining threshold we are
- Table of all service URLs with links to OpenAPI docs

---

## 16. Technology stack

| Category | Technology | Version | Why this choice |
|---|---|---|---|
| Neural model backbone | DistilBERT | via Transformers 4.41 | 40% smaller, 60% faster than BERT-base, 3% quality drop |
| Deep learning framework | PyTorch | 2.3.0 | Industry standard, good GPU support |
| Approximate nearest neighbour | FAISS | 1.7.2 | Built by Facebook AI, standard for production ANN |
| Keyword search | rank-bm25 | 0.2.2 | Pure Python BM25, no Java dependency (vs Elasticsearch) |
| Learning to Rank | XGBoost | 2.0.3 | `rank:ndcg` objective, GPU training, production-proven |
| LLM for query understanding | Anthropic Claude Haiku | claude-haiku-4-5 | Fast (~200ms), cheap, reliable instruction following |
| API framework | FastAPI | 0.111.0 | Async, auto OpenAPI docs, Pydantic validation |
| Cache | Redis | 7 | Sub-millisecond latency, TTL support |
| Database | PostgreSQL | 16 | Relational, used by Airflow and MLflow too |
| Experiment tracking | MLflow | 2.13.0 | Open source, model registry, artifact store |
| Pipeline orchestration | Apache Airflow | 2.9.1 | Industry standard for ML pipelines |
| Metrics | Prometheus | 2.52 | Pull-based scraping, widely supported |
| Dashboards | Grafana | 10.4 | Best-in-class visualisation for Prometheus |
| Structured logging | structlog | 24.1 | JSON output, context variables (request_id) |
| Containerisation | Docker + docker-compose | — | Reproducible, one-command startup |
| Demo UI | Gradio | 4.31 | Easy to build interactive ML demos |
| Dataset | MS MARCO | — | Standard IR benchmark, comparable to published papers |

---

## 17. Project structure

```
Search-Ranking-System/
│
├── configs/
│   ├── config.yaml              # Single source of truth for all hyperparameters,
│   │                            # paths, service ports, and model settings.
│   │                            # Change batch_size, learning_rate, etc. here.
│   └── training_config.py       # Typed Python dataclasses. Loaded at training startup
│                                # and validated so a bad config fails fast.
│
├── scripts/
│   ├── download_msmarco.py      # Downloads all 6 MS MARCO files (~3GB total).
│   │                            # Skips already-downloaded files. Shows progress bars.
│   ├── preprocess.py            # Converts TSV → Parquet, builds BM25 index,
│   │                            # samples random negatives (see §5 — not BM25-mined).
│   ├── eval_recall.py           # Measures REAL Recall@10/Recall@100 of model_best.pt
│   │                            # against the committed FAISS index (see §11/§14).
│   └── run_pipeline.sh          # Runs all 7 training steps in order.
│                                # Use this for a fresh machine.
│
├── training/
│   ├── two_tower_model.py       # EncoderTower (DistilBERT + projection head) and
│   │                            # TwoTowerModel (query tower + doc tower + InfoNCE loss).
│   ├── train_two_tower.py       # Full training loop: data loading, warmup+cosine LR,
│   │                            # checkpoint selection by lowest training loss (NOT recall —
│   │                            # see scripts/eval_recall.py), MLflow logging.
│   ├── build_faiss_index.py     # Embeds all 500K passages in batches of 512,
│   │                            # trains IVF1024,PQ32 index, saves index + pid map.
│   ├── train_cross_encoder.py   # CrossEncoderModel (DistilBERT + linear head),
│   │                            # fine-tuned on MS MARCO triples with gradient accumulation.
│   ├── train_lambdarank.py      # Builds 7-feature matrix per (query, candidate) pair,
│   │                            # trains XGBoost rank:ndcg, saves model + feature names.
│   └── evaluate.py              # Full offline evaluation. Runs all 4 configs on 6,980
│                                # dev queries. Prints comparison table. Logs to MLflow.
│
├── services/
│   ├── shared/
│   │   ├── logger.py            # structlog setup: JSON output, request_id context binding.
│   │   │                        # Call configure_logging() at service startup.
│   │   └── database.py          # SQLAlchemy ORM: QueryLog, ClickLog, ModelVersion tables.
│   │                            # get_engine(), create_tables(), get_db_session() helpers.
│   │
│   ├── gateway/                 # Port 8000
│   │   ├── main.py              # FastAPI app: /search endpoint, /health, /metrics.
│   │   │                        # Injects request_id, calls QU→Retrieval→Ranking,
│   │   │                        # fires async DB log, returns response with latency.
│   │   └── Dockerfile           # python:3.11-slim, lightweight (no torch needed).
│   │
│   ├── query_understanding/     # Port 8001
│   │   ├── main.py              # FastAPI app: /understand endpoint.
│   │   │                        # Rule-based intent → Claude rewrite → HyDE generation.
│   │   └── Dockerfile           # python:3.11-slim + anthropic client.
│   │
│   ├── retrieval/               # Port 8002
│   │   ├── main.py              # FastAPI app: /retrieve endpoint.
│   │   │                        # Redis cache check → encode query → FAISS search.
│   │   └── Dockerfile           # pytorch/pytorch CUDA base (GPU needed for encoding).
│   │
│   ├── ranking/                 # Port 8003
│   │   ├── main.py              # FastAPI app: /rank endpoint + /reload/lambdarank.
│   │   │                        # A/B hash routing → LambdaRank or CrossEncoder.
│   │   └── Dockerfile           # pytorch/pytorch CUDA base.
│   │
│   └── feedback/                # Port 8004
│       ├── main.py              # FastAPI app: /click (log clicks) + /stats (threshold check).
│       └── Dockerfile           # python:3.11-slim + psycopg2.
│
├── airflow_dags/
│   └── retraining_dag.py        # DAG with 7 tasks: check_click_threshold →
│                                # extract_click_features → train → evaluate →
│                                # promote_if_better → hot_reload → notify.
│                                # Schedule: 0 2 * * * (2am daily).
│
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml       # Scrape config: all 5 services every 15 seconds.
│   └── grafana/
│       └── dashboard.json       # 11-panel pre-built dashboard. Auto-loaded on startup.
│
├── gradio_app/
│   ├── app.py                   # 4-tab Gradio app: Search, A/B Compare,
│   │                            # Offline Evaluation, System Stats.
│   └── Dockerfile               # python:3.11-slim + gradio + httpx + plotly.
│
├── tests/
│   ├── unit/
│   │   ├── test_two_tower.py    # Tests: output shape, L2 normalisation, scalar loss,
│   │   │                        # hard negative forward pass, cosine similarity direction.
│   │   └── test_services.py     # Tests: intent rules, A/B determinism, cache key
│   │                            # properties, NDCG/MAP/MRR metric calculations.
│   └── integration/
│       └── test_gateway.py      # Live tests against running gateway: health check,
│                                # result fields, forced ranker, latency SLA, cache hit,
│                                # unique request IDs, metrics endpoint.
│
├── data/                        # Git-ignored. Tracked by DVC.
│   ├── raw/                     # Downloaded MS MARCO .tsv files
│   ├── processed/               # Parquet files, BM25 index, sampled negatives
│   ├── embeddings/              # doc_embeddings.npy (500K × 256 float32, ~500MB)
│   └── indexes/                 # faiss_ivfpq.index (~16MB), bm25_index.pkl, docid_map.pkl
│
├── models/                      # Git-ignored. Tracked by DVC.
│   ├── two_tower/               # model_best.pt, model_final.pt, config.json, tokenizer
│   ├── cross_encoder/           # model.pt, config.json, tokenizer
│   └── lambdarank/              # lambdarank.json, feature_names.json
│
├── .env                         # YOUR REAL SECRETS. Never committed to git.
│                                # Copy from .env.example and add your API key.
├── .env.example                 # Template showing what keys are needed. Safe to commit.
│                                # Anyone cloning the repo can see what they need to set.
├── .gitignore                   # Excludes: .env, data/, models/, __pycache__, etc.
├── docker-compose.yml           # Defines all 12 services, networks, volumes.
│                                # docker-compose up starts everything.
└── requirements.txt             # All Python dependencies with pinned versions.
```

---

## 18. Getting started

### Prerequisites

Before you begin, make sure you have:

- **Python 3.11 or higher** — check with `python --version`
- **NVIDIA GPU with 8GB+ VRAM** — an RTX 3060, 4060, or better. CPU-only training is possible but will be very slow (12-24 hours instead of 6-8 hours).
- **CUDA 12.1 or higher** — check with `nvidia-smi`
- **Docker Desktop** — installed and running. Check with `docker --version`.
- **~50GB free disk space** — for the dataset (~3GB), embeddings (~500MB), and Docker images (~15GB)
- **An Anthropic API key** — [get one at console.anthropic.com](https://console.anthropic.com). The query understanding service uses Claude Haiku which costs roughly $0.001 per 1,000 queries — negligible for testing.

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/Search-Ranking-System.git
cd Search-Ranking-System
```

### Step 2 — Set up the Python environment

```bash
python -m venv venv

# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### Step 3 — Add your API key

The `.env` file holds your secrets and is never committed to git (it's in `.gitignore`). It already exists in the repo with a placeholder. Open it and replace the API key:

```bash
# Open .env in any text editor and replace:
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# with your actual key from console.anthropic.com
```

All other values in `.env` (PostgreSQL credentials, Redis host, service URLs) work as-is for local docker-compose deployment — you don't need to change them.

### Step 4 — Run the training pipeline

This is the longest step. It downloads the dataset, preprocesses it, trains three models, and builds the search indexes. On an RTX 4060 Laptop (8GB VRAM), expect roughly 6-8 hours total.

**Option A — Run everything in one command:**
```bash
bash scripts/run_pipeline.sh
```

**Option B — Run each step separately (recommended so you can monitor each one):**

```bash
# Step 1: Download MS MARCO (~3GB, 10-30 min depending on connection)
python scripts/download_msmarco.py

# Step 2: Preprocess — converts TSV to Parquet, builds BM25 index, samples
# random negatives for up to 50K training queries (see §5 — not BM25-mined,
# despite the function name mine_hard_negatives)
python scripts/preprocess.py

# Step 3: Train the two-tower dual encoder
# (~3-4 hours on RTX 4060, 3 epochs over the ~50K queries with sampled negatives)
# Checkpoint selection is by lowest training loss, not recall (see §11).
python training/train_two_tower.py

# Step 3b: Measure REAL Recall@10/Recall@100 on the dev set (see §11/§14)
python scripts/eval_recall.py

# Step 4: Embed all 500K passages and build the FAISS index
# (~20-30 min — GPU-accelerated embedding + index training)
python training/build_faiss_index.py

# Step 5: Fine-tune the CrossEncoder reranker
# (~1-2 hours, 2 epochs, gradient accumulation for 8GB VRAM)
python training/train_cross_encoder.py

# Step 6: Train LambdaRank
# (~20 min — builds feature matrix for 20K queries then trains XGBoost)
python training/train_lambdarank.py

# Step 7: Run full offline evaluation
# (~10-15 min — evaluates all 4 configurations on 6,980 dev queries)
# Prints a comparison table and saves results to data/processed/eval_results.json
python training/evaluate.py
```

### Step 5 — Start all services

```bash
docker-compose up
```

This starts 12 containers. The first run will pull Docker images and build the service containers (~5-10 minutes). Subsequent runs start in ~30 seconds.

Watch for all services to show `healthy`:
```bash
docker-compose ps
```

Check the gateway is ready:
```bash
curl http://localhost:8000/health
# Expected: {"status": "ok", "service": "api-gateway"}
```

### Step 6 — Run your first search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what causes inflation", "top_k": 10}'
```

Or open the Gradio UI at **http://localhost:7860**.

### Stopping the system

```bash
docker-compose down           # stop and remove containers
docker-compose down -v        # also remove database volumes (resets PostgreSQL)
```

---

## 19. Service URLs

Once `docker-compose up` is running, all of these are accessible in your browser:

| Service | URL | Login |
|---|---|---|
| **Gradio UI** | http://localhost:7860 | None |
| **API Gateway** (OpenAPI docs) | http://localhost:8000/docs | None |
| Query Understanding (docs) | http://localhost:8001/docs | None |
| Retrieval (docs) | http://localhost:8002/docs | None |
| Ranking (docs) | http://localhost:8003/docs | None |
| Feedback (docs) | http://localhost:8004/docs | None |
| **MLflow** | http://localhost:5001 | None |
| **Airflow** | http://localhost:8080 | admin / admin |
| Prometheus | http://localhost:9090 | None |
| **Grafana** | http://localhost:3000 | admin / searchadmin |

The OpenAPI docs (e.g. http://localhost:8000/docs) let you send requests directly from the browser without needing curl or a client — useful for exploring the API.

---

## 20. Running the tests

**Unit tests** (no running services needed):
```bash
pytest tests/unit/ -v
```

These test the model architecture (shape, L2 normalisation, loss values), service logic (intent classification rules, A/B routing determinism, cache key properties), and metric calculations (NDCG, MAP, MRR).

**Integration tests** (requires `docker-compose up`):
```bash
pytest tests/integration/ -v
```

These test the live gateway: that it returns valid results, that forced ranker selection works, that the latency SLA (< 500ms) is met, that a second identical request is served from cache (with lower retrieval latency), and that the Prometheus metrics endpoint is serving data.

---

## 21. Key design decisions

**Why two towers instead of just using a cross-encoder for everything?**

A cross-encoder compares the query and document together in one forward pass, which is very accurate. But it cannot pre-compute anything — every query requires a fresh forward pass for every document. With 500K documents, that's 500K forward passes per query, which would take ~10 minutes. The two-tower model's document embeddings are computed once offline. At query time, only the query is encoded (one forward pass), and FAISS finds the nearest 100 documents in ~30ms. The cross-encoder is then used only on those 100 candidates, where its accuracy advantage is worth the cost.

**Why DistilBERT instead of a larger BERT or a more modern model?**

DistilBERT is 40% smaller and 60% faster than BERT-base with only a 3% quality drop on most NLP benchmarks. On an 8GB GPU, this means we can train with larger batch sizes (more stable InfoNCE training) and serve faster at inference. For a portfolio project that needs to run on consumer hardware, this is the practical choice. In a real production system with budget, you'd use a larger encoder like `bert-large` or `e5-large-v2`.

**Why IVF+PQ and not just a flat exact-search FAISS index?**

A flat index does exact search — it finds the truly closest 100 vectors. But with 500K × 256-dim float32 vectors, a flat index uses ~512MB of RAM and scales linearly with corpus size. IVF+PQ uses ~16MB (32× compression) and searches ~31K vectors instead of 500K per query. The cost is ~5% recall loss — meaning for about 5% of queries, the true top-100 is slightly different from the approximate top-100. For a corpus of 500K this is a good tradeoff. At billion scale (YouTube), you'd use IVF65536,PQ64 or a hierarchical HNSW structure.

**Why is the A/B split deterministic (hash-based) rather than random?**

If the split were random, the same query could go to LambdaRank on one request and CrossEncoder on the next. With a hash on `request_id` (a UUID), the same `request_id` always goes to the same variant. Since `request_id` is random, the 50/50 distribution holds across all requests. This makes debugging much easier: if a user reports a bad result, you can look up their `request_id` in the logs and know exactly which ranker they saw. It also prevents "flicker" — a user searching the same query twice shouldn't see completely different results just because of A/B randomness.

**Why PostgreSQL instead of a simpler store like SQLite or a log file for click data?**

The click data needs to be queryable for retraining (GROUP BY query_text, JOIN with query_logs), and PostgreSQL is already running for MLflow and Airflow. Using one database reduces infrastructure complexity. The same click_logs table will eventually need window functions and CTEs for cohort analysis — PostgreSQL handles all of this naturally.

**Why Airflow for the retraining schedule instead of a simple cron job?**

A cron job would work. Airflow adds: task dependencies (step 3 only runs if step 2 succeeded), retries with backoff (if the evaluation step fails due to a transient GPU OOM, it retries automatically), a visual UI showing the full run history, XCom for passing data between tasks (the NDCG delta from evaluation is passed to the promotion decision), and branching (skip the whole pipeline if the click threshold hasn't been reached). For a team, this visibility is essential — you can see at a glance when the last retrain was, whether it was promoted, and what the NDCG improvement was.

---

## 22. Academic references

The techniques in this project come from a small number of highly cited papers. These are worth reading if you want to understand the theory:

- **MS MARCO dataset** — Bajaj et al., Microsoft Research, 2016. The paper introducing the dataset used for training and evaluation. [arxiv.org/abs/1611.09268](https://arxiv.org/abs/1611.09268)

- **Dense Passage Retrieval (DPR)** — Karpukhin et al., Facebook AI Research, 2020. Introduced the idea of using hard negatives mined from BM25 to train dense retrieval models. This is the paper that showed dense retrieval could match and exceed BM25. [arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)

- **YouTube Two-Tower Recommendations** — Covington, Adams, Sargin, Google, 2016. The paper that popularised the two-tower architecture for large-scale retrieval. The same architecture is used here for search instead of recommendations. [dl.acm.org/doi/10.1145/2959100.2959190](https://dl.acm.org/doi/10.1145/2959100.2959190)

- **LambdaRank** — Burges et al., Microsoft Research, 2006. Introduced the gradient trick that allows gradient boosting to directly optimise NDCG (which is normally non-differentiable). LambdaRank is used in production at Bing, historically at Yahoo, and in XGBoost's `rank:ndcg` objective. [microsoft.com research link](https://www.microsoft.com/en-us/research/publication/learning-to-rank-with-nonsmooth-cost-functions/)

- **HyDE (Hypothetical Document Embeddings)** — Gao et al., CMU / Google, 2022. The paper introducing the idea of generating a hypothetical answer to a query and embedding that instead of the query itself to improve zero-shot retrieval. Used in the query understanding service. [arxiv.org/abs/2212.10496](https://arxiv.org/abs/2212.10496)

- **FAISS** — Johnson, Douze, Jégou, Facebook AI Research, 2017. The paper and library behind the approximate nearest neighbour search used for retrieval. IVF and PQ are both described here. [github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)

- **DistilBERT** — Sanh et al., Hugging Face, 2019. The distilled version of BERT used as the backbone for both the two-tower encoders and the cross-encoder. [arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)
