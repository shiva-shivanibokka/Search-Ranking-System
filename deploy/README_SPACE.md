---
title: Neural Search Ranking System
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Neural Search Ranking System — Live Demo

Two-stage neural search over 500K MS MARCO passages: hybrid retrieval
(FAISS dense + BM25 sparse, fused with Reciprocal Rank Fusion) followed by
learned reranking (LambdaRank, optional CrossEncoder). This Space runs the whole
pipeline consolidated in a single process; the full system is a 5-microservice
architecture (see the GitHub repo).

## Required Space configuration

This Space is built from `deploy/Dockerfile`. Set these in **Settings → Variables and secrets**:

| Name | Type | Required | Purpose |
|---|---|---|---|
| `HF_ARTIFACTS_REPO` | Variable | yes | `shiva-1993/search-ranking-system` — where models/indexes were published (this is the default) |
| `HF_TOKEN` | Secret | only if artifact repo is private | read access to pull artifacts |
| `LLM_PROVIDER` | Variable | no (default `none`) | `groq` / `gemini` / `openai` / `anthropic` for HyDE + intent |
| `GROQ_API_KEY` / `GOOGLE_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Secret | only for the chosen provider | LLM key |
| `DATABASE_URL` | Secret | no | Neon Postgres URL to log clicks |

With no LLM key and no DB, the demo still runs fully (rule-based query
understanding, no click logging) at $0.

See `DEPLOY.md` in the repo for the full step-by-step deployment guide.
