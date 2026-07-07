# 0004 — Dual-mode deployment: microservices locally, consolidated on free infra

**Status:** Accepted — partially superseded by [ADR 0007](0007-cloud-run-vercel-frontend.md)

> **Update:** the *dual-mode* decision (microservices locally, one consolidated
> process on free infra) still holds. What changed is the host and UI: the
> consolidated process is now a **FastAPI API** (`deploy/api.py`) on **Google
> Cloud Run**, fronted by a **SvelteKit** app on **Vercel** with client-side
> BYOK RAG — not a Gradio UI on a Hugging Face Space (that free tier was
> exhausted, and the portfolio needed a production-grade frontend). See ADR 0007.

## Context
The system is 5 FastAPI microservices plus Postgres, Redis, MLflow, Prometheus,
Grafana, and Airflow. That is the right shape to demonstrate distributed-systems
skills, but no free hosting tier can run ~12 always-on containers (PyTorch + a
500K-vector FAISS index + a cross-encoder need real RAM). The project must still
have a public, clickable demo for a portfolio.

## Decision
Run the **same code in two modes**:
- **Microservices mode** — `docker-compose up` locally / on a real box. The full
  architecture, used to develop and to record the walkthrough.
- **Consolidated mode** — `deploy/engine.py` runs the whole pipeline in one
  process behind a Gradio UI on a free **Hugging Face Space** (16GB CPU), backed
  by free serverless **Neon** (Postgres) and **Upstash** (Redis). Heavy internal
  tooling (Airflow/Prometheus/Grafana/MLflow) stays local and documented.

## Consequences
- **Pro:** a real public demo at $0, plus a credible full architecture.
- **Pro:** mirrors how companies operate — they don't expose retraining pipelines
  or Grafana to the public either.
- **Pro:** scheduled retraining moves to free GitHub Actions cron (no always-on
  Airflow needed for the hosted demo).
- **Trade-off:** consolidated mode duplicates some glue logic from the services
  (RRF, the LambdaRank feature vector). Kept faithful and flagged in code; the
  long-term fix is to extract those pure functions into a shared module both modes import.

## At 10× scale
Promote consolidated mode back to the microservice topology on a paid platform
(Render/Fly/AWS), put the vector index behind its own service, and keep Airflow
for the gated retrain/promote pipeline.
