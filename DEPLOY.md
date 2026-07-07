# Deploying the live demo on free infrastructure

This deploys a public, clickable demo of the search system at **~$0** using only
free tiers — no Supabase, no paid hosting. The full 5-microservice stack stays
runnable locally (`docker-compose up`); this guide deploys a **SvelteKit
frontend** (Vercel) talking to a **consolidated retrieval API** (`deploy/api.py`)
on **Google Cloud Run**, which scales to zero.

## Architecture (free-tier)

```
  Vercel (free)  ──►  SvelteKit frontend (web/)
        │  fetch() ─►  Cloud Run (scale-to-zero)  ──►  deploy/api.py (FastAPI)
        │                    │  pulls model + indexes at start ─►  HF Hub (free)
        │                    ├──► Neon (free serverless Postgres) — click logs
        │                    └──► (optional) LLM provider — Groq/Gemini for HyDE
        └─ client-side BYOK RAG:  browser calls the user's own LLM key directly
                                  (Anthropic / Gemini / OpenAI / Groq), never the server

  GitHub Actions (free):  CI on every push  +  weekly click-feedback retraining
```

Why consolidated: the API runs query understanding + hybrid retrieval + ranking
in one process (`deploy/engine.py`) instead of as 5 networked services, so it
fits one small Cloud Run container. Results match the microservice path.

---

## Step 1 — Publish artifacts to Hugging Face Hub

The model weights + FAISS/BM25 indexes + passages are too big for git. Publish
them once so every deploy (and any fresh clone) can pull them.

```bash
pip install huggingface_hub
huggingface-cli login                      # paste an HF write token
export HF_ARTIFACTS_REPO=shiva-1993/search-ranking-system   # the published artifact repo (override with your own fork)
python scripts/publish_artifacts.py        # add --optional to include cross-encoder
```

Verify a fresh pull works:

```bash
python scripts/bootstrap.py                # downloads into models/ and data/
```

## Step 2 — Create a free Neon Postgres database (click logs)

1. Sign up at neon.tech (free tier) and create a project.
2. Copy the connection string (looks like `postgresql://user:pass@ep-xxx.neon.tech/db?sslmode=require`).
3. Apply the schema:

```bash
export DATABASE_URL="postgresql://user:pass@ep-xxx.neon.tech/db?sslmode=require"
pip install -r requirements-dev.txt
alembic upgrade head                       # creates query_logs, click_logs, model_versions
```

(The app also auto-creates tables on first write, but `alembic upgrade head` is
the version-controlled, repeatable path.)

## Step 3 — (Optional) Upstash Redis for caching

1. Sign up at upstash.com (free tier), create a Redis database.
2. Note the host/port/password (or the `rediss://` URL).
3. The consolidated demo runs fine without Redis; for the full microservice
   stack, point `REDIS_HOST`/`REDIS_PORT` at Upstash.

## Step 4 — Deploy the retrieval API to Cloud Run

Full runbook (deploy command, memory/CPU sizing, keep-warm scheduler, verify):
**[`deploy/cloudrun.md`](deploy/cloudrun.md)**. In short, from the repo root:

```bash
gcloud run deploy search-ranking-api --source . --dockerfile deploy/Dockerfile \
  --region us-central1 --allow-unauthenticated --memory 4Gi --cpu 2 \
  --set-env-vars "HF_ARTIFACTS_REPO=shiva-1993/search-ranking-system,ALLOWED_ORIGINS=https://<your-app>.vercel.app"
```

`start.sh` pulls artifacts from HF Hub on boot, then serves `deploy/api.py`.
With no LLM key and no DB it still runs (rule-based understanding, no HyDE, no
logging). Grab the printed service URL — it's the frontend's `PUBLIC_API_URL`.

## Step 5 — Deploy the SvelteKit frontend to Vercel

1. Import the GitHub repo in Vercel → set **Root Directory** to `web/`.
2. Set the env var `PUBLIC_API_URL` = the Cloud Run service URL from Step 4.
3. Deploy. Vercel builds the SvelteKit app; the demo is live at
   `https://<your-app>.vercel.app`.
4. Back in Cloud Run, set `ALLOWED_ORIGINS` to that Vercel URL so CORS only
   admits the frontend. See `web/README.md` for local dev.

The RAG panel is **client-side BYOK**: visitors paste their own LLM key
(Anthropic / Gemini / OpenAI / Groq), which stays in their browser and is sent
straight to the provider — never to your server.

## Step 6 — Wire up CI + scheduled retraining (GitHub)

- **CI** (`.github/workflows/ci.yml`) runs automatically on push/PR: lint + tests + Docker build.
- **Retraining** (`.github/workflows/retrain.yml`) runs weekly. Add repo secrets
  (Settings → Secrets and variables → Actions): `DATABASE_URL`, `HF_ARTIFACTS_REPO`,
  `HF_TOKEN`. It retrains LambdaRank from accumulated clicks and republishes the
  model to HF Hub; the Cloud Run service picks it up on its next cold start (or
  redeploy).

---

## Rollback

The artifact repo on HF Hub is versioned (git-backed). To roll back a model,
restore the previous `models/lambdarank/lambdarank.json` revision in the artifact
repo and redeploy the Cloud Run service. For schema, `alembic downgrade -1`.

## What stays local (documented, not deployed)

Airflow, Prometheus, Grafana, and MLflow run via `docker-compose up` for the full
experience — these are internal tools companies don't expose publicly either.
Capture a short screen recording / screenshots of them for the README.
