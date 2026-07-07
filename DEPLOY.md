# Deploying the live demo on free infrastructure

This deploys a public, clickable demo of the search system at **$0** using only
free tiers — no Supabase, no paid hosting. The full 5-microservice stack stays
runnable locally (`docker-compose up`); this guide deploys the **consolidated
single-process demo** (`deploy/`) which fits a free host.

## Architecture (free-tier)

```
  Hugging Face Space (free, Docker SDK)  ──►  deploy/space_app.py (Gradio UI)
        │  pulls models + indexes at start from ─►  Hugging Face Hub (free)
        ├──► Neon (free serverless Postgres)  — click logs
        └──► (optional) LLM provider          — Groq/Gemini free tier for HyDE

  GitHub Actions (free):  CI on every push  +  weekly click-feedback retraining
```

Why consolidated: a free Space gives one process, so query understanding +
hybrid retrieval + ranking run in-process (`deploy/engine.py`) instead of as 5
networked services. Results match the microservice path.

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

## Step 4 — Create the Hugging Face Space

1. Create a new Space → **Docker** SDK → blank.
2. Push this repo to the Space (or connect the GitHub repo). The Space builds
   from `deploy/Dockerfile`. Copy `deploy/README_SPACE.md` to the Space's
   `README.md` (it carries the required `sdk: docker` / `app_port: 7860` frontmatter).
3. In **Settings → Variables and secrets**, set:
   - `HF_ARTIFACTS_REPO` (variable) = `shiva-1993/search-ranking-system` (the default; override for your own fork)
   - `HF_TOKEN` (secret) — only if the artifact repo is private
   - `DATABASE_URL` (secret) — your Neon URL, to enable click logging
   - `LLM_PROVIDER` (variable) = `groq` (optional) + `GROQ_API_KEY` (secret) for HyDE
4. The Space boots, runs `deploy/start.sh` (pulls artifacts, launches Gradio),
   and your demo is live at `https://huggingface.co/spaces/<you>/<space>`.

With no LLM key and no DB, it still runs (rule-based understanding, no logging).

## Step 5 — Wire up CI + scheduled retraining (GitHub)

- **CI** (`.github/workflows/ci.yml`) runs automatically on push/PR: lint + tests + Docker build.
- **Retraining** (`.github/workflows/retrain.yml`) runs weekly. Add repo secrets
  (Settings → Secrets and variables → Actions): `DATABASE_URL`, `HF_ARTIFACTS_REPO`,
  `HF_TOKEN`. It retrains LambdaRank from accumulated clicks and republishes the
  model to HF Hub; the Space picks it up on its next restart.

---

## Rollback

The artifact repo on HF Hub is versioned (git-backed). To roll back a model,
restore the previous `models/lambdarank/lambdarank.json` revision in the artifact
repo and restart the Space. For schema, `alembic downgrade -1`.

## What stays local (documented, not deployed)

Airflow, Prometheus, Grafana, and MLflow run via `docker-compose up` for the full
experience — these are internal tools companies don't expose publicly either.
Capture a short screen recording / screenshots of them for the README.
