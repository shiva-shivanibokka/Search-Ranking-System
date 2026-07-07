# Deploying the retrieval API to Google Cloud Run

The retrieval API (`deploy/api.py`) is a single-process FastAPI service that
wraps the full hybrid engine (FAISS + BM25 + RRF + LambdaRank). Cloud Run is the
target host: it **scales to zero** (idle cost ~$0) and the free tier (2M req,
360k GB-s, 180k vCPU-s / month) comfortably covers a portfolio demo.

The artifacts (~1.2 GB) are pulled from HF Hub on container start by
`scripts/bootstrap.py` — nothing large lives in the image or in git.

## Prerequisites (one-time)

```bash
gcloud auth login
gcloud config set project <YOUR_GCP_PROJECT_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com
```

## Deploy

From the repo root (Cloud Build builds `deploy/Dockerfile`):

```bash
gcloud run deploy search-ranking-api \
  --source . \
  --dockerfile deploy/Dockerfile \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --concurrency 8 \
  --timeout 120 \
  --min-instances 0 \
  --max-instances 2 \
  --set-env-vars "HF_ARTIFACTS_REPO=shiva-1993/search-ranking-system,ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app,RATE_LIMIT_PER_MINUTE=60"
```

Notes:
- **Memory 4Gi**: the model + BM25 index + passages need ~2–4 GB resident. 2Gi
  OOMs; 4Gi is the safe floor.
- **Cold start ~60–120 s**: image pull + 1.2 GB artifact download + model load.
  Cloud Run's default startup grace is enough at 4Gi/2cpu; if a deploy reports
  "failed to start and listen", raise CPU to 4 for the first boot or pre-bake
  artifacts (see below).
- **ALLOWED_ORIGINS**: set to your Vercel domain so CORS only allows the
  frontend. Use `*` only for local testing.
- Leave `HF_TOKEN` unset — the artifact repo is public.

Grab the service URL (used as `PUBLIC_API_URL` in the SvelteKit frontend):

```bash
gcloud run services describe search-ranking-api --region us-central1 \
  --format 'value(status.url)'
```

## Keep-warm (kill cold starts on the demo)

Cloud Run scales to zero, so the first hit after idle is slow. A tiny scheduled
ping to `/health` keeps one instance warm during the hours you expect traffic
(e.g. while sharing the demo in an interview). This stays within the free tier.

```bash
gcloud scheduler jobs create http keep-warm-search-api \
  --location us-central1 \
  --schedule "*/10 * * * *" \
  --uri "$(gcloud run services describe search-ranking-api --region us-central1 --format 'value(status.url)')/health" \
  --http-method GET
```

Delete it (`gcloud scheduler jobs delete keep-warm-search-api --location us-central1`)
when you don't need the demo hot, to save the warm-instance minutes.

For a *guaranteed* warm instance (no cold start ever, but bills for one
always-on instance's memory — not free), deploy with `--min-instances 1`.

## Optional integrations

| Env var | Effect |
|---|---|
| `LLM_PROVIDER` = `groq`/`gemini`/`openai`/`anthropic` + `<PROVIDER>_API_KEY` | Enables server-side HyDE query expansion. The client-side RAG does **not** use this — it uses the visitor's own key in the browser. |
| `DATABASE_URL` (Neon) | Enables `POST /click` logging for the retraining signal. |
| `FAISS_NPROBE` (default 64) | Recall/latency tradeoff on the FAISS IVF index. |

## Verify

```bash
URL=$(gcloud run services describe search-ranking-api --region us-central1 --format 'value(status.url)')
curl "$URL/health"
curl -X POST "$URL/search" -H 'content-type: application/json' \
  -d '{"query":"what causes inflation","top_k":5}' | jq
```

`/health` returns `engine_ready: true` once artifacts are loaded; `/search`
returns ranked results plus the full pipeline stage breakdown.
