#!/usr/bin/env bash
# Cloud Run startup: pull artifacts from HF Hub (idempotent), then launch the API.
set -e

if [ -f "models/two_tower/model_best.pt" ] && [ -f "data/indexes/faiss_ivfpq.index" ]; then
  echo "Artifacts already present; skipping bootstrap."
else
  echo "Bootstrapping artifacts from HF Hub (HF_ARTIFACTS_REPO=${HF_ARTIFACTS_REPO:-shiva-1993/search-ranking-system})..."
  # --optional pulls the CrossEncoder so the "crossencoder" ranker actually works
  # in the demo (without it the engine silently falls back to LambdaRank).
  python scripts/bootstrap.py --optional || {
    echo "ERROR: bootstrap failed. Ensure HF_ARTIFACTS_REPO points at a repo with"
    echo "published artifacts (scripts/publish_artifacts.py)."
    exit 1
  }
fi

# Single worker: the model + indexes are large; one process per container and let
# Cloud Run scale horizontally. Cloud Run injects $PORT (default 8080).
exec uvicorn deploy.api:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
