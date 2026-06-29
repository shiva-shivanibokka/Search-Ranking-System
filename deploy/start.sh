#!/usr/bin/env bash
# Space startup: pull artifacts from HF Hub (idempotent), then launch the demo.
set -e

if [ -f "models/two_tower/model_best.pt" ] && [ -f "data/indexes/faiss_ivfpq.index" ]; then
  echo "Artifacts already present; skipping bootstrap."
else
  echo "Bootstrapping artifacts from HF Hub (HF_ARTIFACTS_REPO=${HF_ARTIFACTS_REPO})..."
  python scripts/bootstrap.py || {
    echo "ERROR: bootstrap failed. Set the HF_ARTIFACTS_REPO Space variable and"
    echo "publish artifacts with scripts/publish_artifacts.py first."
    exit 1
  }
fi

exec python deploy/space_app.py
