# Cloud Run image for the consolidated retrieval API (deploy/api.py).
# CPU-only, single process. Artifacts (model + FAISS/BM25 indexes + passages)
# are pulled from HF Hub at startup by scripts/bootstrap.py (see deploy/start.sh).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch (no CUDA libs) — keeps the image small and Cloud-Run-friendly.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0

COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/models /home/appuser/.cache \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

ENV PYTHONPATH=/app \
    HF_HOME=/home/appuser/.cache/huggingface \
    PORT=8080

# Cloud Run injects $PORT; start.sh bootstraps artifacts then execs uvicorn.
EXPOSE 8080
CMD ["bash", "deploy/start.sh"]
