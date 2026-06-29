# Retrieval service — CPU-only image (runs on free tiers: Render, Fly.io, HF Spaces).
# Previously used a CUDA base + faiss-gpu==1.7.2 which (a) won't run on free CPU
# hosts and (b) mismatched the faiss-cpu version that built the index.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (smaller wheel, no CUDA libraries pulled in).
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0

COPY services/retrieval/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ services/
COPY training/two_tower_model.py training/two_tower_model.py

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app
EXPOSE 8002

CMD ["uvicorn", "services.retrieval.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
