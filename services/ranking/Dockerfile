# Ranking service — CPU-only image (runs on free tiers).
# Previously used a CUDA base; CPU-only torch keeps it deployable for free.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (smaller wheel, no CUDA libraries pulled in).
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0

COPY services/ranking/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ services/
COPY training/train_cross_encoder.py training/train_cross_encoder.py

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app
EXPOSE 8003

CMD ["uvicorn", "services.ranking.main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
