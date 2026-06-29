# Hugging Face Space image (Docker SDK). CPU-only, single process.
# Pulls model/index artifacts from HF Hub at startup (deploy/start.sh).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch (no CUDA libs).
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0

COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces run as uid 1000; give it a home + a writable HF cache.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/models /home/appuser/.cache \
    && chown -R appuser:appuser /app /home/appuser
USER appuser

ENV PYTHONPATH=/app \
    HF_HOME=/home/appuser/.cache/huggingface

EXPOSE 7860

CMD ["bash", "deploy/start.sh"]
