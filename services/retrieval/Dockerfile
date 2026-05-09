FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.7.1 \
    prometheus-client==0.20.0 \
    structlog==24.1.0 \
    faiss-gpu==1.7.2 \
    transformers==4.41.0 \
    redis==5.0.4 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    python-dotenv==1.0.1

COPY services/ services/
COPY training/two_tower_model.py training/two_tower_model.py

ENV PYTHONPATH=/app
EXPOSE 8002

CMD ["uvicorn", "services.retrieval.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
