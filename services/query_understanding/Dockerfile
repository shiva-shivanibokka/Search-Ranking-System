FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.7.1 \
    prometheus-client==0.20.0 \
    structlog==24.1.0 \
    anthropic==0.28.0 \
    python-dotenv==1.0.1

COPY services/ services/

ENV PYTHONPATH=/app
EXPOSE 8001

CMD ["uvicorn", "services.query_understanding.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
