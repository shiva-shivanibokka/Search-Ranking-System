FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.7.1 \
    prometheus-client==0.20.0 \
    structlog==24.1.0 \
    psycopg2-binary==2.9.9 \
    sqlalchemy==2.0.30 \
    python-dotenv==1.0.1

COPY services/ services/

ENV PYTHONPATH=/app
EXPOSE 8004

CMD ["uvicorn", "services.feedback.main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "2"]
