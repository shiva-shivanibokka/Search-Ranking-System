FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/feedback/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ services/

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app
EXPOSE 8004

CMD ["uvicorn", "services.feedback.main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "2"]
