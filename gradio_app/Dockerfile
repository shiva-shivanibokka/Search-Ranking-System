FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY gradio_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gradio_app/app.py gradio_app/app.py

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app
EXPOSE 7860

CMD ["python", "gradio_app/app.py"]
