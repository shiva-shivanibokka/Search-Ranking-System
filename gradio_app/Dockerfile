FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    gradio==4.31.0 \
    httpx==0.27.0 \
    pandas==2.2.2 \
    plotly==5.22.0 \
    python-dotenv==1.0.1

COPY gradio_app/app.py gradio_app/app.py

ENV PYTHONPATH=/app
EXPOSE 7860

CMD ["python", "gradio_app/app.py"]
