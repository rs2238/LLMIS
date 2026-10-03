FROM python:3.12-slim

WORKDIR /app

# system deps: none needed — torch ships CPU-only prebuilt wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code + the trained artifacts it needs at inference time
COPY config.py model.py serve.py app.py batching.py quantize.py metrics.py drift.py ./
COPY model.pt vocab.json ./

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
