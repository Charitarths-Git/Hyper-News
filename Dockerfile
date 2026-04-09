FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.runtime.txt ./requirements.runtime.txt

RUN pip install --no-cache-dir -r requirements.runtime.txt

RUN mkdir -p /app/data /app/models /app/graph

COPY backend ./backend
COPY data/articles.parquet ./data/articles.parquet
COPY data/article_embeddings.npy ./data/article_embeddings.npy
COPY models/ltr_model.txt ./models/ltr_model.txt
COPY models/ltr_model.txt.weights.json ./models/ltr_model.txt.weights.json

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
