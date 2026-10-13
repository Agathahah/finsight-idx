# FinSight IDX demo, same app as the Hugging Face Space.
#   docker build -t finsight-idx .
#   docker run --rm -p 7860:7860 --env-file .env finsight-idx
# The image serves a pre-built index: run the indexer first so that
# data/vectorstore/ contains chroma.sqlite3 and index_manifest.json.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    FINSIGHT_STRICT_INDEX=1 \
    VECTORSTORE_DIR=/app/data/vectorstore \
    QUERY_LOG_PATH=/app/logs/queries.jsonl

WORKDIR /app
RUN useradd --create-home --uid 1000 app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
COPY requirements_spaces.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
 && pip install -r requirements_spaces.txt gradio==6.12.0

# Bake the embedding model into the image: no download at container start.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

COPY app.py ./
COPY src ./src
COPY data/vectorstore ./data/vectorstore
RUN mkdir -p logs && chown -R app:app /app
USER app

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)" || exit 1
CMD ["python", "app.py"]
