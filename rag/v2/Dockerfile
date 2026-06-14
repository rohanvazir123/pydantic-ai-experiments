# ── Stage 1: base — Python 3.13 + uv ─────────────────────────────────────────
FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1
RUN pip install --no-cache-dir uv
WORKDIR /app

# ── Stage 2: api-deps — install API extras ────────────────────────────────────
# reranker (sentence-transformers/torch) is excluded — it's optional and the
# pipeline degrades gracefully when not available (pure RRF ordering).
# Install separately in production via: pip install sentence-transformers
FROM base AS api-deps
COPY pyproject.toml uv.lock* ./
RUN uv sync --extra ingestion --extra observability --extra mcp --no-dev

# ── Stage 3: worker-deps — install worker extras ──────────────────────────────
FROM base AS worker-deps
COPY pyproject.toml uv.lock* ./
RUN uv sync --extra ingestion --extra audio --extra observability --no-dev

# ── Stage 4: api — production API image ──────────────────────────────────────
FROM api-deps AS api
COPY knowledge/ ./knowledge/
ENV GUNICORN_WORKERS=2 \
    GUNICORN_TIMEOUT=0 \
    GUNICORN_GRACEFUL_TIMEOUT=30 \
    GUNICORN_KEEPALIVE=5
EXPOSE 8000
CMD gunicorn knowledge.api.app:app \
      -k uvicorn.workers.UvicornWorker \
      --workers ${GUNICORN_WORKERS} \
      --bind 0.0.0.0:8000 \
      --timeout ${GUNICORN_TIMEOUT} \
      --graceful-timeout ${GUNICORN_GRACEFUL_TIMEOUT} \
      --keep-alive ${GUNICORN_KEEPALIVE} \
      --access-logfile - \
      --error-logfile -

# ── Stage 5: ingest-worker ────────────────────────────────────────────────────
FROM worker-deps AS ingest-worker
COPY knowledge/ ./knowledge/
CMD ["python", "-m", "knowledge.ingestion.worker"]

# ── Stage 6: retrieval-worker ─────────────────────────────────────────────────
FROM worker-deps AS retrieval-worker
COPY knowledge/ ./knowledge/
CMD ["python", "-m", "knowledge.retrieval.worker"]

# ── Stage 7: eval-worker ──────────────────────────────────────────────────────
FROM worker-deps AS eval-worker
COPY knowledge/ ./knowledge/
CMD ["python", "-m", "knowledge.evaluation.runner"]
