FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# ── Layer 1: install deps via uv (pyproject + lock first for cache efficiency) ──
# --extra docling adds Docling (CPU-only torch). --no-cache keeps uv's download cache
# out of the image, so the wheels are not stored twice.
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --no-dev --frozen --no-install-project --extra docling --no-cache

# Bake the layout + table models into the image so the first upload never downloads them
ENV DOCLING_ARTIFACTS_PATH=/opt/docling-models
RUN uv run --no-sync docling-tools models download layout tableformer -o /opt/docling-models

# ── Layer 2: backend package ───────────────────────────────────────────────────
COPY backend/ backend/

# ── Layer 3: application files (change most often — last for cache efficiency) ─
COPY api/ api/
COPY .env.example .

# Non-root user for security (production best practice)
# Not chown -R: recursive chown copies the whole .venv into a new layer (~2 GB with Docling)
RUN useradd -m -u 10001 papeer && chown papeer:papeer /app
USER papeer

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=8)" || exit 1

# Required env vars (pass via docker run -e or --env-file .env):
#   OPENAI_API_KEY, GOOGLE_API_KEY, TAVILY_API_KEY, QDRANT_URL, QDRANT_API_KEY
#   DATABASE_URL (Neon pooled — optional, falls back to LocalFileStore)

CMD ["uv", "run", "--no-sync", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
