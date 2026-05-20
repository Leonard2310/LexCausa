# LexCausa Flask Backend Dockerfile
# Multi-stage build for optimized image size

FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies (build-essential needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir "poetry>=2.0.0,<3.0.0"

# Copy dependency files first to leverage layer caching
COPY pyproject.toml poetry.lock ./

# Install Python dependencies only (no dev group, no virtualenv inside container)
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --without dev

# ─── Final stage ────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Runtime dependencies only (no compiler needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application code (respects .dockerignore)
COPY . .

# Ensure log directories exist
RUN mkdir -p logs logs/pdf_exports/pipeline logs/pdf_exports/doe

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "src/api_server.py"]
