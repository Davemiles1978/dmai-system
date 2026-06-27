# ─────────────────────────────────────────────────────────────────────────────
# DMAI Web — Production Dockerfile
# Built by GitHub Actions CI; deployed to Render via git push.
# ─────────────────────────────────────────────────────────────────────────────
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

# System deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────
COPY . .

# ── Runtime config ────────────────────────────────────────────────────────
ENV PORT=8080
ENV DATA_PATH=/app/data
ENV PYTHONUNBUFFERED=1

EXPOSE $PORT

# Gunicorn: 1 worker + 2 threads matches Render free-tier memory limits.
# max-requests restarts workers periodically to prevent memory leaks.
CMD gunicorn dmai_core_complete:app \
    --bind 0.0.0.0:$PORT \
    --timeout 120 \
    --workers 1 \
    --threads 2 \
    --max-requests 100 \
    --max-requests-jitter 50 \
    --access-logfile - \
    --error-logfile -
