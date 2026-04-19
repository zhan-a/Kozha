# syntax=docker/dockerfile:1.6
#
# Multi-stage build for the Kozha + chat2hamnosys FastAPI app.
#
# Stage 1 ("builder") creates a virtualenv with all Python deps so the
# runtime image stays small — no pip cache, no compiler toolchain, no
# wheel build artefacts.
#
# Stage 2 ("runtime") copies the venv plus the application source, drops
# privileges to a non-root user, and starts uvicorn on :8000.
#
# Build:    docker build --build-arg BUILD_SHA=$(git rev-parse --short HEAD) -t kozha:local .
# Run:      docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-... kozha:local
# Health:   curl localhost:8000/api/chat2hamnosys/health
#
# Image size budget: <500 MB. Verified in docs/chat2hamnosys/19-deployment.md.

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Build deps for lxml (libxml2 / libxslt headers) and any C extensions.
# Installed only in the builder; the runtime image picks up the compiled
# wheels via the venv copy.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt1-dev \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV" \
 && pip install --upgrade pip wheel setuptools

WORKDIR /build

# Copy only requirement manifests first so the long pip-install layer is
# cached when source files change but deps don't.
COPY backend/chat2hamnosys/requirements.txt /build/backend-requirements.txt
COPY server/requirements.txt /build/server-requirements.txt

# chat2hamnosys deps — the authoring stack proper.
RUN pip install --no-cache-dir -r /build/backend-requirements.txt

# Server-side runtime deps. We install fastapi/uvicorn/gunicorn/spacy
# plus the English small model; the remaining six per-language spaCy
# models and the argostranslate stack are *not* baked in by default —
# they push the image well past the 500 MB budget. Operators who need
# them install at deploy-time (see docs/chat2hamnosys/19-deployment.md
# §"Translation extras"). The chat2hamnosys authoring path is unaffected
# by their absence.
RUN pip install --no-cache-dir \
        "fastapi" \
        "uvicorn[standard]" \
        "gunicorn" \
        "spacy>=3.8.0" \
 && pip install --no-cache-dir \
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz"

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG BUILD_SHA=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    BUILD_SHA=${BUILD_SHA} \
    KOZHA_HOME=/app \
    CHAT2HAMNOSYS_DATA_DIR=/app/data \
    CHAT2HAMNOSYS_LOG_DIR=/app/logs

# libxml2 + libxslt runtime libs for lxml; curl for the HEALTHCHECK.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libxml2 \
        libxslt1.1 \
        curl \
        ca-certificates \
        tini \
 && rm -rf /var/lib/apt/lists/*

# Non-root user — fixed uid so volume mounts behave deterministically
# across hosts (e.g. fly.io volumes, host bind mounts in compose).
RUN groupadd --gid 10001 kozha \
 && useradd --uid 10001 --gid kozha --create-home --shell /bin/bash kozha

# Copy the virtualenv from the builder. Single layer, no recompile.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy application source. .dockerignore keeps the build context lean
# (no .git, no logs, no cached previews, no editor noise). Ownership
# flips to kozha on copy so the runtime user can read everything.
COPY --chown=kozha:kozha backend /app/backend
COPY --chown=kozha:kozha server /app/server
COPY --chown=kozha:kozha public /app/public
COPY --chown=kozha:kozha data /app/data

# Mutable runtime dirs created up-front so volume mounts don't trip on
# missing parents. Modes are deliberately group-writable so a host
# bind-mount run by a different uid can still write (compose pattern).
RUN install -d -o kozha -g kozha -m 0775 \
        /app/data/authored_signs \
        /app/data/preview_cache \
        /app/data/chat2hamnosys \
        /app/logs

USER kozha

EXPOSE 8000

# Use chat2hamnosys' own /health (which surfaces BUILD_SHA) — it is the
# subsystem this image actually exists to serve. Falling back to the
# parent /api/health would also work but says less.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8000/api/chat2hamnosys/health || exit 1

# tini reaps zombies and forwards signals — uvicorn's default workers
# don't always shut down cleanly on SIGTERM otherwise.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Single-process uvicorn by default. Operators who want gunicorn with
# multiple workers override this in compose / fly.toml / systemd unit.
CMD ["uvicorn", "server.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
