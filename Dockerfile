# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.14-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv for reproducible installs from uv.lock
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Use the image's Python (no downloads) and copy files instead of hardlinking
ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.14-slim AS runtime

WORKDIR /app

# Runtime libs (libgomp for XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtualenv from builder (same path, same base image → interpreter link stays valid)
COPY --from=builder /app/.venv /app/.venv

# Copy project files
COPY src/       ./src/
COPY configs/   ./configs/

# Copy trained model artifacts (if present)
COPY models/    ./models/

# Environment
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
