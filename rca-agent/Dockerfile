# Multi-stage build for Python RCA Agent using uv
FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv globally
RUN pip install --no-cache-dir uv

# Stage 1: Dependencies
FROM base AS builder

# Copy dependency files first (create README.md since it's referenced in pyproject.toml)
COPY pyproject.toml uv.lock ./
RUN echo "# RCA Agent\n\nThis agent performs root cause analysis (RCA) for SRE incidents." > README.md

# Create virtual environment and install dependencies
RUN uv venv && \
    .venv/bin/python -m ensurepip --upgrade && \
    uv sync

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser pyproject.toml ./

# Set environment variables for virtual environment
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER appuser

# Expose the port (RCA Agent runs on 8011)
EXPOSE 8011

# Default command to run the RCA agent
CMD ["python", "-m", "app", "--host", "0.0.0.0", "--port", "8011"]