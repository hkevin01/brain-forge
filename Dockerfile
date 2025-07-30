# Brain-Forge API Dockerfile
# Multi-stage build for production-ready Brain-Forge API deployment

# Stage 1: Base Python environment
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash brain-forge
USER brain-forge
WORKDIR /home/brain-forge

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements
COPY --chown=brain-forge:brain-forge requirements/ ./requirements/
COPY --chown=brain-forge:brain-forge requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements/base.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY --chown=brain-forge:brain-forge src/ ./src/
COPY --chown=brain-forge:brain-forge configs/ ./configs/

# Create necessary directories
RUN mkdir -p logs data temp results

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.api.rest_api:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Development (optional)
FROM application as development

# Install development dependencies
RUN pip install --user --no-cache-dir -r requirements/dev.txt

# Install additional dev tools
RUN pip install --user --no-cache-dir \
    jupyter \
    ipython

# Development command with reload
CMD ["python", "-m", "uvicorn", "src.api.rest_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
