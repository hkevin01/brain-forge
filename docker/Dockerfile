# Brain-Forge Docker Image
# Multi-stage build for production-ready brain scanning and simulation platform

ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=11.8

# Base stage with CUDA support for GPU acceleration
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libhdf5-dev \
    libnetcdf-dev \
    libopenmpi-dev \
    openmpi-bin \
    libfftw3-dev \
    libeigen3-dev \
    libboost-all-dev \
    libgsl-dev \
    libltdl-dev \
    libncurses-dev \
    libreadline-dev \
    libtool \
    automake \
    autoconf \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Development stage
FROM base as development

WORKDIR /app

# Copy requirements first for better caching
COPY requirements/ requirements/
RUN pip install -r requirements/base.txt
RUN pip install -r requirements/dev.txt
RUN pip install -r requirements/gpu.txt
RUN pip install -r requirements/visualization.txt
RUN pip install -r requirements/hardware.txt

# Install additional development tools
RUN pip install \
    jupyter \
    jupyterlab \
    ipython \
    notebook \
    jupyter-contrib-nbextensions

# Copy source code
COPY . .

# Install Brain-Forge in development mode
RUN pip install -e .

# Set up Jupyter extensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable --py widgetsnbextension

# Create data directories
RUN mkdir -p /app/data/{test_datasets,brain_atlases,calibration_files}
RUN mkdir -p /app/logs
RUN mkdir -p /app/outputs

# Expose ports for Jupyter, API, and visualization
EXPOSE 8888 8000 8001 8080

# Default command for development
CMD ["bash"]

# Production stage
FROM base as production

WORKDIR /app

# Copy requirements and install production dependencies only
COPY requirements/base.txt requirements/gpu.txt requirements/visualization.txt requirements/hardware.txt requirements/
RUN pip install -r base.txt -r gpu.txt -r visualization.txt -r hardware.txt

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

# Install Brain-Forge
RUN pip install .

# Create non-root user for security
RUN useradd -m -u 1000 brainforge && \
    chown -R brainforge:brainforge /app

# Create necessary directories
RUN mkdir -p /app/data/{test_datasets,brain_atlases,calibration_files} && \
    mkdir -p /app/logs && \
    mkdir -p /app/outputs && \
    chown -R brainforge:brainforge /app/data /app/logs /app/outputs

USER brainforge

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.core.config; print('Brain-Forge is healthy')" || exit 1

# Expose API port
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.api.rest_api"]

# Jupyter stage for interactive development
FROM development as jupyter

# Install additional Jupyter kernels
RUN pip install \
    ipykernel \
    ipywidgets \
    plotly \
    bokeh

# Set up custom Jupyter configuration
RUN mkdir -p /home/brainforge/.jupyter
COPY docker/jupyter_config.py /home/brainforge/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

# GPU-optimized stage
FROM production as gpu

# Install additional GPU libraries
RUN pip install \
    cupy-cuda11x \
    cudf-cu11 \
    cuml-cu11 \
    rapids-core

# Set CUDA environment variables
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=0

# GPU health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import cupy; cupy.cuda.Device(0).use(); print('GPU is available')" || exit 1

# Testing stage
FROM development as testing

# Copy test files
COPY tests/ tests/

# Install testing dependencies
RUN pip install pytest pytest-cov pytest-xdist pytest-benchmark

# Set test environment variables
ENV PYTHONPATH=/app/src
ENV TESTING=1

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=html", "--cov-report=term"]

# Documentation stage
FROM base as docs

WORKDIR /app

# Install documentation dependencies
RUN pip install \
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    sphinx-autodoc-typehints \
    pdoc3

# Copy source and docs
COPY src/ src/
COPY docs/ docs/
COPY pyproject.toml .

# Install Brain-Forge for documentation
RUN pip install .

# Build documentation
WORKDIR /app/docs
RUN make html

# Serve documentation
EXPOSE 8080
CMD ["python", "-m", "http.server", "8080", "--directory", "_build/html"]
