# Brain-Forge Backend Development Container
# Multi-purpose backend development environment for BCI systems
# Supports Python, Node.js, and scientific computing

FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Set up locale
RUN apt-get update && apt-get install -y \
    locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    vim \
    nano \
    htop \
    tree \
    jq \
    # Network tools
    net-tools \
    iputils-ping \
    telnet \
    netcat \
    # SSL and crypto
    ca-certificates \
    gnupg \
    lsb-release \
    # Scientific computing dependencies
    gfortran \
    libblas-dev \
    liblapack-dev \
    libfftw3-dev \
    libhdf5-dev \
    pkg-config \
    # Graphics and GUI (for matplotlib, etc.)
    libx11-dev \
    libxrender1 \
    libxext6 \
    libxft2 \
    libfreetype6-dev \
    libpng-dev \
    # Database clients
    postgresql-client \
    mysql-client \
    redis-tools \
    # Monitoring tools
    sysstat \
    iotop \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (LTS version)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest yarn pnpm

# Install Python (multiple versions for compatibility)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install Poetry for Python dependency management
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Install Pipenv as alternative package manager
RUN pip3 install pipenv

# Install Go (for potential future integrations)
RUN wget https://go.dev/dl/go1.21.0.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz \
    && rm go1.21.0.linux-amd64.tar.gz

# Set Go environment
ENV PATH="/usr/local/go/bin:${PATH}" \
    GOPATH="/go" \
    GOBIN="/go/bin"

# Install Rust (for high-performance computing components)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . /root/.cargo/env

ENV PATH="/root/.cargo/bin:${PATH}"

# Install scientific Python packages globally
RUN pip3 install --no-cache-dir \
    # Core scientific computing
    numpy==1.24.3 \
    scipy==1.11.1 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    plotly==5.15.0 \
    # Neuroscience specific
    mne==1.4.2 \
    nilearn==0.10.1 \
    nibabel==5.1.0 \
    # Machine learning
    scikit-learn==1.3.0 \
    tensorflow==2.13.0 \
    torch==2.0.1 \
    # Async and networking
    asyncio \
    aiohttp==3.8.5 \
    fastapi==0.101.0 \
    uvicorn==0.23.2 \
    # Data processing
    h5py==3.9.0 \
    pyarrow==12.0.1 \
    # Testing and development
    pytest==7.4.0 \
    pytest-asyncio==0.21.1 \
    pytest-cov==4.1.0 \
    black==23.7.0 \
    isort==5.12.0 \
    flake8==6.0.0 \
    mypy==1.5.1 \
    # Jupyter for interactive development
    jupyter==1.0.0 \
    jupyterlab==4.0.5 \
    # Monitoring and profiling
    psutil==5.9.5 \
    memory-profiler==0.60.0

# Install Node.js development tools globally
RUN npm install -g \
    # Development servers
    nodemon \
    pm2 \
    # Build tools
    webpack \
    vite \
    parcel \
    # Testing
    jest \
    mocha \
    cypress \
    # Code quality
    eslint \
    prettier \
    typescript \
    ts-node \
    # API tools
    @apidevtools/swagger-cli \
    postman-collection-transformer

# Install Docker CLI (for Docker-in-Docker scenarios)
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Install docker-compose
RUN curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose \
    && chmod +x /usr/local/bin/docker-compose

# Create development user (non-root for security)
RUN useradd -m -s /bin/bash -u 1000 dev \
    && usermod -aG sudo dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set up development environment
USER dev
WORKDIR /workspace

# Create common development directories
RUN mkdir -p \
    /home/dev/.ssh \
    /home/dev/.config \
    /home/dev/.local/bin \
    /home/dev/projects \
    /home/dev/logs

# Configure Git (will be overridden by user config)
RUN git config --global user.name "Brain-Forge Developer" \
    && git config --global user.email "dev@brain-forge.local" \
    && git config --global init.defaultBranch main

# Install user-level Python packages
RUN pip3 install --user \
    # Development tools
    ipython \
    jupyter-lab \
    # Code formatters
    autopep8 \
    # Documentation
    sphinx \
    sphinx-rtd-theme

# Set up shell environment
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/dev/.bashrc \
    && echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> /home/dev/.bashrc \
    && echo 'alias ll="ls -la"' >> /home/dev/.bashrc \
    && echo 'alias la="ls -A"' >> /home/dev/.bashrc \
    && echo 'alias l="ls -CF"' >> /home/dev/.bashrc

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 --version && node --version && echo "Backend dev container healthy"

# Expose common development ports
EXPOSE 3000 5000 8000 8080 8888 9000

# Default command
CMD ["/bin/bash"]

# Labels for maintenance
LABEL maintainer="Brain-Forge Team" \
      version="1.0" \
      description="Multi-purpose backend development container for BCI systems" \
      python.version="3.11" \
      node.version="LTS" \
      purpose="development"
