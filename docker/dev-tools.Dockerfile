# Brain-Forge Development Tools Container
# Comprehensive tooling environment for code quality, CI/CD, and development utilities
# Includes linters, formatters, security scanners, and documentation tools

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    gnupg \
    jq \
    lsb-release \
    python3 \
    python3-pip \
    software-properties-common \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for JavaScript tools
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Go for Go-based tools
RUN wget https://go.dev/dl/go1.21.0.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz \
    && rm go1.21.0.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"

# Install Rust for Rust-based tools
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python development and security tools
RUN pip3 install --no-cache-dir \
    # Code formatters
    black \
    isort \
    autopep8 \
    # Linters
    flake8 \
    pylint \
    mypy \
    bandit \
    # Security scanners
    safety \
    semgrep \
    # Documentation tools
    sphinx \
    sphinx-rtd-theme \
    mkdocs \
    mkdocs-material \
    # Testing tools
    pytest \
    pytest-cov \
    pytest-xdist \
    # Pre-commit hooks
    pre-commit \
    # Code complexity analysis
    radon \
    xenon \
    # Import sorting and analysis
    isort \
    vulture \
    # Type checking
    mypy \
    # Performance profiling
    py-spy \
    # Scientific code quality
    numpy-stubs \
    pandas-stubs

# Install Node.js development tools
RUN npm install -g \
    # Code formatters
    prettier \
    # Linters
    eslint \
    jshint \
    # TypeScript tools
    typescript \
    tslint \
    # Security scanners
    snyk \
    npm-audit \
    # Documentation generators
    jsdoc \
    typedoc \
    # API documentation
    @apidevtools/swagger-cli \
    # Testing tools
    jest \
    mocha \
    # Bundle analyzers
    webpack-bundle-analyzer \
    # Dependency management
    npm-check-updates \
    depcheck \
    # License checking
    license-checker \
    # Performance testing
    lighthouse \
    # Code complexity
    complexity-report \
    # Git hooks
    husky \
    lint-staged

# Install Go-based tools
RUN go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest \
    && go install github.com/securecodewarrior/sast-scan@latest \
    && go install github.com/mgechev/revive@latest \
    && go install honnef.co/go/tools/cmd/staticcheck@latest \
    && go install github.com/kisielk/errcheck@latest

# Install Rust-based tools
RUN cargo install \
    ripgrep \
    fd-find \
    tokei \
    cargo-audit \
    cargo-outdated \
    cargo-tree

# Install additional security and analysis tools
RUN wget https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64 \
    && chmod +x hadolint-Linux-x86_64 \
    && mv hadolint-Linux-x86_64 /usr/local/bin/hadolint

# Install Terraform for infrastructure as code
RUN wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | tee /usr/share/keyrings/hashicorp-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/hashicorp.list \
    && apt-get update && apt-get install -y terraform \
    && rm -rf /var/lib/apt/lists/*

# Install Docker for container operations
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Install Trivy for container security scanning
RUN wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | apt-key add - \
    && echo deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main | tee -a /etc/apt/sources.list.d/trivy.list \
    && apt-get update && apt-get install -y trivy \
    && rm -rf /var/lib/apt/lists/*

# Install additional development utilities
RUN apt-get update && apt-get install -y \
    # Network tools
    netcat \
    telnet \
    # Text processing
    jq \
    yq \
    # Monitoring
    htop \
    iotop \
    # File operations
    tree \
    && rm -rf /var/lib/apt/lists/*

# Create development user
RUN useradd -m -s /bin/bash -u 1000 dev \
    && usermod -aG sudo dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to development user
USER dev
WORKDIR /workspace

# Set up user environment
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/dev/.bashrc \
    && echo 'alias ll="ls -la"' >> /home/dev/.bashrc \
    && echo 'alias la="ls -A"' >> /home/dev/.bashrc \
    && echo 'alias format-python="black . && isort ."' >> /home/dev/.bashrc \
    && echo 'alias lint-python="flake8 . && pylint ."' >> /home/dev/.bashrc \
    && echo 'alias security-python="bandit -r . && safety check"' >> /home/dev/.bashrc \
    && echo 'alias format-js="prettier --write ."' >> /home/dev/.bashrc \
    && echo 'alias lint-js="eslint ."' >> /home/dev/.bashrc \
    && echo 'alias audit-js="npm audit && snyk test"' >> /home/dev/.bashrc

# Create tool configuration templates
RUN mkdir -p /home/dev/.config/templates

# Pre-commit configuration template
RUN cat > /home/dev/.config/templates/.pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.0.0
    hooks:
      - id: prettier
        types_or: [javascript, jsx, ts, tsx, json, yaml, markdown]
EOF

# ESLint configuration template
RUN cat > /home/dev/.config/templates/.eslintrc.js << 'EOF'
module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: true,
  },
  extends: [
    'eslint:recommended',
    '@typescript-eslint/recommended',
    'prettier',
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
  },
  plugins: ['@typescript-eslint', 'security'],
  rules: {
    'no-console': 'warn',
    'no-unused-vars': 'error',
    '@typescript-eslint/no-unused-vars': 'error',
    'security/detect-object-injection': 'warn',
  },
};
EOF

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 --version && node --version && go version && echo "Dev tools container healthy"

# Default command
CMD ["/bin/bash"]

LABEL maintainer="Brain-Forge Team" \
      version="1.0" \
      description="Development tools container with linters, formatters, and security scanners" \
      purpose="development-tools"
