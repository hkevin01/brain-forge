# Brain-Forge Frontend Development Container
# Isolated frontend development environment for BCI web interfaces
# Supports React, Vue, Angular, and modern build tools

FROM node:18-alpine

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apk update && apk add --no-cache \
    bash \
    git \
    curl \
    wget \
    vim \
    nano \
    htop \
    tree \
    jq \
    # Build dependencies
    python3 \
    make \
    g++ \
    # Graphics libraries for canvas/webgl
    cairo-dev \
    pango-dev \
    giflib-dev \
    librsvg-dev \
    pixman-dev \
    # For Puppeteer/Playwright
    chromium \
    nss \
    freetype \
    harfbuzz \
    ca-certificates \
    ttf-freefont

# Configure Puppeteer to use installed Chromium
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true \
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser

# Install global Node.js tools
RUN npm install -g \
    # Package managers
    yarn \
    pnpm \
    # Development servers and build tools
    vite \
    webpack \
    webpack-cli \
    webpack-dev-server \
    parcel \
    rollup \
    esbuild \
    # Frontend frameworks CLI tools
    @angular/cli \
    @vue/cli \
    create-react-app \
    create-next-app \
    @sveltejs/kit \
    # Testing frameworks
    jest \
    @jest/core \
    cypress \
    @playwright/test \
    # Code quality tools
    eslint \
    prettier \
    typescript \
    ts-node \
    # CSS tools
    sass \
    less \
    postcss \
    autoprefixer \
    tailwindcss \
    # Development utilities
    nodemon \
    concurrently \
    cross-env \
    dotenv-cli \
    # Static site generators
    @11ty/eleventy \
    gatsby-cli \
    # Component libraries
    @storybook/cli \
    # API and GraphQL tools
    graphql-cli \
    apollo \
    # Performance and analysis
    webpack-bundle-analyzer \
    lighthouse \
    # Documentation
    @storybook/cli \
    typedoc

# Install browser testing dependencies
RUN npx playwright install-deps

# Create development user
RUN addgroup -g 1000 dev && \
    adduser -D -s /bin/bash -u 1000 -G dev dev && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to development user
USER dev

# Set up development environment
WORKDIR /workspace

# Create common directories
RUN mkdir -p \
    /home/dev/.npm-global \
    /home/dev/.config \
    /home/dev/.local/bin \
    /home/dev/projects \
    /home/dev/logs

# Configure npm global directory
RUN npm config set prefix '/home/dev/.npm-global'

# Set up environment variables
ENV PATH="/home/dev/.npm-global/bin:$PATH" \
    NODE_ENV=development

# Configure Git (will be overridden by user config)
RUN git config --global user.name "Frontend Developer" \
    && git config --global user.email "frontend@brain-forge.local" \
    && git config --global init.defaultBranch main

# Set up shell aliases and environment
RUN echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> /home/dev/.bashrc && \
    echo 'alias ll="ls -la"' >> /home/dev/.bashrc && \
    echo 'alias la="ls -A"' >> /home/dev/.bashrc && \
    echo 'alias l="ls -CF"' >> /home/dev/.bashrc && \
    echo 'alias npm-check="npm outdated"' >> /home/dev/.bashrc && \
    echo 'alias yarn-check="yarn outdated"' >> /home/dev/.bashrc && \
    echo 'alias build-stats="npx webpack-bundle-analyzer build/static/js/*.js"' >> /home/dev/.bashrc

# Install user-level packages for Brain-Forge specific development
RUN npm install -g \
    # Brain-computer interface visualization
    d3 \
    three \
    plotly.js \
    # Real-time data handling
    socket.io-client \
    ws \
    # Scientific computing in JS
    ml-matrix \
    simple-statistics \
    # Data visualization
    chart.js \
    echarts \
    # UI component libraries
    @mui/material \
    antd \
    bootstrap

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD node --version && npm --version && echo "Frontend dev container healthy"

# Expose common frontend development ports
EXPOSE 3000 3001 4200 5173 8080 8081 9000 9001

# Set default command
CMD ["/bin/bash"]

# Labels
LABEL maintainer="Brain-Forge Team" \
      version="1.0" \
      description="Frontend development container for BCI web interfaces" \
      node.version="18" \
      purpose="frontend-development"
