# Great Expectations Data Quality Platform - Production Ready
# Multi-stage build with comprehensive data validation capabilities

# Stage 1: Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG GE_VERSION=0.18.3
ARG BUILDKIT_INLINE_CACHE=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production stage
FROM python:3.11-slim as production

# Set metadata labels
LABEL maintainer="Data Engineering Team" \
      version="1.0.0" \
      description="Production Great Expectations Platform" \
      org.opencontainers.image.title="Great Expectations" \
      org.opencontainers.image.description="Production-ready data quality validation platform" \
      org.opencontainers.image.vendor="Modern Data Stack Showcase" \
      org.opencontainers.image.version="1.0.0"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    libxml2 \
    libxslt1.1 \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create great_expectations user for security
RUN groupadd -r ge && useradd -r -g ge ge

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /opt/great_expectations

# Copy configuration files
COPY great_expectations.yml /opt/great_expectations/
COPY start-ge.sh /opt/great_expectations/
COPY healthcheck.py /opt/great_expectations/
COPY config/ /opt/great_expectations/config/
COPY expectations/ /opt/great_expectations/expectations/
COPY checkpoints/ /opt/great_expectations/checkpoints/

# Create necessary directories
RUN mkdir -p \
    /opt/great_expectations/data \
    /opt/great_expectations/logs \
    /opt/great_expectations/plugins \
    /opt/great_expectations/uncommitted \
    /opt/great_expectations/data_docs \
    /opt/great_expectations/validations \
    && chown -R ge:ge /opt/great_expectations

# Set permissions
RUN chmod +x /opt/great_expectations/start-ge.sh && \
    chmod +x /opt/great_expectations/healthcheck.py

# Switch to ge user
USER ge

# Set environment variables
ENV PYTHONPATH=/opt/great_expectations:$PYTHONPATH
ENV GE_HOME=/opt/great_expectations
ENV GE_CONFIG_DIR=/opt/great_expectations/config

# Expose port for data docs
EXPOSE 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /opt/great_expectations/healthcheck.py

# Use dumb-init for proper signal handling
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# Start Great Expectations
CMD ["/opt/great_expectations/start-ge.sh"] 