# MLflow Tracking Server - Production Ready
# Multi-stage build with security scanning and optimization

# Stage 1: Build stage with dependencies
FROM python:3.11-slim as builder

# Set build arguments
ARG MLFLOW_VERSION=2.8.1
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
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir mlflow==${MLFLOW_VERSION}

# Stage 2: Production stage
FROM python:3.11-slim as production

# Set metadata labels
LABEL maintainer="Data Engineering Team" \
      version="1.0.0" \
      description="Production MLflow Tracking Server" \
      org.opencontainers.image.title="MLflow Tracking Server" \
      org.opencontainers.image.description="Production-ready MLflow tracking server with PostgreSQL and S3 support" \
      org.opencontainers.image.vendor="Modern Data Stack Showcase" \
      org.opencontainers.image.version="1.0.0"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create mlflow user for security
RUN groupadd -r mlflow && useradd -r -g mlflow mlflow

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /opt/mlflow

# Copy configuration files
COPY mlflow.conf /opt/mlflow/
COPY start-mlflow.sh /opt/mlflow/
COPY healthcheck.py /opt/mlflow/

# Create necessary directories
RUN mkdir -p /opt/mlflow/artifacts /opt/mlflow/logs /opt/mlflow/config && \
    chown -R mlflow:mlflow /opt/mlflow

# Set permissions
RUN chmod +x /opt/mlflow/start-mlflow.sh && \
    chmod +x /opt/mlflow/healthcheck.py

# Switch to mlflow user
USER mlflow

# Set environment variables
ENV MLFLOW_SERVER_HOST=0.0.0.0
ENV MLFLOW_SERVER_PORT=5000
ENV MLFLOW_ARTIFACT_ROOT=/opt/mlflow/artifacts
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/opt/mlflow/artifacts
ENV PYTHONPATH=/opt/mlflow:$PYTHONPATH

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /opt/mlflow/healthcheck.py

# Use dumb-init for proper signal handling
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# Start MLflow server
CMD ["/opt/mlflow/start-mlflow.sh"] 