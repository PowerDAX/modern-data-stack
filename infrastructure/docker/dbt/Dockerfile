# Multi-stage Dockerfile for dbt with advanced capabilities
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    postgresql-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /usr/app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create dbt profiles directory
RUN mkdir -p /usr/app/.dbt

# Copy dbt profiles configuration
COPY profiles.yml /usr/app/.dbt/profiles.yml

# Set proper permissions
RUN chmod -R 755 /usr/app

# Expose dbt docs port
EXPOSE 8080

# Default command
CMD ["dbt", "--version"] 