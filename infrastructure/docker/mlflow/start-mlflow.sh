#!/bin/bash

# MLflow Tracking Server Startup Script
# Production-ready startup with comprehensive error handling and monitoring

set -e

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a /opt/mlflow/logs/startup.log
}

# Function to check if database is ready
wait_for_database() {
    local host=$1
    local port=$2
    local max_tries=30
    local count=0
    
    log "Waiting for database connection at $host:$port..."
    
    while ! nc -z "$host" "$port" > /dev/null 2>&1; do
        count=$((count + 1))
        if [ $count -gt $max_tries ]; then
            log "ERROR: Database not available after $max_tries attempts"
            exit 1
        fi
        log "Database not ready, waiting... ($count/$max_tries)"
        sleep 2
    done
    
    log "Database is ready!"
}

# Function to initialize database
init_database() {
    log "Initializing MLflow database..."
    
    # Run database upgrade
    mlflow db upgrade "$MLFLOW_BACKEND_STORE_URI" || {
        log "ERROR: Database initialization failed"
        exit 1
    }
    
    log "Database initialization completed successfully"
}

# Function to create S3 bucket if it doesn't exist
create_s3_bucket() {
    if [ -n "$MLFLOW_S3_ENDPOINT_URL" ] && [ -n "$AWS_ACCESS_KEY_ID" ]; then
        log "Creating S3 bucket for artifacts..."
        
        # Extract bucket name from artifact root
        BUCKET_NAME=$(echo "$MLFLOW_DEFAULT_ARTIFACT_ROOT" | sed 's|s3://||' | cut -d'/' -f1)
        
        # Check if bucket exists, create if not
        if ! aws s3 --endpoint-url="$MLFLOW_S3_ENDPOINT_URL" ls "s3://$BUCKET_NAME" > /dev/null 2>&1; then
            aws s3 --endpoint-url="$MLFLOW_S3_ENDPOINT_URL" mb "s3://$BUCKET_NAME" || {
                log "WARNING: Failed to create S3 bucket, continuing..."
            }
            log "S3 bucket created: $BUCKET_NAME"
        else
            log "S3 bucket already exists: $BUCKET_NAME"
        fi
    fi
}

# Function to setup prometheus metrics
setup_prometheus() {
    if [ "$ENABLE_PROMETHEUS_METRICS" = "true" ]; then
        log "Setting up Prometheus metrics..."
        
        # Create prometheus multiproc directory
        mkdir -p /opt/mlflow/prometheus
        
        # Set environment variables
        export PROMETHEUS_MULTIPROC_DIR=/opt/mlflow/prometheus
        
        log "Prometheus metrics enabled"
    fi
}

# Function to validate configuration
validate_config() {
    log "Validating MLflow configuration..."
    
    # Check required environment variables
    if [ -z "$MLFLOW_BACKEND_STORE_URI" ]; then
        log "ERROR: MLFLOW_BACKEND_STORE_URI is not set"
        exit 1
    fi
    
    if [ -z "$MLFLOW_DEFAULT_ARTIFACT_ROOT" ]; then
        log "ERROR: MLFLOW_DEFAULT_ARTIFACT_ROOT is not set"
        exit 1
    fi
    
    # Check if artifact root is accessible
    if [[ "$MLFLOW_DEFAULT_ARTIFACT_ROOT" == file://* ]]; then
        ARTIFACT_PATH=$(echo "$MLFLOW_DEFAULT_ARTIFACT_ROOT" | sed 's|file://||')
        if [ ! -d "$ARTIFACT_PATH" ]; then
            log "Creating artifact directory: $ARTIFACT_PATH"
            mkdir -p "$ARTIFACT_PATH"
        fi
    fi
    
    log "Configuration validation completed"
}

# Function to setup logging
setup_logging() {
    log "Setting up logging configuration..."
    
    # Create logs directory
    mkdir -p /opt/mlflow/logs
    
    # Set up log rotation
    if command -v logrotate &> /dev/null; then
        cat > /tmp/mlflow-logrotate.conf << EOF
/opt/mlflow/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 mlflow mlflow
}
EOF
        logrotate -f /tmp/mlflow-logrotate.conf
    fi
    
    log "Logging configuration completed"
}

# Function to start MLflow server
start_mlflow() {
    log "Starting MLflow tracking server..."
    
    # Set default values
    MLFLOW_HOST=${MLFLOW_SERVER_HOST:-0.0.0.0}
    MLFLOW_PORT=${MLFLOW_SERVER_PORT:-5000}
    MLFLOW_WORKERS=${MLFLOW_WORKERS:-4}
    
    # Build MLflow command
    MLFLOW_CMD="mlflow server \
        --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
        --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT \
        --host $MLFLOW_HOST \
        --port $MLFLOW_PORT \
        --workers $MLFLOW_WORKERS \
        --gunicorn-opts '--timeout=120 --keepalive=2 --max-requests=1000'"
    
    # Add S3 endpoint if configured
    if [ -n "$MLFLOW_S3_ENDPOINT_URL" ]; then
        MLFLOW_CMD="$MLFLOW_CMD --artifacts-destination $MLFLOW_DEFAULT_ARTIFACT_ROOT"
    fi
    
    # Add authentication if configured
    if [ "$MLFLOW_AUTH_ENABLED" = "true" ]; then
        MLFLOW_CMD="$MLFLOW_CMD --enable-auth"
    fi
    
    log "MLflow command: $MLFLOW_CMD"
    
    # Start server with proper signal handling
    exec $MLFLOW_CMD
}

# Function to handle shutdown
shutdown_handler() {
    log "Received shutdown signal, gracefully shutting down..."
    
    # Clean up prometheus metrics
    if [ -d "/opt/mlflow/prometheus" ]; then
        rm -rf /opt/mlflow/prometheus/*
    fi
    
    log "Shutdown completed"
    exit 0
}

# Main execution
main() {
    log "Starting MLflow Tracking Server initialization..."
    
    # Set up signal handlers
    trap shutdown_handler SIGTERM SIGINT
    
    # Initialize components
    setup_logging
    validate_config
    setup_prometheus
    
    # Wait for database if configured
    if [[ "$MLFLOW_BACKEND_STORE_URI" == postgresql* ]]; then
        DB_HOST=$(echo "$MLFLOW_BACKEND_STORE_URI" | sed -n 's/.*@\([^:]*\):.*/\1/p')
        DB_PORT=$(echo "$MLFLOW_BACKEND_STORE_URI" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
        wait_for_database "$DB_HOST" "$DB_PORT"
        init_database
    fi
    
    # Create S3 bucket if configured
    create_s3_bucket
    
    log "MLflow Tracking Server initialization completed successfully"
    
    # Start MLflow server
    start_mlflow
}

# Run main function
main "$@" 