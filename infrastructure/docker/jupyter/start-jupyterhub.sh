#!/bin/bash

# JupyterHub Startup Script
# Handles initialization, database setup, and service startup

set -e

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to wait for database
wait_for_db() {
    local host=$1
    local port=$2
    local timeout=30
    local count=0
    
    log "Waiting for database at $host:$port..."
    
    while ! nc -z "$host" "$port" > /dev/null 2>&1; do
        count=$((count + 1))
        if [ $count -gt $timeout ]; then
            log "ERROR: Database not available after $timeout seconds"
            exit 1
        fi
        log "Database not ready, waiting... ($count/$timeout)"
        sleep 1
    done
    
    log "Database is ready!"
}

# Function to setup database
setup_database() {
    log "Setting up JupyterHub database..."
    
    # Run database upgrade
    jupyterhub upgrade-db
    
    if [ $? -eq 0 ]; then
        log "Database setup completed successfully"
    else
        log "ERROR: Database setup failed"
        exit 1
    fi
}

# Function to create admin user
create_admin_user() {
    log "Creating admin user..."
    
    # Check if admin user exists
    if ! id "admin" &>/dev/null; then
        useradd -m -s /bin/bash -G sudo admin
        echo "admin:${ADMIN_PASSWORD:-admin}" | chpasswd
        log "Admin user created successfully"
    else
        log "Admin user already exists"
    fi
}

# Function to setup SSL certificates
setup_ssl() {
    local ssl_enabled=${SSL_ENABLED:-false}
    
    if [ "$ssl_enabled" = "true" ]; then
        log "Setting up SSL certificates..."
        
        # Check if certificates exist
        if [ ! -f "/srv/jupyterhub/ssl/cert.pem" ] || [ ! -f "/srv/jupyterhub/ssl/key.pem" ]; then
            log "Generating self-signed SSL certificates..."
            
            openssl req -x509 -newkey rsa:4096 \
                -keyout /srv/jupyterhub/ssl/key.pem \
                -out /srv/jupyterhub/ssl/cert.pem \
                -days 365 -nodes \
                -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
            
            chmod 600 /srv/jupyterhub/ssl/key.pem
            chmod 644 /srv/jupyterhub/ssl/cert.pem
            
            log "SSL certificates generated successfully"
        else
            log "SSL certificates already exist"
        fi
    else
        log "SSL disabled, using HTTP"
    fi
}

# Function to setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Create metrics directory
    mkdir -p /srv/jupyterhub/metrics
    
    # Set up Prometheus metrics endpoint
    if [ "${PROMETHEUS_ENABLED:-false}" = "true" ]; then
        log "Prometheus metrics enabled"
        export JUPYTERHUB_METRICS_ENABLED=true
    fi
}

# Function to setup logging
setup_logging() {
    log "Setting up logging..."
    
    # Create log directory
    mkdir -p /var/log/jupyterhub
    
    # Set log permissions
    chown -R root:root /var/log/jupyterhub
    chmod 755 /var/log/jupyterhub
    
    # Rotate logs
    if [ -f "/var/log/jupyterhub/jupyterhub.log" ]; then
        log "Rotating old log file"
        mv /var/log/jupyterhub/jupyterhub.log /var/log/jupyterhub/jupyterhub.log.$(date +%Y%m%d_%H%M%S)
    fi
}

# Function to cleanup old containers
cleanup_containers() {
    log "Cleaning up old containers..."
    
    # Remove stopped containers
    if command -v docker &> /dev/null; then
        docker container prune -f || true
        log "Container cleanup completed"
    fi
}

# Function to validate configuration
validate_config() {
    log "Validating JupyterHub configuration..."
    
    # Check configuration syntax
    if ! jupyterhub --config=/srv/jupyterhub/jupyterhub_config.py --generate-config --dry-run; then
        log "ERROR: Configuration validation failed"
        exit 1
    fi
    
    log "Configuration validation passed"
}

# Function to setup environment variables
setup_environment() {
    log "Setting up environment variables..."
    
    # Set default values
    export JUPYTERHUB_ENV=${JUPYTERHUB_ENV:-development}
    export JUPYTERHUB_AUTH_METHOD=${JUPYTERHUB_AUTH_METHOD:-native}
    export POSTGRES_HOST=${POSTGRES_HOST:-postgres}
    export POSTGRES_DB=${POSTGRES_DB:-jupyterhub}
    export POSTGRES_USER=${POSTGRES_USER:-jupyterhub}
    export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-jupyterhub}
    
    # Generate secrets if not provided
    if [ -z "$JUPYTERHUB_CRYPT_KEY" ]; then
        export JUPYTERHUB_CRYPT_KEY=$(openssl rand -hex 32)
        log "Generated new crypt key"
    fi
    
    if [ -z "$JUPYTERHUB_PROXY_TOKEN" ]; then
        export JUPYTERHUB_PROXY_TOKEN=$(openssl rand -hex 32)
        log "Generated new proxy token"
    fi
}

# Function to health check
health_check() {
    log "Performing health check..."
    
    # Check if JupyterHub is responding
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -f -s http://localhost:8000/hub/health > /dev/null 2>&1; then
            log "JupyterHub health check passed"
            return 0
        fi
        
        attempt=$((attempt + 1))
        log "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 2
    done
    
    log "ERROR: Health check failed after $max_attempts attempts"
    return 1
}

# Function to start JupyterHub
start_jupyterhub() {
    log "Starting JupyterHub..."
    
    # Start JupyterHub with configuration
    exec jupyterhub \
        --config=/srv/jupyterhub/jupyterhub_config.py \
        --log-level=${LOG_LEVEL:-INFO} \
        --log-file=/var/log/jupyterhub/jupyterhub.log \
        --pid-file=/var/run/jupyterhub.pid
}

# Main execution
main() {
    log "Starting JupyterHub initialization..."
    
    # Setup environment
    setup_environment
    
    # Setup logging
    setup_logging
    
    # Wait for database
    wait_for_db "${POSTGRES_HOST:-postgres}" 5432
    
    # Setup database
    setup_database
    
    # Create admin user
    create_admin_user
    
    # Setup SSL
    setup_ssl
    
    # Setup monitoring
    setup_monitoring
    
    # Cleanup old containers
    cleanup_containers
    
    # Validate configuration
    validate_config
    
    log "JupyterHub initialization completed successfully"
    
    # Start JupyterHub
    start_jupyterhub
}

# Handle signals
trap 'log "Received SIGTERM, shutting down gracefully..."; kill -TERM $PID; wait $PID' TERM
trap 'log "Received SIGINT, shutting down gracefully..."; kill -INT $PID; wait $PID' INT

# Run main function
main "$@" &
PID=$!
wait $PID 