# Automation & Utility Scripts

## Overview

The `scripts` directory contains automation and utility scripts for the Modern Data Stack Showcase platform. These scripts provide essential functionality for environment setup, deployment automation, and ongoing maintenance operations across the entire data platform ecosystem.

## 🎯 **Script Categories**

### **Setup Scripts** (`setup/`)
Environment initialization and configuration scripts:

- **Development Environment**: Local development setup and dependencies
- **Cloud Infrastructure**: Initial cloud resource provisioning  
- **Database Setup**: Database initialization and schema creation
- **Service Configuration**: Service discovery and configuration management

### **Deployment Scripts** (`deployment/`)
Automated deployment and release management:

- **Application Deployment**: Automated application releases
- **Infrastructure Deployment**: Infrastructure as Code deployment
- **Database Migrations**: Schema updates and data migrations
- **Configuration Management**: Environment-specific configurations

### **Maintenance Scripts** (`maintenance/`)
Ongoing operational and maintenance tasks:

- **Data Cleanup**: Automated data retention and cleanup
- **Performance Optimization**: System optimization and tuning
- **Health Monitoring**: System health checks and diagnostics
- **Backup & Recovery**: Automated backup and recovery procedures

## 📁 **Directory Structure**

```
scripts/
├── README.md                    # This overview document
├── setup/                       # Environment setup and initialization
│   ├── setup_dev_environment.sh # Complete development environment setup
│   ├── install_dependencies.sh  # Install required dependencies
│   ├── configure_database.sh    # Database setup and initialization
│   ├── setup_monitoring.sh      # Monitoring stack configuration
│   └── validate_setup.sh        # Environment validation and testing
├── deployment/                  # Deployment automation scripts
│   ├── deploy_infrastructure.sh # Infrastructure deployment automation
│   ├── deploy_applications.sh   # Application deployment automation
│   ├── run_migrations.sh        # Database migration execution
│   ├── update_configs.sh        # Configuration updates and rollouts
│   └── rollback_deployment.sh   # Automated rollback procedures
└── maintenance/                 # Maintenance and operational scripts
    ├── cleanup_data.sh          # Data retention and cleanup
    ├── optimize_performance.sh  # Performance optimization routines
    ├── health_check.sh          # System health monitoring
    ├── backup_system.sh         # Automated backup procedures
    └── generate_reports.sh      # Operational reporting
```

## 🛠️ **Setup Scripts** (`setup/`)

### **Development Environment Setup**
```bash
# Complete development environment initialization
./scripts/setup/setup_dev_environment.sh

# Individual component setup
./scripts/setup/install_dependencies.sh
./scripts/setup/configure_database.sh
./scripts/setup/setup_monitoring.sh
```

### **Script Features**
- **Dependency Management**: Automated installation of required tools and packages
- **Environment Configuration**: Setup of environment variables and configurations
- **Service Initialization**: Starting and configuring required services
- **Validation Testing**: Verification that setup completed successfully

### **Example: Development Environment Setup**
```bash
#!/bin/bash
# setup_dev_environment.sh - Complete development environment setup

set -e  # Exit on any error

echo "🚀 Setting up Modern Data Stack development environment..."

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting." >&2; exit 1; }

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install poetry
poetry install

# Setup Docker environment
echo "🐳 Setting up Docker environment..."
docker-compose up -d postgres redis mlflow

# Initialize database
echo "🗄️ Initializing database..."
./scripts/setup/configure_database.sh

# Setup monitoring
echo "📊 Setting up monitoring..."
./scripts/setup/setup_monitoring.sh

# Validate setup
echo "✅ Validating setup..."
./scripts/setup/validate_setup.sh

echo "🎉 Development environment setup complete!"
```

## 🚀 **Deployment Scripts** (`deployment/`)

### **Infrastructure Deployment**
```bash
# Deploy complete infrastructure
./scripts/deployment/deploy_infrastructure.sh --env production

# Deploy specific components
./scripts/deployment/deploy_applications.sh --app ml-pipeline
./scripts/deployment/run_migrations.sh --target latest
```

### **Deployment Features**
- **Environment Management**: Support for dev, staging, production environments
- **Rolling Deployments**: Zero-downtime deployment strategies
- **Health Checks**: Automated health validation after deployment
- **Rollback Capability**: Automated rollback on deployment failure

### **Example: Application Deployment**
```bash
#!/bin/bash
# deploy_applications.sh - Automated application deployment

set -e

ENVIRONMENT=${1:-staging}
APP_NAME=${2:-all}

echo "🚀 Deploying applications to $ENVIRONMENT environment..."

# Pre-deployment validation
echo "✅ Running pre-deployment checks..."
./scripts/deployment/validate_environment.sh --env $ENVIRONMENT

# Deploy infrastructure updates
echo "🏗️ Updating infrastructure..."
cd infrastructure/terraform
terraform plan -var-file="$ENVIRONMENT.tfvars"
terraform apply -var-file="$ENVIRONMENT.tfvars" -auto-approve

# Deploy applications
echo "📦 Deploying applications..."
kubectl apply -f infrastructure/kubernetes/$ENVIRONMENT/

# Run database migrations
echo "🗄️ Running database migrations..."
./scripts/deployment/run_migrations.sh --env $ENVIRONMENT

# Health checks
echo "🔍 Running health checks..."
./scripts/deployment/health_check.sh --env $ENVIRONMENT

echo "✅ Deployment to $ENVIRONMENT completed successfully!"
```

## 🔧 **Maintenance Scripts** (`maintenance/`)

### **System Maintenance**
```bash
# Daily maintenance routine
./scripts/maintenance/health_check.sh
./scripts/maintenance/cleanup_data.sh --days 30

# Performance optimization
./scripts/maintenance/optimize_performance.sh

# Backup operations
./scripts/maintenance/backup_system.sh --type full
```

### **Maintenance Features**
- **Automated Scheduling**: Cron-compatible scripts for automated execution
- **Resource Monitoring**: System resource usage tracking and optimization
- **Data Lifecycle**: Automated data retention and cleanup policies
- **Performance Tuning**: Database and application optimization routines

### **Example: System Health Check**
```bash
#!/bin/bash
# health_check.sh - Comprehensive system health monitoring

set -e

echo "🔍 Running system health checks..."

# Check service status
echo "📊 Checking service status..."
kubectl get pods --all-namespaces | grep -v Running | grep -v Completed && echo "❌ Some pods are not running" || echo "✅ All pods are running"

# Check database connectivity
echo "🗄️ Checking database connectivity..."
psql $DATABASE_URL -c "SELECT 1;" > /dev/null && echo "✅ Database connection successful" || echo "❌ Database connection failed"

# Check MLflow tracking server
echo "🤖 Checking MLflow server..."
curl -s http://localhost:5000/health > /dev/null && echo "✅ MLflow server healthy" || echo "❌ MLflow server unhealthy"

# Check disk usage
echo "💾 Checking disk usage..."
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "⚠️ Disk usage is high: ${DISK_USAGE}%"
else
    echo "✅ Disk usage is normal: ${DISK_USAGE}%"
fi

# Generate health report
echo "📋 Generating health report..."
./scripts/maintenance/generate_reports.sh --type health

echo "✅ Health check completed!"
```

## ⚙️ **Script Configuration**

### **Environment Variables**
```bash
# Common environment variables used by scripts
export ENVIRONMENT=development
export DATABASE_URL=postgresql://localhost:5432/showcase_db
export MLFLOW_TRACKING_URI=http://localhost:5000
export KUBERNETES_NAMESPACE=default
export BACKUP_LOCATION=s3://backups/modern-data-stack
export LOG_LEVEL=INFO
```

### **Configuration Files**
```yaml
# scripts/config/default.yaml
environments:
  development:
    database_url: postgresql://localhost:5432/showcase_dev
    kubernetes_context: docker-desktop
  staging:
    database_url: postgresql://staging-db:5432/showcase_staging
    kubernetes_context: staging-cluster
  production:
    database_url: postgresql://prod-db:5432/showcase_prod
    kubernetes_context: production-cluster

backup:
  retention_days: 30
  schedule: "0 2 * * *"  # Daily at 2 AM
  
monitoring:
  health_check_interval: 300  # 5 minutes
  alert_thresholds:
    cpu_usage: 80
    memory_usage: 85
    disk_usage: 80
```

## 🔒 **Security & Best Practices**

### **Security Guidelines**
- **Credential Management**: Use environment variables and secret management
- **Access Control**: Implement proper permissions and access controls
- **Audit Logging**: Log all script executions and changes
- **Input Validation**: Validate all script parameters and inputs

### **Script Best Practices**
```bash
#!/bin/bash
# Best practices template

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# Script metadata
SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG_FILE="/var/log/${SCRIPT_NAME%.sh}.log"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Input validation
validate_input() {
    [[ -z "$1" ]] && error_exit "Required parameter missing"
}

# Main script logic
main() {
    log "Starting $SCRIPT_NAME"
    # Script implementation
    log "Completed $SCRIPT_NAME successfully"
}

# Script execution
main "$@"
```

## 📊 **Monitoring & Logging**

### **Script Monitoring**
- **Execution Tracking**: Monitor script execution status and duration
- **Error Alerting**: Automated alerts for script failures
- **Performance Metrics**: Track script performance and resource usage
- **Audit Trail**: Complete audit trail of all script executions

### **Logging Configuration**
```bash
# Centralized logging for all scripts
exec 1> >(logger -t "$SCRIPT_NAME" -p user.info)
exec 2> >(logger -t "$SCRIPT_NAME" -p user.error)

# Structured logging
log_structured() {
    echo "{\"timestamp\":\"$(date -Iseconds)\",\"script\":\"$SCRIPT_NAME\",\"level\":\"$1\",\"message\":\"$2\"}"
}
```

## 🚀 **Usage Examples**

### **Complete Environment Setup**
```bash
# Setup complete development environment
git clone <repository>
cd modern-data-stack-showcase
./scripts/setup/setup_dev_environment.sh

# Validate setup
./scripts/setup/validate_setup.sh
```

### **Production Deployment**
```bash
# Deploy to production
./scripts/deployment/deploy_infrastructure.sh --env production
./scripts/deployment/deploy_applications.sh --env production

# Verify deployment
./scripts/deployment/health_check.sh --env production
```

### **Regular Maintenance**
```bash
# Daily maintenance (typically run via cron)
0 2 * * * /path/to/scripts/maintenance/backup_system.sh
0 3 * * * /path/to/scripts/maintenance/cleanup_data.sh --days 30
0 4 * * * /path/to/scripts/maintenance/optimize_performance.sh
```

## 📚 **Documentation & Support**

### **Script Documentation**
- **Inline Comments**: Comprehensive commenting for complex logic
- **Usage Documentation**: Clear usage instructions and examples
- **Error Handling**: Detailed error messages and troubleshooting guides
- **Change Log**: Track changes and version history

### **Getting Help**
- **Script Help**: Run scripts with `--help` flag for usage information
- **Documentation**: Refer to individual script documentation
- **Troubleshooting**: Check logs and error messages for debugging
- **Support**: Contact platform team for assistance

---

**🛠️ These scripts provide essential automation for the Modern Data Stack platform, ensuring reliable and consistent operations across all environments.** 