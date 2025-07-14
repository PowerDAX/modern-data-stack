# MODERN DATA STACK DEPLOYMENT GUIDE

## Overview

This comprehensive deployment guide provides step-by-step instructions for deploying the Modern Data Stack Showcase platform. The guide covers infrastructure provisioning, application deployment, security configuration, and operational procedures.

## Prerequisites

### **System Requirements**

#### Development Environment
- **Operating System**: Linux, macOS, or Windows with WSL2
- **Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**: 100GB free space minimum
- **Network**: Stable internet connection for cloud resources

#### Required Tools
```bash
# Core tools (install these first)
terraform >= 1.6.0
kubectl >= 1.28.0
docker >= 24.0.0
helm >= 3.12.0
git >= 2.40.0

# Additional tools
aws-cli >= 2.13.0
azure-cli >= 2.50.0
gcloud >= 440.0.0
jq >= 1.6
curl >= 7.80.0
```

### **Cloud Provider Setup**

#### AWS Configuration
```bash
# Install AWS CLI and configure credentials
aws configure
# Set region, access key, secret key

# Verify configuration
aws sts get-caller-identity
aws eks list-clusters
```

#### Azure Configuration (Optional)
```bash
# Install Azure CLI and login
az login
az account set --subscription "your-subscription-id"

# Verify configuration
az account show
az aks list
```

#### GCP Configuration (Optional)
```bash
# Install gcloud CLI and authenticate
gcloud auth login
gcloud config set project your-project-id

# Verify configuration
gcloud auth list
gcloud container clusters list
```

### **Tool Installation**

#### Install Required Tools (Ubuntu/Debian)
```bash
#!/bin/bash
# Modern Data Stack - Tool Installation Script

set -e

echo "Installing Modern Data Stack deployment tools..."

# Update package manager
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Terraform
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform

# Install Helm
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update && sudo apt-get install helm

# Install additional tools
sudo apt-get install -y jq curl git unzip

echo "Tool installation completed!"
echo "Please log out and log back in to use Docker without sudo."
```

#### Install Required Tools (macOS)
```bash
#!/bin/bash
# Modern Data Stack - Tool Installation Script (macOS)

set -e

echo "Installing Modern Data Stack deployment tools..."

# Install Homebrew if not present
if ! command -v brew &> /dev/null; then
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install tools via Homebrew
brew install terraform
brew install kubectl
brew install helm
brew install docker
brew install aws-cli
brew install azure-cli
brew install google-cloud-sdk
brew install jq
brew install git

echo "Tool installation completed!"
```

## Infrastructure Deployment

### **Phase 1: Terraform Infrastructure**

#### 1. Configure Terraform Backend
```bash
# Navigate to terraform directory
cd infrastructure/terraform

# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit configuration for your environment
nano terraform.tfvars
```

#### 2. Terraform Configuration Example
```hcl
# terraform.tfvars
project_name = "modern-data-stack-showcase"
environment  = "dev"
owner        = "your-team"
cost_center  = "engineering"

# AWS Configuration
aws_region = "us-west-2"

# Network Configuration
vpc_cidr            = "10.0.0.0/16"
allowed_cidr_blocks = ["10.0.0.0/8", "172.16.0.0/12"]

# Kubernetes Configuration
kubernetes_version              = "1.28"
eks_node_instance_types         = ["t3.medium", "t3.large"]
eks_node_group_min_size         = 1
eks_node_group_max_size         = 5
eks_node_group_desired_size     = 2

# Database Configuration
rds_engine                    = "postgres"
rds_engine_version           = "15.4"
rds_instance_class           = "db.t3.medium"
rds_allocated_storage        = 100

# Application Deployment
deploy_airflow           = true
deploy_mlflow           = true
deploy_great_expectations = true
deploy_jupyter          = true

# Monitoring Configuration
create_cloudwatch_dashboard = true
create_prometheus_workspace = true
create_grafana_workspace    = true

# Alerting Configuration
alert_email_endpoints = [
  "your-team@company.com"
]
slack_webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
slack_channel     = "#data-alerts"
```

#### 3. Deploy Infrastructure
```bash
# Initialize Terraform
terraform init

# Validate configuration
terraform validate

# Plan deployment
terraform plan -out=tfplan

# Review the plan thoroughly
terraform show tfplan

# Apply infrastructure changes
terraform apply tfplan

# Save outputs for later use
terraform output -json > terraform-outputs.json
```

#### 4. Verify Infrastructure Deployment
```bash
# Verify EKS cluster
aws eks describe-cluster --name $(terraform output -raw eks_cluster_name)

# Configure kubectl
aws eks update-kubeconfig --region $(terraform output -raw aws_region) --name $(terraform output -raw eks_cluster_name)

# Verify cluster connectivity
kubectl get nodes
kubectl get namespaces
```

### **Phase 2: Kubernetes Application Deployment**

#### 1. Deploy Core Infrastructure Components
```bash
# Deploy namespaces and security policies
kubectl apply -f infrastructure/kubernetes/namespaces/

# Deploy PostgreSQL
kubectl apply -f infrastructure/kubernetes/deployments/postgres.yaml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l app=postgresql -n modern-data-stack --timeout=300s

# Deploy Redis
kubectl apply -f infrastructure/kubernetes/deployments/redis.yaml
```

#### 2. Deploy Application Services
```bash
# Deploy MLflow
kubectl apply -f infrastructure/kubernetes/deployments/mlflow.yaml

# Deploy Great Expectations
kubectl apply -f infrastructure/kubernetes/deployments/great-expectations.yaml

# Deploy Jupyter
kubectl apply -f infrastructure/kubernetes/deployments/jupyter.yaml

# Verify deployments
kubectl get deployments -n modern-data-stack
kubectl get services -n modern-data-stack
```

#### 3. Deploy Monitoring Stack
```bash
# Deploy Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values infrastructure/monitoring/prometheus/values.yaml

# Deploy custom dashboards
kubectl apply -f infrastructure/monitoring/grafana/dashboards/

# Verify monitoring deployment
kubectl get pods -n monitoring
```

### **Phase 3: Security Configuration**

#### 1. Deploy Security Policies
```bash
# Deploy Kyverno policy engine
helm repo add kyverno https://kyverno.github.io/kyverno/
helm install kyverno kyverno/kyverno \
  --namespace kyverno \
  --create-namespace

# Deploy security policies
kubectl apply -f infrastructure/security/governance/security-policies.yaml

# Verify policy deployment
kubectl get clusterpolicies
kubectl get networkpolicies -A
```

#### 2. Configure RBAC
```bash
# Create service accounts and role bindings
kubectl apply -f infrastructure/security/rbac/

# Verify RBAC configuration
kubectl get clusterroles | grep data-
kubectl get rolebindings -A
```

### **Phase 4: Logging Infrastructure**

#### 1. Deploy ELK Stack
```bash
# Deploy Elasticsearch
kubectl apply -f infrastructure/logging/elk-stack/elasticsearch.yaml

# Wait for Elasticsearch cluster to be ready
kubectl wait --for=condition=ready pod -l app=elasticsearch -n logging --timeout=600s

# Deploy Logstash
kubectl apply -f infrastructure/logging/elk-stack/logstash.yaml

# Deploy Kibana
kubectl apply -f infrastructure/logging/elk-stack/kibana.yaml

# Verify ELK stack deployment
kubectl get pods -n logging
kubectl get services -n logging
```

#### 2. Configure Log Shipping
```bash
# Deploy Filebeat for log collection
helm repo add elastic https://helm.elastic.co
helm install filebeat elastic/filebeat \
  --namespace logging \
  --values infrastructure/logging/filebeat/values.yaml

# Verify log shipping
kubectl logs -l app=filebeat -n logging
```

## Application Configuration

### **Phase 1: Database Setup**

#### 1. PostgreSQL Configuration
```bash
# Get PostgreSQL connection details
POSTGRES_HOST=$(kubectl get service postgresql -n modern-data-stack -o jsonpath='{.spec.clusterIP}')
POSTGRES_PASSWORD=$(kubectl get secret postgresql-credentials -n modern-data-stack -o jsonpath='{.data.password}' | base64 -d)

# Connect to PostgreSQL
kubectl exec -it deployment/postgresql -n modern-data-stack -- psql -U postgres

# Create databases
CREATE DATABASE mlflow;
CREATE DATABASE airflow;
CREATE DATABASE superset;
CREATE DATABASE great_expectations;
```

#### 2. Initialize Application Databases
```bash
# Initialize MLflow database
kubectl exec -it deployment/mlflow -n modern-data-stack -- mlflow db upgrade

# Initialize Airflow database
kubectl exec -it deployment/airflow-webserver -n modern-data-stack -- airflow db init

# Create Airflow admin user
kubectl exec -it deployment/airflow-webserver -n modern-data-stack -- \
  airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin123
```

### **Phase 2: ML Platform Setup**

#### 1. MLflow Configuration
```bash
# Access MLflow UI
kubectl port-forward service/mlflow 5000:5000 -n modern-data-stack

# Open browser to http://localhost:5000
echo "MLflow UI available at: http://localhost:5000"

# Test MLflow API
curl http://localhost:5000/api/2.0/mlflow/experiments/list
```

#### 2. Jupyter Configuration
```bash
# Access Jupyter Lab
kubectl port-forward service/jupyter 8888:8888 -n modern-data-stack

# Get Jupyter token
kubectl logs deployment/jupyter -n modern-data-stack | grep "http://127.0.0.1:8888/lab?token="

# Open browser to Jupyter Lab
echo "Jupyter Lab available at: http://localhost:8888"
```

### **Phase 3: Data Pipeline Setup**

#### 1. Great Expectations Configuration
```bash
# Access Great Expectations
kubectl port-forward service/great-expectations 8080:8080 -n modern-data-stack

# Initialize Great Expectations project
kubectl exec -it deployment/great-expectations -n modern-data-stack -- \
  great_expectations init

# Open browser to Great Expectations
echo "Great Expectations available at: http://localhost:8080"
```

#### 2. Airflow Configuration
```bash
# Access Airflow UI
kubectl port-forward service/airflow-webserver 8080:8080 -n modern-data-stack

# Open browser to Airflow
echo "Airflow UI available at: http://localhost:8080"
echo "Username: admin"
echo "Password: admin123"

# Verify DAGs
kubectl exec -it deployment/airflow-webserver -n modern-data-stack -- \
  airflow dags list
```

## Verification and Testing

### **Phase 1: Infrastructure Verification**

#### 1. Cluster Health Check
```bash
#!/bin/bash
# Infrastructure Health Check Script

echo "=== Cluster Health Check ==="

# Check cluster status
echo "Cluster Status:"
kubectl cluster-info

# Check node status
echo -e "\nNode Status:"
kubectl get nodes -o wide

# Check system pods
echo -e "\nSystem Pods:"
kubectl get pods -n kube-system

# Check resource usage
echo -e "\nResource Usage:"
kubectl top nodes
kubectl top pods -A
```

#### 2. Application Health Check
```bash
#!/bin/bash
# Application Health Check Script

echo "=== Application Health Check ==="

# Check application deployments
echo "Application Deployments:"
kubectl get deployments -n modern-data-stack

# Check application services
echo -e "\nApplication Services:"
kubectl get services -n modern-data-stack

# Check application pods
echo -e "\nApplication Pods:"
kubectl get pods -n modern-data-stack -o wide

# Check application logs for errors
echo -e "\nRecent Application Logs:"
kubectl logs -l app=mlflow -n modern-data-stack --tail=10
kubectl logs -l app=jupyter -n modern-data-stack --tail=10
```

#### 3. Security Verification
```bash
#!/bin/bash
# Security Verification Script

echo "=== Security Verification ==="

# Check security policies
echo "Security Policies:"
kubectl get clusterpolicies
kubectl get networkpolicies -A

# Check RBAC configuration
echo -e "\nRBAC Configuration:"
kubectl get clusterroles | grep data-
kubectl get rolebindings -A | grep data-

# Check pod security
echo -e "\nPod Security:"
kubectl get pods -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.securityContext}{"\n"}{end}' -n modern-data-stack
```

### **Phase 2: Functional Testing**

#### 1. ML Pipeline Test
```bash
#!/bin/bash
# ML Pipeline Test Script

echo "=== ML Pipeline Test ==="

# Test MLflow connection
echo "Testing MLflow..."
curl -f http://localhost:5000/api/2.0/mlflow/experiments/list || echo "MLflow test failed"

# Test Jupyter access
echo "Testing Jupyter..."
curl -f http://localhost:8888/api || echo "Jupyter test failed"

# Run sample notebook
echo "Running sample notebook..."
kubectl exec -it deployment/jupyter -n modern-data-stack -- \
  jupyter nbconvert --execute --to notebook /home/jovyan/notebooks/01-feature-engineering.ipynb
```

#### 2. Data Pipeline Test
```bash
#!/bin/bash
# Data Pipeline Test Script

echo "=== Data Pipeline Test ==="

# Test PostgreSQL connection
echo "Testing PostgreSQL..."
kubectl exec -it deployment/postgresql -n modern-data-stack -- \
  psql -U postgres -c "SELECT version();"

# Test Great Expectations
echo "Testing Great Expectations..."
kubectl exec -it deployment/great-expectations -n modern-data-stack -- \
  great_expectations checkpoint run sample_checkpoint

# Test Airflow DAGs
echo "Testing Airflow DAGs..."
kubectl exec -it deployment/airflow-webserver -n modern-data-stack -- \
  airflow dags trigger sample_data_pipeline
```

### **Phase 3: Performance Testing**

#### 1. Load Testing
```bash
#!/bin/bash
# Load Testing Script

echo "=== Load Testing ==="

# Test MLflow under load
echo "Testing MLflow performance..."
for i in {1..10}; do
  curl -X POST http://localhost:5000/api/2.0/mlflow/experiments/create \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"test_experiment_$i\"}" &
done
wait

# Test database performance
echo "Testing database performance..."
kubectl exec -it deployment/postgresql -n modern-data-stack -- \
  pgbench -i -s 10 postgres
kubectl exec -it deployment/postgresql -n modern-data-stack -- \
  pgbench -c 10 -j 2 -t 1000 postgres
```

## Monitoring and Alerting Setup

### **Phase 1: Metrics Configuration**

#### 1. Prometheus Configuration
```bash
# Verify Prometheus targets
kubectl port-forward service/prometheus-kube-prometheus-prometheus 9090:9090 -n monitoring

# Open browser to http://localhost:9090
# Check Status -> Targets to verify all targets are up
```

#### 2. Grafana Dashboard Setup
```bash
# Access Grafana
kubectl port-forward service/prometheus-grafana 3000:3000 -n monitoring

# Get Grafana admin password
kubectl get secret prometheus-grafana -n monitoring -o jsonpath="{.data.admin-password}" | base64 -d

# Open browser to http://localhost:3000
# Username: admin
# Password: [from above command]

# Import custom dashboards
curl -X POST http://admin:password@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @infrastructure/monitoring/grafana/dashboards/modern-data-stack-overview.json
```

### **Phase 2: Alerting Configuration**

#### 1. AlertManager Setup
```bash
# Configure AlertManager
kubectl apply -f infrastructure/monitoring/alerting/alertmanager-config.yaml

# Test alert routing
kubectl exec -it alertmanager-prometheus-kube-prometheus-alertmanager-0 -n monitoring -- \
  amtool config routes test
```

#### 2. Notification Channels
```bash
# Test Slack notifications
curl -X POST "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK" \
  -H "Content-Type: application/json" \
  -d '{"text": "Modern Data Stack deployment test notification"}'

# Test email notifications
kubectl logs alertmanager-prometheus-kube-prometheus-alertmanager-0 -n monitoring
```

## Troubleshooting Guide

### **Common Issues**

#### 1. Pod Startup Issues
```bash
# Check pod status
kubectl get pods -n modern-data-stack

# Describe problematic pod
kubectl describe pod <pod-name> -n modern-data-stack

# Check pod logs
kubectl logs <pod-name> -n modern-data-stack --previous

# Check events
kubectl get events -n modern-data-stack --sort-by='.lastTimestamp'
```

#### 2. Resource Issues
```bash
# Check resource usage
kubectl top nodes
kubectl top pods -A

# Check resource limits
kubectl describe node <node-name>

# Check resource quotas
kubectl get resourcequota -A
```

#### 3. Network Connectivity Issues
```bash
# Test pod-to-pod connectivity
kubectl exec -it <pod-name> -n modern-data-stack -- ping <target-ip>

# Check service discovery
kubectl exec -it <pod-name> -n modern-data-stack -- nslookup postgresql.modern-data-stack.svc.cluster.local

# Check network policies
kubectl get networkpolicies -A
kubectl describe networkpolicy <policy-name> -n <namespace>
```

#### 4. Storage Issues
```bash
# Check persistent volumes
kubectl get pv
kubectl get pvc -A

# Check storage classes
kubectl get storageclass

# Check volume mounts
kubectl describe pod <pod-name> -n modern-data-stack | grep -A 5 Mounts
```

### **Recovery Procedures**

#### 1. Application Recovery
```bash
# Restart deployment
kubectl rollout restart deployment/<deployment-name> -n modern-data-stack

# Scale deployment
kubectl scale deployment/<deployment-name> --replicas=0 -n modern-data-stack
kubectl scale deployment/<deployment-name> --replicas=2 -n modern-data-stack

# Check rollout status
kubectl rollout status deployment/<deployment-name> -n modern-data-stack
```

#### 2. Database Recovery
```bash
# Check database status
kubectl exec -it deployment/postgresql -n modern-data-stack -- pg_isready

# Backup database
kubectl exec -it deployment/postgresql -n modern-data-stack -- \
  pg_dump -U postgres mlflow > mlflow_backup.sql

# Restore database
kubectl exec -i deployment/postgresql -n modern-data-stack -- \
  psql -U postgres mlflow < mlflow_backup.sql
```

## Maintenance Procedures

### **Regular Maintenance Tasks**

#### 1. Daily Tasks
```bash
# Check cluster health
kubectl get nodes
kubectl get pods -A | grep -v Running

# Check resource usage
kubectl top nodes
kubectl top pods -A

# Check for failed jobs
kubectl get jobs -A | grep -v Complete

# Backup critical data
./scripts/backup-databases.sh
```

#### 2. Weekly Tasks
```bash
# Update container images
kubectl set image deployment/mlflow mlflow=mlflow:latest -n modern-data-stack
kubectl rollout status deployment/mlflow -n modern-data-stack

# Clean up old resources
kubectl delete job --field-selector status.successful=1 -A

# Security scan
trivy cluster --report summary
```

#### 3. Monthly Tasks
```bash
# Kubernetes version upgrade
./scripts/upgrade-cluster.sh

# Certificate renewal
./scripts/renew-certificates.sh

# Disaster recovery test
./scripts/test-disaster-recovery.sh

# Performance optimization
./scripts/optimize-resources.sh
```

### **Backup and Recovery**

#### 1. Database Backup
```bash
#!/bin/bash
# Database Backup Script

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/$BACKUP_DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup PostgreSQL databases
kubectl exec -it deployment/postgresql -n modern-data-stack -- \
  pg_dump -U postgres mlflow > $BACKUP_DIR/mlflow_$BACKUP_DATE.sql

kubectl exec -it deployment/postgresql -n modern-data-stack -- \
  pg_dump -U postgres airflow > $BACKUP_DIR/airflow_$BACKUP_DATE.sql

# Upload to S3
aws s3 cp $BACKUP_DIR s3://modern-data-stack-backups/$BACKUP_DATE/ --recursive

echo "Backup completed: $BACKUP_DIR"
```

#### 2. Configuration Backup
```bash
#!/bin/bash
# Configuration Backup Script

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
CONFIG_BACKUP_DIR="/config-backups/$BACKUP_DATE"

# Create backup directory
mkdir -p $CONFIG_BACKUP_DIR

# Backup Kubernetes resources
kubectl get all -n modern-data-stack -o yaml > $CONFIG_BACKUP_DIR/modern-data-stack-resources.yaml
kubectl get configmaps -n modern-data-stack -o yaml > $CONFIG_BACKUP_DIR/configmaps.yaml
kubectl get secrets -n modern-data-stack -o yaml > $CONFIG_BACKUP_DIR/secrets.yaml

# Backup Terraform state
cp infrastructure/terraform/terraform.tfstate $CONFIG_BACKUP_DIR/

# Upload to S3
aws s3 cp $CONFIG_BACKUP_DIR s3://modern-data-stack-config-backups/$BACKUP_DATE/ --recursive

echo "Configuration backup completed: $CONFIG_BACKUP_DIR"
```

## Next Steps

After successful deployment, consider:

1. **Security Hardening**: Review and implement additional security controls
2. **Performance Optimization**: Monitor and optimize resource usage
3. **Disaster Recovery**: Test and validate disaster recovery procedures
4. **Team Training**: Train team members on platform operations
5. **Documentation**: Update documentation based on deployment experience

For detailed operational procedures, see the [Operations Runbook](./operations-runbook.md).

---

*This deployment guide provides comprehensive instructions for deploying the Modern Data Stack Showcase platform. For additional support or questions, refer to the troubleshooting section or contact the platform team.* 