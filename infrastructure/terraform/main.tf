# ===================================================================================
# MODERN DATA STACK INFRASTRUCTURE AS CODE
# ===================================================================================
# Main Terraform configuration for deploying the complete Modern Data Stack
# Supports AWS, Azure, and GCP with modular architecture
# ===================================================================================

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  backend "s3" {
    bucket         = var.terraform_state_bucket
    key            = "modern-data-stack/terraform.tfstate"
    region         = var.aws_region
    encrypt        = true
    dynamodb_table = var.terraform_lock_table
  }
}

# ===================================================================================
# PROVIDER CONFIGURATIONS
# ===================================================================================

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = local.common_tags
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.azure_subscription_id
  tenant_id       = var.azure_tenant_id
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# ===================================================================================
# LOCAL VALUES
# ===================================================================================

locals {
  common_tags = {
    Project     = "modern-data-stack-showcase"
    Environment = var.environment
    ManagedBy   = "terraform"
    Owner       = var.owner
    CostCenter  = var.cost_center
    Backup      = "daily"
    CreatedAt   = formatdate("YYYY-MM-DD", timestamp())
  }
  
  cluster_name = "${var.project_name}-${var.environment}-cluster"
  
  # Network configuration
  vpc_cidr = var.vpc_cidr
  availability_zones = slice(data.aws_availability_zones.available.names, 0, 3)
  
  # Subnet CIDR blocks
  private_subnet_cidrs = [
    cidrsubnet(local.vpc_cidr, 8, 1),
    cidrsubnet(local.vpc_cidr, 8, 2),
    cidrsubnet(local.vpc_cidr, 8, 3)
  ]
  
  public_subnet_cidrs = [
    cidrsubnet(local.vpc_cidr, 8, 101),
    cidrsubnet(local.vpc_cidr, 8, 102),
    cidrsubnet(local.vpc_cidr, 8, 103)
  ]
  
  database_subnet_cidrs = [
    cidrsubnet(local.vpc_cidr, 8, 201),
    cidrsubnet(local.vpc_cidr, 8, 202),
    cidrsubnet(local.vpc_cidr, 8,  203)
  ]
}

# ===================================================================================
# DATA SOURCES
# ===================================================================================

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

# ===================================================================================
# NETWORK INFRASTRUCTURE
# ===================================================================================

module "vpc" {
  source = "./modules/vpc"
  
  project_name           = var.project_name
  environment           = var.environment
  vpc_cidr              = local.vpc_cidr
  availability_zones    = local.availability_zones
  private_subnet_cidrs  = local.private_subnet_cidrs
  public_subnet_cidrs   = local.public_subnet_cidrs
  database_subnet_cidrs = local.database_subnet_cidrs
  
  enable_nat_gateway = true
  enable_vpn_gateway = var.enable_vpn_gateway
  enable_dns_hostnames = true
  enable_dns_support = true
  
  tags = local.common_tags
}

# ===================================================================================
# SECURITY INFRASTRUCTURE
# ===================================================================================

module "security" {
  source = "./modules/security"
  
  project_name = var.project_name
  environment  = var.environment
  vpc_id       = module.vpc.vpc_id
  
  # WAF Configuration
  enable_waf = var.enable_waf
  waf_rate_limit = var.waf_rate_limit
  
  # Security Groups
  allowed_cidr_blocks = var.allowed_cidr_blocks
  
  # KMS Configuration
  kms_key_rotation_enabled = true
  kms_key_deletion_window = var.kms_key_deletion_window
  
  tags = local.common_tags
}

# ===================================================================================
# CONTAINER INFRASTRUCTURE
# ===================================================================================

module "eks" {
  source = "./modules/eks"
  
  project_name    = var.project_name
  environment     = var.environment
  cluster_version = var.kubernetes_version
  
  # Networking
  vpc_id                    = module.vpc.vpc_id
  private_subnet_ids        = module.vpc.private_subnet_ids
  public_subnet_ids         = module.vpc.public_subnet_ids
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access = var.cluster_endpoint_public_access
  
  # Node Groups
  node_groups = {
    main = {
      instance_types = var.eks_node_instance_types
      capacity_type  = "SPOT"
      min_size       = var.eks_node_group_min_size
      max_size       = var.eks_node_group_max_size
      desired_size   = var.eks_node_group_desired_size
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "main"
      }
    }
    
    data_processing = {
      instance_types = var.eks_data_processing_instance_types
      capacity_type  = "ON_DEMAND"
      min_size       = 1
      max_size       = 10
      desired_size   = 2
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "data-processing"
        WorkloadType = "data-intensive"
      }
      
      taints = [
        {
          key    = "data-processing"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
  
  # Add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  # Security
  cluster_security_group_id = module.security.eks_cluster_security_group_id
  node_security_group_id    = module.security.eks_node_security_group_id
  
  # Logging
  cluster_enabled_log_types = [
    "api",
    "audit",
    "authenticator",
    "controllerManager",
    "scheduler"
  ]
  
  tags = local.common_tags
}

# ===================================================================================
# DATABASE INFRASTRUCTURE
# ===================================================================================

module "database" {
  source = "./modules/database"
  
  project_name = var.project_name
  environment  = var.environment
  
  # Network Configuration
  vpc_id                = module.vpc.vpc_id
  database_subnet_ids   = module.vpc.database_subnet_ids
  database_security_group_id = module.security.database_security_group_id
  
  # RDS Configuration
  rds_engine            = var.rds_engine
  rds_engine_version    = var.rds_engine_version
  rds_instance_class    = var.rds_instance_class
  rds_allocated_storage = var.rds_allocated_storage
  rds_storage_encrypted = true
  rds_kms_key_id       = module.security.rds_kms_key_id
  
  # Backup and Maintenance
  rds_backup_retention_period = var.rds_backup_retention_period
  rds_backup_window          = var.rds_backup_window
  rds_maintenance_window     = var.rds_maintenance_window
  
  # Monitoring
  rds_monitoring_interval = var.rds_monitoring_interval
  rds_performance_insights_enabled = true
  
  # Redis Configuration
  redis_node_type            = var.redis_node_type
  redis_num_cache_clusters   = var.redis_num_cache_clusters
  redis_parameter_group_name = var.redis_parameter_group_name
  redis_engine_version       = var.redis_engine_version
  
  tags = local.common_tags
}

# ===================================================================================
# STORAGE INFRASTRUCTURE
# ===================================================================================

module "storage" {
  source = "./modules/storage"
  
  project_name = var.project_name
  environment  = var.environment
  
  # S3 Buckets
  create_data_lake_bucket = var.create_data_lake_bucket
  create_backup_bucket    = var.create_backup_bucket
  create_logs_bucket      = var.create_logs_bucket
  
  # Encryption
  kms_key_id = module.security.s3_kms_key_id
  
  # Lifecycle Management
  enable_lifecycle_rules = var.enable_lifecycle_rules
  
  # Cross-region replication
  enable_cross_region_replication = var.enable_cross_region_replication
  replication_region             = var.replication_region
  
  tags = local.common_tags
}

# ===================================================================================
# MONITORING INFRASTRUCTURE
# ===================================================================================

module "monitoring" {
  source = "./modules/monitoring"
  
  project_name = var.project_name
  environment  = var.environment
  
  # CloudWatch Configuration
  create_cloudwatch_dashboard = var.create_cloudwatch_dashboard
  log_retention_days          = var.log_retention_days
  
  # Prometheus Configuration
  create_prometheus_workspace = var.create_prometheus_workspace
  prometheus_retention_days   = var.prometheus_retention_days
  
  # Grafana Configuration
  create_grafana_workspace = var.create_grafana_workspace
  grafana_authentication_providers = var.grafana_authentication_providers
  
  # Alerting
  sns_topic_arn = module.alerting.sns_topic_arn
  
  tags = local.common_tags
}

# ===================================================================================
# ALERTING INFRASTRUCTURE
# ===================================================================================

module "alerting" {
  source = "./modules/alerting"
  
  project_name = var.project_name
  environment  = var.environment
  
  # SNS Configuration
  alert_email_endpoints = var.alert_email_endpoints
  alert_sms_endpoints   = var.alert_sms_endpoints
  
  # Slack Integration
  slack_webhook_url = var.slack_webhook_url
  slack_channel     = var.slack_channel
  
  # PagerDuty Integration
  pagerduty_integration_key = var.pagerduty_integration_key
  
  tags = local.common_tags
}

# ===================================================================================
# APPLICATION INFRASTRUCTURE
# ===================================================================================

module "applications" {
  source = "./modules/applications"
  
  project_name = var.project_name
  environment  = var.environment
  
  # Kubernetes Configuration
  cluster_name               = module.eks.cluster_name
  cluster_endpoint          = module.eks.cluster_endpoint
  cluster_certificate_authority_data = module.eks.cluster_certificate_authority_data
  
  # Database Configuration
  database_endpoint = module.database.rds_endpoint
  database_port     = module.database.rds_port
  redis_endpoint    = module.database.redis_endpoint
  
  # Storage Configuration
  data_lake_bucket = module.storage.data_lake_bucket
  backup_bucket    = module.storage.backup_bucket
  
  # Monitoring Configuration
  prometheus_endpoint = module.monitoring.prometheus_endpoint
  grafana_endpoint    = module.monitoring.grafana_endpoint
  
  # Security Configuration
  secrets_manager_arn = module.security.secrets_manager_arn
  
  # Applications to Deploy
  deploy_airflow           = var.deploy_airflow
  deploy_mlflow           = var.deploy_mlflow
  deploy_great_expectations = var.deploy_great_expectations
  deploy_superset         = var.deploy_superset
  deploy_jupyter          = var.deploy_jupyter
  
  tags = local.common_tags
}

# ===================================================================================
# OUTPUTS
# ===================================================================================

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.vpc.vpc_id
}

output "private_subnet_ids" {
  description = "List of private subnet IDs"
  value       = module.vpc.private_subnet_ids
}

output "public_subnet_ids" {
  description = "List of public subnet IDs"
  value       = module.vpc.public_subnet_ids
}

output "eks_cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = module.database.rds_endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = module.database.redis_endpoint
  sensitive   = true
}

output "data_lake_bucket" {
  description = "Name of the data lake S3 bucket"
  value       = module.storage.data_lake_bucket
}

output "backup_bucket" {
  description = "Name of the backup S3 bucket"
  value       = module.storage.backup_bucket
}

output "monitoring_dashboard_url" {
  description = "URL for the monitoring dashboard"
  value       = module.monitoring.dashboard_url
}

output "grafana_endpoint" {
  description = "Grafana workspace endpoint"
  value       = module.monitoring.grafana_endpoint
}

output "application_urls" {
  description = "URLs for deployed applications"
  value       = module.applications.application_urls
}

output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
} 