# ===================================================================================
# TERRAFORM VARIABLES CONFIGURATION
# ===================================================================================
# Comprehensive variable definitions for Modern Data Stack infrastructure
# Supports multiple environments and cloud providers
# ===================================================================================

# ===================================================================================
# PROJECT CONFIGURATION
# ===================================================================================

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "modern-data-stack-showcase"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "owner" {
  description = "Owner of the infrastructure"
  type        = string
  default     = "data-team"
}

variable "cost_center" {
  description = "Cost center for billing"
  type        = string
  default     = "engineering"
}

# ===================================================================================
# TERRAFORM STATE CONFIGURATION
# ===================================================================================

variable "terraform_state_bucket" {
  description = "S3 bucket for Terraform state"
  type        = string
  default     = "modern-data-stack-terraform-state"
}

variable "terraform_lock_table" {
  description = "DynamoDB table for Terraform state locking"
  type        = string
  default     = "modern-data-stack-terraform-locks"
}

# ===================================================================================
# CLOUD PROVIDER CONFIGURATION
# ===================================================================================

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "azure_subscription_id" {
  description = "Azure subscription ID"
  type        = string
  default     = ""
}

variable "azure_tenant_id" {
  description = "Azure tenant ID"
  type        = string
  default     = ""
}

variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
  default     = ""
}

variable "gcp_region" {
  description = "GCP region for deployment"
  type        = string
  default     = "us-central1"
}

# ===================================================================================
# NETWORK CONFIGURATION
# ===================================================================================

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_vpn_gateway" {
  description = "Whether to create VPN gateway"
  type        = bool
  default     = false
}

variable "allowed_cidr_blocks" {
  description = "List of CIDR blocks allowed to access resources"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# ===================================================================================
# KUBERNETES CONFIGURATION
# ===================================================================================

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "cluster_endpoint_public_access" {
  description = "Whether the cluster endpoint should be publicly accessible"
  type        = bool
  default     = true
}

variable "eks_node_instance_types" {
  description = "EC2 instance types for EKS node groups"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "eks_data_processing_instance_types" {
  description = "EC2 instance types for data processing node groups"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge", "c5.large"]
}

variable "eks_node_group_min_size" {
  description = "Minimum number of nodes in EKS node group"
  type        = number
  default     = 1
}

variable "eks_node_group_max_size" {
  description = "Maximum number of nodes in EKS node group"
  type        = number
  default     = 5
}

variable "eks_node_group_desired_size" {
  description = "Desired number of nodes in EKS node group"
  type        = number
  default     = 2
}

# ===================================================================================
# DATABASE CONFIGURATION
# ===================================================================================

variable "rds_engine" {
  description = "Database engine for RDS"
  type        = string
  default     = "postgres"
}

variable "rds_engine_version" {
  description = "Database engine version"
  type        = string
  default     = "15.4"
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 100
}

variable "rds_backup_retention_period" {
  description = "RDS backup retention period in days"
  type        = number
  default     = 7
}

variable "rds_backup_window" {
  description = "RDS backup window"
  type        = string
  default     = "03:00-04:00"
}

variable "rds_maintenance_window" {
  description = "RDS maintenance window"
  type        = string
  default     = "sun:04:00-sun:05:00"
}

variable "rds_monitoring_interval" {
  description = "RDS monitoring interval in seconds"
  type        = number
  default     = 60
}

variable "redis_node_type" {
  description = "Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

variable "redis_num_cache_clusters" {
  description = "Number of cache clusters for Redis"
  type        = number
  default     = 1
}

variable "redis_parameter_group_name" {
  description = "Redis parameter group name"
  type        = string
  default     = "default.redis7"
}

variable "redis_engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.0"
}

# ===================================================================================
# STORAGE CONFIGURATION
# ===================================================================================

variable "create_data_lake_bucket" {
  description = "Whether to create data lake S3 bucket"
  type        = bool
  default     = true
}

variable "create_backup_bucket" {
  description = "Whether to create backup S3 bucket"
  type        = bool
  default     = true
}

variable "create_logs_bucket" {
  description = "Whether to create logs S3 bucket"
  type        = bool
  default     = true
}

variable "enable_lifecycle_rules" {
  description = "Whether to enable S3 lifecycle rules"
  type        = bool
  default     = true
}

variable "enable_cross_region_replication" {
  description = "Whether to enable cross-region replication"
  type        = bool
  default     = false
}

variable "replication_region" {
  description = "Region for cross-region replication"
  type        = string
  default     = "us-east-1"
}

# ===================================================================================
# SECURITY CONFIGURATION
# ===================================================================================

variable "enable_waf" {
  description = "Whether to enable WAF"
  type        = bool
  default     = true
}

variable "waf_rate_limit" {
  description = "WAF rate limit per 5 minutes"
  type        = number
  default     = 10000
}

variable "kms_key_deletion_window" {
  description = "KMS key deletion window in days"
  type        = number
  default     = 7
}

# ===================================================================================
# MONITORING CONFIGURATION
# ===================================================================================

variable "create_cloudwatch_dashboard" {
  description = "Whether to create CloudWatch dashboard"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention period in days"
  type        = number
  default     = 30
}

variable "create_prometheus_workspace" {
  description = "Whether to create Amazon Managed Prometheus workspace"
  type        = bool
  default     = true
}

variable "prometheus_retention_days" {
  description = "Prometheus data retention period in days"
  type        = number
  default     = 15
}

variable "create_grafana_workspace" {
  description = "Whether to create Amazon Managed Grafana workspace"
  type        = bool
  default     = true
}

variable "grafana_authentication_providers" {
  description = "List of authentication providers for Grafana"
  type        = list(string)
  default     = ["AWS_SSO"]
}

# ===================================================================================
# ALERTING CONFIGURATION
# ===================================================================================

variable "alert_email_endpoints" {
  description = "List of email addresses for alerts"
  type        = list(string)
  default     = []
}

variable "alert_sms_endpoints" {
  description = "List of SMS numbers for alerts"
  type        = list(string)
  default     = []
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for notifications"
  type        = string
  default     = ""
  sensitive   = true
}

variable "slack_channel" {
  description = "Slack channel for notifications"
  type        = string
  default     = "#data-alerts"
}

variable "pagerduty_integration_key" {
  description = "PagerDuty integration key"
  type        = string
  default     = ""
  sensitive   = true
}

# ===================================================================================
# APPLICATION DEPLOYMENT CONFIGURATION
# ===================================================================================

variable "deploy_airflow" {
  description = "Whether to deploy Apache Airflow"
  type        = bool
  default     = true
}

variable "deploy_mlflow" {
  description = "Whether to deploy MLflow"
  type        = bool
  default     = true
}

variable "deploy_great_expectations" {
  description = "Whether to deploy Great Expectations"
  type        = bool
  default     = true
}

variable "deploy_superset" {
  description = "Whether to deploy Apache Superset"
  type        = bool
  default     = false
}

variable "deploy_jupyter" {
  description = "Whether to deploy JupyterHub"
  type        = bool
  default     = true
}

# ===================================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# ===================================================================================

variable "environment_configs" {
  description = "Environment-specific configurations"
  type = map(object({
    instance_types           = list(string)
    min_nodes               = number
    max_nodes               = number
    desired_nodes           = number
    rds_instance_class      = string
    rds_allocated_storage   = number
    enable_deletion_protection = bool
    enable_backup           = bool
    monitoring_level        = string
  }))
  
  default = {
    dev = {
      instance_types           = ["t3.medium"]
      min_nodes               = 1
      max_nodes               = 3
      desired_nodes           = 1
      rds_instance_class      = "db.t3.micro"
      rds_allocated_storage   = 20
      enable_deletion_protection = false
      enable_backup           = false
      monitoring_level        = "basic"
    }
    
    staging = {
      instance_types           = ["t3.medium", "t3.large"]
      min_nodes               = 1
      max_nodes               = 5
      desired_nodes           = 2
      rds_instance_class      = "db.t3.small"
      rds_allocated_storage   = 50
      enable_deletion_protection = false
      enable_backup           = true
      monitoring_level        = "enhanced"
    }
    
    prod = {
      instance_types           = ["t3.large", "t3.xlarge"]
      min_nodes               = 2
      max_nodes               = 10
      desired_nodes           = 3
      rds_instance_class      = "db.t3.medium"
      rds_allocated_storage   = 100
      enable_deletion_protection = true
      enable_backup           = true
      monitoring_level        = "comprehensive"
    }
  }
}

# ===================================================================================
# FEATURE FLAGS
# ===================================================================================

variable "feature_flags" {
  description = "Feature flags for optional components"
  type = object({
    enable_service_mesh         = bool
    enable_policy_as_code       = bool
    enable_chaos_engineering    = bool
    enable_cost_optimization    = bool
    enable_compliance_scanning  = bool
    enable_advanced_networking  = bool
    enable_disaster_recovery    = bool
    enable_multi_region        = bool
  })
  
  default = {
    enable_service_mesh         = false
    enable_policy_as_code       = false
    enable_chaos_engineering    = false
    enable_cost_optimization    = true
    enable_compliance_scanning  = true
    enable_advanced_networking  = false
    enable_disaster_recovery    = false
    enable_multi_region        = false
  }
}

# ===================================================================================
# ADVANCED CONFIGURATION
# ===================================================================================

variable "advanced_configs" {
  description = "Advanced configuration options"
  type = object({
    enable_spot_instances       = bool
    enable_graviton_instances   = bool
    enable_custom_ami          = bool
    custom_ami_id              = string
    enable_pod_identity        = bool
    enable_fargate_profiles    = bool
    enable_bottlerocket_nodes  = bool
    enable_gpu_nodes           = bool
    gpu_instance_types         = list(string)
  })
  
  default = {
    enable_spot_instances       = true
    enable_graviton_instances   = false
    enable_custom_ami          = false
    custom_ami_id              = ""
    enable_pod_identity        = false
    enable_fargate_profiles    = false
    enable_bottlerocket_nodes  = false
    enable_gpu_nodes           = false
    gpu_instance_types         = ["p3.2xlarge", "g4dn.xlarge"]
  }
}

# ===================================================================================
# SECURITY POLICIES
# ===================================================================================

variable "security_policies" {
  description = "Security policy configurations"
  type = object({
    enforce_pod_security_standards = bool
    enable_network_policies        = bool
    enable_falco_monitoring        = bool
    enable_admission_controllers   = bool
    enable_image_scanning          = bool
    enable_runtime_security        = bool
    enable_secret_encryption       = bool
    enable_audit_logging           = bool
  })
  
  default = {
    enforce_pod_security_standards = true
    enable_network_policies        = true
    enable_falco_monitoring        = false
    enable_admission_controllers   = true
    enable_image_scanning          = true
    enable_runtime_security        = false
    enable_secret_encryption       = true
    enable_audit_logging           = true
  }
}

# ===================================================================================
# BACKUP AND DISASTER RECOVERY
# ===================================================================================

variable "backup_config" {
  description = "Backup and disaster recovery configuration"
  type = object({
    enable_automated_backups      = bool
    backup_retention_days         = number
    enable_point_in_time_recovery = bool
    enable_cross_region_backup    = bool
    backup_schedule               = string
    enable_backup_encryption      = bool
    backup_storage_class          = string
  })
  
  default = {
    enable_automated_backups      = true
    backup_retention_days         = 30
    enable_point_in_time_recovery = true
    enable_cross_region_backup    = false
    backup_schedule               = "cron(0 2 * * ? *)"
    enable_backup_encryption      = true
    backup_storage_class          = "STANDARD_IA"
  }
} 