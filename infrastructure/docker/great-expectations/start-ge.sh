#!/bin/bash

# Great Expectations Startup Script
# Production-ready startup with comprehensive data validation setup

set -e

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a /opt/great_expectations/logs/startup.log
}

# Function to wait for database
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

# Function to initialize Great Expectations
init_great_expectations() {
    log "Initializing Great Expectations context..."
    
    # Create data context if it doesn't exist
    if [ ! -f "/opt/great_expectations/great_expectations.yml" ]; then
        log "Creating new Great Expectations context..."
        cd /opt/great_expectations
        great_expectations init
    fi
    
    # Validate configuration
    great_expectations --v3-api project check-config || {
        log "ERROR: Configuration validation failed"
        exit 1
    }
    
    log "Great Expectations context initialized successfully"
}

# Function to create sample expectations
create_sample_expectations() {
    log "Creating sample expectations suites..."
    
    # Create expectations directory if it doesn't exist
    mkdir -p /opt/great_expectations/expectations
    
    # Create sample retail sales expectations
    cat > /opt/great_expectations/expectations/retail_sales_expectations.json << 'EOF'
{
  "data_asset_type": "Dataset",
  "expectation_suite_name": "retail_sales_expectations",
  "expectations": [
    {
      "expectation_type": "expect_table_row_count_to_be_between",
      "kwargs": {
        "min_value": 1,
        "max_value": null
      }
    },
    {
      "expectation_type": "expect_table_columns_to_match_ordered_list",
      "kwargs": {
        "column_list": ["transaction_id", "product_id", "store_id", "customer_id", "transaction_date", "quantity", "unit_price", "total_amount"]
      }
    },
    {
      "expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {
        "column": "transaction_id"
      }
    },
    {
      "expectation_type": "expect_column_values_to_be_unique",
      "kwargs": {
        "column": "transaction_id"
      }
    },
    {
      "expectation_type": "expect_column_values_to_be_between",
      "kwargs": {
        "column": "quantity",
        "min_value": 1,
        "max_value": 1000
      }
    },
    {
      "expectation_type": "expect_column_values_to_be_between",
      "kwargs": {
        "column": "unit_price",
        "min_value": 0.01,
        "max_value": 10000
      }
    },
    {
      "expectation_type": "expect_column_values_to_match_regex",
      "kwargs": {
        "column": "transaction_date",
        "regex": "^\\d{4}-\\d{2}-\\d{2}$"
      }
    }
  ],
  "ge_cloud_id": null,
  "meta": {
    "great_expectations_version": "0.18.3"
  }
}
EOF

    # Create sample data quality expectations
    cat > /opt/great_expectations/expectations/data_quality_expectations.json << 'EOF'
{
  "data_asset_type": "Dataset",
  "expectation_suite_name": "data_quality_expectations",
  "expectations": [
    {
      "expectation_type": "expect_column_values_to_not_be_null",
      "kwargs": {
        "column": "id",
        "mostly": 1.0
      }
    },
    {
      "expectation_type": "expect_column_values_to_be_unique",
      "kwargs": {
        "column": "id"
      }
    },
    {
      "expectation_type": "expect_column_proportion_of_unique_values_to_be_between",
      "kwargs": {
        "column": "id",
        "min_value": 0.99,
        "max_value": 1.0
      }
    },
    {
      "expectation_type": "expect_column_mean_to_be_between",
      "kwargs": {
        "column": "value",
        "min_value": 0,
        "max_value": 1000000
      }
    },
    {
      "expectation_type": "expect_column_stdev_to_be_between",
      "kwargs": {
        "column": "value",
        "min_value": 0,
        "max_value": 100000
      }
    }
  ],
  "ge_cloud_id": null,
  "meta": {
    "great_expectations_version": "0.18.3"
  }
}
EOF

    log "Sample expectations created successfully"
}

# Function to create sample checkpoints
create_sample_checkpoints() {
    log "Creating sample checkpoints..."
    
    # Create checkpoints directory if it doesn't exist
    mkdir -p /opt/great_expectations/checkpoints
    
    # Create sample retail checkpoint
    cat > /opt/great_expectations/checkpoints/retail_data_checkpoint.yml << 'EOF'
name: retail_data_checkpoint
config_version: 1.0
class_name: SimpleCheckpoint
run_name_template: "%Y%m%d-%H%M%S-retail-data-validation"
validations:
  - batch_request:
      datasource_name: postgres_datasource
      data_connector_name: default_inferred_data_connector
      data_asset_name: retail_sales
    expectation_suite_name: retail_sales_expectations
action_list:
  - name: store_validation_result
    action:
      class_name: StoreValidationResultAction
  - name: store_evaluation_params
    action:
      class_name: StoreEvaluationParametersAction
  - name: update_data_docs
    action:
      class_name: UpdateDataDocsAction
      site_names:
        - local_site
evaluation_parameters: {}
runtime_configuration: {}
EOF

    log "Sample checkpoints created successfully"
}

# Function to generate data docs
generate_data_docs() {
    log "Generating data documentation..."
    
    # Build data docs
    great_expectations --v3-api docs build || {
        log "WARNING: Data docs generation failed, continuing..."
    }
    
    log "Data documentation generated successfully"
}

# Function to start data docs server
start_data_docs_server() {
    log "Starting data docs server..."
    
    # Start simple HTTP server for data docs
    cd /opt/great_expectations/data_docs
    python -m http.server 8082 &
    DATA_DOCS_PID=$!
    
    log "Data docs server started on port 8082 (PID: $DATA_DOCS_PID)"
    echo $DATA_DOCS_PID > /opt/great_expectations/data_docs_server.pid
}

# Function to run sample validation
run_sample_validation() {
    log "Running sample data validation..."
    
    # Run checkpoint if it exists
    if [ -f "/opt/great_expectations/checkpoints/retail_data_checkpoint.yml" ]; then
        great_expectations --v3-api checkpoint run retail_data_checkpoint || {
            log "WARNING: Sample validation failed, continuing..."
        }
    fi
    
    log "Sample validation completed"
}

# Function to setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Create monitoring directory
    mkdir -p /opt/great_expectations/monitoring
    
    # Create metrics collection script
    cat > /opt/great_expectations/monitoring/collect_metrics.py << 'EOF'
#!/usr/bin/env python3
"""
Great Expectations Metrics Collection
"""

import os
import json
import time
from datetime import datetime
from great_expectations.data_context import DataContext

def collect_metrics():
    """Collect Great Expectations metrics"""
    try:
        context = DataContext("/opt/great_expectations")
        
        # Get validation results
        validation_results = context.get_validation_results()
        
        # Calculate metrics
        total_validations = len(validation_results)
        successful_validations = sum(1 for result in validation_results if result.success)
        failed_validations = total_validations - successful_validations
        
        # Get expectation suites
        expectation_suites = context.list_expectation_suites()
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "failed_validations": failed_validations,
            "success_rate": successful_validations / total_validations if total_validations > 0 else 0,
            "total_expectation_suites": len(expectation_suites),
            "expectation_suites": expectation_suites
        }
        
        # Write metrics to file
        with open("/opt/great_expectations/monitoring/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics collected successfully: {metrics}")
        
    except Exception as e:
        print(f"Error collecting metrics: {e}")

if __name__ == "__main__":
    collect_metrics()
EOF

    chmod +x /opt/great_expectations/monitoring/collect_metrics.py
    
    log "Monitoring setup completed"
}

# Function to handle shutdown
shutdown_handler() {
    log "Received shutdown signal, gracefully shutting down..."
    
    # Stop data docs server
    if [ -f "/opt/great_expectations/data_docs_server.pid" ]; then
        DATA_DOCS_PID=$(cat /opt/great_expectations/data_docs_server.pid)
        kill -TERM $DATA_DOCS_PID 2>/dev/null || true
        rm -f /opt/great_expectations/data_docs_server.pid
    fi
    
    log "Shutdown completed"
    exit 0
}

# Function to validate configuration
validate_config() {
    log "Validating Great Expectations configuration..."
    
    # Check if configuration file exists
    if [ ! -f "/opt/great_expectations/great_expectations.yml" ]; then
        log "ERROR: Configuration file not found"
        exit 1
    fi
    
    # Validate data context
    great_expectations --v3-api project check-config || {
        log "ERROR: Configuration validation failed"
        exit 1
    }
    
    log "Configuration validation completed"
}

# Function to setup logging
setup_logging() {
    log "Setting up logging configuration..."
    
    # Create logs directory
    mkdir -p /opt/great_expectations/logs
    
    # Create log rotation configuration
    cat > /opt/great_expectations/logs/logrotate.conf << 'EOF'
/opt/great_expectations/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 ge ge
}
EOF
    
    log "Logging configuration completed"
}

# Main execution
main() {
    log "Starting Great Expectations initialization..."
    
    # Set up signal handlers
    trap shutdown_handler SIGTERM SIGINT
    
    # Initialize components
    setup_logging
    validate_config
    init_great_expectations
    
    # Wait for database if configured
    if [ -n "$POSTGRES_CONNECTION_STRING" ]; then
        DB_HOST=$(echo "$POSTGRES_CONNECTION_STRING" | sed -n 's/.*@\([^:]*\):.*/\1/p')
        DB_PORT=$(echo "$POSTGRES_CONNECTION_STRING" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
        wait_for_database "$DB_HOST" "$DB_PORT"
    fi
    
    # Setup components
    create_sample_expectations
    create_sample_checkpoints
    setup_monitoring
    
    # Generate documentation
    generate_data_docs
    
    # Start services
    start_data_docs_server
    
    # Run sample validation
    run_sample_validation
    
    log "Great Expectations initialization completed successfully"
    log "Data docs available at http://localhost:8082"
    
    # Keep container running
    while true; do
        # Collect metrics every 5 minutes
        python /opt/great_expectations/monitoring/collect_metrics.py
        sleep 300
    done
}

# Run main function
main "$@" 