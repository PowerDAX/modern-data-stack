#!/bin/bash

# Security Scan Orchestrator
# Comprehensive security scanning for Docker containers and images

set -e

# Configuration
SCAN_CONFIG_FILE="${SCAN_CONFIG_FILE:-/security/config/scan-config.json}"
REPORTS_DIR="${REPORTS_DIR:-/security/reports}"
SCANS_DIR="${SCANS_DIR:-/security/scans}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Create timestamp for this scan
SCAN_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
SCAN_ID="${SCAN_ID:-scan-${SCAN_TIMESTAMP}}"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$1] $2" | tee -a "${REPORTS_DIR}/security-scan-${SCAN_TIMESTAMP}.log"
}

# Function to scan Docker image for vulnerabilities
scan_image_vulnerabilities() {
    local image_name=$1
    local output_file="${REPORTS_DIR}/vulnerability-${SCAN_TIMESTAMP}.json"
    
    log "INFO" "Starting vulnerability scan for image: $image_name"
    
    # Run Trivy vulnerability scan
    trivy image \
        --format json \
        --output "$output_file" \
        --severity HIGH,CRITICAL \
        --exit-code 0 \
        "$image_name" || {
        log "ERROR" "Trivy scan failed for image: $image_name"
        return 1
    }
    
    # Parse results
    local critical_vulns=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "$output_file")
    local high_vulns=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH")] | length' "$output_file")
    
    log "INFO" "Vulnerability scan complete - Critical: $critical_vulns, High: $high_vulns"
    
    # Set exit code based on vulnerabilities found
    if [ "$critical_vulns" -gt 0 ]; then
        log "ERROR" "Critical vulnerabilities found in image: $image_name"
        return 2
    elif [ "$high_vulns" -gt 5 ]; then
        log "WARN" "High number of high-severity vulnerabilities found: $high_vulns"
        return 1
    fi
    
    return 0
}

# Function to scan for misconfigurations
scan_misconfigurations() {
    local dockerfile_path=$1
    local output_file="${REPORTS_DIR}/misconfig-${SCAN_TIMESTAMP}.json"
    
    log "INFO" "Starting misconfiguration scan for Dockerfile: $dockerfile_path"
    
    # Run Trivy misconfiguration scan
    trivy config \
        --format json \
        --output "$output_file" \
        --exit-code 0 \
        "$dockerfile_path" || {
        log "ERROR" "Misconfiguration scan failed for: $dockerfile_path"
        return 1
    }
    
    # Parse results
    local misconfigs=$(jq '[.Results[]?.Misconfigurations[]?] | length' "$output_file")
    
    log "INFO" "Misconfiguration scan complete - Issues found: $misconfigs"
    
    return 0
}

# Function to scan for secrets
scan_secrets() {
    local target_path=$1
    local output_file="${REPORTS_DIR}/secrets-${SCAN_TIMESTAMP}.json"
    
    log "INFO" "Starting secrets scan for: $target_path"
    
    # Run Trivy secrets scan
    trivy fs \
        --security-checks secret \
        --format json \
        --output "$output_file" \
        --exit-code 0 \
        "$target_path" || {
        log "ERROR" "Secrets scan failed for: $target_path"
        return 1
    }
    
    # Parse results
    local secrets_found=$(jq '[.Results[]?.Secrets[]?] | length' "$output_file")
    
    log "INFO" "Secrets scan complete - Secrets found: $secrets_found"
    
    if [ "$secrets_found" -gt 0 ]; then
        log "ERROR" "Secrets found in: $target_path"
        return 2
    fi
    
    return 0
}

# Function to run Docker Bench Security
run_docker_bench() {
    local output_file="${REPORTS_DIR}/docker-bench-${SCAN_TIMESTAMP}.json"
    
    log "INFO" "Starting Docker Bench Security scan"
    
    # Run Docker Bench Security
    cd /opt/docker-bench-security
    ./docker-bench-security.sh -l "$output_file" || {
        log "ERROR" "Docker Bench Security scan failed"
        return 1
    }
    
    log "INFO" "Docker Bench Security scan complete"
    
    return 0
}

# Function to run compliance checks
run_compliance_checks() {
    local compliance_standard=$1
    local output_file="${REPORTS_DIR}/compliance-${compliance_standard}-${SCAN_TIMESTAMP}.json"
    
    log "INFO" "Starting compliance check for: $compliance_standard"
    
    case "$compliance_standard" in
        "cis")
            # Run CIS Docker Benchmark
            ./docker-bench-security.sh -c cis -l "$output_file"
            ;;
        "nist")
            # Run NIST compliance checks
            log "INFO" "NIST compliance checks not yet implemented"
            ;;
        "pci-dss")
            # Run PCI-DSS compliance checks
            log "INFO" "PCI-DSS compliance checks not yet implemented"
            ;;
        *)
            log "ERROR" "Unknown compliance standard: $compliance_standard"
            return 1
            ;;
    esac
    
    log "INFO" "Compliance check complete for: $compliance_standard"
    
    return 0
}

# Function to generate security report
generate_security_report() {
    local report_file="${REPORTS_DIR}/security-report-${SCAN_TIMESTAMP}.json"
    
    log "INFO" "Generating comprehensive security report"
    
    # Create report structure
    cat > "$report_file" << EOF
{
    "scan_id": "$SCAN_ID",
    "timestamp": "$SCAN_TIMESTAMP",
    "scanner_version": "1.0.0",
    "scan_type": "comprehensive",
    "results": {
        "vulnerability_scan": null,
        "misconfiguration_scan": null,
        "secrets_scan": null,
        "docker_bench": null,
        "compliance_checks": null
    },
    "summary": {
        "total_issues": 0,
        "critical_issues": 0,
        "high_issues": 0,
        "medium_issues": 0,
        "low_issues": 0,
        "overall_score": 0
    },
    "recommendations": [],
    "artifacts": []
}
EOF

    # Merge scan results
    python3 /security/scripts/security-report.py \
        --input-dir "$REPORTS_DIR" \
        --output "$report_file" \
        --scan-id "$SCAN_ID"
    
    log "INFO" "Security report generated: $report_file"
    
    return 0
}

# Function to check scan thresholds
check_scan_thresholds() {
    local report_file="${REPORTS_DIR}/security-report-${SCAN_TIMESTAMP}.json"
    
    if [ ! -f "$report_file" ]; then
        log "ERROR" "Security report not found: $report_file"
        return 1
    fi
    
    local critical_issues=$(jq '.summary.critical_issues' "$report_file")
    local high_issues=$(jq '.summary.high_issues' "$report_file")
    local overall_score=$(jq '.summary.overall_score' "$report_file")
    
    log "INFO" "Security scan results - Critical: $critical_issues, High: $high_issues, Score: $overall_score"
    
    # Check thresholds
    if [ "$critical_issues" -gt 0 ]; then
        log "ERROR" "Critical security issues found - scan failed"
        return 2
    elif [ "$high_issues" -gt 10 ]; then
        log "WARN" "High number of security issues found - review required"
        return 1
    elif [ "$(echo "$overall_score < 70" | bc -l)" -eq 1 ]; then
        log "WARN" "Security score below threshold: $overall_score"
        return 1
    fi
    
    log "INFO" "Security scan passed all thresholds"
    return 0
}

# Function to cleanup old reports
cleanup_old_reports() {
    local retention_days=${RETENTION_DAYS:-7}
    
    log "INFO" "Cleaning up reports older than $retention_days days"
    
    find "$REPORTS_DIR" -name "*.json" -type f -mtime +$retention_days -delete
    find "$REPORTS_DIR" -name "*.log" -type f -mtime +$retention_days -delete
    
    log "INFO" "Cleanup complete"
}

# Main execution
main() {
    log "INFO" "Starting security scan - ID: $SCAN_ID"
    
    # Create directories
    mkdir -p "$REPORTS_DIR" "$SCANS_DIR"
    
    # Default scan targets
    IMAGE_NAME="${IMAGE_NAME:-modern-data-stack-showcase_mlflow:latest}"
    DOCKERFILE_PATH="${DOCKERFILE_PATH:-/security/Dockerfile}"
    SOURCE_PATH="${SOURCE_PATH:-/security/source}"
    
    local exit_code=0
    
    # Run vulnerability scan
    if scan_image_vulnerabilities "$IMAGE_NAME"; then
        log "INFO" "Vulnerability scan passed"
    else
        log "ERROR" "Vulnerability scan failed"
        exit_code=1
    fi
    
    # Run misconfiguration scan
    if [ -f "$DOCKERFILE_PATH" ]; then
        if scan_misconfigurations "$DOCKERFILE_PATH"; then
            log "INFO" "Misconfiguration scan passed"
        else
            log "ERROR" "Misconfiguration scan failed"
            exit_code=1
        fi
    fi
    
    # Run secrets scan
    if [ -d "$SOURCE_PATH" ]; then
        if scan_secrets "$SOURCE_PATH"; then
            log "INFO" "Secrets scan passed"
        else
            log "ERROR" "Secrets scan failed"
            exit_code=1
        fi
    fi
    
    # Run Docker Bench Security
    if command -v docker >/dev/null 2>&1; then
        if run_docker_bench; then
            log "INFO" "Docker Bench Security passed"
        else
            log "ERROR" "Docker Bench Security failed"
            exit_code=1
        fi
    fi
    
    # Run compliance checks
    COMPLIANCE_STANDARDS="${COMPLIANCE_STANDARDS:-cis}"
    for standard in $COMPLIANCE_STANDARDS; do
        if run_compliance_checks "$standard"; then
            log "INFO" "Compliance check passed for: $standard"
        else
            log "ERROR" "Compliance check failed for: $standard"
            exit_code=1
        fi
    done
    
    # Generate comprehensive report
    generate_security_report
    
    # Check thresholds
    if check_scan_thresholds; then
        log "INFO" "Security scan completed successfully"
    else
        log "ERROR" "Security scan failed threshold checks"
        exit_code=2
    fi
    
    # Cleanup old reports
    cleanup_old_reports
    
    log "INFO" "Security scan completed - Exit code: $exit_code"
    
    exit $exit_code
}

# Run main function
main "$@" 