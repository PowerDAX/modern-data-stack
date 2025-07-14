#!/bin/bash

# Documentation Deployment Script
# Modern Data Stack Showcase - Jupyter Notebooks Documentation
# 
# This script builds and deploys the comprehensive documentation system
# including Jupyter Book, API documentation, and quality reports.
#
# Usage:
#   ./deploy-docs.sh [--mode production|development] [--output-dir OUTPUT_DIR]
#
# Author: Data Science Team
# Date: 2024-01-15

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default values
MODE="development"
OUTPUT_DIR="docs"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${BASE_DIR}/../.." && pwd)"
NOTEBOOK_DIR="${PROJECT_ROOT}/notebooks"
DOCUMENTATION_DIR="${PROJECT_ROOT}/notebooks/documentation"
BUILD_DIR="${PROJECT_ROOT}/build"
REQUIREMENTS_FILE="${DOCUMENTATION_DIR}/requirements.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo "=================================================================="
    echo -e "${BLUE}$1${NC}"
    echo "=================================================================="
}

print_separator() {
    echo "------------------------------------------------------------------"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Command '$1' is not available. Please install it first."
        exit 1
    fi
}

check_python_package() {
    if ! python -c "import $1" &> /dev/null; then
        log_error "Python package '$1' is not available. Please install it first."
        exit 1
    fi
}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--mode production|development] [--output-dir OUTPUT_DIR]"
            echo ""
            echo "Options:"
            echo "  --mode            Deployment mode (development|production)"
            echo "  --output-dir      Output directory for documentation"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

print_header "ENVIRONMENT VALIDATION"

log_info "Validating environment for documentation deployment..."

# Check required commands
log_info "Checking required commands..."
check_command "python"
check_command "pip"
check_command "git"

# Check Python version
PYTHON_VERSION=$(python --version | cut -d' ' -f2)
log_info "Python version: $PYTHON_VERSION"

# Check if we're in a virtual environment
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    log_info "Virtual environment: $VIRTUAL_ENV"
else
    log_warning "No virtual environment detected. Consider using one."
fi

# Check project structure
log_info "Validating project structure..."
if [[ ! -d "$NOTEBOOK_DIR" ]]; then
    log_error "Notebook directory not found: $NOTEBOOK_DIR"
    exit 1
fi

if [[ ! -d "$DOCUMENTATION_DIR" ]]; then
    log_error "Documentation directory not found: $DOCUMENTATION_DIR"
    exit 1
fi

log_success "Environment validation completed"

# =============================================================================
# DEPENDENCY INSTALLATION
# =============================================================================

print_header "DEPENDENCY INSTALLATION"

log_info "Installing/updating Python dependencies..."

if [[ -f "$REQUIREMENTS_FILE" ]]; then
    log_info "Installing from requirements file: $REQUIREMENTS_FILE"
    pip install -r "$REQUIREMENTS_FILE" --upgrade
else
    log_warning "Requirements file not found. Installing core dependencies..."
    pip install jupyter-book nbformat nbconvert pandas numpy matplotlib seaborn plotly scipy scikit-learn
fi

# Check critical packages
log_info "Validating critical packages..."
check_python_package "jupyter_book"
check_python_package "nbformat"
check_python_package "nbconvert"
check_python_package "pandas"
check_python_package "plotly"

log_success "Dependencies installed successfully"

# =============================================================================
# DOCUMENTATION ANALYSIS
# =============================================================================

print_header "DOCUMENTATION ANALYSIS"

log_info "Analyzing notebook documentation quality..."

# Run documentation analysis
cd "$DOCUMENTATION_DIR"
if [[ -f "build-docs.py" ]]; then
    log_info "Running comprehensive documentation analysis..."
    python build-docs.py --mode notebooks --output-dir "$BUILD_DIR/analysis"
    
    if [[ $? -eq 0 ]]; then
        log_success "Documentation analysis completed"
    else
        log_error "Documentation analysis failed"
        exit 1
    fi
else
    log_warning "build-docs.py not found. Skipping detailed analysis."
fi

# =============================================================================
# NOTEBOOK ENHANCEMENT
# =============================================================================

print_header "NOTEBOOK ENHANCEMENT"

log_info "Enhancing notebooks with comprehensive documentation..."

if [[ -f "enhance-notebook-docs.py" ]]; then
    log_info "Running notebook enhancement..."
    python enhance-notebook-docs.py --directory "$NOTEBOOK_DIR" --recursive --output-dir "$BUILD_DIR/enhanced"
    
    if [[ $? -eq 0 ]]; then
        log_success "Notebook enhancement completed"
    else
        log_error "Notebook enhancement failed"
        exit 1
    fi
else
    log_warning "enhance-notebook-docs.py not found. Skipping enhancement."
fi

# =============================================================================
# DOCUMENTATION TESTING
# =============================================================================

print_header "DOCUMENTATION TESTING"

log_info "Testing documentation quality and completeness..."

if [[ -f "test-documentation.py" ]]; then
    log_info "Running documentation tests..."
    python test-documentation.py --directory "$NOTEBOOK_DIR" --recursive --output "$BUILD_DIR/test-results.json"
    
    if [[ $? -eq 0 ]]; then
        log_success "Documentation testing completed"
    else
        log_warning "Documentation testing completed with issues"
    fi
else
    log_warning "test-documentation.py not found. Skipping testing."
fi

# =============================================================================
# JUPYTER BOOK BUILD
# =============================================================================

print_header "JUPYTER BOOK BUILD"

log_info "Building Jupyter Book documentation..."

# Create build directory
mkdir -p "$BUILD_DIR/jupyter-book"

# Copy configuration files
if [[ -f "jupyter-book-config.yml" ]]; then
    cp "jupyter-book-config.yml" "$BUILD_DIR/jupyter-book/_config.yml"
    log_info "Copied Jupyter Book configuration"
fi

if [[ -f "_toc.yml" ]]; then
    cp "_toc.yml" "$BUILD_DIR/jupyter-book/_toc.yml"
    log_info "Copied table of contents"
fi

# Copy notebooks
log_info "Copying notebooks to build directory..."
cp -r "$NOTEBOOK_DIR" "$BUILD_DIR/jupyter-book/notebooks"

# Create index file
log_info "Creating index file..."
cat > "$BUILD_DIR/jupyter-book/index.md" << 'EOF'
# Modern Data Stack Showcase - Jupyter Notebooks Documentation

Welcome to the comprehensive documentation for the Modern Data Stack Showcase Jupyter notebooks.

## 📊 Overview

This documentation provides detailed guidance for using and understanding the Jupyter notebooks in this project.

## 🚀 Quick Start

1. [Getting Started](getting-started/introduction.md)
2. [Environment Setup](getting-started/environment-setup.md)
3. [Documentation Standards](standards/documentation-guidelines.md)

## 📚 Sections

Navigate through the documentation using the table of contents on the left.

## 🔧 Development

This documentation is automatically generated and deployed using our comprehensive documentation system.

---

*Last updated: $(date)*
EOF

# Build the book
log_info "Building Jupyter Book..."
cd "$BUILD_DIR/jupyter-book"
if jupyter-book build . --builder html; then
    log_success "Jupyter Book build completed"
else
    log_error "Jupyter Book build failed"
    exit 1
fi

# =============================================================================
# OUTPUT PREPARATION
# =============================================================================

print_header "OUTPUT PREPARATION"

log_info "Preparing final output directory..."

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Copy built documentation
if [[ -d "$BUILD_DIR/jupyter-book/_build/html" ]]; then
    log_info "Copying built documentation to output directory..."
    cp -r "$BUILD_DIR/jupyter-book/_build/html/"* "$OUTPUT_DIR/"
    log_success "Documentation copied to $OUTPUT_DIR"
fi

# Copy analysis results
if [[ -d "$BUILD_DIR/analysis" ]]; then
    log_info "Copying analysis results..."
    mkdir -p "$OUTPUT_DIR/analysis"
    cp -r "$BUILD_DIR/analysis/"* "$OUTPUT_DIR/analysis/"
    log_success "Analysis results copied"
fi

# Copy test results
if [[ -f "$BUILD_DIR/test-results.json" ]]; then
    log_info "Copying test results..."
    cp "$BUILD_DIR/test-results.json" "$OUTPUT_DIR/"
    log_success "Test results copied"
fi

# =============================================================================
# DEPLOYMENT FINALIZATION
# =============================================================================

print_header "DEPLOYMENT FINALIZATION"

# Create deployment metadata
log_info "Creating deployment metadata..."
cat > "$OUTPUT_DIR/deployment-info.json" << EOF
{
    "deployment_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "mode": "$MODE",
    "python_version": "$PYTHON_VERSION",
    "build_directory": "$BUILD_DIR",
    "output_directory": "$OUTPUT_DIR",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
}
EOF

# Create .nojekyll file for GitHub Pages compatibility
touch "$OUTPUT_DIR/.nojekyll"
log_info "Created .nojekyll file for GitHub Pages compatibility"

# Generate deployment summary
log_info "Generating deployment summary..."
cat > "$OUTPUT_DIR/README.md" << 'EOF'
# Modern Data Stack Showcase - Documentation

This directory contains the comprehensive documentation for the Modern Data Stack Showcase Jupyter notebooks.

## 📁 Directory Structure

- `/` - Main documentation site (Jupyter Book)
- `/analysis/` - Documentation analysis results
- `/test-results.json` - Documentation quality test results
- `/deployment-info.json` - Deployment metadata

## 🚀 Viewing the Documentation

Open `index.html` in your web browser to view the documentation.

## 🔧 Regenerating Documentation

To regenerate this documentation, run:

```bash
./deploy-docs.sh --mode production
```

## 📊 Quality Metrics

Check the analysis and test results for documentation quality metrics and recommendations.

---

*This documentation is automatically generated and deployed.*
EOF

# =============================================================================
# CLEANUP
# =============================================================================

print_header "CLEANUP"

log_info "Cleaning up temporary files..."

# Remove build directory if in production mode
if [[ "$MODE" == "production" ]]; then
    log_info "Removing build directory (production mode)..."
    rm -rf "$BUILD_DIR"
    log_success "Build directory cleaned up"
else
    log_info "Keeping build directory (development mode)"
fi

# =============================================================================
# SUMMARY
# =============================================================================

print_header "DEPLOYMENT SUMMARY"

log_success "Documentation deployment completed successfully!"
echo ""
log_info "📊 Summary:"
log_info "  - Mode: $MODE"
log_info "  - Output Directory: $OUTPUT_DIR"
log_info "  - Python Version: $PYTHON_VERSION"
log_info "  - Deployment Date: $(date)"
echo ""
log_info "📁 Generated Files:"
log_info "  - Main Documentation: $OUTPUT_DIR/index.html"
log_info "  - Analysis Results: $OUTPUT_DIR/analysis/"
log_info "  - Test Results: $OUTPUT_DIR/test-results.json"
log_info "  - Deployment Info: $OUTPUT_DIR/deployment-info.json"
echo ""
log_info "🌐 To view the documentation:"
log_info "  - Open: $OUTPUT_DIR/index.html"
log_info "  - Or serve with: python -m http.server --directory $OUTPUT_DIR"
echo ""

# Open documentation in browser (development mode only)
if [[ "$MODE" == "development" ]] && command -v open &> /dev/null; then
    log_info "Opening documentation in browser..."
    open "$OUTPUT_DIR/index.html"
fi

log_success "All tasks completed! 🎉" 