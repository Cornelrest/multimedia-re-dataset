#!/bin/bash

# Requirements Engineering Dataset - Deployment Script
# ===================================================
# Easy deployment and setup script for all platforms
#
# Author: Cornelius Chimuanya Okechukwu
# Institution: Tomas Bata University in Zlin

set -e  # Exit on error

# ============================================================================
# Configuration and Variables
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="Requirements Engineering Dataset"
VERSION="1.0.0"
PYTHON_MIN_VERSION="3.8"
REQUIRED_SPACE_MB=1000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ============================================================================
# Utility Functions
# ============================================================================

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

header() {
    echo -e "\n${PURPLE}$1${NC}"
    echo "$(printf '=%.0s' {1..50})"
}

# ============================================================================
# System Requirements Check
# ============================================================================

check_system_requirements() {
    header "Checking System Requirements"
    
    # Check Python version
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            success "Python $PYTHON_VERSION found"
        else
            error "Python 3.8+ required, found $PYTHON_VERSION"
        fi
    else
        error "Python 3 not found. Please install Python 3.8 or higher."
    fi
    
    # Check pip
    if command -v pip3 >/dev/null 2>&1; then
        success "pip3 found"
    else
        error "pip3 not found. Please install pip."
    fi
    
    # Check git (optional)
    if command -v git >/dev/null 2>&1; then
        success "Git found"
    else
        warning "Git not found. Some features may be limited."
    fi
    
    # Check available disk space
    if command -v df >/dev/null 2>&1; then
        AVAILABLE_MB=$(df . | tail -1 | awk '{print int($4/1024)}')
        if [ "$AVAILABLE_MB" -ge "$REQUIRED_SPACE_MB" ]; then
            success "Sufficient disk space: ${AVAILABLE_MB}MB available"
        else
            warning "Low disk space: ${AVAILABLE_MB}MB available (${REQUIRED_SPACE_MB}MB recommended)"
        fi
    fi
    
    # Check Docker (optional)
    if command -v docker >/dev/null 2>&1; then
        success "Docker found"
        DOCKER_AVAILABLE=true
    else
        info "Docker not found. Docker features will be unavailable."
        DOCKER_AVAILABLE=false
    fi
}

# ============================================================================
# Installation Functions
# ============================================================================

setup_virtual_environment() {
    header "Setting up Virtual Environment"
    
    if [ -d "venv" ]; then
        warning "Virtual environment already exists"
        read -p "Remove existing environment and create new one? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
        else
            return 0
        fi
    fi
    
    log "Creating virtual environment..."
    python3 -m venv venv
    
    log "Activating virtual environment..."
    source venv/bin/activate || source venv/Scripts/activate  # Windows compatibility
    
    log "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    success "Virtual environment created and activated"
}

install_dependencies() {
    header "Installing Dependencies"
    
    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi
    
    log "Installing Python dependencies..."
    pip install -r requirements.txt
    
    log "Installing package in development mode..."
    pip install -e .
    
    success "Dependencies installed successfully"
}

setup_development_environment() {
    header "Setting up Development Environment"
    
    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi
    
    log "Installing development dependencies..."
    pip install pytest pytest-cov black flake8 mypy pre-commit
    
    if command -v git >/dev/null 2>&1 && [ -d ".git" ]; then
        log "Setting up pre-commit hooks..."
        pre-commit install
        success "Pre-commit hooks installed"
    fi
    
    success "Development environment ready"
}

# ============================================================================
# Dataset Operations
# ============================================================================

generate_dataset() {
    header "Generating Dataset"
    
    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi
    
    log "Generating complete dataset..."
    python dataset_generator.py
    
    if [ $? -eq 0 ]; then
        success "Dataset generated successfully"
        info "Generated files in: generated_data/"
    else
        error "Dataset generation failed"
    fi
}

validate_dataset() {
    header "Validating Dataset"
    
    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi
    
    if [ ! -d "generated_data" ]; then
        error "No dataset found. Run generation first."
    fi
    
    log "Validating dataset..."
    python validate_dataset.py --data-dir generated_data
    
    if [ $? -eq 0 ]; then
        success "Dataset validation passed"
    else
        error "Dataset validation failed"
    fi
}

run_analysis() {
    header "Running Analysis"
    
    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi
    
    if [ ! -d "generated_data" ]; then
        error "No dataset found. Run generation first."
    fi
    
    log "Running comprehensive analysis..."
    python example_analysis.py
    
    if [ $? -eq 0 ]; then
        success "Analysis completed successfully"
        info "Results available in: analysis_output/"
    else
        error "Analysis failed"
    fi
}

run_tests() {
    header "Running Tests"
    
    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi
    
    log "Running test suite..."
    python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term
    
    if [ $? -eq 0 ]; then
        success "All tests passed"
        info "Coverage report: htmlcov/index.html"
    else
        error "Some tests failed"
    fi
}

# ============================================================================
# Docker Operations
# ============================================================================

setup_docker() {
    header "Setting up Docker Environment"
    
    if [ "$DOCKER_AVAILABLE" != true ]; then
        error "Docker not available. Please install Docker first."
    fi
    
    log "Building Docker image..."
    docker build -t requirements-engineering-dataset:latest .
    
    log "Creating Docker network..."
    docker network create requirements-engineering 2>/dev/null || true
    
    log "Creating data directories..."
    mkdir -p data_output analysis_output validation_output logs
    
    success "Docker environment ready"
}

run_docker_pipeline() {
    header "Running Docker Pipeline"
    
    if [ "$DOCKER_AVAILABLE" != true ]; then
        error "Docker not available"
    fi
    
    log "Starting dataset generation service..."
    docker-compose up dataset-generator
    
    log "Starting validation service..."
    docker-compose up dataset-validator
    
    log "Starting analysis service..."
    docker-compose up dataset-analyzer
    
    success "Docker pipeline completed"
    info "Results available in: data_output/, analysis_output/"
}

start_jupyter() {
    header "Starting Jupyter Environment"
    
    if [ "$DOCKER_AVAILABLE" = true ]; then
        log "Starting Jupyter with Docker..."
        docker-compose up -d jupyter
        info "Jupyter available at: http://localhost:8888"
        info "Token: requirements-engineering"
    else
        # Activate virtual environment
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
        elif [ -f "venv/Scripts/activate" ]; then
            source venv/Scripts/activate
        fi
        
        log "Installing Jupyter..."
        pip install jupyter jupyterlab
        
        log "Starting Jupyter Lab..."
        jupyter lab --port=8888 --no-browser --allow-root
    fi
}

# ============================================================================
# Cleanup Functions
# ============================================================================

cleanup() {
    header "Cleaning Up"
    
    log "Removing generated files..."
    rm -rf generated_data analysis_output validation_output
    rm -rf __pycache__ .pytest_cache .mypy_cache
    rm -rf build dist *.egg-info
    rm -rf htmlcov .coverage
    
    if [ "$DOCKER_AVAILABLE" = true ]; then
        log "Stopping Docker services..."
        docker-compose down 2>/dev/null || true
        
        log "Removing Docker images..."
        docker rmi requirements-engineering-dataset:latest 2>/dev/null || true
    fi
    
    success "Cleanup completed"
}

reset_environment() {
    header "Resetting Environment"
    
    cleanup
    
    log "Removing virtual environment..."
    rm -rf venv
    
    success "Environment reset completed"
}

# ============================================================================
# Help and Information
# ============================================================================

show_help() {
    echo -e "${PURPLE}${PROJECT_NAME} v${VERSION}${NC}"
    echo -e "${PURPLE}Deployment and Management Script${NC}"
    echo
    echo "Usage: $0 [OPTION]"
    echo
    echo "Setup Options:"
    echo "  setup           Complete setup (recommended for first-time users)"
    echo "  setup-dev       Setup development environment"
    echo "  install         Install dependencies only"
    echo "  check           Check system requirements"
    echo
    echo "Dataset Operations:"
    echo "  generate        Generate the complete dataset"
    echo "  validate        Validate generated dataset"
    echo "  analyze         Run comprehensive analysis"
    echo "  pipeline        Run complete pipeline (generate + validate + analyze)"
    echo
    echo "Development:"
    echo "  test            Run test suite"
    echo "  lint            Run code linting"
    echo "  format          Format code with black"
    echo "  docs            Build documentation"
    echo
    echo "Docker Operations:"
    echo "  docker-setup    Setup Docker environment"
    echo "  docker-run      Run complete pipeline with Docker"
    echo "  jupyter         Start Jupyter environment"
    echo
    echo "Maintenance:"
    echo "  cleanup         Clean generated files and caches"
    echo "  reset           Reset entire environment"
    echo "  update          Update dependencies"
    echo
    echo "Information:"
    echo "  help            Show this help message"
    echo "  version         Show version information"
    echo "  status          Show current status"
    echo
    echo "Examples:"
    echo "  $0 setup         # First-time setup"
    echo "  $0 pipeline      # Generate and analyze dataset"
    echo "  $0 jupyter       # Start Jupyter for interactive analysis"
    echo
}

show_version() {
    echo -e "${PURPLE}${PROJECT_NAME}${NC}"
    echo "Version: $VERSION"
    echo "Author: Cornelius Chimuanya Okechukwu"
    echo "Institution: Tomas Bata University in Zlin"
    echo "License: MIT with Academic Attribution"
    echo "Repository: https://github.com/multimedia-re-study/dataset"
}

show_status() {
    header "Current Status"
    
    # Python environment
    if [ -d "venv" ]; then
        success "Virtual environment: Present"
    else
        warning "Virtual environment: Missing"
    fi
    
    # Generated data
    if [ -d "generated_data" ]; then
        FILE_COUNT=$(find generated_data -name "*.csv" | wc -l)
        success "Generated dataset: $FILE_COUNT CSV files"
    else
        warning "Generated dataset: Not found"
    fi
    
    # Analysis results
    if [ -d "analysis_output" ]; then
        PLOT_COUNT=$(find analysis_output -name "*.png" | wc -l)
        success "Analysis results: $PLOT_COUNT visualizations"
    else
        warning "Analysis results: Not found"
    fi
    
    # Docker
    if [ "$DOCKER_AVAILABLE" = true ]; then
        if docker images | grep -q "requirements-engineering-dataset"; then
            success "Docker image: Built"
        else
            warning "Docker image: Not built"
        fi
    else
        info "Docker: Not available"
    fi
    
    # Test results
    if [ -f "htmlcov/index.html" ]; then
        success "Test coverage: Available"
    else
        warning "Test coverage: Not generated"
    fi
}

# ============================================================================
# Interactive Setup
# ============================================================================

interactive_setup() {
    header "Interactive Setup"
    echo "Welcome to the $PROJECT_NAME setup!"
    echo
    
    echo "This script will:"
    echo "1. Check system requirements"
    echo "2. Create a virtual environment"
    echo "3. Install all dependencies"
    echo "4. Generate the dataset"
    echo "5. Run validation and analysis"
    echo
    
    read -p "Continue with setup? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        info "Setup cancelled"
        exit 0
    fi
    
    check_system_requirements
    setup_virtual_environment
    install_dependencies
    
    read -p "Generate dataset now? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        generate_dataset
        validate_dataset
        
        read -p "Run analysis? [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            run_analysis
        fi
    fi
    
    success "Setup completed successfully!"
    echo
    info "Next steps:"
    echo "  - Review generated data in: generated_data/"
    echo "  - Check analysis results in: analysis_output/"
    echo "  - Run '$0 jupyter' to start interactive analysis"
    echo "  - Run '$0 help' to see all available commands"
}

# ============================================================================
# Main Script Logic
# ============================================================================

main() {
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Check if running in CI environment
    if [ "$CI" = "true" ]; then
        export DEBIAN_FRONTEND=noninteractive
        warning "Running in CI mode"
    fi
    
    case "${1:-help}" in
        setup)
            interactive_setup
            ;;
        setup-dev)
            check_system_requirements
            setup_virtual_environment
            install_dependencies
            setup_development_environment
            ;;
        install)
            setup_virtual_environment
            install_dependencies
            ;;
        check)
            check_system_requirements
            ;;
        generate)
            generate_dataset
            ;;
        validate)
            validate_dataset
            ;;
        analyze)
            run_analysis
            ;;
        pipeline)
            generate_dataset
            validate_dataset
            run_analysis
            ;;
        test)
            run_tests
            ;;
        lint)
            # Activate virtual environment
            if [ -f "venv/bin/activate" ]; then
                source venv/bin/activate
            elif [ -f "venv/Scripts/activate" ]; then
                source venv/Scripts/activate
            fi
            flake8 .
            ;;
        format)
            # Activate virtual environment
            if [ -f "venv/bin/activate" ]; then
                source venv/bin/activate
            elif [ -f "venv/Scripts/activate" ]; then
                source venv/Scripts/activate
            fi
            black .
            ;;
        docs)
            header "Building Documentation"
            if [ -f "venv/bin/activate" ]; then
                source venv/bin/activate
            elif [ -f "venv/Scripts/activate" ]; then
                source venv/Scripts/activate
            fi
            pip install sphinx sphinx-rtd-theme
            mkdir -p docs
            sphinx-build -b html . docs/_build/html 2>/dev/null || make -C docs html
            success "Documentation built in docs/_build/html/"
            ;;
        docker-setup)
            check_system_requirements
            setup_docker
            ;;
        docker-run)
            check_system_requirements
            run_docker_pipeline
            ;;
        jupyter)
            start_jupyter
            ;;
        cleanup)
            cleanup
            ;;
        reset)
            reset_environment
            ;;
        update)
            header "Updating Dependencies"
            if [ -f "venv/bin/activate" ]; then
                source venv/bin/activate
            elif [ -f "venv/Scripts/activate" ]; then
                source venv/Scripts/activate
            fi
            pip install --upgrade -r requirements.txt
            success "Dependencies updated"
            ;;
        version)
            show_version
            ;;
        status)
            show_status
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown option: $1"
            echo "Run '$0 help' for available options."
            exit 1
            ;;
    esac
}

# ============================================================================
# Script Execution
# ============================================================================

# Ensure script is not sourced
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
else
    error "This script should be executed, not sourced."
fi