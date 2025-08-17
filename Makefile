# Requirements Engineering Dataset - Makefile
# ==========================================
# Automation commands for development and deployment

.PHONY: help install install-dev clean test test-coverage lint format
.PHONY: generate validate analyze build package upload
.PHONY: docs docs-serve docker-build docker-run
.PHONY: ci-check release-check all

# Default Python interpreter
PYTHON := python3
PIP := pip3

# Project directories
SRC_DIR := .
TEST_DIR := tests
DOCS_DIR := docs
BUILD_DIR := build
DIST_DIR := dist
DATA_DIR := generated_data
ANALYSIS_DIR := analysis_output

# Docker settings
DOCKER_IMAGE := requirements-engineering-dataset
DOCKER_TAG := latest

# Help target - default when just running 'make'
help:
	@echo "Requirements Engineering Dataset - Makefile Commands"
	@echo "==================================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install package and dependencies"
	@echo "  install-dev      Install with development dependencies"
	@echo "  clean            Clean build artifacts and generated files"
	@echo ""
	@echo "Development Commands:"
	@echo "  test             Run test suite"
	@echo "  test-coverage    Run tests with coverage report"
	@echo "  lint             Run code linting (flake8)"
	@echo "  format           Format code with black"
	@echo "  type-check       Run static type checking with mypy"
	@echo ""
	@echo "Dataset Commands:"
	@echo "  generate         Generate complete dataset"
	@echo "  validate         Validate generated dataset"
	@echo "  analyze          Run comprehensive analysis"
	@echo "  quick-test       Generate, validate, and analyze (quick pipeline)"
	@echo ""
	@echo "Build Commands:"
	@echo "  build            Build package distributions"
	@echo "  package          Create distribution packages"
	@echo "  upload-test      Upload to TestPyPI"
	@echo "  upload-prod      Upload to PyPI (production)"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs             Build documentation"
	@echo "  docs-serve       Serve documentation locally"
	@echo "  docs-clean       Clean documentation build"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run dataset generation in container"
	@echo "  docker-shell     Open shell in container"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  ci-check         Run all CI checks locally"
	@echo "  release-check    Verify release readiness"
	@echo "  security-check   Run security vulnerability scan"
	@echo ""
	@echo "Utility Commands:"
	@echo "  all              Run complete pipeline (generate, validate, analyze)"
	@echo "  benchmark        Run performance benchmarks"
	@echo "  profile          Profile code performance"

# ============================================================================
# Setup and Installation
# ============================================================================

install:
	@echo "📦 Installing Requirements Engineering Dataset..."
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "✅ Installation complete!"

install-dev:
	@echo "📦 Installing with development dependencies..."
	$(PIP) install -r requirements.txt
	$(PIP) install -e .[dev,docs]
	$(PIP) install pre-commit black flake8 mypy pytest pytest-cov
	pre-commit install
	@echo "✅ Development installation complete!"

clean:
	@echo "🧹 Cleaning build artifacts and generated files..."
	rm -rf $(BUILD_DIR) $(DIST_DIR) .egg-info/
	rm -rf .pytest_cache .coverage htmlcov/ .mypy_cache
	rm -rf $(DATA_DIR) $(ANALYSIS_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "✅ Cleanup complete!"

# ============================================================================
# Development and Testing
# ============================================================================

test:
	@echo "🧪 Running test suite..."
	$(PYTHON) -m pytest $(TEST_DIR)/ -v
	@echo "✅ Tests complete!"

test-coverage:
	@echo "🧪 Running tests with coverage..."
	$(PYTHON) -m pytest $(TEST_DIR)/ -v --cov=. --cov-report=html --cov-report=xml --cov-report=term
	@echo "📊 Coverage report generated in htmlcov/"
	@echo "✅ Tests with coverage complete!"

test-integration:
	@echo "🔄 Running integration tests..."
	$(PYTHON) -m pytest $(TEST_DIR)/test_integration.py -v
	@echo "✅ Integration tests complete!"

lint:
	@echo "🔍 Running code linting..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "✅ Linting complete!"

format:
	@echo "🎨 Formatting code with black..."
	black .
	@echo "✅ Code formatting complete!"

format-check:
	@echo "🎨 Checking code formatting..."
	black --check --diff .
	@echo "✅ Format check complete!"

type-check:
	@echo "🔍 Running static type checking..."
	mypy . --ignore-missing-imports
	@echo "✅ Type checking complete!"

# ============================================================================
# Dataset Operations
# ============================================================================

generate:
	@echo "📊 Generating Requirements Engineering Dataset..."
	$(PYTHON) dataset_generator.py
	@echo "✅ Dataset generation complete!"

validate:
	@echo "🔍 Validating generated dataset..."
	$(PYTHON) validate_dataset.py --data-dir $(DATA_DIR)
	@echo "✅ Dataset validation complete!"

analyze:
	@echo "📈 Running comprehensive analysis..."
	$(PYTHON) example_analysis.py
	@echo "✅ Analysis complete!"

quick-test: generate validate
	@echo "⚡ Quick pipeline test complete!"

# ============================================================================
# Build and Distribution
# ============================================================================

build: clean
	@echo "🔨 Building package distributions..."
	$(PYTHON) -m build
	@echo "✅ Build complete!"

package: clean test build
	@echo "📦 Creating distribution packages..."
	$(PYTHON) setup.py sdist bdist_wheel
	twine check dist/*
	@echo "✅ Package creation complete!"

upload-test: package
	@echo "🚀 Uploading to TestPyPI..."
	twine upload --repository testpypi dist/*
	@echo "✅ Upload to TestPyPI complete!"

upload-prod: package
	@echo "🚀 Uploading to PyPI..."
	@read -p "Are you sure you want to upload to production PyPI? [y/N] " confirm && [ "$$confirm" = "y" ]
	twine upload dist/*
	@echo "✅ Upload to PyPI complete!"

# ============================================================================
# Documentation
# ============================================================================

docs:
	@echo "📚 Building documentation..."
	mkdir -p $(DOCS_DIR)
	@if [ ! -f "$(DOCS_DIR)/conf.py" ]; then \
		echo "Setting up Sphinx documentation..."; \
		sphinx-quickstart -q -p "Requirements Engineering Dataset" \
			-a "Cornelius Chimuanya Okechukwu" \
			-v "1.0.0" --ext-autodoc --ext-viewcode \
			--makefile --no-batchfile $(DOCS_DIR); \
	fi
	cd $(DOCS_DIR) && make html
	@echo "✅ Documentation built in $(DOCS_DIR)/_build/html/"

docs-serve: docs
	@echo "🌐 Serving documentation locally..."
	@echo "Open http://localhost:8000 in your browser"
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

docs-clean:
	@echo "🧹 Cleaning documentation..."
	cd $(DOCS_DIR) && make clean
	@echo "✅ Documentation cleanup complete!"

# ============================================================================
# Docker Commands
# ============================================================================

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "✅ Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)"

docker-run: docker-build
	@echo "🐳 Running dataset generation in Docker..."
	docker run --rm -v $$(pwd)/docker_output:/app/generated_data $(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "✅ Docker run complete! Check docker_output/ directory"

docker-shell: docker-build
	@echo "🐳 Opening shell in Docker container..."
	docker run --rm -it -v $$(pwd):/app/workspace $(DOCKER_IMAGE):$(DOCKER_TAG) /bin/bash

docker-clean:
	@echo "🐳 Cleaning Docker images..."
	docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG) 2>/dev/null || true
	docker system prune -f
	@echo "✅ Docker cleanup complete!"

# ============================================================================
# Quality Assurance
# ============================================================================

ci-check: format-check lint type-check test test-integration
	@echo "🔍 Running all CI checks locally..."
	$(PYTHON) dataset_generator.py
	$(PYTHON) validate_dataset.py --data-dir $(DATA_DIR)
	@echo "✅ All CI checks passed!"

release-check: ci-check build
	@echo "🔍 Checking release readiness..."
	@echo "Verifying package can be installed..."
	$(PIP) install dist/*.whl --force-reinstall --quiet
	$(PYTHON) -c "from dataset_generator import RequirementsDatasetGenerator; print('✅ Package import successful')"
	twine check dist/*
	@echo "✅ Release checks passed!"

security-check:
	@echo "🔒 Running security vulnerability scan..."
	$(PIP) install safety bandit
	safety check
	bandit -r . -f json -o security_report.json || true
	@echo "✅ Security check complete! See security_report.json"

dependency-check:
	@echo "🔍 Checking for outdated dependencies..."
	$(PIP) list --outdated
	@echo "✅ Dependency check complete!"

# ============================================================================
# Performance and Benchmarking
# ============================================================================

benchmark:
	@echo "⏱️  Running performance benchmarks..."
	@echo "Generating dataset with timing..."
	@time $(PYTHON) -c "from dataset_generator import RequirementsDatasetGenerator; g = RequirementsDatasetGenerator(); g.generate_complete_dataset('benchmark_data')"
	@echo "Validating dataset with timing..."
	@time $(PYTHON) validate_dataset.py --data-dir benchmark_data
	@echo "Analyzing dataset with timing..."
	@time $(PYTHON) -c "from example_analysis import RequirementsAnalyzer; r = RequirementsAnalyzer('benchmark_data'); r.analyze_core_findings()"
	rm -rf benchmark_data
	@echo "✅ Benchmark complete!"

profile:
	@echo "📊 Profiling dataset generation..."
	$(PYTHON) -m cProfile -o profile_stats.prof dataset_generator.py
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile_stats.prof'); p.sort_stats('cumulative').print_stats(20)"
	@echo "✅ Profiling complete! See profile_stats.prof"

memory-check:
	@echo "💾 Checking memory usage..."
	$(PIP) install memory-profiler
	$(PYTHON) -m memory_profiler dataset_generator.py
	@echo "✅ Memory check complete!"

# ============================================================================
# Maintenance and Updates
# ============================================================================

update-deps:
	@echo "🔄 Updating dependencies..."
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -r requirements.txt
	@echo "✅ Dependencies updated!"

check-updates:
	@echo "🔍 Checking for available updates..."
	$(PIP) list --outdated
	@echo "ℹ️  Run 'make update-deps' to update all dependencies"

freeze-deps:
	@echo "❄️  Freezing current dependency versions..."
	$(PIP) freeze > requirements-frozen.txt
	@echo "✅ Dependencies frozen to requirements-frozen.txt"

# ============================================================================
# Git and Version Control
# ============================================================================

git-status:
	@echo "📊 Git status summary..."
	@git status --porcelain | wc -l | xargs echo "Modified files:"
	@git log --oneline -5
	@echo "Current branch: $$(git branch --show-current)"

pre-commit-all:
	@echo "🔍 Running pre-commit on all files..."
	pre-commit run --all-files
	@echo "✅ Pre-commit checks complete!"

# ============================================================================
# Complete Workflows
# ============================================================================

all: clean install generate validate analyze
	@echo "🎉 Complete workflow finished!"
	@echo "📁 Generated data: $(DATA_DIR)/"
	@echo "📈 Analysis results: $(ANALYSIS_DIR)/"

dev-setup: install-dev
	@echo "🛠️  Development environment setup complete!"
	@echo "Next steps:"
	@echo "  1. Run 'make quick-test' to verify everything works"
	@echo "  2. Run 'make all' for complete workflow"
	@echo "  3. See 'make help' for all available commands"

production-release: release-check
	@echo "🚀 Production release workflow..."
	@read -p "Enter version tag (e.g., v1.0.0): " version && \
	git tag -a $$version -m "Release $$version" && \
	git push origin $$version
	@echo "✅ Production release tagged and pushed!"

# ============================================================================
# Platform-specific Commands
# ============================================================================

# Windows-specific commands
ifeq ($(OS),Windows_NT)
    PYTHON := python
    PIP := pip
    RM := del /Q
    RMDIR := rmdir /S /Q
endif

# macOS-specific optimizations
ifeq ($(shell uname -s),Darwin)
    # Use faster parallel processing on macOS
    MAKEFLAGS += -j$(shell sysctl -n hw.ncpu)
endif

# Linux-specific optimizations  
ifeq ($(shell uname -s),Linux)
    # Use faster parallel processing on Linux
    MAKEFLAGS += -j$(shell nproc)
endif

# ============================================================================
# Help and Information
# ============================================================================

info:
	@echo "Requirements Engineering Dataset - Project Information"
	@echo "===================================================="
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Git status: $$(git status --porcelain 2>/dev/null | wc -l || echo 'N/A') modified files"
	@echo "Available disk space: $$(df -h . | tail -1 | awk '{print $$4}')"

version:
	@echo "Requirements Engineering Dataset v1.0.0"
	@echo "Author: Cornelius Chimuanya Okechukwu"
	@echo "Institution: Tomas Bata University in Zlin"
	@echo "License: MIT with Academic Attribution"

# ============================================================================
# Error Handling
# ============================================================================

# Check if Python is available
check-python:
	@$(PYTHON) --version >/dev/null 2>&1 || (echo "❌ Python not found! Please install Python 3.8+"; exit 1)
	@echo "✅ Python available: $$($(PYTHON) --version)"

# Check if required tools are available
check-tools: check-python
	@command -v git >/dev/null 2>&1 || (echo "❌ Git not found!"; exit 1)
	@echo "✅ All required tools available"

# Ensure directories exist
create-dirs:
	@mkdir -p $(DATA_DIR) $(ANALYSIS_DIR) $(BUILD_DIR) $(DIST_DIR)

# ============================================================================
# Default target
# ============================================================================

.DEFAULT_GOAL := help
