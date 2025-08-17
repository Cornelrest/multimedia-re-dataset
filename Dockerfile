# Requirements Engineering Dataset - Dockerfile
# ==============================================
# Multi-stage Docker build for efficient containerization

# ============================================================================
# Stage 1: Base Python environment
# ============================================================================
FROM python:3.9-slim-bullseye AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:$PYTHONPATH" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Stage 2: Dependencies installation
# ============================================================================
FROM base AS dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip cache purge

# ============================================================================
# Stage 3: Development environment (optional)
# ============================================================================
FROM dependencies AS development

# Install development dependencies
RUN pip install pytest pytest-cov black flake8 mypy jupyter ipython

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Expose Jupyter port (optional)
EXPOSE 8888

# Default command for development
CMD ["python", "dataset_generator.py"]

# ============================================================================
# Stage 4: Production environment
# ============================================================================
FROM dependencies AS production

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy only necessary files
COPY dataset_generator.py .
COPY validate_dataset.py .
COPY example_analysis.py .
COPY setup.py .
COPY README.md .
COPY LICENSE .

# Install package
RUN pip install .

# Create output directory with proper permissions
RUN mkdir -p /app/generated_data /app/analysis_output && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import dataset_generator; print('Health check passed')" || exit 1

# Default command
CMD ["python", "dataset_generator.py"]

# ============================================================================
# Stage 5: Analysis environment
# ============================================================================
FROM production AS analysis

# Switch back to root to install additional packages
USER root

# Install additional analysis dependencies
RUN pip install plotly dash streamlit

# Copy analysis scripts
COPY example_analysis.py .

# Switch back to appuser
USER appuser

# Expose ports for web interfaces
EXPOSE 8050 8501

# Default command for analysis
CMD ["python", "example_analysis.py"]

# ============================================================================
# Stage 6: Jupyter notebook environment
# ============================================================================
FROM development AS notebook

# Install Jupyter and additional packages
RUN pip install jupyterlab notebook ipywidgets

# Create notebooks directory
RUN mkdir -p /app/notebooks

# Copy example notebooks (if they exist)
COPY notebooks/ ./notebooks/ 2>/dev/null || true

# Expose Jupyter ports
EXPOSE 8888 8889

# Configure Jupyter
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py

# Default command for notebook
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--allow-root", "--ip=0.0.0.0"]

# ============================================================================
# Labels and Metadata
# ============================================================================
LABEL maintainer="Cornelius Chimuanya Okechukwu <okechukwu@utb.cz>"
LABEL version="1.0.0"
LABEL description="Requirements Engineering Multimedia Dataset Generator"
LABEL org.opencontainers.image.title="Requirements Engineering Dataset"
LABEL org.opencontainers.image.description="Synthetic dataset for multimedia-enhanced requirements engineering research"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Cornelius Chimuanya Okechukwu"
LABEL org.opencontainers.image.source="https://github.com/multimedia-re-study/dataset"
LABEL org.opencontainers.image.documentation="https://github.com/multimedia-re-study/dataset/blob/main/README.md"
LABEL org.opencontainers.image.licenses="MIT"

# Build arguments for flexibility
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.revision=$VCS_REF
LABEL org.opencontainers.image.version=$VERSION
