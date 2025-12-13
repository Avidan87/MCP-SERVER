# Multi-stage build for optimal Railway deployment
# Stage 1: Builder - Downloads model and installs dependencies
# Stage 2: Runtime - Lean production image

# ============================================
# STAGE 1: Builder
# ============================================
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies to user directory
RUN pip install --no-cache-dir --user -r requirements.txt

# Set torch cache location for model download
ENV TORCH_HOME=/root/.cache/torch

# Pre-download MiDaS_small model during build (baked into image!)
# This saves 30-60 seconds on every cold start
RUN python -c "import torch; \
    print('========================================'); \
    print('Downloading MiDaS_small model...'); \
    print('========================================'); \
    torch.hub.set_dir('/root/.cache/torch'); \
    model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', verbose=True); \
    print('----------------------------------------'); \
    print('Downloading transforms...'); \
    print('----------------------------------------'); \
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', verbose=True); \
    print('========================================'); \
    print('MiDaS_small model cached successfully!'); \
    print('Model will be available on container startup'); \
    print('========================================');"

# ============================================
# STAGE 2: Runtime
# ============================================
FROM python:3.12-slim

# Install only runtime dependencies (no build tools!)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder (includes all dependencies)
COPY --from=builder /root/.local /root/.local

# Copy pre-downloaded MiDaS model from builder (THIS IS KEY!)
COPY --from=builder /root/.cache/torch /root/.cache/torch

# Update PATH to include user-installed packages
ENV PATH=/root/.local/bin:$PATH

# Environment variables for Railway
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/root/.cache/torch
ENV PORT=8080

# Copy application code
COPY . .

# Expose port (Railway will override with $PORT)
EXPOSE 8080

# Start server with uvicorn
# --workers 1: Single worker for MiDaS (model in memory)
# --timeout-keep-alive 300: 5 min timeout for long-running requests
CMD exec uvicorn server:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --timeout-keep-alive 300 \
    --log-level info
