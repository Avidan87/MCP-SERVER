# Multi-stage build for optimal Railway deployment
# Stage 1: Builder - Downloads model and installs dependencies
# Stage 2: Runtime - Lean production image

# ============================================
# STAGE 1: Builder
# ============================================
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies to user directory
RUN pip install --no-cache-dir --user -r requirements.txt

# Set HuggingFace cache location for Depth Anything V2
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

# Pre-download Depth Anything V2 Small model during build (baked into image!)
# This saves 30-90 seconds on every cold start
# Using AutoModel instead of pipeline for better performance
RUN python -c "from transformers import AutoModelForDepthEstimation, AutoImageProcessor; \
    print('========================================'); \
    print('Downloading Depth Anything V2 Small...'); \
    print('========================================'); \
    model = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf'); \
    processor = AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf'); \
    print('========================================'); \
    print('Depth Anything V2 cached successfully!'); \
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

# Copy pre-downloaded Depth Anything V2 model from builder (THIS IS KEY!)
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Update PATH to include user-installed packages
ENV PATH=/root/.local/bin:$PATH

# Environment variables for Railway
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV PORT=8080

# Copy application code
COPY . .

# Expose port (Railway will override with $PORT)
EXPOSE 8080

# Start server with uvicorn
# --workers 1: Single worker for Depth Anything V2 (model in memory)
# --timeout-keep-alive 300: 5 min timeout for long-running requests
# Use JSON array form with sh -c to ensure $PORT variable expansion works
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 300 --log-level info"]
