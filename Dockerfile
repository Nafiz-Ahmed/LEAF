# Use Python 3.11 with Debian Bookworm (avoids python3-distutils issue)
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies including Git LFS
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-venv \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    libxrandr2 \
    libxss1 \
    libxtst6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    wget \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Initialize Git LFS
RUN git lfs install

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories that your app expects
RUN mkdir -p uploads logs

# Set environment variables for your Flask ML app
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose port (Railway will provide PORT env var)
EXPOSE 5000

# Add healthcheck and better startup
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

# Your app.py already handles Railway's PORT correctly
CMD ["python", "app.py"]