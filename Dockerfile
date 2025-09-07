# Use official Python 3.12 slim image
FROM python:3.12-slim

# Install system dependencies for Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-venv \
    python3-distutils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel to ensure prebuilt wheels are used
RUN python -m ensurepip \
    && pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Copy the rest of your app
COPY . .

# Set environment variables so venv is used
ENV PATH="/opt/venv/bin:$PATH"

# Expose Railway port
EXPOSE 8080

# Start Flask app using gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "--timeout", "60"]
