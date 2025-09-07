# Use official Python 3.12 slim image
FROM python:3.12-slim

# Install system dependencies needed to build Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3.12-venv \
    python3.12-distutils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Create virtual environment and install Python packages
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"

# Expose port for Railway
EXPOSE 8080

# Start the app with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "--timeout", "60"]
