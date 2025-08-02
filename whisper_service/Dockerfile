# Dockerfile for Whisper Service
# =============================

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    build-essential \
    gcc \
    g++ \
    make \
    pkg-config \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
# Ensure no whisper-cpp gets through even if another layer copies an old file
RUN sed -i '/whisper-cpp-python/d' requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 50051

# Start the service
CMD ["python", "server.py"] 