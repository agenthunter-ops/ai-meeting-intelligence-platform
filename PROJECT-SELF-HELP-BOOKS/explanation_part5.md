# AI Meeting Intelligence Platform - Part 5: Deployment & Infrastructure

## Table of Contents
1. [Docker Containerization Strategy](#docker-containerization-strategy)
2. [Multi-Environment Deployment](#multi-environment-deployment)
3. [Service Orchestration](#service-orchestration)
4. [Infrastructure as Code](#infrastructure-as-code)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Scaling and Performance](#scaling-and-performance)
7. [Security and Compliance](#security-and-compliance)
8. [Disaster Recovery](#disaster-recovery)

---

## Docker Containerization Strategy

### Multi-Stage Build Optimization

Our containerization strategy leverages multi-stage builds to optimize image sizes and security. Each service follows container best practices while maintaining development flexibility.

```dockerfile
# Dockerfile.backend - Backend Service Container

# Build stage - Install dependencies and build
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_ENV=production
ARG BUILD_VERSION=latest

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Runtime stage - Minimal production image
FROM python:3.11-slim as runtime

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy user from builder stage
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/group

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --from=builder /app /app

# Set working directory and permissions
WORKDIR /app
RUN chown -R appuser:appuser /app

# Create directories for data persistence
RUN mkdir -p /app/media/uploads /app/media/processed /app/logs \
    && chown -R appuser:appuser /app/media /app/logs

# Switch to non-root user
USER appuser

# Update PATH to include user packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Development stage - Hot reload and debugging
FROM runtime as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython

# Switch back to appuser
USER appuser

# Override for development with hot reload
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]
```

### Service-Specific Containerization

Each AI service has specialized container configurations optimized for their specific requirements:

```dockerfile
# Dockerfile.whisper - Whisper Service with GPU Support

FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Download and cache Whisper models
RUN python3 -c "import whisper; whisper.load_model('base')"
RUN python3 -c "import whisper; whisper.load_model('small')"

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r whisperuser && useradd -r -g whisperuser whisperuser
RUN chown -R whisperuser:whisperuser /app

# Switch to non-root user
USER whisperuser

# Health check for gRPC service
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import grpc; import meeting_pb2_grpc; channel = grpc.insecure_channel('localhost:50051'); stub = meeting_pb2_grpc.WhisperServiceStub(channel)" || exit 1

# Expose gRPC port
EXPOSE 50051

# Command
CMD ["python3", "server.py"]

# CPU-only variant
FROM base as cpu
ENV CUDA_VISIBLE_DEVICES=""

# GPU variant
FROM base as gpu
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

```dockerfile
# Dockerfile.llm - LLM Service with Ollama Integration

FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r llmuser && useradd -r -g llmuser llmuser
RUN chown -R llmuser:llmuser /app

USER llmuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Container Security Hardening

Our containers implement security best practices:

```dockerfile
# Security-hardened base image
FROM python:3.11-slim as security-base

# Update system and remove unnecessary packages
RUN apt-get update && apt-get upgrade -y \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with specific UID/GID
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m -s /bin/bash appuser

# Set security-focused environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/appuser/.local/bin:$PATH"
ENV TMPDIR=/tmp
ENV HOME=/home/appuser

# Create secure directories
RUN mkdir -p /app /tmp \
    && chmod 755 /app \
    && chmod 1777 /tmp \
    && chown appuser:appgroup /app

WORKDIR /app

# Security labels and metadata
LABEL security.scan="enabled"
LABEL security.non-root="true"
LABEL security.hardened="true"
LABEL maintainer="ai-meeting-platform@company.com"

# Remove setuid/setgid permissions
RUN find / -type f \( -perm -4000 -o -perm -2000 \) -exec chmod -s {} \; 2>/dev/null || true

# Switch to non-root user
USER appuser

# Security-focused CMD
CMD ["python", "-u", "app.py"]
```

---

## Multi-Environment Deployment

### Environment Configuration Strategy

We support multiple deployment environments with environment-specific configurations:

```yaml
# docker-compose.yml - Base configuration

version: '3.8'

x-common-variables: &common-variables
  PYTHONPATH: /app
  PYTHONUNBUFFERED: "1"
  LOG_LEVEL: ${LOG_LEVEL:-INFO}

x-restart-policy: &restart-policy
  restart_policy:
    condition: unless-stopped
    delay: 5s
    max_attempts: 3
    window: 60s

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: meeting-postgres
    <<: *restart-policy
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-meeting_intelligence}
      POSTGRES_USER: ${POSTGRES_USER:-meeting_user}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-meeting_secure_password_2025}
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d:ro
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-meeting_user} -d ${POSTGRES_DB:-meeting_intelligence}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    networks:
      - meeting-intelligence
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'

  # Redis Cache and Message Broker
  redis:
    image: redis:7-alpine
    container_name: meeting-redis
    <<: *restart-policy
    command: redis-server /usr/local/etc/redis/redis.conf
    volumes:
      - redis_data:/data
      - ./infrastructure/redis/redis.conf:/usr/local/etc/redis/redis.conf:ro
    ports:
      - "${REDIS_PORT:-6379}:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
      start_period: 10s
    networks:
      - meeting-intelligence
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.25'

  # ChromaDB Vector Database
  chromadb:
    image: chromadb/chroma:latest
    container_name: meeting-chroma
    <<: *restart-policy
    environment:
      CHROMA_HOST: 0.0.0.0
      CHROMA_PORT: 8000
      CHROMA_DB_IMPL: clickhouse
      CLICKHOUSE_HOST: clickhouse
      CLICKHOUSE_PORT: 9000
    volumes:
      - chroma_data:/chroma/chroma
    ports:
      - "${CHROMA_PORT:-8000}:8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    networks:
      - meeting-intelligence
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  # ClickHouse for ChromaDB
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    container_name: meeting-clickhouse
    <<: *restart-policy
    environment:
      CLICKHOUSE_USER: ${CLICKHOUSE_USER:-default}
      CLICKHOUSE_PASSWORD: ${CLICKHOUSE_PASSWORD:-}
      CLICKHOUSE_DB: ${CLICKHOUSE_DB:-chroma}
    volumes:
      - clickhouse_data:/var/lib/clickhouse
      - ./infrastructure/clickhouse/config.xml:/etc/clickhouse-server/config.xml:ro
    networks:
      - meeting-intelligence
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  # Ollama LLM Runtime
  ollama:
    image: ollama/ollama:latest
    container_name: meeting-ollama
    <<: *restart-policy
    environment:
      OLLAMA_HOST: 0.0.0.0:11434
      OLLAMA_MODELS: /root/.ollama/models
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "${OLLAMA_PORT:-11434}:11434"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    networks:
      - meeting-intelligence
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
    # GPU support for Ollama
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

  # Backend API Service
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
      target: ${BUILD_TARGET:-runtime}
      args:
        BUILD_ENV: ${BUILD_ENV:-production}
        BUILD_VERSION: ${BUILD_VERSION:-latest}
    container_name: meeting-backend
    <<: *restart-policy
    environment:
      <<: *common-variables
      DATABASE_URL: postgresql://${POSTGRES_USER:-meeting_user}:${POSTGRES_PASSWORD:-meeting_secure_password_2025}@postgres:5432/${POSTGRES_DB:-meeting_intelligence}
      REDIS_URL: redis://redis:6379/0
      CELERY_BROKER_URL: redis://redis:6379/0
      CELERY_RESULT_BACKEND: redis://redis:6379/0
      CHROMA_HOST: chromadb
      CHROMA_PORT: 8000
      WHISPER_GRPC_URL: whisper-service:50051
      LLM_SERVICE_URL: http://llm-service:8001
      OLLAMA_URL: http://ollama:11434
      SECRET_KEY: ${SECRET_KEY:-development-secret-key-change-in-production}
      DEBUG: ${DEBUG:-false}
      CORS_ORIGINS: ${CORS_ORIGINS:-http://localhost:4200,http://frontend:4200}
    volumes:
      - media_storage:/app/media
      - ./backend/logs:/app/logs
    ports:
      - "${BACKEND_PORT:-8080}:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      chromadb:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    networks:
      - meeting-intelligence
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'

  # Celery Worker for Background Tasks
  celery-worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
      target: ${BUILD_TARGET:-runtime}
    container_name: meeting-celery-worker
    <<: *restart-policy
    environment:
      <<: *common-variables
      DATABASE_URL: postgresql://${POSTGRES_USER:-meeting_user}:${POSTGRES_PASSWORD:-meeting_secure_password_2025}@postgres:5432/${POSTGRES_DB:-meeting_intelligence}
      REDIS_URL: redis://redis:6379/0
      CELERY_BROKER_URL: redis://redis:6379/0
      CELERY_RESULT_BACKEND: redis://redis:6379/0
      CHROMA_HOST: chromadb
      CHROMA_PORT: 8000
      WHISPER_GRPC_URL: whisper-service:50051
      LLM_SERVICE_URL: http://llm-service:8001
      OLLAMA_URL: http://ollama:11434
    volumes:
      - media_storage:/app/media
      - ./backend/logs:/app/logs
    depends_on:
      - postgres
      - redis
      - backend
    command: ["celery", "-A", "celery_config.celery_app", "worker", "--loglevel=info", "--concurrency=2"]
    networks:
      - meeting-intelligence
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  # Whisper Speech-to-Text Service
  whisper-service:
    build:
      context: ./whisper_service
      dockerfile: Dockerfile
      target: ${WHISPER_TARGET:-cpu}
    container_name: meeting-whisper
    <<: *restart-policy
    environment:
      GRPC_PORT: 50051
      MODEL_SIZE: ${WHISPER_MODEL_SIZE:-base}
      DEVICE: ${WHISPER_DEVICE:-cpu}
    volumes:
      - whisper_models:/app/models
      - media_storage:/app/media:ro
    ports:
      - "${WHISPER_PORT:-50051}:50051"
    healthcheck:
      test: ["CMD", "python", "-c", "import grpc; grpc.insecure_channel('localhost:50051').channel_ready()"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    networks:
      - meeting-intelligence
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  # LLM Analysis Service
  llm-service:
    build:
      context: ./llm_service
      dockerfile: Dockerfile
    container_name: meeting-llm
    <<: *restart-policy
    environment:
      OLLAMA_URL: http://ollama:11434
      LLM_MODEL: ${LLM_MODEL:-llama2:7b}
      LOG_LEVEL: ${LOG_LEVEL:-INFO}
    ports:
      - "${LLM_PORT:-8001}:8001"
    depends_on:
      - ollama
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    networks:
      - meeting-intelligence
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

  # Frontend Web Application
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.simple
    container_name: meeting-frontend
    <<: *restart-policy
    ports:
      - "${FRONTEND_PORT:-4200}:80"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    networks:
      - meeting-intelligence
    depends_on:
      - backend
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.25'

# Shared volumes
volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_DIR:-./data}/postgres
  redis_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_DIR:-./data}/redis
  chroma_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_DIR:-./data}/chroma
  clickhouse_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_DIR:-./data}/clickhouse
  ollama_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_DIR:-./data}/ollama
  whisper_models:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_DIR:-./data}/whisper_models
  media_storage:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_DIR:-./data}/media

# Network configuration
networks:
  meeting-intelligence:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
    driver_opts:
      com.docker.network.bridge.name: br-meeting-intel
```

### Environment-Specific Overrides

```yaml
# docker-compose.development.yml - Development Environment

version: '3.8'

services:
  backend:
    build:
      target: development
    environment:
      DEBUG: "true"
      LOG_LEVEL: DEBUG
      RELOAD: "true"
    volumes:
      - ./backend:/app:cached
      - ~/.cache/pip:/root/.cache/pip
    command: ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]
    ports:
      - "8000:8000"  # Direct access for debugging

  celery-worker:
    build:
      target: development
    environment:
      LOG_LEVEL: DEBUG
    volumes:
      - ./backend:/app:cached
    command: ["celery", "-A", "celery_config.celery_app", "worker", "--loglevel=debug", "--concurrency=1", "--pool=solo"]

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      target: development
    volumes:
      - ./frontend:/app:cached
      - /app/node_modules
    environment:
      NODE_ENV: development
      CHOKIDAR_USEPOLLING: "true"
    command: ["npm", "run", "serve:dev"]

  # Development tools
  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: meeting-pgadmin-dev
    restart: unless-stopped
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@meeting-intelligence.com
      PGADMIN_DEFAULT_PASSWORD: admin_password_2025
      PGADMIN_CONFIG_SERVER_MODE: 'False'
    ports:
      - "5050:80"
    volumes:
      - pgadmin_data:/var/lib/pgadmin
    networks:
      - meeting-intelligence

  flower:
    build:
      context: ./backend
      dockerfile: Dockerfile
      target: development
    container_name: meeting-flower-dev
    restart: unless-stopped
    environment:
      CELERY_BROKER_URL: redis://redis:6379/0
      CELERY_RESULT_BACKEND: redis://redis:6379/0
    ports:
      - "5555:5555"
    depends_on:
      - redis
    networks:
      - meeting-intelligence
    command: ["celery", "-A", "celery_config.celery_app", "flower", "--port=5555"]

volumes:
  pgadmin_data:
```

```yaml
# docker-compose.production.yml - Production Environment

version: '3.8'

services:
  backend:
    build:
      target: runtime
    environment:
      DEBUG: "false"
      LOG_LEVEL: INFO
      WORKERS: 4
    deploy:
      replicas: 2
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
        order: start-first
      rollback_config:
        parallelism: 1
        delay: 5s
        failure_action: pause
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 60s
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'
    command: ["gunicorn", "app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

  celery-worker:
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 60s
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  # Production reverse proxy
  nginx:
    image: nginx:alpine
    container_name: meeting-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./infrastructure/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./infrastructure/nginx/conf.d:/etc/nginx/conf.d:ro
      - ./infrastructure/ssl:/etc/nginx/ssl:ro
      - nginx_logs:/var/log/nginx
    depends_on:
      - backend
      - frontend
    networks:
      - meeting-intelligence
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.5'

  # Log aggregation
  fluentd:
    image: fluent/fluentd:latest
    container_name: meeting-fluentd
    restart: unless-stopped
    volumes:
      - ./infrastructure/fluentd/conf:/fluentd/etc:ro
      - ./logs:/var/log/app:ro
      - nginx_logs:/var/log/nginx:ro
    ports:
      - "24224:24224"
    networks:
      - meeting-intelligence

volumes:
  nginx_logs:
```

### Environment Variables Management

```bash
# .env.example - Environment variables template

# Environment
NODE_ENV=development
BUILD_ENV=development
BUILD_TARGET=development
DEBUG=true
LOG_LEVEL=DEBUG

# Application
SECRET_KEY=development-secret-key-change-in-production
CORS_ORIGINS=http://localhost:4200,http://localhost:3000

# Database
POSTGRES_DB=meeting_intelligence
POSTGRES_USER=meeting_user
POSTGRES_PASSWORD=meeting_secure_password_2025
POSTGRES_PORT=5432
DATABASE_URL=postgresql://meeting_user:meeting_secure_password_2025@localhost:5432/meeting_intelligence

# Redis
REDIS_PORT=6379
REDIS_URL=redis://localhost:6379/0

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# AI Services
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_PORT=50051
WHISPER_GRPC_URL=localhost:50051

LLM_MODEL=llama2:7b
LLM_PORT=8001
LLM_SERVICE_URL=http://localhost:8001

OLLAMA_PORT=11434
OLLAMA_URL=http://localhost:11434

# ChromaDB
CHROMA_PORT=8000
CHROMA_HOST=localhost

# Application Ports
BACKEND_PORT=8080
FRONTEND_PORT=4200

# Data Directory
DATA_DIR=./data

# Build Information
BUILD_VERSION=latest
```

```bash
# scripts/env-setup.sh - Environment setup script

#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "All dependencies are installed."
}

# Setup environment files
setup_env() {
    local env_type=${1:-development}
    print_info "Setting up environment for: $env_type"
    
    # Copy environment template if .env doesn't exist
    if [[ ! -f .env ]]; then
        if [[ -f .env.example ]]; then
            cp .env.example .env
            print_success "Created .env file from template"
        else
            print_error ".env.example not found"
            exit 1
        fi
    else
        print_warning ".env file already exists"
    fi
    
    # Update environment-specific variables
    case $env_type in
        "production")
            sed -i.bak 's/NODE_ENV=development/NODE_ENV=production/' .env
            sed -i.bak 's/DEBUG=true/DEBUG=false/' .env
            sed -i.bak 's/LOG_LEVEL=DEBUG/LOG_LEVEL=INFO/' .env
            sed -i.bak 's/BUILD_TARGET=development/BUILD_TARGET=runtime/' .env
            print_info "Updated .env for production"
            ;;
        "staging")
            sed -i.bak 's/NODE_ENV=development/NODE_ENV=staging/' .env
            sed -i.bak 's/DEBUG=true/DEBUG=false/' .env
            sed -i.bak 's/LOG_LEVEL=DEBUG/LOG_LEVEL=INFO/' .env
            print_info "Updated .env for staging"
            ;;
        *)
            print_info "Using development configuration"
            ;;
    esac
    
    # Generate secret key for production
    if [[ $env_type == "production" ]]; then
        SECRET_KEY=$(openssl rand -hex 32)
        sed -i.bak "s/SECRET_KEY=development-secret-key-change-in-production/SECRET_KEY=$SECRET_KEY/" .env
        print_success "Generated secure secret key"
    fi
}

# Setup data directories
setup_directories() {
    print_info "Setting up data directories..."
    
    local data_dir=$(grep DATA_DIR .env | cut -d '=' -f2)
    data_dir=${data_dir:-./data}
    
    mkdir -p "$data_dir"/{postgres,redis,chroma,clickhouse,ollama,whisper_models,media/{uploads,processed}}
    mkdir -p logs
    
    # Set proper permissions
    chmod 755 "$data_dir"
    chmod 777 "$data_dir"/media
    
    print_success "Created data directories"
}

# Pull required images
pull_images() {
    print_info "Pulling required Docker images..."
    
    docker pull postgres:15-alpine
    docker pull redis:7-alpine
    docker pull chromadb/chroma:latest
    docker pull clickhouse/clickhouse-server:latest
    docker pull ollama/ollama:latest
    docker pull nginx:alpine
    
    print_success "Pulled all required images"
}

# Initialize Ollama models
init_ollama() {
    print_info "Initializing Ollama models..."
    
    # Start only Ollama service temporarily
    docker-compose up -d ollama
    
    # Wait for Ollama to be ready
    print_info "Waiting for Ollama to be ready..."
    until docker exec meeting-ollama ollama list >/dev/null 2>&1; do
        sleep 5
        print_info "Still waiting for Ollama..."
    done
    
    # Pull required models
    print_info "Pulling LLM models (this may take a while)..."
    docker exec meeting-ollama ollama pull llama2:7b
    docker exec meeting-ollama ollama pull llama2:7b-chat
    
    # Stop Ollama
    docker-compose stop ollama
    
    print_success "Ollama models initialized"
}

# Main function
main() {
    local env_type=${1:-development}
    local skip_models=${2:-false}
    
    print_info "Setting up AI Meeting Intelligence Platform"
    print_info "Environment: $env_type"
    
    check_dependencies
    setup_env "$env_type"
    setup_directories
    pull_images
    
    if [[ "$skip_models" != "true" ]]; then
        init_ollama
    else
        print_warning "Skipping Ollama model initialization"
    fi
    
    print_success "Environment setup complete!"
    print_info "You can now run: docker-compose up -d"
}

# Script usage
usage() {
    echo "Usage: $0 [environment] [skip-models]"
    echo "  environment: development (default), staging, production"
    echo "  skip-models: true to skip Ollama model download"
    echo ""
    echo "Examples:"
    echo "  $0                          # Development with models"
    echo "  $0 production               # Production with models"
    echo "  $0 development true         # Development without models"
}

# Parse arguments
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi

# Run main function
main "$@"
```

---

## Service Orchestration

### Advanced Docker Compose Orchestration

Our orchestration strategy handles complex service dependencies and startup sequences:

```yaml
# docker-compose.override.yml - Advanced orchestration

version: '3.8'

services:
  # Service dependency management
  backend:
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      chromadb:
        condition: service_started
    environment:
      WAIT_HOSTS: postgres:5432,redis:6379,chromadb:8000
      WAIT_HOSTS_TIMEOUT: 300
      WAIT_SLEEP_INTERVAL: 5
      WAIT_HOST_CONNECT_TIMEOUT: 30
    entrypoint: ["./scripts/wait-for-services.sh"]
    command: ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

  # Graceful shutdown handling
  celery-worker:
    stop_grace_period: 30s
    stop_signal: SIGTERM
    environment:
      CELERY_WORKER_HIJACK_ROOT_LOGGER: "false"
      CELERY_WORKER_LOG_COLOR: "true"
      CELERY_TASK_SERIALIZER: json
      CELERY_ACCEPT_CONTENT: ["json"]
      CELERY_RESULT_SERIALIZER: json
      CELERY_TIMEZONE: UTC
      CELERY_ENABLE_UTC: "true"
      CELERY_TASK_TRACK_STARTED: "true"
      CELERY_TASK_TIME_LIMIT: 1800  # 30 minutes
      CELERY_TASK_SOFT_TIME_LIMIT: 1500  # 25 minutes
      CELERY_WORKER_MAX_TASKS_PER_CHILD: 1000
      CELERY_WORKER_PREFETCH_MULTIPLIER: 1

  # Service scaling configuration
  whisper-service:
    deploy:
      mode: replicated
      replicas: 1
      placement:
        constraints:
          - node.labels.gpu == true  # For GPU nodes
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Health check optimizations
  postgres:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $POSTGRES_USER -d $POSTGRES_DB"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s

  redis:
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
      start_period: 10s

  # Backup service
  backup:
    image: postgres:15-alpine
    container_name: meeting-backup
    restart: "no"
    environment:
      PGPASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - ./backups:/backups
      - ./scripts/backup.sh:/backup.sh:ro
    networks:
      - meeting-intelligence
    depends_on:
      - postgres
    command: /backup.sh
    profiles:
      - backup

  # Log rotation service
  logrotate:
    image: alpine:latest
    container_name: meeting-logrotate
    restart: unless-stopped
    volumes:
      - ./logs:/var/log/app
      - ./infrastructure/logrotate/logrotate.conf:/etc/logrotate.conf:ro
    command: >
      sh -c "
        echo '0 2 * * * /usr/sbin/logrotate /etc/logrotate.conf' > /tmp/crontab &&
        crontab /tmp/crontab &&
        crond -f
      "
    profiles:
      - logging
```

### Service Discovery and Load Balancing

```nginx
# infrastructure/nginx/nginx.conf - Load balancing configuration

user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

# Optimize worker connections
events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging format
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent "$http_referer" '
                   '"$http_user_agent" "$http_x_forwarded_for" '
                   'rt=$request_time uct="$upstream_connect_time" '
                   'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;

    # Performance optimizations
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript 
               application/javascript application/xml+rss 
               application/json application/xml;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=1r/s;

    # Upstream backend servers
    upstream backend_servers {
        least_conn;
        server backend:8000 max_fails=3 fail_timeout=30s;
        # Additional backend instances for scaling
        # server backend-2:8000 max_fails=3 fail_timeout=30s;
        # server backend-3:8000 max_fails=3 fail_timeout=30s;
        
        keepalive 32;
    }

    # Upstream for AI services
    upstream whisper_servers {
        server whisper-service:50051 max_fails=2 fail_timeout=60s;
        keepalive 8;
    }

    upstream llm_servers {
        server llm-service:8001 max_fails=2 fail_timeout=60s;
        keepalive 8;
    }

    # Main server configuration
    server {
        listen 80;
        server_name localhost;
        client_max_body_size 500M;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Referrer-Policy strict-origin-when-cross-origin;
        
        # Frontend static files
        location / {
            proxy_pass http://frontend:80;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Caching for static assets
            location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
                expires 1y;
                add_header Cache-Control "public, immutable";
            }
        }

        # API endpoints with rate limiting
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://backend_servers;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
            
            # Buffering
            proxy_buffering on;
            proxy_buffer_size 128k;
            proxy_buffers 4 256k;
            proxy_busy_buffers_size 256k;
        }

        # File upload endpoint with special handling
        location /api/upload {
            limit_req zone=upload burst=5 nodelay;
            
            proxy_pass http://backend_servers;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Extended timeouts for file uploads
            proxy_connect_timeout 60s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;
            
            # Large file support
            client_max_body_size 500M;
            proxy_request_buffering off;
            proxy_max_temp_file_size 0;
        }

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }

        # Metrics endpoint (protected)
        location /metrics {
            allow 127.0.0.1;
            allow 172.16.0.0/12;  # Docker networks
            deny all;
            
            proxy_pass http://backend_servers;
            proxy_set_header Host $host;
        }

        # Error pages
        error_page 404 /404.html;
        error_page 500 502 503 504 /50x.html;
        
        location = /50x.html {
            root /usr/share/nginx/html;
        }
    }

    # SSL server (production)
    server {
        listen 443 ssl http2;
        server_name your-domain.com;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # SSL configuration
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;
        
        # HSTS
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # Same location blocks as HTTP server
        include /etc/nginx/conf.d/locations.conf;
    }
}
```

### Service Health Monitoring

```bash
# scripts/health-check.sh - Comprehensive health monitoring

#!/bin/bash

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="docker-compose.yml"
HEALTH_CHECK_TIMEOUT=30
RETRY_ATTEMPTS=3
RETRY_DELAY=5

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
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

# Health check functions
check_container_health() {
    local container_name=$1
    local health_status
    
    health_status=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "no-health-check")
    
    case $health_status in
        "healthy")
            log_success "$container_name: healthy"
            return 0
            ;;
        "unhealthy")
            log_error "$container_name: unhealthy"
            return 1
            ;;
        "starting")
            log_warning "$container_name: starting"
            return 2
            ;;
        "no-health-check")
            # Check if container is running
            if docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
                log_warning "$container_name: running (no health check)"
                return 0
            else
                log_error "$container_name: not running"
                return 1
            fi
            ;;
        *)
            log_error "$container_name: unknown status ($health_status)"
            return 1
            ;;
    esac
}

check_service_endpoint() {
    local service_name=$1
    local endpoint=$2
    local expected_status=${3:-200}
    local timeout=${4:-10}
    
    log_info "Checking $service_name endpoint: $endpoint"
    
    local status_code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$timeout" "$endpoint" || echo "000")
    
    if [[ "$status_code" == "$expected_status" ]]; then
        log_success "$service_name: endpoint healthy ($status_code)"
        return 0
    else
        log_error "$service_name: endpoint unhealthy ($status_code)"
        return 1
    fi
}

check_database_connection() {
    local container_name="meeting-postgres"
    local db_name="meeting_intelligence"
    local db_user="meeting_user"
    
    log_info "Checking database connection"
    
    if docker exec "$container_name" pg_isready -U "$db_user" -d "$db_name" >/dev/null 2>&1; then
        log_success "Database: connection healthy"
        return 0
    else
        log_error "Database: connection failed"
        return 1
    fi
}

check_redis_connection() {
    local container_name="meeting-redis"
    
    log_info "Checking Redis connection"
    
    if docker exec "$container_name" redis-cli ping | grep -q "PONG"; then
        log_success "Redis: connection healthy"
        return 0
    else
        log_error "Redis: connection failed"
        return 1
    fi
}

check_ai_services() {
    local services=("whisper-service:50051" "llm-service:8001" "ollama:11434")
    local all_healthy=true
    
    for service in "${services[@]}"; do
        IFS=':' read -r service_name port <<< "$service"
        
        if docker exec "meeting-${service_name}" nc -z localhost "$port" >/dev/null 2>&1; then
            log_success "$service_name: port $port accessible"
        else
            log_error "$service_name: port $port not accessible"
            all_healthy=false
        fi
    done
    
    return $all_healthy
}

wait_for_health() {
    local container_name=$1
    local max_attempts=${2:-30}
    local attempt=1
    
    log_info "Waiting for $container_name to become healthy (max $max_attempts attempts)"
    
    while [[ $attempt -le $max_attempts ]]; do
        if check_container_health "$container_name" >/dev/null 2>&1; then
            return 0
        fi
        
        echo -n "."
        sleep 2
        ((attempt++))
    done
    
    echo
    log_error "$container_name failed to become healthy after $max_attempts attempts"
    return 1
}

# Main health check function
main_health_check() {
    local exit_code=0
    
    log_info "Starting comprehensive health check"
    
    # Check if Docker Compose is running
    if ! docker-compose ps >/dev/null 2>&1; then
        log_error "Docker Compose services are not running"
        return 1
    fi
    
    # Core infrastructure services
    local core_services=("meeting-postgres" "meeting-redis" "meeting-chroma")
    
    log_info "Checking core infrastructure services..."
    for service in "${core_services[@]}"; do
        if ! check_container_health "$service"; then
            exit_code=1
        fi
    done
    
    # Application services
    local app_services=("meeting-backend" "meeting-celery-worker" "meeting-frontend")
    
    log_info "Checking application services..."
    for service in "${app_services[@]}"; do
        if ! check_container_health "$service"; then
            exit_code=1
        fi
    done
    
    # AI services
    local ai_services=("meeting-whisper" "meeting-llm" "meeting-ollama")
    
    log_info "Checking AI services..."
    for service in "${ai_services[@]}"; do
        if ! check_container_health "$service"; then
            exit_code=1
        fi
    done
    
    # Database connectivity
    if ! check_database_connection; then
        exit_code=1
    fi
    
    # Redis connectivity
    if ! check_redis_connection; then
        exit_code=1
    fi
    
    # API endpoints
    local endpoints=(
        "Backend API:http://localhost:8080/health"
        "LLM Service:http://localhost:8001/health"
        "Frontend:http://localhost:4200"
    )
    
    log_info "Checking service endpoints..."
    for endpoint in "${endpoints[@]}"; do
        IFS=':' read -r service_name url <<< "$endpoint"
        if ! check_service_endpoint "$service_name" "$url"; then
            exit_code=1
        fi
    done
    
    # Summary
    if [[ $exit_code -eq 0 ]]; then
        log_success "All health checks passed!"
    else
        log_error "Some health checks failed!"
    fi
    
    return $exit_code
}

# Startup health check with retries
startup_health_check() {
    local max_attempts=60  # 5 minutes with 5-second intervals
    local attempt=1
    
    log_info "Starting startup health check (max ${max_attempts} attempts)"
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt of $max_attempts"
        
        if main_health_check >/dev/null 2>&1; then
            log_success "All services are healthy!"
            return 0
        fi
        
        if [[ $attempt -lt $max_attempts ]]; then
            log_info "Waiting 5 seconds before next attempt..."
            sleep 5
        fi
        
        ((attempt++))
    done
    
    log_error "Services failed to become healthy after $max_attempts attempts"
    return 1
}

# Usage information
usage() {
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "  check     - Run one-time health check (default)"
    echo "  startup   - Wait for all services to become healthy"
    echo "  monitor   - Continuous monitoring"
    echo "  --help    - Show this help message"
}

# Continuous monitoring
monitor_health() {
    log_info "Starting continuous health monitoring (Ctrl+C to stop)"
    
    while true; do
        echo "$(date): Running health check..."
        main_health_check
        echo "---"
        sleep 30
    done
}

# Main execution
case "${1:-check}" in
    "check")
        main_health_check
        ;;
    "startup")
        startup_health_check
        ;;
    "monitor")
        monitor_health
        ;;
    "--help"|"-h")
        usage
        ;;
    *)
        log_error "Unknown option: $1"
        usage
        exit 1
        ;;
esac
```

This completes the first major section of Part 5, covering containerization strategy, multi-environment deployment, and service orchestration. The implementation demonstrates enterprise-grade deployment practices including:

1. **Multi-stage Docker builds** for optimization and security
2. **Environment-specific configurations** for development, staging, and production
3. **Advanced service orchestration** with dependency management
4. **Load balancing and service discovery** using Nginx
5. **Comprehensive health monitoring** with automated checks
6. **Infrastructure automation** through shell scripts

Would you like me to continue with the remaining sections covering Infrastructure as Code, Monitoring & Observability, Scaling & Performance, Security & Compliance, and Disaster Recovery?