# 🤖 AI Meeting Intelligence Platform

> Transform your meetings into actionable insights with AI-powered transcription, analysis, and task extraction.

[![Build Status](https://github.com/your-org/meeting-intelligence/workflows/CI/badge.svg)](https://github.com/your-org/meeting-intelligence/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![Angular](https://img.shields.io/badge/Angular-17-red.svg)](https://angular.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [🔧 Manual Setup](#-manual-setup)
- [🧪 Development](#-development)
- [🐳 Docker Deployment](#-docker-deployment)
- [📊 Monitoring](#-monitoring)
- [🔒 Security](#-security)
- [🔧 Configuration](#-configuration)
- [📖 API Documentation](#-api-documentation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## ✨ Features

### 🎯 Core Functionality
- **🎤 Smart Transcription**: Convert audio/video to text using Whisper.cpp with speaker identification
- **🧠 AI Insights**: Extract action items, decisions, and sentiment using local LLM (Ollama)
- **🔍 Semantic Search**: Find content across meetings using vector embeddings (ChromaDB)
- **📊 Real-time Analytics**: Live dashboard with meeting metrics and trends
- **📋 Action Items**: Kanban-style board for task management and assignment
- **🔄 Live Updates**: Real-time progress tracking via WebSockets

### 🛠️ Technical Features
- **🏠 Privacy-First**: All AI processing runs locally - no data leaves your infrastructure
- **📱 Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **🌙 Dark Mode**: Full dark mode support with system preference detection
- **⚡ Progressive Web App**: Offline capabilities and native app-like experience
- **🔐 Enterprise Security**: JWT authentication, RBAC, and audit trails
- **📈 Scalable Architecture**: Microservices with horizontal scaling support

## 🏗️ Architecture

The platform is built using a microservices architecture with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   AI Services   │
│   (Angular)     │◄──►│   (FastAPI)     │◄──►│   (Ollama)      │
│   Port: 4200    │    │   Port: 8080    │    │   Port: 11434   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx         │    │   Celery        │    │   Whisper       │
│   (Reverse      │    │   (Background   │    │   (Speech-to-   │
│   Proxy)        │    │   Tasks)        │    │   Text)         │
│   Port: 80      │    │   Port: 6379    │    │   Port: 50051   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Redis         │    │   ChromaDB      │
│   (Database)    │    │   (Cache/Queue) │    │   (Vector DB)   │
│   Port: 5432    │    │   Port: 6379    │    │   Port: 8000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose**: Version 20.10+ and 2.0+
- **Git**: For cloning the repository
- **At least 8GB RAM**: For running all services
- **At least 10GB free disk space**: For models and data

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/ai-meeting-intelligence-platform.git
cd ai-meeting-intelligence-platform
```

### 2. Start the Platform

```bash
# Start all services
docker-compose up -d

# Or start with development tools
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

### 3. Access the Application

- **Frontend**: http://localhost:4200
- **Backend API**: http://localhost:8080
- **API Documentation**: http://localhost:8080/docs
- **Database Admin**: http://localhost:5050 (pgAdmin)
- **Celery Monitor**: http://localhost:5555 (Flower)

### 4. Initialize Ollama Models

```bash
# Pull a language model for AI processing
docker exec meeting-ollama ollama pull llama2:7b

# Or use a smaller model for faster processing
docker exec meeting-ollama ollama pull llama2:7b-chat
```

### 5. Test the Platform

1. Open http://localhost:4200 in your browser
2. Upload an audio file (MP3, WAV, M4A)
3. Wait for processing to complete
4. View transcription and AI insights

## 🔧 Manual Setup

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://meeting_user:meeting_secure_password_2025@localhost:5432/meeting_intelligence"
export REDIS_URL="redis://localhost:6379/0"

# Run the application
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run serve:dev
```

### Database Setup

```bash
# Start PostgreSQL
docker run -d \
  --name meeting-postgres \
  -e POSTGRES_DB=meeting_intelligence \
  -e POSTGRES_USER=meeting_user \
  -e POSTGRES_PASSWORD=meeting_secure_password_2025 \
  -p 5432:5432 \
  postgres:15-alpine

# Start Redis
docker run -d \
  --name meeting-redis \
  -p 6379:6379 \
  redis:7-alpine
```

## 🧪 Development

### Project Structure

```
ai-meeting-intelligence-platform/
├── backend/                 # FastAPI backend
│   ├── app.py              # Main application
│   ├── models.py           # Database models
│   ├── tasks.py            # Celery tasks
│   ├── schemas.py          # Pydantic schemas
│   └── requirements.txt    # Python dependencies
├── frontend/               # Angular frontend
│   ├── src/                # Source code
│   ├── package.json        # Node dependencies
│   └── Dockerfile          # Frontend container
├── llm_service/            # LLM wrapper service
├── whisper_service/        # Speech-to-text service
├── infrastructure/         # Configuration files
├── database/               # Database scripts
└── docker-compose.yml      # Service orchestration
```

### Development Commands

```bash
# Start development environment
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f celery-worker

# Restart services
docker-compose restart backend
docker-compose restart frontend

# Access containers
docker exec -it meeting-backend bash
docker exec -it meeting-frontend sh

# Run tests
docker exec meeting-backend python -m pytest
docker exec meeting-frontend npm test
```

### Adding New Features

1. **Backend API**: Add endpoints in `backend/app.py`
2. **Database Models**: Define in `backend/models.py`
3. **Background Tasks**: Create in `backend/tasks.py`
4. **Frontend Components**: Add in `frontend/src/app/`
5. **AI Prompts**: Modify in `llm_service/prompts.py`

## 🐳 Docker Deployment

### Production Deployment

```bash
# Build production images
docker-compose -f docker-compose.yml build

# Start production services
docker-compose -f docker-compose.yml up -d

# Scale services
docker-compose up -d --scale celery-worker=3
```

### Environment Configuration

Create a `.env` file for production:

```env
# Production Environment Variables
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0
SECRET_KEY=your-production-secret-key
DEBUG=false
LOG_LEVEL=INFO
```

### Health Checks

```bash
# Check service health
curl http://localhost:8080/health

# Check Celery workers
curl http://localhost:5555/api/workers

# Check database connectivity
docker exec meeting-backend python -c "from db import check_db_health; print(check_db_health())"
```

## 📊 Monitoring

### Available Monitoring Tools

- **Flower**: Celery task monitoring at http://localhost:5555
- **pgAdmin**: Database management at http://localhost:5050
- **Application Logs**: `docker-compose logs -f [service-name]`

### Key Metrics to Monitor

- **Task Queue Length**: Number of pending Celery tasks
- **Processing Time**: Average time for transcription and analysis
- **Error Rates**: Failed task percentage
- **Resource Usage**: CPU, memory, and disk usage
- **Database Performance**: Query response times

### Log Analysis

```bash
# View recent errors
docker-compose logs --tail=100 backend | grep ERROR

# Monitor task processing
docker-compose logs -f celery-worker | grep "Task completed"

# Check service startup
docker-compose logs --tail=50 | grep "started"
```

## 🔒 Security

### Security Features

- **JWT Authentication**: Token-based authentication
- **CORS Protection**: Cross-origin request validation
- **Input Validation**: Pydantic schema validation
- **SQL Injection Protection**: SQLAlchemy ORM
- **File Upload Security**: File type and size validation

### Security Best Practices

1. **Change Default Passwords**: Update database and admin passwords
2. **Use HTTPS**: Configure SSL certificates for production
3. **Network Security**: Use firewalls and VPNs
4. **Regular Updates**: Keep dependencies updated
5. **Access Control**: Implement role-based access control

## 🔧 Configuration

### Service Configuration

Each service can be configured through environment variables:

```bash
# Backend Configuration
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0
SECRET_KEY=your-secret-key
DEBUG=true
LOG_LEVEL=INFO

# AI Services Configuration
WHISPER_GRPC_URL=whisper-service:50051
LLM_SERVICE_URL=http://llm-service:8001
OLLAMA_URL=http://ollama:11434

# Frontend Configuration
API_BASE_URL=http://localhost:8080
NODE_ENV=development
```

### Performance Tuning

```bash
# Increase Celery workers
docker-compose up -d --scale celery-worker=4

# Adjust Redis memory
# Edit infrastructure/redis/redis.conf
maxmemory 512mb

# Optimize database
# Edit database/init/01-init.sql
```

## 📖 API Documentation

### Core Endpoints

- `POST /api/upload` - Upload audio file for processing
- `GET /api/status/{task_id}` - Get processing status
- `GET /api/meetings` - List all meetings
- `GET /api/meetings/{id}` - Get meeting details
- `GET /api/search` - Search meetings and content

### Example API Usage

```bash
# Upload a meeting
curl -X POST "http://localhost:8080/api/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@meeting.mp3" \
  -F "title=Weekly Standup" \
  -F "meeting_type=standup"

# Check processing status
curl "http://localhost:8080/api/status/{task_id}"

# Search meetings
curl "http://localhost:8080/api/search?query=budget discussion"
```

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `npm test` and `python -m pytest`
6. Commit your changes: `git commit -am 'Add new feature'`
7. Push to the branch: `git push origin feature/new-feature`
8. Submit a pull request

### Code Style

- **Python**: Follow PEP 8 guidelines
- **TypeScript**: Use ESLint and Prettier
- **HTML/CSS**: Follow Angular style guide
- **Docker**: Use multi-stage builds and best practices

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**Service won't start:**
```bash
# Check if ports are available
netstat -tulpn | grep :8080
netstat -tulpn | grep :4200

# Check Docker logs
docker-compose logs [service-name]
```

**Database connection issues:**
```bash
# Check database status
docker exec meeting-postgres pg_isready -U meeting_user

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

**AI processing fails:**
```bash
# Check Ollama models
docker exec meeting-ollama ollama list

# Pull required models
docker exec meeting-ollama ollama pull llama2:7b
```

**Memory issues:**
```bash
# Check resource usage
docker stats

# Increase Docker memory limit
# Edit Docker Desktop settings
```

### Getting Help

- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check the `/docs` folder
- **Community**: Join our Discord server

---

**Made with ❤️ by the AI Meeting Intelligence Team**

