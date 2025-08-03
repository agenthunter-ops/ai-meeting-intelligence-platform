# AI Meeting Intelligence Platform - Part 2: Backend Architecture Deep Dive

## Table of Contents
1. [FastAPI Backend Architecture](#fastapi-backend-architecture)
2. [Database Layer Deep Dive](#database-layer-deep-dive)
3. [API Endpoints and Request Flow](#api-endpoints-and-request-flow)
4. [Authentication and Security](#authentication-and-security)
5. [Error Handling and Validation](#error-handling-and-validation)
6. [Background Task Processing](#background-task-processing)
7. [File Upload and Storage](#file-upload-and-storage)
8. [Health Monitoring and Observability](#health-monitoring-and-observability)

---

## FastAPI Backend Architecture

### Core Application Structure

The backend is built using FastAPI, a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints. Let's examine the complete architecture:

```python
# backend/app.py - Core Application Structure
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional, List
import os
import uuid
import logging

# Import our custom modules
from .db import get_db, init_db
from .models import Meeting, Task, Segment, Base
from .schemas import UploadResponse, TaskStatus, UploadRequest
from .tasks import process_audio_file
```

#### Application Initialization Flow

When the FastAPI application starts, it goes through several initialization phases:

1. **Environment Configuration**: The app reads environment variables to configure database connections, Redis URLs, and service endpoints.

```python
# Environment variables loaded during startup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./meeting_intelligence.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
WHISPER_GRPC_URL = os.getenv("WHISPER_GRPC_URL", "whisper-service:50051")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8001")
```

2. **Database Initialization**: The `init_db()` function is called to create tables and apply optimizations.

3. **CORS Configuration**: Cross-Origin Resource Sharing is configured to allow frontend requests.

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://frontend:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### Request Processing Pipeline

Every HTTP request goes through FastAPI's request processing pipeline:

1. **Request Parsing**: FastAPI automatically parses incoming requests based on endpoint definitions
2. **Validation**: Pydantic models validate request data and convert types
3. **Dependency Injection**: Database sessions and other dependencies are injected
4. **Business Logic**: The endpoint function executes the business logic
5. **Response Serialization**: The response is automatically serialized to JSON
6. **Error Handling**: Any exceptions are caught and converted to appropriate HTTP responses

### Dependency Injection System

FastAPI uses a sophisticated dependency injection system that our application leverages extensively:

```python
# Database dependency
async def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication dependency (when implemented)
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Dependency to get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    # Token validation logic here
    return user
```

---

## Database Layer Deep Dive

### SQLAlchemy ORM Architecture

Our database layer is built using SQLAlchemy, providing a powerful Object-Relational Mapping (ORM) system. The architecture follows the Repository pattern with clear separation of concerns.

#### Database Configuration Strategy

The database configuration handles both development (SQLite) and production (PostgreSQL) environments seamlessly:

```python
# backend/db.py - Database Configuration
import os
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Database configuration that adapts to environment
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./meeting_intelligence.db")

def create_database_engine():
    """Create database engine with environment-specific optimizations"""
    if DATABASE_URL.startswith("sqlite"):
        # SQLite-specific configuration
        engine = create_engine(
            DATABASE_URL,
            pool_size=20,
            max_overflow=0,
            connect_args={
                "check_same_thread": False,
                "timeout": 20,
                "isolation_level": None,
            },
            echo=False
        )
    else:
        # PostgreSQL configuration
        engine = create_engine(
            DATABASE_URL,
            pool_size=20,
            max_overflow=0,
            pool_pre_ping=True,
            echo=False
        )
    return engine

engine = create_database_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

#### Database Models Architecture

Our database models follow a hierarchical structure that mirrors the meeting processing workflow:

```python
# backend/models.py - Complete Model Definitions

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class Meeting(Base):
    """Core meeting entity - represents an uploaded meeting"""
    __tablename__ = "meetings"
    
    # Primary identifiers
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    
    # File metadata
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    mime_type = Column(String(100))
    duration = Column(Float)  # Duration in seconds
    
    # Processing metadata
    status = Column(String(50), default="uploaded")  # uploaded, processing, completed, failed
    upload_time = Column(DateTime, server_default=func.now())
    processing_start_time = Column(DateTime)
    processing_end_time = Column(DateTime)
    
    # Meeting metadata
    meeting_type = Column(String(100))  # standup, review, planning, etc.
    meeting_date = Column(DateTime)
    
    # Relationships
    tasks = relationship("Task", back_populates="meeting", cascade="all, delete-orphan")
    segments = relationship("Segment", back_populates="meeting", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Meeting(id={self.id}, title={self.title}, status={self.status})>"

class Task(Base):
    """Represents a background processing task"""
    __tablename__ = "tasks"
    
    # Primary identifiers
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    meeting_id = Column(String, ForeignKey("meetings.id"), nullable=False)
    
    # Task metadata
    task_type = Column(String(100), nullable=False)  # transcription, analysis, etc.
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    progress = Column(Integer, default=0)  # 0-100
    
    # Timing information
    created_at = Column(DateTime, server_default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Results and errors
    result = Column(JSON)  # Store task results as JSON
    error_message = Column(Text)
    
    # Relationships
    meeting = relationship("Meeting", back_populates="tasks")
    
    def __repr__(self):
        return f"<Task(id={self.id}, type={self.task_type}, status={self.status})>"

class Segment(Base):
    """Represents a segment of transcribed audio with speaker information"""
    __tablename__ = "segments"
    
    # Primary identifiers
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    meeting_id = Column(String, ForeignKey("meetings.id"), nullable=False)
    
    # Timing information
    start_time = Column(Float, nullable=False)  # Start time in seconds
    end_time = Column(Float, nullable=False)    # End time in seconds
    
    # Speaker information
    speaker_id = Column(String)  # Speaker identifier from diarization
    speaker_name = Column(String(255))  # Human-readable speaker name
    
    # Content
    text = Column(Text, nullable=False)  # Transcribed text
    confidence = Column(Float)  # Transcription confidence score
    
    # Analysis results
    sentiment = Column(String(50))  # positive, negative, neutral
    sentiment_score = Column(Float)  # -1.0 to 1.0
    emotion = Column(String(50))  # joy, anger, sadness, etc.
    emotion_confidence = Column(Float)
    
    # Language and quality
    language = Column(String(10), default="en")
    audio_quality = Column(Float)  # Audio quality score
    
    # Relationships
    meeting = relationship("Meeting", back_populates="segments")
    
    def __repr__(self):
        return f"<Segment(id={self.id}, speaker={self.speaker_name}, text={self.text[:50]}...)>"
```

#### Database Initialization and Migration Strategy

The database initialization process handles both schema creation and data migration:

```python
def init_db():
    """Initialize database with optimizations"""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # Apply database-specific optimizations
        if DATABASE_URL.startswith("sqlite"):
            apply_sqlite_optimizations()
        else:
            apply_postgresql_optimizations()
        
        print("✅ Database initialized successfully")
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise

def apply_sqlite_optimizations():
    """Apply SQLite-specific performance optimizations"""
    with engine.connect() as conn:
        # Enable Write-Ahead Logging for better concurrent access
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        
        # Increase cache size to 64MB
        conn.exec_driver_sql("PRAGMA cache_size=-64000")
        
        # Enable foreign key constraints
        conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        
        # Set synchronous mode for balance of safety and performance
        conn.exec_driver_sql("PRAGMA synchronous=NORMAL")
        
        # Set busy timeout for concurrent access
        conn.exec_driver_sql("PRAGMA busy_timeout=30000")

def apply_postgresql_optimizations():
    """Apply PostgreSQL-specific optimizations"""
    with engine.connect() as conn:
        # Set connection-level optimizations
        conn.exec_driver_sql("SET statement_timeout = '300s'")
        conn.exec_driver_sql("SET lock_timeout = '30s'")
```

### Query Optimization Strategies

Our database layer implements several query optimization strategies:

1. **Lazy Loading**: Relationships are loaded only when accessed
2. **Eager Loading**: Critical relationships can be preloaded using `joinedload()`
3. **Connection Pooling**: Reuses database connections for better performance
4. **Index Strategy**: Strategic indexes on frequently queried columns

```python
# Example of optimized queries
def get_meeting_with_segments(db: Session, meeting_id: str):
    """Get meeting with all segments in a single query"""
    return db.query(Meeting)\
        .options(joinedload(Meeting.segments))\
        .filter(Meeting.id == meeting_id)\
        .first()

def get_recent_meetings(db: Session, limit: int = 10):
    """Get recent meetings with optimized query"""
    return db.query(Meeting)\
        .order_by(Meeting.upload_time.desc())\
        .limit(limit)\
        .all()
```

---

## API Endpoints and Request Flow

### REST API Design Principles

Our API follows REST principles with clear resource-based URLs and appropriate HTTP methods:

```
GET    /health              - Health check
POST   /api/upload          - Upload audio file
GET    /api/status/{id}     - Get task status
GET    /api/meetings        - List meetings
GET    /api/meetings/{id}   - Get specific meeting
GET    /api/search          - Search meetings
DELETE /api/meetings/{id}   - Delete meeting
```

### Upload Endpoint Deep Dive

The upload endpoint is the most critical part of our API. Let's examine its complete implementation:

```python
@app.post("/api/upload", response_model=UploadResponse)
async def upload_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    meeting_type: Optional[str] = Form("general"),
    meeting_date: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload and process an audio/video file for meeting analysis
    
    This endpoint handles the complete upload workflow:
    1. File validation and security checks
    2. File storage with unique naming
    3. Database record creation
    4. Background task initiation
    5. Response with task tracking ID
    """
    
    # Step 1: File validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    allowed_extensions = {'.mp3', '.wav', '.m4a', '.mp4', '.avi', '.mov'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}"
        )
    
    # Validate file size (max 500MB)
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    file_content = await file.read()
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 500MB"
        )
    
    # Step 2: Generate unique identifiers
    meeting_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    
    # Step 3: Save file to storage
    upload_dir = Path("media/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique filename to prevent conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{meeting_id}_{timestamp}{file_ext}"
    file_path = upload_dir / safe_filename
    
    # Write file to disk
    with open(file_path, "wb") as buffer:
        buffer.write(file_content)
    
    # Step 4: Create database records
    try:
        # Create meeting record
        meeting = Meeting(
            id=meeting_id,
            title=title or f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            filename=file.filename,
            file_path=str(file_path),
            file_size=len(file_content),
            mime_type=file.content_type,
            meeting_type=meeting_type,
            status="uploaded"
        )
        db.add(meeting)
        
        # Create task record
        task = Task(
            id=task_id,
            meeting_id=meeting_id,
            task_type="complete_processing",
            status="pending",
            progress=0
        )
        db.add(task)
        
        # Commit transaction
        db.commit()
        
    except Exception as e:
        db.rollback()
        # Clean up uploaded file on database error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    # Step 5: Initiate background processing
    background_tasks.add_task(
        process_audio_file,
        task_id=task_id,
        meeting_id=meeting_id,
        file_path=str(file_path)
    )
    
    # Step 6: Return response
    return UploadResponse(
        task_id=task_id,
        meeting_id=meeting_id,
        message="File uploaded successfully. Processing started.",
        status="processing"
    )
```

### Status Endpoint Implementation

The status endpoint provides real-time updates on processing progress:

```python
@app.get("/api/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str, db: Session = Depends(get_db)):
    """
    Get the current status of a processing task
    
    Returns detailed information about task progress including:
    - Current status (pending, running, completed, failed)
    - Progress percentage (0-100)
    - Any error messages
    - Processing results when completed
    """
    
    # Query task from database
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Calculate processing time
    processing_time = None
    if task.started_at and task.completed_at:
        processing_time = (task.completed_at - task.started_at).total_seconds()
    elif task.started_at:
        processing_time = (datetime.now() - task.started_at).total_seconds()
    
    # Get meeting information
    meeting = db.query(Meeting).filter(Meeting.id == task.meeting_id).first()
    
    return TaskStatus(
        task_id=task.id,
        meeting_id=task.meeting_id,
        status=task.status,
        progress=task.progress,
        error_message=task.error_message,
        result=task.result,
        processing_time=processing_time,
        meeting_title=meeting.title if meeting else None,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at
    )
```

### Search Endpoint with Vector Search

The search endpoint provides powerful semantic search capabilities:

```python
@app.get("/api/search")
async def search_meetings(
    query: str,
    limit: int = 10,
    offset: int = 0,
    meeting_type: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db: Session = Depends(get_db)
):
    """
    Search meetings using both text and semantic search
    
    Supports:
    - Full-text search across transcripts
    - Semantic similarity search using embeddings
    - Filtering by meeting type and date range
    - Pagination with limit/offset
    """
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    # Build base query
    query_base = db.query(Meeting).filter(Meeting.status == "completed")
    
    # Apply filters
    if meeting_type:
        query_base = query_base.filter(Meeting.meeting_type == meeting_type)
    if date_from:
        query_base = query_base.filter(Meeting.meeting_date >= date_from)
    if date_to:
        query_base = query_base.filter(Meeting.meeting_date <= date_to)
    
    # Perform text search on segments
    text_results = query_base.join(Segment)\
        .filter(Segment.text.contains(query))\
        .offset(offset)\
        .limit(limit)\
        .all()
    
    # TODO: Implement semantic search using ChromaDB
    # This would involve:
    # 1. Converting query to embedding
    # 2. Searching ChromaDB for similar embeddings
    # 3. Retrieving corresponding meetings
    # 4. Combining with text search results
    
    return {
        "query": query,
        "total_results": len(text_results),
        "meetings": [
            {
                "id": meeting.id,
                "title": meeting.title,
                "meeting_type": meeting.meeting_type,
                "upload_time": meeting.upload_time,
                "duration": meeting.duration,
                "segments_count": len(meeting.segments)
            }
            for meeting in text_results
        ]
    }
```

---

## Authentication and Security

### Security Architecture Overview

While our current implementation focuses on core functionality, the architecture is designed to support enterprise-grade security features:

```python
# backend/auth.py - Authentication Framework (Future Implementation)

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class AuthManager:
    """Handles all authentication and authorization logic"""
    
    def __init__(self):
        self.pwd_context = pwd_context
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password for storage"""
        return self.pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str):
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials"
                )
            return username
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
```

### Input Validation and Sanitization

FastAPI provides robust input validation through Pydantic models:

```python
# backend/schemas.py - Request/Response Validation

from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import re

class UploadRequest(BaseModel):
    """Validation schema for file upload requests"""
    title: Optional[str] = Field(None, max_length=255, description="Meeting title")
    meeting_type: Optional[str] = Field("general", max_length=100, description="Type of meeting")
    meeting_date: Optional[datetime] = Field(None, description="Meeting date")
    
    @validator('title')
    def validate_title(cls, v):
        if v is not None:
            # Remove potentially dangerous characters
            v = re.sub(r'[<>:"/\\|?*]', '', v).strip()
            if not v:
                raise ValueError('Title cannot be empty after sanitization')
        return v
    
    @validator('meeting_type')
    def validate_meeting_type(cls, v):
        allowed_types = ['general', 'standup', 'review', 'planning', 'retrospective']
        if v not in allowed_types:
            raise ValueError(f'Meeting type must be one of: {allowed_types}')
        return v

class TaskStatus(BaseModel):
    """Response schema for task status"""
    task_id: str
    meeting_id: str
    status: str = Field(..., regex='^(pending|running|completed|failed)$')
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    meeting_title: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True
```

### File Upload Security

File upload security is critical for preventing malicious attacks:

```python
import magic
from pathlib import Path

class FileSecurityValidator:
    """Handles file security validation"""
    
    ALLOWED_MIME_TYPES = {
        'audio/mpeg',      # .mp3
        'audio/wav',       # .wav
        'audio/x-wav',     # .wav alternative
        'audio/mp4',       # .m4a
        'video/mp4',       # .mp4
        'video/quicktime', # .mov
        'video/x-msvideo'  # .avi
    }
    
    ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.mp4', '.mov', '.avi'}
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    
    @classmethod
    def validate_file(cls, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Comprehensive file validation"""
        
        # Check file size
        if len(file_content) > cls.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {cls.MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in cls.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension: {file_ext}"
            )
        
        # Check MIME type using python-magic
        try:
            mime_type = magic.from_buffer(file_content, mime=True)
            if mime_type not in cls.ALLOWED_MIME_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type detected: {mime_type}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Could not determine file type"
            )
        
        # Additional security checks
        cls._check_file_headers(file_content)
        
        return {
            "mime_type": mime_type,
            "extension": file_ext,
            "size": len(file_content),
            "is_valid": True
        }
    
    @classmethod
    def _check_file_headers(cls, file_content: bytes):
        """Check file headers for known malicious patterns"""
        
        # Check for executable headers
        malicious_headers = [
            b'\x4d\x5a',  # PE/DOS executable
            b'\x7f\x45\x4c\x46',  # ELF executable
            b'\xca\xfe\xba\xbe',  # Mach-O binary
        ]
        
        for header in malicious_headers:
            if file_content.startswith(header):
                raise HTTPException(
                    status_code=400,
                    detail="Potentially malicious file detected"
                )
```

---

## Error Handling and Validation

### Comprehensive Error Handling Strategy

Our error handling strategy provides clear, actionable error messages while maintaining security:

```python
# backend/exceptions.py - Custom Exception Classes

class MeetingIntelligenceException(Exception):
    """Base exception for all application errors"""
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class FileProcessingError(MeetingIntelligenceException):
    """Raised when file processing fails"""
    pass

class TranscriptionError(MeetingIntelligenceException):
    """Raised when transcription service fails"""
    pass

class DatabaseError(MeetingIntelligenceException):
    """Raised when database operations fail"""
    pass

class ServiceUnavailableError(MeetingIntelligenceException):
    """Raised when external services are unavailable"""
    pass

# Global exception handlers
@app.exception_handler(MeetingIntelligenceException)
async def meeting_intelligence_exception_handler(request: Request, exc: MeetingIntelligenceException):
    """Handle custom application exceptions"""
    return JSONResponse(
        status_code=400,
        content={
            "error": True,
            "message": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4())
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with detailed logging"""
    
    # Log error details
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    
    # Log detailed error information
    logger.exception(f"Unexpected error in {request.url.path}: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "An internal server error occurred",
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4())
        }
    )
```

### Request Validation Pipeline

FastAPI's validation pipeline ensures data integrity at every level:

```python
# Example of comprehensive validation
@app.post("/api/meetings/{meeting_id}/segments")
async def create_segment(
    meeting_id: str = Path(..., regex=r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
    segment_data: SegmentCreate = Body(...),
    db: Session = Depends(get_db)
):
    """Create a new segment with comprehensive validation"""
    
    # Validate meeting exists
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    # Validate business logic
    if segment_data.start_time >= segment_data.end_time:
        raise HTTPException(
            status_code=400, 
            detail="Start time must be less than end time"
        )
    
    if meeting.duration and segment_data.end_time > meeting.duration:
        raise HTTPException(
            status_code=400,
            detail="Segment end time exceeds meeting duration"
        )
    
    # Create segment
    segment = Segment(**segment_data.dict(), meeting_id=meeting_id)
    db.add(segment)
    db.commit()
    db.refresh(segment)
    
    return segment
```

---

## Background Task Processing

### Celery Integration Architecture

Our background task processing uses Celery for reliable, scalable task execution:

```python
# backend/celery_config.py - Celery Configuration

from celery import Celery
import os

# Configure Celery with Redis as broker
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "meeting_intelligence",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["backend.tasks"]
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        'backend.tasks.process_audio_file': {'queue': 'audio_processing'},
        'backend.tasks.transcribe_audio': {'queue': 'transcription'},
        'backend.tasks.analyze_sentiment': {'queue': 'analysis'},
    },
    
    # Task execution settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    
    # Result backend settings
    result_expires=3600,  # 1 hour
    
    # Task timeouts
    task_soft_time_limit=1800,  # 30 minutes soft limit
    task_time_limit=3600,       # 60 minutes hard limit
)
```

### Task Implementation Deep Dive

The main processing task orchestrates the entire meeting analysis pipeline:

```python
# backend/tasks.py - Background Task Implementation

from celery import current_app
from celery.utils.log import get_task_logger
from sqlalchemy.orm import sessionmaker
from typing import Dict, Any
import subprocess
import requests
import json

logger = get_task_logger(__name__)

@celery_app.task(bind=True, name='backend.tasks.process_audio_file')
def process_audio_file(self, task_id: str, meeting_id: str, file_path: str) -> Dict[str, Any]:
    """
    Main task that orchestrates the complete audio processing pipeline
    
    Pipeline stages:
    1. Audio preprocessing and validation
    2. Speaker diarization
    3. Speech-to-text transcription
    4. Sentiment and emotion analysis
    5. Key point extraction
    6. Summary generation
    7. Results storage
    """
    
    # Initialize database session
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    try:
        # Update task status to running
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise Exception(f"Task {task_id} not found")
        
        task.status = "running"
        task.started_at = datetime.now()
        task.progress = 0
        db.commit()
        
        logger.info(f"Starting processing for task {task_id}, meeting {meeting_id}")
        
        # Stage 1: Audio preprocessing (10% progress)
        self.update_state(state='PROGRESS', meta={'progress': 10, 'stage': 'preprocessing'})
        audio_info = preprocess_audio(file_path)
        update_task_progress(db, task_id, 10, "Audio preprocessing completed")
        
        # Stage 2: Speaker diarization (30% progress)
        self.update_state(state='PROGRESS', meta={'progress': 30, 'stage': 'diarization'})
        diarization_result = perform_speaker_diarization(file_path)
        update_task_progress(db, task_id, 30, "Speaker diarization completed")
        
        # Stage 3: Speech-to-text transcription (60% progress)
        self.update_state(state='PROGRESS', meta={'progress': 60, 'stage': 'transcription'})
        transcription_result = transcribe_audio_segments(file_path, diarization_result)
        update_task_progress(db, task_id, 60, "Transcription completed")
        
        # Stage 4: Sentiment and emotion analysis (80% progress)
        self.update_state(state='PROGRESS', meta={'progress': 80, 'stage': 'analysis'})
        analysis_result = analyze_transcript_sentiment(transcription_result)
        update_task_progress(db, task_id, 80, "Sentiment analysis completed")
        
        # Stage 5: Summary and insights generation (90% progress)
        self.update_state(state='PROGRESS', meta={'progress': 90, 'stage': 'summarization'})
        summary_result = generate_meeting_summary(transcription_result, analysis_result)
        update_task_progress(db, task_id, 90, "Summary generation completed")
        
        # Stage 6: Store results in database (100% progress)
        store_processing_results(
            db, meeting_id, audio_info, transcription_result, 
            analysis_result, summary_result
        )
        
        # Mark task as completed
        task.status = "completed"
        task.progress = 100
        task.completed_at = datetime.now()
        task.result = {
            "audio_duration": audio_info.get("duration"),
            "speaker_count": len(diarization_result.get("speakers", [])),
            "transcript_length": len(transcription_result.get("text", "")),
            "sentiment_summary": analysis_result.get("overall_sentiment"),
            "summary": summary_result.get("summary")
        }
        db.commit()
        
        logger.info(f"Processing completed for task {task_id}")
        
        return {
            "status": "completed",
            "message": "Processing completed successfully",
            "result": task.result
        }
        
    except Exception as e:
        # Handle task failure
        logger.error(f"Task {task_id} failed: {str(e)}")
        
        task.status = "failed"
        task.error_message = str(e)
        task.completed_at = datetime.now()
        db.commit()
        
        # Re-raise exception for Celery
        raise self.retry(exc=e, countdown=60, max_retries=3)
    
    finally:
        db.close()

def preprocess_audio(file_path: str) -> Dict[str, Any]:
    """Preprocess audio file and extract metadata"""
    
    try:
        # Use ffprobe to get audio information
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        audio_info = json.loads(result.stdout)
        
        # Extract relevant information
        format_info = audio_info.get('format', {})
        stream_info = audio_info.get('streams', [{}])[0]
        
        return {
            "duration": float(format_info.get('duration', 0)),
            "bit_rate": int(format_info.get('bit_rate', 0)),
            "sample_rate": int(stream_info.get('sample_rate', 0)),
            "channels": int(stream_info.get('channels', 0)),
            "codec": stream_info.get('codec_name', 'unknown')
        }
        
    except subprocess.CalledProcessError as e:
        raise FileProcessingError(f"Failed to analyze audio file: {e}")
    except Exception as e:
        raise FileProcessingError(f"Audio preprocessing failed: {e}")

def perform_speaker_diarization(file_path: str) -> Dict[str, Any]:
    """Perform speaker diarization using Whisper service"""
    
    try:
        # Call Whisper service for diarization
        whisper_url = f"{WHISPER_GRPC_URL}/diarize"
        
        with open(file_path, 'rb') as audio_file:
            files = {'audio': audio_file}
            response = requests.post(whisper_url, files=files, timeout=1800)
        
        if response.status_code != 200:
            raise ServiceUnavailableError(f"Whisper service error: {response.text}")
        
        diarization_data = response.json()
        
        # Process and validate diarization results
        speakers = diarization_data.get('speakers', [])
        segments = diarization_data.get('segments', [])
        
        return {
            "speakers": speakers,
            "segments": segments,
            "speaker_count": len(speakers)
        }
        
    except requests.RequestException as e:
        raise ServiceUnavailableError(f"Failed to connect to Whisper service: {e}")
    except Exception as e:
        raise TranscriptionError(f"Speaker diarization failed: {e}")

def transcribe_audio_segments(file_path: str, diarization_result: Dict) -> Dict[str, Any]:
    """Transcribe audio with speaker information"""
    
    try:
        # Call Whisper service for transcription
        whisper_url = f"{WHISPER_GRPC_URL}/transcribe"
        
        payload = {
            "audio_path": file_path,
            "diarization_data": diarization_result
        }
        
        response = requests.post(
            whisper_url, 
            json=payload, 
            headers={'Content-Type': 'application/json'},
            timeout=1800
        )
        
        if response.status_code != 200:
            raise ServiceUnavailableError(f"Transcription service error: {response.text}")
        
        transcription_data = response.json()
        
        return {
            "text": transcription_data.get('text', ''),
            "segments": transcription_data.get('segments', []),
            "language": transcription_data.get('language', 'en'),
            "confidence": transcription_data.get('confidence', 0.0)
        }
        
    except requests.RequestException as e:
        raise ServiceUnavailableError(f"Failed to connect to transcription service: {e}")
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {e}")

def analyze_transcript_sentiment(transcription_result: Dict) -> Dict[str, Any]:
    """Analyze sentiment and emotions in the transcript"""
    
    try:
        # Call LLM service for sentiment analysis
        llm_url = f"{LLM_SERVICE_URL}/analyze_sentiment"
        
        payload = {
            "text": transcription_result.get('text', ''),
            "segments": transcription_result.get('segments', [])
        }
        
        response = requests.post(
            llm_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=600
        )
        
        if response.status_code != 200:
            raise ServiceUnavailableError(f"LLM service error: {response.text}")
        
        analysis_data = response.json()
        
        return {
            "overall_sentiment": analysis_data.get('overall_sentiment', 'neutral'),
            "sentiment_score": analysis_data.get('sentiment_score', 0.0),
            "emotions": analysis_data.get('emotions', []),
            "segment_sentiments": analysis_data.get('segment_sentiments', [])
        }
        
    except requests.RequestException as e:
        raise ServiceUnavailableError(f"Failed to connect to LLM service: {e}")
    except Exception as e:
        raise Exception(f"Sentiment analysis failed: {e}")

def generate_meeting_summary(transcription_result: Dict, analysis_result: Dict) -> Dict[str, Any]:
    """Generate meeting summary and key insights"""
    
    try:
        # Call LLM service for summary generation
        llm_url = f"{LLM_SERVICE_URL}/generate_summary"
        
        payload = {
            "transcript": transcription_result.get('text', ''),
            "sentiment_data": analysis_result,
            "segments": transcription_result.get('segments', [])
        }
        
        response = requests.post(
            llm_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=600
        )
        
        if response.status_code != 200:
            raise ServiceUnavailableError(f"LLM service error: {response.text}")
        
        summary_data = response.json()
        
        return {
            "summary": summary_data.get('summary', ''),
            "key_points": summary_data.get('key_points', []),
            "action_items": summary_data.get('action_items', []),
            "decisions": summary_data.get('decisions', []),
            "topics": summary_data.get('topics', [])
        }
        
    except requests.RequestException as e:
        raise ServiceUnavailableError(f"Failed to connect to LLM service: {e}")
    except Exception as e:
        raise Exception(f"Summary generation failed: {e}")

def store_processing_results(
    db: Session, 
    meeting_id: str, 
    audio_info: Dict, 
    transcription_result: Dict, 
    analysis_result: Dict, 
    summary_result: Dict
):
    """Store all processing results in the database"""
    
    try:
        # Update meeting with processed information
        meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        if meeting:
            meeting.duration = audio_info.get('duration')
            meeting.status = 'completed'
            meeting.processing_end_time = datetime.now()
        
        # Store transcript segments
        segments = transcription_result.get('segments', [])
        for i, segment_data in enumerate(segments):
            segment = Segment(
                meeting_id=meeting_id,
                start_time=segment_data.get('start', 0),
                end_time=segment_data.get('end', 0),
                speaker_id=segment_data.get('speaker', f'speaker_{i}'),
                text=segment_data.get('text', ''),
                confidence=segment_data.get('confidence', 0.0),
                sentiment=segment_data.get('sentiment', 'neutral'),
                sentiment_score=segment_data.get('sentiment_score', 0.0)
            )
            db.add(segment)
        
        db.commit()
        
    except Exception as e:
        db.rollback()
        raise DatabaseError(f"Failed to store processing results: {e}")

def update_task_progress(db: Session, task_id: str, progress: int, message: str):
    """Update task progress in database"""
    
    task = db.query(Task).filter(Task.id == task_id).first()
    if task:
        task.progress = progress
        db.commit()
```

---

## File Upload and Storage

### Storage Architecture

Our file storage system handles both temporary and permanent storage with proper organization:

```python
# backend/storage.py - File Storage Management

from pathlib import Path
import shutil
import hashlib
from typing import Dict, Optional
import os

class StorageManager:
    """Manages file storage operations"""
    
    def __init__(self):
        self.base_dir = Path("media")
        self.uploads_dir = self.base_dir / "uploads"
        self.processed_dir = self.base_dir / "processed"
        self.temp_dir = self.base_dir / "temp"
        
        # Ensure directories exist
        for directory in [self.uploads_dir, self.processed_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def store_upload(self, file_content: bytes, filename: str, meeting_id: str) -> Dict[str, str]:
        """Store uploaded file with organized structure"""
        
        # Create meeting-specific directory
        meeting_dir = self.uploads_dir / meeting_id
        meeting_dir.mkdir(exist_ok=True)
        
        # Generate secure filename
        file_ext = Path(filename).suffix.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        secure_filename = f"original_{timestamp}{file_ext}"
        
        # Store file
        file_path = meeting_dir / secure_filename
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Generate file hash for integrity checking
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        return {
            "file_path": str(file_path),
            "filename": secure_filename,
            "directory": str(meeting_dir),
            "size": len(file_content),
            "hash": file_hash
        }
    
    def create_working_directory(self, meeting_id: str) -> Path:
        """Create temporary working directory for processing"""
        
        working_dir = self.temp_dir / meeting_id
        working_dir.mkdir(exist_ok=True)
        return working_dir
    
    def store_processed_results(self, meeting_id: str, results: Dict) -> Dict[str, str]:
        """Store processing results and artifacts"""
        
        # Create processed directory for meeting
        processed_dir = self.processed_dir / meeting_id
        processed_dir.mkdir(exist_ok=True)
        
        stored_files = {}
        
        # Store transcript
        if 'transcript' in results:
            transcript_path = processed_dir / "transcript.json"
            with open(transcript_path, 'w') as f:
                json.dump(results['transcript'], f, indent=2)
            stored_files['transcript'] = str(transcript_path)
        
        # Store analysis results
        if 'analysis' in results:
            analysis_path = processed_dir / "analysis.json"
            with open(analysis_path, 'w') as f:
                json.dump(results['analysis'], f, indent=2)
            stored_files['analysis'] = str(analysis_path)
        
        # Store audio segments if available
        if 'segments' in results:
            segments_dir = processed_dir / "segments"
            segments_dir.mkdir(exist_ok=True)
            # Store individual audio segments
            stored_files['segments_dir'] = str(segments_dir)
        
        return stored_files
    
    def cleanup_temp_files(self, meeting_id: str):
        """Clean up temporary files after processing"""
        
        temp_dir = self.temp_dir / meeting_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """Get file information and metadata"""
        
        path = Path(file_path)
        if not path.exists():
            return None
        
        stat = path.stat()
        return {
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "extension": path.suffix.lower(),
            "name": path.name
        }
```

### File Processing Pipeline

The file processing pipeline handles format conversion and optimization:

```python
# backend/file_processor.py - File Processing Operations

import subprocess
from pathlib import Path
from typing import Dict, Tuple
import tempfile

class AudioProcessor:
    """Handles audio file processing and conversion"""
    
    @staticmethod
    def normalize_audio(input_path: str, output_path: str) -> Dict[str, Any]:
        """Normalize audio for better processing"""
        
        cmd = [
            'ffmpeg', '-i', input_path,
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # Mono channel
            '-c:a', 'pcm_s16le',  # 16-bit PCM
            '-y',            # Overwrite output
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Get output file info
            info = AudioProcessor.get_audio_info(output_path)
            
            return {
                "success": True,
                "output_path": output_path,
                "duration": info.get("duration", 0),
                "sample_rate": info.get("sample_rate", 0),
                "channels": info.get("channels", 0)
            }
            
        except subprocess.CalledProcessError as e:
            raise FileProcessingError(f"Audio normalization failed: {e.stderr}")
    
    @staticmethod
    def extract_audio_from_video(video_path: str, audio_path: str) -> Dict[str, Any]:
        """Extract audio track from video file"""
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',           # No video
            '-acodec', 'pcm_s16le',  # Audio codec
            '-ar', '16000',  # Sample rate
            '-ac', '1',      # Mono
            '-y',            # Overwrite
            audio_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return {
                "success": True,
                "audio_path": audio_path,
                "extracted_from": video_path
            }
            
        except subprocess.CalledProcessError as e:
            raise FileProcessingError(f"Audio extraction failed: {e.stderr}")
    
    @staticmethod
    def get_audio_info(file_path: str) -> Dict[str, Any]:
        """Get detailed audio file information"""
        
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            file_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            format_info = data.get('format', {})
            stream_info = next(
                (s for s in data.get('streams', []) if s.get('codec_type') == 'audio'),
                {}
            )
            
            return {
                "duration": float(format_info.get('duration', 0)),
                "bitrate": int(format_info.get('bit_rate', 0)),
                "sample_rate": int(stream_info.get('sample_rate', 0)),
                "channels": int(stream_info.get('channels', 0)),
                "codec": stream_info.get('codec_name', 'unknown'),
                "size": int(format_info.get('size', 0))
            }
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as e:
            raise FileProcessingError(f"Failed to get audio info: {e}")
    
    @staticmethod
    def split_audio_by_segments(audio_path: str, segments: List[Dict], output_dir: str) -> List[str]:
        """Split audio file into segments based on timestamps"""
        
        output_paths = []
        
        for i, segment in enumerate(segments):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            duration = end_time - start_time
            
            if duration <= 0:
                continue
            
            segment_path = Path(output_dir) / f"segment_{i:04d}.wav"
            
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',
                '-y',
                str(segment_path)
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                output_paths.append(str(segment_path))
                
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to create segment {i}: {e.stderr}")
        
        return output_paths
```

---

## Health Monitoring and Observability

### Health Check System

Our health check system provides comprehensive monitoring of all services:

```python
# backend/health.py - Health Monitoring System

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any
import requests
import time
import psutil

health_router = APIRouter()

class HealthChecker:
    """Comprehensive health checking for all services"""
    
    def __init__(self):
        self.checks = {
            'database': self.check_database,
            'redis': self.check_redis,
            'whisper_service': self.check_whisper_service,
            'llm_service': self.check_llm_service,
            'disk_space': self.check_disk_space,
            'memory': self.check_memory,
            'celery': self.check_celery_workers
        }
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status"""
        
        results = {}
        overall_status = "healthy"
        
        for check_name, check_func in self.checks.items():
            try:
                start_time = time.time()
                result = await check_func()
                duration = time.time() - start_time
                
                results[check_name] = {
                    "status": "healthy" if result.get("healthy", False) else "unhealthy",
                    "details": result,
                    "response_time": round(duration * 1000, 2)  # ms
                }
                
                if not result.get("healthy", False):
                    overall_status = "unhealthy"
                    
            except Exception as e:
                results[check_name] = {
                    "status": "error",
                    "error": str(e),
                    "response_time": None
                }
                overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": results,
            "version": "1.0.0"
        }
    
    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        
        try:
            db = SessionLocal()
            
            # Test basic connectivity
            start_time = time.time()
            db.execute("SELECT 1")
            query_time = time.time() - start_time
            
            # Check table existence
            tables_exist = db.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name IN ('meetings', 'tasks', 'segments')"
            ).scalar()
            
            # Get database stats
            meeting_count = db.query(Meeting).count()
            task_count = db.query(Task).count()
            
            db.close()
            
            return {
                "healthy": True,
                "query_time_ms": round(query_time * 1000, 2),
                "tables_exist": tables_exist >= 3,
                "meeting_count": meeting_count,
                "task_count": task_count
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity and performance"""
        
        try:
            import redis
            
            r = redis.from_url(REDIS_URL)
            
            # Test basic operations
            start_time = time.time()
            r.ping()
            ping_time = time.time() - start_time
            
            # Test read/write
            test_key = "health_check_test"
            r.set(test_key, "test_value", ex=10)
            retrieved_value = r.get(test_key)
            r.delete(test_key)
            
            # Get Redis info
            info = r.info()
            
            return {
                "healthy": True,
                "ping_time_ms": round(ping_time * 1000, 2),
                "read_write_test": retrieved_value == b"test_value",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown")
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def check_whisper_service(self) -> Dict[str, Any]:
        """Check Whisper service availability"""
        
        try:
            response = requests.get(
                f"{WHISPER_GRPC_URL}/health",
                timeout=5
            )
            
            return {
                "healthy": response.status_code == 200,
                "status_code": response.status_code,
                "response_data": response.json() if response.status_code == 200 else None
            }
            
        except requests.RequestException as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def check_llm_service(self) -> Dict[str, Any]:
        """Check LLM service availability"""
        
        try:
            response = requests.get(
                f"{LLM_SERVICE_URL}/health",
                timeout=5
            )
            
            return {
                "healthy": response.status_code == 200,
                "status_code": response.status_code,
                "response_data": response.json() if response.status_code == 200 else None
            }
            
        except requests.RequestException as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        
        try:
            disk_usage = psutil.disk_usage('/')
            
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            used_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Consider unhealthy if less than 5GB free or more than 95% used
            healthy = free_gb > 5 and used_percent < 95
            
            return {
                "healthy": healthy,
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_percent": round(used_percent, 2)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        
        try:
            memory = psutil.virtual_memory()
            
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            used_percent = memory.percent
            
            # Consider unhealthy if less than 1GB available or more than 90% used
            healthy = available_gb > 1 and used_percent < 90
            
            return {
                "healthy": healthy,
                "available_gb": round(available_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_percent": round(used_percent, 2)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def check_celery_workers(self) -> Dict[str, Any]:
        """Check Celery worker status"""
        
        try:
            from celery import current_app
            
            # Get active workers
            inspect = current_app.control.inspect()
            stats = inspect.stats()
            active_tasks = inspect.active()
            
            if stats is None:
                return {
                    "healthy": False,
                    "error": "No Celery workers found"
                }
            
            worker_count = len(stats)
            total_active_tasks = sum(len(tasks) for tasks in active_tasks.values()) if active_tasks else 0
            
            return {
                "healthy": worker_count > 0,
                "worker_count": worker_count,
                "active_tasks": total_active_tasks,
                "workers": list(stats.keys()) if stats else []
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

# Health check endpoints
health_checker = HealthChecker()

@health_router.get("/health")
async def basic_health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "meeting-intelligence-backend"
    }

@health_router.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check with all services"""
    return await health_checker.run_all_checks()

# Add health router to main app
app.include_router(health_router)
```

### Logging and Monitoring

Our logging system provides structured logging for observability:

```python
# backend/logging_config.py - Logging Configuration

import logging
import sys
from pathlib import Path
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

def setup_logging():
    """Configure application logging"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "app.log"),
        ]
    )
    
    # Configure JSON logging for production
    json_handler = logging.FileHandler(log_dir / "app.json")
    json_handler.setFormatter(JSONFormatter())
    
    # Add JSON handler to root logger
    logging.getLogger().addHandler(json_handler)
    
    # Configure specific loggers
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# Initialize logging
logger = setup_logging()
```

This completes Part 2 of the comprehensive documentation covering the backend architecture in extreme detail. The next parts will cover the AI services, frontend implementation, deployment strategies, and more advanced topics.

Each section provides implementation details, code examples, design decisions, and architectural patterns that you'll need to master for your hackathon presentation. The documentation includes real-world considerations like error handling, security, monitoring, and scalability.

Would you like me to continue with Part 3, which will cover the AI Services Architecture (Whisper, LLM, and ChromaDB integration)?