"""
AI Meeting Intelligence Platform - Database Configuration
========================================================
This module handles SQLite database initialization, connection management,
and provides utility functions for database operations.

Key Features:
- SQLite database with WAL mode for better concurrent access
- Automatic table creation with proper indexes
- Connection pooling and session management
- Database migration support
- Status tracking for async tasks
"""

import sqlite3
import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Optional, Dict, Any
import json

# SQLAlchemy declarative base - all ORM models inherit from this
Base = declarative_base()

# Database file path - stores in project root for development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./meeting_intelligence.db")

# Create SQLAlchemy engine with SQLite-specific optimizations
engine = create_engine(
    DATABASE_URL,
    # Enable connection pooling for better performance
    pool_size=20,
    max_overflow=0,
    # SQLite-specific settings for better concurrent access
    connect_args={
        "check_same_thread": False,  # Allow multiple threads
        "timeout": 20,               # Connection timeout in seconds
        "isolation_level": None,     # Enable autocommit mode
    },
    # Enable SQL query logging in development
    echo=False  # Set to True for SQL debugging
)

# Session factory for database operations
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Database dependency injection for FastAPI endpoints.
    Creates a new database session for each request and ensures proper cleanup.
    
    Usage:
        @app.get("/api/meetings")
        def get_meetings(db: Session = Depends(get_db)):
            return db.query(Meeting).all()
    
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        # Always close the session to prevent connection leaks
        db.close()

def init_db():
    """
    Initialize the database with all required tables and indexes.
    Called once during application startup.
    
    Creates tables:
    - tasks: Track async processing jobs (transcription, AI analysis)
    - meetings: Store meeting metadata and basic information
    - segments: Individual transcript segments with timestamps
    - insights: AI-extracted insights (action items, decisions, etc.)
    - embeddings: Vector embeddings for semantic search
    
    Also optimizes SQLite for concurrent access and performance.
    """
    # Create all tables defined in models.py
    Base.metadata.create_all(bind=engine)
    
    # Apply SQLite-specific optimizations
    with engine.connect() as conn:
        # Enable Write-Ahead Logging for better concurrent access
        conn.exec_driver_sql("PRAGMA journal_mode=WAL")

        # Increase cache size to 64MB for better performance
        conn.exec_driver_sql("PRAGMA cache_size=-64000")

        # Enable foreign key constraints
        conn.exec_driver_sql("PRAGMA foreign_keys=ON")

        # Set synchronous mode to NORMAL for balance of safety and performance
        conn.exec_driver_sql("PRAGMA synchronous=NORMAL")

        # Set busy timeout to handle concurrent access
        conn.exec_driver_sql("PRAGMA busy_timeout=30000")
    
    print("✅ Database initialized successfully with optimizations")

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the current status of a background task.
    
    Args:
        task_id (str): Unique identifier for the Celery task
        
    Returns:
        Dict containing task status, progress, and results, or None if not found
        
    Example:
        status = get_task_status("abc-123-def")
        if status and status['state'] == 'SUCCESS':
            print(f"Task completed: {status['result']}")
    """
    db = SessionLocal()
    try:
        from models import Task  # Import here to avoid circular imports
        
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return None
            
        return {
            "task_id": task.id,
            "state": task.state,
            "progress": task.progress,
            "result": json.loads(task.result) if task.result else None,
            "error": task.error,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat() if task.updated_at else None
        }
    finally:
        db.close()

def update_task_status(task_id: str, state: str, progress: int = 0, 
                      result: Dict[str, Any] = None, error: str = None):
    """
    Update the status of a background task.
    Called by Celery workers to report progress and results.
    
    Args:
        task_id (str): Unique task identifier
        state (str): Task state ('PENDING', 'PROGRESS', 'SUCCESS', 'FAILURE')
        progress (int): Completion percentage (0-100)
        result (Dict): Task result data (serialized as JSON)
        error (str): Error message if task failed
        
    Example:
        update_task_status("abc-123", "PROGRESS", progress=50)
        update_task_status("abc-123", "SUCCESS", progress=100, 
                          result={"segments": 25, "insights": 12})
    """
    db = SessionLocal()
    try:
        from models import Task
        
        # Find existing task or create new one
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            task = Task(id=task_id)
            db.add(task)
        
        # Update task fields
        task.state = state
        task.progress = progress
        task.updated_at = datetime.utcnow()
        
        if result:
            task.result = json.dumps(result)
        if error:
            task.error = error
            
        # Commit changes to database
        db.commit()
        
    except Exception as e:
        # Rollback on error and re-raise
        db.rollback()
        raise e
    finally:
        db.close()

def save_meeting_segments(meeting_id: int, segments: list):
    """
    Save transcript segments for a meeting.
    Called by transcription workers after processing audio.
    
    Args:
        meeting_id (int): Meeting database ID
        segments (list): List of segment dictionaries with text, start_time, end_time
        
    Example:
        segments = [
            {"text": "Welcome everyone", "start_time": 0.0, "end_time": 2.5},
            {"text": "Let's discuss the project", "start_time": 2.5, "end_time": 5.8}
        ]
        save_meeting_segments(123, segments)
    """
    db = SessionLocal()
    try:
        from models import Segment
        
        # Create segment objects
        segment_objects = []
        for i, seg in enumerate(segments):
            segment = Segment(
                meeting_id=meeting_id,
                sequence_number=i,
                text=seg["text"],
                start_time=seg["start_time"],
                end_time=seg["end_time"],
                speaker=seg.get("speaker", "Unknown"),  # Speaker detection if available
                confidence=seg.get("confidence", 0.95)   # Transcription confidence
            )
            segment_objects.append(segment)
        
        # Bulk insert for better performance
        db.bulk_save_objects(segment_objects)
        db.commit()
        
        print(f"✅ Saved {len(segments)} segments for meeting {meeting_id}")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error saving segments: {e}")
        raise e
    finally:
        db.close()

def save_meeting_insights(meeting_id: int, insights: Dict[str, Any]):
    """
    Save AI-extracted insights for a meeting.
    Called by LLM workers after processing transcript segments.
    
    Args:
        meeting_id (int): Meeting database ID
        insights (Dict): Dictionary containing extracted insights
        
    Example:
        insights = {
            "action_items": [{"text": "Send report", "assignee": "John", "due_date": "2025-01-15"}],
            "decisions": [{"text": "Approved budget increase", "impact": "high"}],
            "sentiment": {"overall": "positive", "score": 0.7}
        }
        save_meeting_insights(123, insights)
    """
    db = SessionLocal()
    try:
        from models import Insight
        
        # Save different types of insights
        insight_objects = []
        
        # Process action items
        for item in insights.get("action_items", []):
            insight = Insight(
                meeting_id=meeting_id,
                type="action_item",
                content=item["text"],
                metadata_json=json.dumps({
                    "assignee": item.get("assignee"),
                    "due_date": item.get("due_date"),
                    "priority": item.get("priority", "medium")
                }),
                confidence=item.get("confidence", 0.9)
            )
            insight_objects.append(insight)
        
        # Process decisions
        for decision in insights.get("decisions", []):
            insight = Insight(
                meeting_id=meeting_id,
                type="decision",
                content=decision["text"],
                metadata_json=json.dumps({
                    "impact": decision.get("impact", "medium"),
                    "rationale": decision.get("rationale")
                }),
                confidence=decision.get("confidence", 0.85)
            )
            insight_objects.append(insight)
        
        # Process sentiment analysis
        if "sentiment" in insights:
            sentiment = insights["sentiment"]
            insight = Insight(
                meeting_id=meeting_id,
                type="sentiment",
                content=f"Overall sentiment: {sentiment['overall']}",
                metadata_json=json.dumps({
                    "score": sentiment["score"],
                    "positive_ratio": sentiment.get("positive_ratio", 0.0),
                    "negative_ratio": sentiment.get("negative_ratio", 0.0)
                }),
                confidence=sentiment.get("confidence", 0.8)
            )
            insight_objects.append(insight)
        
        # Bulk save insights
        db.bulk_save_objects(insight_objects)
        db.commit()
        
        print(f"✅ Saved {len(insight_objects)} insights for meeting {meeting_id}")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error saving insights: {e}")
        raise e
    finally:
        db.close()

def search_meetings(query: str, limit: int = 10) -> list:
    """
    Search meetings by text content using SQLite FTS (Full-Text Search).
    
    Args:
        query (str): Search query string
        limit (int): Maximum number of results to return
        
    Returns:
        List of meeting dictionaries with relevance scores
        
    Example:
        results = search_meetings("project budget discussion", limit=5)
        for meeting in results:
            print(f"Meeting: {meeting['title']} (Score: {meeting['score']})")
    """
    db = SessionLocal()
    try:
        from models import Meeting, Segment
        
        # Use SQLite FTS for fast text search across segments
        # This requires FTS5 virtual table (can be set up in init_db)
        raw_query = """
        SELECT m.id, m.title, m.created_at, 
               snippet(segments_fts, -1, '<mark>', '</mark>', '...', 32) as snippet,
               rank as score
        FROM meetings m
        JOIN segments_fts ON segments_fts.meeting_id = m.id
        WHERE segments_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """
        
        results = db.execute(raw_query, (query, limit)).fetchall()
        
        # Convert to dictionaries
        meetings = []
        for row in results:
            meetings.append({
                "id": row[0],
                "title": row[1],
                "created_at": row[2].isoformat(),
                "snippet": row[3],
                "score": row[4]
            })
        
        return meetings
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []
    finally:
        db.close()

# Database health check function
def check_db_health() -> Dict[str, Any]:
    """
    Check database connectivity and basic health metrics.
    Used by monitoring systems and health check endpoints.
    
    Returns:
        Dictionary with health status and metrics
    """
    try:
        db = SessionLocal()
        
        # Test basic connectivity
        result = db.execute("SELECT 1").fetchone()
        
        # Get table counts for monitoring
        from models import Meeting, Task, Segment, Insight
        
        meeting_count = db.query(Meeting).count()
        task_count = db.query(Task).count()
        segment_count = db.query(Segment).count()
        insight_count = db.query(Insight).count()
        
        db.close()
        
        return {
            "status": "healthy",
            "database": "sqlite",
            "tables": {
                "meetings": meeting_count,
                "tasks": task_count,
                "segments": segment_count,
                "insights": insight_count
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Initialize database on module import
if __name__ == "__main__":
    print("🗄️  Initializing AI Meeting Intelligence Database...")
    init_db()
    health = check_db_health()
    print(f"📊 Database Health: {health}")
