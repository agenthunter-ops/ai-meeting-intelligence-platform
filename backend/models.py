"""
AI Meeting Intelligence Platform - SQLAlchemy ORM Models
=======================================================
This module defines the database schema using SQLAlchemy ORM models.
These models represent the core data structures and relationships
for storing meetings, transcripts, insights, and processing tasks.

Key Features:
- Normalized relational schema with proper foreign keys
- Indexes for query performance optimization
- Automatic timestamps and audit trails
- JSON columns for flexible metadata storage
- Full-text search support preparation
- Cascading deletes for data consistency
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, 
    ForeignKey, Index, UniqueConstraint, CheckConstraint, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

# Base class for all ORM models
Base = declarative_base()

class TimestampMixin:
    """
    Mixin class to add automatic timestamp fields to models.
    Provides created_at and updated_at fields with automatic population.
    """
    # Automatically set when record is created
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when record was created"
    )
    
    # Automatically updated when record is modified
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True,
        comment="Timestamp when record was last updated"
    )

class Task(Base, TimestampMixin):
    """
    Model for tracking background processing tasks (Celery jobs).
    
    Stores the state and progress of asynchronous operations like
    transcription, AI analysis, and embedding generation.
    
    Relationships:
    - One-to-one with Meeting (task creates meeting)
    """
    __tablename__ = 'tasks'
    
    # Primary key - matches Celery task ID
    id = Column(
        String(100), 
        primary_key=True,
        comment="Celery task UUID identifier"
    )
    
    # Task state tracking
    state = Column(
        String(50),
        nullable=False,
        default='PENDING',
        index=True,
        comment="Current task state (PENDING, PROGRESS, SUCCESS, FAILURE)"
    )
    
    # Progress percentage (0-100)
    progress = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Task completion percentage (0-100)"
    )
    
    # Current processing stage description
    current_stage = Column(
        String(100),
        nullable=True,
        comment="Current processing stage (e.g., 'transcribing', 'analyzing')"
    )
    
    # Serialized task result (JSON)
    result = Column(
        JSON,
        nullable=True,
        comment="Task result data as JSON"
    )
    
    # Error message if task failed
    error = Column(
        Text,
        nullable=True,
        comment="Error message if task failed"
    )
    
    # Processing metadata
    metadata_json = Column(
        'metadata',
        JSON,
        nullable=True,
        comment="Additional task metadata (file info, settings, etc.)"
    )
    # Estimated completion time
    eta = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Estimated completion time"
    )
    
    # Foreign key to associated meeting
    meeting_id = Column(
        Integer,
        ForeignKey('meetings.id', ondelete='CASCADE'),
        nullable=True,
        index=True,
        comment="Associated meeting ID"
    )
    
    # Relationship to Meeting
    meeting = relationship(
        "Meeting",
        back_populates="task",
        cascade="all, delete-orphan",
        single_parent=True
    )
    
    # Indexes for query performance
    __table_args__ = (
        Index('idx_task_state_created', 'state', 'created_at'),
        Index('idx_task_progress', 'progress'),
        CheckConstraint('progress >= 0 AND progress <= 100', name='check_progress_range'),
        {'comment': 'Background processing tasks with state tracking'}
    )
    
    def __repr__(self):
        return f"<Task(id='{self.id}', state='{self.state}', progress={self.progress})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'state': self.state,
            'progress': self.progress,
            'current_stage': self.current_stage,
            'result': self.result,
            'error': self.error,
            'meeting_id': self.meeting_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'eta': self.eta.isoformat() if self.eta else None
        }

class Meeting(Base, TimestampMixin):
    """
    Model for storing meeting information and metadata.
    
    Central entity that ties together transcripts, insights, and tasks.
    Stores basic meeting information and file details.
    
    Relationships:
    - One-to-many with Segment (meeting has many transcript segments)
    - One-to-many with Insight (meeting has many AI insights)  
    - One-to-one with Task (meeting created by processing task)
    - One-to-many with Embedding (meeting has many vector embeddings)
    """
    __tablename__ = 'meetings'
    
    # Primary key
    id = Column(
        Integer, 
        primary_key=True, 
        autoincrement=True,
        comment="Unique meeting identifier"
    )
    
    # Meeting basic information
    title = Column(
        String(200),
        nullable=False,
        index=True,
        comment="Meeting title or description"
    )
    
    # Meeting type for categorization
    meeting_type = Column(
        String(50),
        nullable=False,
        default='general',
        index=True,
        comment="Type of meeting (standup, planning, etc.)"
    )
    
    # Meeting date/time
    meeting_date = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When the meeting took place"
    )
    
    # Attendee information (JSON array)
    attendees = Column(
        JSON,
        nullable=True,
        comment="List of meeting attendees as JSON array"
    )
    
    # File and processing information
    original_filename = Column(
        String(255),
        nullable=True,
        comment="Original uploaded filename"
    )
    
    file_path = Column(
        String(500),
        nullable=True,
        comment="Path to stored audio/video file"
    )
    
    file_size_bytes = Column(
        Integer,
        nullable=True,
        comment="File size in bytes"
    )
    
    # Audio/video metadata
    duration_seconds = Column(
        Float,
        nullable=True,
        comment="Audio/video duration in seconds"
    )
    
    audio_format = Column(
        String(20),
        nullable=True,
        comment="Audio format (mp3, wav, m4a, etc.)"
    )
    
    sample_rate = Column(
        Integer,
        nullable=True,
        comment="Audio sample rate in Hz"
    )
    
    # Processing status
    processing_status = Column(
        String(50),
        nullable=False,
        default='pending',
        index=True,
        comment="Current processing status"
    )
    
    # Processing metadata and settings
    processing_metadata = Column(
        JSON,
        nullable=True,
        comment="Processing settings and metadata as JSON"
    )
    
    # Content statistics (updated after processing)
    total_segments = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total number of transcript segments"
    )
    
    total_words = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total word count across all segments"
    )
    
    total_insights = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Total number of AI-extracted insights"
    )
    
    # Relationships
    segments = relationship(
        "Segment",
        back_populates="meeting",
        cascade="all, delete-orphan",
        order_by="Segment.sequence_number"
    )
    
    insights = relationship(
        "Insight",
        back_populates="meeting", 
        cascade="all, delete-orphan",
        order_by="Insight.created_at"
    )
    
    embeddings = relationship(
        "Embedding",
        back_populates="meeting",
        cascade="all, delete-orphan"
    )
    
    task = relationship(
        "Task",
        back_populates="meeting",
        uselist=False  # One-to-one relationship
    )
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_meeting_type_date', 'meeting_type', 'meeting_date'),
        Index('idx_meeting_status', 'processing_status'),
        Index('idx_meeting_created', 'created_at'),
        # Full-text search index on title (SQLite FTS support)
        Index('idx_meeting_title_fts', 'title'),
        {'comment': 'Core meeting information and metadata'}
    )
    
    def __repr__(self):
        return f"<Meeting(id={self.id}, title='{self.title}', type='{self.meeting_type}')>"
    
    def get_attendee_list(self) -> List[str]:
        """Get attendees as a Python list"""
        if self.attendees:
            return self.attendees if isinstance(self.attendees, list) else json.loads(self.attendees)
        return []
    
    def set_attendee_list(self, attendees: List[str]):
        """Set attendees from a Python list"""
        self.attendees = attendees
    
    def get_duration_formatted(self) -> str:
        """Get formatted duration string (HH:MM:SS)"""
        if not self.duration_seconds:
            return "00:00:00"
        
        hours = int(self.duration_seconds // 3600)
        minutes = int((self.duration_seconds % 3600) // 60)
        seconds = int(self.duration_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def to_dict(self, include_relations: bool = False) -> Dict[str, Any]:
        """Convert model to dictionary for JSON serialization"""
        data = {
            'id': self.id,
            'title': self.title,
            'meeting_type': self.meeting_type,
            'meeting_date': self.meeting_date.isoformat() if self.meeting_date else None,
            'attendees': self.get_attendee_list(),
            'original_filename': self.original_filename,
            'file_size_bytes': self.file_size_bytes,
            'duration_seconds': self.duration_seconds,
            'duration_formatted': self.get_duration_formatted(),
            'audio_format': self.audio_format,
            'processing_status': self.processing_status,
            'total_segments': self.total_segments,
            'total_words': self.total_words,
            'total_insights': self.total_insights,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_relations:
            data.update({
                'segments': [seg.to_dict() for seg in self.segments] if self.segments else [],
                'insights': [ins.to_dict() for ins in self.insights] if self.insights else [],
                'task': self.task.to_dict() if self.task else None
            })
        
        return data

class Segment(Base, TimestampMixin):
    """
    Model for storing individual transcript segments.
    
    Each segment represents a portion of the meeting transcript
    with timing information and speaker identification.
    
    Relationships:
    - Many-to-one with Meeting (segments belong to meeting)
    - One-to-many with Embedding (segment can have multiple embeddings)
    """
    __tablename__ = 'segments'
    
    # Primary key
    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Unique segment identifier"
    )
    
    # Foreign key to meeting
    meeting_id = Column(
        Integer,
        ForeignKey('meetings.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment="Associated meeting ID"
    )
    
    # Segment ordering within meeting
    sequence_number = Column(
        Integer,
        nullable=False,
        comment="Segment order within the meeting (0-based)"
    )
    
    # Transcript content
    text = Column(
        Text,
        nullable=False,
        comment="Transcript text content"
    )
    
    # Timing information
    start_time = Column(
        Float,
        nullable=False,
        comment="Segment start time in seconds"
    )
    
    end_time = Column(
        Float,
        nullable=False,
        comment="Segment end time in seconds"
    )
    
    # Duration (calculated field)
    duration = Column(
        Float,
        nullable=True,
        comment="Segment duration in seconds (end_time - start_time)"
    )
    
    # Speaker identification
    speaker = Column(
        String(100),
        nullable=True,
        default='Unknown',
        comment="Identified speaker name or ID"
    )
    
    # Transcription confidence
    confidence = Column(
        Float,
        nullable=False,
        default=0.95,
        comment="Transcription confidence score (0.0-1.0)"
    )
    
    # Language detection
    language = Column(
        String(10),
        nullable=True,
        comment="Detected language code (e.g., 'en', 'es')"
    )
    
    # Content analysis
    word_count = Column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of words in segment"
    )
    
    # Processing metadata
    processing_metadata = Column(
        JSON,
        nullable=True,
        comment="Additional processing information as JSON"
    )
    
    # Relationships
    meeting = relationship(
        "Meeting",
        back_populates="segments"
    )
    
    embeddings = relationship(
        "Embedding",
        back_populates="segment",
        cascade="all, delete-orphan"
    )
    
    # Indexes and constraints
    __table_args__ = (
        # Unique constraint on meeting_id + sequence_number
        UniqueConstraint('meeting_id', 'sequence_number', name='uk_meeting_sequence'),
        
        # Indexes for common queries
        Index('idx_segment_meeting_sequence', 'meeting_id', 'sequence_number'),
        Index('idx_segment_times', 'start_time', 'end_time'),
        Index('idx_segment_speaker', 'speaker'),
        Index('idx_segment_confidence', 'confidence'),
        
        # Check constraints for data validity
        CheckConstraint('start_time >= 0', name='check_start_time_positive'),
        CheckConstraint('end_time > start_time', name='check_end_after_start'),
        CheckConstraint('confidence >= 0.0 AND confidence <= 1.0', name='check_confidence_range'),
        CheckConstraint('word_count >= 0', name='check_word_count_positive'),
        
        {'comment': 'Individual transcript segments with timing and speaker info'}
    )
    
    def __repr__(self):
        return f"<Segment(id={self.id}, meeting_id={self.meeting_id}, seq={self.sequence_number})>"
    
    def calculate_duration(self):
        """Calculate and update segment duration"""
        if self.start_time is not None and self.end_time is not None:
            self.duration = self.end_time - self.start_time
    
    def get_formatted_time_range(self) -> str:
        """Get formatted time range string (MM:SS - MM:SS)"""
        def format_time(seconds):
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        
        if self.start_time is not None and self.end_time is not None:
            return f"{format_time(self.start_time)} - {format_time(self.end_time)}"
        return "00:00 - 00:00"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'meeting_id': self.meeting_id,
            'sequence_number': self.sequence_number,
            'text': self.text,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'time_range': self.get_formatted_time_range(),
            'speaker': self.speaker,
            'confidence': self.confidence,
            'language': self.language,
            'word_count': self.word_count,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Insight(Base, TimestampMixin):
    """
    Model for storing AI-extracted insights from meetings.
    
    Stores structured insights like action items, decisions,
    sentiment analysis, and key topics identified by AI processing.
    
    Relationships:
    - Many-to-one with Meeting (insights belong to meeting)
    """
    __tablename__ = 'insights'
    
    # Primary key
    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Unique insight identifier"
    )
    
    # Foreign key to meeting
    meeting_id = Column(
        Integer,
        ForeignKey('meetings.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment="Associated meeting ID"
    )
    
    # Insight classification
    type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of insight (action_item, decision, sentiment, etc.)"
    )
    
    # Main insight content
    content = Column(
        Text,
        nullable=False,
        comment="Primary insight content or description"
    )
    
    # AI confidence in this insight
    confidence = Column(
        Float,
        nullable=False,
        default=0.8,
        comment="AI confidence score for this insight (0.0-1.0)"
    )
    
    # Structured metadata for different insight types
    metadata_json = Column(
        'metadata',
        JSON,
        nullable=True,
        comment="Type-specific structured data as JSON"
    )
    source_segment_id = Column(
        Integer,
        ForeignKey('segments.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        comment="Source segment ID if insight came from specific segment"
    )
    
    # Priority/importance ranking
    priority = Column(
        String(20),
        nullable=False,
        default='medium',
        comment="Insight priority (low, medium, high, urgent)"
    )
    
    # Status tracking (for action items)
    status = Column(
        String(50),
        nullable=False,
        default='open',
        comment="Insight status (open, in_progress, completed, cancelled)"
    )
    
    # Tags for categorization
    tags = Column(
        JSON,
        nullable=True,
        comment="List of tags for categorization as JSON array"
    )
    
    # Relationships
    meeting = relationship(
        "Meeting",
        back_populates="insights"
    )
    
    source_segment = relationship(
        "Segment",
        foreign_keys=[source_segment_id],
        backref="related_insights"
    )
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_insight_meeting_type', 'meeting_id', 'type'),
        Index('idx_insight_status', 'status'),
        Index('idx_insight_priority', 'priority'),
        Index('idx_insight_confidence', 'confidence'),
        Index('idx_insight_created', 'created_at'),
        
        CheckConstraint('confidence >= 0.0 AND confidence <= 1.0', name='check_insight_confidence_range'),
        CheckConstraint("priority IN ('low', 'medium', 'high', 'urgent')", name='check_insight_priority'),
        CheckConstraint("status IN ('open', 'in_progress', 'completed', 'cancelled')", name='check_insight_status'),
        
        {'comment': 'AI-extracted insights from meeting content'}
    )
    
    def __repr__(self):
        return f"<Insight(id={self.id}, type='{self.type}', meeting_id={self.meeting_id})>"
    
    def get_tags_list(self) -> List[str]:
        """Get tags as a Python list"""
        if self.tags:
            return self.tags if isinstance(self.tags, list) else json.loads(self.tags)
        return []
    
    def set_tags_list(self, tags: List[str]):
        """Set tags from a Python list"""
        self.tags = tags
    
    def get_metadata_value(self, key: str, default=None):
        """Get a specific value from metadata JSON"""
        if self.metadata_json and isinstance(self.metadata_json, dict):
            return self.metadata_json.get(key, default)
        return default
    
    def set_metadata_value(self, key: str, value):
        """Set a specific value in metadata JSON"""
        if not self.metadata_json:
            self.metadata_json = {}
        if isinstance(self.metadata_json, dict):
            self.metadata_json[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'meeting_id': self.meeting_id,
            'type': self.type,
            'content': self.content,
            'confidence': self.confidence,
            'metadata': self.metadata_json,
            'source_segment_id': self.source_segment_id,
            'priority': self.priority,
            'status': self.status,
            'tags': self.get_tags_list(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Embedding(Base, TimestampMixin):
    """
    Model for storing vector embeddings for semantic search.
    
    Stores vector representations of text segments for similarity
    search and semantic analysis. Supports multiple embedding models.
    
    Relationships:
    - Many-to-one with Meeting (embeddings belong to meeting)
    - Many-to-one with Segment (embeddings can be linked to segments)
    """
    __tablename__ = 'embeddings'
    
    # Primary key
    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Unique embedding identifier"
    )
    
    # Foreign keys
    meeting_id = Column(
        Integer,
        ForeignKey('meetings.id', ondelete='CASCADE'),
        nullable=False,
        index=True,
        comment="Associated meeting ID"
    )
    
    segment_id = Column(
        Integer,
        ForeignKey('segments.id', ondelete='CASCADE'),
        nullable=True,
        index=True,
        comment="Associated segment ID (optional)"
    )
    
    # Embedding model information
    model_name = Column(
        String(100),
        nullable=False,
        index=True,
        comment="Name of the embedding model used"
    )
    
    model_version = Column(
        String(50),
        nullable=True,
        comment="Version of the embedding model"
    )
    
    # Vector data (stored as JSON for SQLite compatibility)
    # In production, consider using a dedicated vector database
    vector = Column(
        JSON,
        nullable=False,
        comment="Vector embedding as JSON array"
    )
    
    # Embedding dimensions
    dimensions = Column(
        Integer,
        nullable=False,
        comment="Number of dimensions in the vector"
    )
    
    # Source text that was embedded
    source_text = Column(
        Text,
        nullable=True,
        comment="Original text that was embedded"
    )
    
    # Content hash for deduplication
    content_hash = Column(
        String(64),
        nullable=True,
        index=True,
        comment="SHA-256 hash of source text for deduplication"
    )
    
    # Embedding metadata
    embedding_metadata = Column(
        JSON,
        nullable=True,
        comment="Additional embedding metadata as JSON"
    )
    
    # Relationships
    meeting = relationship(
        "Meeting",
        back_populates="embeddings"
    )
    
    segment = relationship(
        "Segment",
        back_populates="embeddings"
    )
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_embedding_meeting_model', 'meeting_id', 'model_name'),
        Index('idx_embedding_segment', 'segment_id'),
        Index('idx_embedding_hash', 'content_hash'),
        Index('idx_embedding_dimensions', 'dimensions'),
        
        # Unique constraint to prevent duplicate embeddings
        UniqueConstraint('meeting_id', 'segment_id', 'model_name', 'content_hash', 
                        name='uk_embedding_unique'),
        
        CheckConstraint('dimensions > 0', name='check_embedding_dimensions_positive'),
        
        {'comment': 'Vector embeddings for semantic search and analysis'}
    )
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, meeting_id={self.meeting_id}, model='{self.model_name}')>"
    
    def get_vector_array(self) -> List[float]:
        """Get vector as Python list of floats"""
        if self.vector and isinstance(self.vector, list):
            return [float(x) for x in self.vector]
        return []
    
    def set_vector_array(self, vector: List[float]):
        """Set vector from Python list of floats"""
        self.vector = vector
        self.dimensions = len(vector)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'meeting_id': self.meeting_id,
            'segment_id': self.segment_id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'dimensions': self.dimensions,
            'content_hash': self.content_hash,
            'embedding_metadata': self.embedding_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Create all indexes and constraints
def create_additional_indexes(engine):
    """
    Create additional indexes for performance optimization.
    Called after table creation for complex index operations.
    """
    with engine.connect() as conn:
        # Full-text search virtual table for segments (SQLite FTS5)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts 
            USING fts5(
                content='segments',
                content_rowid='id',
                text,
                speaker,
                meeting_id UNINDEXED
            )
        """)
        
        # Triggers to keep FTS table in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS segments_fts_insert 
            AFTER INSERT ON segments 
            BEGIN
                INSERT INTO segments_fts(rowid, text, speaker, meeting_id) 
                VALUES (new.id, new.text, new.speaker, new.meeting_id);
            END
        """)
        
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS segments_fts_delete 
            AFTER DELETE ON segments 
            BEGIN
                DELETE FROM segments_fts WHERE rowid = old.id;
            END
        """)
        
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS segments_fts_update 
            AFTER UPDATE ON segments 
            BEGIN
                DELETE FROM segments_fts WHERE rowid = old.id;
                INSERT INTO segments_fts(rowid, text, speaker, meeting_id) 
                VALUES (new.id, new.text, new.speaker, new.meeting_id);
            END
        """)
        
        print("✅ Additional indexes and FTS tables created successfully")

# Export all models for easy importing
__all__ = [
    "Base", "TimestampMixin", 
    "Task", "Meeting", "Segment", "Insight", "Embedding",
    "create_additional_indexes"
]
