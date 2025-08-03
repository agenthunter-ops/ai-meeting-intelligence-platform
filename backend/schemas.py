"""
AI Meeting Intelligence Platform - Pydantic Schemas
==================================================
This module defines Pydantic models for API request/response validation,
serialization, and documentation. These schemas ensure type safety
and automatic API documentation generation.

Key Features:
- Request/response validation with automatic error messages
- JSON serialization/deserialization
- OpenAPI documentation generation
- Data transformation and cleaning
- Custom validators for business logic
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic import ValidationInfo
from typing import Optional, List, Dict, Any, Union, Literal
from datetime import datetime
from enum import Enum
import re

class TaskState(str, Enum):
    """
    Enumeration of possible task states for background processing.
    Used to track the lifecycle of transcription and AI analysis jobs.
    """
    PENDING = "PENDING"      # Task created but not started
    PROGRESS = "PROGRESS"    # Task is currently running
    SUCCESS = "SUCCESS"      # Task completed successfully
    FAILURE = "FAILURE"      # Task failed with error
    RETRY = "RETRY"          # Task is being retried
    REVOKED = "REVOKED"      # Task was cancelled

class MeetingType(str, Enum):
    """
    Types of meetings supported by the platform.
    Used for categorization and specialized processing.
    """
    STANDUP = "standup"           # Daily standup meetings
    PLANNING = "planning"         # Sprint/project planning
    RETROSPECTIVE = "retrospective"  # Team retrospectives
    ONE_ON_ONE = "one_on_one"    # Manager-employee meetings
    ALL_HANDS = "all_hands"      # Company-wide meetings
    CLIENT_CALL = "client_call"  # External client calls
    INTERVIEW = "interview"      # Job interviews
    GENERAL = "general"          # General-purpose meetings

class InsightType(str, Enum):
    """
    Types of insights that can be extracted from meetings.
    Used to categorize AI-generated insights.
    """
    ACTION_ITEM = "action_item"  # Tasks assigned to team members
    DECISION = "decision"        # Decisions made during meeting
    SENTIMENT = "sentiment"      # Overall meeting sentiment
    SUMMARY = "summary"          # Meeting summary
    TOPIC = "topic"             # Key topics discussed
    RISK = "risk"               # Identified risks or concerns

# Base schemas with common fields
class BaseSchema(BaseModel):
    """
    Base schema with common configuration for all Pydantic models.
    Provides consistent serialization and validation settings.
    """
    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
        validate_assignment=True,
        populate_by_name=True,
    )

# Upload and Task Management Schemas
class UploadRequest(BaseSchema):
    """
    Schema for file upload requests.
    Validates file metadata and meeting information.
    """
    # Meeting title with length validation
    title: str = Field(
        ..., 
        min_length=1, 
        max_length=200,
        description="Meeting title or description"
    )
    
    # Optional meeting type for specialized processing
    meeting_type: MeetingType = Field(
        default=MeetingType.GENERAL,
        description="Type of meeting for optimized processing"
    )
    
    # Optional attendee list
    attendees: Optional[List[str]] = Field(
        default=None,
        description="List of meeting attendees"
    )
    
    # Meeting date/time
    meeting_date: Optional[datetime] = Field(
        default=None,
        description="When the meeting took place"
    )
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Ensure title is not just whitespace"""
        if not v.strip():
            raise ValueError('Title cannot be empty or just whitespace')
        return v.strip()

    @field_validator('attendees')
    @classmethod
    def validate_attendees(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Clean and validate attendee list"""
        if v is None:
            return v
        # Remove empty strings and strip whitespace
        cleaned = [name.strip() for name in v if name.strip()]
        # Remove duplicates while preserving order
        seen: set[str] = set()
        unique_attendees: List[str] = []
        for attendee in cleaned:
            if attendee.lower() not in seen:
                seen.add(attendee.lower())
                unique_attendees.append(attendee)
        return unique_attendees if unique_attendees else None

class UploadResponse(BaseSchema):
    """
    Response schema for successful file uploads.
    Returns task ID for status tracking.
    """
    task_id: str = Field(..., description="Unique task identifier for tracking progress")
    meeting_id: int = Field(..., description="Database ID of the created meeting")
    message: str = Field(default="Upload successful", description="Status message")
    estimated_duration: Optional[int] = Field(
        default=None, 
        description="Estimated processing time in seconds"
    )

class TaskStatus(BaseSchema):
    """
    Schema for task status information.
    Provides detailed progress and result data.
    """
    task_id: str = Field(..., description="Task identifier")
    state: TaskState = Field(..., description="Current task state")
    progress: int = Field(
        default=0, 
        ge=0, 
        le=100, 
        description="Completion percentage (0-100)"
    )
    
    # Current processing stage
    current_stage: Optional[str] = Field(
        default=None,
        description="Current processing stage (e.g., 'transcribing', 'analyzing')"
    )
    
    # Detailed progress information
    stages_completed: Optional[List[str]] = Field(
        default=None,
        description="List of completed processing stages"
    )
    
    # Processing times
    started_at: Optional[datetime] = Field(default=None, description="Task start time")
    updated_at: Optional[datetime] = Field(default=None, description="Last update time")
    completed_at: Optional[datetime] = Field(default=None, description="Task completion time")
    
    # Results and errors
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task results")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    
    # Estimated time remaining
    eta_seconds: Optional[int] = Field(
        default=None,
        description="Estimated time to completion in seconds"
    )


# Alias for backward compatibility
class StatusResponse(TaskStatus):
    """Response model for task status queries."""
    pass

# Meeting and Content Schemas
class SegmentBase(BaseSchema):
    """
    Base schema for transcript segments.
    Contains common fields for segment data.
    """
    text: str = Field(..., min_length=1, description="Transcript text")
    start_time: float = Field(..., ge=0, description="Start time in seconds")
    end_time: float = Field(..., ge=0, description="End time in seconds")
    speaker: Optional[str] = Field(default="Unknown", description="Speaker identification")
    confidence: float = Field(
        default=0.95, 
        ge=0.0, 
        le=1.0, 
        description="Transcription confidence score"
    )
    
    @field_validator('end_time')
    @classmethod
    def validate_end_time(cls, v: float, info: ValidationInfo) -> float:
        """Ensure end_time is after start_time"""
        start = info.data.get('start_time') if info.data else None
        if start is not None and v <= start:
            raise ValueError('end_time must be greater than start_time')
        return v

class SegmentCreate(SegmentBase):
    """Schema for creating new transcript segments"""
    pass

class SegmentResponse(SegmentBase):
    """
    Schema for segment responses with database fields.
    Includes auto-generated IDs and metadata.
    """
    id: int = Field(..., description="Segment database ID")
    meeting_id: int = Field(..., description="Associated meeting ID")
    sequence_number: int = Field(..., description="Segment order in transcript")
    created_at: datetime = Field(..., description="Creation timestamp")

class InsightBase(BaseSchema):
    """
    Base schema for AI-extracted insights.
    Common fields for all insight types.
    """
    type: InsightType = Field(..., description="Type of insight")
    content: str = Field(..., min_length=1, description="Insight content/description")
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="AI confidence in this insight"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional structured data for the insight"
    )

class ActionItemInsight(InsightBase):
    """
    Specialized schema for action item insights.
    Includes assignee, due date, and priority information.
    """
    type: Literal[InsightType.ACTION_ITEM] = Field(default=InsightType.ACTION_ITEM)
    assignee: Optional[str] = Field(default=None, description="Person assigned to the task")
    due_date: Optional[datetime] = Field(default=None, description="Task due date")
    priority: str = Field(
        default="medium",
        pattern="^(low|medium|high|urgent)$",
        description="Task priority level"
    )
    status: str = Field(
        default="open",
        pattern="^(open|in_progress|completed|cancelled)$",
        description="Current task status"
    )

class DecisionInsight(InsightBase):
    """
    Specialized schema for decision insights.
    Includes impact level and rationale.
    """
    type: Literal[InsightType.DECISION] = Field(default=InsightType.DECISION)
    impact: str = Field(
        default="medium",
        pattern="^(low|medium|high|critical)$",
        description="Decision impact level"
    )
    rationale: Optional[str] = Field(default=None, description="Decision reasoning")
    stakeholders: Optional[List[str]] = Field(
        default=None,
        description="People affected by this decision"
    )

class InsightResponse(InsightBase):
    """
    Schema for insight responses with database fields.
    """
    id: int = Field(..., description="Insight database ID")
    meeting_id: int = Field(..., description="Associated meeting ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")

class MeetingBase(BaseSchema):
    """
    Base schema for meeting information.
    Common fields for meeting creation and updates.
    """
    title: str = Field(..., min_length=1, max_length=200, description="Meeting title")
    meeting_type: MeetingType = Field(default=MeetingType.GENERAL, description="Meeting type")
    attendees: Optional[List[str]] = Field(default=None, description="Meeting attendees")
    meeting_date: Optional[datetime] = Field(default=None, description="Meeting date/time")
    duration_minutes: Optional[int] = Field(
        default=None,
        gt=0,
        description="Meeting duration in minutes"
    )
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Clean and validate meeting title"""
        return v.strip()

class MeetingCreate(MeetingBase):
    """Schema for creating new meetings"""
    pass

class MeetingResponse(MeetingBase):
    """
    Complete meeting response with all related data.
    Includes segments, insights, and processing status.
    """
    id: int = Field(..., description="Meeting database ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
    
    # Processing status
    processing_status: TaskState = Field(default=TaskState.PENDING, description="Processing status")
    
    # File information
    file_path: Optional[str] = Field(default=None, description="Original file path")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    audio_duration: Optional[float] = Field(default=None, description="Audio duration in seconds")
    
    # Related data (loaded on request)
    segments: Optional[List[SegmentResponse]] = Field(default=None, description="Transcript segments")
    insights: Optional[List[InsightResponse]] = Field(default=None, description="AI insights")
    
    # Analytics
    total_segments: int = Field(default=0, description="Total number of segments")
    total_insights: int = Field(default=0, description="Total number of insights")
    action_items_count: int = Field(default=0, description="Number of action items")
    decisions_count: int = Field(default=0, description="Number of decisions")

# Search and Query Schemas
class SearchRequest(BaseSchema):
    """
    Schema for search requests.
    Supports both text and semantic search parameters.
    """
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=500,
        description="Search query text"
    )
    
    # Search options
    search_type: str = Field(
        default="hybrid",
        pattern="^(text|semantic|hybrid)$",
        description="Type of search to perform"
    )
    
    # Filters
    meeting_type: Optional[MeetingType] = Field(
        default=None,
        description="Filter by meeting type"
    )
    
    date_from: Optional[datetime] = Field(
        default=None,
        description="Filter meetings from this date"
    )
    
    date_to: Optional[datetime] = Field(
        default=None,
        description="Filter meetings until this date"
    )
    
    attendee: Optional[str] = Field(
        default=None,
        description="Filter by attendee name"
    )
    
    # Pagination
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum results to return"
    )
    
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of results to skip"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Clean and validate search query"""
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', v.strip())
        if not cleaned:
            raise ValueError('Query cannot be empty')
        return cleaned

    @field_validator('date_to')
    @classmethod
    def validate_date_range(cls, v: Optional[datetime], info: ValidationInfo) -> Optional[datetime]:
        """Ensure date_to is after date_from"""
        start = info.data.get('date_from') if info.data else None
        if v and start and v <= start:
            raise ValueError('date_to must be after date_from')
        return v

class SearchResult(BaseSchema):
    """
    Individual search result item.
    Contains meeting info and relevance scoring.
    """
    meeting_id: int = Field(..., description="Meeting database ID")
    title: str = Field(..., description="Meeting title")
    meeting_type: MeetingType = Field(..., description="Meeting type")
    meeting_date: Optional[datetime] = Field(default=None, description="Meeting date")
    
    # Relevance scoring
    score: float = Field(..., ge=0.0, description="Relevance score")
    
    # Snippet with highlighted matches
    snippet: Optional[str] = Field(default=None, description="Text snippet with highlights")
    
    # Match information
    match_type: str = Field(
        ...,
        pattern="^(title|content|insight)$",
        description="Where the match was found"
    )
    
    # Quick stats
    segment_count: int = Field(default=0, description="Number of segments")
    insight_count: int = Field(default=0, description="Number of insights")
    action_items_count: int = Field(default=0, description="Number of action items")

class SearchResponse(BaseSchema):
    """
    Complete search response with results and metadata.
    """
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., ge=0, description="Total number of matching results")
    results: List[SearchResult] = Field(..., description="Search results")
    
    # Search metadata
    search_time_ms: float = Field(..., ge=0, description="Search execution time in milliseconds")
    search_type: str = Field(..., description="Type of search performed")
    
    # Pagination
    limit: int = Field(..., description="Results limit used")
    offset: int = Field(..., description="Results offset used")
    has_more: bool = Field(..., description="Whether more results are available")

# Analytics and Statistics Schemas
class MeetingAnalytics(BaseSchema):
    """
    Analytics data for a specific meeting.
    Contains insights, statistics, and trends.
    """
    meeting_id: int = Field(..., description="Meeting database ID")
    
    # Basic statistics
    total_duration: float = Field(..., ge=0, description="Total meeting duration in seconds")
    speaking_time: Dict[str, float] = Field(
        default={},
        description="Speaking time per participant in seconds"
    )
    
    # Content analysis
    word_count: int = Field(default=0, ge=0, description="Total word count")
    unique_words: int = Field(default=0, ge=0, description="Unique word count")
    
    # Insight statistics
    insights_by_type: Dict[str, int] = Field(
        default={},
        description="Count of insights by type"
    )
    
    # Sentiment analysis
    overall_sentiment: Optional[str] = Field(
        default=None,
        pattern="^(positive|neutral|negative)$",
        description="Overall meeting sentiment"
    )
    sentiment_score: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Sentiment score (-1 to 1)"
    )
    
    # Topic analysis
    key_topics: List[str] = Field(default=[], description="Key topics discussed")
    topic_distribution: Dict[str, float] = Field(
        default={},
        description="Topic distribution percentages"
    )

class SystemHealth(BaseSchema):
    """
    System health and status information.
    Used for monitoring and debugging.
    """
    status: str = Field(
        ...,
        pattern="^(healthy|degraded|unhealthy)$",
        description="Overall system status"
    )
    
    timestamp: datetime = Field(..., description="Health check timestamp")
    
    # Component status
    database: Dict[str, Any] = Field(..., description="Database health info")
    redis: Dict[str, Any] = Field(..., description="Redis health info")
    celery: Dict[str, Any] = Field(..., description="Celery health info")
    
    # System metrics
    active_tasks: int = Field(default=0, ge=0, description="Number of active background tasks")
    total_meetings: int = Field(default=0, ge=0, description="Total meetings in system")
    storage_used_gb: float = Field(default=0.0, ge=0, description="Storage used in GB")

# Error response schemas
class ErrorDetail(BaseSchema):
    """Individual error detail with context"""
    field: Optional[str] = Field(default=None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(default=None, description="Error code")

class ErrorResponse(BaseSchema):
    """
    Standardized error response schema.
    Provides consistent error formatting across all endpoints.
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[List[ErrorDetail]] = Field(
        default=None,
        description="Detailed error information"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for debugging"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )

# Export all schemas for easy importing
__all__ = [
    "TaskState", "MeetingType", "InsightType",
    "UploadRequest", "UploadResponse", "TaskStatus",
    "SegmentCreate", "SegmentResponse", "InsightResponse",
    "ActionItemInsight", "DecisionInsight",
    "MeetingCreate", "MeetingResponse",
    "SearchRequest", "SearchResult", "SearchResponse",
    "MeetingAnalytics", "SystemHealth",
    "ErrorDetail", "ErrorResponse"
]
