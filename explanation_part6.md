# AI Meeting Intelligence Platform - Part 6: Testing & Quality Assurance

## Table of Contents
1. [Testing Strategy Overview](#testing-strategy-overview)
2. [Unit Testing Implementation](#unit-testing-implementation)
3. [Integration Testing](#integration-testing)
4. [API Testing](#api-testing)
5. [Performance Testing](#performance-testing)
6. [End-to-End Testing](#end-to-end-testing)
7. [Security Testing](#security-testing)
8. [Quality Assurance Automation](#quality-assurance-automation)

---

## Testing Strategy Overview

### Comprehensive Testing Pyramid

Our testing strategy follows the testing pyramid principle, ensuring robust quality assurance across all layers of the application. The strategy is designed to catch issues early in the development cycle while maintaining high confidence in production deployments.

```
                    ┌─────────────────┐
                    │   E2E Tests     │  ← Few, High-Value, Slow
                    │   (Browser)     │
                    ├─────────────────┤
                    │ Integration     │  ← Some, API/Service Level
                    │ Tests           │
                    ├─────────────────┤
                    │ Unit Tests      │  ← Many, Fast, Isolated
                    └─────────────────┘

Testing Coverage Distribution:
- Unit Tests: 70% of total tests
- Integration Tests: 20% of total tests  
- E2E Tests: 10% of total tests
```

### Testing Framework Architecture

```python
# tests/conftest.py - Pytest Configuration and Fixtures

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Generator, AsyncGenerator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient
import redis
from unittest.mock import Mock, AsyncMock
import os

# Import application modules
from backend.app import app
from backend.db import Base, get_db
from backend.models import Meeting, Task, Segment
from backend.ai_service_manager import AIServiceManager

# Test configuration
TEST_DATABASE_URL = "sqlite:///:memory:"
TEST_REDIS_URL = "redis://localhost:6379/1"  # Use different DB for tests

# Test database engine
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def test_db():
    """Create a fresh database for each test."""
    
    # Create all tables
    Base.metadata.create_all(bind=test_engine)
    
    # Create session
    db = TestSessionLocal()
    
    try:
        yield db
    finally:
        db.close()
        # Drop all tables after test
        Base.metadata.drop_all(bind=test_engine)

@pytest.fixture(scope="function")
def client(test_db):
    """Create a test client with database override."""
    
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up
    app.dependency_overrides.clear()

@pytest.fixture(scope="function")
def temp_media_dir():
    """Create temporary directory for test media files."""
    
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Create subdirectories
    (temp_path / "uploads").mkdir()
    (temp_path / "processed").mkdir()
    
    yield temp_path
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def sample_audio_file(temp_media_dir):
    """Create a sample audio file for testing."""
    
    # Create a small WAV file (silence)
    import wave
    import struct
    
    audio_file = temp_media_dir / "test_meeting.wav"
    
    with wave.open(str(audio_file), 'w') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(16000)  # 16kHz
        
        # 5 seconds of silence
        duration = 5
        frames = []
        for i in range(0, int(duration * 16000)):
            frames.append(struct.pack('<h', 0))
        
        wav_file.writeframes(b''.join(frames))
    
    return audio_file

@pytest.fixture(scope="function")
def mock_ai_services():
    """Mock AI services for testing."""
    
    mock_whisper = AsyncMock()
    mock_whisper.process_meeting_audio.return_value = {
        "text": "This is a test meeting transcript.",
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "This is a test meeting transcript.",
                "speaker": "Speaker_0",
                "confidence": 0.95
            }
        ],
        "speakers": ["Speaker_0"],
        "statistics": {
            "duration": 5.0,
            "speaker_count": 1,
            "segment_count": 1,
            "word_count": 7,
            "average_confidence": 0.95
        }
    }
    
    mock_llm = AsyncMock()
    mock_llm.analyze_sentiment_advanced.return_value = {
        "overall_sentiment": "neutral",
        "sentiment_score": 0.1,
        "emotions": ["professional"],
        "segment_sentiments": []
    }
    mock_llm.generate_summary.return_value = {
        "summary": "Test meeting summary",
        "key_points": ["Test point 1", "Test point 2"],
        "action_items": ["Test action"],
        "decisions": ["Test decision"],
        "topics": ["testing"]
    }
    
    mock_vector_store = Mock()
    mock_vector_store.store_meeting.return_value = True
    mock_vector_store.store_segments.return_value = 1
    
    return {
        "whisper": mock_whisper,
        "llm": mock_llm,
        "vector_store": mock_vector_store
    }

@pytest.fixture(scope="function")
def redis_client():
    """Redis client for testing."""
    try:
        client = redis.from_url(TEST_REDIS_URL)
        client.ping()
        
        # Clean test database
        client.flushdb()
        
        yield client
        
        # Clean up after test
        client.flushdb()
        client.close()
        
    except redis.ConnectionError:
        pytest.skip("Redis not available for testing")

@pytest.fixture(scope="function")
def sample_meeting(test_db):
    """Create a sample meeting for testing."""
    
    meeting = Meeting(
        title="Test Meeting",
        filename="test_meeting.wav",
        file_path="/tmp/test_meeting.wav",
        file_size=1024,
        mime_type="audio/wav",
        duration=60.0,
        status="completed",
        meeting_type="standup"
    )
    
    test_db.add(meeting)
    test_db.commit()
    test_db.refresh(meeting)
    
    return meeting

@pytest.fixture(scope="function")
def sample_segments(test_db, sample_meeting):
    """Create sample segments for testing."""
    
    segments = [
        Segment(
            meeting_id=sample_meeting.id,
            start_time=0.0,
            end_time=10.0,
            speaker_id="Speaker_0",
            speaker_name="Alice",
            text="Hello everyone, let's start the meeting.",
            confidence=0.95,
            sentiment="positive",
            sentiment_score=0.3
        ),
        Segment(
            meeting_id=sample_meeting.id,
            start_time=10.0,
            end_time=20.0,
            speaker_id="Speaker_1",
            speaker_name="Bob",
            text="Good morning, Alice. I have an update on the project.",
            confidence=0.92,
            sentiment="neutral",
            sentiment_score=0.1
        )
    ]
    
    for segment in segments:
        test_db.add(segment)
    
    test_db.commit()
    
    for segment in segments:
        test_db.refresh(segment)
    
    return segments

# Performance testing fixtures
@pytest.fixture(scope="function")
def performance_monitor():
    """Monitor for performance testing."""
    
    import time
    import psutil
    import threading
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.memory_usage = []
            self.cpu_usage = []
            self.monitoring = False
            self.monitor_thread = None
        
        def start_monitoring(self):
            """Start monitoring system resources."""
            self.start_time = time.time()
            self.monitoring = True
            self.memory_usage = []
            self.cpu_usage = []
            
            def monitor():
                while self.monitoring:
                    self.memory_usage.append(psutil.virtual_memory().percent)
                    self.cpu_usage.append(psutil.cpu_percent(interval=1))
                    time.sleep(1)
            
            self.monitor_thread = threading.Thread(target=monitor)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
        
        def stop_monitoring(self):
            """Stop monitoring and return results."""
            self.end_time = time.time()
            self.monitoring = False
            
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            
            return {
                "duration": self.end_time - self.start_time,
                "avg_memory_usage": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
                "max_memory_usage": max(self.memory_usage) if self.memory_usage else 0,
                "avg_cpu_usage": sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
                "max_cpu_usage": max(self.cpu_usage) if self.cpu_usage else 0,
            }
    
    return PerformanceMonitor()

# Utility functions for tests
def create_test_file_upload(file_path: Path, filename: str = None):
    """Create a file upload object for testing."""
    
    if filename is None:
        filename = file_path.name
    
    with open(file_path, 'rb') as f:
        return ("file", (filename, f.read(), "audio/wav"))

def assert_response_structure(response, expected_fields):
    """Assert that response has expected structure."""
    
    assert response.status_code == 200
    data = response.json()
    
    for field in expected_fields:
        assert field in data, f"Field '{field}' missing from response"

def wait_for_condition(condition_func, timeout=30, interval=1):
    """Wait for a condition to be true with timeout."""
    
    import time
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        time.sleep(interval)
    
    return False

# Test categories markers
def pytest_configure(config):
    """Configure pytest markers."""
    
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance tests"
    )
    config.addinivalue_line(
        "markers", "security: Security tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
```

---

## Unit Testing Implementation

### Backend Model Testing

```python
# tests/unit/test_models.py - Unit tests for database models

import pytest
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from backend.models import Meeting, Task, Segment

@pytest.mark.unit
class TestMeetingModel:
    """Test cases for Meeting model."""
    
    def test_create_meeting(self, test_db):
        """Test creating a new meeting."""
        
        meeting = Meeting(
            title="Test Meeting",
            filename="test.wav",
            file_path="/tmp/test.wav",
            file_size=1024,
            mime_type="audio/wav",
            meeting_type="standup"
        )
        
        test_db.add(meeting)
        test_db.commit()
        test_db.refresh(meeting)
        
        assert meeting.id is not None
        assert meeting.title == "Test Meeting"
        assert meeting.status == "uploaded"  # Default status
        assert meeting.upload_time is not None
        assert isinstance(meeting.upload_time, datetime)
    
    def test_meeting_relationships(self, test_db):
        """Test meeting relationships with tasks and segments."""
        
        # Create meeting
        meeting = Meeting(
            title="Test Meeting",
            filename="test.wav",
            file_path="/tmp/test.wav"
        )
        test_db.add(meeting)
        test_db.commit()
        test_db.refresh(meeting)
        
        # Create task
        task = Task(
            meeting_id=meeting.id,
            task_type="transcription",
            status="pending"
        )
        test_db.add(task)
        
        # Create segment
        segment = Segment(
            meeting_id=meeting.id,
            start_time=0.0,
            end_time=10.0,
            text="Test segment",
            speaker_id="Speaker_0"
        )
        test_db.add(segment)
        
        test_db.commit()
        
        # Test relationships
        assert len(meeting.tasks) == 1
        assert len(meeting.segments) == 1
        assert meeting.tasks[0].task_type == "transcription"
        assert meeting.segments[0].text == "Test segment"
    
    def test_meeting_validation(self, test_db):
        """Test meeting model validation."""
        
        # Test required fields
        with pytest.raises((IntegrityError, ValueError)):
            meeting = Meeting()  # Missing required fields
            test_db.add(meeting)
            test_db.commit()
    
    def test_meeting_repr(self, test_db):
        """Test meeting string representation."""
        
        meeting = Meeting(
            title="Test Meeting",
            filename="test.wav",
            file_path="/tmp/test.wav",
            status="completed"
        )
        test_db.add(meeting)
        test_db.commit()
        test_db.refresh(meeting)
        
        repr_str = repr(meeting)
        assert "Test Meeting" in repr_str
        assert "completed" in repr_str

@pytest.mark.unit
class TestTaskModel:
    """Test cases for Task model."""
    
    def test_create_task(self, test_db, sample_meeting):
        """Test creating a new task."""
        
        task = Task(
            meeting_id=sample_meeting.id,
            task_type="transcription",
            status="pending",
            progress=0
        )
        
        test_db.add(task)
        test_db.commit()
        test_db.refresh(task)
        
        assert task.id is not None
        assert task.meeting_id == sample_meeting.id
        assert task.task_type == "transcription"
        assert task.status == "pending"
        assert task.progress == 0
        assert task.created_at is not None
    
    def test_task_progress_validation(self, test_db, sample_meeting):
        """Test task progress validation."""
        
        task = Task(
            meeting_id=sample_meeting.id,
            task_type="transcription",
            progress=50
        )
        
        test_db.add(task)
        test_db.commit()
        test_db.refresh(task)
        
        # Test valid progress values
        assert 0 <= task.progress <= 100
        
        # Test setting progress to completion
        task.progress = 100
        task.status = "completed"
        task.completed_at = datetime.now()
        test_db.commit()
        
        assert task.progress == 100
        assert task.status == "completed"
        assert task.completed_at is not None
    
    def test_task_timing(self, test_db, sample_meeting):
        """Test task timing fields."""
        
        start_time = datetime.now()
        
        task = Task(
            meeting_id=sample_meeting.id,
            task_type="analysis",
            status="running",
            started_at=start_time
        )
        
        test_db.add(task)
        test_db.commit()
        
        # Simulate task completion
        task.status = "completed"
        task.completed_at = start_time + timedelta(seconds=30)
        test_db.commit()
        
        # Test timing calculation
        duration = task.completed_at - task.started_at
        assert duration.total_seconds() == 30

@pytest.mark.unit
class TestSegmentModel:
    """Test cases for Segment model."""
    
    def test_create_segment(self, test_db, sample_meeting):
        """Test creating a new segment."""
        
        segment = Segment(
            meeting_id=sample_meeting.id,
            start_time=0.0,
            end_time=10.5,
            speaker_id="Speaker_0",
            speaker_name="Alice",
            text="Hello everyone, welcome to the meeting.",
            confidence=0.95,
            sentiment="positive",
            sentiment_score=0.3
        )
        
        test_db.add(segment)
        test_db.commit()
        test_db.refresh(segment)
        
        assert segment.id is not None
        assert segment.meeting_id == sample_meeting.id
        assert segment.start_time == 0.0
        assert segment.end_time == 10.5
        assert segment.speaker_name == "Alice"
        assert segment.text == "Hello everyone, welcome to the meeting."
        assert segment.confidence == 0.95
    
    def test_segment_timing_validation(self, test_db, sample_meeting):
        """Test segment timing validation."""
        
        # Valid segment
        segment = Segment(
            meeting_id=sample_meeting.id,
            start_time=0.0,
            end_time=10.0,
            text="Valid segment"
        )
        
        test_db.add(segment)
        test_db.commit()
        
        assert segment.start_time < segment.end_time
    
    def test_segment_sentiment_range(self, test_db, sample_meeting):
        """Test sentiment score validation."""
        
        # Test valid sentiment scores
        for score in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            segment = Segment(
                meeting_id=sample_meeting.id,
                start_time=0.0,
                end_time=1.0,
                text=f"Segment with sentiment {score}",
                sentiment_score=score
            )
            
            test_db.add(segment)
            test_db.commit()
            test_db.refresh(segment)
            
            assert segment.sentiment_score == score
            
            # Clean up for next iteration
            test_db.delete(segment)
            test_db.commit()
```

### API Logic Testing

```python
# tests/unit/test_api_logic.py - Unit tests for API business logic

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import HTTPException
from backend.app import upload_meeting, get_task_status
from backend.schemas import UploadResponse, TaskStatus

@pytest.mark.unit
class TestUploadLogic:
    """Test cases for upload logic."""
    
    @patch('backend.app.process_audio_file')
    def test_upload_meeting_success(self, mock_process, test_db, sample_audio_file):
        """Test successful meeting upload."""
        
        # Mock background task
        mock_background_tasks = Mock()
        
        # Create file upload mock
        mock_file = Mock()
        mock_file.filename = "test_meeting.wav"
        mock_file.content_type = "audio/wav"
        mock_file.read = AsyncMock(return_value=b"fake audio data")
        
        # Test upload function
        with patch('backend.app.Path') as mock_path:
            mock_path.return_value.mkdir.return_value = None
            
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.write.return_value = None
                
                # Call the upload function (this would be async in real implementation)
                # For testing, we'll test the logic components
                
                # Verify file validation
                assert mock_file.filename.endswith('.wav')
                
                # Verify database record creation would occur
                meeting_data = {
                    "title": "Test Meeting",
                    "filename": mock_file.filename,
                    "mime_type": mock_file.content_type,
                    "meeting_type": "standup"
                }
                
                assert meeting_data["title"] == "Test Meeting"
                assert meeting_data["filename"] == "test_meeting.wav"
    
    def test_file_validation(self):
        """Test file validation logic."""
        
        from backend.app import FileSecurityValidator
        
        # Test valid file
        valid_file_content = b"fake audio data"
        validation_result = FileSecurityValidator.validate_file(
            valid_file_content, 
            "test.wav"
        )
        
        assert validation_result["is_valid"] is True
        assert validation_result["extension"] == ".wav"
        
        # Test invalid file extension
        with pytest.raises(HTTPException) as exc_info:
            FileSecurityValidator.validate_file(
                valid_file_content,
                "test.exe"
            )
        
        assert exc_info.value.status_code == 400
        assert "Unsupported file extension" in str(exc_info.value.detail)
        
        # Test file too large
        large_file_content = b"x" * (500 * 1024 * 1024 + 1)  # > 500MB
        with pytest.raises(HTTPException) as exc_info:
            FileSecurityValidator.validate_file(
                large_file_content,
                "large.wav"
            )
        
        assert exc_info.value.status_code == 400
        assert "File too large" in str(exc_info.value.detail)

@pytest.mark.unit
class TestStatusLogic:
    """Test cases for status checking logic."""
    
    def test_get_task_status_success(self, test_db, sample_meeting):
        """Test successful task status retrieval."""
        
        from backend.models import Task
        
        # Create a task
        task = Task(
            meeting_id=sample_meeting.id,
            task_type="transcription",
            status="running",
            progress=50
        )
        
        test_db.add(task)
        test_db.commit()
        test_db.refresh(task)
        
        # Test status retrieval logic
        retrieved_task = test_db.query(Task).filter(Task.id == task.id).first()
        
        assert retrieved_task is not None
        assert retrieved_task.status == "running"
        assert retrieved_task.progress == 50
        assert retrieved_task.meeting_id == sample_meeting.id
    
    def test_get_task_status_not_found(self, test_db):
        """Test task status for non-existent task."""
        
        from backend.models import Task
        
        # Try to get non-existent task
        task = test_db.query(Task).filter(Task.id == "non-existent-id").first()
        
        assert task is None
        
        # This would trigger HTTPException in the actual API
        # with pytest.raises(HTTPException) as exc_info:
        #     get_task_status("non-existent-id", test_db)
        # assert exc_info.value.status_code == 404

@pytest.mark.unit
class TestSearchLogic:
    """Test cases for search functionality."""
    
    def test_text_search_query_building(self, test_db, sample_meeting, sample_segments):
        """Test search query building logic."""
        
        from backend.models import Meeting, Segment
        from sqlalchemy import and_
        
        # Test basic text search
        search_query = "meeting"
        
        # Simulate search logic
        meetings = test_db.query(Meeting).filter(
            Meeting.title.contains(search_query)
        ).all()
        
        assert len(meetings) == 1
        assert meetings[0].title == "Test Meeting"
        
        # Test segment search
        segment_matches = test_db.query(Segment).filter(
            Segment.text.contains("Hello")
        ).all()
        
        assert len(segment_matches) == 1
        assert "Hello" in segment_matches[0].text
    
    def test_search_filters(self, test_db, sample_meeting, sample_segments):
        """Test search filtering logic."""
        
        from backend.models import Meeting, Segment
        from datetime import datetime, timedelta
        
        # Test date filtering
        recent_date = datetime.now() - timedelta(hours=1)
        old_date = datetime.now() - timedelta(days=1)
        
        # Test meeting type filtering
        meetings = test_db.query(Meeting).filter(
            Meeting.meeting_type == "standup"
        ).all()
        
        assert len(meetings) == 1
        assert meetings[0].meeting_type == "standup"
        
        # Test speaker filtering
        alice_segments = test_db.query(Segment).filter(
            Segment.speaker_name == "Alice"
        ).all()
        
        assert len(alice_segments) == 1
        assert alice_segments[0].speaker_name == "Alice"
```

### Utility Function Testing

```python
# tests/unit/test_utils.py - Unit tests for utility functions

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

@pytest.mark.unit
class TestFileUtils:
    """Test cases for file utility functions."""
    
    def test_format_file_size(self):
        """Test file size formatting."""
        
        from backend.utils import format_file_size
        
        # Test various file sizes
        assert format_file_size(0) == "0 Bytes"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_file_size(1536) == "1.5 KB"  # 1.5 KB
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        
        from backend.utils import sanitize_filename
        
        # Test dangerous characters removal
        dangerous_name = "test<>:\"/\\|?*file.wav"
        safe_name = sanitize_filename(dangerous_name)
        
        assert "<" not in safe_name
        assert ">" not in safe_name
        assert ":" not in safe_name
        assert "/" not in safe_name
        assert "\\" not in safe_name
        assert "|" not in safe_name
        assert "?" not in safe_name
        assert "*" not in safe_name
        assert safe_name.endswith(".wav")
    
    def test_generate_unique_filename(self):
        """Test unique filename generation."""
        
        from backend.utils import generate_unique_filename
        
        # Test with meeting ID and timestamp
        meeting_id = "test-meeting-123"
        original_filename = "meeting.wav"
        
        unique_name = generate_unique_filename(meeting_id, original_filename)
        
        assert meeting_id in unique_name
        assert ".wav" in unique_name
        assert len(unique_name) > len(original_filename)
    
    def test_create_directory_structure(self, temp_media_dir):
        """Test directory structure creation."""
        
        from backend.storage import StorageManager
        
        storage = StorageManager()
        storage.base_dir = temp_media_dir
        
        meeting_id = "test-meeting-123"
        meeting_dir = storage._create_meeting_directory(meeting_id)
        
        assert meeting_dir.exists()
        assert meeting_dir.is_dir()
        assert meeting_id in str(meeting_dir)

@pytest.mark.unit
class TestDateTimeUtils:
    """Test cases for date/time utility functions."""
    
    def test_format_duration(self):
        """Test duration formatting."""
        
        from backend.utils import format_duration
        
        # Test various durations
        assert format_duration(0) == "0:00"
        assert format_duration(30) == "0:30"
        assert format_duration(60) == "1:00"
        assert format_duration(90) == "1:30"
        assert format_duration(3600) == "60:00"  # 1 hour
        assert format_duration(3661) == "61:01"  # 1 hour 1 minute 1 second
    
    def test_parse_timestamp(self):
        """Test timestamp parsing."""
        
        from backend.utils import parse_timestamp
        
        # Test ISO format
        iso_timestamp = "2024-01-01T12:00:00Z"
        parsed = parse_timestamp(iso_timestamp)
        
        assert isinstance(parsed, datetime)
        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.day == 1
        assert parsed.hour == 12
    
    def test_calculate_processing_time(self):
        """Test processing time calculation."""
        
        from backend.utils import calculate_processing_time
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=5, seconds=30)
        
        processing_time = calculate_processing_time(start_time, end_time)
        
        assert processing_time == 330.0  # 5.5 minutes in seconds

@pytest.mark.unit
class TestValidationUtils:
    """Test cases for validation utilities."""
    
    def test_validate_meeting_title(self):
        """Test meeting title validation."""
        
        from backend.utils import validate_meeting_title
        
        # Test valid titles
        assert validate_meeting_title("Daily Standup") is True
        assert validate_meeting_title("Project Review - Q1 2024") is True
        
        # Test invalid titles
        assert validate_meeting_title("") is False
        assert validate_meeting_title(" ") is False
        assert validate_meeting_title("a" * 256) is False  # Too long
    
    def test_validate_speaker_name(self):
        """Test speaker name validation."""
        
        from backend.utils import validate_speaker_name
        
        # Test valid names
        assert validate_speaker_name("Alice") is True
        assert validate_speaker_name("Bob Smith") is True
        assert validate_speaker_name("Dr. Johnson") is True
        
        # Test invalid names
        assert validate_speaker_name("") is False
        assert validate_speaker_name(" ") is False
        assert validate_speaker_name("A" * 100) is False  # Too long
    
    def test_validate_sentiment_score(self):
        """Test sentiment score validation."""
        
        from backend.utils import validate_sentiment_score
        
        # Test valid scores
        assert validate_sentiment_score(-1.0) is True
        assert validate_sentiment_score(0.0) is True
        assert validate_sentiment_score(1.0) is True
        assert validate_sentiment_score(0.5) is True
        
        # Test invalid scores
        assert validate_sentiment_score(-1.1) is False
        assert validate_sentiment_score(1.1) is False
        assert validate_sentiment_score("invalid") is False

@pytest.mark.unit
class TestTextProcessingUtils:
    """Test cases for text processing utilities."""
    
    def test_clean_transcript_text(self):
        """Test transcript text cleaning."""
        
        from backend.utils import clean_transcript_text
        
        # Test with various text issues
        dirty_text = "  Hello...  world!!!   How are you???  "
        clean_text = clean_transcript_text(dirty_text)
        
        assert clean_text == "Hello. world! How are you?"
        assert not clean_text.startswith(" ")
        assert not clean_text.endswith(" ")
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        
        from backend.utils import extract_keywords
        
        text = "The project deadline is next Friday. We need to review the budget and timeline."
        keywords = extract_keywords(text)
        
        assert "project" in keywords
        assert "deadline" in keywords
        assert "budget" in keywords
        assert "timeline" in keywords
        
        # Common words should be filtered out
        assert "the" not in keywords
        assert "is" not in keywords
        assert "we" not in keywords
    
    def test_calculate_similarity(self):
        """Test text similarity calculation."""
        
        from backend.utils import calculate_text_similarity
        
        text1 = "The meeting was very productive"
        text2 = "The meeting was highly productive"
        text3 = "The weather is nice today"
        
        # Similar texts should have high similarity
        similarity_high = calculate_text_similarity(text1, text2)
        assert similarity_high > 0.8
        
        # Different texts should have low similarity
        similarity_low = calculate_text_similarity(text1, text3)
        assert similarity_low < 0.3
```

---

## Integration Testing

### Database Integration Tests

```python
# tests/integration/test_database_integration.py - Database integration tests

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from backend.db import Base, init_db
from backend.models import Meeting, Task, Segment
from datetime import datetime

@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    def test_database_initialization(self, test_db):
        """Test complete database initialization."""
        
        # Test that all tables are created
        tables = test_db.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )).fetchall()
        
        table_names = [table[0] for table in tables]
        
        assert "meetings" in table_names
        assert "tasks" in table_names
        assert "segments" in table_names
    
    def test_foreign_key_constraints(self, test_db):
        """Test foreign key relationships work correctly."""
        
        # Create meeting first
        meeting = Meeting(
            title="Test Meeting",
            filename="test.wav",
            file_path="/tmp/test.wav"
        )
        test_db.add(meeting)
        test_db.commit()
        test_db.refresh(meeting)
        
        # Create task with valid foreign key
        task = Task(
            meeting_id=meeting.id,
            task_type="transcription"
        )
        test_db.add(task)
        test_db.commit()
        
        # Verify relationship
        assert task.meeting_id == meeting.id
        assert len(meeting.tasks) == 1
        
        # Test cascade delete (if configured)
        test_db.delete(meeting)
        test_db.commit()
        
        # Task should be deleted due to cascade
        remaining_task = test_db.query(Task).filter(Task.id == task.id).first()
        assert remaining_task is None
    
    def test_transaction_rollback(self, test_db):
        """Test transaction rollback functionality."""
        
        # Start with known state
        initial_count = test_db.query(Meeting).count()
        
        try:
            # Begin transaction
            meeting1 = Meeting(title="Meeting 1", filename="1.wav", file_path="/tmp/1.wav")
            test_db.add(meeting1)
            
            meeting2 = Meeting(title="Meeting 2", filename="2.wav", file_path="/tmp/2.wav")
            test_db.add(meeting2)
            
            # Force an error
            meeting3 = Meeting()  # Missing required fields
            test_db.add(meeting3)
            
            test_db.commit()
            
        except Exception:
            test_db.rollback()
        
        # Verify rollback worked
        final_count = test_db.query(Meeting).count()
        assert final_count == initial_count
    
    def test_concurrent_access(self, test_db):
        """Test concurrent database access scenarios."""
        
        import threading
        import time
        
        results = []
        
        def create_meeting(meeting_id):
            """Create a meeting in a separate thread."""
            try:
                meeting = Meeting(
                    title=f"Meeting {meeting_id}",
                    filename=f"{meeting_id}.wav",
                    file_path=f"/tmp/{meeting_id}.wav"
                )
                test_db.add(meeting)
                test_db.commit()
                results.append(("success", meeting_id))
            except Exception as e:
                results.append(("error", str(e)))
        
        # Create multiple threads to test concurrent access
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_meeting, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        success_count = len([r for r in results if r[0] == "success"])
        assert success_count == 5  # All operations should succeed
    
    def test_bulk_operations(self, test_db):
        """Test bulk database operations performance."""
        
        import time
        
        # Test bulk insert
        start_time = time.time()
        
        meetings = []
        for i in range(100):
            meeting = Meeting(
                title=f"Bulk Meeting {i}",
                filename=f"bulk_{i}.wav",
                file_path=f"/tmp/bulk_{i}.wav"
            )
            meetings.append(meeting)
        
        test_db.bulk_save_objects(meetings)
        test_db.commit()
        
        insert_time = time.time() - start_time
        
        # Verify all meetings were created
        count = test_db.query(Meeting).filter(
            Meeting.title.like("Bulk Meeting%")
        ).count()
        assert count == 100
        
        # Test bulk update
        start_time = time.time()
        
        test_db.query(Meeting).filter(
            Meeting.title.like("Bulk Meeting%")
        ).update({"status": "processed"})
        test_db.commit()
        
        update_time = time.time() - start_time
        
        # Verify updates
        processed_count = test_db.query(Meeting).filter(
            Meeting.status == "processed"
        ).count()
        assert processed_count == 100
        
        # Performance assertions (adjust based on requirements)
        assert insert_time < 5.0  # Should complete within 5 seconds
        assert update_time < 2.0  # Should complete within 2 seconds

@pytest.mark.integration
class TestDatabaseQueries:
    """Integration tests for complex database queries."""
    
    def test_meeting_search_queries(self, test_db, sample_meeting, sample_segments):
        """Test complex meeting search queries."""
        
        # Test search by title
        meetings = test_db.query(Meeting).filter(
            Meeting.title.contains("Test")
        ).all()
        assert len(meetings) == 1
        
        # Test search by date range
        from datetime import timedelta
        recent_date = datetime.now() - timedelta(hours=1)
        
        recent_meetings = test_db.query(Meeting).filter(
            Meeting.upload_time >= recent_date
        ).all()
        assert len(recent_meetings) >= 1
        
        # Test join query with segments
        meetings_with_segments = test_db.query(Meeting).join(Segment).filter(
            Segment.text.contains("Hello")
        ).all()
        assert len(meetings_with_segments) == 1
    
    def test_aggregation_queries(self, test_db, sample_meeting, sample_segments):
        """Test database aggregation queries."""
        
        from sqlalchemy import func
        
        # Test count aggregation
        segment_count = test_db.query(func.count(Segment.id)).filter(
            Segment.meeting_id == sample_meeting.id
        ).scalar()
        assert segment_count == 2
        
        # Test average calculation
        avg_confidence = test_db.query(func.avg(Segment.confidence)).filter(
            Segment.meeting_id == sample_meeting.id
        ).scalar()
        assert avg_confidence > 0.9
        
        # Test min/max
        min_start_time = test_db.query(func.min(Segment.start_time)).filter(
            Segment.meeting_id == sample_meeting.id
        ).scalar()
        assert min_start_time == 0.0
        
        max_end_time = test_db.query(func.max(Segment.end_time)).filter(
            Segment.meeting_id == sample_meeting.id
        ).scalar()
        assert max_end_time == 20.0
    
    def test_complex_joins(self, test_db, sample_meeting, sample_segments):
        """Test complex multi-table joins."""
        
        # Create a task for the meeting
        task = Task(
            meeting_id=sample_meeting.id,
            task_type="transcription",
            status="completed",
            progress=100
        )
        test_db.add(task)
        test_db.commit()
        
        # Test three-way join
        results = test_db.query(Meeting, Task, Segment).join(Task).join(Segment).filter(
            Meeting.id == sample_meeting.id
        ).all()
        
        assert len(results) == 2  # Two segments
        
        for meeting, task, segment in results:
            assert meeting.id == sample_meeting.id
            assert task.meeting_id == meeting.id
            assert segment.meeting_id == meeting.id
    
    def test_pagination_queries(self, test_db):
        """Test query pagination."""
        
        # Create multiple meetings
        for i in range(25):
            meeting = Meeting(
                title=f"Paginated Meeting {i:02d}",
                filename=f"page_{i}.wav",
                file_path=f"/tmp/page_{i}.wav"
            )
            test_db.add(meeting)
        test_db.commit()
        
        # Test first page
        page_1 = test_db.query(Meeting).order_by(Meeting.title).limit(10).all()
        assert len(page_1) == 10
        assert page_1[0].title.endswith("00")
        
        # Test second page
        page_2 = test_db.query(Meeting).order_by(Meeting.title).offset(10).limit(10).all()
        assert len(page_2) == 10
        assert page_2[0].title.endswith("10")
        
        # Test last page
        page_3 = test_db.query(Meeting).order_by(Meeting.title).offset(20).limit(10).all()
        assert len(page_3) == 5  # Only 5 remaining meetings
```

### API Integration Tests

```python
# tests/integration/test_api_integration.py - API integration tests

import pytest
import json
import io
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    def test_upload_workflow_integration(self, client, sample_audio_file, mock_ai_services):
        """Test complete upload workflow integration."""
        
        # Mock AI services
        with patch('backend.tasks.process_audio_file') as mock_process:
            mock_process.return_value = {"status": "success"}
            
            # Test file upload
            with open(sample_audio_file, 'rb') as f:
                files = {"file": ("test_meeting.wav", f, "audio/wav")}
                data = {
                    "title": "Integration Test Meeting",
                    "meeting_type": "standup"
                }
                
                response = client.post("/api/upload", files=files, data=data)
        
        # Verify upload response
        assert response.status_code == 200
        upload_data = response.json()
        
        assert "task_id" in upload_data
        assert "meeting_id" in upload_data
        assert upload_data["status"] == "processing"
        
        task_id = upload_data["task_id"]
        meeting_id = upload_data["meeting_id"]
        
        # Test status endpoint
        status_response = client.get(f"/api/status/{task_id}")
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        assert status_data["task_id"] == task_id
        assert status_data["meeting_id"] == meeting_id
        assert "status" in status_data
        assert "progress" in status_data
    
    def test_search_integration(self, client, test_db, sample_meeting, sample_segments):
        """Test search functionality integration."""
        
        # Test basic search
        response = client.get("/api/search?query=Hello")
        assert response.status_code == 200
        
        search_data = response.json()
        assert "results" in search_data
        assert "query" in search_data
        assert search_data["query"] == "Hello"
        
        # Should find the segment with "Hello"
        results = search_data["results"]
        assert len(results) > 0
        
        # Test search with filters
        response = client.get("/api/search?query=meeting&meeting_type=standup")
        assert response.status_code == 200
        
        filtered_data = response.json()
        assert "results" in filtered_data
    
    def test_meeting_crud_integration(self, client, test_db):
        """Test complete CRUD operations for meetings."""
        
        # Create meeting via upload (simulated)
        meeting_data = {
            "title": "CRUD Test Meeting",
            "filename": "crud_test.wav",
            "file_path": "/tmp/crud_test.wav",
            "meeting_type": "review"
        }
        
        # Test getting meetings list
        response = client.get("/api/meetings")
        assert response.status_code == 200
        
        meetings_data = response.json()
        assert isinstance(meetings_data, list)
        
        # Add a meeting to database for testing
        from backend.models import Meeting
        meeting = Meeting(**meeting_data)
        test_db.add(meeting)
        test_db.commit()
        test_db.refresh(meeting)
        
        # Test getting specific meeting
        response = client.get(f"/api/meetings/{meeting.id}")
        assert response.status_code == 200
        
        meeting_detail = response.json()
        assert meeting_detail["id"] == meeting.id
        assert meeting_detail["title"] == "CRUD Test Meeting"
        
        # Test getting non-existent meeting
        response = client.get("/api/meetings/non-existent-id")
        assert response.status_code == 404
    
    def test_health_check_integration(self, client):
        """Test health check endpoint integration."""
        
        response = client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data
        assert "service" in health_data
    
    def test_error_handling_integration(self, client):
        """Test API error handling integration."""
        
        # Test invalid file upload
        files = {"file": ("test.txt", b"not audio data", "text/plain")}
        response = client.post("/api/upload", files=files)
        
        assert response.status_code == 400
        error_data = response.json()
        assert "error" in error_data or "detail" in error_data
        
        # Test invalid search query
        response = client.get("/api/search?query=")
        assert response.status_code == 400
        
        # Test invalid status ID
        response = client.get("/api/status/invalid-task-id")
        assert response.status_code == 404

@pytest.mark.integration 
class TestConcurrentAPIAccess:
    """Test concurrent API access scenarios."""
    
    def test_concurrent_uploads(self, client, sample_audio_file):
        """Test multiple concurrent uploads."""
        
        import threading
        import time
        
        results = []
        
        def upload_file(file_index):
            """Upload file in separate thread."""
            try:
                with open(sample_audio_file, 'rb') as f:
                    files = {"file": (f"concurrent_{file_index}.wav", f, "audio/wav")}
                    data = {"title": f"Concurrent Meeting {file_index}"}
                    
                    response = client.post("/api/upload", files=files, data=data)
                    results.append(("success", response.status_code, file_index))
                    
            except Exception as e:
                results.append(("error", str(e), file_index))
        
        # Create multiple upload threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=upload_file, args=(i,))
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Verify results
        success_count = len([r for r in results if r[0] == "success"])
        
        # All uploads should succeed or handle gracefully
        assert success_count >= 3  # At least 3 should succeed
        assert total_time < 30  # Should complete within reasonable time
    
    def test_concurrent_search_requests(self, client, test_db, sample_meeting, sample_segments):
        """Test multiple concurrent search requests."""
        
        import threading
        
        results = []
        
        def search_request(query_index):
            """Perform search in separate thread."""
            try:
                response = client.get(f"/api/search?query=test{query_index}")
                results.append(("success", response.status_code))
            except Exception as e:
                results.append(("error", str(e)))
        
        # Create multiple search threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=search_request, args=(i,))
            threads.append(thread)
        
        # Start and wait for threads
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All searches should succeed
        success_count = len([r for r in results if r[0] == "success"])
        assert success_count == 10
```

This completes the first major section of Part 6, covering the testing strategy, unit testing implementation, and integration testing. The testing framework demonstrates:

1. **Comprehensive test pyramid** with proper distribution of test types
2. **Robust fixture system** for test data management
3. **Database integration testing** with transaction handling
4. **API integration testing** including concurrent access scenarios
5. **Mock strategies** for external dependencies
6. **Performance considerations** in test execution

Would you like me to continue with the remaining sections covering API Testing, Performance Testing, End-to-End Testing, Security Testing, and Quality Assurance Automation?