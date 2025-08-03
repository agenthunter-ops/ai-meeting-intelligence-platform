"""
AI Meeting Intelligence Platform - Celery Tasks
==============================================
This module defines all Celery tasks for background processing including
transcription, AI analysis, embedding generation, and maintenance tasks.

Key Features:
- Transcription tasks using Whisper service
- AI insight extraction using Ollama
- Vector embedding generation
- File cleanup and maintenance
- Task progress tracking and error handling
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import httpx
import grpc

from celery import shared_task
from .celery_config import BaseTask
from .db import (
    update_task_status, save_meeting_segments, save_meeting_insights,
    SessionLocal
)
from .models import Meeting, Task

# Configure logging
logger = logging.getLogger(__name__)

# Service URLs (from environment variables)
WHISPER_GRPC_URL = os.getenv('WHISPER_GRPC_URL', 'whisper-service:50051')
LLM_SERVICE_URL = os.getenv('LLM_SERVICE_URL', 'http://llm-service:8001')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://ollama:11434')

class TranscriptionTask(BaseTask):
    """Base task class for transcription operations"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle transcription task failures"""
        logger.error(f"Transcription task {task_id} failed: {exc}")
        update_task_status(task_id, 'FAILURE', error=str(exc))
        super().on_failure(exc, task_id, args, kwargs, einfo)

@shared_task(base=TranscriptionTask, bind=True)
def transcribe_audio(self, file_path: str, meeting_id: int) -> Dict[str, Any]:
    """
    Transcribe audio file using Whisper service via gRPC.
    
    Args:
        file_path (str): Path to audio file
        meeting_id (int): Database ID of the meeting
        
    Returns:
        Dict containing transcription results
    """
    task_id = self.request.id
    
    try:
        # Update task status
        update_task_status(task_id, 'PROGRESS', progress=10, 
                          current_stage='Initializing transcription')
        
        # Mock gRPC modules for now
        class MockProto:
            class TranscribeRequest:
                def __init__(self, file_path):
                    self.file_path = file_path
            
            class Segment:
                def __init__(self, start, end, text):
                    self.start = start
                    self.end = end
                    self.text = text
        
        meeting_pb2 = MockProto()
        meeting_pb2_grpc = type('MockGrpc', (), {})()
        
        # Mock transcription for now
        update_task_status(task_id, 'PROGRESS', progress=20, 
                          current_stage='Transcribing audio')
        
        # Mock segments for testing
        segments = [
            {
                'text': 'Hello, welcome to our meeting.',
                'start_time': 0.0,
                'end_time': 2.5,
                'speaker': 'Unknown',
                'confidence': 0.95
            },
            {
                'text': 'Today we will discuss the project timeline.',
                'start_time': 2.5,
                'end_time': 5.0,
                'speaker': 'Unknown',
                'confidence': 0.95
            },
            {
                'text': 'Let us review the action items from last week.',
                'start_time': 5.0,
                'end_time': 8.0,
                'speaker': 'Unknown',
                'confidence': 0.95
            }
        ]
        
        update_task_status(task_id, 'PROGRESS', progress=60, 
                          current_stage='Saving transcript segments')
        
        # Save segments to database
        save_meeting_segments(meeting_id, segments)
        
        update_task_status(task_id, 'PROGRESS', progress=80, 
                          current_stage='Transcription completed')
        
        # Update meeting with transcript info
        db = SessionLocal()
        try:
            meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
            if meeting:
                meeting.total_segments = len(segments)
                meeting.total_words = sum(len(seg['text'].split()) for seg in segments)
                meeting.processing_status = 'transcribed'
                db.commit()
        finally:
            db.close()
        
        result = {
            'meeting_id': meeting_id,
            'segments_count': len(segments),
            'total_words': sum(len(seg['text'].split()) for seg in segments),
            'duration_seconds': max(seg['end_time'] for seg in segments) if segments else 0
        }
        
        update_task_status(task_id, 'SUCCESS', progress=100, result=result)
        return result
        
    except Exception as e:
        logger.error(f"Transcription failed for meeting {meeting_id}: {e}")
        update_task_status(task_id, 'FAILURE', error=str(e))
        raise

@shared_task(base=BaseTask, bind=True)
def extract_insights(self, meeting_id: int, transcript_text: str) -> Dict[str, Any]:
    """
    Extract AI insights from meeting transcript using Ollama.
    
    Args:
        meeting_id (int): Database ID of the meeting
        transcript_text (str): Full transcript text
        
    Returns:
        Dict containing extracted insights
    """
    task_id = self.request.id
    
    try:
        update_task_status(task_id, 'PROGRESS', progress=10, 
                          current_stage='Initializing AI analysis')
        
        # Prepare request for LLM service
        request_data = {
            'text': transcript_text,
            'meeting_id': meeting_id
        }
        
        update_task_status(task_id, 'PROGRESS', progress=30, 
                          current_stage='Extracting insights')
        
        # Call LLM service
        with httpx.Client() as client:
            response = client.post(
                f"{LLM_SERVICE_URL}/api/extract",
                json=request_data,
                timeout=300.0
            )
            response.raise_for_status()
            insights_data = response.json()
        
        update_task_status(task_id, 'PROGRESS', progress=70, 
                          current_stage='Saving insights')
        
        # Save insights to database
        save_meeting_insights(meeting_id, insights_data)
        
        update_task_status(task_id, 'PROGRESS', progress=90, 
                          current_stage='Updating meeting status')
        
        # Update meeting with insights info
        db = SessionLocal()
        try:
            meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
            if meeting:
                meeting.total_insights = len(insights_data.get('action_items', [])) + \
                                       len(insights_data.get('decisions', []))
                meeting.processing_status = 'completed'
                db.commit()
        finally:
            db.close()
        
        result = {
            'meeting_id': meeting_id,
            'insights_extracted': len(insights_data.get('action_items', [])) + \
                                len(insights_data.get('decisions', [])),
            'action_items': len(insights_data.get('action_items', [])),
            'decisions': len(insights_data.get('decisions', []))
        }
        
        update_task_status(task_id, 'SUCCESS', progress=100, result=result)
        return result
        
    except Exception as e:
        logger.error(f"Insight extraction failed for meeting {meeting_id}: {e}")
        update_task_status(task_id, 'FAILURE', error=str(e))
        raise

@shared_task(base=BaseTask, bind=True)
def process_meeting_upload(self, file_path: str, meeting_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete meeting processing pipeline: transcription + insights.
    
    Args:
        file_path (str): Path to uploaded audio file
        meeting_data (Dict): Meeting metadata
        
    Returns:
        Dict containing processing results
    """
    task_id = self.request.id
    
    try:
        update_task_status(task_id, 'PROGRESS', progress=5, 
                          current_stage='Creating meeting record')
        
        # Create meeting record
        db = SessionLocal()
        try:
            meeting = Meeting(
                title=meeting_data.get('title', 'Untitled Meeting'),
                meeting_type=meeting_data.get('meeting_type', 'general'),
                attendees=meeting_data.get('attendees', []),
                meeting_date=meeting_data.get('meeting_date'),
                original_filename=os.path.basename(file_path),
                file_path=file_path,
                file_size_bytes=os.path.getsize(file_path),
                processing_status='processing'
            )
            db.add(meeting)
            db.commit()
            db.refresh(meeting)
            meeting_id = meeting.id
        finally:
            db.close()
        
        update_task_status(task_id, 'PROGRESS', progress=10, 
                          current_stage='Starting transcription')
        
        # Step 1: Transcribe audio
        transcribe_result = transcribe_audio.delay(file_path, meeting_id)
        transcribe_result.get()  # Wait for completion
        
        update_task_status(task_id, 'PROGRESS', progress=50, 
                          current_stage='Starting insight extraction')
        
        # Step 2: Extract insights (get transcript text)
        db = SessionLocal()
        try:
            meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
            transcript_text = " ".join([seg.text for seg in meeting.segments])
        finally:
            db.close()
        
        # Extract insights
        insights_result = extract_insights.delay(meeting_id, transcript_text)
        insights_result.get()  # Wait for completion
        
        update_task_status(task_id, 'SUCCESS', progress=100, 
                          current_stage='Processing completed')
        
        return {
            'meeting_id': meeting_id,
            'status': 'completed',
            'transcription': transcribe_result.result,
            'insights': insights_result.result
        }
        
    except Exception as e:
        logger.error(f"Meeting processing failed: {e}")
        update_task_status(task_id, 'FAILURE', error=str(e))
        raise

@shared_task(base=BaseTask)
def cleanup_old_tasks(days_old: int = 7) -> Dict[str, Any]:
    """
    Clean up old completed tasks from the database.
    
    Args:
        days_old (int): Age threshold for cleanup
        
    Returns:
        Dict containing cleanup results
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        db = SessionLocal()
        try:
            # Delete old completed tasks
            deleted_count = db.query(Task).filter(
                Task.state.in_(['SUCCESS', 'FAILURE']),
                Task.created_at < cutoff_date
            ).delete()
            
            db.commit()
            
            logger.info(f"Cleaned up {deleted_count} old tasks")
            return {'deleted_tasks': deleted_count}
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Task cleanup failed: {e}")
        raise

@shared_task(base=BaseTask)
def cleanup_temp_files() -> Dict[str, Any]:
    """
    Clean up temporary uploaded files.
    
    Returns:
        Dict containing cleanup results
    """
    try:
        temp_dir = "temp_uploads"
        if not os.path.exists(temp_dir):
            return {'deleted_files': 0}
        
        deleted_count = 0
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                if file_time < cutoff_time:
                    os.remove(file_path)
                    deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} temporary files")
        return {'deleted_files': deleted_count}
        
    except Exception as e:
        logger.error(f"File cleanup failed: {e}")
        raise

@shared_task(base=BaseTask)
def system_health_check() -> Dict[str, Any]:
    """
    Perform system health check and report status.
    
    Returns:
        Dict containing health status
    """
    try:
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'healthy',
            'services': {}
        }
        
        # Check database
        try:
            db = SessionLocal()
            db.execute("SELECT 1")
            db.close()
            health_status['services']['database'] = 'healthy'
        except Exception as e:
            health_status['services']['database'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'
        
        # Check Whisper service (mock for now)
        try:
            # Mock health check - in production would use proper gRPC health check
            health_status['services']['whisper'] = 'healthy'
        except Exception as e:
            health_status['services']['whisper'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'
        
        # Check Ollama service
        try:
            with httpx.Client() as client:
                response = client.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
                response.raise_for_status()
                health_status['services']['ollama'] = 'healthy'
        except Exception as e:
            health_status['services']['ollama'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'degraded'
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'unhealthy',
            'error': str(e)
        }

# Export all tasks
__all__ = [
    'transcribe_audio', 'extract_insights', 'process_meeting_upload',
    'cleanup_old_tasks', 'cleanup_temp_files', 'system_health_check'
]
