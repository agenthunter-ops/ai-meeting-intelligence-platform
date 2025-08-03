from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import UploadResponse, StatusResponse, UploadRequest
from .tasks import process_meeting_upload
from .db import init_db, get_task_status
import aiofiles
import os
import uuid
from typing import Optional

app = FastAPI(title="AI Meeting Intelligence API")

app.add_middleware(CORSMiddleware, allow_origins=["*"])

try:
    init_db()  # Initialize DB; may fail until Postgres is up
except Exception as e:
    print(f"⚠️  Database init failed during startup: {e}. Will rely on migrations or later retries.")

async def save_temp(file: UploadFile) -> str:
    """Save uploaded file to temporary location and return path"""
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    file_path = os.path.join(temp_dir, f"{file_id}{file_extension}")
    
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return file_path

@app.post("/api/upload", response_model=UploadResponse)
async def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    meeting_type: Optional[str] = "general",
    attendees: Optional[str] = None,
):
    """Upload audio file for processing"""
    try:
        # Save uploaded file
        file_path = await save_temp(file)
        
        # Prepare meeting data
        meeting_data = {
            'title': title or file.filename or 'Untitled Meeting',
            'meeting_type': meeting_type,
            'attendees': attendees.split(',') if attendees else []
        }
        
        # Start background processing
        task = process_meeting_upload.delay(file_path, meeting_data)
        
        return UploadResponse(
            task_id=task.id,
            meeting_id=0,  # Will be set after meeting creation
            message="Upload successful, processing started"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/status/{task_id}", response_model=TaskStatus)
def status(task_id: str):
    """Get task status"""
    status_data = get_task_status(task_id)
    if not status_data:
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatus(**status_data)

@app.get("/")
def read_root():
    return {"message": "AI Meeting Intelligence API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
