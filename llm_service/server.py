from fastapi import FastAPI
import httpx
import json

app = FastAPI()

@app.post("/api/extract")
def extract(body: dict):
    # Mock insight extraction for now
    text = body.get("text", "")
    meeting_id = body.get("meeting_id", 0)
    
    # Simple mock response
    mock_insights = {
        "action_items": [
            {
                "task": "Follow up on action items",
                "assignee": "Team Lead",
                "due_date": "2025-01-15",
                "priority": "high",
                "confidence": 0.9
            }
        ],
        "decisions": [
            {
                "decision": "Proceed with project implementation",
                "rationale": "Team consensus reached",
                "impact": "high",
                "confidence": 0.85
            }
        ],
        "sentiment": {
            "overall": "positive",
            "score": 0.7,
            "confidence": 0.8
        }
    }
    
    return mock_insights

@app.get("/health")
def health():
    return {"status": "healthy", "service": "llm-service"}
