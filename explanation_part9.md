# AI Meeting Intelligence Platform - Part 9: Advanced Features & Integrations

## Table of Contents
1. [Real-time Processing Pipeline](#real-time-processing-pipeline)
2. [Advanced Analytics Engine](#advanced-analytics-engine)
3. [Third-party Integrations](#third-party-integrations)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Workflow Automation](#workflow-automation)
6. [API Extensions](#api-extensions)
7. [Mobile & Cross-platform Support](#mobile--cross-platform-support)
8. [Enterprise Features](#enterprise-features)

---

## Real-time Processing Pipeline

### Event-Driven Architecture

Our real-time processing pipeline uses an event-driven architecture to handle live meeting analysis, real-time transcription, and instant insights generation.

```python
# backend/realtime/event_processor.py - Real-time event processing system

import asyncio
import websockets
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import redis
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class EventType(Enum):
    MEETING_STARTED = "meeting_started"
    MEETING_ENDED = "meeting_ended"
    SPEAKER_CHANGED = "speaker_changed"
    TRANSCRIPT_UPDATE = "transcript_update"
    SENTIMENT_CHANGE = "sentiment_change"
    KEYWORD_DETECTED = "keyword_detected"
    ACTION_ITEM_DETECTED = "action_item_detected"
    QUESTION_ASKED = "question_asked"
    DECISION_MADE = "decision_made"

@dataclass
class RealTimeEvent:
    """Represents a real-time event in the meeting processing pipeline."""
    event_type: EventType
    meeting_id: str
    timestamp: datetime
    data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "meeting_id": self.meeting_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "user_id": self.user_id,
            "session_id": self.session_id
        }

class EventBus:
    """Central event bus for real-time event distribution."""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0")
        )
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: Dict[str, List[RealTimeEvent]] = {}
        self.max_history_size = 1000
        
    async def publish(self, event: RealTimeEvent):
        """Publish event to all subscribers."""
        
        # Store in history
        if event.meeting_id not in self.event_history:
            self.event_history[event.meeting_id] = []
        
        self.event_history[event.meeting_id].append(event)
        
        # Trim history if too large
        if len(self.event_history[event.meeting_id]) > self.max_history_size:
            self.event_history[event.meeting_id] = \
                self.event_history[event.meeting_id][-self.max_history_size:]
        
        # Publish to Redis for distributed systems
        await self._publish_to_redis(event)
        
        # Notify local subscribers
        await self._notify_subscribers(event)
        
        logger.debug(f"Published event: {event.event_type.value} for meeting {event.meeting_id}")
    
    async def _publish_to_redis(self, event: RealTimeEvent):
        """Publish event to Redis channels."""
        
        channels = [
            f"meeting:{event.meeting_id}",
            f"event_type:{event.event_type.value}",
            "all_events"
        ]
        
        event_data = json.dumps(event.to_dict())
        
        for channel in channels:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.redis_client.publish,
                channel,
                event_data
            )
    
    async def _notify_subscribers(self, event: RealTimeEvent):
        """Notify local event subscribers."""
        
        # Notify meeting-specific subscribers
        meeting_subscribers = self.subscribers.get(f"meeting:{event.meeting_id}", [])
        
        # Notify event-type subscribers
        type_subscribers = self.subscribers.get(f"event_type:{event.event_type.value}", [])
        
        # Notify global subscribers
        global_subscribers = self.subscribers.get("all_events", [])
        
        all_subscribers = meeting_subscribers + type_subscribers + global_subscribers
        
        for subscriber in all_subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(event)
                else:
                    subscriber(event)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def subscribe(self, channel: str, callback: Callable):
        """Subscribe to events on a specific channel."""
        
        if channel not in self.subscribers:
            self.subscribers[channel] = []
        
        self.subscribers[channel].append(callback)
        
        logger.info(f"Added subscriber to channel: {channel}")
    
    def get_event_history(self, meeting_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history for a meeting."""
        
        events = self.event_history.get(meeting_id, [])
        recent_events = events[-limit:] if len(events) > limit else events
        
        return [event.to_dict() for event in recent_events]

class RealTimeTranscriptionProcessor:
    """Processes real-time audio streams for live transcription."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.whisper_processor = None  # Initialize with optimized Whisper
        self.buffer_duration = 5.0  # seconds
        
    async def start_session(self, meeting_id: str, session_config: Dict[str, Any]):
        """Start a real-time transcription session."""
        
        session_id = f"session_{meeting_id}_{datetime.now().timestamp()}"
        
        self.active_sessions[session_id] = {
            "meeting_id": meeting_id,
            "start_time": datetime.now(),
            "audio_buffer": [],
            "current_speaker": None,
            "last_transcript_time": datetime.now(),
            "config": session_config
        }
        
        # Publish session started event
        await self.event_bus.publish(RealTimeEvent(
            event_type=EventType.MEETING_STARTED,
            meeting_id=meeting_id,
            timestamp=datetime.now(),
            data={"session_id": session_id, "config": session_config}
        ))
        
        logger.info(f"Started real-time session: {session_id}")
        return session_id
    
    async def process_audio_chunk(self, session_id: str, audio_data: bytes, timestamp: float):
        """Process incoming audio chunk for real-time transcription."""
        
        if session_id not in self.active_sessions:
            logger.warning(f"Unknown session: {session_id}")
            return
        
        session = self.active_sessions[session_id]
        
        # Add to buffer
        session["audio_buffer"].append({
            "data": audio_data,
            "timestamp": timestamp
        })
        
        # Check if buffer is ready for processing
        if self._should_process_buffer(session):
            await self._process_buffer(session_id, session)
    
    def _should_process_buffer(self, session: Dict[str, Any]) -> bool:
        """Check if audio buffer is ready for processing."""
        
        if not session["audio_buffer"]:
            return False
        
        # Process when buffer duration reaches threshold
        buffer_start = session["audio_buffer"][0]["timestamp"]
        buffer_end = session["audio_buffer"][-1]["timestamp"]
        buffer_duration = buffer_end - buffer_start
        
        return buffer_duration >= self.buffer_duration
    
    async def _process_buffer(self, session_id: str, session: Dict[str, Any]):
        """Process accumulated audio buffer."""
        
        try:
            # Combine audio chunks
            audio_data = b"".join([chunk["data"] for chunk in session["audio_buffer"]])
            
            # Transcribe using optimized Whisper
            transcription_result = await self._transcribe_audio(audio_data)
            
            if transcription_result["text"].strip():
                # Detect speaker changes
                current_speaker = self._detect_speaker(transcription_result)
                
                if current_speaker != session["current_speaker"]:
                    await self.event_bus.publish(RealTimeEvent(
                        event_type=EventType.SPEAKER_CHANGED,
                        meeting_id=session["meeting_id"],
                        timestamp=datetime.now(),
                        data={
                            "previous_speaker": session["current_speaker"],
                            "new_speaker": current_speaker
                        }
                    ))
                    session["current_speaker"] = current_speaker
                
                # Publish transcript update
                await self.event_bus.publish(RealTimeEvent(
                    event_type=EventType.TRANSCRIPT_UPDATE,
                    meeting_id=session["meeting_id"],
                    timestamp=datetime.now(),
                    data={
                        "speaker": current_speaker,
                        "text": transcription_result["text"],
                        "confidence": transcription_result["confidence"],
                        "start_time": session["audio_buffer"][0]["timestamp"],
                        "end_time": session["audio_buffer"][-1]["timestamp"]
                    }
                ))
                
                # Analyze for keywords and insights
                await self._analyze_transcript_realtime(
                    session["meeting_id"],
                    transcription_result["text"],
                    current_speaker
                )
            
            # Clear processed buffer
            session["audio_buffer"] = []
            session["last_transcript_time"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}")
    
    async def _transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe audio data using optimized Whisper."""
        
        # Convert audio data to numpy array
        import numpy as np
        import io
        import soundfile as sf
        
        # Decode audio
        audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
        
        # Use optimized Whisper processor
        if self.whisper_processor is None:
            from .ai_optimization import OptimizedWhisperProcessor
            self.whisper_processor = OptimizedWhisperProcessor()
        
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            self.whisper_processor._transcribe_single_segment,
            audio_array,
            {"realtime": True}
        )
        
        return result
    
    def _detect_speaker(self, transcription_result: Dict[str, Any]) -> str:
        """Detect speaker from transcription result."""
        
        # Simple speaker detection based on voice characteristics
        # In production, this would use speaker diarization
        
        segments = transcription_result.get("segments", [])
        if segments and "speaker" in segments[0]:
            return segments[0]["speaker"]
        
        return "Unknown Speaker"
    
    async def _analyze_transcript_realtime(self, meeting_id: str, text: str, speaker: str):
        """Analyze transcript for real-time insights."""
        
        # Keyword detection
        keywords = await self._detect_keywords(text)
        for keyword in keywords:
            await self.event_bus.publish(RealTimeEvent(
                event_type=EventType.KEYWORD_DETECTED,
                meeting_id=meeting_id,
                timestamp=datetime.now(),
                data={
                    "keyword": keyword,
                    "speaker": speaker,
                    "context": text
                }
            ))
        
        # Action item detection
        action_items = await self._detect_action_items(text)
        for action_item in action_items:
            await self.event_bus.publish(RealTimeEvent(
                event_type=EventType.ACTION_ITEM_DETECTED,
                meeting_id=meeting_id,
                timestamp=datetime.now(),
                data={
                    "action_item": action_item,
                    "speaker": speaker,
                    "context": text
                }
            ))
        
        # Question detection
        if self._is_question(text):
            await self.event_bus.publish(RealTimeEvent(
                event_type=EventType.QUESTION_ASKED,
                meeting_id=meeting_id,
                timestamp=datetime.now(),
                data={
                    "question": text,
                    "speaker": speaker
                }
            ))
    
    async def _detect_keywords(self, text: str) -> List[str]:
        """Detect important keywords in text."""
        
        # Simple keyword detection - in production, use NLP models
        important_keywords = [
            "deadline", "priority", "urgent", "action", "decision",
            "budget", "cost", "revenue", "milestone", "deliverable"
        ]
        
        text_lower = text.lower()
        detected = [keyword for keyword in important_keywords if keyword in text_lower]
        
        return detected
    
    async def _detect_action_items(self, text: str) -> List[str]:
        """Detect action items in text."""
        
        action_patterns = [
            r"(?:we need to|should|must|will|going to)\s+(.+?)(?:\.|$)",
            r"(?:action item|todo|task):\s*(.+?)(?:\.|$)",
            r"(?:assigned to|responsible for)\s+(.+?)(?:\.|$)"
        ]
        
        import re
        action_items = []
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            action_items.extend([match.strip() for match in matches])
        
        return action_items
    
    def _is_question(self, text: str) -> bool:
        """Check if text contains a question."""
        
        question_indicators = [
            "?", "what", "how", "when", "where", "why", "who",
            "can we", "should we", "do you think", "any thoughts"
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in question_indicators)

class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Subscribe to events
        self.event_bus.subscribe("all_events", self._broadcast_event)
    
    async def register_connection(self, websocket: websockets.WebSocketServerProtocol, user_id: str, meeting_id: str = None):
        """Register a new WebSocket connection."""
        
        connection_id = f"conn_{user_id}_{datetime.now().timestamp()}"
        
        self.connections[connection_id] = websocket
        self.connection_metadata[connection_id] = {
            "user_id": user_id,
            "meeting_id": meeting_id,
            "connected_at": datetime.now(),
            "last_ping": datetime.now()
        }
        
        logger.info(f"WebSocket connection registered: {connection_id}")
        
        # Send connection confirmation
        await self._send_message(connection_id, {
            "type": "connection_confirmed",
            "connection_id": connection_id,
            "timestamp": datetime.now().isoformat()
        })
        
        return connection_id
    
    async def unregister_connection(self, connection_id: str):
        """Unregister a WebSocket connection."""
        
        if connection_id in self.connections:
            del self.connections[connection_id]
        
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]
        
        logger.info(f"WebSocket connection unregistered: {connection_id}")
    
    async def _broadcast_event(self, event: RealTimeEvent):
        """Broadcast event to relevant WebSocket connections."""
        
        message = {
            "type": "event",
            "event": event.to_dict()
        }
        
        # Send to connections interested in this meeting
        for connection_id, metadata in self.connection_metadata.items():
            should_send = (
                metadata["meeting_id"] == event.meeting_id or
                metadata["meeting_id"] is None  # Global listeners
            )
            
            if should_send:
                await self._send_message(connection_id, message)
    
    async def _send_message(self, connection_id: str, message: Dict[str, Any]):
        """Send message to specific WebSocket connection."""
        
        if connection_id not in self.connections:
            return
        
        try:
            websocket = self.connections[connection_id]
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            await self.unregister_connection(connection_id)
    
    async def handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection."""
        
        connection_id = None
        
        try:
            # Wait for authentication message
            auth_message = await websocket.recv()
            auth_data = json.loads(auth_message)
            
            # Validate authentication
            user_id = await self._authenticate_websocket(auth_data)
            meeting_id = auth_data.get("meeting_id")
            
            # Register connection
            connection_id = await self.register_connection(websocket, user_id, meeting_id)
            
            # Handle messages
            async for message in websocket:
                await self._handle_message(connection_id, json.loads(message))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if connection_id:
                await self.unregister_connection(connection_id)
    
    async def _authenticate_websocket(self, auth_data: Dict[str, Any]) -> str:
        """Authenticate WebSocket connection."""
        
        token = auth_data.get("token")
        if not token:
            raise ValueError("Authentication token required")
        
        # Verify JWT token
        try:
            payload = auth_manager.verify_token(token)
            return payload["sub"]
        except Exception:
            raise ValueError("Invalid authentication token")
    
    async def _handle_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        
        message_type = message.get("type")
        
        if message_type == "ping":
            # Update last ping time
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["last_ping"] = datetime.now()
            
            # Send pong response
            await self._send_message(connection_id, {"type": "pong"})
        
        elif message_type == "subscribe_meeting":
            # Subscribe to specific meeting events
            meeting_id = message.get("meeting_id")
            if meeting_id and connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["meeting_id"] = meeting_id
        
        elif message_type == "audio_chunk":
            # Handle real-time audio for transcription
            await self._handle_audio_chunk(connection_id, message)
    
    async def _handle_audio_chunk(self, connection_id: str, message: Dict[str, Any]):
        """Handle real-time audio chunk for transcription."""
        
        metadata = self.connection_metadata.get(connection_id)
        if not metadata or not metadata.get("meeting_id"):
            return
        
        # Extract audio data
        audio_data = message.get("audio_data")  # Base64 encoded
        timestamp = message.get("timestamp", datetime.now().timestamp())
        
        if audio_data:
            import base64
            audio_bytes = base64.b64decode(audio_data)
            
            # Process with real-time transcription
            session_id = f"realtime_{metadata['meeting_id']}"
            await realtime_processor.process_audio_chunk(session_id, audio_bytes, timestamp)

# Global instances
event_bus = EventBus()
realtime_processor = RealTimeTranscriptionProcessor(event_bus)
websocket_manager = WebSocketManager(event_bus)
```

---

## Advanced Analytics Engine

### Meeting Intelligence Analytics

```python
# backend/analytics/intelligence_engine.py - Advanced meeting analytics

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)

class MeetingIntelligenceEngine:
    """Advanced analytics engine for meeting intelligence."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.topic_model = None
        self.network_analyzer = NetworkAnalyzer()
        self.pattern_detector = PatternDetector()
        
    def analyze_meeting_comprehensive(self, meeting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis of a meeting."""
        
        analysis_results = {
            "meeting_id": meeting_data.get("id"),
            "analysis_timestamp": datetime.now().isoformat(),
            "metrics": {},
            "insights": {},
            "recommendations": []
        }
        
        # Extract segments and transcript
        segments = meeting_data.get("segments", [])
        transcript = meeting_data.get("transcript", "")
        
        if not segments:
            return analysis_results
        
        # Participation analysis
        analysis_results["metrics"]["participation"] = self._analyze_participation(segments)
        
        # Communication patterns
        analysis_results["metrics"]["communication"] = self._analyze_communication_patterns(segments)
        
        # Topic analysis
        analysis_results["metrics"]["topics"] = self._analyze_topics(transcript, segments)
        
        # Sentiment flow
        analysis_results["metrics"]["sentiment_flow"] = self._analyze_sentiment_flow(segments)
        
        # Decision tracking
        analysis_results["insights"]["decisions"] = self._extract_decisions(segments)
        
        # Action items
        analysis_results["insights"]["action_items"] = self._extract_action_items(segments)
        
        # Meeting effectiveness
        analysis_results["metrics"]["effectiveness"] = self._calculate_meeting_effectiveness(
            segments, analysis_results["metrics"]
        )
        
        # Generate recommendations
        analysis_results["recommendations"] = self._generate_recommendations(analysis_results)
        
        return analysis_results
    
    def _analyze_participation(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze participation patterns in the meeting."""
        
        speakers = {}
        total_duration = 0
        
        for segment in segments:
            speaker = segment.get("speaker_name", "Unknown")
            duration = segment.get("end_time", 0) - segment.get("start_time", 0)
            word_count = len(segment.get("text", "").split())
            
            if speaker not in speakers:
                speakers[speaker] = {
                    "speaking_time": 0,
                    "word_count": 0,
                    "segment_count": 0,
                    "interruptions": 0,
                    "average_segment_length": 0
                }
            
            speakers[speaker]["speaking_time"] += duration
            speakers[speaker]["word_count"] += word_count
            speakers[speaker]["segment_count"] += 1
            total_duration += duration
        
        # Calculate participation metrics
        participation_metrics = {
            "total_speakers": len(speakers),
            "total_duration": total_duration,
            "speaker_stats": {},
            "dominance_index": 0,
            "participation_balance": 0
        }
        
        # Calculate per-speaker metrics
        speaking_times = []
        for speaker, stats in speakers.items():
            participation_metrics["speaker_stats"][speaker] = {
                "speaking_time_percent": (stats["speaking_time"] / total_duration * 100) if total_duration > 0 else 0,
                "word_count": stats["word_count"],
                "words_per_minute": (stats["word_count"] / (stats["speaking_time"] / 60)) if stats["speaking_time"] > 0 else 0,
                "average_segment_length": stats["speaking_time"] / stats["segment_count"] if stats["segment_count"] > 0 else 0,
                "segment_count": stats["segment_count"]
            }
            speaking_times.append(stats["speaking_time"])
        
        # Calculate dominance index (Gini coefficient for speaking time)
        if speaking_times:
            speaking_times_sorted = sorted(speaking_times)
            n = len(speaking_times_sorted)
            cumsum = np.cumsum(speaking_times_sorted)
            participation_metrics["dominance_index"] = (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        # Calculate participation balance
        if len(speakers) > 1:
            ideal_time_per_speaker = total_duration / len(speakers)
            variance = np.var([stats["speaking_time"] for stats in speakers.values()])
            participation_metrics["participation_balance"] = 1 - (variance / (ideal_time_per_speaker ** 2)) if ideal_time_per_speaker > 0 else 0
        
        return participation_metrics
    
    def _analyze_communication_patterns(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze communication patterns and dynamics."""
        
        patterns = {
            "turn_taking": {},
            "interruption_rate": 0,
            "response_patterns": {},
            "communication_flow": []
        }
        
        # Analyze turn-taking patterns
        speaker_transitions = []
        previous_speaker = None
        
        for segment in segments:
            current_speaker = segment.get("speaker_name", "Unknown")
            
            if previous_speaker and previous_speaker != current_speaker:
                speaker_transitions.append((previous_speaker, current_speaker))
            
            previous_speaker = current_speaker
        
        # Build transition matrix
        speakers = list(set([seg.get("speaker_name", "Unknown") for seg in segments]))
        transition_matrix = {}
        
        for speaker in speakers:
            transition_matrix[speaker] = {s: 0 for s in speakers}
        
        for from_speaker, to_speaker in speaker_transitions:
            transition_matrix[from_speaker][to_speaker] += 1
        
        patterns["turn_taking"] = transition_matrix
        
        # Calculate interruption rate (simplified)
        short_segments = sum(1 for seg in segments if (seg.get("end_time", 0) - seg.get("start_time", 0)) < 3)
        patterns["interruption_rate"] = short_segments / len(segments) if segments else 0
        
        # Communication flow timeline
        for segment in segments:
            patterns["communication_flow"].append({
                "timestamp": segment.get("start_time", 0),
                "speaker": segment.get("speaker_name", "Unknown"),
                "duration": segment.get("end_time", 0) - segment.get("start_time", 0),
                "word_count": len(segment.get("text", "").split())
            })
        
        return patterns
    
    def _analyze_topics(self, transcript: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze topics and themes in the meeting."""
        
        if not transcript.strip():
            return {"topics": [], "topic_evolution": [], "key_phrases": []}
        
        # Extract key phrases using TF-IDF
        try:
            tfidf_matrix = self.vectorizer.fit_transform([transcript])
            feature_names = self.vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top phrases
            top_indices = scores.argsort()[-20:][::-1]
            key_phrases = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
        except Exception as e:
            logger.warning(f"TF-IDF analysis failed: {e}")
            key_phrases = []
        
        # Topic clustering
        segment_texts = [seg.get("text", "") for seg in segments if seg.get("text", "").strip()]
        topics = []
        
        if len(segment_texts) >= 3:
            try:
                # Vectorize segments
                segment_vectors = self.vectorizer.fit_transform(segment_texts)
                
                # Cluster into topics
                n_clusters = min(5, len(segment_texts) // 2)
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(segment_vectors)
                    
                    # Extract topic keywords for each cluster
                    for cluster_id in range(n_clusters):
                        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                        cluster_texts = [segment_texts[i] for i in cluster_indices]
                        
                        if cluster_texts:
                            combined_text = " ".join(cluster_texts)
                            cluster_vector = self.vectorizer.transform([combined_text])
                            scores = cluster_vector.toarray()[0]
                            
                            top_indices = scores.argsort()[-5:][::-1]
                            topic_keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
                            
                            topics.append({
                                "cluster_id": cluster_id,
                                "keywords": topic_keywords,
                                "segment_count": len(cluster_indices),
                                "segments": cluster_indices
                            })
            except Exception as e:
                logger.warning(f"Topic clustering failed: {e}")
        
        # Topic evolution over time
        topic_evolution = self._analyze_topic_evolution(segments, key_phrases)
        
        return {
            "topics": topics,
            "topic_evolution": topic_evolution,
            "key_phrases": key_phrases[:10]  # Top 10 phrases
        }
    
    def _analyze_topic_evolution(self, segments: List[Dict[str, Any]], key_phrases: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """Analyze how topics evolve throughout the meeting."""
        
        evolution = []
        window_size = 5  # Analyze in windows of 5 segments
        key_words = [phrase for phrase, score in key_phrases[:10]]
        
        for i in range(0, len(segments), window_size):
            window_segments = segments[i:i + window_size]
            window_text = " ".join([seg.get("text", "") for seg in window_segments])
            
            # Count key phrase occurrences in this window
            phrase_counts = {}
            for phrase in key_words:
                phrase_counts[phrase] = window_text.lower().count(phrase.lower())
            
            # Calculate window timestamp
            if window_segments:
                window_start = window_segments[0].get("start_time", 0)
                window_end = window_segments[-1].get("end_time", 0)
                
                evolution.append({
                    "window_start": window_start,
                    "window_end": window_end,
                    "phrase_counts": phrase_counts,
                    "dominant_phrases": sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                })
        
        return evolution
    
    def _analyze_sentiment_flow(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment flow throughout the meeting."""
        
        sentiment_timeline = []
        overall_sentiments = []
        
        for segment in segments:
            text = segment.get("text", "")
            if not text.strip():
                continue
            
            # Get sentiment from segment or calculate
            sentiment_score = segment.get("sentiment_score")
            if sentiment_score is None:
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
            
            sentiment_timeline.append({
                "timestamp": segment.get("start_time", 0),
                "sentiment_score": sentiment_score,
                "speaker": segment.get("speaker_name", "Unknown"),
                "text_sample": text[:100]  # First 100 characters
            })
            
            overall_sentiments.append(sentiment_score)
        
        # Calculate summary statistics
        if overall_sentiments:
            sentiment_stats = {
                "average_sentiment": np.mean(overall_sentiments),
                "sentiment_variance": np.var(overall_sentiments),
                "positive_moments": sum(1 for s in overall_sentiments if s > 0.1),
                "negative_moments": sum(1 for s in overall_sentiments if s < -0.1),
                "neutral_moments": sum(1 for s in overall_sentiments if -0.1 <= s <= 0.1)
            }
        else:
            sentiment_stats = {
                "average_sentiment": 0,
                "sentiment_variance": 0,
                "positive_moments": 0,
                "negative_moments": 0,
                "neutral_moments": 0
            }
        
        # Detect sentiment shifts
        sentiment_shifts = []
        for i in range(1, len(sentiment_timeline)):
            prev_sentiment = sentiment_timeline[i-1]["sentiment_score"]
            curr_sentiment = sentiment_timeline[i]["sentiment_score"]
            
            if abs(curr_sentiment - prev_sentiment) > 0.5:  # Significant shift
                sentiment_shifts.append({
                    "timestamp": sentiment_timeline[i]["timestamp"],
                    "from_sentiment": prev_sentiment,
                    "to_sentiment": curr_sentiment,
                    "magnitude": abs(curr_sentiment - prev_sentiment)
                })
        
        return {
            "timeline": sentiment_timeline,
            "statistics": sentiment_stats,
            "sentiment_shifts": sentiment_shifts
        }
    
    def _extract_decisions(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract decisions made during the meeting."""
        
        decision_patterns = [
            r"(?:we (?:decided|agreed|concluded|determined)|decision (?:is|was)|it was decided)\s+(?:that\s+)?(.+?)(?:\.|$|,)",
            r"(?:final decision|conclusion|resolution):\s*(.+?)(?:\.|$)",
            r"(?:so we will|we're going to|we'll)\s+(.+?)(?:\.|$)"
        ]
        
        decisions = []
        import re
        
        for segment in segments:
            text = segment.get("text", "")
            
            for pattern in decision_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    decision_text = match.strip()
                    if len(decision_text) > 10:  # Filter out very short matches
                        decisions.append({
                            "decision": decision_text,
                            "timestamp": segment.get("start_time", 0),
                            "speaker": segment.get("speaker_name", "Unknown"),
                            "context": text,
                            "confidence": 0.8  # Rule-based confidence
                        })
        
        return decisions
    
    def _extract_action_items(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract action items from the meeting."""
        
        action_patterns = [
            r"(?:action item|todo|task|assignment):\s*(.+?)(?:\.|$)",
            r"(?:assigned to|responsible for|owner is)\s+(\w+).*?(?:to\s+)?(.+?)(?:\.|$)",
            r"(?:we need to|should|must|will|going to)\s+(.+?)(?:\.|$)",
            r"(\w+)\s+(?:will|should|needs to|has to)\s+(.+?)(?:\.|$)"
        ]
        
        action_items = []
        import re
        
        for segment in segments:
            text = segment.get("text", "")
            
            for i, pattern in enumerate(action_patterns):
                matches = re.findall(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    if i == 1 or i == 3:  # Patterns with assignee
                        assignee, action = match
                        action_text = action.strip()
                    else:
                        action_text = match.strip() if isinstance(match, str) else match[0].strip()
                        assignee = None
                    
                    if len(action_text) > 10:
                        action_items.append({
                            "action": action_text,
                            "assignee": assignee,
                            "timestamp": segment.get("start_time", 0),
                            "speaker": segment.get("speaker_name", "Unknown"),
                            "context": text,
                            "priority": self._estimate_action_priority(action_text),
                            "confidence": 0.7
                        })
        
        return action_items
    
    def _estimate_action_priority(self, action_text: str) -> str:
        """Estimate action item priority based on text content."""
        
        high_priority_keywords = ["urgent", "asap", "immediately", "critical", "deadline", "priority"]
        medium_priority_keywords = ["soon", "next week", "important", "should"]
        
        text_lower = action_text.lower()
        
        if any(keyword in text_lower for keyword in high_priority_keywords):
            return "high"
        elif any(keyword in text_lower for keyword in medium_priority_keywords):
            return "medium"
        else:
            return "low"
    
    def _calculate_meeting_effectiveness(self, segments: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall meeting effectiveness score."""
        
        effectiveness = {
            "overall_score": 0,
            "participation_score": 0,
            "outcome_score": 0,
            "engagement_score": 0,
            "efficiency_score": 0
        }
        
        # Participation score (0-100)
        participation_metrics = metrics.get("participation", {})
        dominance_index = participation_metrics.get("dominance_index", 1)
        participation_balance = participation_metrics.get("participation_balance", 0)
        
        effectiveness["participation_score"] = max(0, min(100, (1 - dominance_index) * 50 + participation_balance * 50))
        
        # Outcome score based on decisions and action items
        decisions = metrics.get("insights", {}).get("decisions", [])
        action_items = metrics.get("insights", {}).get("action_items", [])
        
        outcome_score = min(100, len(decisions) * 20 + len(action_items) * 10)
        effectiveness["outcome_score"] = outcome_score
        
        # Engagement score based on sentiment
        sentiment_metrics = metrics.get("sentiment_flow", {}).get("statistics", {})
        avg_sentiment = sentiment_metrics.get("average_sentiment", 0)
        positive_moments = sentiment_metrics.get("positive_moments", 0)
        total_moments = sum([
            sentiment_metrics.get("positive_moments", 0),
            sentiment_metrics.get("negative_moments", 0),
            sentiment_metrics.get("neutral_moments", 0)
        ])
        
        if total_moments > 0:
            positivity_ratio = positive_moments / total_moments
            engagement_score = max(0, min(100, (avg_sentiment + 1) * 25 + positivity_ratio * 50))
        else:
            engagement_score = 50
        
        effectiveness["engagement_score"] = engagement_score
        
        # Efficiency score (inverse of interruption rate and balanced speaking)
        communication_metrics = metrics.get("communication", {})
        interruption_rate = communication_metrics.get("interruption_rate", 0)
        efficiency_score = max(0, min(100, (1 - interruption_rate) * 100))
        effectiveness["efficiency_score"] = efficiency_score
        
        # Overall score (weighted average)
        effectiveness["overall_score"] = (
            effectiveness["participation_score"] * 0.25 +
            effectiveness["outcome_score"] * 0.30 +
            effectiveness["engagement_score"] * 0.25 +
            effectiveness["efficiency_score"] * 0.20
        )
        
        return effectiveness
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis."""
        
        recommendations = []
        metrics = analysis_results.get("metrics", {})
        
        # Participation recommendations
        participation = metrics.get("participation", {})
        dominance_index = participation.get("dominance_index", 0)
        
        if dominance_index > 0.6:
            recommendations.append({
                "type": "participation",
                "priority": "medium",
                "title": "Improve Participation Balance",
                "description": "The meeting shows uneven participation. Consider using techniques like round-robin discussions or time-boxing to ensure all voices are heard.",
                "action": "Implement structured discussion formats in future meetings."
            })
        
        # Sentiment recommendations
        sentiment_stats = metrics.get("sentiment_flow", {}).get("statistics", {})
        avg_sentiment = sentiment_stats.get("average_sentiment", 0)
        
        if avg_sentiment < -0.2:
            recommendations.append({
                "type": "sentiment",
                "priority": "high",
                "title": "Address Negative Sentiment",
                "description": "The meeting had overall negative sentiment. Consider addressing concerns raised and following up with participants.",
                "action": "Schedule follow-up discussions to address concerns and improve team morale."
            })
        
        # Outcome recommendations
        decisions = analysis_results.get("insights", {}).get("decisions", [])
        action_items = analysis_results.get("insights", {}).get("action_items", [])
        
        if len(decisions) == 0 and len(action_items) == 0:
            recommendations.append({
                "type": "outcomes",
                "priority": "high",
                "title": "Clarify Meeting Outcomes",
                "description": "No clear decisions or action items were identified. Consider ending meetings with explicit decision summaries and action item assignments.",
                "action": "Implement structured meeting closure with clear outcomes documentation."
            })
        
        # Efficiency recommendations
        communication = metrics.get("communication", {})
        interruption_rate = communication.get("interruption_rate", 0)
        
        if interruption_rate > 0.3:
            recommendations.append({
                "type": "efficiency",
                "priority": "medium",
                "title": "Reduce Interruptions",
                "description": "High interruption rate detected. Consider establishing ground rules for discussion flow.",
                "action": "Implement meeting etiquette guidelines and consider using a speaking order system."
            })
        
        return recommendations

class NetworkAnalyzer:
    """Analyzes communication networks and relationships."""
    
    def __init__(self):
        self.graph = None
    
    def build_communication_network(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build communication network from meeting segments."""
        
        G = nx.DiGraph()
        speakers = set()
        
        # Add nodes (speakers)
        for segment in segments:
            speaker = segment.get("speaker_name", "Unknown")
            speakers.add(speaker)
            G.add_node(speaker)
        
        # Add edges (communication flow)
        previous_speaker = None
        for segment in segments:
            current_speaker = segment.get("speaker_name", "Unknown")
            
            if previous_speaker and previous_speaker != current_speaker:
                if G.has_edge(previous_speaker, current_speaker):
                    G[previous_speaker][current_speaker]["weight"] += 1
                else:
                    G.add_edge(previous_speaker, current_speaker, weight=1)
            
            previous_speaker = current_speaker
        
        self.graph = G
        
        # Calculate network metrics
        network_metrics = {
            "node_count": G.number_of_nodes(),
            "edge_count": G.number_of_edges(),
            "density": nx.density(G),
            "centrality": {},
            "clusters": []
        }
        
        if G.number_of_nodes() > 1:
            # Calculate centrality measures
            try:
                network_metrics["centrality"]["betweenness"] = nx.betweenness_centrality(G)
                network_metrics["centrality"]["closeness"] = nx.closeness_centrality(G)
                network_metrics["centrality"]["degree"] = nx.degree_centrality(G)
                
                # Find communities/clusters
                if G.number_of_edges() > 0:
                    undirected_G = G.to_undirected()
                    communities = nx.community.greedy_modularity_communities(undirected_G)
                    network_metrics["clusters"] = [list(community) for community in communities]
            
            except Exception as e:
                logger.warning(f"Network analysis failed: {e}")
        
        return network_metrics

class PatternDetector:
    """Detects patterns and anomalies in meeting data."""
    
    def __init__(self):
        self.meeting_patterns = []
    
    def detect_meeting_patterns(self, meetings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns across multiple meetings."""
        
        patterns = {
            "duration_patterns": self._analyze_duration_patterns(meetings),
            "participation_patterns": self._analyze_participation_patterns(meetings),
            "topic_patterns": self._analyze_topic_patterns(meetings),
            "outcome_patterns": self._analyze_outcome_patterns(meetings)
        }
        
        return patterns
    
    def _analyze_duration_patterns(self, meetings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze meeting duration patterns."""
        
        durations = [meeting.get("duration", 0) for meeting in meetings if meeting.get("duration")]
        
        if not durations:
            return {"average_duration": 0, "duration_trend": "no_data"}
        
        average_duration = np.mean(durations)
        duration_std = np.std(durations)
        
        # Detect trend
        if len(durations) >= 3:
            trend_correlation = np.corrcoef(range(len(durations)), durations)[0, 1]
            if trend_correlation > 0.3:
                trend = "increasing"
            elif trend_correlation < -0.3:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "average_duration": average_duration,
            "duration_variance": duration_std ** 2,
            "duration_trend": trend,
            "outliers": [d for d in durations if abs(d - average_duration) > 2 * duration_std]
        }
    
    def _analyze_participation_patterns(self, meetings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze participation patterns across meetings."""
        
        all_speakers = set()
        speaker_frequency = {}
        
        for meeting in meetings:
            segments = meeting.get("segments", [])
            meeting_speakers = set(seg.get("speaker_name", "Unknown") for seg in segments)
            
            all_speakers.update(meeting_speakers)
            
            for speaker in meeting_speakers:
                speaker_frequency[speaker] = speaker_frequency.get(speaker, 0) + 1
        
        # Calculate participation consistency
        total_meetings = len(meetings)
        regular_participants = [
            speaker for speaker, freq in speaker_frequency.items() 
            if freq >= total_meetings * 0.7  # Appears in 70%+ of meetings
        ]
        
        return {
            "total_unique_speakers": len(all_speakers),
            "regular_participants": regular_participants,
            "average_speakers_per_meeting": len(all_speakers) / max(total_meetings, 1),
            "participation_consistency": len(regular_participants) / max(len(all_speakers), 1)
        }
    
    def _analyze_topic_patterns(self, meetings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze topic patterns across meetings."""
        
        all_topics = []
        
        for meeting in meetings:
            topics = meeting.get("analysis", {}).get("metrics", {}).get("topics", {}).get("key_phrases", [])
            meeting_topics = [phrase for phrase, score in topics] if topics else []
            all_topics.extend(meeting_topics)
        
        if not all_topics:
            return {"recurring_topics": [], "topic_diversity": 0}
        
        # Find recurring topics
        from collections import Counter
        topic_counts = Counter(all_topics)
        recurring_topics = [topic for topic, count in topic_counts.most_common(10) if count > 1]
        
        # Calculate topic diversity
        unique_topics = len(set(all_topics))
        total_topics = len(all_topics)
        topic_diversity = unique_topics / max(total_topics, 1)
        
        return {
            "recurring_topics": recurring_topics,
            "topic_diversity": topic_diversity,
            "total_unique_topics": unique_topics
        }
    
    def _analyze_outcome_patterns(self, meetings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze outcome patterns (decisions, action items)."""
        
        total_decisions = 0
        total_action_items = 0
        
        for meeting in meetings:
            insights = meeting.get("analysis", {}).get("insights", {})
            total_decisions += len(insights.get("decisions", []))
            total_action_items += len(insights.get("action_items", []))
        
        total_meetings = len(meetings)
        
        return {
            "average_decisions_per_meeting": total_decisions / max(total_meetings, 1),
            "average_action_items_per_meeting": total_action_items / max(total_meetings, 1),
            "outcome_productivity": (total_decisions + total_action_items) / max(total_meetings, 1)
        }

# Global analytics instances
intelligence_engine = MeetingIntelligenceEngine()
```

This completes the first major section of Part 9, covering real-time processing pipeline and advanced analytics engine. The implementation demonstrates:

1. **Event-driven architecture** for real-time meeting processing
2. **WebSocket-based real-time communication** for live updates
3. **Advanced analytics engine** with comprehensive meeting intelligence
4. **Pattern detection** across multiple meetings
5. **Network analysis** for communication patterns
6. **Real-time transcription** with live insights
7. **Sentiment flow analysis** and effectiveness scoring

Would you like me to continue with the remaining sections covering Third-party Integrations, Machine Learning Pipeline, Workflow Automation, API Extensions, Mobile Support, and Enterprise Features?