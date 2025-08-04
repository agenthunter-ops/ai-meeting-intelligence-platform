# AI Meeting Intelligence Platform - Part 7: Performance & Optimization

## Table of Contents
1. [Performance Architecture Overview](#performance-architecture-overview)
2. [Database Optimization](#database-optimization)
3. [API Performance Optimization](#api-performance-optimization)
4. [AI Services Performance](#ai-services-performance)
5. [Frontend Performance](#frontend-performance)
6. [Caching Strategies](#caching-strategies)
7. [Load Testing & Monitoring](#load-testing--monitoring)
8. [Scalability Planning](#scalability-planning)

---

## Performance Architecture Overview

### Performance Requirements & SLAs

Our performance architecture is designed to meet specific Service Level Agreements (SLAs) that ensure optimal user experience and system reliability:

```
Performance Targets:
┌─────────────────────────────────────────────────────────────┐
│ API Response Times (95th percentile):                      │
│ ├─ Health checks: < 50ms                                   │
│ ├─ File uploads: < 2s (excluding transfer time)            │
│ ├─ Search queries: < 200ms                                 │
│ ├─ Status updates: < 100ms                                 │
│ └─ Meeting retrieval: < 300ms                              │
│                                                             │
│ Processing Performance:                                     │
│ ├─ Audio transcription: < 0.3x real-time                   │
│ ├─ Sentiment analysis: < 30s per meeting                   │
│ ├─ Summary generation: < 60s per meeting                   │
│ └─ Vector embedding: < 120s per meeting                    │
│                                                             │
│ Scalability Targets:                                       │
│ ├─ Concurrent users: 100+ simultaneous                     │
│ ├─ File uploads: 20 concurrent                             │
│ ├─ Processing queue: 50 jobs                               │
│ └─ Storage capacity: 10TB+ meetings                        │
└─────────────────────────────────────────────────────────────┘
```

### Performance Monitoring Framework

```python
# backend/performance/monitor.py - Performance monitoring system

import time
import psutil
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging
import json
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags or {}
        }

class PerformanceCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.start_time = datetime.now()
        self._lock = threading.Lock()
        
        # System monitoring
        self.system_monitor_running = False
        self.system_monitor_thread = None
        
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        with self._lock:
            self.metrics[metric.name].append(metric)
            
        logger.debug(f"Recorded metric: {metric.name} = {metric.value} {metric.unit}")
    
    def record_timing(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timing metric."""
        metric = PerformanceMetric(
            name=name,
            value=duration,
            unit="seconds",
            timestamp=datetime.now(),
            tags=tags
        )
        self.record_metric(metric)
    
    def record_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Record a counter metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit="count",
            timestamp=datetime.now(),
            tags=tags
        )
        self.record_metric(metric)
    
    def record_gauge(self, name: str, value: float, unit: str, tags: Dict[str, str] = None):
        """Record a gauge metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags
        )
        self.record_metric(metric)
    
    def get_statistics(self, metric_name: str, window_minutes: int = 5) -> Dict[str, Any]:
        """Get statistics for a metric within a time window."""
        with self._lock:
            if metric_name not in self.metrics:
                return {"error": f"Metric {metric_name} not found"}
            
            # Filter metrics within time window
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_metrics = [
                m for m in self.metrics[metric_name] 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {"error": f"No recent data for {metric_name}"}
            
            values = [m.value for m in recent_metrics]
            
            return {
                "metric_name": metric_name,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "unit": recent_metrics[0].unit,
                "window_minutes": window_minutes,
                "latest_value": values[-1],
                "latest_timestamp": recent_metrics[-1].timestamp.isoformat()
            }
    
    def get_percentile(self, metric_name: str, percentile: float, window_minutes: int = 5) -> Optional[float]:
        """Get percentile value for a metric."""
        with self._lock:
            if metric_name not in self.metrics:
                return None
            
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_metrics = [
                m for m in self.metrics[metric_name] 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return None
            
            values = sorted([m.value for m in recent_metrics])
            index = int(len(values) * percentile / 100)
            
            return values[min(index, len(values) - 1)]
    
    def start_system_monitoring(self, interval: int = 10):
        """Start monitoring system resources."""
        if self.system_monitor_running:
            return
        
        self.system_monitor_running = True
        
        def monitor_system():
            while self.system_monitor_running:
                try:
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.record_gauge("system.cpu.usage", cpu_percent, "percent")
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.record_gauge("system.memory.usage", memory.percent, "percent")
                    self.record_gauge("system.memory.available", memory.available / 1024**3, "GB")
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    self.record_gauge("system.disk.usage", (disk.used / disk.total) * 100, "percent")
                    self.record_gauge("system.disk.free", disk.free / 1024**3, "GB")
                    
                    # Network metrics (if available)
                    try:
                        network = psutil.net_io_counters()
                        self.record_gauge("system.network.bytes_sent", network.bytes_sent, "bytes")
                        self.record_gauge("system.network.bytes_recv", network.bytes_recv, "bytes")
                    except:
                        pass
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                    time.sleep(interval)
        
        self.system_monitor_thread = threading.Thread(target=monitor_system)
        self.system_monitor_thread.daemon = True
        self.system_monitor_thread.start()
        
        logger.info("Started system monitoring")
    
    def stop_system_monitoring(self):
        """Stop system resource monitoring."""
        self.system_monitor_running = False
        if self.system_monitor_thread:
            self.system_monitor_thread.join(timeout=5)
        logger.info("Stopped system monitoring")
    
    def export_metrics(self, format: str = "json") -> str:
        """Export all metrics in specified format."""
        with self._lock:
            all_metrics = []
            for metric_name, metric_list in self.metrics.items():
                for metric in metric_list:
                    all_metrics.append(metric.to_dict())
            
            if format == "json":
                return json.dumps(all_metrics, indent=2)
            elif format == "prometheus":
                return self._export_prometheus_format(all_metrics)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def _export_prometheus_format(self, metrics: List[Dict]) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        for metric in metrics:
            tags_str = ""
            if metric["tags"]:
                tag_pairs = [f'{k}="{v}"' for k, v in metric["tags"].items()]
                tags_str = "{" + ",".join(tag_pairs) + "}"
            
            line = f'{metric["name"]}{tags_str} {metric["value"]}'
            lines.append(line)
        
        return "\n".join(lines)

# Global performance collector instance
performance_collector = PerformanceCollector()

@contextmanager
def measure_time(metric_name: str, tags: Dict[str, str] = None):
    """Context manager for measuring execution time."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        performance_collector.record_timing(metric_name, duration, tags)

def performance_monitor(metric_name: str = None, tags: Dict[str, str] = None):
    """Decorator for monitoring function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = metric_name or f"function.{func.__name__}.duration"
            with measure_time(name, tags):
                result = func(*args, **kwargs)
                performance_collector.record_counter(f"function.{func.__name__}.calls", tags=tags)
                return result
        return wrapper
    return decorator

# FastAPI middleware for request monitoring
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor API request performance."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Record request start
        performance_collector.record_counter("api.requests.total", 
                                            tags={"method": request.method, "endpoint": str(request.url.path)})
        
        try:
            response = await call_next(request)
            
            # Record successful request
            duration = time.time() - start_time
            performance_collector.record_timing("api.request.duration", duration,
                                               tags={"method": request.method, 
                                                    "endpoint": str(request.url.path),
                                                    "status_code": str(response.status_code)})
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            # Record failed request
            duration = time.time() - start_time
            performance_collector.record_timing("api.request.duration", duration,
                                               tags={"method": request.method, 
                                                    "endpoint": str(request.url.path),
                                                    "status_code": "500"})
            
            performance_collector.record_counter("api.requests.errors",
                                                tags={"method": request.method,
                                                     "endpoint": str(request.url.path),
                                                     "error_type": type(e).__name__})
            raise
```

---

## Database Optimization

### Query Optimization Strategies

```python
# backend/db_optimization.py - Database optimization techniques

from sqlalchemy import Index, text, func, and_, or_
from sqlalchemy.orm import sessionmaker, joinedload, selectinload
from sqlalchemy.engine import Engine
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseOptimizer:
    """Database optimization and query performance manager."""
    
    def __init__(self, engine: Engine):
        self.engine = engine
        self.query_cache = {}
        
    def create_performance_indexes(self):
        """Create indexes for optimal query performance."""
        
        with self.engine.connect() as conn:
            # Meeting indexes
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_meetings_status_upload_time 
                    ON meetings(status, upload_time DESC)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_meetings_meeting_type 
                    ON meetings(meeting_type)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_meetings_title_search 
                    ON meetings(title)
                """))
                
                # Task indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_tasks_meeting_id_status 
                    ON tasks(meeting_id, status)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_tasks_status_created_at 
                    ON tasks(status, created_at DESC)
                """))
                
                # Segment indexes
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_segments_meeting_id_start_time 
                    ON segments(meeting_id, start_time)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_segments_speaker_name 
                    ON segments(speaker_name)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_segments_text_search 
                    ON segments USING gin(to_tsvector('english', text))
                """))
                
                conn.commit()
                logger.info("Performance indexes created successfully")
                
            except Exception as e:
                logger.error(f"Failed to create indexes: {e}")
                conn.rollback()
    
    def optimize_meeting_queries(self, session):
        """Optimized queries for meeting operations."""
        
        class OptimizedMeetingQueries:
            
            @staticmethod
            def get_recent_meetings(limit: int = 10, status: str = None):
                """Get recent meetings with optimal query."""
                query = session.query(Meeting).options(
                    # Eagerly load related data to avoid N+1 queries
                    selectinload(Meeting.tasks),
                    selectinload(Meeting.segments).selectinload(Segment.speaker_name)
                )
                
                if status:
                    query = query.filter(Meeting.status == status)
                
                return query.order_by(Meeting.upload_time.desc()).limit(limit).all()
            
            @staticmethod
            def get_meeting_with_segments(meeting_id: str):
                """Get meeting with all segments efficiently."""
                return session.query(Meeting).options(
                    joinedload(Meeting.segments).joinedload(Segment.speaker_name)
                ).filter(Meeting.id == meeting_id).first()
            
            @staticmethod
            def search_meetings_optimized(query_text: str, filters: Dict = None, limit: int = 20):
                """Optimized meeting search with full-text search."""
                
                # Base query with necessary joins
                base_query = session.query(Meeting).distinct()
                
                # Text search optimization
                if query_text:
                    # Search in meeting titles
                    title_matches = base_query.filter(
                        Meeting.title.ilike(f"%{query_text}%")
                    )
                    
                    # Search in segment text using full-text search
                    segment_matches = base_query.join(Segment).filter(
                        text("to_tsvector('english', segments.text) @@ plainto_tsquery('english', :query)")
                    ).params(query=query_text)
                    
                    # Union the results
                    query = title_matches.union(segment_matches)
                else:
                    query = base_query
                
                # Apply filters
                if filters:
                    if filters.get("meeting_type"):
                        query = query.filter(Meeting.meeting_type == filters["meeting_type"])
                    
                    if filters.get("status"):
                        query = query.filter(Meeting.status == filters["status"])
                    
                    if filters.get("date_from"):
                        query = query.filter(Meeting.upload_time >= filters["date_from"])
                    
                    if filters.get("date_to"):
                        query = query.filter(Meeting.upload_time <= filters["date_to"])
                
                return query.order_by(Meeting.upload_time.desc()).limit(limit).all()
            
            @staticmethod
            def get_meeting_statistics():
                """Get aggregated meeting statistics efficiently."""
                return session.query(
                    func.count(Meeting.id).label("total_meetings"),
                    func.sum(Meeting.duration).label("total_duration"),
                    func.avg(Meeting.duration).label("avg_duration"),
                    func.count(case([(Meeting.status == "completed", 1)])).label("completed_meetings"),
                    func.count(case([(Meeting.status == "processing", 1)])).label("processing_meetings")
                ).first()
        
        return OptimizedMeetingQueries()
    
    def analyze_query_performance(self, session):
        """Analyze and log query performance."""
        
        # Enable query logging
        import sqlalchemy
        
        # Sample queries to analyze
        test_queries = [
            lambda: session.query(Meeting).filter(Meeting.status == "completed").count(),
            lambda: session.query(Segment).join(Meeting).filter(Meeting.id.like("test%")).count(),
            lambda: session.query(func.count(Segment.id)).group_by(Segment.meeting_id).all()
        ]
        
        for i, query_func in enumerate(test_queries):
            start_time = time.time()
            
            try:
                result = query_func()
                duration = time.time() - start_time
                
                performance_collector.record_timing(f"db.query.test_{i}", duration)
                logger.info(f"Query {i} executed in {duration:.3f}s")
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Query {i} failed after {duration:.3f}s: {e}")
    
    def setup_connection_pooling(self, engine):
        """Configure optimal connection pooling."""
        
        # Connection pool settings are typically set during engine creation
        # This method provides guidance and monitoring
        
        pool = engine.pool
        
        def log_pool_status():
            logger.info(f"Connection pool status:")
            logger.info(f"  Pool size: {pool.size()}")
            logger.info(f"  Checked in: {pool.checkedin()}")
            logger.info(f"  Checked out: {pool.checkedout()}")
            logger.info(f"  Overflow: {pool.overflow()}")
        
        # Monitor pool usage
        performance_collector.record_gauge("db.pool.size", pool.size(), "connections")
        performance_collector.record_gauge("db.pool.checked_in", pool.checkedin(), "connections")
        performance_collector.record_gauge("db.pool.checked_out", pool.checkedout(), "connections")
        
        return log_pool_status

# Database query optimization decorators
def cached_query(cache_key: str, ttl: int = 300):
    """Decorator to cache query results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            import json
            
            # Generate cache key
            key = f"{cache_key}:{hash(json.dumps(kwargs, sort_keys=True))}"
            
            # Check cache
            if hasattr(wrapper, '_cache') and key in wrapper._cache:
                cached_result, timestamp = wrapper._cache[key]
                if time.time() - timestamp < ttl:
                    performance_collector.record_counter("db.cache.hits")
                    return cached_result
            
            # Execute query
            result = func(*args, **kwargs)
            
            # Cache result
            if not hasattr(wrapper, '_cache'):
                wrapper._cache = {}
            wrapper._cache[key] = (result, time.time())
            
            performance_collector.record_counter("db.cache.misses")
            return result
        
        return wrapper
    return decorator

def optimized_transaction(func):
    """Decorator for optimized database transactions."""
    def wrapper(*args, **kwargs):
        session = kwargs.get('session') or args[0] if args else None
        
        if not session:
            raise ValueError("Session required for optimized transaction")
        
        start_time = time.time()
        
        try:
            # Set transaction isolation level for better performance
            session.execute(text("SET TRANSACTION ISOLATION LEVEL READ COMMITTED"))
            
            result = func(*args, **kwargs)
            session.commit()
            
            duration = time.time() - start_time
            performance_collector.record_timing("db.transaction.duration", duration)
            
            return result
            
        except Exception as e:
            session.rollback()
            duration = time.time() - start_time
            performance_collector.record_timing("db.transaction.failed_duration", duration)
            raise
    
    return wrapper
```

### Database Schema Optimization

```sql
-- database/optimizations.sql - Database schema optimizations

-- Performance optimizations for PostgreSQL

-- Enable parallel query execution
SET max_parallel_workers_per_gather = 4;
SET parallel_setup_cost = 1000;
SET parallel_tuple_cost = 0.1;

-- Optimize memory settings
SET shared_buffers = '256MB';
SET effective_cache_size = '1GB';
SET work_mem = '64MB';
SET maintenance_work_mem = '128MB';

-- Optimize checkpoints
SET checkpoint_completion_target = 0.9;
SET wal_buffers = '16MB';

-- Partitioning strategy for large tables
CREATE TABLE meetings_partitioned (
    id VARCHAR PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    upload_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'uploaded',
    meeting_type VARCHAR(100),
    -- other columns...
) PARTITION BY RANGE (upload_time);

-- Create monthly partitions
CREATE TABLE meetings_2024_01 PARTITION OF meetings_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE meetings_2024_02 PARTITION OF meetings_partitioned
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Indexes for partitioned tables
CREATE INDEX meetings_2024_01_status_idx ON meetings_2024_01(status);
CREATE INDEX meetings_2024_02_status_idx ON meetings_2024_02(status);

-- Full-text search optimization
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- GIN index for full-text search on segments
CREATE INDEX segments_text_gin_idx ON segments 
    USING gin(to_tsvector('english', text));

-- Trigram index for fuzzy text matching
CREATE INDEX segments_text_trgm_idx ON segments 
    USING gin(text gin_trgm_ops);

-- Composite indexes for common query patterns
CREATE INDEX meetings_status_type_time_idx ON meetings(status, meeting_type, upload_time DESC);
CREATE INDEX segments_meeting_speaker_time_idx ON segments(meeting_id, speaker_name, start_time);

-- Materialized view for dashboard statistics
CREATE MATERIALIZED VIEW meeting_statistics AS
SELECT 
    DATE_TRUNC('day', upload_time) as date,
    COUNT(*) as total_meetings,
    SUM(duration) as total_duration,
    AVG(duration) as avg_duration,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_count,
    COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing_count,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_count
FROM meetings 
GROUP BY DATE_TRUNC('day', upload_time)
ORDER BY date DESC;

-- Index on materialized view
CREATE UNIQUE INDEX meeting_statistics_date_idx ON meeting_statistics(date);

-- Function to refresh statistics efficiently
CREATE OR REPLACE FUNCTION refresh_meeting_statistics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY meeting_statistics;
END;
$$ LANGUAGE plpgsql;

-- Automated statistics refresh (using pg_cron extension)
-- SELECT cron.schedule('refresh-stats', '0 */6 * * *', 'SELECT refresh_meeting_statistics();');

-- VACUUM and ANALYZE automation
CREATE OR REPLACE FUNCTION auto_vacuum_analyze()
RETURNS void AS $$
BEGIN
    -- Analyze tables for query planner
    ANALYZE meetings;
    ANALYZE segments;
    ANALYZE tasks;
    
    -- Log statistics
    RAISE NOTICE 'Auto VACUUM ANALYZE completed at %', NOW();
END;
$$ LANGUAGE plpgsql;

-- Archive old data procedure
CREATE OR REPLACE FUNCTION archive_old_meetings(days_old INTEGER DEFAULT 365)
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
BEGIN
    -- Move old meetings to archive table
    WITH archived AS (
        DELETE FROM meetings 
        WHERE upload_time < NOW() - INTERVAL '%s days' % days_old
        AND status IN ('completed', 'failed')
        RETURNING *
    )
    INSERT INTO meetings_archive SELECT * FROM archived;
    
    GET DIAGNOSTICS archived_count = ROW_COUNT;
    
    RAISE NOTICE 'Archived % meetings older than % days', archived_count, days_old;
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;
```

---

## API Performance Optimization

### Response Optimization

```python
# backend/api_optimization.py - API performance optimization

from fastapi import FastAPI, Request, Response, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.cors import CORSMiddleware
import gzip
import json
import asyncio
from typing import Any, Dict, List
import time
import logging

logger = logging.getLogger(__name__)

class APIOptimizer:
    """API performance optimization utilities."""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.setup_middleware()
        self.setup_response_optimization()
    
    def setup_middleware(self):
        """Setup performance-oriented middleware."""
        
        # GZip compression for responses
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # CORS with optimized settings
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            # Preflight cache to reduce OPTIONS requests
            max_age=3600
        )
    
    def setup_response_optimization(self):
        """Setup response optimization techniques."""
        
        @self.app.middleware("http")
        async def response_optimization_middleware(request: Request, call_next):
            """Middleware for response optimization."""
            
            start_time = time.time()
            
            # Set cache headers for static content
            if request.url.path.startswith('/static/'):
                response = await call_next(request)
                response.headers["Cache-Control"] = "public, max-age=31536000"  # 1 year
                response.headers["ETag"] = f'"{hash(request.url.path)}"'
                return response
            
            # Process API requests
            response = await call_next(request)
            
            # Add performance headers
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            
            # Add appropriate cache headers for API responses
            if request.url.path.startswith('/api/'):
                if request.method == "GET":
                    # Cache GET requests for 5 minutes
                    response.headers["Cache-Control"] = "public, max-age=300"
                else:
                    # No cache for non-GET requests
                    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            
            return response

class OptimizedResponse:
    """Utilities for creating optimized API responses."""
    
    @staticmethod
    def paginated_response(
        data: List[Any], 
        page: int, 
        limit: int, 
        total: int, 
        serializer=None
    ) -> Dict[str, Any]:
        """Create optimized paginated response."""
        
        # Serialize data efficiently
        if serializer:
            serialized_data = [serializer(item) for item in data]
        else:
            serialized_data = data
        
        total_pages = (total + limit - 1) // limit
        
        return {
            "data": serialized_data,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            },
            "meta": {
                "count": len(serialized_data),
                "timestamp": time.time()
            }
        }
    
    @staticmethod
    def streaming_response(data_generator, media_type: str = "application/json"):
        """Create streaming response for large datasets."""
        
        async def generate():
            yield "["
            first = True
            async for item in data_generator:
                if not first:
                    yield ","
                yield json.dumps(item)
                first = False
            yield "]"
        
        return StreamingResponse(generate(), media_type=media_type)
    
    @staticmethod
    def compressed_json_response(data: Any, status_code: int = 200) -> Response:
        """Create compressed JSON response."""
        
        json_data = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        
        # Compress if data is large enough
        if len(json_data) > 1000:
            compressed_data = gzip.compress(json_data.encode('utf-8'))
            
            return Response(
                content=compressed_data,
                status_code=status_code,
                headers={
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                    "Content-Length": str(len(compressed_data))
                }
            )
        else:
            return JSONResponse(content=data, status_code=status_code)

# Optimized endpoint implementations
@performance_monitor("api.meetings.list")
async def optimized_list_meetings(
    page: int = 1,
    limit: int = 20,
    status: str = None,
    meeting_type: str = None,
    db: Session = Depends(get_db)
):
    """Optimized meetings listing endpoint."""
    
    # Validate pagination parameters
    limit = min(limit, 100)  # Cap at 100 items per page
    offset = (page - 1) * limit
    
    # Build optimized query
    query = db.query(Meeting)
    
    # Apply filters
    filters = []
    if status:
        filters.append(Meeting.status == status)
    if meeting_type:
        filters.append(Meeting.meeting_type == meeting_type)
    
    if filters:
        query = query.filter(and_(*filters))
    
    # Get total count efficiently
    total = query.count()
    
    # Get paginated data with optimized loading
    meetings = query.options(
        # Load related data efficiently
        selectinload(Meeting.tasks).load_only(Task.status, Task.progress),
        selectinload(Meeting.segments).load_only(Segment.id)
    ).order_by(Meeting.upload_time.desc()).offset(offset).limit(limit).all()
    
    # Serialize data efficiently
    def serialize_meeting(meeting):
        return {
            "id": meeting.id,
            "title": meeting.title,
            "status": meeting.status,
            "upload_time": meeting.upload_time.isoformat() if meeting.upload_time else None,
            "duration": meeting.duration,
            "meeting_type": meeting.meeting_type,
            "task_count": len(meeting.tasks),
            "segment_count": len(meeting.segments)
        }
    
    return OptimizedResponse.paginated_response(
        data=meetings,
        page=page,
        limit=limit,
        total=total,
        serializer=serialize_meeting
    )

@performance_monitor("api.search.meetings")
async def optimized_search_meetings(
    query: str,
    page: int = 1,
    limit: int = 20,
    filters: Dict[str, Any] = None,
    db: Session = Depends(get_db)
):
    """Optimized meeting search endpoint."""
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    # Use the optimized search function
    optimizer = DatabaseOptimizer(db.bind)
    optimized_queries = optimizer.optimize_meeting_queries(db)
    
    # Perform optimized search
    results = optimized_queries.search_meetings_optimized(
        query_text=query,
        filters=filters or {},
        limit=limit * 2  # Get more results for better pagination
    )
    
    # Apply pagination
    offset = (page - 1) * limit
    paginated_results = results[offset:offset + limit]
    
    def serialize_search_result(meeting):
        return {
            "id": meeting.id,
            "title": meeting.title,
            "status": meeting.status,
            "upload_time": meeting.upload_time.isoformat() if meeting.upload_time else None,
            "meeting_type": meeting.meeting_type,
            "relevance_score": 1.0,  # Would be calculated by search algorithm
            "snippet": meeting.title  # Would include matched text snippet
        }
    
    return OptimizedResponse.paginated_response(
        data=paginated_results,
        page=page,
        limit=limit,
        total=len(results),
        serializer=serialize_search_result
    )

# Batch operations for efficiency
@performance_monitor("api.meetings.batch_status")
async def batch_status_update(
    meeting_ids: List[str],
    status: str,
    db: Session = Depends(get_db)
):
    """Batch update meeting statuses efficiently."""
    
    if len(meeting_ids) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 meetings per batch")
    
    # Use bulk update for efficiency
    updated_count = db.query(Meeting).filter(
        Meeting.id.in_(meeting_ids)
    ).update(
        {"status": status},
        synchronize_session=False
    )
    
    db.commit()
    
    performance_collector.record_counter("api.batch.meetings_updated", updated_count)
    
    return {
        "updated_count": updated_count,
        "meeting_ids": meeting_ids,
        "new_status": status
    }

# Async processing for long-running operations
async def async_bulk_export(
    meeting_ids: List[str],
    format: str,
    db: Session
):
    """Asynchronously export meetings data."""
    
    async def export_generator():
        """Generator for streaming export data."""
        
        # Export metadata
        yield {
            "export_info": {
                "format": format,
                "meeting_count": len(meeting_ids),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Export meetings one by one to avoid memory issues
        for meeting_id in meeting_ids:
            meeting = db.query(Meeting).options(
                joinedload(Meeting.segments),
                joinedload(Meeting.tasks)
            ).filter(Meeting.id == meeting_id).first()
            
            if meeting:
                yield {
                    "meeting": {
                        "id": meeting.id,
                        "title": meeting.title,
                        "segments": [
                            {
                                "start_time": seg.start_time,
                                "end_time": seg.end_time,
                                "text": seg.text,
                                "speaker": seg.speaker_name
                            }
                            for seg in meeting.segments
                        ]
                    }
                }
                
                # Allow other coroutines to run
                await asyncio.sleep(0)
    
    return OptimizedResponse.streaming_response(
        export_generator(),
        media_type="application/json"
    )
```

---

## AI Services Performance

### Whisper Service Optimization

```python
# whisper_service/performance_optimization.py - Whisper service optimization

import torch
import whisper
import numpy as np
from typing import Dict, List, Any, Tuple
import threading
import queue
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class OptimizedWhisperProcessor:
    """Performance-optimized Whisper processor."""
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        self.device = self._select_optimal_device(device)
        self.model_size = model_size
        
        # Load model with optimizations
        self.model = self._load_optimized_model()
        
        # Processing pools
        self.cpu_pool = ThreadPoolExecutor(max_workers=2)
        self.preprocessing_pool = ThreadPoolExecutor(max_workers=4)
        
        # Performance cache
        self.segment_cache = {}
        self.cache_size = 1000
        
        logger.info(f"Initialized Whisper processor: {model_size} on {self.device}")
    
    def _select_optimal_device(self, device: str) -> str:
        """Select optimal device for processing."""
        
        if device == "auto":
            if torch.cuda.is_available():
                # Check CUDA memory
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if memory_gb >= 4:
                    return "cuda"
                else:
                    logger.warning(f"CUDA available but limited memory ({memory_gb:.1f}GB), using CPU")
                    return "cpu"
            else:
                return "cpu"
        
        return device
    
    def _load_optimized_model(self) -> whisper.Whisper:
        """Load model with performance optimizations."""
        
        model = whisper.load_model(self.model_size, device=self.device)
        
        # Apply optimizations
        if self.device == "cuda":
            # Enable cuDNN optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Use half precision for faster inference
            if hasattr(model, 'half'):
                model = model.half()
        
        # Set to evaluation mode
        model.eval()
        
        # Compile model for better performance (PyTorch 2.0+)
        try:
            import torch._dynamo
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile for better performance")
        except:
            logger.info("torch.compile not available, using standard model")
        
        return model
    
    def preprocess_audio_optimized(self, audio_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimized audio preprocessing."""
        
        import librosa
        import soundfile as sf
        
        start_time = time.time()
        
        # Load audio with optimal parameters
        audio, sr = librosa.load(
            audio_path, 
            sr=16000,  # Whisper's expected sample rate
            mono=True,
            dtype=np.float32
        )
        
        # Audio quality analysis
        audio_stats = {
            "duration": len(audio) / sr,
            "sample_rate": sr,
            "channels": 1,
            "rms_energy": np.sqrt(np.mean(audio**2)),
            "max_amplitude": np.max(np.abs(audio)),
            "dynamic_range": np.max(audio) - np.min(audio)
        }
        
        # Noise reduction (optional, based on audio quality)
        if audio_stats["rms_energy"] < 0.01:  # Very quiet audio
            audio = self._enhance_quiet_audio(audio)
        
        # Normalize audio
        if audio_stats["max_amplitude"] > 0:
            audio = audio / audio_stats["max_amplitude"]
        
        # Pad to Whisper's expected length
        audio = whisper.pad_or_trim(audio)
        
        preprocessing_time = time.time() - start_time
        performance_collector.record_timing("whisper.preprocessing", preprocessing_time)
        
        return audio, audio_stats
    
    def _enhance_quiet_audio(self, audio: np.ndarray) -> np.ndarray:
        """Enhance quiet audio for better transcription."""
        
        # Apply gentle noise reduction
        from scipy.signal import butter, filtfilt
        
        # High-pass filter to remove low-frequency noise
        nyquist = 8000  # Half of 16kHz
        low_cutoff = 80  # Hz
        high_cutoff = 7000  # Hz
        
        # Design filter
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_audio = filtfilt(b, a, audio)
        
        # Gentle compression
        compressed_audio = np.sign(filtered_audio) * np.power(np.abs(filtered_audio), 0.8)
        
        return compressed_audio
    
    def batch_transcribe_segments(
        self, 
        audio_segments: List[Tuple[np.ndarray, Dict]], 
        batch_size: int = 4
    ) -> List[Dict[str, Any]]:
        """Batch transcribe multiple audio segments for efficiency."""
        
        results = []
        
        # Process segments in batches
        for i in range(0, len(audio_segments), batch_size):
            batch = audio_segments[i:i + batch_size]
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=min(len(batch), 4)) as executor:
                futures = []
                
                for audio, metadata in batch:
                    future = executor.submit(self._transcribe_single_segment, audio, metadata)
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Segment transcription failed: {e}")
                        results.append({
                            "text": "",
                            "confidence": 0.0,
                            "error": str(e)
                        })
        
        return results
    
    def _transcribe_single_segment(self, audio: np.ndarray, metadata: Dict) -> Dict[str, Any]:
        """Transcribe a single audio segment."""
        
        start_time = time.time()
        
        try:
            # Check cache first
            audio_hash = hash(audio.tobytes())
            if audio_hash in self.segment_cache:
                performance_collector.record_counter("whisper.cache.hits")
                return self.segment_cache[audio_hash]
            
            # Convert to mel spectrogram
            mel = whisper.log_mel_spectrogram(audio)
            
            if self.device == "cuda":
                mel = mel.cuda()
                if mel.dtype != torch.float16 and hasattr(self.model, 'half'):
                    mel = mel.half()
            
            # Transcribe with optimized parameters
            with torch.no_grad():
                result = self.model.transcribe(
                    audio,
                    language="en",
                    task="transcribe",
                    beam_size=1,  # Faster than default 5
                    best_of=1,    # Faster than default 5
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,  # Disable for batch processing
                    fp16=self.device == "cuda"  # Use FP16 on GPU
                )
            
            # Extract results
            transcription_result = {
                "text": result["text"].strip(),
                "language": result["language"],
                "segments": result.get("segments", []),
                "confidence": self._calculate_confidence(result),
                "processing_time": time.time() - start_time
            }
            
            # Cache result
            if len(self.segment_cache) < self.cache_size:
                self.segment_cache[audio_hash] = transcription_result
            
            performance_collector.record_counter("whisper.cache.misses")
            performance_collector.record_timing("whisper.transcribe_segment", time.time() - start_time)
            
            return transcription_result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate average confidence score from Whisper result."""
        
        if "segments" not in result or not result["segments"]:
            return 0.5  # Default confidence
        
        confidences = []
        for segment in result["segments"]:
            if "avg_logprob" in segment:
                # Convert log probability to confidence (0-1)
                confidence = max(0.0, min(1.0, (segment["avg_logprob"] + 1.0)))
                confidences.append(confidence)
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def cleanup_cache(self):
        """Clean up caches to free memory."""
        self.segment_cache.clear()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info("Whisper caches cleared")

# GPU memory management
class GPUMemoryManager:
    """Manages GPU memory for optimal performance."""
    
    def __init__(self):
        self.memory_threshold = 0.9  # 90% memory usage threshold
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Check current GPU memory usage."""
        
        if not torch.cuda.is_available():
            return {"available": 0, "used": 0, "total": 0}
        
        memory_stats = torch.cuda.memory_stats(0)
        allocated = memory_stats["allocated_bytes.all.current"] / 1e9
        reserved = memory_stats["reserved_bytes.all.current"] / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "usage_percent": (reserved / total) * 100
        }
    
    def should_cleanup_memory(self) -> bool:
        """Check if memory cleanup is needed."""
        
        memory_info = self.check_memory_usage()
        return memory_info.get("usage_percent", 0) > self.memory_threshold * 100
    
    def cleanup_memory(self):
        """Cleanup GPU memory."""
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            memory_info = self.check_memory_usage()
            logger.info(f"GPU memory after cleanup: {memory_info['usage_percent']:.1f}% used")

# Performance monitoring for Whisper service
gpu_memory_manager = GPUMemoryManager()

def monitor_whisper_performance():
    """Monitor Whisper service performance metrics."""
    
    while True:
        try:
            # GPU memory monitoring
            memory_info = gpu_memory_manager.check_memory_usage()
            performance_collector.record_gauge("whisper.gpu.memory_used", 
                                             memory_info.get("usage_percent", 0), "percent")
            
            # Cleanup if needed
            if gpu_memory_manager.should_cleanup_memory():
                gpu_memory_manager.cleanup_memory()
                performance_collector.record_counter("whisper.gpu.cleanup")
            
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
            time.sleep(60)
```

This completes the first major section of Part 7, covering performance architecture, database optimization, API optimization, and AI services performance. The implementation demonstrates:

1. **Comprehensive performance monitoring** with metrics collection
2. **Database optimization strategies** including indexing and query optimization
3. **API response optimization** with caching and streaming
4. **AI service performance tuning** with batching and GPU memory management
5. **Real-world performance targets** and SLA definitions

Would you like me to continue with the remaining sections covering Frontend Performance, Caching Strategies, Load Testing & Monitoring, and Scalability Planning?