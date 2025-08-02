"""
AI Meeting Intelligence Platform - Celery Configuration
=====================================================
This module configures Celery for asynchronous task processing.
Handles background jobs for transcription, AI analysis, and embedding generation.

Key Features:
- Redis as message broker and result backend
- Task routing and priority queues
- Retry policies and error handling
- Task monitoring and health checks
- Auto-discovery of task modules
- Production-ready configuration
"""

import os
from celery import Celery
from kombu import Queue, Exchange
from datetime import timedelta

# Environment variables with defaults
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', REDIS_URL)

# Application name for Celery
CELERY_APP_NAME = 'meeting_intelligence'

# Create Celery application instance
celery_app = Celery(
    CELERY_APP_NAME,
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        'tasks.transcription',    # Transcription tasks
        'tasks.insights',         # AI insight extraction tasks  
        'tasks.embeddings',       # Vector embedding tasks
        'tasks.notifications',    # Notification tasks
        'tasks.cleanup'          # Cleanup and maintenance tasks
    ]
)

# Celery configuration dictionary
celery_config = {
    # Broker settings
    'broker_url': CELERY_BROKER_URL,
    'result_backend': CELERY_RESULT_BACKEND,
    
    # Task execution settings
    'task_serializer': 'json',           # Use JSON for task serialization
    'accept_content': ['json'],          # Only accept JSON content
    'result_serializer': 'json',         # Use JSON for results
    'timezone': 'UTC',                   # Use UTC timezone
    'enable_utc': True,                  # Enable UTC
    
    # Task routing and queues
    'task_routes': {
        # Route transcription tasks to high-priority queue
        'tasks.transcription.*': {
            'queue': 'transcription',
            'routing_key': 'transcription',
            'priority': 8
        },
        
        # Route AI tasks to GPU queue if available
        'tasks.insights.*': {
            'queue': 'ai_processing', 
            'routing_key': 'ai',
            'priority': 7
        },
        
        # Route embedding tasks to CPU queue
        'tasks.embeddings.*': {
            'queue': 'embeddings',
            'routing_key': 'embeddings', 
            'priority': 6
        },
        
        # Route notifications to low-priority queue
        'tasks.notifications.*': {
            'queue': 'notifications',
            'routing_key': 'notifications',
            'priority': 3
        },
        
        # Route cleanup tasks to maintenance queue
        'tasks.cleanup.*': {
            'queue': 'maintenance',
            'routing_key': 'maintenance',
            'priority': 1
        }
    },
    
    # Queue definitions with different priorities
    'task_queues': (
        # High-priority transcription queue
        Queue(
            'transcription',
            Exchange('transcription', type='direct'),
            routing_key='transcription',
            queue_arguments={'x-max-priority': 10}
        ),
        
        # AI processing queue (may need GPU resources)
        Queue(
            'ai_processing',
            Exchange('ai', type='direct'), 
            routing_key='ai',
            queue_arguments={'x-max-priority': 10}
        ),
        
        # Embedding generation queue
        Queue(
            'embeddings',
            Exchange('embeddings', type='direct'),
            routing_key='embeddings',
            queue_arguments={'x-max-priority': 10}
        ),
        
        # Low-priority notification queue
        Queue(
            'notifications',
            Exchange('notifications', type='direct'),
            routing_key='notifications',
            queue_arguments={'x-max-priority': 10}
        ),
        
        # Maintenance and cleanup queue
        Queue(
            'maintenance', 
            Exchange('maintenance', type='direct'),
            routing_key='maintenance',
            queue_arguments={'x-max-priority': 10}
        ),
        
        # Default queue for uncategorized tasks
        Queue(
            'default',
            Exchange('default', type='direct'),
            routing_key='default',
            queue_arguments={'x-max-priority': 10}
        )
    ),
    
    # Task execution limits
    'task_acks_late': True,              # Acknowledge tasks after completion
    'task_reject_on_worker_lost': True,  # Reject tasks if worker dies
    'task_track_started': True,          # Track when tasks start
    
    # Worker settings
    'worker_prefetch_multiplier': 1,     # Prefetch one task at a time for fairness
    'worker_max_tasks_per_child': 1000,  # Restart worker after 1000 tasks
    'worker_disable_rate_limits': False, # Enable rate limiting
    
    # Task time limits (in seconds)
    'task_soft_time_limit': 300,        # Soft limit: 5 minutes
    'task_time_limit': 600,              # Hard limit: 10 minutes  
    
    # Retry settings
    'task_default_retry_delay': 60,      # Default retry delay: 1 minute
    'task_max_retries': 3,               # Maximum retries per task
    
    # Result backend settings
    'result_expires': 3600,              # Results expire after 1 hour
    'result_cache_max': 10000,           # Cache up to 10k results
    
    # Monitoring and logging
    'worker_send_task_events': True,     # Send task events for monitoring
    'task_send_sent_event': True,        # Send task sent events
    'worker_log_format': '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    'worker_task_log_format': '[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s',
    
    # Security settings
    'worker_hijack_root_logger': False,  # Don't hijack root logger
    'worker_log_color': False,           # Disable colored logs in production
    
    # Memory and resource management
    'worker_max_memory_per_child': 200000,  # 200MB per worker child
    'worker_autoscaler': 'celery.worker.autoscale:Autoscaler',
    
    # Beat scheduler settings (for periodic tasks)
    'beat_schedule': {
        # Cleanup old tasks every hour
        'cleanup-old-tasks': {
            'task': 'tasks.cleanup.cleanup_old_tasks',
            'schedule': timedelta(hours=1),
            'kwargs': {'days_old': 7}
        },
        
        # Generate system health report every 5 minutes
        'system-health-check': {
            'task': 'tasks.monitoring.system_health_check',
            'schedule': timedelta(minutes=5)
        },
        
        # Backup database every day at 2 AM
        'daily-backup': {
            'task': 'tasks.maintenance.backup_database',
            'schedule': timedelta(days=1),
            'options': {'queue': 'maintenance'}
        }
    },
    
    # Custom task annotations for specific behavior
    'task_annotations': {
        # Transcription tasks get more memory
        'tasks.transcription.*': {
            'rate_limit': '10/m',        # Max 10 transcription tasks per minute
            'time_limit': 1800,          # 30 minutes for long audio
            'soft_time_limit': 1500      # 25 minutes soft limit
        },
        
        # AI tasks may need GPU resources
        'tasks.insights.*': {
            'rate_limit': '5/m',         # Max 5 AI tasks per minute
            'time_limit': 900,           # 15 minutes for AI processing
            'soft_time_limit': 600       # 10 minutes soft limit
        },
        
        # Embedding tasks are CPU intensive
        'tasks.embeddings.*': {
            'rate_limit': '20/m',        # Max 20 embedding tasks per minute
            'time_limit': 300,           # 5 minutes for embeddings
            'soft_time_limit': 240       # 4 minutes soft limit
        },
        
        # Notification tasks are lightweight
        'tasks.notifications.*': {
            'rate_limit': '100/m',       # Max 100 notifications per minute
            'time_limit': 60,            # 1 minute for notifications
            'soft_time_limit': 30        # 30 seconds soft limit
        }
    }
}

# Apply configuration to Celery app
celery_app.config_from_object(celery_config)

# Task base class with common functionality
class BaseTask(celery_app.Task):
    """
    Base task class with common error handling and logging.
    All custom tasks should inherit from this class.
    """
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """
        Called when task fails. Updates database with error information
        and sends notifications if needed.
        """
        from db import update_task_status
        
        # Update task status in database
        error_message = f"{exc.__class__.__name__}: {str(exc)}"
        update_task_status(
            task_id=task_id,
            state='FAILURE',
            error=error_message
        )
        
        # Log the error
        self.get_logger().error(
            f"Task {task_id} failed: {error_message}",
            exc_info=einfo
        )
        
        # Send failure notification for critical tasks
        if self.name.startswith('tasks.transcription') or self.name.startswith('tasks.insights'):
            from tasks.notifications import send_task_failure_notification
            send_task_failure_notification.delay(
                task_id=task_id,
                task_name=self.name,
                error_message=error_message
            )
    
    def on_success(self, retval, task_id, args, kwargs):
        """
        Called when task succeeds. Updates database with success status.
        """
        from db import update_task_status
        
        # Update task status in database
        update_task_status(
            task_id=task_id,
            state='SUCCESS',
            progress=100,
            result=retval
        )
        
        # Log success
        self.get_logger().info(f"Task {task_id} completed successfully")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """
        Called when task is retried. Updates database with retry status.
        """
        from db import update_task_status
        
        # Update task status in database  
        update_task_status(
            task_id=task_id,
            state='RETRY',
            error=f"Retry due to: {str(exc)}"
        )
        
        # Log retry
        self.get_logger().warning(f"Task {task_id} is being retried: {str(exc)}")
    
    def update_progress(self, task_id, progress, current_stage=None):
        """
        Helper method to update task progress in database.
        
        Args:
            task_id (str): Task identifier
            progress (int): Progress percentage (0-100)
            current_stage (str): Description of current processing stage
        """
        from db import update_task_status
        
        update_task_status(
            task_id=task_id,
            state='PROGRESS',
            progress=progress
        )
        
        # Update current stage if provided
        if current_stage:
            # This would update the current_stage field in the task record
            # Implementation depends on your database schema
            pass

# Set the base task class
celery_app.Task = BaseTask

# Utility functions for Celery management
def get_celery_worker_status():
    """
    Get status of all Celery workers.
    
    Returns:
        dict: Worker status information
    """
    try:
        # Get active workers
        active_workers = celery_app.control.inspect().active()
        
        # Get worker statistics
        stats = celery_app.control.inspect().stats()
        
        # Get registered tasks
        registered_tasks = celery_app.control.inspect().registered()
        
        return {
            'status': 'healthy',
            'active_workers': active_workers or {},
            'worker_stats': stats or {},
            'registered_tasks': registered_tasks or {},
            'total_workers': len(active_workers) if active_workers else 0
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'total_workers': 0
        }

def get_queue_status():
    """
    Get status of all Celery queues.
    
    Returns:
        dict: Queue status information with message counts
    """
    try:
        # Get queue lengths (requires Redis broker)
        from celery.backends.redis import RedisBackend
        
        backend = RedisBackend(app=celery_app)
        redis_client = backend.client
        
        queues_info = {}
        
        # Check each defined queue
        for queue in celery_config['task_queues']:
            queue_name = queue.name
            # Get queue length from Redis
            queue_length = redis_client.llen(f"celery:{queue_name}")
            
            queues_info[queue_name] = {
                'length': queue_length,
                'routing_key': queue.routing_key,
                'exchange': queue.exchange.name
            }
        
        return {
            'status': 'healthy',
            'queues': queues_info
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy', 
            'error': str(e),
            'queues': {}
        }

def purge_all_queues():
    """
    Purge all messages from all queues.
    Use with caution - this will delete all pending tasks.
    
    Returns:
        dict: Results of purge operations
    """
    try:
        results = {}
        
        for queue in celery_config['task_queues']:
            queue_name = queue.name
            purged_count = celery_app.control.purge()
            results[queue_name] = purged_count
        
        return {
            'status': 'success',
            'purged_queues': results
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

# Health check function
def celery_health_check():
    """
    Comprehensive health check for Celery system.
    
    Returns:
        dict: Complete health status
    """
    worker_status = get_celery_worker_status()
    queue_status = get_queue_status()
    
    # Overall health is healthy if both workers and queues are healthy
    overall_healthy = (
        worker_status['status'] == 'healthy' and 
        queue_status['status'] == 'healthy' and
        worker_status['total_workers'] > 0
    )
    
    return {
        'status': 'healthy' if overall_healthy else 'unhealthy',
        'workers': worker_status,
        'queues': queue_status,
        'broker_url': CELERY_BROKER_URL,
        'result_backend': CELERY_RESULT_BACKEND
    }

# Export configuration and utility functions
__all__ = [
    'celery_app', 'celery_config', 'BaseTask',
    'get_celery_worker_status', 'get_queue_status', 
    'purge_all_queues', 'celery_health_check'
]

# Initialize Celery when module is imported
if __name__ == '__main__':
    print("🔄 Celery configuration loaded")
    print(f"📡 Broker: {CELERY_BROKER_URL}")
    print(f"💾 Backend: {CELERY_RESULT_BACKEND}")
    
    # Print health check
    health = celery_health_check()
    print(f"🏥 Health Status: {health['status']}")
    print(f"👷 Active Workers: {health['workers']['total_workers']}")
