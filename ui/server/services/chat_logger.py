"""Chat logging service for tracking user prompts and responses."""

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure structured logger
logger = logging.getLogger(__name__)


class ChatLogger:
    """Service for logging chat interactions with structured data."""
    
    def __init__(self, log_file_path: str = "/tmp/databricks-app-chat.log"):
        """Initialize chat logger with file and console handlers."""
        self.log_file_path = log_file_path
        self.chat_logger = logging.getLogger("chat_audit")
        self.chat_logger.setLevel(logging.INFO)
        
        # Ensure log directory exists
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # File handler with JSON formatting
        if not any(isinstance(h, logging.FileHandler) for h in self.chat_logger.handlers):
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(JsonFormatter())
            self.chat_logger.addHandler(file_handler)
        
        # Console handler for important events
        if not any(isinstance(h, logging.StreamHandler) for h in self.chat_logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - CHAT - %(message)s')
            )
            self.chat_logger.addHandler(console_handler)
    
    def log_request(
        self,
        request_id: str,
        user_info: Dict[str, Any],
        prompt: str,
        messages_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Log incoming chat request with user and prompt information."""
        log_entry = {
            "event_type": "chat_request",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": request_id,
            "user": user_info,
            "prompt": prompt[:1000],  # Truncate very long prompts
            "prompt_length": len(prompt),
            "conversation_length": len(messages_history) if messages_history else 1,
        }
        
        self.chat_logger.info(json.dumps(log_entry))
        logger.info(f"Chat request from {user_info.get('username', 'unknown')}: {prompt[:100]}...")
        
        return log_entry
    
    def log_response(
        self,
        request_id: str,
        response: str,
        elapsed_time_ms: float,
        stream_events_count: int = 0,
        metadata: Optional[Dict] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Log chat response with metrics and outcome."""
        # Calculate response summary
        response_summary = response[:500] if response else ""
        
        log_entry = {
            "event_type": "chat_response",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": request_id,
            "response_summary": response_summary,
            "response_length": len(response) if response else 0,
            "metrics": {
                "response_time_ms": elapsed_time_ms,
                "stream_events": stream_events_count,
                "estimated_tokens": len(response.split()) * 1.3 if response else 0,  # Rough estimate
            },
            "status": "error" if error else "success",
            "error": error,
            "metadata": metadata
        }
        
        self.chat_logger.info(json.dumps(log_entry))
        
        status_emoji = "❌" if error else "✅"
        logger.info(
            f"{status_emoji} Chat response for {request_id}: "
            f"{elapsed_time_ms:.0f}ms, {len(response) if response else 0} chars"
        )
        
        return log_entry
    
    def log_streaming_event(
        self,
        request_id: str,
        event_type: str,
        phase: Optional[str] = None,
        progress: Optional[float] = None
    ):
        """Log streaming progress events for monitoring."""
        log_entry = {
            "event_type": "streaming_progress",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": request_id,
            "stream_event_type": event_type,
            "phase": phase,
            "progress_percentage": progress
        }
        
        # Only log significant events to avoid spam
        if event_type in ["research_update", "error", "stream_end"]:
            self.chat_logger.debug(json.dumps(log_entry))
    
    def log_error(
        self,
        request_id: str,
        error_message: str,
        error_type: str = "unknown",
        traceback_str: Optional[str] = None
    ):
        """Log error events with full context."""
        log_entry = {
            "event_type": "chat_error",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": request_id,
            "error_type": error_type,
            "error_message": error_message,
            "traceback": traceback_str
        }
        
        self.chat_logger.error(json.dumps(log_entry))
        logger.error(f"Chat error for {request_id}: {error_type} - {error_message}")
    
    def get_stats(self, last_hours: int = 24) -> Dict[str, Any]:
        """Get chat statistics from logs (optional analytics feature)."""
        # This could parse the log file and provide statistics
        # For now, return a placeholder
        return {
            "message": "Statistics calculation not yet implemented",
            "last_hours": last_hours
        }


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        # If the message is already JSON, return it as-is
        if isinstance(record.msg, str) and record.msg.startswith('{'):
            return record.msg
        
        # Otherwise, create a JSON structure
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'request_id'):
            log_obj['request_id'] = record.request_id
        
        return json.dumps(log_obj)


class ChatRequestContext:
    """Context manager for tracking chat request lifecycle."""
    
    def __init__(self, logger: ChatLogger, user_info: Dict, prompt: str, messages: Optional[List] = None):
        """Initialize request context."""
        self.logger = logger
        self.request_id = str(uuid.uuid4())
        self.user_info = user_info
        self.prompt = prompt
        self.messages = messages
        self.start_time = None
        self.stream_events_count = 0
        self.response_content = ""
    
    def __enter__(self):
        """Start request tracking."""
        self.start_time = time.time()
        self.logger.log_request(
            self.request_id,
            self.user_info,
            self.prompt,
            self.messages
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete request tracking."""
        elapsed_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0
        
        if exc_val:
            # Log error case
            self.logger.log_error(
                self.request_id,
                str(exc_val),
                exc_type.__name__ if exc_type else "unknown",
                traceback_str=str(exc_tb) if exc_tb else None
            )
            self.logger.log_response(
                self.request_id,
                self.response_content,
                elapsed_ms,
                self.stream_events_count,
                error=str(exc_val)
            )
        else:
            # Log success case
            self.logger.log_response(
                self.request_id,
                self.response_content,
                elapsed_ms,
                self.stream_events_count
            )
    
    def add_stream_event(self, event_type: str, content: str = "", metadata: Dict = None):
        """Track streaming events."""
        self.stream_events_count += 1
        if content:
            self.response_content += content
        
        # Log significant streaming events
        if event_type == "research_update" and metadata:
            self.logger.log_streaming_event(
                self.request_id,
                event_type,
                phase=metadata.get("phase"),
                progress=metadata.get("progress_percentage")
            )