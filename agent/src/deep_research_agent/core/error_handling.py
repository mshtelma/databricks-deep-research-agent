"""
Enhanced error handling and wrapping for agent operations.

Provides comprehensive error handling, user-friendly error formatting,
and proper error propagation from agent to UI.
"""

import traceback
import logging
from typing import Dict, Any, Optional, Union, Type
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AgentErrorType(Enum):
    """Types of agent errors for categorization."""
    
    # State management errors  
    STATE_UPDATE_ERROR = "state_update_error"
    CONCURRENT_UPDATE_ERROR = "concurrent_update_error"
    
    # Agent execution errors
    AGENT_EXECUTION_ERROR = "agent_execution_error"
    WORKFLOW_ERROR = "workflow_error"
    
    # Search and tool errors
    SEARCH_ERROR = "search_error"
    TOOL_ERROR = "tool_error" 
    API_ERROR = "api_error"
    
    # Configuration errors
    CONFIG_ERROR = "config_error"
    AUTHENTICATION_ERROR = "authentication_error"
    
    # Content processing errors
    CONTENT_ERROR = "content_error"
    PARSING_ERROR = "parsing_error"
    
    # Resource errors
    RATE_LIMIT_ERROR = "rate_limit_error"
    QUOTA_ERROR = "quota_error"
    TIMEOUT_ERROR = "timeout_error"
    
    # Unknown errors
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class AgentError:
    """Structured agent error with user-friendly messaging."""
    
    # Core error info
    error_type: AgentErrorType
    error_code: str
    internal_message: str  # For logs
    user_message: str     # For UI display
    
    # Context
    timestamp: datetime
    agent_name: Optional[str] = None
    operation: Optional[str] = None
    request_id: Optional[str] = None
    
    # Technical details (for debugging)
    exception_type: Optional[str] = None
    traceback: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    
    # Recovery suggestions
    retry_possible: bool = False
    suggested_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error_type": self.error_type.value,
            "error_code": self.error_code,
            "user_message": self.user_message,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "operation": self.operation,
            "request_id": self.request_id,
            "retry_possible": self.retry_possible,
            "suggested_action": self.suggested_action,
            # Technical details only included in debug mode
            "technical_details": {
                "internal_message": self.internal_message,
                "exception_type": self.exception_type,
                "context": self.context
            } if logger.isEnabledFor(logging.DEBUG) else None
        }


class AgentErrorHandler:
    """
    Centralized error handling for agent operations.
    
    Provides error categorization, user-friendly messaging,
    and proper logging for debugging.
    """
    
    # Error type detection patterns
    ERROR_PATTERNS = {
        AgentErrorType.CONCURRENT_UPDATE_ERROR: [
            "Can receive only one value per step",
            "INVALID_CONCURRENT_GRAPH_UPDATE",
            "concurrent update",
            "multiple values"
        ],
        AgentErrorType.STATE_UPDATE_ERROR: [
            "state update",
            "invalid state", 
            "state conflict",
            "state validation"
        ],
        AgentErrorType.SEARCH_ERROR: [
            "search failed",
            "search provider",
            "search timeout",
            "no search results"
        ],
        AgentErrorType.API_ERROR: [
            "api error",
            "http error",
            "connection error",
            "api timeout"
        ],
        AgentErrorType.RATE_LIMIT_ERROR: [
            "rate limit",
            "too many requests",
            "quota exceeded",
            "429"
        ],
        AgentErrorType.TIMEOUT_ERROR: [
            "timeout",
            "timed out",
            "deadline exceeded"
        ],
        AgentErrorType.AUTHENTICATION_ERROR: [
            "authentication",
            "unauthorized",
            "invalid credentials",
            "401",
            "403"
        ],
        AgentErrorType.CONFIG_ERROR: [
            "configuration",
            "config error",
            "missing config",
            "invalid config"
        ],
        AgentErrorType.WORKFLOW_ERROR: [
            "workflow transition",
            "workflow error",
            "invalid workflow",
            "workflow failed"
        ]
    }
    
    # User-friendly error messages
    USER_MESSAGES = {
        AgentErrorType.CONCURRENT_UPDATE_ERROR: "The research agent encountered a workflow conflict. This is typically a temporary issue that resolves on retry.",
        AgentErrorType.STATE_UPDATE_ERROR: "There was an issue updating the research progress. Please try your request again.", 
        AgentErrorType.SEARCH_ERROR: "Unable to complete web searches at this time. Please check your internet connection and try again.",
        AgentErrorType.API_ERROR: "There was a problem connecting to external services. Please try again in a few moments.",
        AgentErrorType.RATE_LIMIT_ERROR: "Too many requests have been made recently. Please wait a moment and try again.",
        AgentErrorType.TIMEOUT_ERROR: "The research request took too long to complete. Please try again with a simpler question.",
        AgentErrorType.AUTHENTICATION_ERROR: "Authentication failed. Please check your credentials and try again.",
        AgentErrorType.CONFIG_ERROR: "There's a configuration issue preventing the research agent from working properly.",
        AgentErrorType.CONTENT_ERROR: "Unable to process the provided content. Please try rephrasing your question.",
        AgentErrorType.PARSING_ERROR: "There was an issue parsing the research results. Please try again.",
        AgentErrorType.WORKFLOW_ERROR: "The research workflow encountered an unexpected issue. Please try again.",
        AgentErrorType.TOOL_ERROR: "One of the research tools failed to execute properly. Please try again.",
        AgentErrorType.QUOTA_ERROR: "The service quota has been exceeded. Please try again later.",
        AgentErrorType.AGENT_EXECUTION_ERROR: "The research agent encountered an error while processing your request.",
        AgentErrorType.UNKNOWN_ERROR: "An unexpected error occurred. Please try again, and contact support if the issue persists."
    }
    
    # Recovery suggestions
    RECOVERY_SUGGESTIONS = {
        AgentErrorType.CONCURRENT_UPDATE_ERROR: "Try your request again - this type of error usually resolves automatically.",
        AgentErrorType.SEARCH_ERROR: "Check your internet connection and try again.",
        AgentErrorType.RATE_LIMIT_ERROR: "Please wait a few minutes before making another request.",
        AgentErrorType.TIMEOUT_ERROR: "Try asking a more specific or simpler question.",
        AgentErrorType.AUTHENTICATION_ERROR: "Please check your authentication settings.",
        AgentErrorType.CONFIG_ERROR: "Contact your administrator to check the agent configuration.",
        AgentErrorType.UNKNOWN_ERROR: "Try again, and contact support if the problem continues."
    }
    
    def __init__(self, request_id: Optional[str] = None):
        """
        Initialize error handler.
        
        Args:
            request_id: Optional request ID for tracking
        """
        self.request_id = request_id
    
    def handle_exception(
        self,
        exception: Exception,
        agent_name: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentError:
        """
        Handle an exception and convert to structured AgentError.
        
        Args:
            exception: The exception to handle
            agent_name: Name of the agent where error occurred
            operation: Operation being performed when error occurred
            context: Additional context information
            
        Returns:
            AgentError: Structured error object
        """
        error_type = self._classify_error(exception)
        error_code = self._generate_error_code(error_type, exception)
        internal_message = str(exception)
        user_message = self._get_user_message(error_type, exception)
        
        # Get traceback for debugging
        tb = traceback.format_exc()
        
        # Check if retry is possible
        retry_possible = self._is_retry_possible(error_type, exception)
        
        # Get recovery suggestion
        suggested_action = self._get_recovery_suggestion(error_type)
        
        agent_error = AgentError(
            error_type=error_type,
            error_code=error_code,
            internal_message=internal_message,
            user_message=user_message,
            timestamp=datetime.now(),
            agent_name=agent_name,
            operation=operation,
            request_id=self.request_id,
            exception_type=type(exception).__name__,
            traceback=tb,
            context=context,
            retry_possible=retry_possible,
            suggested_action=suggested_action
        )
        
        # Log the error appropriately
        self._log_error(agent_error)
        
        return agent_error
    
    def _classify_error(self, exception: Exception) -> AgentErrorType:
        """Classify exception into error type."""
        error_msg = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        # Check specific exception types first
        if "langgraph" in exception_type or "concurrent" in error_msg:
            if "can receive only one value per step" in error_msg:
                return AgentErrorType.CONCURRENT_UPDATE_ERROR
            if "invalid_concurrent_graph_update" in error_msg:
                return AgentErrorType.CONCURRENT_UPDATE_ERROR
            return AgentErrorType.STATE_UPDATE_ERROR
        
        # Check error message patterns
        for error_type, patterns in self.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in error_msg:
                    return error_type
        
        # Check exception types
        if "timeout" in exception_type:
            return AgentErrorType.TIMEOUT_ERROR
        elif "connection" in exception_type or "http" in exception_type:
            return AgentErrorType.API_ERROR
        elif "auth" in exception_type:
            return AgentErrorType.AUTHENTICATION_ERROR
        elif "config" in exception_type:
            return AgentErrorType.CONFIG_ERROR
        
        return AgentErrorType.UNKNOWN_ERROR
    
    def _generate_error_code(self, error_type: AgentErrorType, exception: Exception) -> str:
        """Generate unique error code for tracking."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Include microseconds for uniqueness
        type_code = error_type.value.upper()[:8]
        exception_code = type(exception).__name__[:4].upper()
        
        return f"AGT_{type_code}_{exception_code}_{timestamp}"
    
    def _get_user_message(self, error_type: AgentErrorType, exception: Exception) -> str:
        """Get user-friendly error message."""
        base_message = self.USER_MESSAGES.get(error_type, self.USER_MESSAGES[AgentErrorType.UNKNOWN_ERROR])
        
        # Add specific details for certain error types
        if error_type == AgentErrorType.CONCURRENT_UPDATE_ERROR:
            base_message += " This usually happens during complex multi-step research and typically resolves when you try again."
        elif error_type == AgentErrorType.SEARCH_ERROR:
            if "no results" in str(exception).lower():
                base_message += " No relevant information was found for your query."
        
        return base_message
    
    def _is_retry_possible(self, error_type: AgentErrorType, exception: Exception) -> bool:
        """Determine if the error is retryable."""
        retryable_types = {
            AgentErrorType.CONCURRENT_UPDATE_ERROR,
            AgentErrorType.SEARCH_ERROR,
            AgentErrorType.API_ERROR,
            AgentErrorType.RATE_LIMIT_ERROR,
            AgentErrorType.TIMEOUT_ERROR,
            AgentErrorType.WORKFLOW_ERROR,
            AgentErrorType.TOOL_ERROR
        }
        
        non_retryable_types = {
            AgentErrorType.AUTHENTICATION_ERROR,
            AgentErrorType.CONFIG_ERROR,
            AgentErrorType.QUOTA_ERROR
        }
        
        # Check for specific non-retryable patterns in exception message
        exception_msg = str(exception).lower()
        non_retryable_patterns = ["permanent", "invalid permanent state", "configuration"]
        
        if any(pattern in exception_msg for pattern in non_retryable_patterns):
            return False
        
        if error_type in retryable_types:
            return True
        elif error_type in non_retryable_types:
            return False
        else:
            # For unknown types, default to retryable unless explicitly permanent
            return True
    
    def _get_recovery_suggestion(self, error_type: AgentErrorType) -> Optional[str]:
        """Get recovery suggestion for error type."""
        return self.RECOVERY_SUGGESTIONS.get(error_type)
    
    def _log_error(self, agent_error: AgentError):
        """Log error with appropriate level."""
        log_msg = (
            f"[{agent_error.error_code}] {agent_error.error_type.value}: {agent_error.internal_message}"
        )
        
        if agent_error.agent_name:
            log_msg = f"[{agent_error.agent_name}] {log_msg}"
        
        if agent_error.operation:
            log_msg += f" (during {agent_error.operation})"
        
        # Log level based on error type
        if agent_error.error_type in [
            AgentErrorType.CONFIG_ERROR,
            AgentErrorType.AUTHENTICATION_ERROR,
            AgentErrorType.QUOTA_ERROR
        ]:
            logger.error(log_msg)
            logger.error(f"Traceback: {agent_error.traceback}")
        elif agent_error.error_type in [
            AgentErrorType.CONCURRENT_UPDATE_ERROR,
            AgentErrorType.RATE_LIMIT_ERROR
        ]:
            logger.warning(log_msg)
            logger.debug(f"Traceback: {agent_error.traceback}")
        else:
            logger.info(log_msg)
            logger.debug(f"Traceback: {agent_error.traceback}")


def handle_agent_error(
    exception: Exception,
    agent_name: Optional[str] = None,
    operation: Optional[str] = None,
    request_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> AgentError:
    """
    Convenience function for handling agent errors.
    
    Args:
        exception: The exception to handle
        agent_name: Name of the agent where error occurred
        operation: Operation being performed
        request_id: Request ID for tracking
        context: Additional context
        
    Returns:
        AgentError: Structured error object
    """
    handler = AgentErrorHandler(request_id)
    return handler.handle_exception(exception, agent_name, operation, context)


def create_error_response(agent_error: AgentError) -> Dict[str, Any]:
    """
    Create standardized error response for API.
    
    Args:
        agent_error: Structured error object
        
    Returns:
        Dict containing error response
    """
    return {
        "success": False,
        "error": {
            "code": agent_error.error_code,
            "message": agent_error.user_message,
            "type": agent_error.error_type.value,
            "timestamp": agent_error.timestamp.isoformat(),
            "retry_possible": agent_error.retry_possible,
            "suggested_action": agent_error.suggested_action
        },
        "request_id": agent_error.request_id
    }