"""
Custom exception classes for the research agent.

This module provides specific exception types for better error handling
and debugging throughout the agent codebase.
"""

from typing import Optional, Dict, Any


class ResearchAgentError(Exception):
    """Base exception for all research agent errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(ResearchAgentError):
    """Raised when there's an issue with agent configuration."""
    pass


class LLMEndpointError(ResearchAgentError):
    """Raised when there's an issue with LLM endpoint communication."""
    pass


class SearchToolError(ResearchAgentError):
    """Raised when there's an issue with search tools (Tavily, Vector Search)."""
    pass


class VectorSearchError(SearchToolError):
    """Raised when there's an issue with vector search specifically."""
    pass


class TavilySearchError(SearchToolError):
    """Raised when there's an issue with Tavily search specifically."""
    pass


class WorkflowError(ResearchAgentError):
    """Raised when there's an issue in the LangGraph workflow execution."""
    pass


class MessageConversionError(ResearchAgentError):
    """Raised when there's an issue converting between message formats."""
    pass


class ContentExtractionError(ResearchAgentError):
    """Raised when there's an issue extracting content from responses."""
    pass


class ToolInitializationError(ResearchAgentError):
    """Raised when there's an issue initializing tools."""
    pass


class ValidationError(ResearchAgentError):
    """Raised when input validation fails."""
    pass


class TimeoutError(ResearchAgentError):
    """Raised when an operation times out."""
    pass


class CircuitBreakerError(ResearchAgentError):
    """Raised when a circuit breaker is open."""
    pass


class RetryExhaustedError(ResearchAgentError):
    """Raised when retry attempts are exhausted."""
    pass