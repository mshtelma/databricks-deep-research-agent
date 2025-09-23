"""
LLM Response Parser utilities.

Handles extraction of text content from structured LLM responses.
This module provides backward compatibility while using the new abstract response handlers.
"""

from typing import Any, Union, List, Dict, Tuple, Optional
from deep_research_agent.core import get_logger

logger = get_logger(__name__)


def extract_text_from_response(response: Any) -> str:
    """
    Extract text content from various LLM response formats.
    
    This function provides backward compatibility while using the new
    abstract response handling system.
    
    Args:
        response: LLM response in various formats
        
    Returns:
        Extracted text content as string
    """
    # Use the new abstract response handler
    from deep_research_agent.core.response_handlers import extract_text_from_response as new_extract
    return new_extract(response)


def parse_response_with_reasoning(response: Any) -> Tuple[str, Optional[str]]:
    """
    Extract both content and reasoning from LLM response.
    
    Args:
        response: LLM response in various formats
        
    Returns:
        Tuple of (content, reasoning) where reasoning may be None
    """
    from deep_research_agent.core.response_handlers import extract_content_and_reasoning
    return extract_content_and_reasoning(response)


def get_response_metadata(response: Any) -> Dict[str, Any]:
    """
    Get metadata about the response structure.
    
    Args:
        response: LLM response in various formats
        
    Returns:
        Dictionary with metadata about the response
    """
    from deep_research_agent.core.response_handlers import parse_structured_response
    parsed = parse_structured_response(response)
    return {
        'response_type': parsed.response_type.value,
        'has_reasoning': parsed.reasoning is not None,
        'content_length': len(parsed.content),
        'reasoning_length': len(parsed.reasoning) if parsed.reasoning else 0,
        'metadata': parsed.metadata or {}
    }

