"""
Centralized schema conversion for all message and response formats.

This module eliminates the duplicate conversion logic scattered across multiple files
and provides a single source of truth for MLflow, OpenAI, and LangChain format conversions.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from uuid import uuid4
import json
import logging

from mlflow.types.responses import (
    ResponsesAgentRequest, 
    ResponsesAgentResponse, 
    ResponsesAgentStreamEvent
)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from deep_research_agent.core.error_handler import retry, safe_call

logger = logging.getLogger(__name__)


@dataclass
class ConversionResult:
    """Result of a schema conversion with metadata."""
    
    data: Any
    format: str
    metadata: Dict[str, Any]
    warnings: List[str]


class SchemaConverter:
    """
    Centralized schema conversion for all message and response formats.
    
    This class provides consistent conversion between:
    - MLflow ResponsesAgent format
    - OpenAI API format  
    - LangChain message format
    - Databricks chat format
    """
    
    def __init__(self):
        self.conversion_stats = {
            "requests_converted": 0,
            "responses_converted": 0,
            "stream_events_created": 0,
            "warnings_generated": 0
        }
    
    # =========================================================================
    # Request Conversions
    # =========================================================================
    
    @retry("request_to_messages", "io_operation")
    def request_to_messages(self, request: ResponsesAgentRequest) -> List[BaseMessage]:
        """
        Convert MLflow ResponsesAgentRequest to LangChain messages.
        
        Args:
            request: MLflow request object
            
        Returns:
            List of LangChain BaseMessage objects
        """
        messages = []
        warnings = []
        
        try:
            input_data = request.input if hasattr(request, 'input') else []
            
            for msg in input_data:
                try:
                    langchain_msg = self._convert_single_message(msg)
                    if langchain_msg:
                        messages.append(langchain_msg)
                except Exception as e:
                    warnings.append(f"Failed to convert message: {e}")
                    logger.warning(f"Message conversion failed: {e}", extra={"message": msg})
            
            self.conversion_stats["requests_converted"] += 1
            self.conversion_stats["warnings_generated"] += len(warnings)
            
            return messages
            
        except Exception as e:
            logger.error(f"Request conversion failed: {e}")
            raise ValueError(f"Failed to convert request to messages: {e}")
    
    def _convert_single_message(self, msg: Dict[str, Any]) -> Optional[BaseMessage]:
        """Convert single message dictionary to LangChain format."""
        if not isinstance(msg, dict):
            return None
        
        role = msg.get("role", "user").lower()
        content = self._extract_content(msg)
        
        if not content:
            return None
        
        # Convert based on role
        if role == "system":
            return SystemMessage(content=content)
        elif role == "assistant":
            return AIMessage(content=content)
        else:  # Default to user
            return HumanMessage(content=content)
    
    def _extract_content(self, msg: Dict[str, Any]) -> str:
        """
        Extract content from various message formats with robust handling.
        
        Supports:
        - Simple string content
        - Structured content arrays
        - OpenAI-style content format
        - Databricks message format
        """
        if "content" not in msg:
            return ""
        
        content = msg["content"]
        
        # Handle string content directly
        if isinstance(content, str):
            return content
        
        # Handle structured content (list format)
        elif isinstance(content, list):
            return self._extract_from_content_list(content)
        
        # Handle other formats
        else:
            return str(content)
    
    def _extract_from_content_list(self, content_list: List[Dict]) -> str:
        """Extract text from structured content list."""
        text_parts = []
        
        for item in content_list:
            if not isinstance(item, dict):
                continue
            
            # Handle different content item types
            if "text" in item:
                text_parts.append(str(item["text"]))
            elif item.get("type") == "output_text":
                text_parts.append(str(item.get("text", "")))
            elif item.get("type") == "text":
                text_parts.append(str(item.get("content", "")))
        
        return " ".join(text_parts)
    
    # =========================================================================
    # Response Conversions
    # =========================================================================
    
    @retry("format_response", "io_operation")
    def format_response(
        self,
        content: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        format_type: str = "responses_agent"
    ) -> Dict[str, Any]:
        """
        Format response in specified schema format.
        
        Args:
            content: Response content text
            citations: List of citation dictionaries
            metadata: Response metadata
            format_type: Target format (responses_agent, openai_chat, databricks)
            
        Returns:
            Formatted response dictionary
        """
        citations = citations or []
        metadata = metadata or {}
        
        if format_type == "openai_chat":
            return self._format_openai_response(content, citations, metadata)
        elif format_type == "databricks":
            return self._format_databricks_response(content, citations, metadata)
        else:  # Default to responses_agent
            return self._format_responses_agent_response(content, citations, metadata)
    
    def _format_responses_agent_response(
        self, 
        content: str, 
        citations: List[Dict], 
        metadata: Dict
    ) -> Dict[str, Any]:
        """Format response for MLflow ResponsesAgent compliance."""
        return {
            "type": "message",
            "role": "assistant",
            "id": str(uuid4()),
            "content": [{
                "type": "output_text",
                "text": content
            }],
            "citations": self._standardize_citations(citations),
            "metadata": metadata
        }
    
    def _format_openai_response(
        self, 
        content: str, 
        citations: List[Dict], 
        metadata: Dict
    ) -> Dict[str, Any]:
        """Format response in OpenAI chat completion style."""
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "usage": metadata.get("usage", {}),
            "citations": self._standardize_citations(citations),
            "metadata": metadata
        }
    
    def _format_databricks_response(
        self, 
        content: str, 
        citations: List[Dict], 
        metadata: Dict
    ) -> Dict[str, Any]:
        """Format response for Databricks chat format."""
        return {
            "content": content,
            "citations": self._standardize_citations(citations),
            "metadata": metadata
        }
    
    def _standardize_citations(self, citations: List[Dict]) -> List[Dict]:
        """Standardize citation format across all response types."""
        standardized = []
        
        for citation in citations:
            if isinstance(citation, dict):
                standardized.append({
                    "title": citation.get("title", ""),
                    "url": citation.get("url", ""),
                    "snippet": citation.get("snippet", citation.get("content", "")),
                    "source": citation.get("source", ""),
                    "score": float(citation.get("score", 0.0))
                })
        
        return standardized
    
    # =========================================================================
    # Stream Event Creation
    # =========================================================================
    
    def create_stream_event(
        self,
        event_type: str,
        content: Optional[str] = None,
        item: Optional[Dict] = None,
        item_id: Optional[str] = None
    ) -> ResponsesAgentStreamEvent:
        """
        Create standardized stream events for MLflow compliance.
        
        Args:
            event_type: Type of event ("delta" or "done")
            content: Text content for delta events
            item: Item data for done events
            item_id: Unique identifier for the event
            
        Returns:
            ResponsesAgentStreamEvent object
        """
        try:
            if event_type == "delta":
                return self._create_delta_event(content or "", item_id)
            elif event_type == "done":
                return self._create_done_event(item or {})
            else:
                raise ValueError(f"Unknown event type: {event_type}")
        
        except Exception as e:
            logger.error(f"Failed to create stream event: {e}")
            # Return a fallback error event
            return self._create_error_event(str(e))
    
    def _create_delta_event(self, content: str, item_id: Optional[str] = None) -> ResponsesAgentStreamEvent:
        """Create delta event with text content."""
        self.conversion_stats["stream_events_created"] += 1
        
        return ResponsesAgentStreamEvent(
            type="response.output_text.delta",
            item_id=item_id or str(uuid4()),
            delta=content
        )
    
    def _create_done_event(self, item: Dict) -> ResponsesAgentStreamEvent:
        """Create done event with final response data."""
        self.conversion_stats["stream_events_created"] += 1
        
        return ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=item
        )
    
    def _create_error_event(self, error_message: str) -> ResponsesAgentStreamEvent:
        """Create error event for stream failures."""
        self.conversion_stats["stream_events_created"] += 1
        
        error_item = {
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text", 
                "text": f"Error: {error_message}"
            }],
            "error": True
        }
        
        return ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=error_item
        )
    
    # =========================================================================
    # Validation and Utilities
    # =========================================================================
    
    def validate_request_format(self, request: Any) -> ConversionResult:
        """
        Validate and analyze request format.
        
        Args:
            request: Request object to validate
            
        Returns:
            ConversionResult with validation details
        """
        warnings = []
        metadata = {}
        
        try:
            if hasattr(request, 'input'):
                # MLflow ResponsesAgentRequest format
                format_type = "responses_agent"
                metadata["message_count"] = len(request.input)
            elif isinstance(request, dict):
                if "messages" in request:
                    # OpenAI chat format
                    format_type = "openai_chat"
                    metadata["message_count"] = len(request["messages"])
                elif "input" in request:
                    # Dictionary-wrapped ResponsesAgent format
                    format_type = "responses_agent_dict"
                    metadata["message_count"] = len(request["input"])
                else:
                    format_type = "unknown"
                    warnings.append("Unknown dictionary format")
            else:
                format_type = "unknown"
                warnings.append(f"Unrecognized request type: {type(request)}")
            
            return ConversionResult(
                data=request,
                format=format_type,
                metadata=metadata,
                warnings=warnings
            )
            
        except Exception as e:
            warnings.append(f"Validation failed: {e}")
            return ConversionResult(
                data=request,
                format="invalid",
                metadata={},
                warnings=warnings
            )
    
    @safe_call("get_conversion_stats", fallback={})
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get conversion statistics for monitoring."""
        return {
            **self.conversion_stats,
            "success_rate": self._calculate_success_rate(),
            "total_conversions": (
                self.conversion_stats["requests_converted"] + 
                self.conversion_stats["responses_converted"]
            )
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate conversion success rate."""
        total = (
            self.conversion_stats["requests_converted"] + 
            self.conversion_stats["responses_converted"]
        )
        warnings = self.conversion_stats["warnings_generated"]
        
        if total == 0:
            return 1.0
        
        return max(0.0, 1.0 - (warnings / total))
    
    def reset_stats(self):
        """Reset conversion statistics."""
        for key in self.conversion_stats:
            self.conversion_stats[key] = 0


# Global schema converter instance for easy access
global_schema_converter = SchemaConverter()

# Convenience functions
def convert_request_to_messages(request: ResponsesAgentRequest) -> List[BaseMessage]:
    """Convenience function for request conversion."""
    return global_schema_converter.request_to_messages(request)

def create_delta_event(content: str, item_id: Optional[str] = None) -> ResponsesAgentStreamEvent:
    """Convenience function for delta event creation."""
    return global_schema_converter.create_stream_event("delta", content=content, item_id=item_id)

def create_done_event(item: Dict) -> ResponsesAgentStreamEvent:
    """Convenience function for done event creation."""
    return global_schema_converter.create_stream_event("done", item=item)

def format_standard_response(
    content: str,
    citations: Optional[List[Dict]] = None,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """Convenience function for standard response formatting."""
    return global_schema_converter.format_response(content, citations, metadata)