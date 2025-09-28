#!/usr/bin/env python3
"""
Databricks-compliant response builder.

This builds responses that match the EXACT schema requirements from Databricks serving endpoints,
based on the retrieved OpenAPI schema.

IMPORTANT SCHEMA COMPLIANCE:
- All responses must follow MLflow ResponsesAgent or OpenAI ChatCompletion formats
- Delta events contain ONLY plain text, never JSON objects
- Done events have structured message format with content arrays
- See SCHEMA_REQUIREMENTS.md for complete specifications
"""

import json
import time
from typing import Dict, List, Any, Optional, Union
from uuid import uuid4

from mlflow.types.responses import ResponsesAgentResponse
from .core import (
    get_logger,
    ResearchContext,
    WorkflowMetrics,
    Citation,
    SearchResult,
    format_duration
)

logger = get_logger(__name__)


class DatabricksResponseBuilder:
    """Builds responses that comply with Databricks serving endpoint schemas."""
    
    def __init__(self, schema_format: str = "auto"):
        """
        Initialize response builder.
        
        Args:
            schema_format: Format to use - "auto", "openai_chat", "responses_agent", or "simple"
        """
        self.schema_format = schema_format
        
    def build_openai_chat_completion(
        self,
        content: str,
        research_context: Optional[ResearchContext] = None,
        metrics: Optional[WorkflowMetrics] = None
    ) -> Dict[str, Any]:
        """
        Build OpenAI-compatible chat completion response.
        
        This format matches the "chat.completion" schema that Databricks expects.
        
        Schema Structure:
        - id: Unique completion ID (format: "chatcmpl-{hex}")
        - object: Must be "chat.completion"
        - created: Unix timestamp (not "created_at")
        - model: Model identifier
        - choices: Array with message objects containing role and content
        - usage: Token usage statistics
        
        See SCHEMA_REQUIREMENTS.md for complete schema documentation.
        """
        response = {
            "id": f"chatcmpl-{uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),  # Note: "created", not "created_at"
            "model": "databricks-research-agent",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(content.split()),
                "total_tokens": len(content.split())
            }
        }
        
        # Add custom research metadata if available
        if research_context or metrics:
            # Store research metadata in a way that doesn't interfere with validation
            response["metadata"] = self._build_research_metadata(research_context, metrics)
        
        logger.info(
            "Built OpenAI chat completion response",
            content_length=len(content),
            has_choices=True,
            choices_count=len(response["choices"])
        )
        
        return response
    
    def build_chat_agent_response(
        self,
        content: str,
        research_context: Optional[ResearchContext] = None
    ) -> Dict[str, Any]:
        """
        Build ChatAgent-compatible response.
        
        This format matches the "ChatAgentResponseSchema" that Databricks validates.
        
        Schema Structure:
        - messages: Array of message objects with role and content
        - metadata: Optional research metadata
        
        Critical Requirements:
        - Content must be a string, not structured data
        - Role must be "assistant" for responses
        
        See SCHEMA_REQUIREMENTS.md for complete schema documentation.
        """
        response = {
            "messages": [
                {
                    "role": "assistant",
                    "content": content,
                    "id": str(uuid4())
                }
            ],
            "model": "databricks-research-agent",
            "created": int(time.time()),
            "object": "chat.agent"
        }
        
        if research_context:
            response["context"] = {
                "citations": [self._citation_to_dict(c) for c in research_context.citations],
                "research_loops": research_context.research_loops,
                "sources_count": len(research_context.web_results) + len(research_context.vector_results)
            }
        
        logger.info(
            "Built ChatAgent response",
            content_length=len(content),
            has_messages=True,
            messages_count=len(response["messages"])
        )
        
        return response
    
    def build_responses_agent_response(
        self,
        content: str,
        research_context: Optional[ResearchContext] = None,
        metrics: Optional[WorkflowMetrics] = None
    ) -> Dict[str, Any]:
        """
        Build standard ResponsesAgent format, but with field name fixes.
        
        This addresses the field name mismatches (created_at vs created).
        """
        # Create the ResponsesAgentResponse
        response_obj = ResponsesAgentResponse(
            output=[{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
                "id": str(uuid4())
            }],
            model="databricks-research-agent",
            object="responses.agent",
            created_at=int(time.time()),  # This will be renamed
            id=str(uuid4()),
            status="completed"
        )
        
        # Convert to dict and fix field names
        response_dict = response_obj.model_dump()
        
        # Fix field name mismatches
        if "created_at" in response_dict:
            response_dict["created"] = response_dict.pop("created_at")
        
        # Add research metadata
        if research_context or metrics:
            response_dict["custom_outputs"] = self._build_custom_outputs(research_context, metrics)
        
        logger.info(
            "Built ResponsesAgent format with field fixes",
            content_length=len(content),
            has_output=True,
            output_count=len(response_dict.get("output", []))
        )
        
        return response_dict
    
    def build_simple_output(
        self,
        content: str,
        research_context: Optional[ResearchContext] = None
    ) -> Dict[str, Any]:
        """
        Build simplified output format.
        
        Returns just the essential fields that might match multiple schemas.
        """
        response = {
            "output": content,
            "model": "databricks-research-agent",
            "created": int(time.time()),
            "status": "completed"
        }
        
        if research_context and research_context.citations:
            response["sources"] = [self._citation_to_dict(c) for c in research_context.citations]
        
        logger.info("Built simple output format", content_length=len(content))
        return response
    
    def build_adaptive_response(
        self,
        content: str,
        research_context: Optional[ResearchContext] = None,
        metrics: Optional[WorkflowMetrics] = None,
        schema_hints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Build response that tries to satisfy multiple schema requirements.
        
        This creates a response with fields from multiple schemas to maximize
        the chance of passing validation.
        """
        base_id = str(uuid4())
        created_time = int(time.time())
        
        # Start with OpenAI chat completion structure (most common)
        response = {
            # OpenAI chat completion fields - required for chat.completion schema
            "id": f"chatcmpl-{uuid4().hex[:8]}",
            "object": "chat.completion", 
            "created": created_time,  # Note: "created" not "created_at"
            "model": "databricks-research-agent",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }
            ],
            
            # ChatAgent fields - required for ChatAgentResponseSchema
            "messages": [
                {
                    "role": "assistant",  # Required for ChatAgentChunkSchema
                    "content": content,
                    "id": base_id
                }
            ],
            
            # ResponsesAgent fields - required for ResponsesResponseSchema
            "output": [
                {
                    "type": "message",
                    "role": "assistant", 
                    "content": [{"type": "output_text", "text": content}],
                    "id": base_id
                }
            ],
            
            # Additional compatibility fields
            "status": "completed",
            "content": content,  # For strOutputSchema compatibility - required field
            "usage": {
                "input_tokens": 0,
                "input_tokens_details": {
                    "cached_tokens": 0
                },
                "output_tokens": len(content.split()),
                "output_tokens_details": {
                    "reasoning_tokens": 0
                },
                "total_tokens": len(content.split())
            }
        }
        
        # Add research metadata
        if research_context or metrics:
            response["custom_outputs"] = self._build_custom_outputs(research_context, metrics)
            response["metadata"] = self._build_research_metadata(research_context, metrics)
        
        logger.info(
            "Built adaptive multi-schema response",
            content_length=len(content),
            has_choices=True,
            has_messages=True,
            has_output=True
        )
        
        return response
    
    def _build_research_metadata(
        self,
        research_context: Optional[ResearchContext],
        metrics: Optional[WorkflowMetrics]
    ) -> Dict[str, Any]:
        """Build research metadata for inclusion in responses."""
        metadata = {}
        
        if research_context:
            metadata.update({
                "research_loops": str(research_context.research_loops),
                "citations_count": str(len(research_context.citations)),
                "web_results_count": str(len(research_context.web_results)),
                "vector_results_count": str(len(research_context.vector_results))
            })
            
            if research_context.citations:
                # Convert citations list to JSON string for ResponsesAgentResponse compatibility
                citations_list = [self._citation_to_dict(c) for c in research_context.citations]
                metadata["sources"] = json.dumps(citations_list)
        
        if metrics:
            metadata.update({
                "execution_time": format_duration(metrics.execution_time_seconds),
                "queries_generated": str(metrics.total_queries_generated),
                "success_rate": str(metrics.success_rate)
            })
        
        return metadata
    
    def _build_custom_outputs(
        self,
        research_context: Optional[ResearchContext],
        metrics: Optional[WorkflowMetrics]
    ) -> Dict[str, Any]:
        """Build custom outputs for ResponsesAgent format."""
        custom_outputs = {}
        
        if research_context:
            custom_outputs.update({
                "citations": [self._citation_to_dict(c) for c in research_context.citations],
                "research_metadata": {
                    "original_question": research_context.original_question,
                    "research_loops": research_context.research_loops,
                    "queries_generated": len(research_context.generated_queries),
                    "web_results_count": len(research_context.web_results),
                    "reflection": research_context.reflection
                }
            })
        
        if metrics:
            custom_outputs["metrics"] = {
                "execution_time": format_duration(metrics.execution_time_seconds),
                "total_queries": metrics.total_queries_generated,
                "success_rate": metrics.success_rate,
                "error_count": metrics.error_count
            }
        
        return custom_outputs
    
    def _citation_to_dict(self, citation: Citation) -> Dict[str, Any]:
        """Convert Citation to dictionary."""
        return {
            "source": citation.source,
            "url": citation.url,
            "title": citation.title,
            "snippet": citation.snippet
        }
    
    def build_error_response(
        self,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        format_type: str = "adaptive"
    ) -> Dict[str, Any]:
        """Build error response in specified format."""
        if format_type == "openai_chat":
            return {
                "id": f"chatcmpl-error-{uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "databricks-research-agent",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": error_message
                        },
                        "finish_reason": "error"
                    }
                ],
                "error": error_details or {"message": error_message}
            }
        elif format_type == "chat_agent":
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": error_message,
                        "id": str(uuid4())
                    }
                ],
                "model": "databricks-research-agent",
                "created": int(time.time()),
                "error": error_details or {"message": error_message}
            }
        else:
            # Adaptive format with multiple schema compatibility
            error_response = self.build_adaptive_response(
                content=error_message,
                research_context=None,
                metrics=None
            )
            # Add error details to the response for better debugging
            if error_details:
                error_response["error"] = error_details
            else:
                error_response["error"] = {"message": error_message}
            
            # Ensure error responses also have status indicating failure
            error_response["status"] = "error"
            
            return error_response

    # ===============================
    # Event Emission Helper Functions
    # ===============================
    
    def emit_plan_structure_event(
        self, 
        plan_id: str, 
        plan_name: str, 
        plan_description: str, 
        steps: List[Dict[str, Any]],
        complexity: str = "moderate",
        estimated_time: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a plan_structure intermediate event that matches the UI's expected format.
        
        Args:
            plan_id: Unique identifier for the plan
            plan_name: Human-readable name for the plan
            plan_description: Description of what the plan will accomplish
            steps: List of step dictionaries with step_id, name, description, status
            complexity: Plan complexity ("simple", "moderate", "complex")
            estimated_time: Estimated execution time in minutes
            
        Returns:
            Dict formatted as MLflow ResponsesAgentStreamEvent intermediate_event
        """
        event_data = {
            "type": "intermediate_event",
            "event_type": "plan_structure", 
            "data": {
                "plan_id": plan_id,
                "plan": {
                    "name": plan_name,
                    "description": plan_description,
                    "complexity": complexity,
                    "estimated_time": estimated_time or len(steps) * 5,  # 5 minutes per step default
                    "steps": [
                        {
                            "step_id": step.get("step_id", f"step_{i+1:03d}"),
                            "name": step.get("name", f"Step {i+1}"),
                            "description": step.get("description", ""),
                            "status": step.get("status", "pending"),
                            "type": step.get("type", "research"),
                            "estimated_time": step.get("estimated_time", 5)
                        }
                        for i, step in enumerate(steps)
                    ]
                }
            }
        }
        
        logger.info(
            "Created plan_structure event",
            plan_id=plan_id,
            steps_count=len(steps),
            complexity=complexity
        )
        
        return event_data

    def emit_step_activated_event(
        self, 
        step_id: str, 
        step_index: int, 
        step_name: str,
        step_type: str = "research"
    ) -> Dict[str, Any]:
        """
        Create a step_activated intermediate event.
        
        Args:
            step_id: Unique identifier for the step
            step_index: Zero-based index of the step in the plan
            step_name: Human-readable name for the step
            step_type: Type of step (research, analysis, etc.)
            
        Returns:
            Dict formatted as MLflow ResponsesAgentStreamEvent intermediate_event
        """
        event_data = {
            "type": "intermediate_event", 
            "event_type": "step_activated",
            "data": {
                "step_id": step_id,
                "step_index": step_index,
                "step_name": step_name,
                "step_type": step_type,
                "status": "active",
                "timestamp": int(time.time())
            }
        }
        
        logger.info(
            "Created step_activated event", 
            step_id=step_id,
            step_index=step_index,
            step_name=step_name
        )
        
        return event_data

    def emit_step_completed_event(
        self, 
        step_id: str, 
        step_index: int, 
        step_name: str,
        step_type: str = "research",
        success: bool = True,
        summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a step_completed intermediate event.
        
        Args:
            step_id: Unique identifier for the step
            step_index: Zero-based index of the step in the plan
            step_name: Human-readable name for the step
            step_type: Type of step (research, analysis, etc.)
            success: Whether the step completed successfully
            summary: Optional summary of what was accomplished
            
        Returns:
            Dict formatted as MLflow ResponsesAgentStreamEvent intermediate_event
        """
        event_data = {
            "type": "intermediate_event",
            "event_type": "step_completed", 
            "data": {
                "step_id": step_id,
                "step_index": step_index,
                "step_name": step_name,
                "step_type": step_type,
                "status": "completed" if success else "failed",
                "success": success,
                "summary": summary,
                "timestamp": int(time.time())
            }
        }
        
        logger.info(
            "Created step_completed event",
            step_id=step_id, 
            step_index=step_index,
            step_name=step_name,
            success=success
        )
        
        return event_data


# Create instances for different formats
openai_response_builder = DatabricksResponseBuilder("openai_chat")
chat_agent_response_builder = DatabricksResponseBuilder("chat_agent")
adaptive_response_builder = DatabricksResponseBuilder("adaptive")

# Default builder (adaptive)
databricks_response_builder = adaptive_response_builder