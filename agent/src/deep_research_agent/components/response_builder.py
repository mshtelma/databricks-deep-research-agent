"""
Response building utilities for the research agent.

This module provides utilities for building responses in various formats
including ResponsesAgent format, streaming responses, and citations.
"""

import json
import time
from typing import Dict, List, Any, Optional, Generator
from uuid import uuid4

from mlflow.types.responses import (
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent
)

from ..core import (
    get_logger,
    SearchResult,
    Citation,
    ResearchContext,
    WorkflowMetrics,
    safe_json_dumps,
    format_duration
)

logger = get_logger(__name__)


class ResponseBuilder:
    """Builds responses in various formats for the research agent."""
    
    def __init__(self):
        """Initialize response builder."""
        pass
    
    # Note: create_text_output_item and create_text_delta methods removed
    # These should be inherited from ResponsesAgent in classes that need them
    
    def create_function_call_item(
        self, 
        call_id: str, 
        name: str, 
        arguments: str,
        item_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a function call item for ResponsesAgent.
        
        Args:
            call_id: Function call ID
            name: Function name
            arguments: Function arguments as JSON string
            item_id: Optional item ID
            
        Returns:
            Function call item dictionary
        """
        return {
            "type": "function_call",
            "id": item_id or str(uuid4()),
            "call_id": call_id,
            "name": name,
            "arguments": arguments
        }
    
    def create_function_call_output_item(
        self, 
        call_id: str, 
        output: str,
        item_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a function call output item.
        
        Args:
            call_id: Function call ID
            output: Function output
            item_id: Optional item ID
            
        Returns:
            Function call output item dictionary
        """
        return {
            "type": "function_call_output",
            "id": item_id or str(uuid4()),
            "call_id": call_id,
            "output": output
        }
    
    def build_research_response(
        self, 
        final_answer: str,
        research_context: ResearchContext,
        metrics: Optional[WorkflowMetrics] = None
    ) -> ResponsesAgentResponse:
        """
        Build a complete research response that matches expected schemas.
        
        Args:
            final_answer: Final synthesized answer
            research_context: Research context with citations and metadata
            metrics: Optional workflow metrics
            
        Returns:
            ResponsesAgentResponse object
        """
        try:
            # Apply final table cleanup to the answer - LAST LINE OF DEFENSE
            import re
            clean_final_answer = re.sub(r'\|\|+', '|', final_answer)
            
            # Log if we caught any double pipes here
            double_pipes_caught = final_answer.count('||') - clean_final_answer.count('||')
            if double_pipes_caught > 0:
                logger.warning(f"Response builder caught {double_pipes_caught} double pipes in final answer")
            
            # Create main response item using correct format for ResponsesAgentResponse
            response_item = {
                "type": "message",
                "role": "assistant",
                "content": [{"text": clean_final_answer, "type": "output_text"}],
                "id": str(uuid4())
            }
            
            # Build custom outputs with research metadata
            custom_outputs = self._build_custom_outputs(research_context, metrics)
            
            # Create response using native ResponsesAgentResponse fields
            response = ResponsesAgentResponse(
                output=[response_item],
                custom_outputs=custom_outputs,
                model="databricks-research-agent",
                object="responses.agent",
                created_at=int(time.time()),  # Note: MLflow ResponsesAgentResponse expects created_at
                id=str(uuid4()),
                status="completed"
            )
            
            logger.info(
                "Built research response",
                answer_length=len(final_answer),
                citations_count=len(research_context.citations),
                research_loops=research_context.research_loops
            )
            
            return response
        except Exception as e:
            logger.error("Failed to build research response", error=e)
            # Return error response
            return self.build_error_response(f"I encountered an error while preparing the response: {str(e)}")
    
    def _build_custom_outputs(
        self, 
        research_context: ResearchContext,
        metrics: Optional[WorkflowMetrics] = None
    ) -> Dict[str, Any]:
        """
        Build custom outputs dictionary with research metadata.
        
        Args:
            research_context: Research context
            metrics: Optional workflow metrics
            
        Returns:
            Custom outputs dictionary
        """
        custom_outputs = {}
        
        if research_context:
            custom_outputs.update({
                "citations": [self._citation_to_dict(citation) for citation in research_context.citations],
                "research_metadata": {
                    "original_question": research_context.original_question,
                    "research_loops": research_context.research_loops,
                    "max_loops": research_context.max_loops,
                    "queries_generated": len(research_context.generated_queries),
                    "web_results_count": len(research_context.web_results),
                    "vector_results_count": len(research_context.vector_results),
                    "reflection": research_context.reflection
                }
            })
        
        # Add metrics if available
        if metrics:
            custom_outputs["workflow_metrics"] = {
                "total_queries_generated": metrics.total_queries_generated,
                "total_web_results": metrics.total_web_results,
                "total_vector_results": metrics.total_vector_results,
                "total_research_loops": metrics.total_research_loops,
                "execution_time": format_duration(metrics.execution_time_seconds),
                "error_count": metrics.error_count,
                "success_rate": metrics.success_rate
            }
        
        # Add search result summaries
        if research_context and research_context.web_results:
            custom_outputs["web_sources"] = [
                self._search_result_to_dict(result) for result in research_context.web_results[:5]
            ]
        
        if research_context and research_context.vector_results:
            custom_outputs["internal_sources"] = [
                self._search_result_to_dict(result) for result in research_context.vector_results[:3]
            ]
        
        return custom_outputs
    
    def _citation_to_dict(self, citation: Citation) -> Dict[str, Any]:
        """Convert Citation object to dictionary."""
        return {
            "source": citation.source,
            "url": citation.url,
            "title": citation.title,
            "snippet": citation.snippet
        }
    
    def _search_result_to_dict(self, result: SearchResult) -> Dict[str, Any]:
        """Convert SearchResult object to dictionary."""
        return {
            "source": result.source,
            "url": result.url,
            "title": result.title,
            "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content,
            "score": result.score,
            "result_type": result.result_type.value
        }
    
    def build_streaming_response(
        self, 
        content_generator: Generator[str, None, None],
        research_context: Optional[ResearchContext] = None
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Build streaming response from content generator.
        
        Args:
            content_generator: Generator yielding content strings
            research_context: Optional research context for final metadata
            
        Yields:
            ResponsesAgentStreamEvent objects
        """
        try:
            item_id = str(uuid4())
            
            # Stream content deltas
            for content_delta in content_generator:
                if content_delta:
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.delta",
                        item={
                            "type": "response.output_text.delta",
                            "item_id": item_id,
                            "delta": content_delta
                        }
                    )
            
            # Send final done event with metadata if available
            if research_context:
                custom_outputs = self._build_custom_outputs(research_context)
                final_item = {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"text": "", "type": "output_text"}],
                    "id": item_id
                }
                final_item["custom_outputs"] = custom_outputs
                
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=final_item
                )
        except Exception as e:
            logger.error("Failed to build streaming response", error=e)
            # Send error event
            error_item = {
                "type": "message",
                "role": "assistant",
                "content": [{"text": f"Streaming error: {str(e)}", "type": "output_text"}],
                "id": str(uuid4())
            }
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=error_item
            )
    
    def build_progress_event(self, step_name: str, details: str) -> ResponsesAgentStreamEvent:
        """
        Build a progress event for streaming.
        
        Args:
            step_name: Name of the current step
            details: Details about the step
            
        Returns:
            ResponsesAgentStreamEvent for progress
        """
        progress_text = f"[{step_name}] {details}\n"
        return ResponsesAgentStreamEvent(
            type="response.output_item.delta",
            item={
                "type": "response.output_text.delta",
                "item_id": str(uuid4()),
                "delta": progress_text
            }
        )
    
    def build_error_response(self, error_message: str, error_details: Optional[Dict[str, Any]] = None) -> ResponsesAgentResponse:
        """
        Build an error response using native ResponsesAgentResponse fields.
        
        Args:
            error_message: Error message for the user
            error_details: Optional error details for debugging
            
        Returns:
            ResponsesAgentResponse with error information
        """
        try:
            error_item = {
                "type": "message",
                "role": "assistant",
                "content": [{"text": error_message, "type": "output_text"}],
                "id": str(uuid4())
            }
            
            custom_outputs = {
                "error": True,
                "error_message": error_message
            }
            
            if error_details:
                custom_outputs["error_details"] = error_details
            
            # Create error response using native ResponsesAgentResponse fields
            response = ResponsesAgentResponse(
                output=[error_item],
                custom_outputs=custom_outputs,
                model="databricks-research-agent",
                object="responses.agent",
                created_at=int(time.time()),
                id=str(uuid4()),
                status="failed"
            )
            
            logger.error("Built error response", error_message=error_message)
            return response
        except Exception as e:
            logger.error("Failed to build error response", error=e)
            # Final fallback - create minimal response
            error_text = "An unexpected error occurred while building response."
            minimal_item = {
                "type": "message",
                "role": "assistant",
                "content": [{"text": error_text, "type": "output_text"}],
                "id": str(uuid4())
            }
            return ResponsesAgentResponse(
                output=[minimal_item],
                model="databricks-research-agent",
                object="responses.agent",
                created_at=int(time.time()),
                id=str(uuid4()),
                status="failed"
            )
    
    def format_citations_text(self, citations: List[Citation]) -> str:
        """
        Format citations as text for inclusion in responses.
        
        Args:
            citations: List of citations
            
        Returns:
            Formatted citations text
        """
        if not citations:
            return ""
        
        citation_lines = []
        for i, citation in enumerate(citations, 1):
            if citation.url:
                citation_line = f"{i}. {citation.source} - {citation.url}"
            else:
                citation_line = f"{i}. {citation.source}"
            
            if citation.title:
                citation_line += f" ({citation.title})"
            
            citation_lines.append(citation_line)
        
        return "\n\nSources:\n" + "\n".join(citation_lines)
    
    def build_synthesis_with_citations(
        self, 
        main_content: str, 
        citations: List[Citation],
        include_citations_in_text: bool = True
    ) -> str:
        """
        Build synthesized response with citations.
        
        Args:
            main_content: Main response content
            citations: List of citations
            include_citations_in_text: Whether to append citations to text
            
        Returns:
            Complete response with citations
        """
        if not include_citations_in_text or not citations:
            return main_content
        
        citations_text = self.format_citations_text(citations)
        return main_content + citations_text
    
    def validate_response_format(self, response: ResponsesAgentResponse) -> List[str]:
        """
        Validate response format and return list of issues.
        
        Args:
            response: Response to validate
            
        Returns:
            List of validation issues
        """
        issues = []
        
        if not response.output:
            issues.append("Response has no output items")
        
        for i, item in enumerate(response.output):
            if not isinstance(item, dict):
                issues.append(f"Output item {i} is not a dictionary")
                continue
            
            if "type" not in item:
                issues.append(f"Output item {i} missing 'type' field")
            
            if "id" not in item:
                issues.append(f"Output item {i} missing 'id' field")
            
            item_type = item.get("type")
            if item_type == "message":
                if "content" not in item:
                    issues.append(f"Message item {i} missing 'content' field")
            elif item_type == "function_call":
                required_fields = ["call_id", "name", "arguments"]
                for field in required_fields:
                    if field not in item:
                        issues.append(f"Function call item {i} missing '{field}' field")
        
        return issues


# Create singleton instance
response_builder = ResponseBuilder()