"""
Response handling utilities for the research agent.

This module contains methods for building responses, detecting environments,
and handling various response formats.
"""

import os
from typing import Dict, Any, Optional
from uuid import uuid4

from langchain_core.messages import HumanMessage
from mlflow.types.responses import ResponsesAgentResponse

from deep_research_agent.core import get_logger
from deep_research_agent.databricks_response_builder import (
    databricks_response_builder,
    openai_response_builder,
    chat_agent_response_builder,
    adaptive_response_builder
)
from deep_research_agent.components import response_builder, message_converter

logger = get_logger(__name__)


class ResponseUtils:
    """Utilities for handling responses in various formats."""
    
    @staticmethod
    def build_error_responses_agent_response(error_message: str, error_details: Optional[Dict] = None) -> ResponsesAgentResponse:
        """Helper to build a proper ResponsesAgentResponse for errors."""
        # Note: This needs to be updated to not use create_text_output_item
        # For now, create the error output structure directly
        error_output = {
            "type": "response.output_text",
            "id": str(uuid4()),
            "content": error_message
        }
        return ResponsesAgentResponse(
            output=[error_output],
            custom_outputs={"error": error_message, **(error_details or {})}
        )
    
    @staticmethod
    def get_response_builder(is_serving_endpoint=False):
        """Get the appropriate response builder based on environment."""
        # Check for Databricks-specific response format
        response_format = os.getenv("DATABRICKS_RESPONSE_FORMAT", "auto")
        
        if response_format == "openai_chat":
            logger.info("Using OpenAI chat completion response builder")
            return openai_response_builder
        elif response_format == "chat_agent":
            logger.info("Using ChatAgent response builder")  
            return chat_agent_response_builder
        elif response_format == "adaptive":
            logger.info("Using adaptive multi-schema response builder")
            return adaptive_response_builder
        elif response_format == "standard":
            logger.info("Using standard MLflow ResponsesAgent builder")
            return response_builder
        else:
            # Auto-detect: only use adaptive builder for actual serving endpoints
            if is_serving_endpoint:
                logger.info("Detected Databricks serving endpoint - using adaptive response builder")
                return adaptive_response_builder
            else:
                # Use standard response builder for tests and development
                logger.info("Using standard response builder for backward compatibility")
                return response_builder
    
    @staticmethod
    def is_databricks_environment() -> bool:
        """Check if running in Databricks environment."""
        # Check for Databricks-specific environment variables
        databricks_indicators = [
            "DATABRICKS_RUNTIME_VERSION",
            "DATABRICKS_HOST", 
            "DATABRICKS_WORKSPACE_URL",
            "DATABRICKS_TOKEN",
            "DATABRICKS_WORKSPACE_ID",
            "DATABRICKS_CLUSTER_ID",
            "DATABRICKS_APPLICATION_ID",
            # Serving endpoint specific indicators
            "DATABRICKS_SERVING_ENDPOINT",
            "DATABRICKS_MODEL_SERVING"
        ]
        
        detected_indicators = []
        for indicator in databricks_indicators:
            if os.getenv(indicator):
                detected_indicators.append(indicator)
        
        if detected_indicators:
            logger.info(f"Databricks environment detected via: {', '.join(detected_indicators)}")
            return True
            
        # Also check for force flag
        if os.getenv("FORCE_DATABRICKS_RESPONSE_FORMAT", "").lower() in ["true", "1", "yes"]:
            logger.info("Databricks environment forced via FORCE_DATABRICKS_RESPONSE_FORMAT")
            return True
        
        # Check for Databricks file system paths
        if os.path.exists("/databricks") or os.getcwd().startswith("/Workspace"):
            logger.info("Databricks environment detected via filesystem paths")
            return True
            
        logger.debug("No Databricks environment indicators found")
        return False
    
    @staticmethod
    def is_serving_endpoint_environment() -> bool:
        """Check if running in a Databricks serving endpoint context where schema compliance is critical."""
        # Check for serving endpoint specific indicators
        serving_indicators = [
            "DATABRICKS_SERVING_ENDPOINT",
            "DATABRICKS_MODEL_SERVING",
            "DATABRICKS_ENDPOINT_NAME",
            "DATABRICKS_MODEL_NAME",
            # MLflow serving context
            "MLFLOW_SERVING_PORT",
            "SERVING_PORT"
        ]
        
        detected_serving = []
        for indicator in serving_indicators:
            if os.getenv(indicator):
                detected_serving.append(indicator)
        
        if detected_serving:
            logger.info(f"Serving endpoint context detected via: {', '.join(detected_serving)}")
            return True
            
        # Check for explicit schema compliance requirement
        schema_compliance = os.getenv("REQUIRE_DATABRICKS_SCHEMA_COMPLIANCE", "").lower()
        if schema_compliance in ["true", "1", "yes"]:
            logger.info("Schema compliance required via REQUIRE_DATABRICKS_SCHEMA_COMPLIANCE")
            return True
            
        # Check if we're in a production deployment context
        deployment_context = os.getenv("DEPLOYMENT_CONTEXT", "").lower()
        if deployment_context in ["production", "serving", "endpoint"]:
            logger.info(f"Production deployment context detected: {deployment_context}")
            return True
            
        logger.debug("No serving endpoint indicators found - using standard response format")
        return False
    
    @staticmethod
    def apply_return_format(response):
        """Apply return format based on environment variable and debugging."""
        return_format = os.getenv("AGENT_RETURN_FORMAT", "object")
        
        logger.info(
            "=== _APPLY_RETURN_FORMAT CALLED ===",
            env_var=return_format,
            response_type=type(response).__name__,
            is_dict=isinstance(response, dict),
            response_has_model_dump=hasattr(response, 'model_dump'),
            response_has_model_dump_json=hasattr(response, 'model_dump_json'),
            response_has_output='output' in response if isinstance(response, dict) else hasattr(response, 'output')
        )
        
        # If response is already a dict (from Databricks builder), validate and return it
        if isinstance(response, dict):
            # Validate the dict has required fields
            required_fields = ['output', 'messages', 'choices', 'created', 'model', 'object', 'content']
            missing_fields = [f for f in required_fields if f not in response]
            if missing_fields:
                logger.warning(f"Response dict missing required fields: {missing_fields}")
            
            # Dict is already in correct format, just return it
            logger.info("Response is already a dict, returning as-is", keys=list(response.keys()))
            return response
        
        # Add validation before returning for non-dict responses
        validation_issues = response_builder.validate_response_format(response)
        if validation_issues:
            logger.warning("Response validation issues found", issues=validation_issues)
        
        if return_format == "dict":
            logger.info("Returning response as dictionary")
            try:
                if hasattr(response, 'model_dump'):
                    result = response.model_dump()
                    logger.info("Successfully converted to dict", keys=list(result.keys()))
                    return result
                else:
                    result = response.__dict__
                    logger.info("Used __dict__ for conversion", keys=list(result.keys()))
                    return result
            except Exception as e:
                logger.error("Failed to convert to dict", error=e)
                return response
                
        elif return_format == "json":
            logger.info("Returning response as JSON string")
            try:
                if hasattr(response, 'model_dump_json'):
                    result = response.model_dump_json()
                    logger.info("Successfully converted to JSON", length=len(result))
                    return result
                else:
                    import json
                    result = json.dumps(response.__dict__)
                    logger.info("Used JSON dumps for conversion", length=len(result))
                    return result
            except Exception as e:
                logger.error("Failed to convert to JSON", error=e)
                return response
        
        else:
            logger.info("Returning response as object (default)")
            return response
    
    @staticmethod
    def handle_alternative_request_formats(request):
        """Handle alternative request formats when standard conversion fails."""
        messages = []
        
        # Case 1: Request has input field with ChatMessage-like objects
        if hasattr(request, 'input') and request.input:
            for item in request.input:
                try:
                    # Try to extract content from different formats
                    if hasattr(item, 'content'):
                        content = item.content
                    elif hasattr(item, 'message'):
                        content = item.message
                    elif isinstance(item, dict) and 'content' in item:
                        content = item['content']
                    elif isinstance(item, dict) and 'message' in item:
                        content = item['message']
                    elif isinstance(item, str):
                        content = item
                    else:
                        content = str(item)
                    
                    # Extract text content if it's complex
                    if isinstance(content, list):
                        content = ' '.join([str(c) for c in content])
                    elif isinstance(content, dict):
                        content = content.get('text', str(content))
                    
                    messages.append(HumanMessage(content=str(content)))
                    logger.info("Converted alternative format item", content_preview=str(content)[:100])
                    
                except Exception as item_error:
                    logger.warning("Failed to convert item", item=str(item)[:100], error=item_error)
                    continue
        
        # Case 2: Direct string input
        elif isinstance(request, str):
            messages.append(HumanMessage(content=request))
        
        # Case 3: Dict with messages or input field
        elif isinstance(request, dict):
            input_data = request.get('input', request.get('messages', []))
            if input_data:
                for item in input_data:
                    content = item if isinstance(item, str) else str(item)
                    messages.append(HumanMessage(content=content))
        
        if not messages:
            # Final fallback - create a generic message
            messages.append(HumanMessage(content="Please provide information on this topic."))
            logger.warning("Used fallback generic message")
        
        return messages
    
    @staticmethod
    def extract_text_content(content) -> str:
        """
        Extract text content from various formats.
        
        This method provides backward compatibility with existing tests and
        matches the behavior expected by the test suite.
        
        Args:
            content: Content in various formats (string, list, dict)
            
        Returns:
            Extracted text content as string
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
            return ''.join(text_parts)
        return str(content)  # Fallback