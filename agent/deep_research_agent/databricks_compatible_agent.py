"""
Databricks-compatible wrapper for the Research Agent.

This module provides a fully compatible ResponsesAgent implementation that follows
the exact pattern from the Databricks working example while preserving all research
functionality.
"""

import sys
import os
from pathlib import Path

# Add the model directory to Python path for imports when deployed
# This ensures deep_research_agent module is available in MLflow serving environment
model_dir = Path(__file__).parent.parent
if str(model_dir) not in sys.path:
    sys.path.insert(0, str(model_dir))

import json
from typing import Any, Dict, Generator, List, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from deep_research_agent.research_agent_refactored import RefactoredResearchAgent
from deep_research_agent.core import get_logger

logger = get_logger(__name__)


class DatabricksCompatibleAgent(ResponsesAgent):
    """
    Databricks-compatible wrapper that follows the exact pattern from the working example.
    
    This wrapper ensures full schema compliance with Databricks Agent Framework while
    maintaining all research capabilities of the underlying agent.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, yaml_path: Optional[str] = None):
        """Initialize the wrapper with the research agent."""
        try:
            self.agent = RefactoredResearchAgent(config=config, yaml_path=yaml_path)
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            # Import ConfigurationError for proper error handling
            from deep_research_agent.core.exceptions import ConfigurationError
            if "configuration file not found" in str(e).lower() or "yaml" in str(e).lower():
                raise ConfigurationError(f"Agent initialization failed - YAML config required: {e}")
            else:
                raise ConfigurationError(f"Agent initialization failed: {e}")
    
    def _responses_to_cc(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert from a Responses API output item to ChatCompletion messages."""
        msg_type = message.get("type")
        if msg_type == "function_call":
            return [
                {
                    "role": "assistant",
                    "content": "tool call",
                    "tool_calls": [
                        {
                            "id": message["call_id"],
                            "type": "function",
                            "function": {
                                "arguments": message["arguments"],
                                "name": message["name"],
                            },
                        }
                    ],
                }
            ]
        elif msg_type == "message" and isinstance(message["content"], list):
            return [
                {"role": message["role"], "content": content["text"]}
                for content in message["content"]
            ]
        elif msg_type == "reasoning":
            return [{"role": "assistant", "content": json.dumps(message["summary"])}]
        elif msg_type == "function_call_output":
            return [
                {
                    "role": "tool",
                    "content": message["output"],
                    "tool_call_id": message["call_id"],
                }
            ]
        compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
        filtered = {k: v for k, v in message.items() if k in compatible_keys}
        return [filtered] if filtered else []
    
    def _prep_msgs_for_cc_llm(self, responses_input) -> list[dict[str, Any]]:
        """Convert from Responses input items to ChatCompletion dictionaries."""
        cc_msgs = []
        for msg in responses_input:
            cc_msgs.extend(self._responses_to_cc(msg.model_dump()))
        return cc_msgs
    
    def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Convert from LangChain messages to Responses output item dictionaries."""
        responses_items = []
        for message in messages:
            if hasattr(message, 'model_dump'):
                message_dict = message.model_dump()
            else:
                # Fallback for messages without model_dump
                message_dict = {
                    "type": message.__class__.__name__.lower().replace("message", ""),
                    "content": str(message.content) if hasattr(message, 'content') else str(message)
                }
            
            role = message_dict.get("type", "")
            if role == "ai":
                if tool_calls := message_dict.get("tool_calls"):
                    for tool_call in tool_calls:
                        responses_items.append(
                            self.create_function_call_item(
                                id=message_dict.get("id") or str(uuid4()),
                                call_id=tool_call["id"],
                                name=tool_call["name"],
                                arguments=json.dumps(tool_call.get("args", {})),
                            )
                        )
                else:
                    responses_items.append(
                        self.create_text_output_item(
                            text=message_dict.get("content", ""),
                            id=message_dict.get("id") or str(uuid4()),
                        )
                    )
            elif role == "tool":
                responses_items.append(
                    self.create_function_call_output_item(
                        call_id=message_dict["tool_call_id"],
                        output=message_dict["content"],
                    )
                )
            elif role == "user" or role == "human":
                responses_items.append({
                    "role": "user",
                    "content": message_dict.get("content", "")
                })
        return responses_items
    
    def create_function_call_item(self, id: str, call_id: str, name: str, arguments: str) -> dict[str, Any]:
        """Create a function call item for responses."""
        return {
            "type": "function_call",
            "id": id,
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        }
    
    def create_function_call_output_item(self, call_id: str, output: str) -> dict[str, Any]:
        """Create a function call output item for responses."""
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }
    
    
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Process request - exact pattern from working example."""
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=getattr(request, 'custom_inputs', None) or {})
    
    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Stream agent responses with proper delta and done events.
        
        Implements streaming following Databricks Agent Framework requirements,
        ensuring compatibility with Unity Catalog model serving endpoints.
        
        Args:
            request (ResponsesAgentRequest): Input request with messages
        
        Yields:
            ResponsesAgentStreamEvent: Two event types:
                - Delta events: Text chunks with "response.output_text.delta" type
                - Done events: Complete messages with "response.output_item.done" type
        
        Critical Schema Rules:
            - NO JSON objects in delta events (plain text only)
            - Consistent item_id across related events
            - Single done event per response
            - Messages converted through: Request → CC format → LangChain → Stream
        
        See SCHEMA_REQUIREMENTS.md for complete schema documentation.
        """
        try:
            # Use the wrapped agent's predict_stream and ensure proper format
            for event in self.agent.predict_stream(request):
                # Ensure proper event format
                # Pass through all events directly since they're already properly formatted
                yield event
                    
        except Exception as e:
            logger.error(f"Error in predict_stream: {e}")
            # Return error as properly formatted output
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(f"Error processing request: {str(e)}", str(uuid4()))
            )


# For MLflow model registration
import mlflow
mlflow.langchain.autolog()

# Create default agent instance - let ConfigManager find the YAML file
# In MLflow serving contexts, the YAML may be in various locations
try:
    default_agent = DatabricksCompatibleAgent(yaml_path=None)  # Let ConfigManager find it
    mlflow.models.set_model(default_agent)
except Exception as e:
    # If that fails, try the explicit path as a fallback
    logger.warning(f"Failed to create agent without explicit path: {e}")
    try:
        default_yaml_path = str(Path(__file__).parent / "agent_config.yaml")
        default_agent = DatabricksCompatibleAgent(yaml_path=default_yaml_path)
        mlflow.models.set_model(default_agent)
    except Exception as fallback_error:
        logger.error(f"Failed to create agent even with explicit path: {fallback_error}")
        # Create a minimal default agent as last resort
        raise Exception(f"Could not initialize agent: tried auto-discovery ({e}) and explicit path ({fallback_error})")