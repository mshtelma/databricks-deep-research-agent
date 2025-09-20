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
import threading
from typing import Any, Dict, Generator, List, Optional
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from deep_research_agent.enhanced_research_agent import EnhancedResearchAgent
from deep_research_agent.research_agent_refactored import (
    RefactoredResearchAgentDeprecated as RefactoredResearchAgent,
)
from deep_research_agent.core.report_styles import ReportStyle
from deep_research_agent.core.grounding import VerificationLevel
from deep_research_agent.core import get_logger
from deep_research_agent.constants import AGENT_CONFIG_FILENAME, CONFIG_SEARCH_ORDER
from deep_research_agent.core.error_handling import handle_agent_error, create_error_response

logger = get_logger(__name__)


class DatabricksCompatibleAgent(ResponsesAgent):
    """
    Databricks-compatible wrapper that follows the exact pattern from the working example.
    
    This wrapper ensures full schema compliance with Databricks Agent Framework while
    maintaining all research capabilities of the underlying agent.
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None, 
        yaml_path: Optional[str] = None,
        config_override: Optional[Dict[str, Any]] = None
    ):
        """Initialize with enhanced multi-agent system.
        
        Args:
            config: Legacy config parameter (deprecated)
            yaml_path: Path to agent configuration file
            config_override: Configuration override for testing (replaces TEST_MODE)
        """
        try:
            # Use centralized config file discovery
            if not yaml_path:
                # Search for config files in order of preference
                if os.path.exists("/model/code"):
                    # MLflow deployment - use explicit paths to avoid path construction issues
                    search_paths = [
                        Path("/model/code/conf"),  # Primary location for conf/ directory
                        Path("/model/code"),  # Fallback in code root
                        Path("/model/code/deep_research_agent"),  # Legacy location in package
                    ]
                else:
                    # Local development - use relative paths
                    search_paths = [
                        Path(__file__).parent.parent / "conf",  # conf/ directory
                        Path(__file__).parent,  # Same directory as this file
                        Path(__file__).parent.parent,  # Parent directory (agent/)
                        Path.cwd() / "deep_research_agent",  # Current working directory
                        Path.cwd(),  # Current working directory
                    ]
                
                for config_dir in search_paths:
                    for config_name in CONFIG_SEARCH_ORDER:
                        config_path = config_dir / config_name
                        # Skip backup files
                        if ".backup" in config_name:
                            continue
                        if config_path.exists():
                            yaml_path = str(config_path)
                            logger.info(f"Found config file: {config_name} at {config_path}")
                            break
                    if yaml_path:
                        break
                
                if not yaml_path:
                    # Final fallback - use the default config from the package directory
                    default_path = Path(__file__).parent / AGENT_CONFIG_FILENAME
                    if default_path.exists():
                        yaml_path = str(default_path)
                        logger.info(f"Using package default config: {default_path}")
                    else:
                        # CRITICAL ERROR: No config found anywhere
                        # Detect runtime environment for better error message
                        env_context = "unknown"
                        if os.path.exists("/model"):
                            env_context = "MLflow deployment"
                        elif os.getenv("PYTEST_CURRENT_TEST"):
                            env_context = "test environment"
                        elif os.path.exists("deep_research_agent") and os.path.exists("tests"):
                            env_context = "development"
                        
                        # Prepare detailed search information
                        search_locations = []
                        for config_dir in search_paths:
                            # Debug: log what path values we're getting
                            logger.debug(f"search_path: {config_dir}, __file__: {__file__}, parent: {Path(__file__).parent}, parent.parent: {Path(__file__).parent.parent}")
                            for config_name in CONFIG_SEARCH_ORDER:
                                if ".backup" not in config_name:  # Skip backup files
                                    search_locations.append(str(config_dir / config_name))
                        
                        error_msg = (
                            f"CRITICAL: Agent configuration not found in {env_context}!\n"
                            f"\nThe agent requires a configuration file but none were found in these locations:\n"
                            + "\n".join(f"  - {loc}" for loc in search_locations[:8])  # Show first 8 locations
                            + (f"\n  ... and {len(search_locations) - 8} more locations" if len(search_locations) > 8 else "")
                            + f"\n\nTo fix this:\n"
                            f"  1. For local development: Create conf/base.yaml in your project root\n"
                            f"  2. For MLflow deployment: Ensure conf/ directory is included in model packaging\n"
                            f"  3. For legacy systems: Use agent_config.yaml in the agent directory\n"
                            f"\nUse CONFIG_SEARCH_ORDER in constants.py to see the complete search order."
                        )
                        logger.error(error_msg)
                        from deep_research_agent.core.exceptions import ConfigurationError
                        raise ConfigurationError(error_msg)
            
            # Always use EnhancedResearchAgent - no TEST_MODE branching
            if config_override:
                # Use config override for testing
                self.agent = EnhancedResearchAgent(
                    config_override=config_override,
                    enable_grounding=True,
                    enable_background_investigation=True,
                    default_report_style=ReportStyle.PROFESSIONAL,
                    verification_level=VerificationLevel.MODERATE,
                    stream_emitter=None  # Will be set in predict_stream if needed
                )
                logger.info("Initialized DatabricksCompatibleAgent with config override")
            else:
                # Use config file for production
                self.agent = EnhancedResearchAgent(
                    config_path=yaml_path,
                    enable_grounding=True,
                    enable_background_investigation=True,
                    default_report_style=ReportStyle.PROFESSIONAL,
                    verification_level=VerificationLevel.MODERATE,
                    stream_emitter=None  # Will be set in predict_stream if needed
                )
                logger.info(
                    f"Initialized DatabricksCompatibleAgent with EnhancedResearchAgent using {yaml_path}"
                )
            
            # Add threading lock for concurrent request safety
            self._predict_lock = threading.Lock()
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced agent: {e}")
            # Import ConfigurationError for proper error handling
            from deep_research_agent.core.exceptions import ConfigurationError
            
            # Provide helpful error message based on the error type
            if "configuration file not found" in str(e).lower() or "yaml" in str(e).lower():
                # Detect runtime environment for better error message
                env_context = "unknown"
                if os.path.exists("/model"):
                    env_context = "MLflow deployment"
                elif os.getenv("PYTEST_CURRENT_TEST"):
                    env_context = "test environment"
                elif os.path.exists("deep_research_agent") and os.path.exists("tests"):
                    env_context = "development"
                
                error_msg = (
                    f"CRITICAL: Agent configuration not found in {env_context}!\n"
                    f"Error: {str(e)}\n"
                    f"\nThe agent requires a configuration file. Please ensure one of these exists:\n"
                    f"  - /model/code/conf/base.yaml (for MLflow deployment)\n"
                    f"  - conf/base.yaml (for local development)\n"
                    f"  - agent_config.yaml (legacy format)\n"
                    f"\nTo fix this in deployment:\n"
                    f"  1. Ensure conf/base.yaml exists locally\n"
                    f"  2. Re-deploy with updated model packaging (include conf/ directory)\n"
                    f"\nSearched in: {', '.join([str(p) for p in search_paths[:3]])}..."
                )
                logger.error(error_msg)
                raise ConfigurationError(error_msg)
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
        """Process request by aggregating only done items from streaming events."""
        # Serialize concurrent requests to avoid thread safety issues
        with self._predict_lock:
            try:
                outputs = [
                event.item
                for event in self.predict_stream(request)
                if event.type == "response.output_item.done"
                ]
                return ResponsesAgentResponse(
                    output=outputs, custom_outputs=getattr(request, "custom_inputs", None) or {}
                )
            except Exception as e:
                # Handle errors and return structured error response
                agent_error = handle_agent_error(
                    exception=e,
                    agent_name="DatabricksCompatibleAgent",
                operation="predict",
                request_id=str(uuid4()),
                    context={"request_type": "sync"}
                )
                
                # Return error as properly formatted response
                error_response = self.create_text_output_item(
                    text=agent_error.user_message,
                    id=str(uuid4())
                )
                
                return ResponsesAgentResponse(
                    output=[error_response],
                    custom_outputs={
                        "error": agent_error.to_dict(),
                        "success": False
                    }
                )
    
    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Stream responses with full intermediate events from multi-agent system.
        
        Delegates to EnhancedResearchAgent which provides complete multi-agent
        workflow with event emission from all agents (Coordinator, Planner, 
        Researcher, Fact Checker, Reporter).
        """
        try:
            logger.info("DatabricksCompatibleAgent.predict_stream delegating to EnhancedResearchAgent")
            for event in self.agent.predict_stream(request):
                yield event
        except Exception as e:
            # Handle errors with proper error wrapping
            agent_error = handle_agent_error(
                exception=e,
                agent_name="DatabricksCompatibleAgent", 
                operation="predict_stream",
                request_id=str(uuid4()),
                context={"request_type": "streaming"}
            )
            
            # Return structured error as properly formatted output
            error_item = self.create_text_output_item(
                text=agent_error.user_message,
                id=str(uuid4())
            )
            
            # Add error metadata to the item
            error_item["error_details"] = {
                "error_code": agent_error.error_code,
                "retry_possible": agent_error.retry_possible,
                "suggested_action": agent_error.suggested_action,
                "error_type": agent_error.error_type.value
            }
            
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=error_item
            )

    def create_text_delta(self, text_chunk: str, item_id: str) -> dict[str, Any]:
        """Create a delta event dictionary compliant with MLflow ResponsesAgent schema."""
        return {
            "type": "response.output_text.delta",
            "item_id": item_id,
            "delta": text_chunk,
        }

    def create_text_output_item(self, text: str, id: str) -> dict[str, Any]:
        """Create a completed message item dictionary for done events."""
        return {
            "type": "message",
            "role": "assistant", 
            "content": [{"type": "output_text", "text": text}],
            "id": id,
        }
    
    # Property accessors to expose internal agent attributes for backward compatibility
    @property
    def llm(self):
        """Access to the underlying LLM."""
        return getattr(self.agent, 'llm', None)
    
    @llm.setter
    def llm(self, value):
        if hasattr(self.agent, 'llm'):
            setattr(self.agent, 'llm', value)
        else:
            self.agent.llm = value
    
    @llm.deleter
    def llm(self):
        # Allow tests using patch.object to clean up
        try:
            delattr(self.agent, 'llm')
        except Exception:
            # If underlying agent doesn't allow delete, set to None
            try:
                setattr(self.agent, 'llm', None)
            except Exception:
                pass
    
    @property
    def agent_config(self):
        """Access to the agent configuration."""
        return getattr(self.agent, 'agent_config', None) or getattr(self.agent, 'config', None)
    
    @property
    def tool_registry(self):
        """Access to the tool registry."""
        return getattr(self.agent, 'tool_registry', None)
    
    @property
    def graph(self):
        """Access to the workflow graph."""
        # Check for test override first
        if hasattr(self.agent, '_test_graph_override'):
            return self.agent._test_graph_override
        return getattr(self.agent, 'graph', None)
    
    @graph.setter
    def graph(self, value):
        """Set the workflow graph (for test compatibility)."""
        if hasattr(self.agent, 'graph'):
            # If the agent has a graph attribute, try to set it
            try:
                setattr(self.agent, 'graph', value)
            except AttributeError:
                # If it can't be set, store it as a private attribute for tests
                self.agent._test_graph_override = value
        else:
            # If no graph attribute exists, create one
            self.agent.graph = value
    
    @graph.deleter
    def graph(self):
        # Allow cleanup in tests
        try:
            delattr(self.agent, 'graph')
        except Exception:
            try:
                setattr(self.agent, 'graph', None)
            except Exception:
                pass
    
    @property
    def event_emitter(self):
        """Access to the event emitter."""
        return getattr(self.agent, 'event_emitter', None)
    
    @property
    def reasoning_tracer(self):
        """Access to the reasoning tracer."""
        return getattr(self.agent, 'reasoning_tracer', None)
    
    # Backward compatibility methods for tests
    def _extract_text_content(self, content) -> str:
        """
        Extract text content from various content formats.
        
        Handles:
        - String content: Return as-is
        - List of dicts with 'text' key: Extract and concatenate
        - Mixed lists: Handle both strings and dicts
        - Empty content: Return empty string
        - Other types: Convert to string
        """
        if content is None:
            # Tests expect string "None" for explicit None values
            return "None"
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            if not content:
                return ""
            
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, dict):
                    # Skip items without text key (like images)
                    continue
                else:
                    # Fallback to string conversion
                    text_parts.append(str(item))
            
            return "".join(text_parts)
        
        # Fallback to string conversion
        return str(content)
    
    def _chunk_content_preserving_markdown(self, content: str, chunk_size: int = 500) -> List[str]:
        """
        Chunk content while preserving markdown structure, especially tables.
        
        Simple implementation that avoids splitting tables and code blocks.
        
        Args:
            content: Text content to chunk
            chunk_size: Target size for each chunk
            
        Returns:
            List of content chunks that preserve markdown structure
        """
        from typing import List
        
        if not content or len(content) <= chunk_size:
            return [content] if content else []
        
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        in_table = False
        in_code_block = False
        
        for line in lines:
            # Check if we're entering/leaving a code block
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
            
            # Check if this looks like a table row (contains |)
            is_table_line = '|' in line and not in_code_block
            
            # Track table state
            if is_table_line and not in_table:
                in_table = True
            elif not is_table_line and in_table:
                in_table = False
            
            line_length = len(line)
            
            # Don't split if we're in a table or code block
            if (current_size + line_length + 1 > chunk_size and 
                current_chunk and 
                not in_table and 
                not in_code_block):
                # Emit current chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_length
            else:
                # Add to current chunk
                current_chunk.append(line)
                current_size += line_length + 1
        
        # Add remaining content
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _generate_search_queries(self, user_question: str, query_complexity=None) -> list[str]:
        """
        Generate search queries from a user question.
        
        Simple implementation that creates basic variations of the input question.
        In the new architecture, this functionality is handled by the Planner agent.
        
        Args:
            user_question: The user's question
            query_complexity: Complexity level (unused in simple implementation)
            
        Returns:
            List of search queries
        """
        if not user_question:
            return []
        
        # Basic query generation - create variations
        queries = [user_question.strip()]
        
        # Add some basic variations
        words = user_question.split()
        if len(words) > 3:
            # Create a shorter version with key terms
            key_words = [word for word in words if len(word) > 3 and word.lower() not in ['what', 'how', 'when', 'where', 'why', 'who', 'is', 'are', 'the', 'and', 'or']]
            if key_words:
                queries.append(" ".join(key_words[:5]))
        
        # Add question without question words
        clean_query = user_question.lower()
        for qword in ['what is', 'how to', 'how do', 'when is', 'where is', 'why is', 'who is']:
            if clean_query.startswith(qword):
                clean_query = clean_query[len(qword):].strip()
                if clean_query:
                    queries.append(clean_query)
                break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in queries:
            if query not in seen and len(query.strip()) > 2:
                seen.add(query)
                unique_queries.append(query)
        
        return unique_queries[:3]  # Limit to 3 queries


# For MLflow model registration - Register when loaded by MLflow OR not in test environment
def _is_mlflow_loading_context():
    """Check if we're being loaded by MLflow for model logging/serving."""
    
    # NEW: Check if we're in MLflow deployment structure (most reliable indicator)
    if os.path.exists("/model/code"):
        logger.debug("MLflow deployment structure detected (/model/code exists)")
        return True
    
    # Check for code_model modules (for other MLflow loading patterns)
    code_model_modules = [name for name in sys.modules.keys() if "code_model_" in name]
    if code_model_modules:
        logger.debug(f"Found code_model modules: {code_model_modules[:3]}")
        return True
    
    # Check for MLflow-specific environment variables
    mlflow_env_vars = ["MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID", "DATABRICKS_MODEL_SERVING"]
    set_vars = [var for var in mlflow_env_vars if os.getenv(var)]
    if set_vars:
        logger.debug(f"Found MLflow environment variables: {set_vars}")
        return True
    
    return False

def _should_skip_registration():
    """Check if we should skip agent registration (true test environments only)."""
    # Always register if we're in MLflow loading context
    if _is_mlflow_loading_context():
        logger.debug("MLflow loading context detected - will register model")
        return False
    
    # Check test environment indicators
    test_conditions = {
        "PYTEST_CURRENT_TEST": os.getenv("PYTEST_CURRENT_TEST") is not None,
        "PYTEST_RUNNING": bool(os.getenv("PYTEST_CURRENT_TEST")), 
        "pytest_in_modules": "pytest" in sys.modules
    }
    
    active_conditions = [k for k, v in test_conditions.items() if v]
    if active_conditions:
        logger.debug(f"Test environment detected: {active_conditions}")
        return True
    
    logger.debug("No test environment indicators found - will register model")
    return False

# Register the model unless we're in a true test environment
if not _should_skip_registration():
    try:
        import mlflow
        # Try enabling autologging if available, but don't let it block registration
        try:
            _ = getattr(mlflow, "langchain", None)
            if _ is not None and hasattr(mlflow.langchain, "autolog"):
                mlflow.langchain.autolog()
            else:
                logger.info("mlflow.langchain.autolog not available; continuing without autolog")
        except Exception as autolog_error:
            logger.info(f"Skipping mlflow.langchain.autolog due to error: {autolog_error}")
        # Use standard config discovery logic (searches CONFIG_SEARCH_ORDER)
        default_agent = DatabricksCompatibleAgent()
        logger.info("Successfully initialized with discovered configuration")
        mlflow.models.set_model(default_agent)
        logger.info("Successfully registered enhanced agent system with MLflow")
    except Exception as e:
        logger.error(f"Failed to register agent with MLflow: {e}")
        # Try to initialize with minimal config as last resort, but do not raise on failure
        try:
            default_agent = DatabricksCompatibleAgent(yaml_path=None)
            mlflow.models.set_model(default_agent)
            logger.warning("Used fallback agent initialization")
        except Exception as final_error:
            logger.error(f"Complete agent initialization failure: {final_error}")
            # Silently skip registration in non-critical contexts
            default_agent = None
else:
    if _is_mlflow_loading_context():
        logger.warning("Agent registration skipped despite MLflow context - this may cause code-based logging to fail")
    else:
        logger.info("Skipping agent initialization in test environment")