"""
Refactored LangGraph Research Agent with improved architecture.

This module implements a clean, maintainable research agent using the new
core libraries and components for better separation of concerns and robustness.
"""

import json
import time
import threading
from typing import Dict, Any, Optional, Generator, List
from uuid import uuid4
from pathlib import Path
import sys

# Ensure current directory and parent are in Python path for imports
# This handles both local development and MLflow deployment contexts
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# For MLflow deployment, also add parent directory to handle package structure
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage
import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent
)
from mlflow.entities import SpanType

from deep_research_agent.core import (
    get_logger,
    ConfigManager,
    AgentConfiguration,
    ResearchContext,
    ResearchAgentError,
    WorkflowError,
    format_duration,
    URLResolver
)
from deep_research_agent.components import (
    message_converter,
    create_tool_registry,
    response_builder
)

# Import refactored modules
from deep_research_agent.state_management import AgentStateDict
from deep_research_agent.agent_initialization import AgentInitializer
from deep_research_agent.response_utils import ResponseUtils
from deep_research_agent.workflow_nodes import WorkflowNodes

logger = get_logger(__name__)


class RefactoredResearchAgent(ResponsesAgent):
    """
    Refactored research agent with improved architecture and error handling.
    
    This agent uses the new core libraries and components for better maintainability,
    robustness, and separation of concerns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, yaml_path: Optional[str] = None):
        """
        Initialize the refactored research agent.
        
        Args:
            config: Optional configuration dictionary or AgentConfiguration object
            yaml_path: Path to YAML configuration file (optional)
        """
        logger.info("Initializing refactored research agent")
        
        try:
            # Initialize configuration - YAML config is mandatory
            if isinstance(config, AgentConfiguration):
                # If we receive an AgentConfiguration object, convert to dict for ConfigManager
                from dataclasses import asdict
                config_dict = asdict(config)
                self.config_manager = ConfigManager(config_dict, yaml_path)
                self.agent_config = config  # Use the provided configuration directly
            else:
                # Handle dictionary config, YAML path, or None
                self.config_manager = ConfigManager(config, yaml_path)
                self.agent_config = self.config_manager.get_agent_config()
            
            # Initialize tool registry
            self.tool_registry = create_tool_registry(self.config_manager)
            
            # Initialize rate limiting semaphore for parallel searches
            self.search_semaphore = threading.Semaphore(self.agent_config.max_concurrent_searches)
            
            # Initialize LLM
            self.llm = AgentInitializer.initialize_llm(self.agent_config)
            
            # Initialize Phase 2 components
            self._initialize_phase2_components()
            
            # Initialize URL resolver
            self.url_resolver = URLResolver()
            
            # Initialize workflow nodes
            self.workflow_nodes = WorkflowNodes(self)
            
            # Build workflow graph
            self.graph = self._build_workflow_graph()
            
            # Initialize response utils
            self.response_utils = ResponseUtils()
            
            logger.info(
                "Successfully initialized refactored research agent",
                llm_endpoint=self.agent_config.llm_endpoint,
                max_loops=self.agent_config.max_research_loops,
                phase2_enabled=True
            )
        except Exception as e:
            # Re-raise configuration errors with clear messaging about YAML requirement
            if "configuration file not found" in str(e).lower():
                error_msg = f"Agent initialization failed - YAML configuration is mandatory: {e}"
                logger.error(error_msg)
                raise ResearchAgentError(error_msg)
            else:
                # Re-raise other errors as ResearchAgentError
                logger.error(f"Failed to initialize research agent: {e}")
                raise ResearchAgentError(f"Agent initialization failed: {e}")
    
    def _initialize_phase2_components(self):
        """Initialize Phase 2 components for enhanced research capabilities."""
        (
            self.embedding_client,
            self.model_manager,
            self.deduplicator,
            self.query_analyzer,
            self.result_evaluator,
            self.adaptive_generator
        ) = AgentInitializer.initialize_phase2_components(self.config_manager)
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        try:
            workflow = StateGraph(AgentStateDict)  # Use TypedDict for state
            
            # Add workflow nodes
            workflow.add_node("generate_queries", self.workflow_nodes.generate_queries_node)
            workflow.add_node("batch_controller", self.workflow_nodes.batch_controller_node)
            workflow.add_node("parallel_web_search", self.workflow_nodes.parallel_web_search_node)
            workflow.add_node("aggregate_search_results", self.workflow_nodes.aggregate_search_results_node)
            workflow.add_node("vector_research", self.workflow_nodes.vector_research_node)
            workflow.add_node("reflect", self.workflow_nodes.reflect_node)
            workflow.add_node("synthesize_answer", self.workflow_nodes.synthesize_answer_node)
            
            # Define workflow edges
            workflow.set_entry_point("generate_queries")
            workflow.add_edge("generate_queries", "batch_controller")
            workflow.add_conditional_edges(
                "batch_controller", 
                self.workflow_nodes.route_to_parallel_search,
                ["parallel_web_search"]
            )
            workflow.add_edge("parallel_web_search", "aggregate_search_results")
            workflow.add_edge("aggregate_search_results", "vector_research")
            workflow.add_edge("vector_research", "reflect")
            
            # Conditional edge for research loops
            workflow.add_conditional_edges(
                "reflect",
                self.workflow_nodes.should_continue_research,
                {
                    "continue": "generate_queries",
                    "finish": "synthesize_answer"
                }
            )
            workflow.add_edge("synthesize_answer", END)
            
            compiled_graph = workflow.compile()
            logger.info("Successfully built workflow graph")
            return compiled_graph
        except Exception as e:
            logger.error("Failed to build workflow graph", error=e)
            raise WorkflowError(f"Failed to build workflow: {e}")
    
    # Public interface methods
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Process research request synchronously following MLflow ResponsesAgent interface.
        
        This method collects complete responses from the streaming predict_stream method
        and returns them in the ResponsesAgentResponse format required by Databricks
        Agent Framework.
        
        Args:
            request (ResponsesAgentRequest): Input request containing:
                - input: List[Dict[str, str]] with message dictionaries
                  Each message must have 'role' (user/assistant/system) and 'content' (string)
                - custom_inputs: Optional[Dict[str, Any]] for custom parameters
        
        Returns:
            ResponsesAgentResponse: Response containing:
                - output: List of done event items with structured message format
                - custom_outputs: Dictionary (typically from request.custom_inputs)
        
        Schema Requirements:
            - Input messages must follow {"role": str, "content": str} format
            - Output items come only from "response.output_item.done" events
            - Each output item has type, role, id, and content fields
            - Content is a list with {"type": "output_text", "text": str} items
        
        See SCHEMA_REQUIREMENTS.md for complete schema documentation.
        """
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        response = ResponsesAgentResponse(output=outputs,
                                          custom_outputs=getattr(request, 'custom_inputs', None) or {}
                                          )
        return response
    
    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Stream research responses with delta and done events per MLflow ResponsesAgent spec.
        
        This method processes research requests through the LangGraph workflow and streams
        responses following the exact schema required by Databricks serving endpoints.
        
        Args:
            request (ResponsesAgentRequest): Input request containing:
                - input: List[Dict[str, str]] with message dictionaries
                - custom_inputs: Optional[Dict[str, Any]] for custom parameters
        
        Yields:
            ResponsesAgentStreamEvent: Stream events of two types:
                1. Delta events ("response.output_text.delta"):
                   - item_id: Unique identifier for the message
                   - delta: Plain text chunk (NEVER JSON objects)
                2. Done events ("response.output_item.done"):
                   - item: Complete message with type, role, id, and content
        
        Schema Requirements:
            - Delta events MUST contain only plain text, no JSON objects
            - Item IDs must be consistent between related delta and done events
            - Exactly one done event per response with complete message
            - Messages are converted: Request → ChatCompletion → LangChain → Stream
        
        Implementation Flow:
            1. Convert request messages to ChatCompletion format
            2. Convert to LangChain messages for graph processing
            3. Execute research workflow through LangGraph
            4. Stream delta events during synthesis phase only
            5. Send done event with complete response
        
        See SCHEMA_REQUIREMENTS.md for complete schema documentation.
        """
        # Convert request messages to ChatCompletion format
        cc_msgs = []
        for msg in request.input:
            if isinstance(msg, dict):
                cc_msgs.extend(self._responses_to_cc(msg))
            else:
                cc_msgs.extend(self._responses_to_cc(msg.model_dump()))
        
        # Convert CC messages to LangChain messages for the graph
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        langchain_msgs = []
        for msg in cc_msgs:
            if msg.get("role") == "user":
                langchain_msgs.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                langchain_msgs.append(AIMessage(content=msg.get("content", "")))
            elif msg.get("role") == "system":
                langchain_msgs.append(SystemMessage(content=msg.get("content", "")))
        
        # Extract user question from messages
        user_question = ""
        for msg in langchain_msgs:
            if isinstance(msg, HumanMessage):
                user_question = msg.content
                break
        
        # Initialize state for the graph
        initial_state = {
            "messages": langchain_msgs,
            "research_context": ResearchContext(original_question=user_question),
            "start_time": time.time(),
            "url_resolver": self.url_resolver,
            "search_tasks": {},
            "batch_info": None,
            "parallel_search_results": []
        }
        
        try:
            # Track the item_id for delta events
            item_id = str(uuid4())
            collected_content = []
            synthesis_started = False
            
            # Stream the graph execution - use updates mode to track node outputs
            for event in self.graph.stream(initial_state, stream_mode=["updates"]):
                # Process updates from nodes
                if isinstance(event, dict):
                    items = event.items()
                else:
                    # Handle potential tuple format (mode, data)
                    continue
                
                for node_name, node_data in items:
                    # Only stream content from the synthesis node
                    if node_name == "synthesize_answer":
                        synthesis_started = True
                        # Extract the final message from synthesis
                        if isinstance(node_data, dict) and "messages" in node_data:
                            messages = node_data["messages"]
                            if messages and len(messages) > 0:
                                last_msg = messages[-1]
                                if hasattr(last_msg, 'content') and last_msg.content:
                                    # Skip JSON-formatted intermediate outputs
                                    content = last_msg.content
                                    if not (content.strip().startswith('{') and content.strip().endswith('}')):
                                        # Generate delta event for streaming content
                                        yield ResponsesAgentStreamEvent(
                                            type="response.output_text.delta",
                                            item_id=item_id,
                                            delta=content
                                        )
                                        collected_content.append(content)
                    
                    # Log intermediate outputs as debug info instead of streaming them
                    elif node_name in ["generate_queries", "reflect"]:
                        logger.debug(f"[Intermediate] {node_name}: Processing...")
            
            # If no synthesis occurred but we have messages, get the last AI message
            if not synthesis_started and not collected_content:
                # Run the graph without streaming to get the final state
                final_state = self.graph.invoke(initial_state)
                if isinstance(final_state, dict) and "messages" in final_state:
                    for msg in reversed(final_state["messages"]):
                        if isinstance(msg, AIMessage) and msg.content:
                            content = msg.content
                            # Skip JSON outputs
                            if not (content.strip().startswith('{') and content.strip().endswith('}')):
                                collected_content.append(content)
                                break
            
            # Emit final done event with complete response
            if collected_content:
                # Join all collected content
                final_content = "".join(collected_content)
                
                # If no delta events were generated yet (synthesis_started is False), generate them now
                if not synthesis_started:
                    yield ResponsesAgentStreamEvent(
                        type="response.output_text.delta",
                        item_id=item_id,
                        delta=final_content
                    )
                
                final_item = self.create_text_output_item(final_content, item_id)
                yield ResponsesAgentStreamEvent(type="response.output_item.done", item=final_item)
            else:
                # Fallback response if no content was collected
                fallback_content = "I'm ready to help you research any topic. Please provide a specific question or topic you'd like me to explore."
                # Generate delta event for fallback
                yield ResponsesAgentStreamEvent(
                    type="response.output_text.delta",
                    item_id=item_id,
                    delta=fallback_content
                )
                fallback_item = self.create_text_output_item(fallback_content, item_id)
                yield ResponsesAgentStreamEvent(type="response.output_item.done", item=fallback_item)
        except Exception as e:
            logger.error("Stream execution failed", error=e)
            # Create error response using inherited method
            error_item = self.create_text_output_item(f"Error: {str(e)}", str(uuid4()))
            stream_event = ResponsesAgentStreamEvent(type="response.output_item.done", item=error_item)
            yield stream_event
    
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

    def _langchain_to_responses(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Convert from LangChain messages to Responses output item dictionaries."""
        responses_items = []
        for message in messages:
            if hasattr(message, 'model_dump'):
                message_dict = message.model_dump()
            else:
                # Fallback for messages without model_dump
                message_dict = {"type": message.__class__.__name__.lower().replace("message", ""), 
                               "content": str(message.content) if hasattr(message, 'content') else str(message)}
            
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
                # Skip user/human messages as they are inputs, not outputs
                pass
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


    def _create_enhanced_progress_event(
        self, 
        node_name: str, 
        node_data: Dict[str, Any], 
        progress_tracker: Dict[str, Any],
        progress_percentage: float
    ) -> Optional[ResponsesAgentStreamEvent]:
        """Create enhanced progress event with detailed information."""
        
        # Enhanced progress messages with percentage
        enhanced_messages = {
            "generate_queries": f"🔍 Generating search queries... ({progress_percentage:.0f}%)",
            "batch_controller": f"📋 Preparing batch execution... ({progress_percentage:.0f}%)",
            "route_to_parallel_search": f"📤 Routing to parallel search... ({progress_percentage:.0f}%)",
            "parallel_web_search": f"🌐 Rate-limited searches in progress... ({progress_percentage:.0f}%)",
            "aggregate_search_results": f"📊 Aggregating search results... ({progress_percentage:.0f}%)",
            "vector_research": f"🗄️ Searching internal knowledge... ({progress_percentage:.0f}%)",
            "reflect": f"🤔 Evaluating research quality... ({progress_percentage:.0f}%)",
            "synthesize_answer": f"✍️ Synthesizing final answer... ({progress_percentage:.0f}%)"
        }
        
        # Add specific details based on node type
        if node_name == "generate_queries" and isinstance(node_data, dict):
            if "research_context" in node_data:
                context = node_data["research_context"]
                if hasattr(context, 'generated_queries'):
                    query_count = len(context.generated_queries)
                    enhanced_messages[node_name] = f"🔍 Generated {query_count} search queries ({progress_percentage:.0f}%)"
        
        elif node_name == "aggregate_search_results" and isinstance(node_data, dict):
            if "research_context" in node_data:
                context = node_data["research_context"]
                if hasattr(context, 'web_results'):
                    result_count = len(context.web_results)
                    enhanced_messages[node_name] = f"📊 Aggregated {result_count} results ({progress_percentage:.0f}%)"
        
        elif node_name == "vector_research" and isinstance(node_data, dict):
            if "research_context" in node_data:
                context = node_data["research_context"]
                if hasattr(context, 'vector_results'):
                    vector_count = len(context.vector_results)
                    enhanced_messages[node_name] = f"🗄️ Found {vector_count} internal results ({progress_percentage:.0f}%)"
        
        # Calculate estimated time remaining
        elapsed_time = time.time() - progress_tracker["start_time"]
        if progress_percentage > 0:
            estimated_total = elapsed_time * (100 / progress_percentage)
            time_remaining = estimated_total - elapsed_time
            time_str = f" • ETA: {format_duration(time_remaining)}" if time_remaining > 0 else ""
        else:
            time_str = ""
        
        message = enhanced_messages.get(node_name, f"Processing {node_name}... ({progress_percentage:.0f}%)")
        message += time_str
        
        return response_builder.build_progress_event(node_name, message)
    
    # Backward compatibility methods for tests
    def _extract_text_content(self, content) -> str:
        """Extract text content - backward compatibility method."""
        return self.response_utils.extract_text_content(content)
    
    def _generate_search_queries(self, user_question: str, query_complexity=None) -> list[str]:
        """Generate search queries - backward compatibility method."""
        return self.workflow_nodes._generate_search_queries(user_question, query_complexity)