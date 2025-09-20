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

from typing_extensions import deprecated

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
    AgentConfiguration,
    ResearchContext,
    ResearchAgentError,
    WorkflowError,
    format_duration,
    URLResolver
)
from deep_research_agent.core.markdown_utils import extract_and_fix_tables
from deep_research_agent.core.event_emitter import initialize_event_emitter, get_event_emitter
from deep_research_agent.core.reasoning_tracer import initialize_reasoning_tracer, get_reasoning_tracer
from deep_research_agent.core.types import ReasoningVisibility
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

@deprecated("Use EnhancedResearchAgent instead - this class is maintained for backward compatibility only")
class RefactoredResearchAgentDeprecated(ResponsesAgent):
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
                from deep_research_agent.core.unified_config import get_config_manager
                self.config_manager = get_config_manager(override_config=config_dict, yaml_path=yaml_path)
                self.agent_config = config  # Use the provided configuration directly
            else:
                # Handle dictionary config, YAML path, or None
                from deep_research_agent.core.unified_config import get_config_manager
                self.config_manager = get_config_manager(override_config=config, yaml_path=yaml_path)
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
            
            # Initialize intermediate event emitter and reasoning tracer
            self._initialize_intermediate_events()
            
            logger.info(
                "Successfully initialized refactored research agent",
                llm_endpoint=self.agent_config.llm_endpoint,
                max_loops=self.agent_config.max_research_loops,
                phase2_enabled=True,
                intermediate_events_enabled=self.agent_config.emit_intermediate_events
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
    
    def _initialize_intermediate_events(self):
        """Initialize intermediate event emitter and reasoning tracer."""
        if not self.agent_config.emit_intermediate_events:
            logger.info("Intermediate events disabled by configuration")
            return
        
        # Get intermediate events configuration
        intermediate_config = self.config_manager.load_yaml_config().get("intermediate_events", {})
        
        # Initialize event emitter with stream emission
        self.event_emitter = initialize_event_emitter(
            stream_emitter=self._emit_intermediate_event,
            max_events_per_second=intermediate_config.get("max_events_per_second", 10),
            batch_events=intermediate_config.get("batch_events", True),
            batch_size=intermediate_config.get("batch_size", 5),
            batch_timeout_ms=intermediate_config.get("batch_timeout_ms", 100),
            redaction_patterns=intermediate_config.get("redact_patterns", [])
        )
        
        # Initialize reasoning tracer
        visibility = ReasoningVisibility(self.agent_config.reasoning_visibility)
        self.reasoning_tracer = initialize_reasoning_tracer(
            visibility=visibility,
            token_interval=self.agent_config.thought_snapshot_interval_tokens,
            time_interval_ms=self.agent_config.thought_snapshot_interval_ms,
            max_chars_per_step=self.agent_config.max_thought_chars_per_step,
            redaction_patterns=intermediate_config.get("redact_patterns", []),
            event_emitter=lambda event: self.event_emitter.emit(
                event.event_type,
                event.data,
                correlation_id=event.correlation_id,
                stage_id=event.stage_id,
                meta=event.meta
            )
        )
        
        logger.info(
            "Initialized intermediate events",
            visibility=visibility.value,
            max_events_per_second=intermediate_config.get("max_events_per_second", 10)
        )
    
    def _emit_intermediate_event(self, event_data: dict):
        """Emit intermediate event through the stream interface."""
        try:
            # Convert to ResponsesAgentStreamEvent format for consistency
            stream_event = ResponsesAgentStreamEvent(
                type="response.intermediate_event",
                delta=json.dumps(event_data),
                item_id=event_data.get("id", str(uuid4()))
            )
            
            # Store for potential batch emission (this could be enhanced to queue)
            if not hasattr(self, '_pending_intermediate_events'):
                self._pending_intermediate_events = []
            self._pending_intermediate_events.append(stream_event)
            
            logger.debug(f"Queued intermediate event: {event_data.get('event_type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to emit intermediate event: {e}")
    
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
        
        # Extract LATEST user question from messages (not first)
        user_question = ""
        for msg in reversed(langchain_msgs):
            if isinstance(msg, HumanMessage):
                user_question = msg.content
                break
        
        # Build conversation history (all messages except current question)
        conversation_history = []
        human_message_count = 0
        for msg in langchain_msgs:
            if isinstance(msg, HumanMessage):
                human_message_count += 1
                if human_message_count < len([m for m in langchain_msgs if isinstance(m, HumanMessage)]):
                    conversation_history.append(msg)
            elif len(conversation_history) > 0 or human_message_count == 0:
                conversation_history.append(msg)
        
        # Extract previous research topics from conversation history
        previous_topics = []
        for msg in conversation_history:
            if isinstance(msg, HumanMessage):
                # Simple topic extraction - could be enhanced with NLP
                topic = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                previous_topics.append(topic)
        
        # Initialize state for the graph
        initial_state = {
            "messages": langchain_msgs,
            "research_context": ResearchContext(
                original_question=user_question,
                conversation_history=conversation_history,
                current_turn_index=human_message_count - 1,
                previous_research_topics=previous_topics,
                enable_streaming=True,  # Enable streaming by default
                streaming_chunk_size=50  # Default chunk size
            ),
            "start_time": time.time(),
            "url_resolver": self.url_resolver,
            "search_tasks": {},
            "batch_info": None,
            "parallel_search_results": []
        }
        
        try:
            # Track the item_id for delta events - use same ID for all events in this response
            item_id = str(uuid4())
            collected_content = []
            synthesis_started = False
            
            # Initialize intermediate event emission
            correlation_id = str(uuid4())
            if hasattr(self, 'event_emitter') and self.event_emitter:
                self.event_emitter.emit_action_start(
                    action="research_workflow",
                    query=user_question,
                    correlation_id=correlation_id,
                    stage_id="workflow_start"
                )
            
            # Start reasoning tracer
            if hasattr(self, 'reasoning_tracer') and self.reasoning_tracer:
                self.reasoning_tracer.start_step(correlation_id, "workflow_start")
                self.reasoning_tracer.add_reasoning_step(
                    "Starting research workflow",
                    f"Question: {user_question[:100]}..." if len(user_question) > 100 else user_question
                )
            
            # Progress tracking
            progress_tracker = {
                "start_time": time.time(),
                "nodes_completed": 0,
                "total_nodes": 8,  # Approximate workflow nodes count
                "seen_nodes": set(),  # Track unique nodes for progress calculation
                "parallel_executions": {}  # Track parallel executions separately
            }
            
            # Stream the graph execution - use updates mode to track node outputs
            for event in self.graph.stream(initial_state, stream_mode=["updates"]):
                # Process updates from nodes
                if isinstance(event, dict):
                    items = event.items()
                elif isinstance(event, tuple) and len(event) == 2:
                    # Handle LangGraph tuple format: (mode, data)
                    mode, data = event
                    if mode == "updates" and isinstance(data, dict):
                        items = data.items()
                    else:
                        # Fallback: support tests that yield ("messages", [AIMessage-like])
                        if isinstance(mode, str) and mode == "messages" and isinstance(data, list):
                            # Stream message contents directly as deltas
                            for msg in data:
                                try:
                                    chunk_text = getattr(msg, "content", str(msg))
                                except Exception:
                                    chunk_text = str(msg)
                                if chunk_text:
                                    yield ResponsesAgentStreamEvent(
                                        type="response.output_text.delta",
                                        item_id=item_id,
                                        delta=chunk_text,
                                    )
                                    collected_content.append(chunk_text)
                            # No structured items to iterate in this fallback
                            continue
                        continue
                else:
                    # Unknown format, skip
                    continue
                
                for node_name, node_data in items:
                    # Track parallel executions separately
                    if node_name == "parallel_web_search":
                        progress_tracker["parallel_executions"][node_name] = \
                            progress_tracker["parallel_executions"].get(node_name, 0) + 1
                    
                    # Only increment progress for first occurrence of each unique node
                    # This prevents parallel executions from inflating progress percentage
                    if node_name not in progress_tracker["seen_nodes"]:
                        progress_tracker["nodes_completed"] += 1
                        progress_tracker["seen_nodes"].add(node_name)
                    
                    # Calculate progress percentage based on unique nodes only
                    progress_percentage = min(
                        (progress_tracker["nodes_completed"] / progress_tracker["total_nodes"]) * 100,
                        99  # Cap at 99% until final completion
                    )
                    
                    # Emit intermediate action events
                    if hasattr(self, 'event_emitter') and self.event_emitter:
                        self.event_emitter.emit_action_progress(
                            action=node_name,
                            status="executing",
                            progress={"percentage": progress_percentage, "step": progress_tracker["nodes_completed"]},
                            correlation_id=correlation_id,
                            stage_id=node_name
                        )
                    
                    # Add reasoning step
                    if hasattr(self, 'reasoning_tracer') and self.reasoning_tracer:
                        self.reasoning_tracer.add_reasoning_step(
                            f"Executing {node_name}",
                            f"Progress: {progress_percentage:.0f}%"
                        )
                    
                    # Emit any pending intermediate events
                    if hasattr(self, '_pending_intermediate_events'):
                        for pending_event in self._pending_intermediate_events:
                            yield pending_event
                        self._pending_intermediate_events.clear()
                    
                    # Emit progress event for EVERY node (this is the key enhancement!)
                    # Show progress for ALL nodes including synthesis for better UX
                    logger.info(f"Creating progress event for node: {node_name}, progress: {progress_percentage:.0f}%")
                    progress_event = self._create_progress_delta_event(
                        node_name, 
                        node_data, 
                        progress_tracker, 
                        progress_percentage, 
                        item_id
                    )
                    if progress_event:
                        logger.debug(f"Yielding progress event: type={progress_event.type}, delta_preview='{progress_event.delta[:100]}...'")
                        yield progress_event
                    else:
                        logger.warning(f"No progress event created for node: {node_name}")
                    
                    # Handle synthesis content with proper streaming support
                    if node_name == "synthesize_answer":
                        synthesis_started = True
                        
                        # Check if we have streaming chunks from the synthesis node
                        research_context = node_data.get('research_context')
                        if research_context and hasattr(research_context, 'synthesis_chunks'):
                            synthesis_chunks = research_context.synthesis_chunks
                        elif isinstance(research_context, dict) and 'synthesis_chunks' in research_context:
                            synthesis_chunks = research_context['synthesis_chunks']
                        else:
                            synthesis_chunks = []
                        
                        if synthesis_chunks and len(synthesis_chunks) > 1:
                            # True streaming: emit multiple delta events
                            chunk_size = getattr(research_context, 'streaming_chunk_size', 50) if hasattr(research_context, 'streaming_chunk_size') else 50
                            
                            for chunk in synthesis_chunks:
                                if chunk.strip():  # Skip empty chunks
                                    # Stream raw chunk; avoid per-chunk table fixing to prevent duplication
                                    fixed_chunk = chunk
                                    # For very large chunks, split them further for better streaming UX
                                    if len(fixed_chunk) > chunk_size * 2:
                                        # Use markdown-preserving chunking
                                        for sub_chunk in self._chunk_content_preserving_markdown(fixed_chunk):
                                            yield ResponsesAgentStreamEvent(
                                                type="response.output_text.delta",
                                                item_id=item_id,
                                                delta=sub_chunk
                                            )
                                            collected_content.append(sub_chunk)
                                    else:
                                        # Chunk is already appropriately sized
                                        yield ResponsesAgentStreamEvent(
                                            type="response.output_text.delta",
                                            item_id=item_id,
                                            delta=fixed_chunk
                                        )
                                        collected_content.append(fixed_chunk)
                        else:
                            # Fallback to existing logic for single chunk or no chunks
                            if isinstance(node_data, dict) and "messages" in node_data:
                                messages = node_data["messages"]
                                if messages and len(messages) > 0:
                                    last_msg = messages[-1]
                                    if hasattr(last_msg, 'content') and last_msg.content:
                                        content = last_msg.content
                                        if not (content.strip().startswith('{') and content.strip().endswith('}')):
                                            # Content already fixed during synthesis; stream as-is
                                            fixed_content = content
                                            # For single chunk, simulate streaming with proper markdown preservation
                                            if len(fixed_content) > 100:  # Only simulate for longer content
                                                # Chunk content preserving markdown structure
                                                for chunk in self._chunk_content_preserving_markdown(fixed_content):
                                                    yield ResponsesAgentStreamEvent(
                                                        type="response.output_text.delta",
                                                        item_id=item_id,
                                                        delta=chunk
                                                    )
                                                    collected_content.append(chunk)
                                            else:
                                                # Short content - emit as single delta
                                                yield ResponsesAgentStreamEvent(
                                                    type="response.output_text.delta",
                                                    item_id=item_id,
                                                    delta=fixed_content
                                                )
                                                collected_content.append(fixed_content)
                    
                    # Enhanced logging for all intermediate outputs
                    else:
                        logger.debug(f"[Progress] {node_name}: {progress_percentage:.0f}% complete")
            
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
                                # Content already fixed during synthesis; collect as-is
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
                
                # Note: Final progress event removed to maintain schema compliance
                # The UI can infer completion when the done event is received
                
                # ULTIMATE TABLE CLEANUP - catch any remaining double pipes from ANY source
                import re
                absolutely_clean_content = re.sub(r'\|\|+', '|', final_content)
                double_pipes_removed = final_content.count('||') - absolutely_clean_content.count('||')
                if double_pipes_removed > 0:
                    logger.warning(f"FINAL CLEANUP: Removed {double_pipes_removed} more double pipes at response creation")
                
                final_item = self.create_text_output_item(absolutely_clean_content, item_id)
                
                # Emit completion events
                if hasattr(self, 'event_emitter') and self.event_emitter:
                    self.event_emitter.emit_action_complete(
                        action="research_workflow",
                        result_summary=f"Generated {len(absolutely_clean_content)} character response",
                        results_count=len(collected_content),
                        correlation_id=correlation_id,
                        stage_id="workflow_complete"
                    )
                
                # End reasoning tracer
                if hasattr(self, 'reasoning_tracer') and self.reasoning_tracer:
                    self.reasoning_tracer.add_reasoning_step(
                        "Completed research workflow",
                        f"Final response ready ({len(absolutely_clean_content)} chars)"
                    )
                    self.reasoning_tracer.end_step()
                
                yield ResponsesAgentStreamEvent(type="response.output_item.done", item=final_item)
            else:
                # Fallback response if no content was collected
                fallback_content = "I'm ready to help you research any topic. Please provide a specific question or topic you'd like me to explore."
                # Apply table cleanup even to fallback (shouldn't be needed but for consistency)
                import re
                fallback_content = re.sub(r'\|\|+', '|', fallback_content)
                
                # Emit completion events for fallback
                if hasattr(self, 'event_emitter') and self.event_emitter:
                    self.event_emitter.emit_action_complete(
                        action="research_workflow",
                        result_summary="Generated fallback response",
                        results_count=0,
                        correlation_id=correlation_id,
                        stage_id="workflow_complete"
                    )
                
                # End reasoning tracer for fallback
                if hasattr(self, 'reasoning_tracer') and self.reasoning_tracer:
                    self.reasoning_tracer.add_reasoning_step(
                        "Generated fallback response",
                        "No research content collected"
                    )
                    self.reasoning_tracer.end_step()
                
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
    
    def _create_progress_delta_event(
        self, 
        node_name: str, 
        node_data: Dict[str, Any], 
        progress_tracker: Dict[str, Any],
        progress_percentage: float,
        item_id: str
    ) -> Optional[ResponsesAgentStreamEvent]:
        """Create progress delta event with parseable markers for UI streaming."""
        import time
        
        # Map nodes to research phases that UI understands
        phase_mapping = {
            "generate_queries": "QUERYING",
            "batch_controller": "PREPARING", 
            "route_to_parallel_search": "PREPARING",
            "parallel_web_search": "SEARCHING",
            "aggregate_search_results": "AGGREGATING", 
            "vector_research": "SEARCHING_INTERNAL",
            "reflect": "ANALYZING",
            "synthesize_answer": "SYNTHESIZING",
            # Additional nodes that might appear during execution
            "web_search": "SEARCHING",
            "search": "SEARCHING",
            "analysis": "ANALYZING",
            "synthesis": "SYNTHESIZING",
            "prepare": "PREPARING",
            "process": "PROCESSING"
        }
        
        phase = phase_mapping.get(node_name, "PROCESSING")
        logger.debug(f"Progress phase mapping: {node_name} -> {phase}")
        
        # Extract rich metadata from node_data
        metadata_parts = []
        
        if node_name == "generate_queries" and isinstance(node_data, dict):
            if "research_context" in node_data:
                context = node_data["research_context"]
                if hasattr(context, 'generated_queries'):
                    query_count = len(context.generated_queries)
                    metadata_parts.append(f"[META:queries:{query_count}]")
                    
        elif node_name == "aggregate_search_results" and isinstance(node_data, dict):
            if "research_context" in node_data:
                context = node_data["research_context"]
                if hasattr(context, 'web_results'):
                    result_count = len(context.web_results)
                    metadata_parts.append(f"[META:results:{result_count}]")
        
        elif node_name == "vector_research" and isinstance(node_data, dict):
            if "research_context" in node_data:
                context = node_data["research_context"]
                if hasattr(context, 'vector_results'):
                    vector_count = len(context.vector_results)
                    metadata_parts.append(f"[META:vector_results:{vector_count}]")
        
        # Build progress message with UI-parseable markers
        message = f"[PHASE:{phase}] "
        
        # Add human-readable description with emojis (percentages shown separately in UI)
        descriptions = {
            "QUERYING": "🔍 Analyzing your query and generating search strategies",
            "PREPARING": "📋 Preparing search execution",
            "SEARCHING": "🌐 Searching across multiple sources",
            "SEARCHING_INTERNAL": "🗄️ Searching internal knowledge base",
            "AGGREGATING": "📊 Aggregating search results",
            "ANALYZING": "🤔 Analyzing search results and extracting insights",
            "SYNTHESIZING": "✍️ Synthesizing comprehensive response",
            "PROCESSING": "⚙️ Processing research data"
        }
        
        message += descriptions.get(phase, f"⚙️ Processing {node_name}")
        
        # Add metadata markers
        if metadata_parts:
            message += "\n" + "\n".join(metadata_parts)
        
        # Add timing and progress metadata
        elapsed = time.time() - progress_tracker["start_time"]
        message += f"\n[META:elapsed:{elapsed:.1f}]"
        message += f"\n[META:progress:{progress_percentage:.0f}]"
        message += f"\n[META:node:{node_name}]"
        
        # Add separator for UI parsing (helps distinguish progress events from content)
        message += "\n---\n"
        
        # Return as plain text delta event (MLflow compliant - no JSON!)
        return ResponsesAgentStreamEvent(
            type="response.output_text.delta",
            item_id=item_id,
            delta=message
        )
    
    def _chunk_content_preserving_markdown(self, content: str, chunk_size: int = 500) -> List[str]:
        """
        Chunk content while preserving markdown structure, especially tables.
        
        This enhanced method ensures tables are NEVER split mid-row and treats
        complete tables as atomic units when possible.
        
        Args:
            content: Text content to chunk
            chunk_size: Target size for each chunk (increased default for tables)
            
        Returns:
            List of content chunks that preserve markdown structure
        """
        # Import table validator for table detection
        from deep_research_agent.core.table_validator import TableValidator
        
        # Quick return for small content
        if not content or len(content) <= chunk_size:
            return [content] if content else []
        
        # For tables, use a larger minimum chunk size; otherwise respect requested size
        MIN_TABLE_CHUNK_SIZE = 1000
        content_has_tables = '|' in content
        effective_chunk_size = max(chunk_size, MIN_TABLE_CHUNK_SIZE) if content_has_tables else chunk_size
        
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        in_table = False
        table_buffer = []
        
        for i, line in enumerate(lines):
            # Check if this line is part of a table
            is_table_line = TableValidator._is_table_row(line.strip()) if line.strip() else False
            
            if is_table_line:
                # We're in a table
                if not in_table:
                    # Table is starting
                    in_table = True
                    table_buffer = [line]
                else:
                    # Continue collecting table
                    table_buffer.append(line)
            else:
                # Not a table line
                if in_table:
                    # Table just ended - add entire table as atomic unit
                    in_table = False
                    table_content = '\n'.join(table_buffer)
                    
                    # Check if current chunk + table would be too large
                    if current_chunk and (current_size + len(table_content) + 1 > effective_chunk_size):
                        # Emit current chunk first
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    
                    # Add table to current chunk (tables are atomic)
                    if table_content:
                        # If table itself is larger than chunk size, it still goes as one unit
                        if not current_chunk and len(table_content) > effective_chunk_size:
                            # Table is very large, emit it as its own chunk
                            chunks.append(table_content)
                        else:
                            # Add table to current chunk
                            current_chunk.append(table_content)
                            current_size += len(table_content) + 1
                    
                    table_buffer = []
                
                # Process non-table line
                line_length = len(line)
                
                # Check if adding this line would exceed chunk size
                if current_size + line_length + 1 > chunk_size and current_chunk:
                    # Emit current chunk
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_length
                else:
                    # Add to current chunk
                    current_chunk.append(line)
                    current_size += line_length + 1
        
        # Handle any remaining table
        if in_table and table_buffer:
            table_content = '\n'.join(table_buffer)
            if current_chunk and (current_size + len(table_content) + 1 > effective_chunk_size):
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
            current_chunk.append(table_content)
        
        # Add remaining content
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        # Helper to split non-table text at word boundaries
        def split_by_word_boundaries(text: str, target_size: int) -> List[str]:
            parts: List[str] = []
            start = 0
            n = len(text)
            while start < n:
                end = min(start + target_size, n)
                if end < n:
                    # backtrack to last whitespace
                    ws = text.rfind(' ', start, end)
                    if ws != -1 and ws > start:
                        end = ws + 1
                parts.append(text[start:end])
                start = end
            return [p for p in parts if p]

        # Final pass: ensure chunks are reasonable and no empty chunks
        final_chunks = []
        for chunk in chunks:
            if chunk.strip():
                if '|' not in chunk and len(chunk) > effective_chunk_size:
                    # Prefer paragraph splitting first
                    paragraphs = chunk.split('\n\n')
                    if len(paragraphs) > 1:
                        current_para_chunk = ""
                        for para in paragraphs:
                            if len(current_para_chunk) + len(para) + 2 > effective_chunk_size and current_para_chunk:
                                final_chunks.append(current_para_chunk.rstrip())
                                current_para_chunk = para
                            else:
                                current_para_chunk += ('\n\n' if current_para_chunk else '') + para
                        if current_para_chunk.strip():
                            final_chunks.append(current_para_chunk.rstrip())
                    else:
                        # No paragraph boundaries; split by words
                        final_chunks.extend(split_by_word_boundaries(chunk, effective_chunk_size))
                else:
                    final_chunks.append(chunk)
        
        # Ensure no empty chunks
        return [chunk for chunk in final_chunks if chunk.strip()]

    # Backward compatibility methods for tests
    def _extract_text_content(self, content) -> str:
        """Extract text content - backward compatibility method."""
        return self.response_utils.extract_text_content(content)
    
    def _generate_search_queries(self, user_question: str, query_complexity=None) -> list[str]:
        """Generate search queries - backward compatibility method."""
        return self.workflow_nodes._generate_search_queries(user_question, query_complexity)