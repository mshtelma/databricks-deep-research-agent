"""
Enhanced Research Agent with Multi-Agent Architecture.

Complete integration of all multi-agent components with planning,
grounding, and report generation capabilities.
"""

import asyncio
import os
import time
import yaml
import threading
from typing import Dict, Any, Optional, AsyncIterator, Callable, Generator, List
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from deep_research_agent.core.noop_checkpointer import NoOpCheckpointSaver
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from deep_research_agent.core import get_logger
from deep_research_agent.core.exceptions import ConfigurationError
from deep_research_agent.core.multi_agent_state import (
    EnhancedResearchState,
    StateManager,
)
from deep_research_agent.core.report_styles import ReportStyle
from deep_research_agent.core.grounding import VerificationLevel
from deep_research_agent.core.event_emitter import (
    EventEmitter,
    initialize_event_emitter,
    IntermediateEventType,
)
from deep_research_agent.core.memory_monitor import get_memory_monitor
from deep_research_agent.core.state_validator import StateValidator, global_propagation_tracker
from deep_research_agent.core.validated_command import ValidatedCommand
from deep_research_agent.workflow_nodes_enhanced import EnhancedWorkflowNodes
from deep_research_agent.agents import (
    CoordinatorAgent,
    PlannerAgent,
    ResearcherAgent,
    ReporterAgent,
    FactCheckerAgent,
)
from deep_research_agent.components import create_tool_registry
from deep_research_agent.core.exceptions import SearchToolsFailedException


logger = get_logger(__name__)


# Phase mapping for UI-compatible PHASE markers
PHASE_MAPPING = {
    "coordinator": "ANALYZING",
    "background_investigation": "SEARCHING",  # Changed from INVESTIGATING for UI compatibility
    "planner": "QUERYING",  # Changed from PLANNING for UI compatibility
    "researcher": "RESEARCHING",
    "fact_checker": "VERIFYING",
    "reporter": "SYNTHESIZING",
}

# Node ordering for progress calculation
NODE_ORDER = [
    "coordinator",
    "background_investigation",
    "planner",
    "researcher",
    "fact_checker",
    "reporter",
]


class EnhancedResearchAgent(ResponsesAgent):
    """
    Enhanced Research Agent with multi-agent architecture.

    Now implements ResponsesAgent interface for Databricks deployment.

    Features:
    - Multi-agent coordination (Coordinator, Planner, Researcher, Reporter, Fact Checker)
    - Background investigation before planning
    - Iterative plan refinement with quality assessment
    - Step-by-step execution with context accumulation
    - Comprehensive grounding and factuality checking
    - Multiple report styles
    - Citation management
    - Reflexion-style self-improvement
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        llm=None,
        tool_registry=None,
        config_override: Optional[Dict] = None,
        stream_emitter: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs,
    ):
        """
        Initialize the enhanced research agent.

        Args:
            config_path: Path to configuration file
            llm: Language model to use
            tool_registry: Registry of available tools
            config_override: Configuration override for testing (replaces TEST_MODE)
            stream_emitter: Optional callback for streaming events
            **kwargs: Additional configuration overrides
        """
        # Load configuration (no TEST_MODE check!)
        if config_override:
            self.config = config_override  # Use provided config for tests
        else:
            self.config = self._load_config(config_path)

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if value is not None:
                self.config[key] = value

        # Always initialize all components - no TEST_MODE branching!
        logger.info("Enhanced Research Agent initializing with full components")

        # Initialize event emitter for detailed progress tracking
        self.event_emitter = initialize_event_emitter(
            stream_emitter=stream_emitter,
            max_events_per_second=self.config.get("events", {}).get(
                "max_per_second", 20
            ),
            batch_events=self.config.get("events", {}).get("batch_events", True),
            batch_size=self.config.get("events", {}).get("batch_size", 3),
            batch_timeout_ms=self.config.get("events", {}).get("batch_timeout_ms", 200),
        )

        # Initialize memory monitor with 2GB limit for safety
        self.memory_monitor = get_memory_monitor(memory_limit_mb=2048)

        # Initialize config manager and agent config early
        # Convert Pydantic model to dict for agent consumption while preserving structure
        if hasattr(self.config, 'model_dump'):
            # Pydantic v2 style - preserve all nested structures
            self.agent_config = self.config.model_dump(mode='python', exclude_none=False)
            logger.info(f"Converted Pydantic config to dict with keys: {list(self.agent_config.keys())}")
        else:
            # Already a dict
            self.agent_config = self.config
            logger.info(f"Using dict config with keys: {list(self.agent_config.keys()) if isinstance(self.agent_config, dict) else 'non-dict'}")
        
        # Create a simple config manager for compatibility
        from deep_research_agent.core.unified_config import get_config_manager

        self.config_manager = get_config_manager(override_config=self.agent_config)

        # Initialize tool registry - use provided or create new
        if tool_registry:
            self.tool_registry = tool_registry
        else:
            self.tool_registry = create_tool_registry(self.config_manager)

        # Initialize LLM - use provided or create new
        if llm:
            self.llm = llm
        else:
            # Initialize LLM using factory pattern for clean dependency injection
            from deep_research_agent.core.llm_factory import create_llm

            try:
                # Use factory to create LLM, allowing test injection via registry
                self.llm = create_llm(self.config)
                if self.llm is None:
                    raise ValueError("LLM factory returned None")
                logger.info("Successfully initialized LLM via factory")
            except Exception as e:
                logger.error(f"Failed to initialize LLM via factory: {e}")
                # Fail fast rather than continuing with None
                raise RuntimeError(f"LLM initialization is required but failed: {e}")

        # Initialize semaphore early (before workflow nodes)
        self.search_semaphore = asyncio.Semaphore(
            self.config.get("rate_limiting", {}).get("max_parallel_requests", 10)
        )

        # Create workflow nodes (after config_manager, tool_registry and semaphore are set)
        self.workflow_nodes = EnhancedWorkflowNodes(self)

        # Store graph builder for thread-local compilation instead of shared instance
        self._graph_builder = self._build_graph
        self._thread_local = threading.local()
        # Note: self.graph is now a property that returns thread-local instances

        logger.info("Enhanced Research Agent initialized with full components")

    @property
    def graph(self):
        """Get or create thread-local graph instance for thread safety."""
        if not hasattr(self._thread_local, "graph"):
            logger.debug(
                f"Creating new graph instance for thread {threading.current_thread().name}"
            )
            self._thread_local.graph = self._graph_builder()
        return self._thread_local.graph

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration using the new ConfigLoader."""
        try:
            from deep_research_agent.config_loader import ConfigLoader

            if config_path:
                # Load with custom config path
                logger.info(f"Loading configuration from: {config_path}")
                config = ConfigLoader.load_agent(config_path=Path(config_path))
            else:
                # Load with auto-detection
                config = ConfigLoader.load_agent()

            # Convert Pydantic model to dict for backward compatibility (Pydantic v2)
            return config.model_dump()

        except Exception as e:
            logger.error(f"Failed to load configuration with ConfigLoader: {e}")
            # No legacy fallback - force proper config system usage
            raise ConfigurationError(
                f"Configuration loading failed: {e}. Please check config files in conf/ directory."
            )

    def _load_legacy_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Legacy configuration loading for backward compatibility."""
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            # Use default configuration
            config = {
                "multi_agent": {"enabled": True},
                "planning": {
                    "enable_iterative_planning": True,
                    "max_plan_iterations": 3,
                    "plan_quality_threshold": 0.7,
                    "auto_accept_plan": True,
                },
                "background_investigation": {"enabled": True},
                "grounding": {"enabled": True, "verification_level": "moderate"},
                "report": {"default_style": "default"},
                "reflexion": {"enabled": True},
                # Controls maximum number of graph transitions before forcing a stop
                # Note: This is a low-level guardrail across ALL node transitions, not just research loops.
                # Keep high (e.g., 100) and rely on high-level loop caps for behavior.
                "recursion_limit": int(self.config.get("workflow", {}).get("recursion_limit", 200)),
            }

        return config

    def _calculate_progress(self, node_name: str) -> int:
        """Calculate progress percentage based on current node."""
        try:
            if node_name in NODE_ORDER:
                current_index = NODE_ORDER.index(node_name)
                # Calculate progress as percentage (0-100)
                progress = int(((current_index + 1) / len(NODE_ORDER)) * 100)
                return min(progress, 100)
            return 0
        except (ValueError, ZeroDivisionError):
            return 0

    def _emit_phase_marker(
        self, node_name: str, item_id: str, start_time: float = None, **extra_meta
    ) -> Optional[ResponsesAgentStreamEvent]:
        """
        Emit PHASE marker with comprehensive META fields.

        Args:
            node_name: Name of the current workflow node
            item_id: Stream item ID
            start_time: Request start time for elapsed calculation
            **extra_meta: Additional metadata to include
        """
        if node_name not in PHASE_MAPPING:
            return None

        phase = PHASE_MAPPING[node_name]
        progress = self._calculate_progress(node_name)

        # Calculate elapsed time from request start
        elapsed = 0.0
        if start_time is not None:
            elapsed = time.time() - start_time

        # Build META fields
        meta_parts = [
            f"[META:node:{node_name}]",
            f"[META:progress:{progress}]",
            f"[META:elapsed:{elapsed:.1f}]",
        ]

        # Add any extra metadata
        for key, value in extra_meta.items():
            if isinstance(value, (int, float)):
                meta_parts.append(f"[META:{key}:{value}]")
            else:
                meta_parts.append(f"[META:{key}:{str(value)}]")

        # Format: [PHASE:X] Description [META:...] [META:...] ...
        delta = f"[PHASE:{phase}] {' '.join(meta_parts)}"

        return ResponsesAgentStreamEvent(
            type="response.output_text.delta", item_id=item_id, delta=delta
        )

    def _build_graph(self) -> Any:
        """Build the LangGraph workflow."""

        # Create state graph with proper state schema
        # Use EnhancedResearchState to define all valid state fields
        workflow = StateGraph(EnhancedResearchState)

        # Add nodes
        workflow.add_node("coordinator", self.workflow_nodes.coordinator_node)
        workflow.add_node(
            "background_investigation",
            self.workflow_nodes.background_investigation_node,
        )
        workflow.add_node("planner", self.workflow_nodes.planner_node)
        workflow.add_node("researcher", self.workflow_nodes.researcher_node)
        workflow.add_node("fact_checker", self.workflow_nodes.fact_checker_node)
        workflow.add_node("reporter", self.workflow_nodes.reporter_node)
        workflow.add_node("human_feedback", self.workflow_nodes.human_feedback_node)

        # Set entry point
        workflow.set_entry_point("coordinator")

        # Add conditional edges for coordinator
        def coordinator_router(state):
            """Route from coordinator based on state."""
            total_steps = state.get("total_workflow_steps", 0)
            logger.info(f"[ROUTER] coordinator_router - step {total_steps}, research_loops={state.get('research_loops', 0)}, fact_check_loops={state.get('fact_check_loops', 0)}")
            
            # Check if we have a valid research topic to proceed
            if not state.get("research_topic"):
                logger.warning("[ROUTER] No research topic found, ending workflow")
                return END
            
            # Normal progression: coordinator -> background_investigation or planner
            if state.get("enable_background_investigation", True):
                # Check if background investigation was already completed
                if state.get("background_investigation_completed", False):
                    return "planner"
                return "background_investigation"
            return "planner"

        workflow.add_conditional_edges(
            "coordinator",
            coordinator_router,
            {
                "background_investigation": "background_investigation",
                "planner": "planner",
                END: END,
            },
        )

        # Background investigation always goes to planner
        workflow.add_edge("background_investigation", "planner")

        # Add conditional edges for planner
        def planner_router(state):
            """Route from planner based on plan and settings."""
            plan = state.get("current_plan")
            if not plan:
                return END

            if state.get("enable_human_feedback") and not state.get("auto_accept_plan"):
                return "human_feedback"

            if hasattr(plan, "has_enough_context") and plan.has_enough_context:
                return "reporter"

            return "researcher"

        workflow.add_conditional_edges(
            "planner",
            planner_router,
            {
                "researcher": "researcher",
                "human_feedback": "human_feedback",
                "reporter": "reporter",
                END: END,
            },
        )

        # Human feedback routing
        def human_feedback_router(state):
            """Route from human feedback."""
            # Could go back to planner or proceed
            return "researcher"

        workflow.add_conditional_edges(
            "human_feedback",
            human_feedback_router,
            {"planner": "planner", "researcher": "researcher"},
        )

        # Researcher routing
        def researcher_router(state):
            """Route from researcher based on step completion."""
            plan = state.get("current_plan")
            
            # Enhanced diagnostic logging
            if plan and hasattr(plan, "steps"):
                from deep_research_agent.core.plan_models import StepStatus
                
                # Count steps by status
                status_counts = {}
                for step in plan.steps:
                    status = step.status.value if hasattr(step.status, 'value') else str(step.status)
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                total_steps = len(plan.steps)
                
                # Check both lowercase and uppercase variants since the enum might use either
                completed = status_counts.get('completed', 0) + status_counts.get('COMPLETED', 0)
                pending = status_counts.get('pending', 0) + status_counts.get('PENDING', 0) 
                failed = status_counts.get('failed', 0) + status_counts.get('FAILED', 0)
                
                logger.info(f"[ROUTER] Plan Status: {completed}/{total_steps} completed, {pending} pending, {failed} failed")
                
                # Log section information if available
                if hasattr(plan, 'section_specifications') and plan.section_specifications:
                    logger.info(f"[ROUTER] Plan has {len(plan.section_specifications)} sections")
            
            # Circuit breaker for infinite loops
            researcher_loops = state.get("researcher_loops", 0)
            max_researcher_loops = state.get("max_researcher_loops", 5)
            
            logger.info(f"[ROUTER] Researcher loops: {researcher_loops}/{max_researcher_loops}")
            
            if researcher_loops >= max_researcher_loops:
                logger.warning(f"[ROUTER] CIRCUIT BREAKER: Max researcher loops reached ({researcher_loops}/{max_researcher_loops})")
                state["researcher_loops"] = 0  # Reset counter
                return "fact_checker" if state.get("enable_grounding", True) else "reporter"
            
            # Step-based completion check with enhanced logging
            if plan and hasattr(plan, "get_next_step"):
                try:
                    next_step = plan.get_next_step()
                    if next_step is None:
                        logger.info("[ROUTER] ✓ All steps complete - proceeding to fact checker")
                        state["researcher_loops"] = 0  # Reset counter
                        return "fact_checker" if state.get("enable_grounding", True) else "reporter"
                    else:
                        # Log detailed next step information
                        step_info = f"{next_step.step_id} - {next_step.title[:50]}"
                        if hasattr(next_step, 'metadata') and next_step.metadata:
                            section_title = next_step.metadata.get('section_title', 'N/A')
                            step_info += f" (section: {section_title})"
                        
                        logger.info(f"[ROUTER] → Routing to specific step: {step_info}")
                        
                        # CRITICAL FIX: Set current_step in state to communicate with researcher
                        state["current_step"] = next_step
                        
                        # Increment loop counter
                        state["researcher_loops"] = researcher_loops + 1
                        return "researcher"
                        
                except Exception as e:
                    logger.warning(f"[ROUTER] Error checking step completion: {e}")
                    # Fallback: check if we have section results
                    section_results = state.get("section_research_results", {})
                    if section_results:
                        logger.info("[ROUTER] Found section results - assuming complete")
                        return "fact_checker" if state.get("enable_grounding", True) else "reporter"
            else:
                # No plan - should not happen but handle gracefully
                logger.warning("[ROUTER] No plan found - assuming complete")
                return "fact_checker" if state.get("enable_grounding", True) else "reporter"
            
            # Continue researching - increment loop counter
            state["researcher_loops"] = researcher_loops + 1
            logger.info(f"[ROUTER] Continuing research (loop {state['researcher_loops']}/{max_researcher_loops})")
            return "researcher"

        workflow.add_conditional_edges(
            "researcher",
            researcher_router,
            {
                "fact_checker": "fact_checker",
                "reporter": "reporter",
                "researcher": "researcher",  # Continue research
                "planner": "planner",  # Re-plan if needed
            },
        )

        # Import routing policy for bulletproof routing decisions
        from .core.routing_policy import determine_next_node, TerminationReason

        # Fact checker routing - now bulletproof with centralized policy
        def fact_checker_router(state):
            """Route from fact checker using centralized routing policy."""
            total_steps = state.get("total_workflow_steps", 0)
            fact_check_loops = state.get("fact_check_loops", 0)
            max_fact_check_loops = state.get("max_fact_check_loops", 2)
            
            logger.info(f"[ROUTER] fact_checker_router - step {total_steps}, fact_check_loops={fact_check_loops}, max_fact_check_loops={max_fact_check_loops}")

            # CRITICAL: Add state validation to prevent data loss
            try:
                StateValidator.validate_state_for_agent('fact_checker', state)
            except ValueError as e:
                logger.error(f"Fact checker router: State validation failed: {e}")
                # Continue but flag the issue
                state.setdefault('warnings', []).append(f"State validation failed in fact_checker_router: {e}")

            # FIX: Hard stop when fact check loops are exhausted
            if fact_check_loops >= max_fact_check_loops:
                logger.warning(f"HARD STOP: Fact check loops exhausted ({fact_check_loops}/{max_fact_check_loops}), routing to reporter")
                # Record transition for debugging
                global_propagation_tracker.record_transition(
                    from_agent='fact_checker_router',
                    to_agent='reporter',
                    state_snapshot=state
                )
                return "reporter"
            
            # Check total steps limit
            max_total_steps = state.get("max_total_steps", 20)
            if total_steps >= max_total_steps:
                logger.warning(f"HARD STOP: Total steps exhausted ({total_steps}/{max_total_steps}), routing to reporter")
                # Record transition for debugging
                global_propagation_tracker.record_transition(
                    from_agent='fact_checker_router',
                    to_agent='reporter',
                    state_snapshot=state
                )
                return "reporter"

            # Get factuality report from state
            factuality_report = state.get("factuality_report")
            if not factuality_report:
                # Fallback if no report available
                logger.warning("No factuality report in state - defaulting to reporter")
                global_propagation_tracker.record_transition(
                    from_agent='fact_checker_router',
                    to_agent='reporter',
                    state_snapshot=state
                )
                return "reporter"

            # Use centralized routing policy
            next_node, reasoning = determine_next_node(state, factuality_report)

            # Log the routing decision for observability
            logger.info(
                f"Fact checker routing decision: {next_node}",
                reasoning=reasoning,
                fact_check_loops=state.get("fact_check_loops", 0),
                max_fact_check_loops=state.get("max_fact_check_loops", 2),
                factuality_score=factuality_report.overall_factuality_score,
                total_steps=state.get("total_workflow_steps", 0),
            )

            # Store routing decision in state for audit trail
            if "routing_decisions" not in state:
                state["routing_decisions"] = []
            state["routing_decisions"].append(
                {
                    "from": "fact_checker",
                    "to": next_node,
                    "reasoning": reasoning,
                    "timestamp": time.time(),
                    "fact_check_loops": state.get("fact_check_loops", 0),
                    "factuality_score": factuality_report.overall_factuality_score,
                }
            )

            # Record state transition for debugging
            global_propagation_tracker.record_transition(
                from_agent='fact_checker_router',
                to_agent=next_node,
                state_snapshot=state
            )

            return next_node

        workflow.add_conditional_edges(
            "fact_checker",
            fact_checker_router,
            {"reporter": "reporter", "researcher": "researcher", "planner": "planner"},
        )

        # Reporter ends the workflow
        workflow.add_edge("reporter", END)

        # Compile with checkpointing
        # Use NoOpCheckpointSaver to reduce memory usage by default in production
        # Always use NoOpCheckpointer in production (Databricks runtime) for memory optimization
        is_databricks_runtime = os.getenv("DATABRICKS_RUNTIME_VERSION") is not None
        use_noop_checkpointer = (
            os.getenv("DISABLE_CHECKPOINTER", "true").lower() == "true" or
            is_databricks_runtime  # Always use NoOpCheckpointer in Databricks runtime
        )

        if use_noop_checkpointer:
            checkpointer = NoOpCheckpointSaver()
            logger.info("Using NoOpCheckpointSaver - memory usage will be reduced")
        else:
            checkpointer = MemorySaver()
            logger.info("Using MemorySaver for full checkpoint functionality")

        return workflow.compile(checkpointer=checkpointer)

    async def research(
        self,
        query: str,
        report_style: Optional[ReportStyle] = None,
        verification_level: Optional[VerificationLevel] = None,
        enable_streaming: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute research on a query.

        Args:
            query: Research question or topic
            report_style: Style for the final report
            verification_level: Strictness of fact checking
            enable_streaming: Whether to stream results
            **kwargs: Additional configuration overrides

        Returns:
            Research results including final report and metadata
        """
        logger.info(f"Starting research for: {query}")

        # Initialize state
        initial_state = StateManager.initialize_state(
            research_topic=query, config=self.config
        )

        # Apply overrides
        if report_style:
            initial_state["report_style"] = report_style
        if verification_level:
            initial_state["verification_level"] = verification_level

        # Add user message
        initial_state["messages"] = [HumanMessage(content=query)]

        # Configure from kwargs
        for key, value in kwargs.items():
            if value is not None:
                initial_state[key] = value

        # Execute workflow
        if enable_streaming:
            # Return an async generator directly for the caller to iterate over
            return self._research_streaming(initial_state)
        # Batch mode returns a regular dictionary after awaiting
        return await self._research_batch(initial_state)

    async def _research_batch(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research in batch mode."""
        try:
            # Run the graph
            config = {
                "configurable": {"thread_id": "research_thread"},
                "recursion_limit": int(self.config.get("workflow", {}).get("recursion_limit", 200)),
            }
            final_state = await self.graph.ainvoke(initial_state, config)

            # Extract results
            return self._extract_results(final_state)

        except SearchToolsFailedException as e:
            logger.error(f"Search tools failed: {e.message}")
            error_message = f"Research could not be completed due to search tool failures.\n\nDetails:\n{e.message}"
            if not e.details.get("has_brave_key", True):
                error_message += (
                    "\n\n💡 Tip: Check that BRAVE_API_KEY is properly configured."
                )
            if e.details.get("failed_tools"):
                error_message += (
                    f"\n\nFailed tools: {', '.join(e.details['failed_tools'])}"
                )

            return {
                "success": False,
                "error": "search_tools_failed",
                "error_message": error_message,
                "final_report": error_message,
                "details": e.details,
            }
        except Exception as e:
            logger.error(f"Research failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "final_report": f"Research failed: {str(e)}",
            }

    async def _research_streaming(
        self, initial_state: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute research in streaming mode."""
        try:
            config = {
                "configurable": {"thread_id": "research_thread"},
                "recursion_limit": int(self.config.get("workflow", {}).get("recursion_limit", 200)),
            }

            # Stream events from the graph
            async for event in self.graph.astream_events(initial_state, config):
                yield self._process_stream_event(event)

        except SearchToolsFailedException as e:
            logger.error(f"Search tools failed: {e.message}")
            error_message = f"Research could not be completed due to search tool failures.\n\nDetails:\n{e.message}"
            if not e.details.get("has_brave_key", True):
                error_message += (
                    "\n\n💡 Tip: Check that BRAVE_API_KEY is properly configured."
                )
            if e.details.get("failed_tools"):
                error_message += (
                    f"\n\nFailed tools: {', '.join(e.details['failed_tools'])}"
                )

            yield {
                "type": "error",
                "content": error_message,
                "error_type": "search_tools_failed",
                "details": e.details,
            }
        except Exception as e:
            # COMPREHENSIVE ERROR LOGGING for streaming research failures
            import traceback
            stack_trace = traceback.format_exc()
            
            logger.error(
                f"STREAMING RESEARCH FAILED: {type(e).__name__}: {e}\n"
                f"Full stack trace:\n{stack_trace}\n"
                f"This occurred during graph.astream_events execution"
            )
            
            yield {
                "type": "error", 
                "content": f"Error processing request: {str(e)}",
                "error_details": {
                    "error_type": type(e).__name__,
                    "stack_trace": stack_trace
                }
            }

    def _extract_results(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract final results from state."""
        return {
            "success": True,
            "final_report": state.get("final_report", ""),
            "research_topic": state.get("research_topic", ""),
            "report_style": state.get("report_style", ReportStyle.PROFESSIONAL),
            "factuality_score": state.get("factuality_score", None),
            "confidence_score": state.get("confidence_score", None),
            "citations": state.get("citations", []),
            "observations": state.get("observations", []),
            "plan": state.get("current_plan", None),
            "grounding_results": state.get("grounding_results", []),
            "contradictions": state.get("contradictions", []),
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
            "reflections": state.get("reflections", []),
            "metadata": {
                "total_duration": state.get("total_duration_seconds"),
                "plan_iterations": state.get("plan_iterations", 0),
                "research_loops": len(state.get("completed_steps", [])),
                "total_sources": len(state.get("citations", [])),
            },
        }

    def _process_stream_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a streaming event with enhanced detail tracking."""
        try:
            # Extract event information
            event_type = event.get("event", "unknown")
            event_name = event.get("name", "")
            event_data = event.get("data", {})

            # Emit detailed intermediate event based on the LangGraph event
            correlation_id = event.get("run_id", "default")

            # Map LangGraph events to detailed progress events
            if "on_chain_start" in event_type or "on_llm_start" in event_type:
                # Starting an agent or LLM call
                agent_name = self._extract_agent_from_event(event_name)
                self.event_emitter.emit(
                    IntermediateEventType.AGENT_HANDOFF,
                    {
                        "current_agent": agent_name,
                        "action": f"Starting {agent_name} processing",
                        "stage": event_name,
                    },
                    correlation_id=correlation_id,
                )

                return {
                    "type": "agent_start",
                    "content": f"🎯 {agent_name} starting...",
                    "metadata": {"agent": agent_name, "stage": event_name},
                }

            elif "on_llm_stream" in event_type:
                # LLM streaming content
                chunk = event_data.get("chunk", {})
                content = (
                    chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
                )
                agent_name = self._extract_agent_from_event(event_name)

                if content.strip():
                    self.event_emitter.emit(
                        IntermediateEventType.SYNTHESIS_PROGRESS,
                        {
                            "agent": agent_name,
                            "content_preview": content[:100] + "..."
                            if len(content) > 100
                            else content,
                            "progress_type": "llm_generation",
                        },
                        correlation_id=correlation_id,
                    )

                return {
                    "type": "llm_streaming",
                    "content": content,
                    "metadata": {"agent": agent_name, "is_streaming": True},
                }

            elif "on_tool_start" in event_type:
                # Tool execution started
                tool_name = event_data.get("name", "unknown_tool")
                tool_input = event_data.get("input", {})

                # Extract search query if it's a search tool
                query = ""
                if "search" in tool_name.lower():
                    query = tool_input.get("query", "") or str(tool_input)[:100]

                self.event_emitter.emit(
                    IntermediateEventType.TOOL_CALL_START,
                    {
                        "tool_name": tool_name,
                        "parameters": {"query": query} if query else tool_input,
                        "action": f"Executing {tool_name}",
                    },
                    correlation_id=correlation_id,
                )

                return {
                    "type": "tool_start",
                    "content": f"🔍 Searching: {query}"
                    if query
                    else f"🛠️ Using {tool_name}",
                    "metadata": {"tool": tool_name, "query": query},
                }

            elif "on_tool_end" in event_type:
                # Tool execution completed
                tool_name = event_data.get("name", "unknown_tool")
                output = event_data.get("output", {})

                # Count results if it's search output
                result_count = 0
                if isinstance(output, list):
                    result_count = len(output)
                elif isinstance(output, dict) and "results" in output:
                    result_count = len(output.get("results", []))

                self.event_emitter.emit(
                    IntermediateEventType.TOOL_CALL_COMPLETE,
                    {
                        "tool_name": tool_name,
                        "success": True,
                        "results_count": result_count,
                        "result_summary": f"Found {result_count} results"
                        if result_count > 0
                        else "Completed successfully",
                    },
                    correlation_id=correlation_id,
                )

                return {
                    "type": "tool_complete",
                    "content": f"✓ Found {result_count} results"
                    if result_count > 0
                    else f"✓ {tool_name} completed",
                    "metadata": {"tool": tool_name, "result_count": result_count},
                }

            elif "on_chain_end" in event_type:
                # Agent completed
                agent_name = self._extract_agent_from_event(event_name)

                self.event_emitter.emit(
                    IntermediateEventType.ACTION_COMPLETE,
                    {
                        "agent": agent_name,
                        "action": f"{agent_name} completed",
                        "stage": event_name,
                    },
                    correlation_id=correlation_id,
                )

                return {
                    "type": "agent_complete",
                    "content": f"✓ {agent_name} completed",
                    "metadata": {"agent": agent_name, "stage": event_name},
                }

            # Handle data-specific events
            if isinstance(event_data, dict):
                if "observation" in event_data:
                    return {
                        "type": "observation",
                        "content": event_data["observation"],
                        "metadata": {"phase": "research"},
                    }
                elif "plan" in event_data:
                    self.event_emitter.emit(
                        IntermediateEventType.PLAN_CREATED,
                        {
                            "description": "Research plan created",
                            "step_count": len(event_data["plan"].steps)
                            if hasattr(event_data["plan"], "steps")
                            and event_data["plan"].steps
                            else (
                                len(event_data["plan"].get("steps", []))
                                if isinstance(event_data["plan"], dict)
                                else 0
                            ),
                        },
                        correlation_id=correlation_id,
                    )
                    return {
                        "type": "plan",
                        "content": event_data["plan"],
                        "metadata": {"phase": "planning"},
                    }
                elif "final_report" in event_data:
                    return {
                        "type": "final_report",
                        "content": event_data["final_report"],
                        "metadata": {"phase": "complete"},
                    }

            # Fallback for unhandled events
            return {
                "type": "progress",
                "content": f"Processing: {event_type}",
                "metadata": {"raw_event": event_type},
            }

        except Exception as e:
            logger.warning(f"Error processing stream event: {str(e)}")
            return {"type": "event", "content": event}

    def _extract_agent_from_event(self, event_name: str) -> str:
        """Extract agent name from LangGraph event name."""
        if "coordinator" in event_name.lower():
            return "Coordinator"
        elif "planner" in event_name.lower():
            return "Planner"
        elif "researcher" in event_name.lower():
            return "Researcher"
        elif "fact_checker" in event_name.lower():
            return "Fact Checker"
        elif "reporter" in event_name.lower():
            return "Reporter"
        elif "background" in event_name.lower():
            return "Background Investigation"
        else:
            return "Research Agent"

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """
        Synchronous prediction for MLflow ResponsesAgent interface.
        Routes through multi-agent workflow.
        """
        # Extract last message as query from input format
        query = ""
        if hasattr(request, "messages") and request.messages:
            # Old format with messages attribute
            query = request.messages[-1].content
        elif hasattr(request, "input") and request.input:
            # New format with input list
            for item in reversed(request.input):
                # Handle both dict and Message object formats
                if hasattr(item, "get"):
                    # Dict format
                    if item.get("type") == "message" and item.get("role") == "user":
                        query = item.get("content", "")
                        break
                elif hasattr(item, "role") and hasattr(item, "content"):
                    # ChatMessage object format (direct MLflow input)
                    if item.role == "user":
                        query = item.content
                        break
                else:
                    # Legacy Message object format
                    if (
                        hasattr(item, "type")
                        and hasattr(item, "role")
                        and hasattr(item, "content")
                    ):
                        if item.type == "message" and item.role == "user":
                            query = item.content
                            break
        elif isinstance(request, list):
            # Direct list input (MLflow schema compatibility)
            for item in reversed(request):
                if hasattr(item, "role") and hasattr(item, "content"):
                    if item.role == "user":
                        query = item.content
                        break
        
        # Additional debugging for message extraction
        if not query:
            logger.warning(f"No query extracted from request - agents may fail to find user message")
        else:
            logger.info(f"Successfully extracted query: {query[:100]}...")

        logger.info(f"Processing predict request: {query[:100]}...")

        # No mock responses - always run real workflow
        # Run async research in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.research(query))

            # Convert result to ResponsesAgentResponse format
            final_report = result.get("final_report", "No report generated")

            # Create response item
            output_item = {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": final_report}],
                "id": str(uuid4()),
            }

            # Add custom outputs with grounding report
            custom_outputs = {}
            if result.get("grounding_results"):
                custom_outputs["grounding"] = {
                    "factuality_score": result.get("factuality_score", 0),
                    "claims_verified": len(result.get("grounding_results", [])),
                    "verification_level": result.get("verification_level", "moderate"),
                }

            if result.get("citations"):
                custom_outputs["citations"] = result["citations"]

            logger.info("Successfully completed predict request")

            return ResponsesAgentResponse(
                output=[output_item], custom_outputs=custom_outputs
            )
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            # Return error response
            error_item = {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": f"Error processing request: {str(e)}",
                    }
                ],
                "id": str(uuid4()),
            }
            return ResponsesAgentResponse(
                output=[error_item], custom_outputs={"error": str(e)}
            )
        finally:
            loop.close()

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Stream responses with intermediate events for real-time UI updates.
        Routes through multi-agent workflow with full event emission.
        """
        # Extract last message as query from input format
        query = ""
        if hasattr(request, "messages") and request.messages:
            # Old format with messages attribute
            query = request.messages[-1].content
        elif hasattr(request, "input") and request.input:
            # New format with input list
            for item in reversed(request.input):
                # Handle both dict and Message object formats
                if hasattr(item, "get"):
                    # Dict format
                    if item.get("type") == "message" and item.get("role") == "user":
                        query = item.get("content", "")
                        break
                elif hasattr(item, "role") and hasattr(item, "content"):
                    # ChatMessage object format (direct MLflow input)
                    if item.role == "user":
                        query = item.content
                        break
                else:
                    # Legacy Message object format
                    if (
                        hasattr(item, "type")
                        and hasattr(item, "role")
                        and hasattr(item, "content")
                    ):
                        if item.type == "message" and item.role == "user":
                            query = item.content
                            break
        elif isinstance(request, list):
            # Direct list input (MLflow schema compatibility)
            for item in reversed(request):
                if hasattr(item, "role") and hasattr(item, "content"):
                    if item.role == "user":
                        query = item.content
                        break
        
        # Additional debugging for message extraction
        if not query:
            logger.warning(f"No query extracted from request - agents may fail to find user message")
        else:
            logger.info(f"Successfully extracted query: {query[:100]}...")

        item_id = str(uuid4())

        # CRITICAL: Clear any residual state from previous queries to prevent contamination
        self._clear_previous_state()
        
        # Generate query fingerprint for isolation
        query_hash = self._generate_query_hash(query)
        logger.info(f"Processing predict_stream request: {query[:100]}...")
        logger.info(f"Starting fresh query session: {query_hash}")

        # No mock streaming - always run real workflow

        # Import workflow limit initializer
        from .core.routing_policy import initialize_workflow_limits

        # Create async loop for streaming
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Initialize state
            initial_state = StateManager.initialize_state(
                research_topic=query, config=self.config
            )

            # Add user message to state - this is critical for agent coordination
            from langchain_core.messages import HumanMessage

            if query:
                initial_state["messages"] = [HumanMessage(content=query)]
            else:
                logger.warning(
                    "No query extracted from request - agents may fail to find user message"
                )

            # Initialize workflow limits and counters
            initialize_workflow_limits(initial_state, self.config)

            # Add run ID and session ID for tracking
            initial_state["run_id"] = str(uuid4())
            initial_state["session_id"] = getattr(
                request, "session_id", initial_state["run_id"]
            )

            # Log initialization for debugging
            logger.info(
                f"Initialized workflow limits",
                max_fact_check_loops=initial_state.get("max_fact_check_loops"),
                max_total_steps=initial_state.get("max_total_steps"),
                max_wall_clock_seconds=initial_state.get("max_wall_clock_seconds"),
            )

            # Monitor memory before workflow execution
            if hasattr(self, "memory_monitor"):
                self.memory_monitor.log_memory_usage("before_workflow", initial_state)

            # Track start time for elapsed calculations
            start_time = time.time()

            # Stream through workflow with event capture
            collected_content = []

            async def run_streaming():
                # Add required config for LangGraph checkpointer
                # FIX: Ensure recursion_limit is properly set
                recursion_limit = int(self.config.get("workflow", {}).get("recursion_limit", 200))
                # Also check for recursion_limit at root level (for backward compatibility)
                if "recursion_limit" in self.config:
                    recursion_limit = int(self.config["recursion_limit"])
                
                # Log the actual limit being used
                logger.info(f"Using recursion_limit: {recursion_limit}")
                
                config = {
                    "configurable": {
                        "thread_id": str(uuid4()),
                        "checkpoint_ns": "enhanced_research_agent",
                    },
                    "recursion_limit": recursion_limit,
                }
                async for event in self.graph.astream(initial_state, config=config):
                    yield event

            # Process events from async generator
            event_generator = run_streaming()

            try:
                while True:
                    event = loop.run_until_complete(event_generator.__anext__())

                    # Process LangGraph events
                    if isinstance(event, dict):
                        for node_name, node_data in event.items():
                            # Defensive check: Skip if node returned None
                            if node_data is None:
                                logger.warning(
                                    f"Node {node_name} returned None, skipping"
                                )
                                continue

                            # Emit intermediate events from event emitter
                            if self.event_emitter and hasattr(
                                self, "_pending_intermediate_events"
                            ):
                                for pending in self._pending_intermediate_events:
                                    yield pending
                                self._pending_intermediate_events.clear()

                            # Handle different node outputs with progress updates
                            if node_name == "coordinator":
                                # Monitor memory after coordinator
                                if hasattr(self, "memory_monitor"):
                                    self.memory_monitor.log_memory_usage(
                                        "after_coordinator", node_data
                                    )

                                # Emit PHASE marker first
                                phase_event = self._emit_phase_marker(
                                    node_name, item_id, start_time
                                )
                                if phase_event:
                                    yield phase_event

                                yield ResponsesAgentStreamEvent(
                                    type="response.output_text.delta",
                                    item_id=item_id,
                                    delta="🎯 Analyzing request and routing to appropriate agents...\n",
                                )

                            elif node_name == "background_investigation":
                                investigations = node_data.get(
                                    "background_investigation_results", ""
                                )
                                if investigations:
                                    # Emit PHASE marker first
                                    phase_event = self._emit_phase_marker(
                                        node_name, item_id, start_time
                                    )
                                    if phase_event:
                                        yield phase_event

                                    yield ResponsesAgentStreamEvent(
                                        type="response.output_text.delta",
                                        item_id=item_id,
                                        delta="🔍 Gathering background information...\n",
                                    )

                            elif node_name == "planner":
                                plan = node_data.get("current_plan")
                                if plan:
                                    # Emit PHASE marker first
                                    phase_event = self._emit_phase_marker(
                                        node_name, item_id, start_time
                                    )
                                    if phase_event:
                                        yield phase_event

                                    steps_count = (
                                        len(plan.steps) if hasattr(plan, "steps") else 0
                                    )
                                    yield ResponsesAgentStreamEvent(
                                        type="response.output_text.delta",
                                        item_id=item_id,
                                        delta=f"📋 Created research plan with {steps_count} steps\n",
                                    )

                            elif node_name == "researcher":
                                # Monitor memory after researcher (most memory-intensive node)
                                if hasattr(self, "memory_monitor"):
                                    self.memory_monitor.log_memory_usage(
                                        "after_researcher", node_data
                                    )
                                    # Force garbage collection if memory is getting high
                                    if not self.memory_monitor.check_memory_limit(
                                        "researcher"
                                    ):
                                        self.memory_monitor.force_garbage_collection(
                                            "researcher_cleanup"
                                        )

                                observations = node_data.get("observations", [])
                                if observations:
                                    # Emit PHASE marker first
                                    phase_event = self._emit_phase_marker(
                                        node_name, item_id, start_time
                                    )
                                    if phase_event:
                                        yield phase_event

                                    yield ResponsesAgentStreamEvent(
                                        type="response.output_text.delta",
                                        item_id=item_id,
                                        delta=f"🔍 Research in progress - found {len(observations)} key findings\n",
                                    )

                            elif node_name == "fact_checker":
                                score = node_data.get("factuality_score", 0)
                                if score is not None and score > 0:
                                    # Emit PHASE marker first
                                    phase_event = self._emit_phase_marker(
                                        node_name, item_id, start_time
                                    )
                                    if phase_event:
                                        yield phase_event

                                    yield ResponsesAgentStreamEvent(
                                        type="response.output_text.delta",
                                        item_id=item_id,
                                        delta=f"✅ Factuality verification complete - score: {score:.1%}\n",
                                    )

                            elif node_name == "reporter":
                                # Enhanced reporter content extraction with comprehensive logging
                                report = ""

                                # Primary path: node_data is the full state dict with final_report
                                if isinstance(node_data, dict):
                                    report = node_data.get("final_report", "")
                                    if not report:
                                        # Try alternative locations
                                        report = node_data.get("report", "")
                                        report = report or node_data.get("content", "")
                                        # Log available keys for debugging
                                        logger.warning(
                                            f"No final_report in reporter output. Available keys: {list(node_data.keys())[:10]}"
                                        )
                                    else:
                                        logger.info(
                                            f"Successfully extracted final_report: {len(report)} characters"
                                        )

                                # Defensive path: handle Command if it somehow gets through
                                elif hasattr(node_data, "update") and node_data.update:
                                    logger.warning(
                                        f"Reporter returned Command instead of dict - applying defensive handling"
                                    )
                                    report = node_data.update.get("final_report", "")
                                else:
                                    logger.error(
                                        f"Unexpected reporter node_data type: {type(node_data)}"
                                    )

                                if report:
                                    # Log content stats
                                    logger.info(
                                        f"Reporter generated {len(report)} chars, starting streaming..."
                                    )

                                    # Validate content quality
                                    if len(report) < 100:
                                        logger.warning(
                                            f"Report seems very short: {len(report)} chars"
                                        )

                                    # Emit PHASE marker with content metadata
                                    phase_event = self._emit_phase_marker(
                                        node_name,
                                        item_id,
                                        start_time,
                                        report_length=len(report),
                                    )
                                    if phase_event:
                                        yield phase_event

                                    # Add header
                                    yield ResponsesAgentStreamEvent(
                                        type="response.output_text.delta",
                                        item_id=item_id,
                                        delta="\n📄 **Research Report**\n\n",
                                    )

                                    # Stream in intelligent chunks that preserve structure
                                    chunks = self._smart_chunk_report(report)
                                    for chunk in chunks:
                                        yield ResponsesAgentStreamEvent(
                                            type="response.output_text.delta",
                                            item_id=item_id,
                                            delta=chunk,
                                        )
                                        collected_content.append(chunk)

                                    # Log streaming completion stats
                                    total_streamed = len("".join(collected_content))
                                    logger.info(
                                        f"Streamed {len(chunks)} chunks, total: {total_streamed} chars"
                                    )
                                else:
                                    logger.warning(
                                        f"No report found in reporter output. node_data type: {type(node_data)}"
                                    )

            except StopAsyncIteration:
                # Async generator finished normally
                pass

            # Emit final done event
            final_content = "".join(collected_content)
            if not final_content.strip():
                final_content = "Research completed successfully, but no final report was generated."

            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item={
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": final_content}],
                    "id": item_id,
                },
            )

            logger.info("Successfully completed predict_stream request")

        except Exception as e:
            # COMPREHENSIVE ERROR LOGGING for debugging
            import traceback
            stack_trace = traceback.format_exc()
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "stack_trace": stack_trace
            }
            
            logger.error(
                f"PREDICT_STREAM ERROR: {type(e).__name__}: {e}\n"
                f"Full stack trace:\n{stack_trace}\n"
                f"This error occurred during streaming prediction"
            )
            
            # Emit error event
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item={
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"Error processing request: {str(e)}",
                        }
                    ],
                    "id": item_id,
                },
            )
        finally:
            loop.close()

    def _smart_chunk_report(
        self, report: str, base_chunk_size: int = 1500
    ) -> List[str]:
        """
        Intelligently chunk report preserving markdown structure.

        Critical improvements:
        1. Never break within tables
        2. Prefer breaking at paragraph boundaries
        3. Preserve code blocks intact
        4. Keep headers with following content
        """
        if len(report) <= base_chunk_size:
            return [report]

        chunks = []
        lines = report.split("\n")
        current_chunk = []
        current_size = 0

        # State tracking for structure preservation
        in_table = False
        in_code_block = False

        for i, line in enumerate(lines):
            # Detect structural boundaries
            is_table_line = "|" in line
            is_table_separator = "|" in line and set(line.strip()) <= {"|", "-", " "}
            is_code_fence = line.strip().startswith("```")

            # Update state
            if is_table_separator:
                in_table = True
            elif in_table and not is_table_line:
                in_table = False

            if is_code_fence:
                in_code_block = not in_code_block

            # Determine if we can break here
            can_break = (
                not in_table
                and not in_code_block
                and line.strip() == ""  # Empty line is good break point
                and current_size > base_chunk_size / 2  # Don't create tiny chunks
            )

            # Add line to current chunk
            current_chunk.append(line)
            current_size += len(line) + 1

            # Emit chunk if appropriate
            if can_break and current_size >= base_chunk_size:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_size = 0

        # Add remaining content
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def _clear_previous_state(self):
        """
        Clear any residual state from previous queries to prevent contamination.
        This is critical for ensuring each query starts with a clean slate.
        """
        # Clear any cached state that might persist between requests
        if hasattr(self, '_previous_observations'):
            delattr(self, '_previous_observations')
        
        if hasattr(self, '_previous_search_results'):
            delattr(self, '_previous_search_results')
            
        if hasattr(self, '_previous_citations'):
            delattr(self, '_previous_citations')
            
        # Clear any workflow node state
        if hasattr(self, 'workflow_nodes') and self.workflow_nodes:
            # Reset any cached data in workflow nodes
            for node_name in ['coordinator', 'planner', 'researcher', 'fact_checker', 'reporter']:
                if hasattr(self.workflow_nodes, f'{node_name}_agent'):
                    agent = getattr(self.workflow_nodes, f'{node_name}_agent')
                    if hasattr(agent, '_clear_cache'):
                        agent._clear_cache()
        
        logger.info("Cleared previous query state to prevent contamination")

    def _generate_query_hash(self, query: str) -> str:
        """
        Generate a unique hash for the query to enable session isolation.
        
        Args:
            query: The research query string
            
        Returns:
            str: A unique hash identifying this query session
        """
        import hashlib
        from datetime import datetime
        
        # Create hash from query content + timestamp for uniqueness
        content = f"{query}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def diagnose_data_flow(self) -> Dict[str, Any]:
        """
        Generate a diagnostic report of data flow between agents.
        
        This method uses the global propagation tracker to analyze how data
        flows through the multi-agent system and identify potential issues.
        
        Returns:
            Dict containing diagnostic information about state transitions
        """
        logger.info("=== Generating Data Flow Diagnostic Report ===")
        
        # Generate the diagnostic report from the global tracker
        diagnostic_report = global_propagation_tracker.diagnose_data_loss()
        
        # Add additional system-level diagnostics
        diagnostic_report.update({
            "agent_configuration": {
                "multi_agent_system": True,
                "validation_enabled": True,
                "propagation_tracking": True,
                "critical_fields_monitored": list(StateValidator.CRITICAL_FIELDS),
                "accumulating_fields_monitored": list(StateValidator.ACCUMULATING_FIELDS)
            },
            "validation_status": {
                "state_validator_active": True,
                "agent_contracts_active": True,
                "validated_command_active": True,
                "global_tracker_active": bool(global_propagation_tracker.transitions)
            }
        })
        
        # Log summary
        logger.info(f"Diagnostic report generated with {len(diagnostic_report)} sections")
        if diagnostic_report.get('data_loss_events'):
            logger.warning(f"Found {len(diagnostic_report['data_loss_events'])} data loss events")
        else:
            logger.info("No data loss events detected")
        
        return diagnostic_report


# Example usage
async def main():
    """Example usage of the Enhanced Research Agent."""

    # Initialize agent with configuration
    agent = EnhancedResearchAgent(
        config_path="agent_config_enhanced.yaml",
        enable_grounding=True,
        enable_background_investigation=True,
        default_report_style=ReportStyle.ACADEMIC,
        verification_level=VerificationLevel.STRICT,
    )

    # Execute research
    query = "What are the latest breakthroughs in quantum computing and their implications for cryptography?"

    # Batch mode
    results = await agent.research(
        query=query,
        report_style=ReportStyle.TECHNICAL,
        verification_level=VerificationLevel.STRICT,
        enable_iterative_planning=True,
        enable_reflexion=True,
    )

    # Display results
    print(f"Research Topic: {results['research_topic']}")
    print(f"Report Style: {results['report_style']}")
    print(f"Factuality Score: {results.get('factuality_score', 'N/A')}")
    print(f"Confidence Score: {results.get('confidence_score', 'N/A')}")
    print(f"Total Sources: {results['metadata']['total_sources']}")
    print(f"Plan Iterations: {results['metadata']['plan_iterations']}")
    print("\n" + "=" * 50 + "\n")
    print(results["final_report"])

    # Streaming mode example
    print("\n" + "=" * 50 + "\n")
    print("Streaming mode example:")

    async for event in await agent.research(
        query="Explain the concept of neural networks",
        report_style=ReportStyle.POPULAR_SCIENCE,
        enable_streaming=True,
    ):
        if event["type"] == "observation":
            print(f"📊 Observation: {event['content'][:100]}...")
        elif event["type"] == "plan":
            print(f"📋 Plan update: {event['content']}")
        elif event["type"] == "final_report":
            print(f"📄 Final Report Ready!")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
