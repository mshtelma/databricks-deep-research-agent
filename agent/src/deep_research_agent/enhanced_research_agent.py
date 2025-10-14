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
from datetime import datetime
from typing import Dict, Any, Optional, AsyncIterator, Callable, Generator, List
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_core.runnables import RunnableConfig
from typing import Iterator, Optional, Dict, Any, Sequence, Tuple
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from .core import get_logger
from .core.exceptions import ConfigurationError
from .core.multi_agent_state import (
    EnhancedResearchState,
    StateManager,
)
from .core.report_styles import ReportStyle
from .core.grounding import VerificationLevel
from .core.event_emitter import (
    EventEmitter,
    initialize_event_emitter,
    IntermediateEventType,
)
from .core.memory_monitor import get_memory_monitor
from .core.state_validator import (
    StateValidator,
    global_propagation_tracker,
)
from .core.validated_command import ValidatedCommand
from .workflow_nodes_enhanced import EnhancedWorkflowNodes
from .agents import (
    CoordinatorAgent,
    PlannerAgent,
    ResearcherAgent,
    ReporterAgent,
    FactCheckerAgent,
)
from .components import create_tool_registry
from .core.exceptions import (
    SearchToolsFailedException,
    PermanentWorkflowError,
    AuthenticationError,
)
from .core.content_sanitizer import sanitize_agent_content
from .core.id_generator import PlanIDGenerator


logger = get_logger(__name__)


# Simplified inline NoOpCheckpointSaver (replaces 175-line external file)
class NoOpCheckpointSaver(BaseCheckpointSaver):
    """Memory-optimized no-op checkpointer that doesn't store any data."""
    
    def put(self, config: RunnableConfig, checkpoint, metadata, new_versions=None) -> RunnableConfig:
        return config
    
    def get(self, config: RunnableConfig):
        return None
    
    def get_tuple(self, config: RunnableConfig):
        return None
    
    def list(self, config=None, *, filter=None, before=None, limit=None) -> Iterator:
        return iter([])
    
    def put_writes(self, config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str, task_path: str = ""):
        pass
    
    # Async versions (required by LangGraph)
    async def aput(self, config: RunnableConfig, checkpoint, metadata, new_versions=None) -> RunnableConfig:
        return config
    
    async def aget(self, config: RunnableConfig):
        return None
    
    async def aget_tuple(self, config: RunnableConfig):
        return None
    
    async def alist(self, config=None, *, filter=None, before=None, limit=None) -> Iterator:
        return iter([])
    
    async def aput_writes(self, config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str, task_path: str = ""):
        pass
    
    @property
    def config_specs(self) -> list:
        return []


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
        emitter = initialize_event_emitter(
            stream_emitter=stream_emitter,
            max_events_per_second=self.config.get("events", {}).get(
                "max_per_second", 20
            ),
            batch_events=self.config.get("events", {}).get("batch_events", True),
            batch_size=self.config.get("events", {}).get("batch_size", 3),
            batch_timeout_ms=self.config.get("events", {}).get("batch_timeout_ms", 200),
        )

        self.event_emitter = emitter
        self._pending_intermediate_events = []

        # Store stream_callback for immediate event flushing in tests
        self.stream_emitter = stream_emitter  # CRITICAL FIX: Enable event streaming
        self.stream_callback = stream_emitter

        # Initialize memory monitor with 2GB limit for safety
        self.memory_monitor = get_memory_monitor(memory_limit_mb=2048)

        # Initialize config manager and agent config early
        # Convert Pydantic model to dict for agent consumption while preserving structure
        if hasattr(self.config, "model_dump"):
            # Pydantic v2 style - preserve all nested structures
            self.agent_config = self.config.model_dump(
                mode="python", exclude_none=False
            )
            logger.info(
                f"Converted Pydantic config to dict with keys: {list(self.agent_config.keys())}"
            )
        else:
            # Already a dict
            self.agent_config = self.config
            logger.info(
                f"Using dict config with keys: {list(self.agent_config.keys()) if isinstance(self.agent_config, dict) else 'non-dict'}"
            )

        # Create a simple config manager for compatibility
        from .core.unified_config import get_config_manager

        self.config_manager = get_config_manager(override_config=self.agent_config)

        # Initialize tool registry - use provided or create new
        if tool_registry:
            self.tool_registry = tool_registry
        else:
            self.tool_registry = create_tool_registry(self.config_manager)

        # Initialize LLM and model manager - use provided or create new
        self.model_manager = None
        if llm:
            self.llm = llm
        else:
            # Initialize LLM and model manager using factory pattern
            from .core.model_config_loader import create_model_manager_from_config
            from .core.model_selector import ModelSelector

            try:
                # CRITICAL: Create ModelSelector for rate limiting BEFORE ModelManager
                # Without this, ModelManager has no rate limiting and uses OpenAI SDK's
                # fast retry logic (0.4s) which triggers 429 errors constantly
                model_selector = None
                if self.config_manager and self.agent_config:
                    try:
                        model_selector = ModelSelector(self.agent_config)
                        logger.info("✅ ModelSelector initialized | Rate limiting ENABLED")
                    except Exception as e:
                        logger.warning(f"Failed to create ModelSelector: {e} | Rate limiting DISABLED")

                # Create model manager for multi-model support (WITH rate limiting if available)
                self.model_manager = create_model_manager_from_config(model_selector=model_selector)
                # Get default LLM from model manager
                self.llm = self.model_manager.get_chat_model("default")
                if self.llm is None:
                    raise ValueError("ModelManager returned None for default LLM")
                logger.info("Successfully initialized LLM and ModelManager via factory")
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

        # Store compiled graph directly (removed thread-local for simplicity)
        # Thread-local was causing issues with async execution
        self._compiled_graph = None
        self._graph_builder = self._build_graph

        logger.info("Enhanced Research Agent initialized with full components")

    def _flush_pending_intermediate_events(self) -> None:
        """Immediately stream any queued intermediate events."""
        # Check both instance-level pending events and state-level intermediate_events
        pending = getattr(self, "_pending_intermediate_events", None)
        stream_emitter = getattr(self, "stream_emitter", None)

        if not stream_emitter:
            if pending:
                pending.clear()
            return

        # First flush instance-level pending events
        if pending:
            for event in list(pending):
                if not event:
                    continue
                try:
                    stream_emitter(event)
                except Exception as exc:
                    logger.warning(
                        "Failed to stream pending intermediate event", exc_info=exc
                    )
            pending.clear()

        # NEW: Also flush any intermediate_events from the state
        # This is critical for step events that are added directly to state
        # Note: State access during execution is handled differently now without thread-local

    @property
    def graph(self):
        """Get or create graph instance (simplified from thread-local)."""
        # Initialize graph if not yet created
        if self._compiled_graph is None:
            logger.info("Compiling workflow graph for first use")
            # Ensure _graph_builder exists (for MLflow deserialization compatibility)
            if not hasattr(self, '_graph_builder'):
                logger.warning("_graph_builder missing, initializing from _build_graph method")
                self._graph_builder = self._build_graph
            self._compiled_graph = self._graph_builder()
        return self._compiled_graph

    @graph.setter
    def graph(self, value):
        """Set the workflow graph (for test compatibility)."""
        self._compiled_graph = value

    @graph.deleter
    def graph(self):
        """Delete the workflow graph (for test cleanup)."""
        self._compiled_graph = None

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration using the new ConfigLoader."""
        try:
            from .config_loader import ConfigLoader

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
                # Keep high (e.g., 500) and rely on high-level loop caps for behavior.
                "recursion_limit": int(
                    self.config.get("workflow", {}).get("recursion_limit", 50)
                ),
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
        Emit PHASE marker as intermediate event only (not as content delta).

        Note: Progress markers should not appear in displayed content.
        The UI handles progress tracking through intermediate events and the event system.

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

        # Emit as intermediate event instead of content delta to avoid showing progress markers in UI
        event_data = {
            "phase": phase,
            "node": node_name,
            "progress": progress,
            "elapsed": elapsed,
            **extra_meta
        }

        return ResponsesAgentStreamEvent(
            type="intermediate_event",
            intermediate_event={
                "event_type": "phase_marker",
                "data": event_data,
                "timestamp": time.time(),
                "correlation_id": f"phase_{node_name}_{item_id}"
            }
        )

    def _emit_plan_event(self, plan, item_id: str) -> ResponsesAgentStreamEvent:
        """Emit detailed plan event with all step information."""
        try:
            plan_steps = []

            if hasattr(plan, "steps") and plan.steps:
                for i, step in enumerate(plan.steps):
                    step_data = {
                        "id": getattr(step, "id", f"step_{i+1}"),
                        "index": i + 1,
                        "description": getattr(step, "description", str(step)),
                        "status": "pending",
                        "dependencies": getattr(step, "dependencies", []),
                    }
                    plan_steps.append(step_data)

            # Send as intermediate event
            return ResponsesAgentStreamEvent(
                type="intermediate_event",
                intermediate_event={
                    "event_type": "plan_structure",
                    "data": {
                        "plan": {
                            "steps": plan_steps,
                            "total_steps": len(plan_steps),
                            "quality": getattr(plan, "quality", None),
                            "has_enough_context": getattr(plan, "has_enough_context", True),
                        }
                    }
                }
            )
        except Exception as e:
            logger.warning(f"Failed to emit plan event: {e}")
            return None

    def _emit_step_progress_event(
        self, step_id: str, status: str, result: str, item_id: str
    ) -> ResponsesAgentStreamEvent:
        """Emit step progress update event."""
        normalized_step_id = (
            PlanIDGenerator.normalize_id(step_id)
            if step_id
            else PlanIDGenerator.generate_step_id(1)
        )
        if status not in {"in_progress", "completed", "failed"}:
            logger.debug(
                "Normalizing unexpected step status '%s' to 'in_progress'",
                status,
            )
            status = "in_progress"

        event_type = (
            "step_completed"
            if status == "completed"
            else "step_failed"
            if status == "failed"
            else "step_activated"
        )

        logger.debug(
            "STREAM: preparing %s for step=%s status=%s result_preview=%s",
            event_type,
            normalized_step_id,
            status,
            (result[:80] + "…") if isinstance(result, str) and len(result) > 80 else result,
        )

        return ResponsesAgentStreamEvent(
            type="intermediate_event",
            intermediate_event={
                "event_type": event_type,
                "data": {
                    "step_id": normalized_step_id,
                    "status": status,  # "in_progress", "completed", "failed"
                    "description": result if result else "",
                    "timestamp": time.time(),
                }
            }
        )

    def _emit_search_query_event(
        self, query: str, provider: str, item_id: str
    ) -> ResponsesAgentStreamEvent:
        """Emit search query execution event."""
        return ResponsesAgentStreamEvent(
            type="response.metadata",
            item_id=item_id,
            metadata={
                "searchQuery": {
                    "query": query,
                    "provider": provider,
                    "timestamp": time.time(),
                }
            },
        )

    def _emit_factuality_event(
        self, claim: str, verdict: str, score: float, evidence: str, item_id: str
    ) -> ResponsesAgentStreamEvent:
        """Emit factuality check event."""
        return ResponsesAgentStreamEvent(
            type="response.metadata",
            item_id=item_id,
            metadata={
                "factualityCheck": {
                    "claim": claim,
                    "verdict": verdict,  # "supported", "contradicted", "uncertain"
                    "score": score,
                    "evidence": evidence,
                    "timestamp": time.time(),
                }
            },
        )

    def _emit_report_chunk(
        self, content: str, item_id: str
    ) -> ResponsesAgentStreamEvent:
        """Emit report content chunk as proper content_delta."""
        return ResponsesAgentStreamEvent(
            type="response.output_text.delta", item_id=item_id, delta=content
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
        workflow.add_node("calculation_planning", self.workflow_nodes.calculation_planning_node)
        workflow.add_node("fact_checker", self.workflow_nodes.fact_checker_node)
        workflow.add_node("reporter", self.workflow_nodes.reporter_node)
        workflow.add_node("human_feedback", self.workflow_nodes.human_feedback_node)

        # Set entry point
        workflow.set_entry_point("coordinator")

        # Add conditional edges for coordinator
        def coordinator_router(state):
            """Route from coordinator based on state."""
            global logger
            total_steps = state.get("total_workflow_steps", 0)
            logger.info(
                f"[ROUTER] coordinator_router - step {total_steps}, research_loops={state.get('research_loops', 0)}, fact_check_loops={state.get('fact_check_loops', 0)}"
            )

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
            global logger
            plan = state.get("current_plan")
            if not plan:
                return END

            # CRITICAL: Check if coordination completed and decided SUFFICIENT
            if state.get("coordination_completed"):
                decision = state.get("gap_decision", {})
                action = decision.get("action")

                # SAFETY: Handle None or invalid action
                if not action:
                    logger.warning(f"[PLANNER_ROUTER] Coordinator returned no action - defaulting to SUFFICIENT")
                    if state.get("metric_capability_enabled", False):
                        return "calculation_planning"
                    return "fact_checker" if state.get("enable_grounding", True) else "reporter"

                if action == "SUFFICIENT":
                    logger.info(f"[PLANNER_ROUTER] Coordination decided SUFFICIENT - routing to next phase")
                    if state.get("metric_capability_enabled", False):
                        return "calculation_planning"
                    return "fact_checker" if state.get("enable_grounding", True) else "reporter"

                elif action in ["EXTEND", "VERIFY"]:
                    logger.info(f"[PLANNER_ROUTER] Coordination decided {action} - adding steps, routing to researcher")
                    # Plan was extended with new steps - continue research
                    return "researcher"

                else:
                    logger.warning(f"[PLANNER_ROUTER] Unexpected action '{action}' - defaulting to next phase")
                    if state.get("metric_capability_enabled", False):
                        return "calculation_planning"
                    return "fact_checker" if state.get("enable_grounding", True) else "reporter"

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
                "fact_checker": "fact_checker",  # CRITICAL: Add this edge for coordination
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
            global logger

            # DIAGNOSTIC: Track all researcher_router calls
            logger.info("🎯 [ROUTER] ===== RESEARCHER_ROUTER CALLED =====")
            logger.info(f"🎯 [ROUTER] State keys: {list(state.keys())[:30]}")
            logger.info(f"🎯 [ROUTER] State object ID: {id(state)}")

            # SAFETY CHECK: Check for permanent error flags in state
            if state.get("permanent_failure"):
                logger.error(
                    "[ROUTER] Permanent failure detected in state - halting workflow"
                )
                if "warnings" not in state:
                    state["warnings"] = []
                state["warnings"].append(
                    "Research stopped due to a permanent error that cannot be resolved by retrying."
                )
                return (
                    "reporter"  # Skip to reporter to generate final report with error
                )

            # SAFETY CHECK: Check for permanent error conditions in warnings
            warnings = state.get("warnings", [])
            for warning in warnings:
                if any(
                    indicator in warning.lower()
                    for indicator in [
                        "permanent error",
                        "ip acl",
                        "blocked",
                        "authentication",
                    ]
                ):
                    logger.error(
                        f"[ROUTER] Permanent error condition detected in warnings: {warning}"
                    )
                    return "reporter"  # Skip to reporter to generate final report with error

            plan = state.get("current_plan")

            # Enhanced diagnostic logging
            if plan and hasattr(plan, "steps"):
                from .core.plan_models import StepStatus

                # Count steps by status
                status_counts = {}
                for step in plan.steps:
                    status = (
                        step.status.value
                        if hasattr(step.status, "value")
                        else str(step.status)
                    )
                    status_counts[status] = status_counts.get(status, 0) + 1

                total_steps = len(plan.steps)

                # Check both lowercase and uppercase variants since the enum might use either
                completed = status_counts.get("completed", 0) + status_counts.get(
                    "COMPLETED", 0
                )
                pending = status_counts.get("pending", 0) + status_counts.get(
                    "PENDING", 0
                )
                failed = status_counts.get("failed", 0) + status_counts.get("FAILED", 0)

                logger.info(
                    f"[ROUTER] Plan Status: {completed}/{total_steps} completed, {pending} pending, {failed} failed"
                )

                # Log section information if available
                if (
                    hasattr(plan, "section_specifications")
                    and plan.section_specifications
                ):
                    logger.info(
                        f"[ROUTER] Plan has {len(plan.section_specifications)} sections"
                    )

            # Step-based completion check with enhanced logging
            if plan and hasattr(plan, "get_next_step"):
                try:
                    next_step = plan.get_next_step()
                    if next_step is None:
                        # All steps complete - check research loop limit before coordinating
                        research_loops = state.get("research_loops", 0)
                        max_research_loops = state.get("max_research_loops", 2)

                        # CRITICAL FIX: Enforce research loop limit to prevent infinite coordination loops
                        if research_loops >= max_research_loops:
                            logger.warning(
                                f"[ROUTER] Research loops limit reached ({research_loops}/{max_research_loops}) - "
                                f"skipping coordination, routing to next phase"
                            )
                            if state.get("metric_capability_enabled", False):
                                return "calculation_planning"
                            return "fact_checker" if state.get("enable_grounding", True) else "reporter"

                        # Under loop limit - planner will coordinate
                        logger.info(
                            f"[ROUTER] All steps complete - routing to planner for coordination "
                            f"(loop {research_loops + 1}/{max_research_loops})"
                        )
                        state["coordination_completed"] = False  # Let planner decide
                        return "planner"
                    else:
                        # Continue executing steps
                        logger.info(f"[ROUTER] Next step: {next_step.step_id} - {next_step.title}")
                        state["current_step"] = next_step
                        state["current_step_number"] = state.get("current_step_number", 0) + 1
                        return "researcher"

                except Exception as e:
                    import traceback
                    logger.error(f"❌ [ROUTER] EXCEPTION in researcher_router!")
                    logger.error(f"❌ [ROUTER] Exception type: {type(e).__name__}")
                    logger.error(f"❌ [ROUTER] Exception message: {str(e)}")
                    logger.error(f"❌ [ROUTER] Traceback:\n{traceback.format_exc()}")
                    logger.warning(f"[ROUTER] Error checking step completion: {e}")
                    # Fallback: check if we have section results
                    section_results = state.get("section_research_results", {})
                    if section_results:
                        logger.info(
                            "[ROUTER] Found section results - assuming complete"
                        )
                        # Check if calculation planning should be done first
                        if state.get("metric_capability_enabled", False):
                            return "calculation_planning"
                        return (
                            "fact_checker"
                            if state.get("enable_grounding", True)
                            else "reporter"
                        )
                    else:
                        # No section results and error occurred - complete research phase
                        logger.warning(
                            "[ROUTER] Error in step checking and no section results - completing research phase"
                        )
                        # Check if calculation planning should be done first
                        if state.get("metric_capability_enabled", False):
                            return "calculation_planning"
                        return (
                            "fact_checker"
                            if state.get("enable_grounding", True)
                            else "reporter"
                        )
            else:
                # No plan - should not happen but handle gracefully
                logger.warning("[ROUTER] No plan found - assuming complete")
                # Check if calculation planning should be done first
                if state.get("metric_capability_enabled", False):
                    return "calculation_planning"
                return (
                    "fact_checker"
                    if state.get("enable_grounding", True)
                    else "reporter"
                )

            # This should never be reached - defensive programming
            logger.error("[ROUTER] CRITICAL: Reached end of researcher_router without returning - defaulting to reporter")
            return "reporter"

        workflow.add_conditional_edges(
            "researcher",
            researcher_router,
            {
                "calculation_planning": "calculation_planning",  # NEW: Route to calculation planning
                "fact_checker": "fact_checker",
                "reporter": "reporter",
                "researcher": "researcher",  # Continue research
                "planner": "planner",  # Re-plan if needed
            },
        )
        
        # Calculation planning routing
        # Note: The calculation_planning_node returns Command(goto=...) which handles routing
        # So edges are implicit via Command, but we add them here for graph completeness
        workflow.add_edge("calculation_planning", "reporter")  # Default path

        # Import routing policy for bulletproof routing decisions
        from .core.routing_policy import determine_next_node, TerminationReason

        # Fact checker routing - now bulletproof with centralized policy
        def fact_checker_router(state):
            """Route from fact checker using centralized routing policy."""
            global logger
            total_steps = state.get("total_workflow_steps", 0)
            fact_check_loops = state.get("fact_check_loops", 0)
            max_fact_check_loops = state.get("max_fact_check_loops", 2)

            logger.info(
                f"[ROUTER] fact_checker_router - step {total_steps}, fact_check_loops={fact_check_loops}, max_fact_check_loops={max_fact_check_loops}"
            )

            # CRITICAL: Add state validation to prevent data loss
            try:
                StateValidator.validate_state_for_agent("fact_checker", state)
            except ValueError as e:
                logger.error(f"Fact checker router: State validation failed: {e}")
                # Continue but flag the issue
                state.setdefault("warnings", []).append(
                    f"State validation failed in fact_checker_router: {e}"
                )

            # FIX: Hard stop when fact check loops are exhausted
            if fact_check_loops >= max_fact_check_loops:
                logger.warning(
                    f"HARD STOP: Fact check loops exhausted ({fact_check_loops}/{max_fact_check_loops}), routing to reporter"
                )
                # Record transition for debugging
                global_propagation_tracker.record_transition(
                    from_agent="fact_checker_router",
                    to_agent="reporter",
                    state_snapshot=state,
                )
                return "reporter"

            # Check total steps limit
            max_total_steps = state.get("max_total_steps", 20)
            if total_steps >= max_total_steps:
                logger.warning(
                    f"HARD STOP: Total steps exhausted ({total_steps}/{max_total_steps}), routing to reporter"
                )
                # Record transition for debugging
                global_propagation_tracker.record_transition(
                    from_agent="fact_checker_router",
                    to_agent="reporter",
                    state_snapshot=state,
                )
                return "reporter"

            # Get factuality report from state
            factuality_report = state.get("factuality_report")
            if not factuality_report:
                # Fallback if no report available
                logger.warning("No factuality report in state - defaulting to reporter")
                global_propagation_tracker.record_transition(
                    from_agent="fact_checker_router",
                    to_agent="reporter",
                    state_snapshot=state,
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
                from_agent="fact_checker_router",
                to_agent=next_node,
                state_snapshot=state,
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
        # CRITICAL FIX: Use MemorySaver by default for proper state persistence
        # NoOpCheckpointSaver was causing state loss between nodes
        use_noop_checkpointer = (
            os.getenv("DISABLE_CHECKPOINTER", "false").lower() == "true"  # Changed default to false
            # Don't force NoOpCheckpointer in Databricks runtime anymore
            # State persistence is critical for multi-agent workflow
        )

        if use_noop_checkpointer:
            checkpointer = NoOpCheckpointSaver()
            logger.warning("Using NoOpCheckpointSaver - state may not persist between nodes!")
        else:
            checkpointer = MemorySaver()
            logger.info("Using MemorySaver for proper state persistence between workflow nodes")

        # Compile the workflow
        # recursion_limit is set via config during graph execution, not during compilation
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
                "recursion_limit": int(
                    self.config.get("workflow", {}).get("recursion_limit", 50)
                ),
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
                "recursion_limit": int(
                    self.config.get("workflow", {}).get("recursion_limit", 50)
                ),
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
                    "stack_trace": stack_trace,
                },
            }

    def _sanitize_for_json(self, obj: Any) -> Any:
        """
        Recursively sanitize object for JSON serialization.

        Converts datetime objects to Unix timestamps and handles Pydantic models.
        This ensures all data passed to ResponsesAgentStreamEvent can be JSON-serialized.

        Args:
            obj: Any Python object that may contain non-JSON-serializable types

        Returns:
            JSON-serializable version of the object
        """
        if obj is None:
            return None
        elif isinstance(obj, datetime):
            return obj.timestamp()
        elif isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        elif hasattr(obj, 'to_dict') and callable(obj.to_dict):
            # Pydantic models with custom to_dict() method
            return obj.to_dict()
        elif hasattr(obj, 'model_dump') and callable(obj.model_dump):
            # Pydantic v2 models
            return self._sanitize_for_json(obj.model_dump())
        elif hasattr(obj, 'dict') and callable(obj.dict):
            # Pydantic v1 models
            return self._sanitize_for_json(obj.dict())
        else:
            # Return primitive types as-is (str, int, float, bool, None)
            return obj

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

    def _extract_streaming_metadata(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract research metadata from final state for UI consumption."""
        try:
            # Extract basic research metadata
            metadata = {
                "searchQueries": state.get("queries", []),
                "sources": self._format_sources_for_ui(state.get("citations", [])),
                "researchIterations": len(state.get("completed_steps", [])),
                "confidenceScore": state.get("confidence_score"),
                "factualityScore": state.get("factuality_score"),
                "currentAgent": state.get("current_agent", "reporter"),
                "reportStyle": state.get("report_style", "professional"),
                "verificationLevel": state.get("verification_level", "moderate"),
            }

            logger.debug(
                "STREAM_METADATA: base=%s",
                {k: metadata[k] for k in metadata if k not in {"sources"}},
            )

            # Extract and format plan details
            current_plan = state.get("current_plan")
            if current_plan:
                plan_metadata = self._format_plan_for_ui(current_plan, state)
                metadata["planDetails"] = plan_metadata
                logger.debug(
                    "STREAM_METADATA: planDetails steps=%s status=%s",
                    len(plan_metadata.get("steps", [])),
                    plan_metadata.get("status"),
                )

            # Extract grounding metadata
            grounding_results = state.get("grounding_results", [])
            contradictions = state.get("contradictions", [])
            if grounding_results or contradictions:
                grounding_metadata = self._format_grounding_for_ui(
                    grounding_results, contradictions, state
                )
                metadata["grounding"] = grounding_metadata

            # Sanitize all metadata to ensure JSON serialization works
            metadata = self._sanitize_for_json(metadata)
            return metadata

        except Exception as e:
            logger.warning(f"Failed to extract streaming metadata: {e}")
            return {
                "searchQueries": [],
                "sources": [],
                "researchIterations": 0,
                "currentAgent": "reporter",
            }

    def _format_sources_for_ui(self, citations: list) -> list:
        """Format citations as sources for UI consumption."""
        sources = []
        for citation in citations[:20]:  # Limit to 20 sources for UI performance
            if isinstance(citation, dict):
                source = {
                    "url": citation.get("url", ""),
                    "title": citation.get("title", "Unknown Source"),
                    "relevanceScore": citation.get("relevance_score"),
                    "snippet": citation.get("snippet", "")[
                        :200
                    ],  # Limit snippet length
                }
                sources.append(source)
        return sources

    def _format_plan_for_ui(self, plan, state: Dict[str, Any]) -> Dict[str, Any]:
        """Format plan metadata for UI visualization."""
        try:
            plan_steps = []
            completed_steps = state.get("completed_steps", [])
            completed_step_ids = {
                PlanIDGenerator.normalize_id(step.get("id") or step.get("step_id"))
                for step in completed_steps
                if isinstance(step, dict) and (step.get("id") or step.get("step_id"))
            }

            # Extract steps from plan
            if hasattr(plan, "steps") and plan.steps:
                for idx, step in enumerate(plan.steps):
                    raw_step_id = (
                        getattr(step, "id", None)
                        or getattr(step, "step_id", None)
                        or PlanIDGenerator.generate_step_id(idx + 1)
                    )
                    normalized_id = PlanIDGenerator.normalize_id(raw_step_id)
                    status_value = getattr(step, "status", None)
                    if not status_value:
                        status_value = (
                            "completed"
                            if normalized_id in completed_step_ids
                            else "pending"
                        )

                    step_data = {
                        "id": normalized_id,
                        "step_id": normalized_id,
                        "description": getattr(step, "description", str(step)),
                        "status": status_value,
                    }

                    # Add completion timestamp if available
                    for completed_step in completed_steps:
                        if (
                            isinstance(completed_step, dict)
                            and PlanIDGenerator.normalize_id(
                                completed_step.get("id")
                                or completed_step.get("step_id")
                            )
                            == step_data["id"]
                        ):
                            # Sanitize timestamp to prevent datetime serialization error
                            timestamp = completed_step.get("timestamp")
                            if isinstance(timestamp, datetime):
                                step_data["completedAt"] = timestamp.timestamp()
                            elif timestamp is not None:
                                step_data["completedAt"] = timestamp
                            step_data["result"] = completed_step.get("result", "")
                            break

                    plan_steps.append(step_data)

            return {
                "steps": plan_steps,
                "quality": getattr(plan, "quality", None),
                "iterations": state.get(
                    "plan_iterations", getattr(plan, "iterations", 1)
                ),
                "status": getattr(plan, "status", None) or "executing",
                "hasEnoughContext": getattr(plan, "has_enough_context", True),
            }

        except Exception as e:
            logger.warning(f"Failed to format plan for UI: {e}")
            return {
                "steps": [],
                "iterations": 1,
                "status": "completed",
                "hasEnoughContext": True,
            }

    def _format_grounding_for_ui(
        self, grounding_results: list, contradictions: list, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format grounding metadata for UI."""
        try:
            formatted_contradictions = []
            for contradiction in contradictions[:10]:  # Limit for UI performance
                if isinstance(contradiction, dict):
                    formatted_contradictions.append(
                        {
                            "id": contradiction.get(
                                "id", str(len(formatted_contradictions))
                            ),
                            "claim": contradiction.get("claim", ""),
                            "evidence": contradiction.get("evidence", ""),
                            "severity": contradiction.get("severity", "medium"),
                            "resolved": contradiction.get("resolved", False),
                            "resolution": contradiction.get("resolution", ""),
                        }
                    )

            formatted_verifications = []
            for result in grounding_results[:20]:  # Limit for UI performance
                if isinstance(result, dict):
                    formatted_verifications.append(
                        {
                            "id": result.get("id", str(len(formatted_verifications))),
                            "fact": result.get("claim", result.get("fact", "")),
                            "verified": result.get("verified", True),
                            "confidence": result.get("confidence", 0.8),
                            "sources": result.get("sources", []),
                        }
                    )

            return {
                "factualityScore": state.get("factuality_score", 0.8),
                "contradictions": formatted_contradictions,
                "verifications": formatted_verifications,
                "verificationLevel": state.get("verification_level", "moderate"),
            }

        except Exception as e:
            logger.warning(f"Failed to format grounding for UI: {e}")
            return {
                "factualityScore": 0.8,
                "contradictions": [],
                "verifications": [],
                "verificationLevel": "moderate",
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
            logger.warning(
                f"No query extracted from request - agents may fail to find user message"
            )
        else:
            logger.info(f"Successfully extracted query: {query[:100]}...")

        logger.info(f"Processing predict request: {query[:100]}...")

        # No mock responses - always run real workflow
        # Use safe async executor to avoid event loop conflicts
        from .core.async_utils import AsyncExecutor

        try:
            # Run async research with timeout using safe executor
            result = AsyncExecutor.run_async_safe(
                self.research(query),
                timeout=300.0  # 5 minute timeout
            )

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
        except asyncio.TimeoutError:
            logger.error("Research timed out after 5 minutes")
            error_item = {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Request timed out. The research topic may be too complex. Please try a simpler query.",
                    }
                ],
                "id": str(uuid4()),
            }
            return ResponsesAgentResponse(
                output=[error_item], custom_outputs={"error": "timeout"}
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

    async def predict_stream_async(
        self, request: ResponsesAgentRequest
    ):
        """
        Native async streaming API for FastAPI and async clients.

        This method provides direct async generator access without thread overhead,
        ideal for async web frameworks like FastAPI. For sync clients (MLflow),
        use predict_stream() instead which provides sync generator interface.

        Args:
            request: ResponsesAgentRequest with user query

        Yields:
            ResponsesAgentStreamEvent objects with intermediate events and final response

        Example:
            ```python
            # FastAPI usage
            @app.post("/stream")
            async def stream_endpoint(request: Request):
                agent = EnhancedResearchAgent(...)
                async for event in agent.predict_stream_async(req):
                    yield event
            ```
        """
        # Delegate to predict_stream which now uses context-aware bridge
        # The bridge will detect async context and return async generator
        async for event in self.predict_stream(request):
            yield event

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Stream responses with intermediate events for real-time UI updates.

        HYBRID: This method works in both sync and async contexts:
        - Sync context (MLflow): Returns sync generator using thread-based bridge
        - Async context (FastAPI): Returns async generator directly

        For async-native code, prefer predict_stream_async() for clearer semantics.

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
            logger.warning(
                f"No query extracted from request - agents may fail to find user message"
            )
        else:
            logger.info(f"Successfully extracted query: {query[:100]}...")

        # Import uuid4 locally to avoid scoping issues
        from uuid import uuid4
        item_id = str(uuid4())

        # CRITICAL: Clear any residual state from previous queries to prevent contamination
        self._clear_previous_state()

        # Generate query fingerprint for isolation
        query_hash = self._generate_query_hash(query)
        logger.info(f"Processing predict_stream request: {query[:100]}...")
        logger.info(f"Starting fresh query session: {query_hash}")

        # No mock streaming - always run real workflow

        # Import workflow limit initializer and async executor
        from .core.routing_policy import initialize_workflow_limits
        from .core.async_utils import AsyncExecutor

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
        final_state = {}  # Track final state for metadata

        async def run_streaming():
            # Add required config for LangGraph checkpointer
            # FIX: Ensure recursion_limit is properly set from workflow config
            recursion_limit = int(
                self.config.get("workflow", {}).get("recursion_limit", 50)
            )

            # Log the actual limit being used
            logger.info(f"Using recursion_limit: {recursion_limit}")

            config = {
                "configurable": {
                    "thread_id": str(uuid4()),
                    "checkpoint_ns": "enhanced_research_agent",
                },
                "recursion_limit": recursion_limit,
            }

            # DIAGNOSTIC: Track stream lifecycle
            logger.info("🎯 [STREAM] ===== STARTING WORKFLOW EVENT STREAM =====")
            node_count = 0

            async for event in self.graph.astream(initial_state, config=config):
                node_count += 1
                event_summary = list(event.keys()) if isinstance(event, dict) else type(event).__name__
                logger.debug(f"🎯 [STREAM] Event {node_count}: {event_summary}")
                yield event

            logger.info(f"🎯 [STREAM] ===== ASTREAM LOOP COMPLETED after {node_count} events =====")
            logger.info("🎯 [STREAM] Workflow has terminated, proceeding to emit final event")

        # Process events from async generator using safe executor
        event_generator = run_streaming()

        try:
            # Calculate streaming timeouts from workflow config
            # This ensures stream stays alive for entire workflow duration
            workflow_config = self.config.get("workflow", {})
            max_wall_clock = float(workflow_config.get("max_wall_clock_seconds", 300))

            # Allow explicit overrides for streaming timeouts
            timeout_per_item = workflow_config.get("stream_timeout_per_item_seconds")
            first_item_timeout = workflow_config.get("stream_first_item_timeout_seconds")

            if timeout_per_item is None:
                # Give downstream nodes enough time to finish long LLM calls (planner ~300s)
                computed_timeout = max(
                    60.0,
                    max_wall_clock * 0.5,
                    max(0.0, max_wall_clock - 30.0)
                )
                timeout_per_item = min(max_wall_clock, computed_timeout)
            else:
                timeout_per_item = float(timeout_per_item)

            if first_item_timeout is None:
                first_item_timeout = max(
                    timeout_per_item,
                    max(180.0, max_wall_clock * 0.75)
                )
            else:
                first_item_timeout = float(first_item_timeout)

            logger.info(
                f"Streaming timeouts: per_item={timeout_per_item:.0f}s, "
                f"first_item={first_item_timeout:.0f}s (based on max_wall_clock={max_wall_clock}s)"
            )
            logger.info(
                f"Workflow config keys: {list(workflow_config.keys())}, "
                f"enable_background_investigation={workflow_config.get('enable_background_investigation', 'NOT SET')}"
            )

            # Use AsyncExecutor context-aware bridge for async/sync compatibility
            # This automatically detects if we're in async (FastAPI) or sync (MLflow) context
            for event in AsyncExecutor.stream_async_bridge(
                event_generator,
                timeout_per_item=timeout_per_item,
                first_item_timeout=first_item_timeout
            ):

                    # Process LangGraph events
                    if isinstance(event, dict):
                        for node_name, node_data in event.items():
                            # DIAGNOSTIC: Log every node execution
                            logger.info(f"🎯 [EVENT] Processing node: {node_name}")

                            # Defensive check: Skip if node returned None
                            if node_data is None:
                                logger.warning(
                                    f"Node {node_name} returned None, skipping"
                                )
                                continue

                            # DIAGNOSTIC: Check for reporter completion
                            if node_name == "reporter" and isinstance(node_data, dict):
                                has_completion = node_data.get("reporter_completed", False)
                                should_terminate = node_data.get("workflow_should_terminate", False)
                                has_error = node_data.get("workflow_failed", False)
                                logger.info(f"🎯 [EVENT] Reporter event received")
                                logger.info(f"🎯 [EVENT] Reporter flags: completed={has_completion}, terminate={should_terminate}, failed={has_error}")

                            # DIAGNOSTIC: Detect unexpected coordinator-after-reporter loop
                            elif node_name == "coordinator" and isinstance(node_data, dict):
                                if node_data.get("reporter_completed"):
                                    logger.error("🎯 [EVENT] ❌ BUG DETECTED: COORDINATOR RUNNING AFTER REPORTER!")
                                    logger.error("🎯 [EVENT] State has reporter_completed=True but coordinator is executing again")
                                    logger.error("🎯 [EVENT] This indicates the workflow is looping instead of terminating")

                            # Track state updates for final metadata
                            if isinstance(node_data, dict):
                                final_state.update(node_data)
                                
                                # Process intermediate_events from state and add to pending events
                                if "intermediate_events" in node_data and node_data["intermediate_events"]:
                                    events_to_add = node_data["intermediate_events"]
                                    logger.info("🎯 Processing %d intermediate events from state for node %s",
                                               len(events_to_add), node_name)
                                    
                                    # Add these events to our pending list for streaming
                                    if not hasattr(self, "_pending_intermediate_events"):
                                        self._pending_intermediate_events = []
                                    
                                    for event in events_to_add:
                                        # Convert event dict to ResponsesAgentStreamEvent if needed
                                        if isinstance(event, dict):
                                            # ResponsesAgentStreamEvent already imported at module level
                                            from uuid import uuid4
                                            
                                            # Create proper streaming event format
                                            stream_event = ResponsesAgentStreamEvent(
                                                type="intermediate_event",
                                                item_id=str(uuid4()),
                                                intermediate_event=event
                                            )
                                            self._pending_intermediate_events.append(stream_event)
                                            logger.info("🎯 Added event %s to pending stream events", 
                                                       event.get("event_type", "unknown"))
                                        else:
                                            # Already a streaming event
                                            self._pending_intermediate_events.append(event)
                                            
                            elif hasattr(node_data, "update") and node_data.update:
                                final_state.update(node_data.update)

                            # Emit intermediate events from event emitter
                            if self.event_emitter and hasattr(
                                self, "_pending_intermediate_events"
                            ):
                                pending_events = list(
                                    self._pending_intermediate_events or []
                                )
                                pending_count = len(pending_events)
                                if pending_count:
                                    logger.info(
                                        "🎯 Emitting %s pending intermediate events from queue",
                                        pending_count,
                                    )
                                    logger.info(
                                        "🎯 Pending intermediate event types: %s",
                                        [
                                            getattr(evt, "type", "unknown")
                                            for evt in pending_events
                                        ],
                                    )
                                for pending in pending_events:
                                    if pending:
                                        pending_type = getattr(
                                            pending, "type", "unknown"
                                        )
                                        if pending_type:
                                            logger.debug(
                                                "🎯 Streaming intermediate event chunk: type=%s",
                                                pending_type,
                                            )
                                        metadata_keys = []
                                        if (
                                            hasattr(pending, "metadata")
                                            and pending.metadata
                                        ):
                                            metadata_keys = list(
                                                pending.metadata.keys()
                                            )[:10]
                                        elif (
                                            hasattr(pending, "event") and pending.event
                                        ):
                                            metadata_keys = list(pending.event.keys())[
                                                :10
                                            ]
                                        if metadata_keys:
                                            logger.debug(
                                                "🎯 Intermediate event metadata keys: %s",
                                                metadata_keys,
                                            )
                                    yield pending
                                self._pending_intermediate_events.clear()
                            elif self.event_emitter:
                                logger.debug(
                                    "🎯 No pending intermediate events to emit for node %s",
                                    node_name,
                                )

                            # Handle different node outputs with progress updates
                            if node_name == "coordinator":
                                # Import locally to avoid scoping issues
                                # ResponsesAgentStreamEvent already imported at module level
                                
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

                                # REMOVED: Progress message now handled by ResearchProgress component
                                # yield ResponsesAgentStreamEvent(
                                #     type="response.output_text.delta",
                                #     item_id=item_id,
                                #     delta="🎯 Analyzing request and routing to appropriate agents...\n",
                                # )

                            elif node_name == "background_investigation":
                                # Import locally to avoid scoping issues
                                # ResponsesAgentStreamEvent already imported at module level
                                
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

                                    # REMOVED: Progress message now handled by ResearchProgress component
                                    # yield ResponsesAgentStreamEvent(
                                    #     type="response.output_text.delta",
                                    #     item_id=item_id,
                                    #     delta="🔍 Gathering background information...\n",
                                    # )

                            elif node_name == "planner":
                                # Import locally to avoid scoping issues
                                # ResponsesAgentStreamEvent already imported at module level
                                
                                plan = node_data.get("current_plan")
                                if plan:
                                    # Emit PHASE marker first
                                    phase_event = self._emit_phase_marker(
                                        node_name, item_id, start_time
                                    )
                                    if phase_event:
                                        yield phase_event

                                    # Emit detailed plan event
                                    plan_event = self._emit_plan_event(plan, item_id)
                                    if plan_event:
                                        yield plan_event

                                    # Emit simple plan creation notification 
                                    # (Plan steps will be shown in unified ResearchProgress component)
                                if plan:
                                    steps_count = (
                                        len(plan.steps) if hasattr(plan, "steps") else 0
                                    )
                                    # REMOVED: Progress message now handled by ResearchProgress component
                                    # yield ResponsesAgentStreamEvent(
                                    #     type="response.output_text.delta",
                                    #     item_id=item_id,
                                    #     delta=f"📋 Created research plan with {steps_count} steps\n",
                                    # )

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

                                # Emit PHASE marker first
                                phase_event = self._emit_phase_marker(
                                    node_name, item_id, start_time
                                )
                                if phase_event:
                                    yield phase_event

                                # Extract and emit search queries if available
                                queries = node_data.get("queries") or []
                                for query in queries:
                                    if isinstance(query, str):
                                        yield self._emit_search_query_event(
                                            query, "brave", item_id
                                        )

                                # Emit step progress events
                                completed_steps = node_data.get("completed_steps") or []
                                for step in completed_steps:
                                    if isinstance(step, dict):
                                        yield self._emit_step_progress_event(
                                            step.get("id", "unknown"),
                                            "completed",
                                            step.get("result", ""),
                                            item_id,
                                        )

                                # Check if we're executing a specific step
                                current_step = node_data.get("current_research_step")
                                if current_step:
                                    yield self._emit_step_progress_event(
                                        current_step.get("id", "unknown")
                                        if isinstance(current_step, dict)
                                        else str(current_step),
                                        "in_progress",
                                        "",
                                        item_id,
                                    )

                                observations = node_data.get("observations") or []
                                if observations:
                                    # REMOVED: Progress message now handled by ResearchProgress component
                                    # yield ResponsesAgentStreamEvent(
                                    #     type="response.output_text.delta",
                                    #     item_id=item_id,
                                    #     delta=f"🔍 Research in progress - found {len(observations)} key findings\n",
                                    # )
                                    pass

                            elif node_name == "fact_checker":
                                # Emit PHASE marker first
                                phase_event = self._emit_phase_marker(
                                    node_name, item_id, start_time
                                )
                                if phase_event:
                                    yield phase_event

                                # Emit factuality check details
                                grounding_results = (
                                    node_data.get("grounding_results") or []
                                )
                                for result in grounding_results:
                                    if isinstance(result, dict):
                                        yield self._emit_factuality_event(
                                            result.get("claim", ""),
                                            result.get("verdict", "uncertain"),
                                            result.get("confidence", 0.0),
                                            result.get("evidence", ""),
                                            item_id,
                                        )

                                score = node_data.get("factuality_score", 0)
                                if score is not None and score > 0:
                                    yield ResponsesAgentStreamEvent(
                                        type="response.output_text.delta",
                                        item_id=item_id,
                                        delta=f"✅ Factuality verification complete - score: {score:.1%}\n",
                                    )

                                    # If there are contradictions, emit them
                                    contradictions = (
                                        node_data.get("contradictions") or []
                                    )
                                    if contradictions:
                                        yield ResponsesAgentStreamEvent(
                                            type="response.output_text.delta",
                                            item_id=item_id,
                                            delta=f"⚠️ Found {len(contradictions)} potential contradictions to review\n",
                                        )

                            elif node_name == "reporter":
                                # Emit PHASE marker first
                                phase_event = self._emit_phase_marker(
                                    node_name, item_id, start_time
                                )
                                if phase_event:
                                    yield phase_event

                                # Enhanced reporter content extraction with comprehensive logging
                                report = ""

                                # Primary path: node_data is the full state dict with final_report
                                if isinstance(node_data, dict):
                                    raw_report = node_data.get("final_report", "")
                                    if not raw_report:
                                        # Try alternative locations
                                        raw_report = node_data.get("report", "")
                                        raw_report = raw_report or node_data.get(
                                            "content", ""
                                        )
                                        # Log available keys for debugging
                                        logger.warning(
                                            f"No final_report in reporter output. Available keys: {list(node_data.keys())[:10]}"
                                        )
                                    else:
                                        logger.info(
                                            f"Successfully extracted final_report: {len(raw_report)} characters"
                                        )

                                    # Apply content sanitization to separate JSON from markdown
                                    if raw_report:
                                        sanitization_result = sanitize_agent_content(
                                            raw_report
                                        )
                                        report = sanitization_result.clean_content

                                        # Log sanitization results
                                        if sanitization_result.sanitization_applied:
                                            logger.info(
                                                f"Content sanitization applied: {len(raw_report)} -> {len(report)} chars, "
                                                f"content_type={sanitization_result.content_type.value}, "
                                                f"reasoning_blocks={len(sanitization_result.extracted_reasoning)}, "
                                                f"warnings={len(sanitization_result.warnings)}"
                                            )

                                            # Log warnings if any
                                            for warning in sanitization_result.warnings:
                                                logger.warning(
                                                    f"Content sanitization warning: {warning}"
                                                )

                                            # Emit extracted reasoning as separate metadata events if present
                                            for reasoning_block in (
                                                sanitization_result.extracted_reasoning
                                            ):
                                                reasoning_event = ResponsesAgentStreamEvent(
                                                    type="response.metadata",
                                                    item_id=item_id,
                                                    metadata={
                                                        "type": "reasoning_extracted",
                                                        "content": reasoning_block,
                                                    },
                                                )
                                                yield reasoning_event
                                        else:
                                            logger.info(
                                                f"Content was already clean: {len(report)} chars, "
                                                f"content_type={sanitization_result.content_type.value}"
                                            )
                                    else:
                                        report = ""

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

                                    # Check for content structure issues
                                    if (
                                        "Research Report" in report
                                        and "References" in report
                                    ):
                                        content_between = (
                                            report.split("Research Report")[1].split(
                                                "References"
                                            )[0]
                                            if "References"
                                            in report.split("Research Report")[1]
                                            else report.split("Research Report")[1]
                                        )
                                        if len(content_between.strip()) < 50:
                                            logger.warning(
                                                "Very little content between header and references - possible content extraction issue"
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

            # Emit final done event with metadata
            final_content = "".join(collected_content)
            if not final_content.strip():
                final_content = "Research completed successfully, but no final report was generated."

            # Extract research metadata for UI
            research_metadata = self._extract_streaming_metadata(final_state)

            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item={
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": final_content}],
                    "id": item_id,
                },
                metadata=research_metadata,
            )

            logger.info("Successfully completed predict_stream request")

        except SearchToolsFailedException as e:
            # Graceful degradation for search failures
            logger.warning(f"Search tools failed during streaming: {e.message} - returning partial results")

            # Emit warning event with helpful message
            yield ResponsesAgentStreamEvent(
                type="response.output_text.delta",
                item_id=item_id,
                delta=f"\n\n⚠️ **Search Error**: {e.message}\n\n"
                     f"Some searches failed, but continuing with available results...\n\n"
            )

            # Check if we have any content to return
            final_content = "".join(collected_content)
            if not final_content.strip():
                final_content = f"Research encountered search failures: {e.message}\n\n"
                if not e.details.get("has_brave_key", True):
                    final_content += "💡 Tip: Check that BRAVE_API_KEY is properly configured.\n\n"
                final_content += "Please try rephrasing your query or try again later."

            # Extract what metadata we can
            research_metadata = self._extract_streaming_metadata(final_state) if final_state else {}

            # Emit final response with partial results
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item={
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": final_content}],
                    "id": item_id,
                },
                metadata=research_metadata,
            )

        except Exception as e:
            # COMPREHENSIVE ERROR LOGGING for debugging (DON'T SWALLOW ERRORS!)
            import traceback

            stack_trace = traceback.format_exc()
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "stack_trace": stack_trace,
            }

            logger.error("❌ [STREAM] ===== CRITICAL: STREAMING FAILED WITH EXCEPTION =====")
            logger.error(f"❌ [STREAM] Exception type: {type(e).__name__}")
            logger.error(f"❌ [STREAM] Exception message: {str(e)}")
            logger.error(f"❌ [STREAM] Full traceback:\n{stack_trace}")
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
        if hasattr(self, "_previous_observations"):
            delattr(self, "_previous_observations")

        if hasattr(self, "_previous_search_results"):
            delattr(self, "_previous_search_results")

        if hasattr(self, "_previous_citations"):
            delattr(self, "_previous_citations")

        # Clear any workflow node state
        if hasattr(self, "workflow_nodes") and self.workflow_nodes:
            # Reset any cached data in workflow nodes
            for node_name in [
                "coordinator",
                "planner",
                "researcher",
                "fact_checker",
                "reporter",
            ]:
                if hasattr(self.workflow_nodes, f"{node_name}_agent"):
                    agent = getattr(self.workflow_nodes, f"{node_name}_agent")
                    if hasattr(agent, "_clear_cache"):
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
        diagnostic_report.update(
            {
                "agent_configuration": {
                    "multi_agent_system": True,
                    "validation_enabled": True,
                    "propagation_tracking": True,
                    "critical_fields_monitored": list(StateValidator.CRITICAL_FIELDS),
                    "accumulating_fields_monitored": list(
                        StateValidator.ACCUMULATING_FIELDS
                    ),
                },
                "validation_status": {
                    "state_validator_active": True,
                    "agent_contracts_active": True,
                    "validated_command_active": True,
                    "global_tracker_active": bool(
                        global_propagation_tracker.transitions
                    ),
                },
            }
        )

        # Log summary
        logger.info(
            f"Diagnostic report generated with {len(diagnostic_report)} sections"
        )
        if diagnostic_report.get("data_loss_events"):
            logger.warning(
                f"Found {len(diagnostic_report['data_loss_events'])} data loss events"
            )
        else:
            logger.info("No data loss events detected")

        return diagnostic_report

    def _emit_plan_stream_event(
        self,
        event_type: str,
        *,
        plan_data: Optional[Dict[str, Any]] = None,
        step_id: Optional[str] = None,
        status: Optional[str] = None,
        result: Optional[str] = None,
        description: Optional[str] = None,
        correlation_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        raw_event: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit plan/step events through the event emitter for visualization."""
        if not self.event_emitter:
            logger.debug("🔕 _emit_plan_stream_event called without event_emitter")
            return

        event_type_upper = (event_type or "").upper()
        logger.debug(
            "⚙️ _emit_plan_stream_event invoked",
            extra={
                "event_type": event_type_upper,
                "step_id": step_id,
                "status": status,
                "has_plan_data": bool(plan_data),
            },
        )

        if event_type_upper in {"PLAN_CREATED", "PLAN_UPDATED"}:
            if not plan_data:
                logger.warning("PLAN event emitted without plan_data payload")
                return

            if isinstance(plan_data, dict):
                plan_payload = plan_data
            else:
                formatted_plan = self._format_plan_for_ui(plan_data, {})
                plan_payload = {
                    **formatted_plan,
                    "steps": [
                        {
                            **step,
                            "id": PlanIDGenerator.normalize_id(
                                step.get("id")
                                or step.get("step_id")
                                or PlanIDGenerator.generate_step_id(idx + 1)
                            ),
                            "step_id": PlanIDGenerator.normalize_id(
                                step.get("step_id")
                                or step.get("id")
                                or PlanIDGenerator.generate_step_id(idx + 1)
                            ),
                        }
                        for idx, step in enumerate(formatted_plan.get("steps", []))
                    ],
                }

            logger.info(
                "STREAM: %s plan event with %s steps (plan_keys=%s)",
                event_type_upper,
                len(plan_payload.get("steps", [])),
                list(plan_payload.keys())[:10],
            )

            metadata_event = ResponsesAgentStreamEvent(
                type="response.metadata",
                item_id=str(uuid4()),
                metadata={"planDetails": plan_payload},
            )
            self._pending_intermediate_events.append(metadata_event)

            # Also emit via EventEmitter for logging
            self.event_emitter.emit(
                IntermediateEventType.PLAN_CREATED
                if event_type_upper == "PLAN_CREATED"
                else IntermediateEventType.PLAN_UPDATED,
                data={"plan": plan_payload},
                correlation_id=correlation_id,
                stage_id=stage_id or "planner",
            )
            logger.debug(
                "STREAM: plan event forwarded to event_emitter (%s steps)",
                len(plan_payload.get("steps", [])),
            )

            # Flush plan metadata immediately for streaming UI
            self._flush_pending_intermediate_events()

            # Immediately call stream_callback if available (for tests)
            if hasattr(self, "stream_callback") and self.stream_callback:
                import time

                callback_event = {
                    "type": "intermediate_event",
                    "event": {
                        "event_type": event_type_upper,
                        "data": {"plan": plan_payload},
                        "timestamp": time.time(),
                    },
                }
                self.stream_callback(callback_event)
                logger.debug(f"Called stream_callback for {event_type_upper}")

            return

        if event_type_upper in {"STEP_ACTIVATED", "STEP_COMPLETED", "STEP_FAILED"}:
            if not step_id:
                logger.warning("STEP event emitted without step_id")
                return

            normalized_step_id = PlanIDGenerator.normalize_id(step_id)
            status_payload = status or (
                "completed"
                if event_type_upper == "STEP_COMPLETED"
                else "failed"
                if event_type_upper == "STEP_FAILED"
                else "in_progress"
            )
            if event_type_upper == "STEP_ACTIVATED":
                event_type_enum = IntermediateEventType.STEP_ACTIVATED
            elif event_type_upper == "STEP_COMPLETED":
                event_type_enum = IntermediateEventType.STEP_COMPLETED
            elif event_type_upper == "STEP_FAILED":
                event_type_enum = IntermediateEventType.STEP_FAILED
            else:
                event_type_enum = IntermediateEventType.STEP_ACTIVATED

            logger.info(
                "STREAM: %s event for step=%s status=%s result_preview=%s",
                event_type_upper,
                normalized_step_id,
                status_payload,
                (result[:80] + "…") if isinstance(result, str) and len(result) > 80 else result,
            )

            step_event = self._emit_step_progress_event(
                normalized_step_id,
                status_payload,
                result or "",
                item_id=str(uuid4()),
            )
            if step_event:
                # Add the properly formatted intermediate event to the queue
                self._pending_intermediate_events.append(step_event)

            self.event_emitter.emit(
                event_type_enum,
                data={
                    "step_id": normalized_step_id,
                    "status": status_payload,
                    "result": result or "",
                    "description": description or "",
                },
                correlation_id=correlation_id,
                stage_id=stage_id or "researcher",
            )

            # Ensure step metadata is streamed immediately
            self._flush_pending_intermediate_events()

            # Immediately call stream_callback if available (for tests)
            if hasattr(self, "stream_callback") and self.stream_callback:
                import time

                callback_event = {
                    "type": "intermediate_event",
                    "event": {
                        "event_type": event_type_upper,
                        "data": {
                            "step_id": normalized_step_id,
                            "status": status_payload,
                            "result": result or "",
                            "description": description or "",
                        },
                        "timestamp": time.time(),
                    },
                }
                self.stream_callback(callback_event)
                logger.debug(
                    f"Called stream_callback for {event_type_upper} step {normalized_step_id}"
                )

            # Force immediate flush of step events for real-time UI updates
            if hasattr(self.event_emitter, 'flush_batch'):
                self.event_emitter.flush_batch()
                logger.debug(f"Flushed event emitter for immediate step update: {normalized_step_id}")

            return

        # Fallback: emit generic intermediate event
        self.event_emitter.emit(
            event_type_upper,
            raw_event or {},
            correlation_id=correlation_id,
            stage_id=stage_id,
        )


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
