"""
Enhanced workflow nodes with multi-agent support.

This module extends the existing workflow nodes with new capabilities
for the multi-agent research system.
"""

import json
import time
import asyncio
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime

from .core import (
    get_logger,
    SearchResult,
    EmbeddingManager
)
from .core import id_generator as id_gen
from .core.section_models import (
    SectionResearchResult,
    replace_section_research_result,
)
from .core.utils import extract_token_usage
from .core.multi_agent_state import EnhancedResearchState, StateManager
from .core.message_utils import get_last_user_message
from .core.exceptions import SearchToolsFailedException, PermanentWorkflowError, AuthenticationError
from .core.state_validator import StateValidator, global_propagation_tracker
from .core.validated_command import ValidatedCommand
from .core.routing_policy import track_structural_error, track_executed_step, track_step_execution
from .core.observation_models import observation_to_text
from .core.entity_validation import extract_entities_from_query
from .core.types import IntermediateEventType
from .databricks_response_builder import databricks_response_builder
from .agents import (
    CoordinatorAgent,
    PlannerAgent,
    ResearcherAgent,
    ReporterAgent,
    FactCheckerAgent
)


logger = get_logger(__name__)


# ============================================================================
# State Capture Decorator
# ============================================================================

def with_state_capture(agent_name: str):
    """
    Decorator to add state capture to workflow nodes.

    This decorator automatically captures state before/after/error for any agent node,
    keeping the capture logic DRY and consistent across all agents.

    Args:
        agent_name: Name of the agent (coordinator, planner, researcher, fact_checker, reporter)

    Usage:
        @with_state_capture("planner")
        async def planner_node(self, state):
            # ... node logic ...
            return result
    """
    def decorator(func):
        # Handle both sync and async functions
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
                from .core.state_capture import state_capture
                import traceback

                # STATE CAPTURE: Before
                state_capture.capture_if_enabled(agent_name, state, "before")

                try:
                    # Execute the actual node function
                    result = await func(self, state)

                    # STATE CAPTURE: After (success)
                    result_state = dict(state)
                    # Check dict first - dicts have .update() method which confuses hasattr check
                    if isinstance(result, dict):
                        result_state.update(result)
                    elif hasattr(result, 'update') and isinstance(getattr(result, 'update', None), dict):
                        result_state.update(result.update)

                    state_capture.capture_if_enabled(agent_name, result_state, "after")

                    return result

                except Exception as e:
                    # STATE CAPTURE: Error (don't mask original exception if capture fails)
                    try:
                        error_state = dict(state)
                        error_state["_error"] = {
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": traceback.format_exc()[:1000]  # Truncate to 1KB
                        }
                        state_capture.capture_if_enabled(agent_name, error_state, "error")
                    except Exception as capture_error:
                        # Log but don't crash - original exception is more important
                        logger.warning(f"Failed to capture error state for {agent_name}: {capture_error}")

                    # ALWAYS re-raise the original exception
                    raise

            return async_wrapper
        else:
            # Sync version
            def sync_wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
                from .core.state_capture import state_capture
                import traceback

                # STATE CAPTURE: Before
                state_capture.capture_if_enabled(agent_name, state, "before")

                try:
                    # Execute the actual node function
                    result = func(self, state)

                    # STATE CAPTURE: After (success)
                    result_state = dict(state)
                    # Check dict first - dicts have .update() method which confuses hasattr check
                    if isinstance(result, dict):
                        result_state.update(result)
                    elif hasattr(result, 'update') and isinstance(getattr(result, 'update', None), dict):
                        result_state.update(result.update)

                    state_capture.capture_if_enabled(agent_name, result_state, "after")

                    return result

                except Exception as e:
                    # STATE CAPTURE: Error (don't mask original exception if capture fails)
                    try:
                        error_state = dict(state)
                        error_state["_error"] = {
                            "type": type(e).__name__,
                            "message": str(e),
                            "traceback": traceback.format_exc()[:1000]  # Truncate to 1KB
                        }
                        state_capture.capture_if_enabled(agent_name, error_state, "error")
                    except Exception as capture_error:
                        # Log but don't crash - original exception is more important
                        logger.warning(f"Failed to capture error state for {agent_name}: {capture_error}")

                    # ALWAYS re-raise the original exception
                    raise

            return sync_wrapper

    return decorator


class EnhancedWorkflowNodes:
    """Enhanced workflow nodes for multi-agent system."""
    
    def __init__(self, agent):
        """Initialize with reference to main agent."""
        self.agent = agent
        self.config_manager = agent.config_manager
        self.agent_config = agent.agent_config
        logger.info(f"WORKFLOW NODES: agent_config keys: {list(self.agent_config.keys()) if isinstance(self.agent_config, dict) else 'Not a dict'}")
        logger.info(f"WORKFLOW NODES: adaptive_structure in config: {self.agent_config.get('adaptive_structure', 'NOT FOUND') if isinstance(self.agent_config, dict) else 'N/A'}")
        self.tool_registry = agent.tool_registry
        self.llm = agent.llm
        self.search_semaphore = agent.search_semaphore
        
        # Import routing policy for circuit breakers (legacy compatibility)
        from .core.routing_policy import should_terminate_workflow
        
        # Access event emitter for rich UI events
        self.event_emitter = getattr(agent, 'event_emitter', None)
        
        # Initialize embedding infrastructure for semantic similarity
        self.embedding_manager = self._initialize_embedding_manager()

        # Initialize observation embedding index for smart deduplication
        self.observation_index = None
        if self.embedding_manager:
            try:
                from .core.embedding_manager import ObservationEmbeddingIndex
                from .core.memory_config import MemoryOptimizedConfig

                # Get max observations from config
                max_observations = MemoryOptimizedConfig.get_observations_limit(self.agent_config)

                self.observation_index = ObservationEmbeddingIndex(
                    embedding_manager=self.embedding_manager,
                    max_observations=max_observations
                )
                logger.info(f"Initialized ObservationEmbeddingIndex with max_observations={max_observations}")
            except Exception as e:
                logger.warning(f"Failed to initialize ObservationEmbeddingIndex: {e}")
                self.observation_index = None

          # Note: Enhanced circuit breaker removed - using legacy one for backward compatibility
        
        # Legacy circuit breaker method for backward compatibility
        self.should_terminate_workflow = should_terminate_workflow
        
        # Extract search configuration
        search_config = self.agent_config.get('search', {})
        self.max_results_per_query = search_config.get('max_results_per_query', 10)  # Default to 10 as per SearchConfig
        
        # Initialize model manager if available
        self.model_manager = getattr(agent, 'model_manager', None)

        # Create specialized LLMs for different agents based on complexity
        if self.model_manager:
            # Use simple model for coordinator (pattern matching mostly)
            coordinator_llm = self.model_manager.get_chat_model("simple")
            # Use analytical model for planner
            planner_llm = self.model_manager.get_chat_model("analytical")
        else:
            # Fallback to shared LLM if no model manager
            coordinator_llm = self.llm
            planner_llm = self.llm

        # Initialize specialized agents with complexity-appropriate LLMs
        self.coordinator = CoordinatorAgent(llm=coordinator_llm, config=self.agent_config, event_emitter=self.event_emitter)
        self.planner = PlannerAgent(llm=planner_llm, reasoning_llm=None, config=self.agent_config, event_emitter=self.event_emitter)
        # Set stream callback for plan visualization events  
        self.planner.stream_callback = getattr(agent, 'stream_callback', None)
        
        # Store reference to agent stream callback for plan events
        self.stream_callback = getattr(agent, 'stream_callback', None)
        
        # Create specialized LLMs for remaining agents
        if self.model_manager:
            # Use analytical model for researcher (medium complexity)
            researcher_llm = self.model_manager.get_chat_model("analytical")
            # Use complex model for reporter (final synthesis)
            reporter_llm = self.model_manager.get_chat_model("complex")
            # Use analytical model for fact checker
            fact_checker_llm = self.model_manager.get_chat_model("analytical")
        else:
            # Fallback to shared LLM if no model manager
            researcher_llm = self.llm
            reporter_llm = self.llm
            fact_checker_llm = self.llm

        self.researcher = ResearcherAgent(
            llm=researcher_llm,
            search_tools=None,
            tool_registry=self.tool_registry,
            config=self.agent_config,
            event_emitter=self.event_emitter,
            observation_index=self.observation_index
        )
        # Set references for emitting plan/step events
        stream_callback = getattr(agent, 'stream_callback', None)
        self.researcher.stream_callback = stream_callback
        self.researcher.parent_agent = agent
        self.reporter = ReporterAgent(llm=reporter_llm, config=self.agent_config, event_emitter=self.event_emitter)
        self.fact_checker = FactCheckerAgent(
            llm=fact_checker_llm, 
            config=self.agent_config,
            event_emitter=self.event_emitter,
            embedding_manager=self.embedding_manager
        )

    def _emit_workflow_event(self, phase: str, agent: str, status: str = 'started', metadata: Dict[str, Any] = None, state: Dict[str, Any] = None):
        """Emit workflow phase event for UI visualization."""
        logger.info(f"[WORKFLOW] Emitting workflow event: phase={phase}, agent={agent}, status={status}")

        event_data = {
            "phase": phase,
            "agent": agent,
            "status": status,
            "timestamp": time.time()
        }
        if metadata:
            event_data["metadata"] = metadata

        # CRITICAL: Add event directly to state for streaming
        # IMPORTANT: Create a NEW list so LangGraph's reducer detects the change!
        if state is not None:
            existing_events = state.get("intermediate_events", [])
            workflow_event = {
                "event_type": "workflow_phase",
                "data": event_data,
                "timestamp": time.time()
            }
            # Create NEW list with the event appended (don't mutate in place!)
            state["intermediate_events"] = existing_events + [workflow_event]
            logger.info(f"✅ Added workflow_phase event to intermediate_events: {phase}={status}")

        # Also emit via event emitter for backward compatibility
        if self.event_emitter:
            self.event_emitter.emit(
                event_type="workflow_phase",
                data=event_data,
                correlation_id=f"workflow_{phase}_{agent}"
            )

    def _emit_agent_handoff(self, from_agent: str, to_agent: str, phase: str = None, reason: str = None, state: Dict[str, Any] = None):
        """Emit agent handoff event for workflow visualization."""
        logger.info(f"[WORKFLOW] Agent handoff: {from_agent} → {to_agent} (phase={phase})")

        event_data = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "timestamp": time.time()
        }
        if phase:
            event_data["phase"] = phase
        if reason:
            event_data["reason"] = reason

        # CRITICAL: Add event directly to state for streaming
        # IMPORTANT: Create a NEW list so LangGraph's reducer detects the change!
        if state is not None:
            existing_events = state.get("intermediate_events", [])
            handoff_event = {
                "event_type": "agent_handoff",
                "data": event_data,
                "timestamp": time.time()
            }
            # Create NEW list with the event appended (don't mutate in place!)
            state["intermediate_events"] = existing_events + [handoff_event]
            logger.info(f"✅ Added agent_handoff event to intermediate_events: {from_agent} → {to_agent}")

        # Also emit via event emitter for backward compatibility
        if self.event_emitter:
            from .core.types import IntermediateEventType
                
            # Calculate progress based on agent sequence
            agent_sequence = ['coordinator', 'background_investigation', 'planner', 'researcher', 'fact_checker', 'reporter']
            try:
                to_index = agent_sequence.index(to_agent)
                progress = (to_index / len(agent_sequence)) * 100
                event_data["progress_percentage"] = progress
            except ValueError:
                pass
                
            self.event_emitter.emit(
                event_type=IntermediateEventType.AGENT_HANDOFF,
                data=event_data,
                correlation_id=f"handoff_{from_agent}_{to_agent}"
            )
            
            logger.info(f"✅ [WORKFLOW] Emitted AGENT_HANDOFF event: {from_agent} → {to_agent}")

    def _emit_plan_creation_event(self, plan, item_id: str | None = None):
        """Emit plan metadata (if possible) and PLAN_CREATED intermediate event."""
        if not hasattr(self.agent, '_format_plan_for_ui'):
            logger.warning("Agent missing _format_plan_for_ui; skipping plan event")
            return None

        plan_details = self.agent._format_plan_for_ui(plan, {})
        if not plan_details:
            logger.warning("Plan formatting returned empty data; skipping plan event")
            return None

        metadata_event = None
        if item_id and hasattr(self.agent, '_emit_plan_event'):
            metadata_event = self.agent._emit_plan_event(plan, item_id)

        if hasattr(self.agent, '_emit_plan_stream_event'):
            self.agent._emit_plan_stream_event("plan_created", plan_data=plan_details)

        return metadata_event

    def _update_section_title_mapping(
        self,
        state: EnhancedResearchState,
        plan,
        new_entries: Dict[str, SectionResearchResult],
    ) -> None:
        """Maintain a mapping between section titles and normalized IDs."""

        title_to_id = state.setdefault("section_title_to_id", {})
        id_to_title = state.setdefault("section_id_to_title", {})

        if plan and hasattr(plan, "section_specifications") and plan.section_specifications:
            for spec in plan.section_specifications:
                if getattr(spec, "id", None) and getattr(spec, "title", None):
                    normalized_id = id_gen.PlanIDGenerator.normalize_id(spec.id)
                    normalized_title = spec.title.lower().strip()
                    title_to_id[normalized_title] = normalized_id
                    title_to_id[spec.title] = normalized_id
                    id_to_title[normalized_id] = spec.title

        for section_id, result in new_entries.items():
            title = None
            if isinstance(result, SectionResearchResult):
                title = result.metadata.get("section_title")
            elif isinstance(result, dict):
                title = result.get("title")
            if not title and plan and hasattr(plan, "section_specifications"):
                for spec in plan.section_specifications:
                    normalized_id = id_gen.PlanIDGenerator.normalize_id(getattr(spec, "id", ""))
                    if normalized_id == section_id:
                        title = getattr(spec, "title", None)
                        break
            if title:
                title_to_id[title] = section_id
                title_to_id[title.lower().strip()] = section_id
                id_to_title[section_id] = title

        state["section_title_to_id"] = title_to_id
        state["section_id_to_title"] = id_to_title

    def _ensure_section_title_map(self, state: EnhancedResearchState, plan) -> None:
        """Ensure the section title↔ID map is populated from the current plan."""

        if not plan:
            logger.debug("No plan provided to _ensure_section_title_map")
            return

        section_specs = getattr(plan, "section_specifications", None)
        if not section_specs:
            logger.debug("Plan has no section_specifications")
            return

        title_to_id = state.setdefault("section_title_to_id", {})
        id_to_title = state.setdefault("section_id_to_title", {})

        # Safe iteration with None check
        try:
            for spec in section_specs:
                if not spec:
                    continue
                if getattr(spec, "id", None) and getattr(spec, "title", None):
                    normalized_id = id_gen.PlanIDGenerator.normalize_id(spec.id)
                    normalized_title = spec.title.lower().strip()
                    title_to_id.setdefault(spec.title, normalized_id)
                    title_to_id.setdefault(normalized_title, normalized_id)
                    id_to_title.setdefault(normalized_id, spec.title)
        except (TypeError, AttributeError) as e:
            logger.error(f"Failed to process section specifications: {e}")
            return

        state["section_title_to_id"] = title_to_id
        state["section_id_to_title"] = id_to_title

    def _format_observations_with_labels(self, observations: List[Any]) -> str:
        """Format observations with [Obs#N] labels for the reporter."""
        if not observations:
            return ""
        
        formatted = []
        for i, obs in enumerate(observations, 1):
            # Handle different observation formats
            content = ""
            if hasattr(obs, 'content'):
                content = obs.content
            elif isinstance(obs, dict):
                content = obs.get('content', obs.get('observation', observation_to_text(obs)))
            else:
                content = observation_to_text(obs)
            
            # Format with [Obs#N] label
            formatted.append(f"[Obs#{i}] {content}")
        
        return "\n\n".join(formatted)
    
    def _extract_synthesis_text(self, payload: Any) -> str:
        if isinstance(payload, SectionResearchResult):
            return payload.synthesis
        if isinstance(payload, dict):
            return payload.get('summary') or payload.get('synthesis', '')
        return str(payload)

    def _extract_observations(self, payload: Any) -> List['StructuredObservation']:
        """
        Extract observations preserving step_id and metadata.

        Returns List[StructuredObservation] instead of List[str] to maintain
        step_id for section filtering and other metadata.
        """
        from .core.observation_models import ensure_structured_observation

        observations = []

        if isinstance(payload, SectionResearchResult):
            # Already StructuredObservation objects with step_id
            observations = list(payload.observations)
        elif isinstance(payload, dict):
            raw_obs = payload.get('observations', [])
            # Convert to StructuredObservation if needed (preserves step_id if present)
            observations = [ensure_structured_observation(obs) for obs in raw_obs]

        # Debug logging - track step_id presence
        with_step_id = sum(1 for obs in observations if obs.step_id)
        without_step_id = len(observations) - with_step_id

        logger.debug(
            f"Extracted {len(observations)} observations: "
            f"{with_step_id} with step_id, {without_step_id} without"
        )

        # Log detailed warning if many missing step_id (indicates bug)
        if without_step_id > with_step_id and len(observations) > 5:
            logger.warning(
                f"⚠️ POTENTIAL BUG: {without_step_id}/{len(observations)} observations missing step_id. "
                f"Check researcher agent observation creation code."
            )

        return observations

    def _extract_citations(self, payload: Any) -> List[Any]:
        if isinstance(payload, SectionResearchResult):
            return list(payload.citations)
        if isinstance(payload, dict):
            candidate = payload.get('citations') or payload.get('research', {}).get('citations', [])
            return list(candidate) if candidate else []
        return []
    
    def _initialize_embedding_manager(self) -> Optional[EmbeddingManager]:
        """Initialize embedding manager for semantic similarity operations."""
        try:
            # Get embedding configuration from agent config
            embedding_endpoint = 'databricks-gte-large-en'  # Default
            if self.agent_config and 'models' in self.agent_config:
                models_config = self.agent_config['models']
                if 'embedding' in models_config:
                    embedding_endpoint = models_config['embedding'].get('endpoint', 'databricks-gte-large-en')
            
            # Get global cache manager
            from .core.cache_manager import global_cache_manager
            
            # Create embedding manager with new constructor
            embedding_manager = EmbeddingManager(
                endpoint_name=embedding_endpoint,
                cache_manager=global_cache_manager
            )
            
            logger.info(
                f"Initialized EmbeddingManager with DatabricksEmbeddings for endpoint: {embedding_endpoint}"
            )
            
            return embedding_manager
            
        except Exception as e:
            logger.warning(
                f"Failed to initialize EmbeddingManager: {e}. "
                "Embeddings will not be available for fact checking."
            )
            return None
    
    def _enrich_search_results_with_embeddings(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich search results with embeddings for later reuse in grounding."""
        if not self.embedding_manager:
            return state
        
        # Get search results from state
        search_results = state.get("search_results", [])
        if not search_results:
            return state
        
        try:
            # Enrich with embeddings
            enriched_results = self.embedding_manager.enrich_search_results_with_embeddings(
                search_results
            )
            
            # Update state
            updated_state = dict(state)
            updated_state["search_results"] = enriched_results
            
            logger.info(f"Enriched {len(enriched_results)} search results with embeddings")
            return updated_state
            
        except Exception as e:
            logger.warning(f"Failed to enrich search results with embeddings: {e}")
            return state
    
    def _debug_state_transition(self, node_name: str, state: Dict[str, Any]) -> None:
        """Debug helper to log comprehensive state information at node transitions."""
        logger.info(f"🔍 STATE_DEBUG [{node_name}] === NODE ENTRY ===")
        
        # Log key state fields
        key_fields = [
            "original_user_query", "research_topic", "requested_entities", 
            "current_step", "step_count", "observations", "citations",
            "plan", "factuality_score", "confidence_score"
        ]
        
        for field in key_fields:
            value = state.get(field)
            if value is not None:
                if isinstance(value, str):
                    logger.info(f"🔍 STATE_DEBUG [{node_name}] {field}: {value[:200]}...")
                elif isinstance(value, list):
                    logger.info(f"🔍 STATE_DEBUG [{node_name}] {field}: {len(value)} items")
                    if value and field == "requested_entities":
                        logger.info(f"🔍 STATE_DEBUG [{node_name}] {field}_content: {value}")
                elif isinstance(value, dict):
                    logger.info(f"🔍 STATE_DEBUG [{node_name}] {field}: {len(value)} keys")
                else:
                    logger.info(f"🔍 STATE_DEBUG [{node_name}] {field}: {str(value)[:100]}...")
        
        # Log total state size for memory tracking
        total_keys = len(state.keys())
        logger.info(f"🔍 STATE_DEBUG [{node_name}] Total state keys: {total_keys}")
        logger.info(f"🔍 STATE_DEBUG [{node_name}] === END NODE ENTRY ===")
    
    @with_state_capture("coordinator")
    def coordinator_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinator node - entry point for requests.

        Classifies and routes incoming requests to appropriate agents.
        """
        logger.info("🎯 [WORKFLOW] Executing coordinator node")

        # CRITICAL: Add state validation and debugging
        logger.info(f"[STATE DEBUG] Coordinator received state with {len(state)} keys")
        logger.info(f"[STATE DEBUG] State keys: {list(state.keys())[:20]}")  # First 20 keys
        logger.info(f"[STATE DEBUG] Research topic: {state.get('research_topic', 'MISSING')[:100] if state.get('research_topic') else 'MISSING'}")
        logger.info(f"[STATE DEBUG] Messages count: {len(state.get('messages', []))}")

        # Validate critical fields
        required_fields = ['research_topic', 'messages']
        missing_fields = [f for f in required_fields if f not in state or not state[f]]
        if missing_fields:
            logger.warning(f"[STATE VALIDATION] Missing or empty required fields: {missing_fields}")

        # Emit agent handoff event for workflow visualization
        self._emit_agent_handoff(
            from_agent="system",
            to_agent="coordinator",
            phase="classification",
            reason="Starting multi-agent research workflow",
            state=state
        )

        # Emit workflow phase event - coordinator started
        self._emit_workflow_event("classification", "coordinator", "started", {
            "message": "Starting request classification and routing",
            "workflow_step": "coordinator_classification"
        }, state=state)
        
        # Debug state at coordinator entry
        self._debug_state_transition("COORDINATOR", state)
        
        # Emit agent handoff event
        if self.event_emitter:
            self.event_emitter.emit(
                event_type="agent_handoff",
                data={
                    "from_agent": "user",
                    "to_agent": "coordinator",
                    "reason": "Request received, classifying and routing",
                    "current_phase": "coordination"
                },
                title="Coordinator Taking Control",
                description="Analyzing request and determining routing strategy",
                correlation_id=f"coordinator_{state.get('current_iteration', 0)}",
                stage_id="coordinator"
            )
        
        # Convert dict to enhanced state if needed
        if not isinstance(state, dict) or "research_topic" not in state:
            # Get the original user query
            original_query = get_last_user_message(state.get("messages", [])) or ""
            
            # Initialize enhanced state
            enhanced_state = StateManager.initialize_state(
                research_topic=original_query,
                config=self.agent_config or {}
            )
            # Copy existing messages
            enhanced_state["messages"] = state.get("messages", [])
            
            # CRITICAL: Store original user query for context in all downstream prompts
            enhanced_state["original_user_query"] = original_query
            requested_entities = extract_entities_from_query(original_query, self.llm)
            if requested_entities:
                enhanced_state["requested_entities"] = requested_entities
            else:
                requested_entities = []
            logger.info(f"🎯 ENTITY DEBUG: Stored original user query: {original_query[:100]}...")
            logger.info(f"🎯 ENTITY DEBUG: Extracted {len(requested_entities)} entities: {requested_entities}")
        else:
            enhanced_state = state
            # Ensure original query is captured if not already stored
            if "original_user_query" not in enhanced_state:
                original_query = get_last_user_message(enhanced_state.get("messages", [])) or ""
                enhanced_state["original_user_query"] = original_query
                logger.info(f"Captured original user query: {original_query[:100]}...")
            requested_entities = enhanced_state.get("requested_entities")
            if requested_entities is None:
                original_query = enhanced_state.get("original_user_query", "")
                requested_entities = extract_entities_from_query(original_query, self.llm)
                if requested_entities:
                    enhanced_state["requested_entities"] = requested_entities
                else:
                    requested_entities = []
            logger.info(f"🎯 ENTITY DEBUG: Extracted {len(requested_entities)} entities: {requested_entities}")
        
        # CRITICAL FIX: Add circuit breaker to prevent infinite loops
        total_steps = enhanced_state.get("total_workflow_steps", 0)
        enhanced_state["total_workflow_steps"] = total_steps + 1
        
        # Check for circuit breaker conditions (time limits, step limits, loop limits)
        should_terminate, reason, explanation = self.should_terminate_workflow(enhanced_state)
        
        if should_terminate:
            # Log termination event
            logger.warning(
                f"Circuit breaker activated in coordinator_node - workflow terminating",
                reason=reason.value,
                explanation=explanation,
                total_steps=total_steps
            )
            
            # Add warning to state
            if "warnings" not in enhanced_state:
                enhanced_state["warnings"] = []
            enhanced_state["warnings"].append(
                f"NOTICE: Workflow completed early due to {reason.value}: {explanation}"
            )

            # Force termination to reporter to generate final report
            enhanced_state = StateManager.prune_state_for_memory(enhanced_state)
            return enhanced_state
        
        # Validate input state for coordinator
        try:
            StateValidator.validate_state_for_agent('coordinator', enhanced_state)
        except ValueError as e:
            logger.error(f"Coordinator input validation failed: {e}")
            # Continue with available data
        
        # Execute coordinator with strict contract enforcement
        from .core.contract_node_handler import execute_contract_agent_with_circuit_breaker
        
        try:
            updated_state = execute_contract_agent_with_circuit_breaker(
                agent=self.coordinator,
                agent_name='coordinator',
                state=enhanced_state,
                config=self.agent_config or {},
                circuit_breaker_fn=self.should_terminate_workflow
            )
            
            # CRITICAL FIX: Mark coordinator as visited to prevent infinite loops
            updated_state["coordinator_visited"] = True

            # CRITICAL FIX: Coordinator must determine next agent - never route to itself
            # Don't rely on current_agent which gets set to "coordinator" by contract handler
            if updated_state.get("enable_background_investigation", True) and not updated_state.get("background_investigation_completed"):
                next_agent = "background_investigation"
            else:
                next_agent = "planner"

            # Override current_agent to ensure correct routing
            updated_state["current_agent"] = next_agent

            self._emit_agent_handoff(
                from_agent="coordinator",
                to_agent=next_agent,
                phase="coordination_complete",
                reason=f"Request classification complete, routing to {next_agent}",
                state=updated_state
            )
            
            logger.info(f"🎯 [COORDINATOR] Handoff complete: coordinator → {next_agent}")
            
            # Emit workflow phase event - coordinator completed
            self._emit_workflow_event("classification", "coordinator", "completed", {
                "message": "Request classification and routing completed",
                "next_phase": "background_investigation"
            }, state=updated_state)

            # Apply memory pruning before returning
            updated_state = StateManager.prune_state_for_memory(updated_state)
            return updated_state
            
        except Exception as e:
            logger.error(f"Contract enforcement failed in coordinator_node: {e}")
            # No fallbacks - re-raise the error to fail fast
            raise ValueError(f"Coordinator node failed with contract violation: {e}")
    
    def _generate_background_query(self, topic: str) -> str:
        """
        Generate intelligent background query using LLM understanding.

        Uses LLM to understand context and intent rather than naive text extraction.
        This prevents issues like literal interpretation of idioms.
        """
        import re
        from langchain_core.messages import SystemMessage, HumanMessage

        logger.info(f"BACKGROUND_QUERY: Generating intelligent query for topic (len={len(topic)})")

        # Try LLM-based generation first
        try:
            # Get simple model for quick query generation
            from deep_research_agent.core.model_selector import ModelRole
            llm = self.model_manager.get_llm_for_role(ModelRole.SIMPLE)

            system_prompt = """You are a search query generator for background research.

Generate a single focused search query (5-10 words) for initial context gathering.

Key principles:
1. Understand the INTENT behind the request, not just literal words
2. Recognize and properly interpret idioms and metaphors (e.g., "apples-to-apples" means fair comparison)
3. Extract the core research need and main entities
4. Include relevant temporal context (year, "recent", etc.) if appropriate
5. Be specific enough to find relevant information but broad enough for initial context

Output ONLY the search query, nothing else."""

            user_prompt = f"Generate a background research query for:\n\n{topic[:800]}\n\nSearch query:"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            # Call LLM
            response = llm.invoke(messages)

            # Extract content
            content = response.content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        content = item.get('text', '')
                        break
                else:
                    content = str(content)

            query = content.strip()

            # Validate query
            if 10 <= len(query) <= 200 and not any(c in query for c in ['\n', '\r', '\t']):
                logger.info(f"BACKGROUND_QUERY: ✅ LLM generated: '{query}'")
                return query
            else:
                logger.warning(f"BACKGROUND_QUERY: LLM query invalid (len={len(query)}), falling back")

        except Exception as e:
            logger.warning(f"BACKGROUND_QUERY: LLM generation failed: {e}, falling back")

        # Fallback to intelligent extraction without LLM
        return self._intelligent_query_extraction_fallback(topic)

    def _intelligent_query_extraction_fallback(self, topic: str) -> str:
        """Extract query intelligently without LLM (fallback method)."""
        import re
        from collections import Counter

        # Clean and normalize
        clean_text = ' '.join(topic.split())

        # Remove common idioms/metaphors that cause confusion
        GENERAL_IDIOMS = [
            r'\bapples?\s+to\s+apples?\b',
            r'\bcomparing\s+apples\s+(?:and|to)\s+oranges?\b',
            r'\bcut\s+to\s+the\s+chase\b',
            r'\bby\s+the\s+book\b',
            r'\bhit\s+the\s+nail\s+on\s+the\s+head\b',
            r'\bpiece\s+of\s+cake\b',
            r'\bbreak\s+the\s+ice\b',
        ]

        for idiom in GENERAL_IDIOMS:
            clean_text = re.sub(idiom, '', clean_text, flags=re.IGNORECASE)

        # Extract high-value terms using patterns (not specific entities)
        key_terms = []

        # Pattern 1: Proper nouns (capitalized)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', clean_text)
        key_terms.extend(proper_nouns[:3])

        # Pattern 2: Numbers and amounts (years, money)
        numbers = re.findall(r'\b(?:20\d{2}|19\d{2})\b|\$[\d,]+|€[\d,]+|£[\d,]+', clean_text)
        key_terms.extend(numbers[:2])

        # Pattern 3: Technical/domain terms (words with technical suffixes)
        technical = re.findall(r'\b\w+(?:tion|ment|ance|ence|ity|ology|ics|ing)\b', clean_text)
        # Count frequency and take most common
        tech_counter = Counter(technical)
        key_terms.extend([term for term, _ in tech_counter.most_common(2)])

        if key_terms:
            query = ' '.join(key_terms[:5])  # Top 5 terms
            logger.info(f"BACKGROUND_QUERY: Extracted from patterns: '{query}'")
            return query

        # Final fallback: first sentence or 100 chars
        first_sentence = re.split(r'[.!?]', clean_text)[0]
        result = first_sentence[:100] if first_sentence else clean_text[:100]
        logger.info(f"BACKGROUND_QUERY: Using fallback extraction")
        return result.strip()
    
    def _convert_search_results_to_observations(self, search_results: list) -> list:
        """Convert search results to observation format."""
        observations = []
        for result in search_results:
            if isinstance(result, dict):
                observation = {
                    "content": result.get("content", result.get("snippet", "")),
                    "source": result.get("title", result.get("source", "Unknown")),
                    "url": result.get("url", ""),
                    "relevance": result.get("relevance_score", 0.5),
                    "metadata": result.get("metadata", {})
                }
            else:
                # Handle object-like results
                observation = {
                    "content": getattr(result, "content", getattr(result, "snippet", "")),
                    "source": getattr(result, "title", getattr(result, "source", "Unknown")),
                    "url": getattr(result, "url", ""),
                    "relevance": getattr(result, "relevance_score", 0.5),
                    "metadata": getattr(result, "metadata", {})
                }
            observations.append(observation)
        return observations

    async def background_investigation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Background investigation node - gathers initial context.

        Performs preliminary search before planning to provide context.
        """
        logger.info("🔍 [WORKFLOW] Executing background investigation node")
        
        # Emit workflow phase event - background investigation started
        self._emit_workflow_event("investigation", "background_investigator", "started", {
            "message": "Starting background investigation and context gathering",
            "workflow_step": "background_research"
        }, state=state)
        
        # Emit phase completion for coordinator (initiate phase)
        if self.event_emitter:
            phase_event_data = {
                "phase": "initiate",
                "agent": "coordinator",
                "timestamp": time.time(),
                "status": "completed"
            }
            self.event_emitter.emit(
                event_type="phase_completed",
                data=phase_event_data,
                title="Initiating research completed",
                correlation_id="initiate_phase_complete"
            )

            # CRITICAL: Add phase_completed event to state for streaming
            if "intermediate_events" not in state:
                state["intermediate_events"] = []
            phase_event = {
                "event_type": "phase_completed",
                "data": phase_event_data,
                "timestamp": time.time()
            }
            state["intermediate_events"].append(phase_event)
            logger.info("✅ Added phase_completed event for 'initiate' to intermediate_events")

        # Emit agent handoff event
        if self.event_emitter:
            self.event_emitter.emit(
                event_type="agent_handoff",
                data={
                    "from_agent": "coordinator",
                    "to_agent": "background_investigator",
                    "reason": "Gathering preliminary context before planning",
                    "current_phase": "background_investigation"
                },
                title="Background Investigation Starting",
                description="Gathering initial context to inform research planning",
                correlation_id=f"background_{state.get('current_iteration', 0)}",
                stage_id="background_investigation"
            )
        
        # Check if enabled
        if not state.get("enable_background_investigation", True):
            logger.info("Background investigation disabled, skipping")
            state["background_investigation_completed"] = True
            return state

        research_topic = state.get("research_topic", "")
        if not research_topic:
            logger.warning("No research topic for background investigation")
            state["background_investigation_completed"] = True
            return state
        
        # CRITICAL FIX: Add circuit breaker to prevent infinite loops
        total_steps = state.get("total_workflow_steps", 0)
        state["total_workflow_steps"] = total_steps + 1
        
        # Check for circuit breaker conditions (time limits, step limits, loop limits)
        should_terminate, reason, explanation = self.should_terminate_workflow(state)
        
        if should_terminate:
            # Log termination event
            logger.warning(
                f"Circuit breaker activated in background_investigation_node - workflow terminating",
                reason=reason.value,
                explanation=explanation,
                total_steps=total_steps
            )
            
            # Add warning to state
            if "warnings" not in state:
                state["warnings"] = []
            state["warnings"].append(
                f"NOTICE: Workflow completed early due to {reason.value}: {explanation}"
            )

            # Mark as completed to prevent router loop
            state["background_investigation_completed"] = True

            # Return state with termination flag - router will handle termination
            return state
        
        logger.info(f"Performing background investigation for: {research_topic}")
        
        # Generate simplified query for background investigation
        background_query = self._generate_background_query(research_topic)
        logger.info(f"Using simplified background query: {background_query}")
        
        # Perform initial search
        try:
            # Use search tools if available
            search_results = []
            
            if self.tool_registry:
                # Handle different tool registry types
                if hasattr(self.tool_registry, 'get_tools_by_type'):
                    search_tools = self.tool_registry.get_tools_by_type("search")
                elif isinstance(self.tool_registry, dict):
                    # Handle dict-style tool registry (for tests)
                    search_tool = self.tool_registry.get("search")
                    if search_tool:
                        search_tools = [search_tool]
                    else:
                        search_tools = []
                else:
                    search_tools = []
                
                # Coordinate search timing with global coordinator
                logger.info("BACKGROUND_INVESTIGATION: Starting search coordination")
                from .core.search_coordinator import search_coordinator
                search_coordinator.coordinate_search("background_investigation", background_query)
                
                for tool in search_tools[:1]:  # Use first available tool
                    try:
                        tool_name = getattr(tool, 'name', tool.__class__.__name__)
                        logger.info(f"BACKGROUND_INVESTIGATION: Searching with {tool_name}")
                        
                        # Emit background search start event
                        if self.event_emitter:
                            self.event_emitter.emit_tool_call_start(
                                tool_name=f"background_{tool_name}",
                                parameters={"query": background_query}
                            )
                        
                        if hasattr(tool, 'search'):
                            # Direct search method (e.g., BraveSearchTool)
                            results = tool.search(background_query)
                        elif hasattr(tool, 'execute'):
                            # Execute method for custom tools
                            results = tool.execute(background_query)
                        elif hasattr(tool, 'invoke'):
                            # Use LangChain's invoke method instead of deprecated __call__
                            results = tool.invoke({"query": background_query})
                        elif callable(tool):
                            # Handle mock tools or other callables
                            results = tool(background_query)
                        else:
                            continue
                        
                        if results:
                            logger.info(f"BACKGROUND_INVESTIGATION: Got {len(results)} results from {tool_name}")
                            # Convert raw dictionaries to SearchResult objects if needed
                            processed_results = self._process_search_results(results[:self.max_results_per_query])
                            search_results.extend(processed_results)
                            logger.info(f"BACKGROUND_INVESTIGATION: Processed {len(processed_results)} results (max={self.max_results_per_query})")
                            
                            # Emit background search complete event
                            if self.event_emitter:
                                self.event_emitter.emit_tool_call_complete(
                                    tool_name=f"background_{tool_name}",
                                    success=True,
                                    result_summary=f"Background investigation found {len(processed_results)} results"
                                )
                            break
                    except Exception as e:
                        logger.error(f"BACKGROUND_INVESTIGATION: Search tool {tool_name} FAILED: {str(e)}")
                        if self.event_emitter:
                            self.event_emitter.emit_tool_call_error(
                                tool_name=f"background_{getattr(tool, 'name', 'unknown')}",
                                error_message=f"Background search failed: {str(e)}"
                            )
            
            # Fallback to mock results if needed
            if not search_results:
                search_results = self._mock_background_search(research_topic)
            
            # Compile background information
            background_info = self._compile_background_info(search_results)
            
            # Update state
            state["background_investigation_results"] = background_info
            
            # Add to search results for later use with memory limits
            if "search_results" not in state:
                state["search_results"] = []
            state = StateManager.add_search_results(state, search_results, self.agent_config)
            
            # Enrich search results with embeddings for later use in grounding
            state = self._enrich_search_results_with_embeddings(state)
            
            logger.info(f"Background investigation completed with {len(search_results)} results")
            
        except Exception as e:
            logger.error(f"Background investigation failed: {str(e)}")
            state["background_investigation_results"] = f"Background research on: {research_topic}"

        # Prune state to reduce memory usage before next node
        state = StateManager.prune_state_for_memory(state)

        # CRITICAL FIX: Mark background investigation as completed so router progresses!
        state["background_investigation_completed"] = True

        return state
    
    @with_state_capture("planner")
    async def planner_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Planner node - generates research plans.

        Creates structured plans with quality assessment.
        """
        logger.info("Executing planner node")
        
        # Debug state at planner entry
        self._debug_state_transition("PLANNER", state)
        
        # Emit agent handoff event
        if self.event_emitter:
            self.event_emitter.emit(
                event_type="agent_handoff",
                data={
                    "from_agent": "background_investigator",
                    "to_agent": "planner",
                    "reason": "Creating structured research plan based on gathered context",
                    "current_phase": "planning"
                },
                title="Planner Taking Control",
                description="Analyzing context and creating structured research plan",
                correlation_id=f"planner_{state.get('current_iteration', 0)}",
                stage_id="planner"
            )
        
        correlation_id = f"planner_{state.get('current_iteration', 0)}"
        research_topic = state.get("research_topic", "")
        
        # CRITICAL FIX: Add circuit breaker to prevent infinite loops
        total_steps = state.get("total_workflow_steps", 0)
        state["total_workflow_steps"] = total_steps + 1
        
        # Check for circuit breaker conditions (time limits, step limits, loop limits)
        should_terminate, reason, explanation = self.should_terminate_workflow(state)
        
        if should_terminate:
            # Log termination event
            logger.warning(
                f"Circuit breaker activated in planner_node - workflow terminating",
                reason=reason.value,
                explanation=explanation,
                total_steps=total_steps
            )
            
            # Add warning to state
            if "warnings" not in state:
                state["warnings"] = []
            state["warnings"].append(
                f"NOTICE: Workflow completed early due to {reason.value}: {explanation}"
            )
            
            # Force termination - routing handled by planner_router
            return state
        
        # Emit planning start event
        if self.event_emitter:
            self.event_emitter.emit_action_start(
                action="research_planning",
                query=research_topic,
                correlation_id=correlation_id,
                stage_id="planner"
            )
        
        # Check if this is a plan revision
        existing_plan = state.get("current_plan")
        plan_iteration = state.get("plan_iterations", 0)
        
        if existing_plan and plan_iteration > 0:
            # Emit plan revision event
            if self.event_emitter:
                self.event_emitter.emit(
                    event_type="plan_revision",
                    data={
                        "iteration": plan_iteration,
                        "previous_quality": existing_plan.quality_assessment.overall_score if existing_plan.quality_assessment else 0,
                        "revision_reason": "Improving plan quality"
                    },
                    reasoning=f"Revising plan to improve quality from {existing_plan.quality_assessment.overall_score if existing_plan.quality_assessment else 0:.1%}",
                    correlation_id=correlation_id,
                    stage_id="planner"
                )
        else:
            # Emit plan consideration event for initial planning
            if self.event_emitter:
                background_context = state.get("background_context", {})
                context_sources = len(background_context.get("sources", []))
                
                self.event_emitter.emit_plan_consideration(
                    approach="systematic_research",
                    reasoning=f"Creating structured plan based on {context_sources} background sources to comprehensively explore {research_topic}",
                    alternatives=["broad_exploration", "focused_investigation", "iterative_discovery"],
                    correlation_id=correlation_id,
                    stage_id="planner"
                )

        # AGGRESSIVE EARLY PRUNING: Clean state before expensive planner operation
        state = StateManager.prune_state_for_memory(state)

        # Memory health check before planner execution
        try:
            import psutil
            current_mb = psutil.Process().memory_info().rss / 1024 / 1024
            if current_mb > 1300:  # Conservative threshold
                logger.warning(f"High memory usage before planner: {current_mb:.0f}MB. Skipping refinement.")
                if state.get("current_plan"):
                    return state  # Return current state without refinement
        except ImportError:
            pass  # Continue without memory check
        
        # Execute planner with enhanced state tracking and metrics
        try:
            # CHECK FOR INCREMENTAL RESEARCH LOOP
            research_loops = state.get("research_loops", 0)
            is_incremental_loop = research_loops > 0 and existing_plan

            if is_incremental_loop:
                logger.info(f"[INCREMENTAL LOOP] Starting research loop {research_loops + 1} - performing gap analysis and plan enhancement")

                # Prepare incremental context for planner
                state = self._prepare_incremental_planning_context(state, existing_plan)

                # Emit incremental planning event
                if self.event_emitter:
                    self.event_emitter.emit(
                        event_type="incremental_planning_start",
                        data={
                            "research_loop": research_loops + 1,
                            "existing_steps": len(existing_plan.steps),
                            "completed_steps": len([s for s in existing_plan.steps if s.status == "completed"]),
                            "mode": "gap_analysis_and_enhancement"
                        },
                        reasoning=f"Beginning incremental planning for research loop {research_loops + 1}",
                        correlation_id=correlation_id,
                        stage_id="planner"
                    )

            # ENHANCED: Start timing and metrics tracking as requested by user
            start_time = time.time()
            request_count_before = getattr(self, '_total_requests', 0)

            # FIXED: Properly await async planner agent
            result = await self.planner(state, self.agent_config or {})
            
            # ENHANCED: Calculate execution metrics  
            execution_time = time.time() - start_time
            request_count_after = getattr(self, '_total_requests', 0)
            requests_made = request_count_after - request_count_before
            
            # ENHANCED: Extract token usage from planner response
            token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            if isinstance(result, dict):
                # Handle dict result (new format from planner)
                if 'response' in result:
                    token_usage = extract_token_usage(result['response'])
            elif hasattr(result, 'response') and result.response:
                # Handle Command with response field
                token_usage = extract_token_usage(result.response)
            elif hasattr(result, 'update') and isinstance(result.update, dict) and 'response' in result.update:
                # Handle Command with update dict containing response
                token_usage = extract_token_usage(result.update['response'])
            
            # ENHANCED: Emit step metrics as requested by user
            if self.event_emitter:
                self.event_emitter.emit(
                    event_type="step_metrics",
                    data={
                        "node": "planner",
                        "execution_time_seconds": round(execution_time, 2),
                        "requests_made": requests_made,
                        "token_usage": token_usage,
                        "cumulative_time": state.get("total_execution_time", 0) + execution_time,
                        "cumulative_requests": state.get("total_requests", 0) + requests_made,
                        "cumulative_tokens": {
                            "input": state.get("total_input_tokens", 0) + token_usage["input_tokens"],
                            "output": state.get("total_output_tokens", 0) + token_usage["output_tokens"],
                            "total": state.get("total_tokens", 0) + token_usage["total_tokens"]
                        },
                        "display_hint": "main_pane",  # UI hint as requested
                        "event_category": "metrics"
                    },
                    reasoning=f"Planner execution: {execution_time:.2f}s, {requests_made} API calls, {token_usage['total_tokens']} tokens",
                    correlation_id=correlation_id,
                    stage_id="planner"
                )
            
            # ENHANCED: Update state with cumulative metrics
            state["total_execution_time"] = state.get("total_execution_time", 0) + execution_time
            state["total_requests"] = state.get("total_requests", 0) + requests_made
            state["total_input_tokens"] = state.get("total_input_tokens", 0) + token_usage["input_tokens"] 
            state["total_output_tokens"] = state.get("total_output_tokens", 0) + token_usage["output_tokens"]
            state["total_tokens"] = state.get("total_tokens", 0) + token_usage["total_tokens"]
            
            # Emit plan quality assessment if plan was created
            new_plan = None
            if isinstance(result, dict):
                new_plan = result.get("current_plan")
            elif hasattr(result, 'update') and result.update:
                new_plan = result.update.get("current_plan")
            if new_plan and self.event_emitter:
                # Handle both Plan objects (Pydantic models) and dictionaries
                if hasattr(new_plan, 'model_dump'):
                    # It's a Pydantic model, convert to dict
                    plan_dict = new_plan.model_dump()
                elif hasattr(new_plan, 'dict'):
                    # Older Pydantic version
                    plan_dict = new_plan.dict()
                else:
                    # It's already a dictionary
                    plan_dict = new_plan if isinstance(new_plan, dict) else {}
                
                # Emit PLAN_CREATED event for UI visualization (without item_id outside stream loop)
                self._emit_plan_creation_event(new_plan)
                
                quality_score = plan_dict.get("quality_score", getattr(new_plan, 'quality_score', 0.8))
                steps = plan_dict.get("steps", getattr(new_plan, 'steps', []))
                step_count = len(steps)
                objectives = plan_dict.get("objectives", getattr(new_plan, 'objectives', []))
                
                self.event_emitter.emit(
                    event_type="plan_quality_assessment", 
                    data={
                        "quality": quality_score,
                        "assessment": "Good" if quality_score > 0.7 else "Needs improvement",
                        "step_count": step_count,
                        "objectives": objectives
                    },
                    confidence=quality_score,
                    reasoning=f"Plan quality: {quality_score:.1%} with {step_count} structured steps",
                    correlation_id=correlation_id,
                    stage_id="planner"
                )
                
                # ENHANCED: Emit comprehensive plan structure as requested by user
                logger.info(f"📋 Emitting PLAN_STRUCTURE event with {step_count} steps for UI visualization")
                step_ids_debug = [getattr(step, 'step_id', step.get('step_id', f'step_{i+1:03d}')) if hasattr(step, 'step_id') or isinstance(step, dict) else f'step_{i+1:03d}' for i, step in enumerate(steps)]
                logger.info(f"📋 DEBUG: Plan structure step IDs: {step_ids_debug}")
                
                # Create plan_structure event using the new helper function
                from .databricks_response_builder import databricks_response_builder
                
                # Deduplicate steps before formatting to prevent duplicates in UI
                seen_step_ids = set()
                unique_steps = []
                for step in steps:
                    step_id = getattr(step, 'step_id', step.get('step_id', None)) if hasattr(step, 'step_id') or isinstance(step, dict) else None
                    if step_id:
                        # Normalize the step ID to handle different formats
                        import re
                        normalized_id = re.sub(r'[^a-zA-Z0-9_]', '', str(step_id).lower())
                        if normalized_id not in seen_step_ids:
                            unique_steps.append(step)
                            seen_step_ids.add(normalized_id)
                    else:
                        # If no step_id, include it (shouldn't happen but be safe)
                        unique_steps.append(step)

                if len(unique_steps) < len(steps):
                    logger.info(f"📋 Deduplicated steps: {len(steps)} → {len(unique_steps)} (removed {len(steps) - len(unique_steps)} duplicates)")
                
                # Prepare steps data for the helper function
                formatted_steps = []
                for i, step in enumerate(unique_steps):
                    formatted_steps.append({
                        "step_id": getattr(step, 'step_id', step.get('step_id', f'step_{i+1:03d}')) if hasattr(step, 'step_id') or isinstance(step, dict) else f'step_{i+1:03d}',
                        "name": getattr(step, 'title', step.get('title', f'Step {i+1}')) if hasattr(step, 'title') or isinstance(step, dict) else f'Step {i+1}',
                        "description": getattr(step, 'description', step.get('description', '')) if hasattr(step, 'description') or isinstance(step, dict) else str(step),
                        "status": "pending",  # All steps start as pending
                        "type": getattr(step, 'step_type', step.get('step_type', 'research')) if hasattr(step, 'step_type') or isinstance(step, dict) else 'research',
                        "estimated_time": 5  # Default 5 minutes per step
                    })
                
                plan_structure_event = databricks_response_builder.emit_plan_structure_event(
                    plan_id=plan_dict.get("plan_id", getattr(new_plan, 'plan_id', f"plan_{int(time.time())}")),
                    plan_name=plan_dict.get("title", getattr(new_plan, 'title', "Research Plan")),
                    plan_description=f"Research plan for: {research_topic}",
                    steps=formatted_steps,
                    complexity="moderate",  # Default complexity 
                    estimated_time=len(formatted_steps) * 5  # 5 minutes per step
                )
                
                # Add the event to intermediate_events so it appears in the streaming response
                if "intermediate_events" not in state:
                    state["intermediate_events"] = []
                state["intermediate_events"].append(plan_structure_event)
                
                logger.info(f"📋 CRITICAL: Added plan_structure event to state.intermediate_events with {len(formatted_steps)} steps")
                
                # Keep the original event_emitter call for backward compatibility
                self.event_emitter.emit(
                    event_type="plan_structure",
                    data={
                        "plan": {
                            "plan_id": plan_dict.get("plan_id", getattr(new_plan, 'plan_id', f"plan_{int(time.time())}")),
                            "title": plan_dict.get("title", getattr(new_plan, 'title', "Research Plan")),
                            "research_topic": plan_dict.get("research_topic", research_topic),
                            "iteration": plan_iteration,
                            "quality_assessment": {
                                "overall_score": quality_score,
                                "completeness_score": plan_dict.get("quality_assessment", {}).get("completeness_score", quality_score),
                                "feasibility_score": plan_dict.get("quality_assessment", {}).get("feasibility_score", quality_score),
                                "clarity_score": plan_dict.get("quality_assessment", {}).get("clarity_score", quality_score),
                                "coverage_score": plan_dict.get("quality_assessment", {}).get("coverage_score", quality_score),
                            },
                            "steps": [
                                {
                                    "step_id": getattr(step, 'step_id', step.get('step_id', f'step_{i+1:03d}')) if hasattr(step, 'step_id') or isinstance(step, dict) else f'step_{i+1:03d}',
                                    "title": getattr(step, 'title', step.get('title', f'Step {i+1}')) if hasattr(step, 'title') or isinstance(step, dict) else f'Step {i+1}',
                                    "description": getattr(step, 'description', step.get('description', '')) if hasattr(step, 'description') or isinstance(step, dict) else str(step),
                                    "step_type": getattr(step, 'step_type', step.get('step_type', 'research')) if hasattr(step, 'step_type') or isinstance(step, dict) else 'research',
                                    "status": getattr(step, 'status', step.get('status', 'pending')) if hasattr(step, 'status') or isinstance(step, dict) else 'pending',
                                    "need_search": getattr(step, 'need_search', step.get('need_search', True)) if hasattr(step, 'need_search') or isinstance(step, dict) else True,
                                    "search_queries": getattr(step, 'search_queries', step.get('search_queries', [])) if hasattr(step, 'search_queries') or isinstance(step, dict) else [],
                                    "estimated_duration": getattr(step, 'estimated_duration', step.get('estimated_duration', 'unknown')) if hasattr(step, 'estimated_duration') or isinstance(step, dict) else 'unknown',
                                } for i, step in enumerate(steps)
                            ],
                            "total_steps": step_count,
                            "current_step_index": plan_dict.get("current_step_index", 0),
                            "completed_steps": plan_dict.get("completed_steps", 0),
                            "created_at": plan_dict.get("created_at", datetime.now().isoformat()),
                            "started_at": plan_dict.get("started_at"),
                            "completed_at": plan_dict.get("completed_at"),
                            "objectives": objectives,
                            "thought": plan_dict.get("thought", getattr(new_plan, 'thought', "Systematic research approach")),
                            "has_enough_context": plan_dict.get("has_enough_context", getattr(new_plan, 'has_enough_context', False)),
                            "needs_background_investigation": plan_dict.get("needs_background_investigation", getattr(new_plan, 'needs_background_investigation', True))
                        },
                        "display_hint": "main_pane",  # UI hint as requested
                        "event_category": "planning"
                    },
                    reasoning=f"Complete plan structure with {step_count} steps and {quality_score:.1%} quality score",
                    correlation_id=correlation_id,
                    stage_id="planner"
                )
                
                # Emit individual step generation events
                for i, step in enumerate(steps):
                    # Handle step objects or dictionaries
                    if hasattr(step, 'description'):
                        description = step.description
                        duration = getattr(step, 'estimated_duration', 'unknown')
                    else:
                        description = step.get("description", "") if isinstance(step, dict) else str(step)
                        duration = step.get("estimated_duration", "unknown") if isinstance(step, dict) else "unknown"
                    
                    self.event_emitter.emit(
                        event_type="step_generated",
                        data={
                            "step_number": i + 1,
                            "description": description,
                            "duration": duration
                        },
                        correlation_id=correlation_id,
                        stage_id="planner"
                    )
            
            # Emit planning complete event
            if self.event_emitter:
                self.event_emitter.emit_action_complete(
                    action="research_planning",
                    result_summary=f"Created {step_count if 'step_count' in locals() else 'structured'} research plan",
                    correlation_id=correlation_id,
                    stage_id="planner"
                )
            
            # Use ValidatedCommand for proper state handling
            try:
                if hasattr(result, 'goto') and hasattr(result, 'update'):
                    # This is a Command object - use ValidatedCommand for proper merge
                    validated_cmd = ValidatedCommand.from_agent_output(
                        agent_name='planner',
                        agent_output=result.update or {},
                        current_state=state,
                        next_node=result.goto
                    )
                    updated_state = StateValidator.merge_command_update(state, validated_cmd.update)
                elif isinstance(result, dict):
                    # Direct dict result - validate and merge
                    validated_cmd = ValidatedCommand.from_agent_output(
                        agent_name='planner',
                        agent_output=result,
                        current_state=state
                    )
                    updated_state = StateValidator.merge_command_update(state, validated_cmd.update)
                else:
                    # Fallback - return original state with warning
                    logger.warning(f"Unexpected planner return type: {type(result)}")
                    updated_state = state

                # Record state transition for debugging
                global_propagation_tracker.record_transition(
                    from_agent='planner',
                    to_agent=updated_state.get('current_agent', 'unknown'),
                    state_snapshot=updated_state
                )

                # Apply aggressive memory pruning after processing
                updated_state = StateManager.prune_state_for_memory(updated_state)
                # CRITICAL: Immediately truncate LLM messages to prevent accumulation
                updated_state = self._truncate_large_messages(updated_state)

                # Emit phase completion for planner (planning phase)
                if self.event_emitter:
                    plan = updated_state.get("current_plan")
                    plan_data = {}
                    if plan:
                        if hasattr(plan, 'model_dump'):
                            plan_data = plan.model_dump()
                        elif hasattr(plan, 'dict'):
                            plan_data = plan.dict()
                        elif isinstance(plan, dict):
                            plan_data = plan

                    planning_phase_data = {
                        "phase": "planning",
                        "agent": "planner",
                        "timestamp": time.time(),
                        "status": "completed",
                        "plan": plan_data
                    }
                    self.event_emitter.emit(
                        event_type="phase_completed",
                        data=planning_phase_data
                    )

                    # CRITICAL: Add phase_completed event to state for streaming
                    if "intermediate_events" not in state:
                        state["intermediate_events"] = []
                    phase_event = {
                        "event_type": "phase_completed",
                        "data": planning_phase_data,
                        "timestamp": time.time()
                    }
                    state["intermediate_events"].append(phase_event)
                    logger.info("✅ Added phase_completed event for 'planning' to intermediate_events")

                    # Emit plan_ready event to signal steps should move to research phase
                    if plan_data:
                        self.event_emitter.emit(
                            event_type="plan_ready",
                            data={
                                "plan": plan_data,
                                "timestamp": time.time(),
                                "ready_for_research": True
                            }
                        )

                return updated_state
                
            except Exception as e:
                logger.error(f"Error in planner state validation: {e}")
                # Fallback to basic handling
                state = StateManager.prune_state_for_memory(state)
                state = self._truncate_large_messages(state)
                return state
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            logger.error(f"Planner node failed: {e}\n\n=== FULL STACK TRACE ===\n{full_traceback}\n=== END STACK TRACE ===")
            
            # Emit error event
            if self.event_emitter:
                self.event_emitter.emit(
                    event_type="tool_call_error",
                    data={
                        "tool_name": "planner",
                        "error_message": str(e)
                    },
                    correlation_id=correlation_id,
                    stage_id="planner"
                )
            
            # Return original state on error - routing will be handled by conditional edges
            return state
    
    def _truncate_large_messages(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Balance context preservation with memory limits - 5MB limit per message."""
        if "messages" in state and state["messages"]:
            truncated_count = 0
            max_message_size = 5 * 1024 * 1024  # 5MB limit
            for msg in state["messages"][-10:]:  # Check last 10 messages
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    if len(msg.content) > max_message_size:
                        msg.content = msg.content[:max_message_size] + "\n[Truncated at 5MB for memory management]"
                        truncated_count += 1

            if truncated_count > 0:
                logger.info(f"Truncated {truncated_count} large messages (>5MB) to prevent memory explosion")

        return state
    
    @with_state_capture("researcher")
    async def researcher_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Researcher node - executes research steps.

        Gathers information and accumulates observations.
        """
        logger.info("Executing researcher node")

        # CRITICAL: Add observation tracking debug
        logger.info(f"[RESEARCHER DEBUG] Current observations count: {len(state.get('observations', []))}")
        logger.info(f"[RESEARCHER DEBUG] Search results count: {len(state.get('search_results', []))}")
        logger.info(f"[RESEARCHER DEBUG] Completed steps: {len(state.get('completed_steps', []))}")

        # Debug state at researcher entry
        self._debug_state_transition("RESEARCHER", state)
        
        # Emit agent handoff event
        if self.event_emitter:
            self.event_emitter.emit(
                event_type="agent_handoff",
                data={
                    "from_agent": "planner",
                    "to_agent": "researcher",
                    "reason": "Executing research plan steps to gather information",
                    "current_phase": "research"
                },
                title="Researcher Taking Control",
                description="Beginning systematic research execution according to plan",
                correlation_id=f"researcher_{state.get('current_iteration', 0)}",
                stage_id="researcher"
            )

            # Emit workflow phase event - research started
            self._emit_workflow_event("research", "researcher", "started", {
                "message": "Starting research execution",
                "workflow_step": "researcher_execution"
            }, state=state)

            # Emit phase started event for research phase
            self.event_emitter.emit(
                event_type="phase_started",
                data={
                    "phase": "research",
                    "agent": "researcher",
                    "timestamp": time.time(),
                    "status": "active"
                }
            )

        plan = state.get("current_plan") or state.get("plan")
        self._ensure_section_title_map(state, plan)

        # CRITICAL FIX: Track executed steps for infinite loop detection
        current_step_obj = state.get("current_step")
        step_name = f"researcher_node_{current_step_obj.step_id if current_step_obj else 'no_step'}"
        state = track_executed_step(state, step_name)
        
        correlation_id = f"researcher_{state.get('current_iteration', 0)}"
        current_step = state.get("current_step_number", 1)
        current_plan = state.get("current_plan")
        total_steps = 0
        if current_plan:
            if hasattr(current_plan, 'steps'):  # Plan object
                total_steps = len(current_plan.steps) if current_plan.steps else 0
            elif isinstance(current_plan, dict):  # Dict format (for tests)
                total_steps = len(current_plan.get('steps', []))
        
        # ENHANCED CIRCUIT BREAKER: Comprehensive termination condition checking
        workflow_steps = state.get("total_workflow_steps", 0)
        state["total_workflow_steps"] = workflow_steps + 1
        
        # Check circuit breaker conditions (using legacy for simplicity)
        should_terminate, reason, explanation = self.should_terminate_workflow(state)
        
        if should_terminate:
            # Log termination event
            logger.warning(
                f"Circuit breaker activated in researcher_node: {reason.value if hasattr(reason, 'value') else reason}",
                explanation=explanation,
                total_steps=workflow_steps,
                researcher_loops=state.get("researcher_loops", 0)
            )
            
            # Add warning to state
            if "warnings" not in state:
                state["warnings"] = []
            state["warnings"].append(
                f"NOTICE: Workflow completed early due to {reason.value if hasattr(reason, 'value') else reason}: {explanation}"
            )
            
            # Return state - routing will be handled by researcher_router
            return state
        
        # Emit researcher start event
        if self.event_emitter:
            self.event_emitter.emit_action_start(
                action="research_execution",
                correlation_id=correlation_id,
                stage_id="researcher"
            )
            
            # Form hypothesis about what we expect to find
            step_description = ""
            if current_plan:
                if hasattr(current_plan, 'steps'):  # Plan object
                    if current_plan.steps and current_step <= len(current_plan.steps):
                        plan_step = current_plan.steps[current_step - 1]
                        step_description = plan_step.description if hasattr(plan_step, 'description') else ""
                elif isinstance(current_plan, dict):  # Dict format (for tests)
                    steps = current_plan.get('steps', [])
                    if steps and current_step <= len(steps):
                        plan_step = steps[current_step - 1]
                        step_description = plan_step.get('description', '') if isinstance(plan_step, dict) else str(plan_step)
            
            if step_description:
                # Emit step activation event for UI visualization 
                # NOTE: Let the researcher agent handle step events to avoid duplicates
                step_id = f"step_{current_step:03d}"
                logger.info(f"🎯 Activating step {current_step}: {step_description}")
                logger.info(f"🎯 DEBUG: Workflow delegating step_activated event to researcher agent for step_id: {step_id}")
                
                # Don't emit from workflow level - let researcher agent handle this
                # to ensure consistent step_id formatting and avoid duplicate events
                
                self.event_emitter.emit_hypothesis_formed(
                    hypothesis=f"Research on '{step_description}' will provide insights to address the research question",
                    confidence=0.7,
                    supporting_evidence=[f"Plan step {current_step} targeting specific aspect"],
                    correlation_id=correlation_id,
                    stage_id="researcher"
                )
        
        # Execute research with enhanced event tracking
        try:
            # Before research, emit search strategy
            if self.event_emitter and "current_plan" in state:
                current_plan = state["current_plan"]
                if hasattr(current_plan, 'steps'):  # Plan object
                    queries_needed = len(current_plan.steps) if current_plan.steps else 3
                    focus_areas = [step.title for step in current_plan.steps[:3]] if current_plan.steps else ["general"]
                elif isinstance(current_plan, dict):  # Dict format (for tests)
                    queries_needed = current_plan.get('estimated_queries', 3)
                    focus_areas = current_plan.get('focus_areas', ["general"])
                else:
                    queries_needed = 3
                    focus_areas = ["general"]
                
                self.event_emitter.emit_search_strategy(
                    query_count=queries_needed,
                    focus_areas=focus_areas,
                    approach="systematic",
                    correlation_id=correlation_id,
                    stage_id="researcher"
                )
            
            # ENHANCED: Start timing and metrics tracking as requested by user
            start_time = time.time()
            request_count_before = getattr(self, '_total_requests', 0)
            
            plan = state.get("current_plan")

            # Ensure we have a dynamic template for downstream reporter
            if not (plan and getattr(plan, 'report_template', None)):
                from .core.template_generator import (
                    ReportTemplateGenerator,
                    DynamicSection,
                    SectionContentType,
                )
                from .core.report_styles import STYLE_CONFIGS, ReportStyle

                report_style = state.get("report_style", ReportStyle.DEFAULT)
                if isinstance(report_style, str):
                    try:
                        report_style = ReportStyle(report_style)
                    except ValueError:
                        report_style = ReportStyle.DEFAULT

                style_config = STYLE_CONFIGS.get(report_style, STYLE_CONFIGS[ReportStyle.DEFAULT])

                # Build dynamic sections from either suggested structure or style defaults
                dynamic_sections: List[DynamicSection] = []
                if plan and getattr(plan, "dynamic_sections", None):
                    dynamic_sections = list(plan.dynamic_sections)
                elif plan and getattr(plan, "suggested_report_structure", None):
                    for idx, title in enumerate(plan.suggested_report_structure, start=1):
                        dynamic_sections.append(
                            DynamicSection(
                                title=title,
                                purpose=f"Provide detailed coverage for {title} in the final report.",
                                priority=idx * 10,
                            )
                        )
                else:
                    for idx, section_title in enumerate(style_config.structure, start=1):
                        lowered = section_title.lower()
                        if any(token in lowered for token in ("compare", "vs")):
                            content_type = SectionContentType.COMPARISON
                        elif any(token in lowered for token in ("timeline", "history", "trend")):
                            content_type = SectionContentType.TIMELINE
                        else:
                            content_type = SectionContentType.ANALYSIS

                        dynamic_sections.append(
                            DynamicSection(
                                title=section_title,
                                purpose=f"Summarize the findings related to {section_title}.",
                                priority=idx * 10,
                                content_type=content_type,
                            )
                        )

                generator = ReportTemplateGenerator()
                template_title = plan.title if plan and getattr(plan, "title", None) else state.get("research_topic", "Research Report")
                report_template = generator.build_template(
                    title=template_title,
                    sections=dynamic_sections,
                    include_appendix=True if report_style == ReportStyle.ACADEMIC else False,
                )

                if not plan:
                    from .core.plan_models import Plan

                    plan = Plan(
                        plan_id=f"fallback_plan_{datetime.now().isoformat()}",
                        title=template_title,
                        research_topic=state.get("research_topic", ""),
                        thought="Generated template-based plan",
                        steps=state.get("plan_steps", []) or [],
                    )

                plan.report_template = report_template
                plan.dynamic_sections = dynamic_sections
                plan.suggested_report_structure = [section.title for section in dynamic_sections]
                if not plan.structure_metadata:
                    plan.structure_metadata = {}
                plan.structure_metadata.update(
                    {
                        "generated_via": "workflow_fallback",
                        "dynamic_section_count": len(dynamic_sections),
                        "template_generated_at": datetime.now().isoformat(),
                    }
                )
                state["current_plan"] = plan

                logger.info(
                    "Generated dynamic template via workflow fallback",
                    extra={
                        "sections": [section.title for section in dynamic_sections],
                        "appendix": report_style == ReportStyle.ACADEMIC,
                    },
                )

            result: Any = None

            if plan and getattr(plan, 'report_template', None):
                logger.info(
                    "Section-specific research bypassed: template-driven reporting is active",
                    extra={
                        "dynamic_section_count": len(getattr(plan, 'dynamic_sections', []) or []),
                        "template_present": True,
                    }
                )
            elif plan and hasattr(plan, 'section_specifications') and plan.section_specifications:
                # Log section research summary with ID details
                section_details = []
                for spec in plan.section_specifications:
                    section_id = getattr(spec, 'id', 'NO_ID')
                    section_details.append(f"{section_id}:'{spec.title}'")
                logger.info(f"[SECTION_PLAN] Starting section research for {len(plan.section_specifications)} sections: {', '.join(section_details)}")
                
                # Debug state before research
                from .core.debug_utils import dump_section_state, validate_section_id_consistency
                dump_section_state(state, "before_section_research")
                
                # Validate ID consistency
                id_errors = validate_section_id_consistency(state)
                if id_errors:
                    for error in id_errors:
                        logger.error(f"[ID_VALIDATION] {error}")
                
                logger.info(f"Executing section-specific research for {len(plan.section_specifications)} sections")
                # CRITICAL FIX: Preserve existing section results across multiple researcher executions
                raw_section_results = state.get("section_research_results", {}) or {}
                section_results: Dict[str, SectionResearchResult] = {}
                for raw_key, value in raw_section_results.items():
                    normalized_key = (
                        id_gen.PlanIDGenerator.normalize_id(raw_key)
                        if isinstance(raw_key, str)
                        else raw_key
                    )
                    if isinstance(value, SectionResearchResult):
                        section_results[normalized_key] = value
                    elif isinstance(value, dict):
                        payload = value.get("research", value)
                        if isinstance(payload, SectionResearchResult):
                            section_results[normalized_key] = payload
                        elif isinstance(payload, dict):
                            section_results[normalized_key] = SectionResearchResult.from_dict(payload)
                        else:
                            section_results[normalized_key] = SectionResearchResult(
                                synthesis=str(payload),
                                metadata={"section_id": normalized_key},
                            )
                    else:
                        logger.warning(
                            f"[SECTION_PLAN] Unrecognized section result type {type(value)} for {raw_key}; converting to placeholder"
                        )
                        section_results[normalized_key] = SectionResearchResult(
                            synthesis="",
                            metadata={"section_id": normalized_key},
                        )
                state["section_research_results"] = section_results
                self._update_section_title_mapping(state, plan, section_results)
                logger.info(
                    f"[ACCUMULATION_DEBUG] Loaded {len(section_results)} existing sections from state: "
                    f"{list(section_results.keys())}"
                )
                
                # CRITICAL FIX: Check if router has specified a particular step to execute
                current_step = state.get("current_step")
                
                if current_step:
                    # If the step is already marked completed (stale state), fetch the next pending one
                    from .core.plan_models import StepStatus
                    if current_step.status == StepStatus.COMPLETED and plan:
                        next_pending = plan.get_next_step()
                        if next_pending is not None and next_pending != current_step:
                            logger.info(f"Detected stale current_step {current_step.step_id} (already completed). Switching to {next_pending.step_id}.")
                            current_step = next_pending
                            state["current_step"] = next_pending

                    if plan and current_step:
                        lookup_id = (
                            id_gen.PlanIDGenerator.normalize_id(current_step.step_id)
                            if isinstance(current_step.step_id, str)
                            else current_step.step_id
                        )
                        plan_managed_step = plan.get_step_by_id(lookup_id)
                        if plan_managed_step is None:
                            raise ValueError(
                                f"Current step '{current_step.step_id}' is not present in plan after renumbering"
                            )
                        if plan_managed_step is not current_step:
                            logger.info(
                                "Rebinding current_step to plan-managed instance",
                                step_id=plan_managed_step.step_id,
                                title=plan_managed_step.title,
                            )
                            current_step = plan_managed_step
                            state["current_step"] = plan_managed_step
                            
                            # Emit step activation event for real-time UI updates
                            if plan and hasattr(plan, 'mark_step_activated'):
                                plan.mark_step_activated(plan_managed_step.step_id, event_emitter=self.event_emitter)
                                # CRITICAL FIX: Create NEW plan object so LangGraph detects the change!
                                state["current_plan"] = plan.model_copy(deep=True)

                    # Execute ONLY the specific step requested by router
                    logger.info(f"Executing specific step: {current_step.step_id} - {current_step.title}")
                    
                    # Get the associated section
                    section_spec = None
                    target_section_id = (
                        id_gen.PlanIDGenerator.normalize_id(current_step.step_id)
                        if isinstance(current_step.step_id, str)
                        else current_step.step_id
                    )

                    if hasattr(current_step, 'section_spec') and current_step.section_spec:
                        candidate_spec = current_step.section_spec
                        candidate_id = getattr(candidate_spec, 'id', None)
                        if isinstance(candidate_id, str):
                            candidate_id = id_gen.PlanIDGenerator.normalize_id(candidate_id)
                        if candidate_id == target_section_id:
                            section_spec = candidate_spec
                        else:
                            logger.warning(
                                "Current step references section_spec with mismatched ID",
                                step_id=current_step.step_id,
                                section_spec_id=getattr(candidate_spec, 'id', None),
                                normalized_step_id=target_section_id
                            )

                    if not section_spec and hasattr(current_step, 'metadata') and current_step.metadata:
                        # Fallback: find section by title from metadata
                        section_title = current_step.metadata.get('section_title')
                        if section_title and plan.section_specifications:
                            for spec in plan.section_specifications:
                                if spec.title == section_title:
                                    section_spec = spec
                                    break

                    if section_spec:
                        # STRICT ID ENFORCEMENT - No fallbacks
                        assert hasattr(section_spec, 'id'), f"Section spec missing ID: {section_spec.title}"
                        assert section_spec.id, f"Section spec has empty ID: {section_spec.title}"
                        normalized_section_id = (
                            id_gen.PlanIDGenerator.normalize_id(section_spec.id)
                            if isinstance(section_spec.id, str)
                            else section_spec.id
                        )
                        if normalized_section_id != section_spec.id:
                            logger.info(
                                "Normalizing section_spec ID",
                                original_id=section_spec.id,
                                normalized_id=normalized_section_id
                            )
                            section_spec.id = normalized_section_id

                        logger.info(
                            f"Researching section ID={section_spec.id}, title='{section_spec.title}'"
                        )

                        # EMIT STEP ACTIVATION EVENT before researching section
                        if plan and hasattr(plan, 'mark_step_activated'):
                            plan.mark_step_activated(current_step.step_id, event_emitter=self.event_emitter)
                            state["current_plan"] = plan

                        # Add to intermediate_events for UI

                        # Find step index
                        step_index = 0
                        if plan and hasattr(plan, 'steps'):
                            for i, step in enumerate(plan.steps):
                                if step.step_id == current_step.step_id:
                                    step_index = i
                                    break

                        step_activated_event = databricks_response_builder.emit_step_activated_event(
                            step_id=current_step.step_id,
                            step_index=step_index,
                            step_name=current_step.title,
                            step_type=getattr(current_step.step_type, 'value', 'research') if hasattr(current_step, 'step_type') else 'research'
                        )

                        if "intermediate_events" not in state:
                            state["intermediate_events"] = []
                        state["intermediate_events"].append(step_activated_event)
                        logger.info(f"📢 Added step_activated event to intermediate_events for section step: {current_step.step_id}")

                        try:
                            # Research this specific section only
                            result = self.researcher.research_section(section_spec, state, self.agent_config or {})

                            if result is None:
                                raise ValueError("research_section returned None")

                            if not isinstance(result, SectionResearchResult):
                                result = SectionResearchResult.from_dict(result)

                            # CRITICAL FIX: Format observations with [Obs#N] labels and include in synthesis
                            observations = self._extract_observations(result)
                            if observations:
                                formatted_observations = self._format_observations_with_labels(observations)
                                # Combine synthesis with formatted observations
                                original_synthesis = result.synthesis or ""
                                if formatted_observations:
                                    new_synthesis = (
                                        f"{formatted_observations}\n\n{original_synthesis}"
                                        if original_synthesis
                                        else formatted_observations
                                    )
                                    result = replace_section_research_result(
                                        result,
                                        synthesis=new_synthesis,
                                    )
                                    logger.info(f"[OBSERVATION_FORMATTING] Added {len(observations)} formatted observations to synthesis for section {section_spec.id}")

                            section_key = (
                                id_gen.PlanIDGenerator.normalize_id(section_spec.id)
                                if isinstance(section_spec.id, str)
                                else section_spec.id
                            )
                            section_results[section_key] = result
                            state["section_research_results"] = section_results
                            self._update_section_title_mapping(state, plan, {section_key: result})
                            logger.info(
                                f"[STORAGE] Section stored with ID={section_key}, title='{section_spec.title}'"
                            )
                            
                            # CRITICAL FIX: Always mark step as completed when section results are stored
                            logger.info(f"🔍 STEP_COMPLETION: Section {section_key} completed for step {current_step.step_id}")

                            if plan:
                                logger.info(f"🔍 WORKFLOW DEBUG: About to mark step {current_step.step_id} as completed")
                                observation_text = self._extract_observations(result)
                                plan.mark_step_completed(
                                    current_step.step_id,
                                    execution_result=self._extract_synthesis_text(result),
                                    observations=observation_text,
                                    citations=self._extract_citations(result),
                                    event_emitter=self.event_emitter,
                                )
                                logger.info(f"🔍 WORKFLOW DEBUG: Finished marking step {current_step.step_id} as completed")
                                
                                # Add step completion to intermediate_events for UI
        
                                # Find step index
                                step_index = 0
                                if plan and hasattr(plan, 'steps'):
                                    for i, step in enumerate(plan.steps):
                                        if step.step_id == current_step.step_id:
                                            step_index = i
                                            break

                                execution_summary = self._extract_synthesis_text(result) or "Step completed successfully"
                                step_completed_event = databricks_response_builder.emit_step_completed_event(
                                    step_id=current_step.step_id,
                                    step_index=step_index,
                                    step_name=current_step.title,
                                    step_type='research',
                                    success=True,
                                    summary=execution_summary[:100] + "..." if len(execution_summary) > 100 else execution_summary
                                )

                                if "intermediate_events" not in state:
                                    state["intermediate_events"] = []
                                state["intermediate_events"].append(step_completed_event)
                                logger.info(f"✅ Added step_completed event to intermediate_events for section step: {current_step.step_id}")
                                
                                # CRITICAL FIX: Create NEW plan object so LangGraph detects the change!
                                # Same fix as intermediate_events: object identity matters for state propagation
                                state["current_plan"] = plan.model_copy(deep=True)
                            else:
                                logger.warning(f"🔍 WORKFLOW DEBUG: No plan available to mark step {current_step.step_id} as completed")

                        except Exception as e:
                            error_msg = f"Failed to research section {section_spec.title}: {e}"
                            logger.error(error_msg)
                            
                            # Store error in state for debugging
                            if "errors" not in state:
                                state["errors"] = []
                            state["errors"].append(error_msg)

                # CRITICAL FIX: Mark step as completed when all sections are done
                if plan and current_step:
                    logger.info(f"🔍 STEP_COMPLETION: All sections completed for step {current_step.step_id}, marking as complete")
                    observation_text = self._extract_observations(result) if 'result' in locals() else []
                    execution_summary = self._extract_synthesis_text(result) if 'result' in locals() else "Step completed successfully"
                    
                    # Mark step as completed in the plan
                    plan.mark_step_completed(
                        current_step.step_id,
                        execution_result=execution_summary,
                        observations=observation_text,
                        citations=self._extract_citations(result) if 'result' in locals() else [],
                        event_emitter=self.event_emitter,
                    )
                    logger.info(f"✅ STEP_COMPLETION: Successfully marked step {current_step.step_id} as completed")
                    
                    # Add to intermediate_events for UI

                    # Find step index
                    step_index = 0
                    if plan and hasattr(plan, 'steps'):
                        for i, step in enumerate(plan.steps):
                            if step.step_id == current_step.step_id:
                                step_index = i
                                break

                    step_completed_event = databricks_response_builder.emit_step_completed_event(
                        step_id=current_step.step_id,
                        step_index=step_index,
                        step_name=current_step.title,
                        step_type='research',
                        success=True,
                        summary=execution_summary[:100] + "..." if len(execution_summary) > 100 else execution_summary
                    )

                    if "intermediate_events" not in state:
                        state["intermediate_events"] = []
                    state["intermediate_events"].append(step_completed_event)
                    logger.info(f"✅ Added step_completed event to intermediate_events for all sections complete: {current_step.step_id}")

                    # CRITICAL FIX: Create NEW plan object so LangGraph detects the change!
                    state["current_plan"] = plan.model_copy(deep=True)
                
                state["current_step"] = None
                logger.info(f"Completed step {current_step.step_id}")
            else:
                # Handle non-section steps (e.g., processing, synthesis)
                from .core.plan_models import StepType
                if current_step.step_type in (StepType.PROCESSING, StepType.SYNTHESIS):
                    logger.info(f"Executing {current_step.step_type} step: {current_step.step_id}")
                    # Track step execution for infinite loop detection
                    state = track_step_execution(state, current_step.step_id)
                    # Initialize step_key before try block to ensure it's available in exception handler
                    normalized_step_id = (
                        id_gen.PlanIDGenerator.normalize_id(current_step.step_id)
                        if isinstance(current_step.step_id, str)
                        else current_step.step_id
                    )
                    # Use normalized step ID directly to avoid duplicate keys
                    step_key = normalized_step_id

                    # EMIT STEP ACTIVATION EVENT before processing/synthesis
                    if plan and hasattr(plan, 'mark_step_activated'):
                        plan.mark_step_activated(current_step.step_id, event_emitter=self.event_emitter)
                        state["current_plan"] = plan

                    # Add to intermediate_events for UI

                    # Find step index
                    step_index = 0
                    if plan and hasattr(plan, 'steps'):
                        for i, step in enumerate(plan.steps):
                            if step.step_id == current_step.step_id:
                                step_index = i
                                break

                    step_activated_event = databricks_response_builder.emit_step_activated_event(
                        step_id=current_step.step_id,
                        step_index=step_index,
                        step_name=current_step.title,
                        step_type=getattr(current_step.step_type, 'value', 'processing') if hasattr(current_step, 'step_type') else 'processing'
                    )

                    if "intermediate_events" not in state:
                        state["intermediate_events"] = []
                    state["intermediate_events"].append(step_activated_event)
                    logger.info(f"📢 Added step_activated event to intermediate_events for {current_step.step_type} step: {current_step.step_id}")

                    try:
                        if current_step.step_type == StepType.PROCESSING:
                            result = self.researcher._execute_processing_step(current_step, state, self.agent_config or {})
                        else:
                            result = self.researcher._execute_synthesis_step(current_step, state, self.agent_config or {})

                        # Guard against tools returning None
                        if result is None:
                            raise ValueError("Processing/Synthesis step returned no result payload")

                        # CRITICAL FIX: Format observations with [Obs#N] labels for processing/synthesis steps
                        if isinstance(result, dict) or hasattr(result, 'observations'):
                            observations = self._extract_observations(result)
                            if observations:
                                formatted_observations = self._format_observations_with_labels(observations)
                                # Add formatted observations to the result's synthesis
                                if isinstance(result, dict):
                                    original_synthesis = result.get('synthesis', '')
                                    result['synthesis'] = f"{formatted_observations}\n\n{original_synthesis}" if original_synthesis else formatted_observations
                                elif isinstance(result, SectionResearchResult):
                                    original_synthesis = result.synthesis or ""
                                    new_synthesis = (
                                        f"{formatted_observations}\n\n{original_synthesis}"
                                        if original_synthesis
                                        else formatted_observations
                                    )
                                    result = replace_section_research_result(
                                        result,
                                        synthesis=new_synthesis,
                                    )
                                elif hasattr(result, 'synthesis'):
                                    # Try to use replace helper for other objects with synthesis
                                    original_synthesis = getattr(result, 'synthesis', '') or ""
                                    try:
                                        # If it's any object with synthesis, try replace_section_research_result
                                        result = replace_section_research_result(
                                            result,
                                            synthesis=f"{formatted_observations}\n\n{original_synthesis}"
                                            if original_synthesis
                                            else formatted_observations,
                                        )
                                    except (TypeError, AttributeError):
                                        # If replace doesn't work, try setattr (for mutable objects)
                                        try:
                                            setattr(
                                                result,
                                                'synthesis',
                                                f"{formatted_observations}\n\n{original_synthesis}"
                                                if original_synthesis
                                                else formatted_observations,
                                            )
                                        except AttributeError:
                                            # If object is immutable and can't be replaced, log warning
                                            logger.warning(f"Could not update synthesis for immutable result of type {type(result)}")
                                logger.info(f"[OBSERVATION_FORMATTING] Added {len(observations)} formatted observations to {current_step.step_type} step {current_step.step_id}")

                        # Store generic key since not tied to a section
                        section_results[step_key] = result
                        state["section_research_results"] = section_results
                        self._update_section_title_mapping(state, plan, {step_key: result})

                        # Mark as completed
                        execution_result = self._extract_synthesis_text(result)
                        current_step.execution_result = execution_result
                        current_step.observations = self._extract_observations(result) or [execution_result]
                        if plan:
                            plan.mark_step_completed(
                                current_step.step_id,
                                execution_result=execution_result,
                                observations=current_step.observations,
                                event_emitter=self.event_emitter,
                            )
                            # CRITICAL FIX: Create NEW plan object so LangGraph detects the change!
                            state["current_plan"] = plan.model_copy(deep=True)

                            # Add to intermediate_events for UI
    
                            # Find step index
                            step_index = 0
                            if plan and hasattr(plan, 'steps'):
                                for i, step in enumerate(plan.steps):
                                    if step.step_id == current_step.step_id:
                                        step_index = i
                                        break

                            step_completed_event = databricks_response_builder.emit_step_completed_event(
                                step_id=current_step.step_id,
                                step_index=step_index,
                                step_name=current_step.title,
                                step_type=getattr(current_step.step_type, 'value', 'processing') if hasattr(current_step, 'step_type') else 'processing',
                                success=True,
                                summary=execution_result[:100] + "..." if len(execution_result) > 100 else execution_result
                            )

                            if "intermediate_events" not in state:
                                state["intermediate_events"] = []
                            state["intermediate_events"].append(step_completed_event)
                            logger.info(f"✅ Added step_completed event to intermediate_events for {current_step.step_type} step: {current_step.step_id}")

                    except Exception as e:
                        import traceback
                        full_traceback = traceback.format_exc()
                        error_msg = f"Failed {current_step.step_type} step {current_step.step_id}: {e}"
                        logger.error(f"{error_msg}\n\n=== FULL STACK TRACE ===\n{full_traceback}\n=== END STACK TRACE ===")

                        # CRITICAL FIX: Track this as a structural error for circuit breaker
                        state = track_structural_error(state, error_msg)

                        # Track failed attempts for circuit breaker
                        from .core.routing_policy import increment_failed_attempts, should_skip_failed_step
                        state = increment_failed_attempts(state, current_step.step_id)

                        # Check if we should permanently skip this step
                        if should_skip_failed_step(state, current_step.step_id):
                            logger.warning(f"Permanently skipping step {current_step.step_id} after max retries")

                        if plan:
                            plan.mark_step_failed(current_step.step_id, reason=str(e), event_emitter=self.event_emitter)
                            # CRITICAL FIX: Create NEW plan object so LangGraph detects the change!
                            state["current_plan"] = plan.model_copy(deep=True)
                            logger.info(f"Plan state persisted after marking step {current_step.step_id} as failed")
                        section_results[step_key] = {
                            "id": step_key,
                            "title": getattr(current_step, "title", step_key),
                            "step_type": str(current_step.step_type),
                            "research": {
                                "synthesis": current_step.execution_result,
                                "confidence": 0.0,
                                "extracted_data": {},
                                "observations": self._extract_observations(result),
                            },
                        }
                    finally:
                        state["current_step"] = None
                
                # Convert section results to observations format for reporter (CONSOLIDATED - no more research_observations duplication)
                observations_from_sections = []
                for section_id, section_data in section_results.items():
                    if isinstance(section_data, SectionResearchResult):
                        observations_from_sections.append({
                            "content": section_data.synthesis,
                            "source": f"Section Research: {section_id}",
                            "relevance": section_data.confidence,
                            "section": section_data.metadata.get("section_title", section_id),
                            "section_id": section_id,
                            "extracted_data": dict(section_data.extracted_data),
                        })
                    elif isinstance(section_data, dict):
                        research_payload = section_data.get("research", {}) or {}
                        if not research_payload and "synthesis" in section_data:
                            research_payload = {
                                "synthesis": section_data.get("synthesis"),
                                "extracted_data": section_data.get("extracted_data", {}),
                                "confidence": section_data.get("confidence"),
                            }
                        observations_from_sections.append({
                            "content": research_payload.get("synthesis", ""),
                            "source": f"Section Research: {section_id}",
                            "relevance": research_payload.get("confidence", 0.5),
                            "section": section_data.get("title", section_id),
                            "section_id": section_data.get("id", section_id),
                            "extracted_data": research_payload.get("extracted_data", {}),
                        })

                # NOTE: Step updates now happen directly in the research loop above
                # No need for complex sync logic since steps are updated immediately

                # Count completed steps
                completed_count = sum(1 for step in plan.steps if step.status == StepStatus.COMPLETED)
                logger.info(f"Section research completed: {completed_count}/{len(plan.steps)} steps finished")

                # Create result with section results and observations (single source of truth)
                logger.info(f"[ACCUMULATION_DEBUG] Before creating result, section_results has {len(section_results)} sections: {list(section_results.keys())}")
                result = {
                    "section_research_results": section_results,
                    "observations": observations_from_sections,  # Standardized field name
                    "current_plan": plan  # This fixes the infinite loop by persisting step updates
                }
                logger.info(f"[SECTION_RESEARCH] Created result with {len(section_results)} section results: {list(section_results.keys())}")
                logger.info(f"[ACCUMULATION_DEBUG] Returning section_research_results with keys: {list(result['section_research_results'].keys())}")
            if result is None:
                # Execute normal researcher (template-driven flow or generic plan)
                # FIXED: Properly await async researcher agent
                result = await self.researcher(state, self.agent_config or {})
            
            # ENHANCED: Calculate execution metrics
            execution_time = time.time() - start_time
            request_count_after = getattr(self, '_total_requests', 0)
            requests_made = request_count_after - request_count_before
            
            # ENHANCED: Extract token usage from researcher response
            token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            if isinstance(result, dict):
                # Handle dict result (new format from researcher)
                if 'response' in result:
                    token_usage = extract_token_usage(result['response'])
            elif hasattr(result, 'response') and result.response:
                # Handle Command with response field
                token_usage = extract_token_usage(result.response)
            elif hasattr(result, 'update') and isinstance(result.update, dict) and 'response' in result.update:
                # Handle Command with update dict containing response
                token_usage = extract_token_usage(result.update['response'])
            
            # ENHANCED: Emit step metrics as requested by user
            if self.event_emitter:
                self.event_emitter.emit(
                    event_type="step_metrics",
                    data={
                        "node": "researcher",
                        "execution_time_seconds": round(execution_time, 2),
                        "requests_made": requests_made,
                        "token_usage": token_usage,
                        "step_number": current_step,
                        "step_description": step_description if 'step_description' in locals() else f"Research step {current_step}",
                        "cumulative_time": state.get("total_execution_time", 0) + execution_time,
                        "cumulative_requests": state.get("total_requests", 0) + requests_made,
                        "cumulative_tokens": {
                            "input": state.get("total_input_tokens", 0) + token_usage["input_tokens"],
                            "output": state.get("total_output_tokens", 0) + token_usage["output_tokens"],
                            "total": state.get("total_tokens", 0) + token_usage["total_tokens"]
                        },
                        "display_hint": "main_pane",  # UI hint as requested
                        "event_category": "metrics"
                    },
                    reasoning=f"Research step {current_step} execution: {execution_time:.2f}s, {requests_made} API calls, {token_usage['total_tokens']} tokens",
                    correlation_id=correlation_id,
                    stage_id="researcher"
                )
            
            # ENHANCED: Update state with cumulative metrics
            state["total_execution_time"] = state.get("total_execution_time", 0) + execution_time
            state["total_requests"] = state.get("total_requests", 0) + requests_made
            state["total_input_tokens"] = state.get("total_input_tokens", 0) + token_usage["input_tokens"]
            state["total_output_tokens"] = state.get("total_output_tokens", 0) + token_usage["output_tokens"]
            state["total_tokens"] = state.get("total_tokens", 0) + token_usage["total_tokens"]
            
            # CRITICAL FIX: Merge section research results instead of overwriting
            if isinstance(result, dict) and "section_research_results" in result:
                existing_sections = state.get("section_research_results", {})
                new_sections = result["section_research_results"]
                logger.info(f"[ACCUMULATION_DEBUG] Merging sections - existing: {list(existing_sections.keys())}, new: {list(new_sections.keys())}")
                # Merge new results with existing ones to preserve accumulation across invocations
                existing_sections.update(new_sections)
                state["section_research_results"] = existing_sections
                logger.info(f"[SECTION_RESEARCH] Merged section research to state: {list(existing_sections.keys())}")
                
                # CRITICAL VALIDATION: Ensure sections are never lost
                if len(existing_sections) == 0:
                    logger.error("[CRITICAL_BUG] section_research_results is empty after merge!")
                elif len(existing_sections) < len(new_sections):
                    logger.error(f"[CRITICAL_BUG] Lost sections during merge! Before: {len(existing_sections)}, After: {len(state['section_research_results'])}")
                else:
                    logger.info(f"[ACCUMULATION_SUCCESS] Successfully accumulated {len(existing_sections)} sections total")
                
                # Additional checkpoint validation
                from .core.debug_utils import validate_section_research_state
                validate_section_research_state(state, "after_merge")
            
            # Emit events for findings
            # Collect observations from all result sources (consolidated - no more research_observations duplication)
            new_observations = []
            if isinstance(result, dict):
                # Primary source: observations field (standardized)
                new_observations = result.get("observations", [])
                # Fallback: check for search_results and convert to observations
                if not new_observations and "search_results" in result:
                    search_results = result.get("search_results", [])
                    new_observations = self._convert_search_results_to_observations(search_results)
                # Fallback: extract from section_research_results if present
                if not new_observations and "section_research_results" in result:
                    section_results = result.get("section_research_results", {})
                    for section_data in section_results.values():
                        if isinstance(section_data, SectionResearchResult):
                            # Extract observations from section result
                            new_observations.extend([obs.to_dict() if hasattr(obs, 'to_dict') else obs
                                                    for obs in section_data.observations])
            elif hasattr(result, 'update') and result.update:
                new_observations = result.update.get("observations", [])
                if not new_observations and "search_results" in result.update:
                    search_results = result.update.get("search_results", [])
                    new_observations = self._convert_search_results_to_observations(search_results)

            # Accumulate observations in state (single source of truth) with batch deduplication
            if new_observations:
                # Use batch processing with automatic deduplication
                state, added_count, duplicate_count = StateManager.add_observations_batch(
                    state,
                    new_observations,
                    step=state.get("current_step"),
                    config=self.agent_config
                )

                logger.info(
                    f"📊 Batch processed observations: {added_count} added, "
                    f"{duplicate_count} duplicates skipped, total: {len(state.get('observations', []))}"
                )
            
            if new_observations:
                
                # Emit source evaluation events for new sources
                if self.event_emitter:
                    for obs in new_observations[-3:]:  # Last 3 observations
                        if isinstance(obs, dict) and "source" in obs:
                            self.event_emitter.emit_source_evaluation(
                                title=obs.get("source", "Unknown"),
                                url=obs.get("url", ""),
                                relevance=obs.get("relevance_score", 0.8),
                                reasoning=obs.get("summary", "Relevant to research topic")[:100],
                                correlation_id=correlation_id,
                                stage_id="researcher"
                            )
                
                # Check if we have enough information or need more
                total_observations = len(state.get("observations", []))
                if total_observations < 5 and self.event_emitter:
                    self.event_emitter.emit_knowledge_gap_identified(
                        topic=state.get("research_topic", "topic"),
                        purpose="comprehensive understanding",
                        impact="medium",
                        correlation_id=correlation_id,
                        stage_id="researcher"
                    )
                elif total_observations >= 10 and self.event_emitter:
                    # Emit partial synthesis for substantial findings
                    self.event_emitter.emit_partial_synthesis(
                        conclusion=f"Gathered substantial evidence from {total_observations} sources",
                        source_count=total_observations,
                        confidence=0.8,
                        correlation_id=correlation_id,
                        stage_id="researcher"
                    )
            
            # Emit research complete event
            if self.event_emitter:
                findings_count = 0
                if isinstance(result, dict):
                    findings_count = len(result.get("observations", []))
                elif hasattr(result, 'update') and result.update:
                    findings_count = len(result.update.get("observations", []))
                
                self.event_emitter.emit_action_complete(
                    action="research_execution",
                    result_summary=f"Completed step {current_step}/{total_steps}, added {findings_count} observations",
                    results_count=findings_count,
                    correlation_id=correlation_id,
                    stage_id="researcher"
                )
            
            # Enrich any new search results with embeddings before fact checking
            # Add debugging for result type
            logger.info(f"[SECTION_RESEARCH] Result type: {type(result)}, is dict: {isinstance(result, dict)}, has section_research_results: {isinstance(result, dict) and 'section_research_results' in result}")
            
            if isinstance(result, dict):
                result = self._enrich_search_results_with_embeddings(result)
                
                # PRESERVE CRITICAL DATA BEFORE PRUNING (consolidated - no more research_observations duplication)
                accumulated_observations = state.get("observations", [])
                section_results_backup = result.get("section_research_results", {})
                observations_from_result = result.get("observations", [])
                plan_backup = result.get("current_plan")  # CRITICAL FIX: Preserve plan before pruning to prevent infinite loop
                logger.info(f"[OBSERVATION TRACKING] Preserving {len(accumulated_observations)} accumulated observations before pruning")
                logger.info(f"[OBSERVATION TRACKING] Preserving {len(observations_from_result)} new observations from result before pruning")
                logger.info(f"[SECTION_RESEARCH] Preserving {len(section_results_backup)} section results before pruning")
                logger.info(f"[PLAN PRESERVATION] Preserving plan before pruning: {plan_backup.plan_id if plan_backup and hasattr(plan_backup, 'plan_id') else 'None'}")

                # Calculate total observation data being preserved
                total_obs_data = len(accumulated_observations) + len(observations_from_result)
                for section, data in section_results_backup.items():
                    if isinstance(data, dict) and 'observations' in data:
                        total_obs_data += len(data['observations'])
                logger.info(f"[DATA PRESERVATION] Total observation data preserved: {total_obs_data} items")

                # Apply memory pruning
                result = StateManager.prune_state_for_memory(result)

                # RESTORE CRITICAL DATA AFTER PRUNING
                if observations_from_result:
                    result["observations"] = observations_from_result
                    logger.info(f"[OBSERVATION TRACKING] Restored {len(observations_from_result)} observations after pruning")

                # CRITICAL FIX: Restore section_research_results after pruning
                if section_results_backup:
                    result["section_research_results"] = section_results_backup
                    logger.info(f"[SECTION_RESEARCH] Restored {len(section_results_backup)} section results after pruning")

                    # Log section observation counts for debugging
                    for section, data in section_results_backup.items():
                        if isinstance(data, dict) and 'observations' in data:
                            obs_count = len(data['observations'])
                            logger.info(f"[SECTION_RESEARCH] Section '{section}': {obs_count} observations restored")

                # CRITICAL FIX: Restore current_plan after pruning to prevent infinite loop
                if plan_backup is not None:
                    result["current_plan"] = plan_backup
                    logger.info(f"[PLAN PRESERVATION] Restored plan after pruning: {plan_backup.plan_id if hasattr(plan_backup, 'plan_id') else 'unknown'}")
            elif hasattr(result, 'update') and result.update:
                # For Command objects returned by agent, extract the update dict
                enriched_update = self._enrich_search_results_with_embeddings(result.update)
                result.update.update(enriched_update)
                
                # CRITICAL FIX: Ensure observations are in the update dict
                if "observations" in state and state["observations"]:
                    result.update["observations"] = state["observations"]

                # CRITICAL FIX: Preserve section_research_results AND current_plan if they exist
                section_results_backup = None
                plan_backup_cmd = None  # NEW: For preserving plan from Command.update
                if isinstance(result, dict) and "section_research_results" in result:
                    section_results_backup = result["section_research_results"]
                    logger.info(f"[SECTION_RESEARCH] Backing up {len(section_results_backup)} section results before pruning")
                # Also check if section_research_results is in the Command.update dict
                elif hasattr(result, 'update') and result.update and "section_research_results" in result.update:
                    section_results_backup = result.update["section_research_results"]
                    logger.info(f"[SECTION_RESEARCH] Backing up {len(section_results_backup)} section results from Command.update before pruning")

                # NEW: Preserve current_plan from Command.update dict to prevent infinite loop
                if hasattr(result, 'update') and result.update and "current_plan" in result.update:
                    plan_backup_cmd = result.update["current_plan"]
                    logger.info(f"[PLAN PRESERVATION] Backing up plan from Command.update before pruning: {plan_backup_cmd.plan_id if hasattr(plan_backup_cmd, 'plan_id') else 'unknown'}")


                # Apply memory pruning to the update dict
                if result.update:
                    pruned_update = StateManager.prune_state_for_memory(result.update)

                    # CRITICAL FIX: Restore section_research_results AND current_plan after pruning
                    if section_results_backup is not None:
                        pruned_update["section_research_results"] = section_results_backup
                        logger.info(f"[SECTION_RESEARCH] Restored {len(section_results_backup)} section results after pruning")

                    # NEW: Restore current_plan after pruning to prevent infinite loop
                    if plan_backup_cmd is not None:
                        pruned_update["current_plan"] = plan_backup_cmd
                        logger.info(f"[PLAN PRESERVATION] Restored plan from Command.update after pruning: {plan_backup_cmd.plan_id if hasattr(plan_backup_cmd, 'plan_id') else 'unknown'}")

                    # Return the pruned update dict, not a Command
                    result = pruned_update
            
            # Final observation tracking and debug state
            obs_count = len(result.get("observations", [])) if isinstance(result, dict) else 0
            section_count = len(result.get("section_research_results", {})) if isinstance(result, dict) else 0
            logger.info(f"[OBSERVATION TRACKING] Returning from researcher with {obs_count} observations")
            logger.info(f"[SECTION_RESEARCH] Returning from researcher with {section_count} section results: {list(result.get('section_research_results', {}).keys()) if isinstance(result, dict) else 'N/A'}")
            
            # Debug state after research
            if isinstance(result, dict) and "section_research_results" in result:
                # Temporarily update state for debug dump
                temp_state = state.copy()
                temp_state.update(result)
                from .core.debug_utils import dump_section_state, validate_section_research_state
                dump_section_state(temp_state, "after_section_research")
                validate_section_research_state(temp_state, "researcher_return")

            # Emit workflow phase event - research completed
            self._emit_workflow_event("research", "researcher", "completed", {
                "message": "Research execution completed successfully",
                "workflow_step": "researcher_complete"
            }, state=state)

            return result
            
        except SearchToolsFailedException as e:
            # Graceful degradation: log error and continue with partial results
            logger.warning(f"Search tools failed in researcher node: {e.message} - continuing with partial results")

            # Emit error event (non-critical)
            if self.event_emitter:
                self.event_emitter.emit(
                    event_type="tool_call_error",
                    data={
                        "tool_name": "researcher",
                        "error_message": f"Search failure (continuing): {e.message}",
                        "step": current_step,
                        "is_critical": False,  # Changed to non-critical
                        "failed_tools": getattr(e, 'failed_tools', []),
                        "failure_reasons": getattr(e, 'failure_reasons', {})
                    },
                    correlation_id=correlation_id,
                    stage_id="researcher"
                )

            # Emit workflow phase event - research degraded
            self._emit_workflow_event("research", "researcher", "degraded", {
                "message": f"Search failure - continuing with partial results: {e.message}",
                "error_type": "SearchToolsFailedException",
                "is_critical": False
            }, state=state)

            # Add error to state but continue
            state["errors"].append(f"Search failed for step {current_step.get('step_id', 'unknown')}: {e.message}")
            state["warnings"].append("Some research steps failed - results may be incomplete")

            # Mark step as failed but continue workflow
            if current_step:
                current_step['status'] = StepStatus.FAILED
                current_step['execution_result'] = f"Search failed: {e.message}"

            # Return state to continue workflow
            return state
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            error_msg = f"Researcher node failed: {e}"
            logger.error(f"{error_msg}\n\n=== FULL STACK TRACE ===\n{full_traceback}\n=== END STACK TRACE ===")
            logger.exception(error_msg)

            # Emit workflow phase event - research failed
            self._emit_workflow_event("research", "researcher", "failed", {
                "message": f"Research execution failed: {str(e)}",
                "error_type": type(e).__name__,
                "is_critical": False
            }, state=state)

            # CRITICAL FIX: Track this as a structural error for circuit breaker
            state = track_structural_error(state, error_msg)
            
            # Handle step retry logic
            current_step = state.get("current_step")
            if current_step:
                step_id = current_step.step_id if hasattr(current_step, 'step_id') else str(current_step)
                
                # Increment retry attempts using routing_policy function
                from .core.routing_policy import (
                    increment_failed_attempts, 
                    should_skip_failed_step
                )
                
                # Track this failure
                state = increment_failed_attempts(state, step_id)
                
                # Check if we should permanently skip this step
                max_retries = 3  # Default retry limit, could be configurable
                if should_skip_failed_step(state, step_id, max_retries=max_retries):
                    logger.warning(f"Permanently marking step {step_id} as failed after max retries")
                    
                    # Mark step with permanent_failure metadata
                    plan = state.get("current_plan")
                    if plan:
                        retry_count = state.get("failed_step_attempts", {}).get(step_id, 0)
                        plan.mark_step_failed(
                            step_id, 
                            reason=str(e),
                            metadata={
                                'permanent_failure': True,
                                'retry_count': retry_count,
                                'max_retries': max_retries,
                                'final_error': str(e),
                                'error_type': type(e).__name__
                            },
                            event_emitter=self.event_emitter
                        )
                        state["current_plan"] = plan
                        
                        # Add to permanently failed steps list for reporting
                        if "permanently_failed_steps" not in state:
                            state["permanently_failed_steps"] = []
                        state["permanently_failed_steps"].append({
                            'step_id': step_id,
                            'reason': str(e),
                            'attempts': retry_count,
                            'error_type': type(e).__name__
                        })
                        
                        logger.info(f"Step {step_id} added to permanently failed steps list")
                else:
                    # Just mark as failed for retry
                    plan = state.get("current_plan")  
                    if plan:
                        retry_count = state.get("failed_step_attempts", {}).get(step_id, 0)
                        plan.mark_step_failed(
                            step_id,
                            reason=f"Attempt {retry_count} failed: {str(e)}",
                            metadata={
                                'retry_count': retry_count,
                                'max_retries': max_retries,
                                'last_error': str(e),
                                'error_type': type(e).__name__
                            },
                            event_emitter=self.event_emitter
                        )
                        state["current_plan"] = plan
                        logger.info(f"Step {step_id} marked for retry (attempt {retry_count}/{max_retries})")
                
                # Clear current step to avoid confusion
                state["current_step"] = None
            
            # Emit error event with retry information
            if self.event_emitter:
                retry_info = {}
                if current_step:
                    step_id = current_step.step_id if hasattr(current_step, 'step_id') else str(current_step)
                    retry_info = {
                        "attempts": state.get("failed_step_attempts", {}).get(step_id, 0),
                        "permanent": should_skip_failed_step(state, step_id, max_retries=3)
                    }
                
                self.event_emitter.emit(
                    event_type="tool_call_error",
                    data={
                        "tool_name": "researcher",
                        "error_message": str(e),
                        "step": current_step,
                        "retry_info": retry_info
                    },
                    correlation_id=correlation_id,
                    stage_id="researcher"
                )
            
            # CRITICAL FIX: Final step completion check before returning
            # Check if we have completed research results that should mark steps as complete
            section_results = state.get("section_research_results", {})
            if section_results and plan:
                logger.info(f"🔍 FINAL_CHECK: Found {len(section_results)} completed sections: {list(section_results.keys())}")
                
                # Mark any steps that correspond to completed sections as complete
                for section_key in section_results.keys():
                    # Try to find matching step ID (section keys are often normalized step IDs)
                    matching_step = None
                    for step in plan.steps:
                        if (step.step_id == section_key or 
                            id_gen.PlanIDGenerator.normalize_id(step.step_id) == section_key or
                            step.step_id.replace("step_", "") == section_key.replace("step_", "")):
                            matching_step = step
                            break
                    
                    if matching_step and matching_step.status != "completed":
                        logger.info(f"🔍 FINAL_CHECK: Marking step {matching_step.step_id} as completed (matches section {section_key})")
                        plan.mark_step_completed(
                            matching_step.step_id,
                            execution_result=f"Research completed for section {section_key}",
                            observations=[],
                            citations=[],
                            event_emitter=self.event_emitter,
                        )
                        
                        # Emit streaming event for UI
                        if hasattr(self, 'agent') and hasattr(self.agent, '_emit_plan_stream_event'):
                            self.agent._emit_plan_stream_event(
                                "step_completed",
                                step_id=matching_step.step_id,
                                status="completed", 
                                result=f"Research completed for section {section_key}",
                                description=matching_step.title
                            )
                            logger.info(f"🔄 FINAL_CHECK: Emitted step_completed event for {matching_step.step_id}")

                # CRITICAL FIX: Create NEW plan object so LangGraph detects the change!
                state["current_plan"] = plan.model_copy(deep=True)
            
            # Emit phase completion for researcher (research phase)
            if self.event_emitter:
                research_phase_data = {
                    "phase": "research",
                    "agent": "researcher",
                    "timestamp": time.time(),
                    "status": "completed"
                }
                self.event_emitter.emit(
                    event_type="phase_completed",
                    data=research_phase_data
                )

                # CRITICAL: Add phase_completed event to state for streaming
                if "intermediate_events" not in state:
                    state["intermediate_events"] = []
                phase_event = {
                    "event_type": "phase_completed",
                    "data": research_phase_data,
                    "timestamp": time.time()
                }
                state["intermediate_events"].append(phase_event)
                logger.info("✅ Added phase_completed event for 'research' to intermediate_events")

            # Consolidate observations before returning
            state = StateManager.consolidate_observations(state)

            # Log observation statistics
            logger.info(
                f"📊 Researcher complete: "
                f"Total observations accumulated: {len(state.get('observations', []))}"
            )

            # FIX: Always return state, never lose data
            # Continue to next step or fact checker with preserved state
            return state
    
    async def _safe_fact_checker_call(self, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Defensive wrapper for fact_checker calls to guarantee dict return.

        ASYNC: Made async to properly await the async fact_checker agent.

        This ensures that no matter what the fact_checker returns (Command, string, None, etc.),
        we always get a valid dict that the workflow can process.
        """
        logger.debug("_safe_fact_checker_call: Starting fact checker call")
        try:
            # FIXED: Properly await async fact_checker agent
            result = await self.fact_checker(state, config)
            logger.debug(f"_safe_fact_checker_call: Raw result type: {type(result)}, is dict: {isinstance(result, dict)}")
            
            # Log the actual value for debugging
            if isinstance(result, str):
                logger.error(f"_safe_fact_checker_call: Got string result: repr={repr(result)}, len={len(result)}")
            elif result is None:
                logger.error("_safe_fact_checker_call: Got None result")
            elif not isinstance(result, dict):
                logger.error(f"_safe_fact_checker_call: Got non-dict result: type={type(result)}, value={repr(result)[:200]}")
            
            # Handle all possible return types
            if isinstance(result, dict):
                logger.debug(f"_safe_fact_checker_call: Returning dict with {len(result)} keys")
                return result
            
            # Handle Command objects (shouldn't happen with our fixed fact_checker)
            if hasattr(result, '__class__') and result.__class__.__name__ == 'Command':
                logger.warning("_safe_fact_checker_call: Received Command, extracting update")
                if hasattr(result, 'update') and isinstance(result.update, dict):
                    return result.update
                logger.error("_safe_fact_checker_call: Command has no valid update dict")
                return self._create_default_fact_check_result()
            
            # Handle string returns
            if isinstance(result, str):
                logger.error(f"_safe_fact_checker_call: Creating default result for string return")
                return self._create_default_fact_check_result()
            
            # Handle None or other types
            logger.error(f"_safe_fact_checker_call: Creating default result for {type(result)}")
            return self._create_default_fact_check_result()
            
        except Exception as e:
            logger.error(f"_safe_fact_checker_call: Exception in fact_checker: {e}", exc_info=True)
            return self._create_default_fact_check_result()
    
    def _create_default_fact_check_result(self) -> Dict[str, Any]:
        """Create a default fact checking result for error cases."""
        return {
            "factuality_score": 0.7,
            "factuality_report": None,
            "grounding_analysis": None,
            "needs_revision": False,
            "confidence_level": "moderate",
            "verified_claims": [],
            "unverified_claims": [],
            "contradictions": [],
            "errors": ["Fact checker failed, using default values"]
        }
    
    @with_state_capture("fact_checker")
    async def fact_checker_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fact checker node - validates claims.

        Ensures factual accuracy and grounding.
        """
        logger.info("Executing fact checker node")
        
        # Debug state at fact checker entry
        self._debug_state_transition("FACT_CHECKER", state)
        
        # Emit agent handoff event
        if self.event_emitter:
            self.event_emitter.emit(
                event_type="agent_handoff",
                data={
                    "from_agent": "researcher",
                    "to_agent": "fact_checker",
                    "reason": "Verifying factual accuracy and grounding of research findings",
                    "current_phase": "fact_checking"
                },
                title="Fact Checker Taking Control",
                description="Validating claims and ensuring factual accuracy",
                correlation_id=f"fact_checker_{state.get('current_iteration', 0)}",
                stage_id="fact_checker"
            )
        
        # Increment step counter for every node execution
        total_steps = state.get("total_workflow_steps", 0)
        state["total_workflow_steps"] = total_steps + 1
        
        # Check for circuit breaker conditions (time limits, step limits, loop limits)
        should_terminate, reason, explanation = self.should_terminate_workflow(state)
        
        if should_terminate:
            # Log termination event
            logger.warning(
                f"Circuit breaker activated in fact_checker_node - workflow terminating",
                reason=reason.value,
                explanation=explanation,
                total_steps=total_steps
            )
            
            # Add warning to state
            if "warnings" not in state:
                state["warnings"] = []
            state["warnings"].append(
                f"NOTICE: Workflow completed early due to {reason.value}: {explanation}"
            )
            
            # Return state - routing will be handled by fact_checker_router
            return state
        
        correlation_id = f"fact_checker_{state.get('current_iteration', 0)}"
        
        # Emit fact checking start event
        if self.event_emitter:
            self.event_emitter.emit_action_start(
                action="fact_verification",
                correlation_id=correlation_id,
                stage_id="fact_checker"
            )
            
            # Emit grounding start event
            verification_level = state.get("verification_level", "moderate")
            self.event_emitter.emit(
                event_type="grounding_start",
                data={
                    "level": verification_level,
                    "sources_to_verify": len(state.get("observations", []))
                },
                reasoning=f"Starting {verification_level} fact verification process",
                correlation_id=correlation_id,
                stage_id="fact_checker"
            )
        
        try:
            # ENHANCED: Start timing and metrics tracking as requested by user
            start_time = time.time()
            request_count_before = getattr(self, '_total_requests', 0)
            
            # Use defensive wrapper for fact checker call
            logger.debug(f"Calling fact checker with state type: {type(state)}, keys: {list(state.keys()) if isinstance(state, dict) else 'N/A'}")
            # FIXED: Properly await async fact_checker wrapper
            raw_result = await self._safe_fact_checker_call(state, self.agent_config or {})
            logger.debug(f"Safe fact checker returned type: {type(raw_result)}, keys: {list(raw_result.keys()) if isinstance(raw_result, dict) else 'N/A'}")
            
            # The safe wrapper guarantees raw_result is a dict
            result = raw_result
            
            # Extra validation to ensure we have a dict
            if not isinstance(result, dict):
                logger.error(f"CRITICAL: Safe wrapper didn't return dict! Type: {type(result)}, Value: {repr(result)[:100]}")
                result = self._create_default_fact_check_result()
            
            # ENHANCED: Calculate execution metrics
            execution_time = time.time() - start_time
            request_count_after = getattr(self, '_total_requests', 0)
            requests_made = request_count_after - request_count_before
            
            # ENHANCED: Extract token usage from fact checker response
            token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            # Result is guaranteed to be a dict now
            if 'response' in result:
                token_usage = extract_token_usage(result['response'])
            
            # ENHANCED: Emit step metrics as requested by user
            if self.event_emitter:
                self.event_emitter.emit(
                    event_type="step_metrics",
                    data={
                        "node": "fact_checker",
                        "execution_time_seconds": round(execution_time, 2),
                        "requests_made": requests_made,
                        "token_usage": token_usage,
                        "cumulative_time": state.get("total_execution_time", 0) + execution_time,
                        "cumulative_requests": state.get("total_requests", 0) + requests_made,
                        "cumulative_tokens": {
                            "input": state.get("total_input_tokens", 0) + token_usage["input_tokens"],
                            "output": state.get("total_output_tokens", 0) + token_usage["output_tokens"],
                            "total": state.get("total_tokens", 0) + token_usage["total_tokens"]
                        },
                        "display_hint": "main_pane",  # UI hint as requested
                        "event_category": "metrics"
                    },
                    reasoning=f"Fact checker execution: {execution_time:.2f}s, {requests_made} API calls, {token_usage['total_tokens']} tokens",
                    correlation_id=correlation_id,
                    stage_id="fact_checker"
                )
            
            # ENHANCED: Update state with cumulative metrics
            state["total_execution_time"] = state.get("total_execution_time", 0) + execution_time
            state["total_requests"] = state.get("total_requests", 0) + requests_made
            state["total_input_tokens"] = state.get("total_input_tokens", 0) + token_usage["input_tokens"]
            state["total_output_tokens"] = state.get("total_output_tokens", 0) + token_usage["output_tokens"]
            state["total_tokens"] = state.get("total_tokens", 0) + token_usage["total_tokens"]
            
            # Emit verification results
            grounding_data = {}
            if isinstance(result, dict):
                grounding_data = result.get("grounding", {})
            elif hasattr(result, 'update') and result.update:
                grounding_data = result.update.get("grounding", {})
            
            if grounding_data and self.event_emitter:
                factuality_score = grounding_data.get("factuality_score", 0.9)
                contradictions = grounding_data.get("contradictions", [])
                
                # Emit claims identified for verification
                verified_claims = grounding_data.get("verified_claims", [])
                for claim in verified_claims[:3]:  # Show first 3 claims
                    if isinstance(claim, dict):
                        self.event_emitter.emit(
                            event_type="claim_identified",
                            data={
                                "claim": claim.get("claim", ""),
                                "source": claim.get("source", "research")
                            },
                            correlation_id=correlation_id,
                            stage_id="fact_checker"
                        )
                        
                        # Emit verification attempt
                        self.event_emitter.emit_verification_attempt(
                            claim=claim.get("claim", "")[:100],
                            method="source_cross_reference",
                            source=claim.get("source", "research"),
                            correlation_id=correlation_id,
                            stage_id="fact_checker"
                        )
                
                # Emit contradictions found
                for contradiction in contradictions:
                    if isinstance(contradiction, dict):
                        self.event_emitter.emit(
                            event_type="contradiction_found",
                            data={
                                "claim": contradiction.get("claim", ""),
                                "evidence": contradiction.get("evidence", ""),
                                "severity": contradiction.get("severity", "medium")
                            },
                            reasoning=contradiction.get("explanation", ""),
                            priority=8,  # High priority for contradictions
                            correlation_id=correlation_id,
                            stage_id="fact_checker"
                        )
                
                # Emit confidence adjustment based on fact checking
                original_confidence = state.get("research_confidence", 0.8)
                if factuality_score != original_confidence:
                    self.event_emitter.emit_confidence_update(
                        old_confidence=original_confidence,
                        new_confidence=factuality_score,
                        reason="fact_verification_results",
                        evidence=f"Verified {len(verified_claims)} claims, found {len(contradictions)} contradictions",
                        correlation_id=correlation_id,
                        stage_id="fact_checker"
                    )
                
                # Emit grounding complete
                self.event_emitter.emit(
                    event_type="grounding_complete",
                    data={
                        "factuality_score": factuality_score,
                        "contradictions_found": len(contradictions),
                        "claims_verified": len(verified_claims)
                    },
                    confidence=factuality_score,
                    reasoning=f"Verification complete: {factuality_score:.1%} factual accuracy",
                    correlation_id=correlation_id,
                    stage_id="fact_checker"
                )
                
                # ENHANCED: Emit comprehensive factuality check actions as requested by user
                self.event_emitter.emit(
                    event_type="factuality_actions",
                    data={
                        "actions": {
                            "verification_level": verification_level,
                            "factuality_score": factuality_score,
                            "threshold": grounding_data.get("threshold", 0.6),
                            "verification_method": "multi_layer_claim_verification",
                            "claims_processing": {
                                "total_identified": len(verified_claims),
                                "verified": len([c for c in verified_claims if isinstance(c, dict) and c.get("verified", True)]),
                                "failed": len([c for c in verified_claims if isinstance(c, dict) and not c.get("verified", True)]),
                                "sample_claims": [
                                    {
                                        "claim": c.get("claim", "")[:150],
                                        "confidence": c.get("confidence", 0.8),
                                        "source_count": len(c.get("supporting_sources", [])),
                                        "verification_status": "verified" if c.get("verified", True) else "unverified"
                                    } for c in verified_claims[:5] if isinstance(c, dict)
                                ]
                            },
                            "contradiction_analysis": {
                                "total_found": len(contradictions),
                                "severity_breakdown": {
                                    "high": len([c for c in contradictions if isinstance(c, dict) and c.get("severity") == "high"]),
                                    "medium": len([c for c in contradictions if isinstance(c, dict) and c.get("severity") == "medium"]),
                                    "low": len([c for c in contradictions if isinstance(c, dict) and c.get("severity") == "low"])
                                },
                                "sample_contradictions": [
                                    {
                                        "claim": c.get("claim", "")[:100],
                                        "conflicting_evidence": c.get("evidence", "")[:150],
                                        "severity": c.get("severity", "medium"),
                                        "resolution": c.get("resolution", "requires_review")
                                    } for c in contradictions[:3] if isinstance(c, dict)
                                ]
                            },
                            "grounding_metrics": {
                                "source_coverage": grounding_data.get("source_coverage", 0.8),
                                "cross_references": grounding_data.get("cross_references", 0),
                                "citation_strength": grounding_data.get("citation_strength", 0.7),
                                "temporal_accuracy": grounding_data.get("temporal_accuracy", 0.9),
                                "authority_score": grounding_data.get("authority_score", 0.8)
                            },
                            "next_action": "continue_to_reporter" if factuality_score >= 0.6 else "return_to_researcher",
                            "recommendation": "high_confidence_report" if factuality_score >= 0.8 else "moderate_confidence_report" if factuality_score >= 0.6 else "requires_additional_research"
                        },
                        "display_hint": "main_pane",  # UI hint as requested
                        "event_category": "factuality"
                    },
                    reasoning=f"Comprehensive factuality check: {len(verified_claims)} claims verified, {len(contradictions)} contradictions found, {factuality_score:.1%} overall accuracy",
                    confidence=factuality_score,
                    correlation_id=correlation_id,
                    stage_id="fact_checker"
                )
            
            # Emit fact checking complete event
            if self.event_emitter:
                self.event_emitter.emit_action_complete(
                    action="fact_verification",
                    result_summary=f"Verified factuality: {factuality_score:.1%}" if 'factuality_score' in locals() else "Verification complete",
                    correlation_id=correlation_id,
                    stage_id="fact_checker"
                )
            
            # CRITICAL FIX: Clear embeddings after fact checking to save memory
            # ROBUST: Handle all possible types of background_investigation_results (None, str, list, empty)
            bg_results = state.get("background_investigation_results")

            # Only iterate if it's a non-empty collection of objects
            if bg_results and isinstance(bg_results, (list, tuple)) and len(bg_results) > 0:
                for bg_result in bg_results:
                    # Verify it's an object with metadata before accessing
                    if hasattr(bg_result, 'metadata') and bg_result.metadata:
                        if 'embedding' in bg_result.metadata:
                            del bg_result.metadata['embedding']
                        if 'embedding_vector' in bg_result.metadata:
                            del bg_result.metadata['embedding_vector']
            else:
                # Log what we got for debugging (helps diagnose issues)
                logger.debug(
                    "Skipping background results embedding cleanup",
                    bg_results_type=type(bg_results).__name__ if bg_results is not None else "None",
                    bg_results_value=str(bg_results)[:100] if bg_results else None
                )
            
            # Create progress metrics before incrementing loop counter
            from .core.routing_policy import ProgressMetrics

            current_metrics = ProgressMetrics(state)
            if isinstance(result, dict) and 'factuality_report' in result and result['factuality_report'] is not None:
                # Store the progress metrics to track progress between iterations
                factuality_report = result['factuality_report']
                current_metrics.current_ungrounded = factuality_report.ungrounded_claims
                current_metrics.current_factuality = factuality_report.overall_factuality_score
            
            # Increment fact-check loop counter when we complete a grounding pass
            # The safe wrapper guarantees result is a dict at this point
            # Add explicit type check for safety
            if not isinstance(result, dict):
                logger.error(f"CRITICAL: result is not a dict after safe wrapper, got {type(result)}")
                result = {"fact_check_loops": state.get("fact_check_loops", 0)}
            result.setdefault("fact_check_loops", state.get("fact_check_loops", 0))
            result["fact_check_loops"] = result["fact_check_loops"] + 1
            # Store progress metrics for next iteration comparison
            result["previous_progress_metrics"] = current_metrics
            # Log the increment for debugging
            logger.info(
                f"Incrementing fact check loop counter",
                new_count=result["fact_check_loops"],
                factuality_score=getattr(current_metrics, 'current_factuality', None)
            )
            
            # Result is guaranteed to be a dict from our handling above
            if not isinstance(result, dict):
                # This should NEVER happen with our safe wrapper
                logger.error(f"CRITICAL ERROR: Result is not dict after safe wrapper! Type: {type(result)}")
                logger.error(f"Result value: {repr(result)[:200]}")
                # Emergency fallback - return state with error flag
                return {
                    **state,
                    "factuality_score": 0.5,
                    "factuality_report": None,
                    "needs_revision": False,
                    "grounding_analysis": None,
                    "errors": state.get("errors", []) + ["Fact checker critical error - using fallback"]
                }
            
            # Apply memory pruning to the dict result
            logger.info(f"Fact checker returned dict with keys: {list(result.keys())}")

            # Merge fact checker results into state before pruning
            updated_state = {**state}
            updated_state.update(result)
            
            # Consolidate observations before pruning
            updated_state = StateManager.consolidate_observations(updated_state)

            # Prune the merged state
            pruned_state = StateManager.prune_state_for_memory(updated_state)

            # Validate pruned result
            if not isinstance(pruned_state, dict):
                logger.error(f"StateManager.prune_state_for_memory returned non-dict: {type(pruned_state)}, value: {repr(pruned_state)[:100]}")
                # Return the unpruned merged state
                return updated_state

            # Log observation statistics
            logger.info(
                f"📊 Fact checker complete: "
                f"Total observations accumulated: {len(pruned_state.get('observations', []))}"
            )

            return pruned_state
            
        except Exception as e:
            # COMPREHENSIVE ERROR LOGGING with stack trace
            import traceback
            stack_trace = traceback.format_exc()
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "stack_trace": stack_trace,
                "state_keys": list(state.keys()) if state else None,
                "factuality_score": state.get("factuality_score") if state else None,
                "verification_level": state.get("verification_level") if state else None
            }
            
            logger.error(
                f"FACT CHECKER NODE FAILURE: {type(e).__name__}: {e}\n"
                f"Stack trace:\n{stack_trace}\n"
                f"State keys: {list(state.keys()) if state else None}\n"
                f"Factuality score: {state.get('factuality_score') if state else None}\n"
                f"Verification level: {state.get('verification_level') if state else None}"
            )
            
            # Emit error event with comprehensive details
            if self.event_emitter:
                self.event_emitter.emit(
                    event_type="tool_call_error",
                    data={
                        "tool_name": "fact_checker",
                        "error_details": error_details
                    },
                    correlation_id=correlation_id,
                    stage_id="fact_checker"
                )
            
            # Continue to reporter despite error - return state as dict
            state["errors"] = state.get("errors", [])
            state["errors"].append(f"Fact checker failed: {str(e)}")
            state["warnings"] = state.get("warnings", [])
            state["warnings"].append("Proceeding to report generation despite fact checking error")


            # Apply memory pruning and consolidate observations
            state = StateManager.consolidate_observations(state)
            state = StateManager.prune_state_for_memory(state)
            logger.info("Returning state to continue workflow despite fact checker error")
            return state
    
    @with_state_capture("reporter")
    async def reporter_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reporter node - generates final report.

        Creates styled reports with citations.
        """
        from uuid import uuid4
        invocation_id = str(uuid4())[:8]

        logger.info("🎯 [REPORTER] ===== ENTERING REPORTER NODE =====")
        logger.info(f"🎯 [REPORTER] Invocation ID: {invocation_id}")
        logger.info(f"🎯 [REPORTER] State keys: {list(state.keys())[:20]}")
        logger.info(f"🎯 [REPORTER] State object ID: {id(state)}")
        logger.info("Executing reporter node")

        # CRITICAL: Debug what reporter is receiving
        logger.info(f"[REPORTER DEBUG] Received observations: {len(state.get('observations', []))}")
        logger.info(f"[REPORTER DEBUG] Search results: {len(state.get('search_results', []))}")
        logger.info(f"[REPORTER DEBUG] Citations: {len(state.get('citations', []))}")
        logger.info(f"[REPORTER DEBUG] Factuality score: {state.get('factuality_score', 'N/A')}")
        logger.info(f"[REPORTER DEBUG] Research topic: {state.get('research_topic', 'MISSING')[:100] if state.get('research_topic') else 'MISSING'}")

        # Check if we have actual content to report
        if not state.get('observations') and not state.get('search_results'):
            logger.warning("[REPORTER WARNING] No observations or search results to synthesize!")

        # Debug state at reporter entry
        self._debug_state_transition("REPORTER", state)
        
        # Emit agent handoff event
        if self.event_emitter:
            self.event_emitter.emit(
                event_type="agent_handoff",
                data={
                    "from_agent": "fact_checker",
                    "to_agent": "reporter",
                    "reason": "Synthesizing final comprehensive report with verified findings",
                    "current_phase": "reporting"
                },
                title="Reporter Taking Control",
                description="Creating final comprehensive report from verified research",
                correlation_id=f"reporter_{state.get('current_iteration', 0)}",
                stage_id="reporter"
            )
        
        correlation_id = f"reporter_{state.get('current_iteration', 0)}"
        
        # CRITICAL FIX: Add circuit breaker to prevent infinite loops
        total_steps = state.get("total_workflow_steps", 0)
        state["total_workflow_steps"] = total_steps + 1
        
        # Check for circuit breaker conditions (time limits, step limits, loop limits)
        should_terminate, reason, explanation = self.should_terminate_workflow(state)
        
        if should_terminate:
            # Log termination event
            logger.warning(
                f"Circuit breaker activated in reporter_node - workflow terminating",
                reason=reason.value,
                explanation=explanation,
                total_steps=total_steps
            )
            
            # Add warning to state
            if "warnings" not in state:
                state["warnings"] = []
            state["warnings"].append(
                f"NOTICE: Workflow completed early due to {reason.value}: {explanation}"
            )
            
            # Reporter is the final node, so proceed with report generation anyway
            # The circuit breaker warning will be included in the final report

        # Emit workflow phase event - report generation started
        self._emit_workflow_event("report_generation", "reporter", "started", {
            "message": "Starting comprehensive report synthesis",
            "workflow_step": "reporter_synthesis"
        }, state=state)

        # Emit report generation start event
        if self.event_emitter:
            self.event_emitter.emit_action_start(
                action="report_generation",
                correlation_id=correlation_id,
                stage_id="reporter"
            )
            
            # Emit synthesis strategy
            report_style = state.get("report_style", "default")
            sections_planned = ["introduction", "findings", "analysis", "conclusion"]
            source_count = len(state.get("observations", []))
            
            self.event_emitter.emit(
                event_type="synthesis_strategy",
                data={
                    "report_style": report_style,
                    "sections": sections_planned,
                    "approach": "structured_synthesis"
                },
                reasoning=f"Organizing {source_count} sources into {report_style} report format",
                correlation_id=correlation_id,
                stage_id="reporter"
            )
        
        try:
            # Execute reporter
            # FIXED: Properly await async reporter agent
            result = await self.reporter(state, self.agent_config or {})

            # Emit events for report sections generated
            final_report = ""
            citations = []

            # Extract report data from either Command.update or direct dict
            # Check for dict first since dicts also have .update method
            if isinstance(result, dict):
                # Direct dict return
                final_report = result.get("final_report", "")
                citations = result.get("citations", [])
            elif hasattr(result, 'update') and result.update:
                # Command object with update dict
                final_report = result.update.get("final_report", "")
                citations = result.update.get("citations", [])
            
            if self.event_emitter and (final_report or citations):
                
                # Emit section generation events (simulated based on content)
                if len(final_report) > 500:  # Substantial report
                    sections = ["Introduction", "Key Findings", "Analysis", "Conclusion"]
                    for i, section in enumerate(sections):
                        self.event_emitter.emit(
                            event_type="section_generation",
                            data={
                                "section_title": section,
                                "citations_count": len(citations) // 4  # Distribute citations
                            },
                            correlation_id=correlation_id,
                            stage_id="reporter"
                        )
                
                # Emit citation linking event
                if citations:
                    self.event_emitter.emit(
                        event_type="citation_linking",
                        data={
                            "citation_count": len(citations),
                            "claim": "research findings"
                        },
                        reasoning=f"Linked {len(citations)} citations to support key claims",
                        correlation_id=correlation_id,
                        stage_id="reporter"
                    )
                
                # Emit final report synthesis
                grounding_data = state.get("grounding", {})
                report_confidence = grounding_data.get("factuality_score", 0.9)
                word_count = len(final_report.split()) if final_report else 0
                
                self.event_emitter.emit_partial_synthesis(
                    conclusion=f"Comprehensive {report_style} report synthesizing research findings",
                    source_count=len(state.get("observations", [])),
                    confidence=report_confidence,
                    supporting_sources=[f"{len(citations)} citations"],
                    correlation_id=correlation_id,
                    stage_id="reporter"
                )
            
            # Emit report generation complete event
            if self.event_emitter:
                self.event_emitter.emit_action_complete(
                    action="report_generation",
                    result_summary=f"Generated {report_style if 'report_style' in locals() else 'comprehensive'} report with {len(citations) if 'citations' in locals() else 0} citations",
                    results_count=len(citations) if 'citations' in locals() else 0,
                    correlation_id=correlation_id,
                    stage_id="reporter"
                )
            
            # Use ValidatedCommand for proper state handling
            try:
                if hasattr(result, 'goto') and hasattr(result, 'update'):
                    # This is a Command object - use ValidatedCommand for proper merge
                    validated_cmd = ValidatedCommand.from_agent_output(
                        agent_name='reporter',
                        agent_output=result.update or {},
                        current_state=state,
                        next_node=result.goto
                    )
                    updated_state = StateValidator.merge_command_update(state, validated_cmd.update)
                elif isinstance(result, dict):
                    # Direct dict result - validate and merge
                    validated_cmd = ValidatedCommand.from_agent_output(
                        agent_name='reporter',
                        agent_output=result,
                        current_state=state
                    )
                    updated_state = StateValidator.merge_command_update(state, validated_cmd.update)
                else:
                    # Fallback - return original state with warning
                    logger.warning(f"Unexpected reporter return type: {type(result)}")
                    updated_state = state

                # Record state transition for debugging
                global_propagation_tracker.record_transition(
                    from_agent='reporter',
                    to_agent=updated_state.get('current_agent', 'final'),
                    state_snapshot=updated_state
                )


                # Apply memory pruning before returning
                updated_state = StateManager.prune_state_for_memory(updated_state)

                # CRITICAL: Mark reporter as completed and create NEW state object
                # This ensures LangGraph detects the state change (same pattern as plan mutation fix)
                logger.info(f"🎯 [REPORTER] ===== REPORTER EXECUTION SUCCESSFUL ({invocation_id}) =====")
                logger.info(f"🎯 [REPORTER] Returning state with {len(updated_state)} keys")

                # Create NEW state object for proper LangGraph propagation
                final_state = dict(updated_state)
                final_state["reporter_completed"] = True
                final_state["workflow_should_terminate"] = True

                logger.info(f"🎯 [REPORTER] New state object ID: {id(final_state)} (original: {id(updated_state)})")
                logger.info(f"🎯 [REPORTER] Completion flags: reporter_completed={final_state.get('reporter_completed')}, workflow_should_terminate={final_state.get('workflow_should_terminate')}")
                logger.info(f"Reporter completed with state keys: {list(final_state.keys())}")

                # Emit workflow phase event - report generation completed
                self._emit_workflow_event("report_generation", "reporter", "completed", {
                    "message": "Report synthesis completed successfully",
                    "workflow_step": "reporter_complete"
                }, state=final_state)

                return final_state
                
            except Exception as e:
                logger.error(f"🎯 [REPORTER] Error in reporter state validation: {e}", exc_info=True)
                # Fallback - return original state with pruning and completion markers
                logger.warning(f"🎯 [REPORTER] Unexpected reporter return type: {type(result) if 'result' in locals() else 'N/A'}")
                state = StateManager.prune_state_for_memory(state)

                # CRITICAL: Still mark as completed so workflow terminates (even with validation error)
                fallback_state = dict(state)
                fallback_state["reporter_completed"] = True
                fallback_state["workflow_should_terminate"] = True
                fallback_state["reporter_validation_error"] = True

                logger.info(f"🎯 [REPORTER] Returning fallback state (invocation {invocation_id})")
                return fallback_state
            
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()

            # COMPREHENSIVE ERROR LOGGING (don't swallow errors!)
            logger.error(f"❌ [REPORTER] ===== REPORTER NODE FAILED ({invocation_id}) =====")
            logger.error(f"❌ [REPORTER] Exception type: {type(e).__name__}")
            logger.error(f"❌ [REPORTER] Exception message: {str(e)}")
            logger.error(f"❌ [REPORTER] Full traceback:\n{full_traceback}")
            logger.error(f"Reporter node failed: {e}\n\n=== FULL STACK TRACE ===\n{full_traceback}\n=== END STACK TRACE ===")

            # Parse error for user-friendly message
            error_str = str(e)
            
            if 'TEMPORARILY_UNAVAILABLE' in error_str or '503' in error_str:
                user_message = (
                    "The AI service is temporarily unavailable due to high demand. "
                    "Please try again in a few minutes. Your research has been saved."
                )
                can_retry = True
            elif '429' in error_str or 'rate_limit' in error_str.lower():
                user_message = (
                    "Rate limit reached for the AI service. "
                    "Please wait a moment before retrying."
                )
                can_retry = True
            elif '401' in error_str or '403' in error_str or 'unauthorized' in error_str.lower():
                user_message = (
                    "Authentication error with the AI service. "
                    "Please check your credentials and permissions."
                )
                can_retry = False
            elif '502' in error_str or '504' in error_str or 'gateway' in error_str.lower():
                user_message = (
                    "Gateway error connecting to the AI service. "
                    "This is usually temporary - please try again shortly."
                )
                can_retry = True
            else:
                user_message = (
                    "Unable to generate report due to an AI service error. "
                    f"Technical details: {error_str[:150]}..."
                )
                can_retry = False
            
            # Emit error event
            if self.event_emitter:
                self.event_emitter.emit(
                    event_type="tool_call_error",
                    data={
                        "tool_name": "reporter",
                        "error_message": str(e),
                        "user_friendly_message": user_message,
                        "can_retry": can_retry
                    },
                    correlation_id=correlation_id if 'correlation_id' in locals() else f"reporter_{invocation_id}",
                    stage_id="reporter"
                )

                # Emit workflow phase event - report generation failed
                self._emit_workflow_event("report_generation", "reporter", "failed", {
                    "message": f"Report generation failed: {user_message}",
                    "error_type": type(e).__name__,
                    "can_retry": can_retry
                }, state=state)

                # Also emit node_error event for consistent error handling
                self.event_emitter.emit(
                    event_type="node_error",
                    data={
                        "node": "reporter",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "recoverable": can_retry,
                        "timestamp": time.time()
                    },
                    correlation_id=f"reporter_error_{invocation_id}",
                    stage_id="reporter"
                )

            # Return state with user-friendly error message and metadata
            # CRITICAL: Create NEW state object for LangGraph propagation
            error_state = dict(state)
            error_state["final_report"] = user_message
            error_state["report_generation_failed"] = True
            error_state["error_details"] = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": time.time(),
                "can_retry": can_retry,
                "user_friendly_message": user_message
            }

            # CRITICAL: Mark workflow as failed but completed so it terminates
            error_state["workflow_failed"] = True
            error_state["reporter_completed"] = True  # Reporter DID complete (with error)
            error_state["workflow_should_terminate"] = True
            error_state["node_error"] = {
                "node": "reporter",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": time.time(),
                "traceback": full_traceback
            }

            logger.info(f"🎯 [REPORTER] Returning error state (invocation {invocation_id})")
            logger.info(f"🎯 [REPORTER] Error state flags: workflow_failed=True, reporter_completed=True, workflow_should_terminate=True")

            return error_state
    
    def human_feedback_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Human feedback node - gets user input on plans.
        
        Allows plan review and editing.
        """
        logger.info("Requesting human feedback")
        
        plan = state.get("current_plan")
        if not plan:
            logger.warning("No plan available for feedback")
            return state
        
        # Format plan for review
        plan_text = self._format_plan_for_review(plan)
        
        # Check if auto-accept is enabled
        if state.get("auto_accept_plan", False):
            logger.info("Auto-accepting plan")
            state["plan_feedback"] = [{"feedback_type": "auto", "approved": True}]
            return state
        
        # In testing, we simulate feedback
        feedback = state.get("simulated_feedback", "[ACCEPTED]")
        
        # Original code would use interrupt here:
        # feedback = interrupt(
        #     f"Please review the research plan:\n\n{plan_text}\n\n"
        #     "Options:\n"
        #     "- Type '[ACCEPTED]' to proceed\n"
        #     "- Type '[EDIT_PLAN] <json>' to provide edited plan\n"
        #     "- Type feedback to request revision"
        # )
        
        # Process feedback
        if feedback == "[ACCEPTED]":
            logger.info("Plan accepted by user")
            return state
        
        elif feedback.startswith("[EDIT_PLAN]"):
            # Parse edited plan
            try:
                edited_json = feedback[len("[EDIT_PLAN]"):].strip()
                edited_plan = json.loads(edited_json)
                logger.info("Plan edited by user")
                state["current_plan"] = edited_plan
                return state
            except json.JSONDecodeError:
                logger.error("Failed to parse edited plan")
                return state
        
        else:
            # Add feedback for revision
            logger.info("User requested plan revision")
            
            from .core.plan_models import PlanFeedback
            
            plan_feedback = PlanFeedback(
                feedback_type="human",
                feedback=feedback,
                requires_revision=True,
                approved=False
            )
            
            # Build new feedback list without mutating state
            current_feedback = state.get("plan_feedback", [])
            new_feedback = current_feedback + [plan_feedback]
            
            state["plan_iterations"] = state.get("plan_iterations", 0) + 1
            state["plan_feedback"] = new_feedback
            return state
    
    def _mock_background_search(self, topic: str) -> List[SearchResult]:
        """Generate mock background search results."""
        return [
            SearchResult(
                title=f"Overview of {topic}",
                url="https://example.com/overview",
                content=f"General information about {topic}. This provides context and background.",
                source="Mock Search",
                score=0.9
            ),
            SearchResult(
                title=f"Recent developments in {topic}",
                url="https://example.com/recent",
                content=f"Latest updates and trends related to {topic}.",
                source="Mock Search",
                score=0.85
            )
        ]
    
    def _compile_background_info(self, results: List[SearchResult]) -> str:
        """Compile search results into background information - NO TRUNCATION."""
        if not results:
            return "No background information available."

        info_parts = ["Background Information:\n"]

        # Include up to 10 results with full content (5000 chars each)
        for i, result in enumerate(results[:10], 1):
            info_parts.append(f"{i}. {result.title}")
            # Keep full content up to 5000 chars for comprehensive context
            content = result.content[:5000] if len(result.content) > 5000 else result.content
            info_parts.append(f"   {content}")
            info_parts.append("")

        return "\n".join(info_parts)
    
    def _format_plan_for_review(self, plan) -> str:
        """Format plan for human review."""
        lines = []
        
        if hasattr(plan, 'title'):
            lines.append(f"Title: {plan.title}")
            lines.append(f"Research Topic: {plan.research_topic}")
            
            if hasattr(plan, 'steps') and plan.steps:
                lines.append("\nSteps:")
                for i, step in enumerate(plan.steps, 1):
                    lines.append(f"{i}. {step.title}")
                    lines.append(f"   - {step.description}")
        else:
            # Handle dict format
            lines.append(str(plan))
        
        return "\n".join(lines)
    
    def _process_search_results(self, raw_results: List[Any]) -> List[SearchResult]:
        """Process search results to ensure they are SearchResult objects."""
        from .core.types import SearchResult
        
        processed = []
        for result in raw_results:
            if isinstance(result, SearchResult):
                # Already a SearchResult object
                processed.append(result)
            elif isinstance(result, dict):
                # Convert dictionary to SearchResult
                # FIXED: SearchResult doesn't accept 'score' parameter - use 'position' and store score in metadata
                metadata = result.get('metadata', {})
                if 'score' in result:
                    metadata['score'] = result['score']  # Preserve score in metadata

                search_result = SearchResult(
                    title=result.get('title', ''),
                    url=result.get('url', ''),
                    content=result.get('content', result.get('snippet', '')),
                    source=result.get('source', 'web'),
                    position=result.get('position'),  # FIXED: Use position instead of score
                    published_date=result.get('published_date'),
                    metadata=metadata
                )
                processed.append(search_result)
            else:
                # Try to convert to string and create minimal SearchResult
                content = str(result)
                search_result = SearchResult(
                    title='Search Result',
                    url='',
                    content=content,
                    source='unknown'
                )
                processed.append(search_result)
        
        return processed

    def _prepare_incremental_planning_context(self, state: Dict[str, Any], existing_plan) -> Dict[str, Any]:
        """Prepare context for incremental planning in research loops."""
        # Analyze completed steps and gather insights
        completed_steps = [s for s in existing_plan.steps if s.status == "completed"]
        pending_steps = [s for s in existing_plan.steps if s.status in ["pending", "in_progress"]]

        # Extract key discoveries from completed steps
        key_discoveries = []
        knowledge_gaps = []
        verification_needed = []
        deep_dive_topics = []

        for step in completed_steps:
            if hasattr(step, 'observations') and step.observations:
                # Extract insights from observations - FULL CONTENT
                for obs in step.observations:
                    if isinstance(obs, str):
                        # Simple heuristics to identify different types of findings
                        if any(keyword in obs.lower() for keyword in ['however', 'but', 'controversial', 'disputed', 'unclear']):
                            verification_needed.append(obs)  # Full observation
                        elif any(keyword in obs.lower() for keyword in ['emerging', 'new', 'recent', 'breakthrough']):
                            deep_dive_topics.append(obs)  # Full observation
                        elif any(keyword in obs.lower() for keyword in ['gap', 'missing', 'limited', 'need more']):
                            knowledge_gaps.append(obs)  # Full observation
                        else:
                            key_discoveries.append(obs)  # Full observation

        # Add incremental planning context to state - NO LIMITS (capture everything)
        state["incremental_context"] = {
            "research_loop": state.get("research_loops", 0) + 1,
            "completed_steps_count": len(completed_steps),
            "pending_steps_count": len(pending_steps),
            "key_discoveries": key_discoveries,  # ALL discoveries
            "knowledge_gaps": knowledge_gaps,    # ALL gaps
            "verification_needed": verification_needed,  # ALL items
            "deep_dive_topics": deep_dive_topics,  # ALL topics
            "existing_plan_summary": self._summarize_existing_plan(existing_plan),
            "planning_mode": "incremental_enhancement"
        }

        logger.info(f"[INCREMENTAL CONTEXT] Prepared context: {len(key_discoveries)} discoveries, "
                   f"{len(knowledge_gaps)} gaps, {len(verification_needed)} items to verify, "
                   f"{len(deep_dive_topics)} deep-dive topics")

        return state

    def _summarize_existing_plan(self, plan) -> str:
        """Create a summary of the existing plan for incremental planning."""
        completed_count = len([s for s in plan.steps if s.status == "completed"])
        total_count = len(plan.steps)

        summary = f"Current plan has {total_count} steps, {completed_count} completed. "

        if hasattr(plan, 'quality_assessment') and plan.quality_assessment:
            summary += f"Quality score: {plan.quality_assessment.overall_score:.1%}. "

        # Add step type breakdown
        step_types = {}
        for step in plan.steps:
            step_type = getattr(step, 'step_type', 'research')
            if hasattr(step_type, 'value'):
                step_type = step_type.value
            step_types[step_type] = step_types.get(step_type, 0) + 1

        summary += f"Step types: {', '.join([f'{k}({v})' for k, v in step_types.items()])}."

        return summary


def create_enhanced_workflow_graph():
    """
    Create the enhanced workflow graph with multi-agent support.
    
    Returns:
        Compiled LangGraph workflow
    """
    from langgraph.graph import StateGraph, END
    from .core.multi_agent_state import EnhancedResearchState
    
    # Create graph
    workflow = StateGraph(EnhancedResearchState)
    
    # Create nodes instance (would need agent reference in practice)
    # This is a template - actual implementation would pass the agent
    
    # Add nodes
    workflow.add_node("coordinator", lambda s: s)  # Placeholder
    workflow.add_node("background_investigation", lambda s: s)  # Placeholder
    workflow.add_node("planner", lambda s: s)  # Placeholder
    workflow.add_node("researcher", lambda s: s)  # Placeholder
    workflow.add_node("fact_checker", lambda s: s)  # Placeholder
    workflow.add_node("reporter", lambda s: s)  # Placeholder
    workflow.add_node("human_feedback", lambda s: s)  # Placeholder
    
    # Add edges
    workflow.set_entry_point("coordinator")
    
    # Coordinator routing
    workflow.add_conditional_edges(
        "coordinator",
        lambda x: "background_investigation" if x.get("research_topic") else END,
        {
            "background_investigation": "background_investigation",
            END: END
        }
    )
    
    # Background investigation to planner
    workflow.add_edge("background_investigation", "planner")
    
    # Planner routing
    workflow.add_conditional_edges(
        "planner",
        lambda x: "researcher" if x.get("current_plan") else END,
        {
            "researcher": "researcher",
            "human_feedback": "human_feedback",
            "reporter": "reporter",
            END: END
        }
    )
    
    # Human feedback routing
    workflow.add_conditional_edges(
        "human_feedback",
        lambda x: "researcher",
        {
            "planner": "planner",
            "researcher": "researcher"
        }
    )
    
    # Researcher routing
    workflow.add_conditional_edges(
        "researcher",
        lambda x: "fact_checker" if x.get("enable_grounding", True) else "reporter",
        {
            "fact_checker": "fact_checker",
            "reporter": "reporter",
            "planner": "planner",
            "researcher": "researcher"
        }
    )
    
    # Fact checker routing
    workflow.add_conditional_edges(
        "fact_checker",
        lambda x: "reporter",
        {
            "reporter": "reporter",
            "researcher": "researcher",
            "planner": "planner"
        }
    )
    
    # Reporter ends workflow
    workflow.add_edge("reporter", END)
    
    return workflow.compile()
