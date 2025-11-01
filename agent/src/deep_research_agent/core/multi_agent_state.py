"""
Enhanced state management for multi-agent research system.

Extends the existing state with support for planning, grounding,
and multi-agent coordination.
"""

from typing import List, Dict, Any, Optional, Literal, Union, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict

from .plan_models import Plan, Step, PlanFeedback, PlanQuality
from .report_styles import ReportStyle
from .constraint_system import QueryConstraints
from .grounding import (
    GroundingResult,
    FactualityReport,
    Contradiction,
    VerificationLevel
)
from . import (
    ResearchContext,
    SearchResult,
    Citation,
    ResearchQuery,
    get_logger
)
from .observation_models import StructuredObservation
# Legacy MetricPipelineState removed - now using calculation_results from calculation_agent
from .metrics.unified_models import UnifiedPlan
from .report_generation.models import CalculationContext

# Initialize logger
logger = get_logger(__name__)


# Reducer functions for concurrent state updates
def use_latest_plan(left: Optional[Plan], right: Optional[Plan]) -> Optional[Plan]:
    """Reducer that uses the most recent non-None plan."""
    return right if right is not None else left


def merge_lists(left: List[Any], right: List[Any]) -> List[Any]:
    """
    Reducer that merges two lists with memory-conscious limits.

    Uses TypeAwareListMergeManager for proper type detection with Pydantic validation
    instead of fragile dict key inspection. This is critical for correctly handling
    observation dicts, search results, and other state list fields.

    This is a LangGraph reducer function - maintains compatibility with existing
    state annotations while using the refactored TypeAwareListMergeManager internally.
    """
    from .advanced_utilities import TypeAwareListMergeManager

    # Use TypeAwareListMergeManager for type-aware merging
    # This replaces 72 lines of fragile dict inspection with robust type detection
    merger = TypeAwareListMergeManager()
    return merger.merge_with_type_limits(
        left=left,
        right=right,
        deduplicate=False  # LangGraph handles deduplication at higher level
    )


def use_latest_value(left: Any, right: Any) -> Any:
    """Reducer that uses the most recent non-None value."""
    return right if right is not None else left


def merge_dicts(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reducer that merges two dictionaries with right-hand precedence."""
    if not left:
        return right or {}
    if not right:
        return left

    merged = dict(left)
    merged.update(right)
    return merged


def ensure_state_hydrated(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure state is hydrated at perimeter entry points (routers and nodes).

    This function provides a lightweight perimeter guard that hydrates dict representations
    back to Pydantic models. It's idempotent - if state is already hydrated, this is a no-op.

    CRITICAL: This is needed because LangGraph serializes Pydantic models to dicts when
    propagating state between nodes. Without hydration at boundaries, downstream code
    that expects Pydantic objects (e.g., query_constraints.entities) will crash with
    AttributeError.

    Implementation:
    - Checks if query_constraints is a dict (needs hydration)
    - Delegates to existing StateManager.hydrate_state() for full hydration
    - Returns state as-is if already hydrated or not present

    Args:
        state: State dict from LangGraph (may contain serialized Pydantic objects)

    Returns:
        Hydrated state (modified in-place) with Pydantic objects restored
    """
    # Fast path: Only hydrate if query_constraints exists and is a dict
    query_constraints = state.get("query_constraints")
    if query_constraints is not None and isinstance(query_constraints, dict):
        # State needs hydration - delegate to existing hydration logic
        # Note: StateManager is defined later in this file, but Python resolves names at
        # runtime, so this will work when the function is called
        return StateManager.hydrate_state(state)

    # Already hydrated or not present - no-op
    return state


class EnhancedResearchState(TypedDict):
    """
    Enhanced state for multi-agent research system.
    
    Extends the basic state with planning, grounding, and style capabilities.
    """
    
    # Core conversation state
    messages: Annotated[List, add_messages]
    
    # Research context
    research_topic: Annotated[str, use_latest_value]
    research_context: Annotated[Optional[ResearchContext], use_latest_value]
    
    # Planning state
    current_plan: Annotated[Optional[Plan], use_latest_plan]
    plan_iterations: Annotated[int, use_latest_value]
    plan_feedback: Annotated[Optional[List[PlanFeedback]], merge_lists]
    plan_quality: Annotated[Optional[PlanQuality], use_latest_value]
    enable_iterative_planning: Annotated[bool, use_latest_value]
    max_plan_iterations: Annotated[int, use_latest_value]
    
    # Background investigation
    enable_background_investigation: Annotated[bool, use_latest_value]
    background_investigation_results: Annotated[Optional[str], use_latest_value]
    
    # Execution state (observations as Pydantic objects throughout)
    observations: Annotated[
        List[StructuredObservation],  # StructuredObservation objects throughout
        use_latest_value  # FIXED: Changed from merge_lists to prevent duplication
    ]  # Accumulated observations from all steps - single source of truth
    new_observations: Annotated[
        List[StructuredObservation],  # Delta: new observations from current pass only
        merge_lists  # Appends for tracking what each pass contributed
    ]  # Temporary holding for new observations before consolidation
    completed_steps: Annotated[List[Step], merge_lists]
    current_step: Annotated[Optional[Step], use_latest_value]
    current_step_index: Annotated[int, use_latest_value]
    # Loop control (separate counters)
    research_loops: Annotated[int, use_latest_value]  # reserved for researcher-side loops (not used for cap here)
    max_research_loops: Annotated[int, use_latest_value]
    fact_check_loops: Annotated[int, use_latest_value]
    max_fact_check_loops: Annotated[int, use_latest_value]
    
    # Research results
    search_results: Annotated[List[SearchResult], merge_lists]
    search_queries: Annotated[List[ResearchQuery], merge_lists]
    section_research_results: Annotated[Dict[str, Any], merge_dicts]
    
    # Grounding and factuality
    enable_grounding: Annotated[bool, use_latest_value]
    grounding_results: Annotated[Optional[List[GroundingResult]], merge_lists]
    factuality_report: Annotated[Optional[FactualityReport], use_latest_value]  # FIX: Add proper annotation
    contradictions: Annotated[Optional[List[Contradiction]], merge_lists]
    factuality_score: Annotated[Optional[float], use_latest_value]
    verification_level: Annotated[VerificationLevel, use_latest_value]
    
    # Entity validation and constraints
    query_constraints: Annotated[Optional[QueryConstraints], use_latest_value]  # NEW: Extracted constraints (entities, metrics, scenarios)
    # requested_entities: REMOVED - Use query_constraints.entities instead
    entity_violations: Annotated[List[Dict[str, Any]], merge_lists]
    
    # Citations and references
    citations: Annotated[List[Citation], merge_lists]
    citation_style: Annotated[str, use_latest_value]  # APA, MLA, Chicago, etc.
    
    # Report generation
    report_style: Annotated[ReportStyle, use_latest_value]
    final_report: Annotated[Optional[str], use_latest_value]
    report_sections: Annotated[Optional[Dict[str, str]], use_latest_value]  # Section name -> content
    metric_state: Annotated[Optional[Dict[str, Any]], use_latest_value]
    unified_plan: Annotated[Optional[UnifiedPlan], use_latest_value]  # NEW: Unified plan for metric extraction/calculation
    calculation_results: Annotated[Optional['CalculationContext'], use_latest_value]  # NEW: CalculationContext from calculation_agent (Pydantic model)
    metric_capability_enabled: Annotated[bool, use_latest_value]
    pending_calculation_research: Annotated[Optional[List[str]], use_latest_value]  # Queries needed for calculations
    
    # Reflexion and self-improvement
    enable_reflexion: Annotated[bool, use_latest_value]
    reflections: Annotated[List[str], merge_lists]  # Self-reflection feedback
    reflection_memory_size: Annotated[int, use_latest_value]
    
    # Agent coordination
    current_agent: Annotated[str, use_latest_value]  # Which agent is currently active
    agent_handoffs: Annotated[List[Dict[str, Any]], merge_lists]  # History of agent handoffs
    coordination_completed: Annotated[bool, use_latest_value]  # Whether gap analysis coordination was performed
    gap_decision: Annotated[Optional[Dict[str, Any]], use_latest_value]  # Last gap analysis decision

    # Quality metrics
    research_quality_score: Annotated[Optional[float], use_latest_value]
    coverage_score: Annotated[Optional[float], use_latest_value]
    confidence_score: Annotated[Optional[float], use_latest_value]
    
    # Configuration
    enable_deep_thinking: Annotated[bool, use_latest_value]
    enable_human_feedback: Annotated[bool, use_latest_value]
    auto_accept_plan: Annotated[bool, use_latest_value]
    
    # Timing and metadata
    start_time: Annotated[datetime, use_latest_value]
    end_time: Annotated[Optional[datetime], use_latest_value]
    total_duration_seconds: Annotated[Optional[float], use_latest_value]
    
    # Error handling
    errors: Annotated[List[str], merge_lists]
    warnings: Annotated[List[str], merge_lists]

    # Streaming events for UI
    intermediate_events: Annotated[List[Dict[str, Any]], merge_lists]

    # ========================================================================
    # Enhanced Research Loop State Management
    # ========================================================================

    # Loop discovery tracking
    loop_discoveries: Annotated[List[Dict[str, Any]], merge_lists]
    knowledge_gaps: Annotated[List[str], merge_lists]
    verification_needed: Annotated[List[str], merge_lists]
    deep_dive_topics: Annotated[List[str], merge_lists]

    # Research quality tracking
    quality_issues: Annotated[List[str], merge_lists]
    source_diversity_score: Annotated[float, use_latest_value]
    factuality_confidence: Annotated[float, use_latest_value]

    # Incremental context for planner
    incremental_context: Annotated[Optional[Dict[str, Any]], use_latest_value]

    # Loop execution tracking
    loop_execution_history: Annotated[List[Dict[str, Any]], merge_lists]
    current_loop_focus: Annotated[Optional[str], use_latest_value]  # "gap_filling", "verification", "deep_dive"

    # User preferences
    user_preferences: Annotated[Optional[Dict[str, Any]], use_latest_value]


class AgentHandoff(BaseModel):
    """Represents a handoff between agents."""
    from_agent: str
    to_agent: str
    reason: str
    context: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class ResearchProgress(BaseModel):
    """Tracks overall research progress."""
    total_steps: int
    completed_steps: int
    current_phase: str  # planning, research, synthesis, etc.
    estimated_completion: Optional[float]  # Percentage
    blockers: List[str] = Field(default_factory=list)
    
    def get_status_message(self) -> str:
        """Get human-readable status message."""
        if self.current_phase == "planning":
            return "Creating research plan..."
        elif self.current_phase == "background_investigation":
            return "Gathering background context..."
        elif self.current_phase == "research":
            return f"Researching... ({self.completed_steps}/{self.total_steps} steps)"
        elif self.current_phase == "grounding":
            return "Verifying factual accuracy..."
        elif self.current_phase == "synthesis":
            return "Synthesizing findings..."
        elif self.current_phase == "report_generation":
            return "Generating final report..."
        else:
            return f"Phase: {self.current_phase}"


class StateManager:
    """Utilities for managing enhanced research state."""
    
    @staticmethod
    def _initialize_report_style(config: Dict[str, Any]) -> 'ReportStyle':
        """Initialize report style with detailed logging for debugging."""
        from .report_styles import ReportStyle
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Get values from different config paths
        report_section = config.get("report", {})
        default_style_from_report = report_section.get("default_style")
        default_report_style_legacy = config.get("default_report_style")
        
        logger.info("STATE_INIT: Initializing report style...")
        logger.info(f"STATE_INIT: config.report = {report_section}")
        logger.info(f"STATE_INIT: config.report.default_style = {default_style_from_report}")
        logger.info(f"STATE_INIT: config.default_report_style = {default_report_style_legacy}")
        
        # Determine the final value using the same logic as before
        final_value = default_style_from_report or default_report_style_legacy or "default"
        logger.info(f"STATE_INIT: Final style string value: '{final_value}'")
        
        # Convert to ReportStyle enum
        report_style = ReportStyle(final_value)
        logger.info(f"STATE_INIT: Final ReportStyle enum: {report_style}")
        logger.info(f"STATE_INIT: ReportStyle == ReportStyle.DEFAULT: {report_style == ReportStyle.DEFAULT}")
        
        return report_style
    
    @staticmethod
    def initialize_state(
        research_topic: str,
        config: Dict[str, Any]
    ) -> EnhancedResearchState:
        """Initialize a new enhanced research state."""
        return EnhancedResearchState(
            messages=[],
            research_topic=research_topic,
            research_context=None,
            
            # Planning
            current_plan=None,
            plan_iterations=0,
            plan_feedback=None,
            plan_quality=None,
            enable_iterative_planning=config.get("enable_iterative_planning", True),
            max_plan_iterations=config.get("max_plan_iterations", 3),
            
            # Background investigation (from workflow config)
            enable_background_investigation=config.get("workflow", {}).get("enable_background_investigation", True),
            background_investigation_results=None,
            
            # Execution
            observations=[],
            completed_steps=[],
            current_step=None,
            current_step_index=0,
            research_loops=0,
            max_research_loops=config.get("research", {}).get("max_research_loops", 3),
            fact_check_loops=0,
            max_fact_check_loops=config.get("research", {}).get("max_fact_check_loops", 2),
            
            # Research
            search_results=[],
            search_queries=[],
            section_research_results={},

            # Grounding
            enable_grounding=config.get("grounding", {}).get("enabled", True),
            grounding_results=None,
            factuality_report=None,
            contradictions=None,
            factuality_score=None,
            verification_level=VerificationLevel(
                config.get("grounding", {}).get("verification_level", "moderate")
            ),
            
            # Entity validation and constraints
            query_constraints=None,  # Will be populated by planner during constraint extraction
            # requested_entities=[], # REMOVED - Use query_constraints.entities instead
            entity_violations=[],
            
            # Citations
            citations=[],
            citation_style=config.get("citation_style", "APA"),
            
            # Report - enhanced logging for debugging adaptive structure
            report_style=StateManager._initialize_report_style(config),
            final_report=None,
            report_sections=None,

            # Metric calculation and extraction (CRITICAL FIX - these fields were missing!)
            metric_state=None,
            unified_plan=None,
            calculation_results=None,
            metric_capability_enabled=config.get("metrics", {}).get("enabled", False),
            pending_calculation_research=None,

            # Reflexion
            enable_reflexion=config.get("enable_reflexion", True),
            reflections=[],
            reflection_memory_size=config.get("reflection_memory_size", 5),
            
            # Coordination
            current_agent="coordinator",
            agent_handoffs=[],
            coordination_completed=False,
            gap_decision=None,

            # Quality
            research_quality_score=None,
            coverage_score=None,
            confidence_score=None,
            
            # Configuration
            enable_deep_thinking=config.get("enable_deep_thinking", False),
            enable_human_feedback=config.get("enable_human_feedback", False),
            auto_accept_plan=config.get("auto_accept_plan", True),
            
            # Timing
            start_time=datetime.now(),
            end_time=None,
            total_duration_seconds=None,

            # Error handling
            errors=[],
            warnings=[],

            # Streaming events
            intermediate_events=[],

            # User preferences
            user_preferences=config.get("user_preferences", {}),

            # ========================================================================
            # Enhanced Research Loop State Management Fields
            # ========================================================================

            # Loop discovery tracking
            loop_discoveries=[],
            knowledge_gaps=[],
            verification_needed=[],
            deep_dive_topics=[],

            # Research quality tracking
            quality_issues=[],
            source_diversity_score=0.0,
            factuality_confidence=0.0,

            # Incremental context for planner
            incremental_context=None,

            # Loop execution tracking
            loop_execution_history=[],
            current_loop_focus=None,
        )

    @staticmethod
    def hydrate_state(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hydrate state by converting dict representations back to Pydantic/dataclass objects.

        This is the PERIMETER CONVERSION point - converts serialized dicts from JSON fixtures
        back to proper Pydantic BaseModel and dataclass objects so all code can use attribute access.

        Handles:
        - Pydantic BaseModel objects (Plan, Step, Citation, etc.) - use .model_validate()
        - Dataclass objects (QueryConstraints, SearchResult, etc.) - use ClassName(**dict)
        - Lists of objects
        - Nested objects (QueryConstraints contains ScenarioDefinition list)
        - Already-hydrated objects (skip gracefully)
        - None/missing values

        Args:
            state_dict: State dictionary from JSON.load() or similar

        Returns:
            State dictionary with all objects hydrated
        """
        import logging
        import re
        logger = logging.getLogger(__name__)

        # ====================================================================
        # PRE-HYDRATION NORMALIZATION HELPERS
        # ====================================================================

        def normalize_enum_strings(data):
            """
            Normalize enum strings from 'EnumName.VALUE' to 'value'.

            Fixes state_capture bug where json.dumps(..., default=str) converts
            StepType.RESEARCH → 'StepType.RESEARCH' instead of 'research'.

            Recursively processes dicts and lists.
            """
            if isinstance(data, dict):
                return {k: normalize_enum_strings(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [normalize_enum_strings(item) for item in data]
            elif isinstance(data, str):
                # Pattern: 'EnumName.VALUE' → 'value'
                # Examples: 'StepType.RESEARCH' → 'research'
                #           'StepStatus.COMPLETED' → 'completed'
                match = re.match(r'^[A-Z][a-zA-Z]*\.([A-Z_]+)$', data)
                if match:
                    # Extract the value part and convert to lowercase
                    enum_value = match.group(1).lower()
                    logger.debug(f"Normalized enum string: '{data}' → '{enum_value}'")
                    return enum_value
                return data
            else:
                return data

        def add_plan_backward_compatibility(plan_dict):
            """Add missing required fields for Plan model."""
            if not isinstance(plan_dict, dict):
                return plan_dict

            # Required field: 'thought'
            if 'thought' not in plan_dict:
                plan_dict['thought'] = "Plan loaded from test fixture"
                logger.debug("Added missing 'thought' field to plan")

            return plan_dict

        # ====================================================================
        # HYDRATION HELPERS (WITH FAIL-FAST ERROR HANDLING)
        # ====================================================================

        # Helper: Check if object is already hydrated
        def is_hydrated(obj, expected_type):
            """Check if object is already the expected type."""
            if obj is None:
                return True
            if isinstance(obj, expected_type):
                return True
            return False

        # Helper: Hydrate a single Pydantic BaseModel object
        def hydrate_pydantic(obj, model_class, field_name):
            """Hydrate a Pydantic BaseModel from dict with FAIL-FAST error handling."""
            if obj is None:
                return None
            if is_hydrated(obj, model_class):
                return obj
            if isinstance(obj, dict):
                try:
                    # Handle both Pydantic v1 and v2
                    if hasattr(model_class, 'model_validate'):
                        # Pydantic v2
                        return model_class.model_validate(obj)
                    elif hasattr(model_class, 'parse_obj'):
                        # Pydantic v1
                        return model_class.parse_obj(obj)
                    else:
                        raise AttributeError(f"{model_class.__name__} has neither model_validate nor parse_obj")
                except Exception as e:
                    logger.error(
                        f"❌ HYDRATION FAILED for {field_name}:\n"
                        f"  Expected: {model_class.__name__}\n"
                        f"  Error: {e}\n"
                        f"  Data keys: {list(obj.keys())[:10]}"
                    )
                    raise ValueError(
                        f"Cannot hydrate {field_name} to {model_class.__name__}. "
                        f"Validation failed - state contains dict when object is required."
                    ) from e
            return obj

        # Helper: Hydrate a list of Pydantic BaseModel objects
        def hydrate_pydantic_list(obj_list, model_class, field_name):
            """Hydrate a list of Pydantic BaseModel objects from dicts with FAIL-FAST."""
            if not obj_list:
                return obj_list
            if not isinstance(obj_list, list):
                return obj_list

            hydrated = []
            for idx, item in enumerate(obj_list):
                if isinstance(item, dict):
                    try:
                        # Handle both Pydantic v1 and v2
                        if hasattr(model_class, 'model_validate'):
                            hydrated.append(model_class.model_validate(item))
                        elif hasattr(model_class, 'parse_obj'):
                            hydrated.append(model_class.parse_obj(item))
                        else:
                            raise AttributeError(f"{model_class.__name__} has neither model_validate nor parse_obj")
                    except Exception as e:
                        logger.error(
                            f"❌ HYDRATION FAILED for {field_name}[{idx}]:\n"
                            f"  Expected: {model_class.__name__}\n"
                            f"  Error: {e}"
                        )
                        raise ValueError(
                            f"Cannot hydrate {field_name}[{idx}] to {model_class.__name__}"
                        ) from e
                else:
                    # Already hydrated or wrong type
                    hydrated.append(item)
            return hydrated

        # Helper: Hydrate a dataclass object
        def hydrate_dataclass(obj, dataclass_type, field_name):
            """Hydrate a dataclass from dict with FAIL-FAST error handling."""
            if obj is None:
                return None
            if is_hydrated(obj, dataclass_type):
                return obj
            if isinstance(obj, dict):
                try:
                    return dataclass_type(**obj)
                except Exception as e:
                    logger.error(
                        f"❌ HYDRATION FAILED for {field_name}:\n"
                        f"  Expected: {dataclass_type.__name__}\n"
                        f"  Error: {e}"
                    )
                    raise ValueError(
                        f"Cannot hydrate {field_name} to {dataclass_type.__name__}"
                    ) from e
            return obj

        # Helper: Hydrate a list of dataclass objects
        def hydrate_dataclass_list(obj_list, dataclass_type, field_name):
            """Hydrate a list of dataclass objects from dicts with FAIL-FAST."""
            if not obj_list:
                return obj_list
            if not isinstance(obj_list, list):
                return obj_list

            hydrated = []
            for idx, item in enumerate(obj_list):
                if isinstance(item, dict):
                    try:
                        hydrated.append(dataclass_type(**item))
                    except Exception as e:
                        logger.error(
                            f"❌ HYDRATION FAILED for {field_name}[{idx}]:\n"
                            f"  Expected: {dataclass_type.__name__}\n"
                            f"  Error: {e}"
                        )
                        raise ValueError(
                            f"Cannot hydrate {field_name}[{idx}] to {dataclass_type.__name__}"
                        ) from e
                else:
                    # Already hydrated or wrong type
                    hydrated.append(item)
            return hydrated

        # Import all required types
        from .plan_models import Plan, Step, PlanQuality, PlanFeedback
        from .constraint_system import QueryConstraints, ScenarioDefinition
        from .types import Citation, SearchResult, ResearchQuery, ResearchContext
        from .observation_models import StructuredObservation
        from .grounding import FactualityReport, GroundingResult, Contradiction
        from .metrics.unified_models import UnifiedPlan

        # Create a copy to avoid modifying original
        state = dict(state_dict)

        # ====================================================================
        # PHASE 1: NORMALIZE ENTIRE STATE (enums, missing fields)
        # ====================================================================
        logger.debug("Phase 1: Normalizing entire state for enum strings...")
        state = normalize_enum_strings(state)

        # Add backward compatibility for specific models
        if "current_plan" in state and isinstance(state["current_plan"], dict):
            state["current_plan"] = add_plan_backward_compatibility(state["current_plan"])

        # Add description to completed_steps if missing
        if "completed_steps" in state and isinstance(state["completed_steps"], list):
            for step_dict in state["completed_steps"]:
                if isinstance(step_dict, dict) and "description" not in step_dict:
                    step_dict["description"] = step_dict.get("title", "Step from fixture")
                    logger.debug(f"Added description to step {step_dict.get('step_id', 'unknown')}")

        # ========================================================================
        # PHASE 2: HYDRATE ALL OBJECTS
        # ========================================================================
        logger.debug("Phase 2: Hydrating all Pydantic/dataclass objects...")

        # Single Plan object
        if "current_plan" in state and isinstance(state["current_plan"], dict):
            state["current_plan"] = hydrate_pydantic(
                state["current_plan"], Plan, "current_plan"
            )

        # Single PlanQuality object
        if "plan_quality" in state:
            state["plan_quality"] = hydrate_pydantic(
                state["plan_quality"], PlanQuality, "plan_quality"
            )

        # Single UnifiedPlan object
        if "unified_plan" in state:
            state["unified_plan"] = hydrate_pydantic(
                state["unified_plan"], UnifiedPlan, "unified_plan"
            )

        # Single CalculationContext object
        # CRITICAL FIX (Bug #4): Now that calculation agent returns correct schema,
        # we can hydrate calculation_results like other Pydantic objects.
        # Previous issue was schema mismatch (data_points vs extracted_data), now resolved.
        # See: agents/calculation_agent.py:404-419 for updated return structure
        if "calculation_results" in state and state["calculation_results"] is not None:
            try:
                # Import CalculationContext
                from ..core.report_generation.models import CalculationContext

                # Hydrate using standard pattern
                if isinstance(state["calculation_results"], dict):
                    state["calculation_results"] = CalculationContext.model_validate(
                        state["calculation_results"]
                    )
                    logger.info(
                        f"✅ Hydrated calculation_results: "
                        f"{len(state['calculation_results'].extracted_data)} data points, "
                        f"{len(state['calculation_results'].calculations)} calculations"
                    )
                elif not isinstance(state["calculation_results"], CalculationContext):
                    logger.warning(
                        f"⚠️ calculation_results has unexpected type: "
                        f"{type(state['calculation_results'])}"
                    )
            except Exception as e:
                logger.error(
                    f"❌ Failed to hydrate calculation_results: {e}\n"
                    f"  Dict keys: {list(state['calculation_results'].keys()) if isinstance(state['calculation_results'], dict) else 'N/A'}",
                    exc_info=True
                )
                # Don't fail hard, leave as dict for reporter to handle gracefully

        # Single FactualityReport object
        if "factuality_report" in state:
            state["factuality_report"] = hydrate_pydantic(
                state["factuality_report"], FactualityReport, "factuality_report"
            )

        # Single Step object
        if "current_step" in state:
            state["current_step"] = hydrate_pydantic(
                state["current_step"], Step, "current_step"
            )

        # List of Steps
        if "completed_steps" in state:
            state["completed_steps"] = hydrate_pydantic_list(
                state["completed_steps"], Step, "completed_steps"
            )

        # List of PlanFeedback
        if "plan_feedback" in state:
            state["plan_feedback"] = hydrate_pydantic_list(
                state["plan_feedback"], PlanFeedback, "plan_feedback"
            )

        # List of Citations
        if "citations" in state:
            state["citations"] = hydrate_pydantic_list(
                state["citations"], Citation, "citations"
            )

        # List of StructuredObservations
        if "observations" in state:
            state["observations"] = hydrate_pydantic_list(
                state["observations"], StructuredObservation, "observations"
            )

        if "new_observations" in state:
            state["new_observations"] = hydrate_pydantic_list(
                state["new_observations"], StructuredObservation, "new_observations"
            )

        # List of GroundingResults
        if "grounding_results" in state:
            state["grounding_results"] = hydrate_pydantic_list(
                state["grounding_results"], GroundingResult, "grounding_results"
            )

        # List of Contradictions
        if "contradictions" in state:
            state["contradictions"] = hydrate_pydantic_list(
                state["contradictions"], Contradiction, "contradictions"
            )

        # ========================================================================
        # Hydrate Dataclass objects
        # ========================================================================

        # QueryConstraints (with nested ScenarioDefinition list) - THE PRIMARY FIX!
        if "query_constraints" in state and state["query_constraints"] is not None:
            qc = state["query_constraints"]
            if isinstance(qc, dict):
                try:
                    # First hydrate nested ScenarioDefinition list
                    hydrated_scenarios = []
                    if "scenarios" in qc and isinstance(qc["scenarios"], list):
                        for scenario in qc["scenarios"]:
                            if isinstance(scenario, dict):
                                hydrated_scenarios.append(ScenarioDefinition(**scenario))
                            elif isinstance(scenario, str):
                                # state_capture bug: scenarios serialized as strings
                                logger.warning(f"⚠️ Skipping string scenario (state_capture bug): {scenario[:100]}")
                            else:
                                # Already a ScenarioDefinition object
                                hydrated_scenarios.append(scenario)
                        qc["scenarios"] = hydrated_scenarios

                    # Then hydrate QueryConstraints (filter to only valid fields)
                    valid_fields = {
                        'entities', 'metrics', 'scenarios', 'comparison_type', 'topics',
                        'comparisons', 'data_format', 'specifics', 'time_constraints', 'monetary_values'
                    }
                    filtered_qc = {k: v for k, v in qc.items() if k in valid_fields}
                    state["query_constraints"] = QueryConstraints(**filtered_qc)
                    logger.info(f"✅ Hydrated query_constraints with {len(qc.get('entities', []))} entities, {len(hydrated_scenarios)} scenarios")
                except Exception as e:
                    logger.error(f"❌ Could not hydrate query_constraints: {e}")
                    raise ValueError(f"Cannot hydrate query_constraints") from e

        # Single ResearchContext object
        if "research_context" in state:
            state["research_context"] = hydrate_dataclass(
                state["research_context"], ResearchContext, "research_context"
            )

        # List of SearchResults
        if "search_results" in state:
            state["search_results"] = hydrate_dataclass_list(
                state["search_results"], SearchResult, "search_results"
            )

        # List of ResearchQueries
        if "search_queries" in state:
            state["search_queries"] = hydrate_dataclass_list(
                state["search_queries"], ResearchQuery, "search_queries"
            )

        logger.debug(f"✅ State hydration complete - {len(state)} fields processed")
        return state

    @staticmethod
    def safe_get_observation_content(obs: Union[Dict[str, Any], StructuredObservation, str]) -> str:
        """
        Safely extract content from observation regardless of format.

        This compatibility helper handles:
        - StructuredObservation objects (Pydantic BaseModel)
        - Dict format (legacy JSON representation)
        - String format (plain text observation)
        - Any unexpected type (defensive fallback)

        Provides defense-in-depth during migration to Pydantic-everywhere architecture.
        """
        if isinstance(obs, StructuredObservation):
            # Pydantic object - access attribute directly
            return str(obs.content if obs.content else "")
        elif isinstance(obs, dict):
            # Legacy dict format - use .get() for safety
            return str(obs.get("content", ""))
        elif isinstance(obs, str):
            # Plain string observation
            return obs
        else:
            # Unexpected type - defensive fallback
            logger.warning(f"Unexpected observation type: {type(obs)}, converting to string")
            return str(obs)

    @staticmethod
    def update_plan(
        state: EnhancedResearchState,
        new_plan: Plan
    ) -> EnhancedResearchState:
        """Update the state with a new plan."""
        state["current_plan"] = new_plan
        state["plan_iterations"] += 1

        # Reset execution state for new plan
        state["current_step_index"] = 0
        state["completed_steps"] = []
        state["current_step"] = new_plan.get_next_step() if new_plan else None

        return state
    
    @staticmethod
    def record_handoff(
        state: EnhancedResearchState,
        from_agent: str,
        to_agent: str,
        reason: str,
        context: Optional[Dict] = None
    ) -> EnhancedResearchState:
        """Record an agent handoff."""
        handoff = AgentHandoff(
            from_agent=from_agent,
            to_agent=to_agent,
            reason=reason,
            context=context or {}
        )
        
        state["agent_handoffs"].append(handoff.model_dump())
        state["current_agent"] = to_agent
        
        return state
    
    @staticmethod
    def update_state(
        state: EnhancedResearchState,
        updates: Dict[str, Any]
    ) -> EnhancedResearchState:
        """Update state with provided updates."""
        for key, value in updates.items():
            if key in state:
                state[key] = value
        return state
    
    @staticmethod
    def add_observation(
        state: EnhancedResearchState,
        observation: Union[str, Any],
        step: Optional[Step] = None,
        config: Optional[Dict[str, Any]] = None,
        observation_index: Optional['ObservationEmbeddingIndex'] = None
    ) -> EnhancedResearchState:
        """
        Add an observation to the state with smart deduplication and memory limits.

        Args:
            state: The current state
            observation: The observation to add (string or StructuredObservation)
            step: Optional step to associate with observation
            config: Optional configuration with memory limits
            observation_index: Optional ObservationEmbeddingIndex for semantic deduplication

        Returns:
            Updated state with observation added (if not duplicate)
        """
        from .observation_models import ensure_structured_observation
        from .memory_config import MemoryOptimizedConfig
        import logging
        logger = logging.getLogger(__name__)

        # Normalize observation to StructuredObservation
        structured_obs = ensure_structured_observation(observation)

        # CRITICAL: Set step_id for section-specific filtering in reporter
        if step:
            old_step_id = getattr(structured_obs, 'step_id', None)
            structured_obs.step_id = step.step_id
            if old_step_id and old_step_id != step.step_id:
                logger.error(f"🔥 [add_observation] STEP_ID OVERWRITE: {old_step_id} → {step.step_id} (object_id={id(structured_obs)})")
            logger.debug(f"🔍 [add_observation] Set step_id={step.step_id} on obs (object_id={id(structured_obs)})")

        # FIXED: Keep as Pydantic object (Pydantic Everywhere strategy)
        # No conversion to dict - we use objects throughout

        # Add to new_observations (delta tracking for current pass)
        if "new_observations" not in state:
            state["new_observations"] = []
        state["new_observations"].append(structured_obs)

        # Add to complete observations with smart deduplication
        if "observations" not in state:
            state["observations"] = []

        # Extract content for deduplication using safe helper
        content = StateManager.safe_get_observation_content(structured_obs).strip()
        if not content:
            logger.debug("Skipping empty observation")
            return state

        # Use ObservationEmbeddingIndex if available for smart deduplication
        if observation_index:
            # Let the index handle all deduplication logic (hash + semantic)
            step_id = structured_obs.step_id if hasattr(structured_obs, 'step_id') else None
            is_new = observation_index.add_observation(content, step_id)

            if is_new:
                state["observations"].append(structured_obs)
                logger.debug(f"Added observation to state (passed index dedup): {content[:50]}...")
            else:
                logger.debug(f"Skipped duplicate observation (detected by index): {content[:50]}...")
        else:
            # Fallback to hash-based deduplication only
            content_lower = content.lower()
            obs_hash = hash(content_lower) if content_lower else 0
            existing_hashes = {
                hash(StateManager.safe_get_observation_content(o).lower().strip())
                for o in state["observations"]
                if StateManager.safe_get_observation_content(o).strip()
            }

            if obs_hash not in existing_hashes:
                state["observations"].append(structured_obs)
                logger.debug(f"Added observation to state (hash dedup): {content[:50]}...")
            else:
                logger.debug(f"Skipped duplicate observation (hash match): {content[:50]}...")

        # Use configurable observation limit (defaults to 50, can be overridden in base.yaml)
        max_observations = MemoryOptimizedConfig.get_observations_limit(config)
        if len(state["observations"]) > max_observations:
            state["observations"] = state["observations"][-max_observations:]

        if step:
            if not step.observations:
                step.observations = []
            step.observations.append(structured_obs)

            # Use configurable per-step observation limit
            max_observations_per_step = MemoryOptimizedConfig.get_observations_per_step_limit(config)
            if len(step.observations) > max_observations_per_step:
                step.observations = step.observations[-max_observations_per_step:]

        return state

    @staticmethod
    def consolidate_observations(state: EnhancedResearchState) -> EnhancedResearchState:
        """Consolidate new_observations into observations with deduplication.

        Called at workflow node boundaries to merge delta observations into the complete list.
        Uses content-based deduplication to prevent duplicate observations.
        """
        import logging
        logger = logging.getLogger(__name__)

        if "new_observations" not in state or not state["new_observations"]:
            return state

        if "observations" not in state:
            state["observations"] = []

        # Build hash set of existing observations using full content
        existing_hashes = {
            hash(StateManager.safe_get_observation_content(o).lower().strip())
            for o in state["observations"]
            if StateManager.safe_get_observation_content(o).strip()
        }

        # Add new observations that aren't duplicates
        added_count = 0
        duplicate_count = 0

        for new_obs in state["new_observations"]:
            content = StateManager.safe_get_observation_content(new_obs).lower().strip()
            obs_hash = hash(content) if content else 0
            if obs_hash not in existing_hashes:
                state["observations"].append(new_obs)
                existing_hashes.add(obs_hash)
                added_count += 1
            else:
                duplicate_count += 1

        logger.info(
            f"📊 Consolidated observations: "
            f"{len(state['new_observations'])} new → "
            f"{added_count} unique added, "
            f"{duplicate_count} duplicates skipped, "
            f"{len(state['observations'])} total"
        )

        # Clear new_observations after consolidation
        state["new_observations"] = []

        return state

    @staticmethod
    def add_observations_batch(
        state: EnhancedResearchState,
        observations: List[Any],
        step: Optional[Step] = None,
        config: Optional[Dict[str, Any]] = None,
        observation_index: Optional['ObservationEmbeddingIndex'] = None
    ) -> Tuple[EnhancedResearchState, int, int]:
        """
        Add multiple observations with efficient batch deduplication.

        This is the recommended way to add observations in bulk as it:
        - Deduplicates against existing observations
        - Applies memory limits
        - Tracks step association
        - Returns metrics for monitoring

        Args:
            state: Current state
            observations: List of observations to add
            step: Optional step to associate observations with
            config: Optional configuration with memory limits
            observation_index: Optional embedding index for semantic deduplication

        Returns:
            Tuple of (updated_state, added_count, duplicate_count)
        """
        from .observation_models import ensure_structured_observation
        from .memory_config import MemoryOptimizedConfig
        import logging
        logger = logging.getLogger(__name__)

        if not observations:
            return state, 0, 0

        # Initialize observations list if needed
        if "observations" not in state:
            state["observations"] = []

        # Build hash set of existing observations
        existing_hashes = {
            hash(StateManager.safe_get_observation_content(o).lower().strip())
            for o in state["observations"]
            if StateManager.safe_get_observation_content(o).strip()
        }

        # Process new observations
        added = []
        duplicate_count = 0
        excessive_duplicate_warning = False

        for obs in observations:
            # Normalize to StructuredObservation
            structured_obs = ensure_structured_observation(obs)

            # Set step_id for section-specific filtering
            if step:
                structured_obs.step_id = step.step_id

            # Don't convert to dict anymore - keep as Pydantic object
            # obs_dict = structured_obs.to_dict()  # REMOVED

            # Check for duplication
            content = StateManager.safe_get_observation_content(structured_obs).strip()
            if not content:
                continue

            content_lower = content.lower()
            obs_hash = hash(content_lower)

            # Circuit breaker: detect excessive duplicates (indicates bug)
            if obs_hash in existing_hashes:
                duplicate_count += 1
                # Count how many times this specific observation exists
                specific_duplicate_count = sum(
                    1 for o in state["observations"]
                    if hash(StateManager.safe_get_observation_content(o).lower().strip()) == obs_hash
                )
                if specific_duplicate_count >= 10 and not excessive_duplicate_warning:
                    logger.error(
                        f"🚨 CIRCUIT BREAKER: Observation already exists {specific_duplicate_count} times! "
                        f"Content: {content[:100]}..."
                    )
                    excessive_duplicate_warning = True
                continue

            # Add unique observation
            added.append(structured_obs)
            existing_hashes.add(obs_hash)

            # Also add to step if provided
            if step:
                if not step.observations:
                    step.observations = []
                step.observations.append(structured_obs)

        # Batch add all unique observations
        if added:
            state["observations"].extend(added)

            # Apply memory limits
            max_observations = MemoryOptimizedConfig.get_observations_limit(config)
            if len(state["observations"]) > max_observations:
                trimmed = len(state["observations"]) - max_observations
                state["observations"] = state["observations"][-max_observations:]
                logger.info(f"📉 Trimmed {trimmed} old observations to maintain limit of {max_observations}")

        added_count = len(added)

        # Log summary
        if added_count > 0 or duplicate_count > 0:
            logger.info(
                f"📊 Batch processed {len(observations)} observations: "
                f"{added_count} added, {duplicate_count} duplicates skipped, "
                f"total: {len(state['observations'])}"
            )

            # Warn if excessive duplicates (indicates potential bug)
            if duplicate_count > 100:
                logger.warning(
                    f"⚠️  Excessive duplicates: {duplicate_count}/{len(observations)}. "
                    f"This may indicate a bug in observation generation."
                )

        return state, added_count, duplicate_count

    @staticmethod
    def add_search_results(
        state: EnhancedResearchState,
        search_results: List[Any],
        config: Optional[Dict[str, Any]] = None
    ) -> EnhancedResearchState:
        """Add search results to the state with configurable memory limits."""
        from .memory_config import MemoryOptimizedConfig

        # Add new search results
        state["search_results"].extend(search_results)

        # Use configurable search results limit (defaults to 500, can be overridden in base.yaml)
        max_search_results = MemoryOptimizedConfig.get_search_results_limit(config)
        if len(state["search_results"]) > max_search_results:
            state["search_results"] = state["search_results"][-max_search_results:]

        return state
    
    @staticmethod
    def prune_search_results_by_relevance(
        search_results: List['SearchResult'], 
        target_count: int = None
    ) -> List['SearchResult']:
        """
        Intelligently prune search results keeping the most relevant and diverse.
        
        Args:
            search_results: List of SearchResult objects to prune
            target_count: Target number of results to keep (defaults to config constant)
            
        Returns:
            Pruned list of search results sorted by relevance
        """
        from .memory_config import MemoryOptimizedConfig
        from .types import SearchResultType
        
        # Use config constant if no target specified
        if target_count is None:
            target_count = MemoryOptimizedConfig.MAX_PRUNED_SEARCH_RESULTS
        
        if len(search_results) <= target_count:
            return search_results
        
        # Calculate composite scores for each result
        scored_results = []
        for result in search_results:
            # Base relevance score (0-1)
            base_score = getattr(result, 'relevance_score', 0) or getattr(result, 'score', 0) or 0.5
            
            # Source type bonus - prioritize academic sources
            source_bonus = 0.0
            result_type = getattr(result, 'result_type', SearchResultType.WEB)
            if result_type == SearchResultType.ACADEMIC_PAPER:
                source_bonus = 0.3
            elif result_type == SearchResultType.JOURNAL_ARTICLE:
                source_bonus = 0.25
            elif result_type == SearchResultType.WEB:
                source_bonus = 0.1
            
            # Content richness (longer, more detailed content gets bonus)
            content = getattr(result, 'content', '') or ''
            content_score = min(len(content) / 10000, 0.2)  # Up to 0.2 bonus for rich content
            
            # Title/source authority bonus
            title = getattr(result, 'title', '') or ''
            source = getattr(result, 'source', '') or ''
            authority_bonus = 0.0
            
            # Check for authoritative domains
            url = getattr(result, 'url', '') or ''
            if any(domain in url.lower() for domain in ['gov', 'edu', 'org', 'wikipedia']):
                authority_bonus = 0.15
            elif any(domain in url.lower() for domain in ['reuters', 'bloomberg', 'wsj', 'ft.com']):
                authority_bonus = 0.1
            
            # Calculate composite score with weights
            composite_score = (
                base_score * 0.4 +         # 40% weight on base relevance
                source_bonus * 0.25 +      # 25% weight on source type
                content_score * 0.15 +     # 15% weight on content richness
                authority_bonus * 0.2      # 20% weight on source authority
            )
            
            scored_results.append((composite_score, result))
        
        # Sort by composite score (highest first)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Select top results with diversity consideration
        selected = []
        seen_domains = {}
        
        for score, result in scored_results:
            # Always include top 50 results regardless of domain
            if len(selected) < 50:
                selected.append(result)
                # Track domain count
                url = getattr(result, 'url', '') or ''
                if url and '//' in url:
                    domain = url.split('//')[1].split('/')[0]
                    seen_domains[domain] = seen_domains.get(domain, 0) + 1
            else:
                # For remaining slots, ensure diversity
                url = getattr(result, 'url', '') or ''
                domain = None
                if url and '//' in url:
                    domain = url.split('//')[1].split('/')[0]
                
                # Avoid too many results from same domain (max 5 per domain)
                if domain and seen_domains.get(domain, 0) >= 5:
                    continue
                    
                selected.append(result)
                if domain:
                    seen_domains[domain] = seen_domains.get(domain, 0) + 1
            
            if len(selected) >= target_count:
                break
        
        return selected

    @staticmethod
    def prune_state_for_memory(
        state: EnhancedResearchState
    ) -> EnhancedResearchState:
        """
        Prune state to reduce memory usage between workflow nodes.
        
        AGGRESSIVE pruning to prevent worker memory exhaustion.
        Keeps essential data while removing or truncating non-essential accumulated data.
        """
        logger = get_logger(__name__)
        pruned_items = []
        
        # CRITICAL FIX: DO NOT PRUNE OBSERVATIONS - they are the core research data!
        # Observations are already filtered by entity validation, so they're all relevant.
        # Pruning them causes hallucinations and wrong countries in reports.
        # OLD CODE (caused issues): state["observations"] = state["observations"][-3:]
        
        # Log observation count but DO NOT prune
        obs_count = len(state.get("observations", []))
        if obs_count > 0:
            logger.info(f"Preserving ALL {obs_count} observations (no pruning)")
            
        # CRITICAL FIX: DO NOT PRUNE SECTION_RESEARCH_RESULTS - core section mapping data!
        # Section research results are essential for report generation and must never be lost.
        section_research = state.get("section_research_results", {})
        section_count = len(section_research)
        if section_count > 0:
            logger.info(f"Preserving ALL {section_count} section research results (no pruning)")
        
        # Smart pruning for search results using relevance-based selection
        from .memory_config import MemoryOptimizedConfig
        max_pruned = MemoryOptimizedConfig.MAX_PRUNED_SEARCH_RESULTS  # 200
        if len(state.get("search_results", [])) > max_pruned:
            old_count = len(state["search_results"])
            state["search_results"] = StateManager.prune_search_results_by_relevance(
                state["search_results"], 
                target_count=max_pruned
            )
            pruned_items.append(f"search_results: {old_count} -> {max_pruned} (relevance-based)")
            
        # Keep only 2 reflections (was 2, originally 5) - no change needed
        if len(state.get("reflections", [])) > 2:
            old_count = len(state["reflections"])
            state["reflections"] = state["reflections"][-2:]
            pruned_items.append(f"reflections: {old_count} -> 2")
            
        # DO NOT prune step observations - they're needed for section-specific content
        # Each step's observations are critical for its section in the report
        completed_steps = state.get("completed_steps", [])
        for step in completed_steps:
            if hasattr(step, 'observations') and step.observations:
                # Log but don't prune
                obs_count = len(step.observations)
                if obs_count > 0:
                    logger.debug(f"Step {getattr(step, 'step_id', 'unknown')}: preserving {obs_count} observations")
        
        # Limit agent handoffs history more aggressively (keep last 5, was 10)
        if len(state.get("agent_handoffs", [])) > 5:
            old_count = len(state["agent_handoffs"])
            state["agent_handoffs"] = state["agent_handoffs"][-5:]
            pruned_items.append(f"agent_handoffs: {old_count} -> 5")
            
        # Clear error and warning lists more aggressively (keep last 3, was 5)
        if len(state.get("errors", [])) > 3:
            old_count = len(state["errors"])
            state["errors"] = state["errors"][-3:]
            pruned_items.append(f"errors: {old_count} -> 3")
        if len(state.get("warnings", [])) > 3:
            old_count = len(state["warnings"])
            state["warnings"] = state["warnings"][-3:]
            pruned_items.append(f"warnings: {old_count} -> 3")
        
        # Less aggressive message pruning - keep more context for better quality
        old_message_count = len(state.get("messages", []))
        state = StateManager.prune_messages(state, max_messages=20, max_content_length=20000)  # Significantly increased for comprehensive research
        new_message_count = len(state.get("messages", []))
        if new_message_count != old_message_count:
            pruned_items.append(f"messages: {old_message_count} -> {new_message_count}")
        
        # Remove embeddings from search results if they exist (major memory saver)
        search_results = state.get("search_results", [])
        embeddings_removed = 0
        for result in search_results:
            if hasattr(result, 'metadata') and result.metadata:
                if 'embedding' in result.metadata:
                    del result.metadata['embedding']
                    embeddings_removed += 1
                if 'embedding_vector' in result.metadata:
                    del result.metadata['embedding_vector']
                    embeddings_removed += 1
        
        if embeddings_removed > 0:
            pruned_items.append(f"embeddings: removed {embeddings_removed}")
        
        if pruned_items:
            logger.info(f"Memory pruning applied: {', '.join(pruned_items)}")
        
        return state
    
    @staticmethod
    def add_reflection(
        state: EnhancedResearchState,
        reflection: str
    ) -> EnhancedResearchState:
        """Add a self-reflection to the state."""
        # CRITICAL FIX: Ensure state is a dict
        if not isinstance(state, dict):
            raise ValueError(f"Invalid state type for add_reflection: {type(state)}")
            
        # Ensure reflections list exists
        if "reflections" not in state:
            state["reflections"] = []
        
        # Ensure reflection_memory_size exists
        if "reflection_memory_size" not in state:
            state["reflection_memory_size"] = 5  # Default value
            
        state["reflections"].append(reflection)
        
        # Maintain memory size limit
        if len(state["reflections"]) > state["reflection_memory_size"]:
            state["reflections"] = state["reflections"][-state["reflection_memory_size"]:]
        
        return state
    
    @staticmethod
    def update_quality_metrics(
        state: EnhancedResearchState,
        research_quality: Optional[float] = None,
        coverage: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> EnhancedResearchState:
        """Update quality metrics."""
        if research_quality is not None:
            state["research_quality_score"] = research_quality
        if coverage is not None:
            state["coverage_score"] = coverage
        if confidence is not None:
            state["confidence_score"] = confidence
        
        return state
    
    @staticmethod
    def prune_messages(
        state: EnhancedResearchState, 
        max_messages: int = 15, 
        max_content_length: int = 15000
    ) -> EnhancedResearchState:
        """
        Prune messages to prevent unbounded growth.
        
        This is critical for memory management as messages accumulate LLM responses
        which can be very large (thousands of tokens each).
        """
        if "messages" in state and len(state["messages"]) > max_messages:
            # Keep only the most recent messages
            old_count = len(state["messages"])
            state["messages"] = state["messages"][-max_messages:]
            logger = get_logger(__name__)
            logger.info(f"Pruned messages: {old_count} -> {len(state['messages'])}")
        
        # Truncate long message content to prevent memory explosion
        truncated_count = 0
        for msg in state.get("messages", []):
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                if len(msg.content) > max_content_length:
                    msg.content = msg.content[:max_content_length] + "\n[Content truncated for memory management]"
                    truncated_count += 1
        
        if truncated_count > 0:
            logger = get_logger(__name__)
            logger.info(f"Truncated {truncated_count} long messages")
        
        return state
    
    @staticmethod
    def finalize_state(
        state: EnhancedResearchState
    ) -> EnhancedResearchState:
        """Finalize the state at the end of research."""
        state["end_time"] = datetime.now()

        # Safe access to start_time (may not exist in test fixtures)
        start_time = state.get("start_time")
        if start_time:
            duration = (state["end_time"] - start_time).total_seconds()
            state["total_duration_seconds"] = duration

        # Calculate final metrics if not already set
        current_plan = state.get("current_plan")
        if current_plan and not state.get("research_quality_score"):
            # Handle both Plan object and dict
            if hasattr(current_plan, 'steps'):
                steps = current_plan.steps
            elif isinstance(current_plan, dict):
                steps = current_plan.get('steps', [])
            else:
                steps = []

            completed = len([s for s in steps
                           if (hasattr(s, 'status') and s.status == "completed") or
                              (isinstance(s, dict) and s.get('status') == 'completed')])
            total = len(steps)
            state["research_quality_score"] = completed / total if total > 0 else 0

        return state
    
    @staticmethod
    def get_progress(state: EnhancedResearchState) -> ResearchProgress:
        """Get current research progress."""
        plan = state.get("current_plan")
        
        if not plan:
            phase = "planning" if state["plan_iterations"] == 0 else "initialization"
            return ResearchProgress(
                total_steps=0,
                completed_steps=0,
                current_phase=phase,
                estimated_completion=0.0
            )
        
        total_steps = len(plan.steps)
        completed_steps = len([s for s in plan.steps if s.status == "completed"])
        
        # Determine current phase
        if state.get("final_report"):
            phase = "complete"
        elif state.get("current_agent") == "reporter":
            phase = "report_generation"
        elif state.get("enable_grounding") and state.get("current_agent") == "fact_checker":
            phase = "grounding"
        elif completed_steps < total_steps:
            phase = "research"
        else:
            phase = "synthesis"
        
        # Calculate estimated completion
        if total_steps > 0:
            base_completion = (completed_steps / total_steps) * 0.7  # Research is 70%
            if phase == "synthesis":
                base_completion += 0.1
            elif phase == "report_generation":
                base_completion += 0.2
            elif phase == "complete":
                base_completion = 1.0
            
            estimated = min(base_completion, 0.99) if phase != "complete" else 1.0
        else:
            estimated = 0.1 if phase == "planning" else 0.0
        
        # Identify blockers
        blockers = []
        if state.get("errors"):
            blockers.extend(state["errors"][-3:])  # Last 3 errors
        
        if state.get("grounding_results"):
            ungrounded = [r for r in state["grounding_results"] 
                         if r.status == "ungrounded"]
            if ungrounded:
                blockers.append(f"{len(ungrounded)} ungrounded claims need revision")
        
        return ResearchProgress(
            total_steps=total_steps,
            completed_steps=completed_steps,
            current_phase=phase,
            estimated_completion=estimated,
            blockers=blockers
        )

    # ========================================================================
    # Enhanced Research Loop State Management Methods
    # ========================================================================

    @staticmethod
    def update_loop_discoveries(
        state: EnhancedResearchState,
        discovery: str,
        category: str = "general",
        confidence: float = 0.8
    ) -> EnhancedResearchState:
        """Record a new discovery during research loops."""
        discovery_entry = {
            "discovery": discovery,
            "category": category,
            "confidence": confidence,
            "research_loop": state.get("research_loops", 0) + 1,
            "timestamp": datetime.now().isoformat()
        }

        state["loop_discoveries"].append(discovery_entry)
        return state

    @staticmethod
    def update_knowledge_gaps(
        state: EnhancedResearchState,
        gaps: List[str]
    ) -> EnhancedResearchState:
        """Update the list of identified knowledge gaps."""
        # Merge with existing gaps, avoiding duplicates
        existing_gaps = set(state.get("knowledge_gaps", []))
        new_gaps = [gap for gap in gaps if gap not in existing_gaps]

        state["knowledge_gaps"].extend(new_gaps)
        return state

    @staticmethod
    def update_verification_needed(
        state: EnhancedResearchState,
        claims: List[str]
    ) -> EnhancedResearchState:
        """Update the list of claims needing verification."""
        existing_claims = set(state.get("verification_needed", []))
        new_claims = [claim for claim in claims if claim not in existing_claims]

        state["verification_needed"].extend(new_claims)
        return state

    @staticmethod
    def update_deep_dive_topics(
        state: EnhancedResearchState,
        topics: List[str]
    ) -> EnhancedResearchState:
        """Update the list of topics needing deeper investigation."""
        existing_topics = set(state.get("deep_dive_topics", []))
        new_topics = [topic for topic in topics if topic not in existing_topics]

        state["deep_dive_topics"].extend(new_topics)
        return state

    @staticmethod
    def set_loop_focus(
        state: EnhancedResearchState,
        focus: str
    ) -> EnhancedResearchState:
        """Set the current research loop focus."""
        valid_focuses = ["gap_filling", "verification", "deep_dive", "comprehensive_review"]

        if focus not in valid_focuses:
            logger.warning(f"Invalid loop focus '{focus}'. Valid options: {valid_focuses}")
            return state

        state["current_loop_focus"] = focus

        # Record in execution history
        history_entry = {
            "research_loop": state.get("research_loops", 0) + 1,
            "focus": focus,
            "timestamp": datetime.now().isoformat(),
            "context": {
                "gaps_count": len(state.get("knowledge_gaps", [])),
                "verification_count": len(state.get("verification_needed", [])),
                "deep_dive_count": len(state.get("deep_dive_topics", []))
            }
        }

        state["loop_execution_history"].append(history_entry)
        return state

    @staticmethod
    def update_research_quality_metrics(
        state: EnhancedResearchState,
        source_diversity: float,
        factuality_confidence: float,
        quality_issues: List[str] = None
    ) -> EnhancedResearchState:
        """Update research quality metrics for loop assessment."""
        state["source_diversity_score"] = max(0.0, min(1.0, source_diversity))
        state["factuality_confidence"] = max(0.0, min(1.0, factuality_confidence))

        if quality_issues:
            # Merge with existing issues, avoiding duplicates
            existing_issues = set(state.get("quality_issues", []))
            new_issues = [issue for issue in quality_issues if issue not in existing_issues]
            state["quality_issues"].extend(new_issues)

        return state

    @staticmethod
    def prepare_incremental_context(
        state: EnhancedResearchState,
        gap_analysis: Dict[str, Any]
    ) -> EnhancedResearchState:
        """Prepare context for incremental planning based on gap analysis."""
        current_plan = state.get("current_plan")

        incremental_context = {
            "research_loop": state.get("research_loops", 0) + 1,
            "gap_analysis": gap_analysis,
            "existing_plan_summary": "",
            "completed_steps_context": [],
            "planning_mode": "incremental_enhancement",
            "timestamp": datetime.now().isoformat()
        }

        # Add existing plan context
        if current_plan:
            completed_steps = [s for s in current_plan.steps if s.status == "completed"]
            incremental_context["existing_plan_summary"] = f"{len(completed_steps)}/{len(current_plan.steps)} steps completed"
            incremental_context["completed_steps_context"] = [
                {
                    "title": step.title,
                    "step_type": step.step_type.value if hasattr(step.step_type, 'value') else str(step.step_type),
                    "execution_result": getattr(step, 'execution_result', '')[:200] if hasattr(step, 'execution_result') else ""
                }
                for step in completed_steps[-5:]  # Last 5 completed steps
            ]

        state["incremental_context"] = incremental_context
        return state

    @staticmethod
    def should_continue_research_loop(state: EnhancedResearchState) -> bool:
        """
        Determine if research should continue based on COVERAGE, not quality scores.

        Quality scores (diversity, factuality) are calculated AFTER research completes
        by the fact_checker agent, so they cannot be used to decide if research should continue.

        Research continues if:
        1. We haven't exceeded max loops
        2. We have explicit knowledge gaps identified
        3. We have incomplete coverage of required sections
        4. We have insufficient observations for a complete report
        """
        current_loops = state.get("research_loops", 0)
        max_loops = state.get("max_research_loops", 2)  # Reduced from 3 to prevent long runs

        # Hard limit on loops
        if current_loops >= max_loops:
            logger.info(f"[LOOP DECISION] Reached max loops ({max_loops}), stopping research")
            return False

        # ✅ COVERAGE-BASED DECISION LOGIC (replaces quality scores)

        # Check 1: Explicit knowledge gaps identified during research
        gaps = state.get("knowledge_gaps", [])
        if len(gaps) > 0:
            logger.info(f"[LOOP DECISION] Loop {current_loops + 1}: Found {len(gaps)} explicit knowledge gaps - continuing")
            return True

        # Check 2: Incomplete section coverage
        plan = state.get("current_plan")
        if plan and hasattr(plan, "suggested_report_structure"):
            sections = plan.suggested_report_structure or []
            section_research = state.get("section_research_results", {})

            incomplete_sections = [
                s for s in sections
                if s not in section_research or not section_research[s]
            ]

            if len(incomplete_sections) > 2:  # More than 2 sections missing
                logger.info(
                    f"[LOOP DECISION] Loop {current_loops + 1}: {len(incomplete_sections)}/{len(sections)} sections lack research - continuing"
                )
                logger.info(f"[LOOP DECISION] Incomplete sections (first 3): {incomplete_sections[:3]}")
                return True

        # Check 3: Insufficient observations for complete report
        observations = state.get("observations", [])
        min_observations = state.get("min_observations_for_report", 15)

        if len(observations) < min_observations:
            logger.info(
                f"[LOOP DECISION] Loop {current_loops + 1}: Insufficient observations "
                f"({len(observations)}/{min_observations}) - continuing"
            )
            return True

        # Check 4: Significant verification needs (higher threshold than before)
        verification = state.get("verification_needed", [])
        if len(verification) > 5:  # Increased from 2 to avoid over-researching
            logger.info(
                f"[LOOP DECISION] Loop {current_loops + 1}: {len(verification)} items need verification - continuing"
            )
            return True

        # Check 5: Deep dive topics (but must be substantial)
        deep_dives = state.get("deep_dive_topics", [])
        if len(deep_dives) > 2:  # Increased threshold
            logger.info(
                f"[LOOP DECISION] Loop {current_loops + 1}: {len(deep_dives)} deep-dive topics identified - continuing"
            )
            return True

        # All checks passed - research is complete
        logger.info(
            f"[LOOP DECISION] Loop {current_loops + 1}: Research complete - "
            f"observations={len(observations)}, gaps={len(gaps)}, "
            f"verification={len(verification)}, deep_dives={len(deep_dives)} - proceeding to fact checking"
        )
        return False
