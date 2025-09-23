"""
Enhanced state management for multi-agent research system.

Extends the existing state with support for planning, grounding,
and multi-agent coordination.
"""

from typing import List, Dict, Any, Optional, Literal, Union
from datetime import datetime
from pydantic import BaseModel, Field

from langgraph.graph import add_messages
from typing_extensions import Annotated, TypedDict

from deep_research_agent.core.plan_models import Plan, Step, PlanFeedback, PlanQuality
from deep_research_agent.core.report_styles import ReportStyle
from deep_research_agent.core.grounding import (
    GroundingResult, 
    FactualityReport, 
    Contradiction,
    VerificationLevel
)
from deep_research_agent.core import (
    ResearchContext,
    SearchResult,
    Citation,
    ResearchQuery,
    get_logger
)
from deep_research_agent.core.observation_models import StructuredObservation


# Reducer functions for concurrent state updates
def use_latest_plan(left: Optional[Plan], right: Optional[Plan]) -> Optional[Plan]:
    """Reducer that uses the most recent non-None plan."""
    return right if right is not None else left


def merge_lists(left: List[Any], right: List[Any]) -> List[Any]:
    """Reducer that merges two lists, avoiding duplicates."""
    if not left:
        return right
    if not right:
        return left
    
    # MEMORY OPTIMIZATION: Limit list sizes to prevent unbounded growth
    combined = left + right
    
    # Apply memory-conscious limits based on list content type
    # Import constants from memory_config for consistency
    from deep_research_agent.core.memory_config import MemoryOptimizedConfig
    
    MAX_OBSERVATIONS = MemoryOptimizedConfig.MAX_OBSERVATIONS        # Keep most recent observations
    MAX_SEARCH_RESULTS = MemoryOptimizedConfig.MAX_SEARCH_RESULTS   # Keep most relevant search results
    MAX_CITATIONS = MemoryOptimizedConfig.MAX_CITATIONS             # Citations are smaller, allow more
    MAX_REFLECTIONS = MemoryOptimizedConfig.MAX_REFLECTIONS         # Limit reflection history
    MAX_AGENT_HANDOFFS = MemoryOptimizedConfig.MAX_AGENT_HANDOFFS   # Limit handoff history
    MAX_GENERAL = MemoryOptimizedConfig.MAX_GENERAL_LIST_SIZE        # Default limit for other lists
    
    # Determine appropriate limit (heuristic based on list content)
    if combined and hasattr(combined[0], 'content') and len(str(combined[0])) > 1000:
        # Large content items (like SearchResults)
        limit = MAX_SEARCH_RESULTS if 'SearchResult' in str(type(combined[0])) else MAX_GENERAL
    elif combined and isinstance(combined[0], str) and len(combined[0]) > 100:
        # Text observations/reflections
        limit = MAX_OBSERVATIONS
    elif combined and isinstance(combined[0], dict) and 'from_agent' in str(combined[0]):
        # Agent handoffs
        limit = MAX_AGENT_HANDOFFS
    else:
        limit = MAX_GENERAL
    
    # Keep most recent items if over limit
    if len(combined) > limit:
        combined = combined[-limit:]
    
    return combined


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
    
    # Execution state
    observations: Annotated[
        List[Union[str, StructuredObservation, Dict[str, Any]]],
        merge_lists
    ]  # Accumulated observations from all steps
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
    
    # Entity validation
    requested_entities: Annotated[List[str], use_latest_value]
    entity_violations: Annotated[List[Dict[str, Any]], merge_lists]
    
    # Citations and references
    citations: Annotated[List[Citation], merge_lists]
    citation_style: Annotated[str, use_latest_value]  # APA, MLA, Chicago, etc.
    
    # Report generation
    report_style: Annotated[ReportStyle, use_latest_value]
    final_report: Annotated[Optional[str], use_latest_value]
    report_sections: Annotated[Optional[Dict[str, str]], use_latest_value]  # Section name -> content
    
    # Reflexion and self-improvement
    enable_reflexion: Annotated[bool, use_latest_value]
    reflections: Annotated[List[str], merge_lists]  # Self-reflection feedback
    reflection_memory_size: Annotated[int, use_latest_value]
    
    # Agent coordination
    current_agent: Annotated[str, use_latest_value]  # Which agent is currently active
    agent_handoffs: Annotated[List[Dict[str, Any]], merge_lists]  # History of agent handoffs
    
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
        from deep_research_agent.core.report_styles import ReportStyle
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
            
            # Background investigation
            enable_background_investigation=config.get("enable_background_investigation", True),
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
            
            # Grounding
            enable_grounding=config.get("grounding", {}).get("enabled", True),
            grounding_results=None,
            factuality_report=None,
            contradictions=None,
            factuality_score=None,
            verification_level=VerificationLevel(
                config.get("grounding", {}).get("verification_level", "moderate")
            ),
            
            # Entity validation
            requested_entities=[],
            entity_violations=[],
            
            # Citations
            citations=[],
            citation_style=config.get("citation_style", "APA"),
            
            # Report - enhanced logging for debugging adaptive structure
            report_style=StateManager._initialize_report_style(config),
            final_report=None,
            report_sections=None,
            
            # Reflexion
            enable_reflexion=config.get("enable_reflexion", True),
            reflections=[],
            reflection_memory_size=config.get("reflection_memory_size", 5),
            
            # Coordination
            current_agent="coordinator",
            agent_handoffs=[],
            
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

            # User preferences
            user_preferences=config.get("user_preferences", {}),

            # Section research accumulation
            section_research_results={},
        )
    
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
        step: Optional[Step] = None
    ) -> EnhancedResearchState:
        """Add an observation to the state with memory limits, normalizing to StructuredObservation."""
        from deep_research_agent.core.observation_models import ensure_structured_observation
        
        # Normalize observation to StructuredObservation
        structured_obs = ensure_structured_observation(observation)
        
        # Add to global observations with size limit
        state["observations"].append(structured_obs)
        
        # Keep only the last 20 observations to prevent unbounded growth
        max_observations = 20
        if len(state["observations"]) > max_observations:
            state["observations"] = state["observations"][-max_observations:]
        
        if step:
            if not step.observations:
                step.observations = []
            step.observations.append(structured_obs)
            
            # Also limit step observations
            if len(step.observations) > 10:  # Smaller limit per step
                step.observations = step.observations[-10:]
        
        return state
    
    @staticmethod
    def add_search_results(
        state: EnhancedResearchState,
        search_results: List[Any]
    ) -> EnhancedResearchState:
        """Add search results to the state with memory limits."""
        # Add new search results
        state["search_results"].extend(search_results)
        
        # Keep only the last 50 search results to prevent unbounded growth
        max_search_results = 50
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
        from deep_research_agent.core.memory_config import MemoryOptimizedConfig
        from deep_research_agent.core.types import SearchResultType
        
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
        elif "section_research_results" in state:
            logger.warning("section_research_results exists but is empty - this may indicate a bug")
        
        # Smart pruning for search results using relevance-based selection
        from deep_research_agent.core.memory_config import MemoryOptimizedConfig
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
        
        if state["start_time"]:
            duration = (state["end_time"] - state["start_time"]).total_seconds()
            state["total_duration_seconds"] = duration
        
        # Calculate final metrics if not already set
        if state["current_plan"] and not state["research_quality_score"]:
            completed = len([s for s in state["current_plan"].steps 
                           if s.status == "completed"])
            total = len(state["current_plan"].steps)
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
