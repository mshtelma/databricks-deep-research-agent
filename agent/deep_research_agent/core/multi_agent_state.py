"""
Enhanced state management for multi-agent research system.

Extends the existing state with support for planning, grounding,
and multi-agent coordination.
"""

from typing import List, Dict, Any, Optional, Literal
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
    ResearchQuery
)


class EnhancedResearchState(TypedDict):
    """
    Enhanced state for multi-agent research system.
    
    Extends the basic state with planning, grounding, and style capabilities.
    """
    
    # Core conversation state
    messages: Annotated[List, add_messages]
    
    # Research context
    research_topic: str
    research_context: Optional[ResearchContext]
    
    # Planning state
    current_plan: Optional[Plan]
    plan_iterations: int
    plan_feedback: Optional[List[PlanFeedback]]
    plan_quality: Optional[PlanQuality]
    enable_iterative_planning: bool
    max_plan_iterations: int
    
    # Background investigation
    enable_background_investigation: bool
    background_investigation_results: Optional[str]
    
    # Execution state
    observations: List[str]  # Accumulated observations from all steps
    completed_steps: List[Step]
    current_step: Optional[Step]
    current_step_index: int
    
    # Research results
    search_results: List[SearchResult]
    search_queries: List[ResearchQuery]
    
    # Grounding and factuality
    enable_grounding: bool
    grounding_results: Optional[List[GroundingResult]]
    factuality_report: Optional[FactualityReport]
    contradictions: Optional[List[Contradiction]]
    factuality_score: Optional[float]
    verification_level: VerificationLevel
    
    # Citations and references
    citations: List[Citation]
    citation_style: str  # APA, MLA, Chicago, etc.
    
    # Report generation
    report_style: ReportStyle
    final_report: Optional[str]
    report_sections: Optional[Dict[str, str]]  # Section name -> content
    
    # Reflexion and self-improvement
    enable_reflexion: bool
    reflections: List[str]  # Self-reflection feedback
    reflection_memory_size: int
    
    # Agent coordination
    current_agent: str  # Which agent is currently active
    agent_handoffs: List[Dict[str, Any]]  # History of agent handoffs
    
    # Quality metrics
    research_quality_score: Optional[float]
    coverage_score: Optional[float]
    confidence_score: Optional[float]
    
    # Configuration
    enable_deep_thinking: bool
    enable_human_feedback: bool
    auto_accept_plan: bool
    
    # Timing and metadata
    start_time: datetime
    end_time: Optional[datetime]
    total_duration_seconds: Optional[float]
    
    # Error handling
    errors: List[str]
    warnings: List[str]
    
    # User preferences
    user_preferences: Optional[Dict[str, Any]]


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
            
            # Citations
            citations=[],
            citation_style=config.get("citation_style", "APA"),
            
            # Report
            report_style=ReportStyle(config.get("report", {}).get("default_style", config.get("default_report_style", "professional"))),
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
            user_preferences=config.get("user_preferences", {})
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
        observation: str,
        step: Optional[Step] = None
    ) -> EnhancedResearchState:
        """Add an observation to the state."""
        state["observations"].append(observation)
        
        if step:
            if not step.observations:
                step.observations = []
            step.observations.append(observation)
        
        return state
    
    @staticmethod
    def add_reflection(
        state: EnhancedResearchState,
        reflection: str
    ) -> EnhancedResearchState:
        """Add a self-reflection to the state."""
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