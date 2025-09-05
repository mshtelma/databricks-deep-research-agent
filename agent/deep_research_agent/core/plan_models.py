"""
Plan and Step data models for multi-agent research system.

Based on deer-flow implementation patterns for structured planning.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class StepType(str, Enum):
    """Types of steps in a research plan."""
    RESEARCH = "research"
    PROCESSING = "processing"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"


class StepStatus(str, Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Step(BaseModel):
    """Individual step in a research plan."""
    
    step_id: str = Field(description="Unique identifier for the step")
    title: str = Field(description="Brief title of the step")
    description: str = Field(description="Detailed description of what to accomplish")
    step_type: StepType = Field(description="Type of the step")
    status: StepStatus = Field(default=StepStatus.PENDING)
    
    # Execution details
    need_search: bool = Field(default=True, description="Whether this step requires web search")
    search_queries: Optional[List[str]] = Field(default=None, description="Specific search queries for this step")
    
    # Dependencies and context
    depends_on: Optional[List[str]] = Field(default=None, description="Step IDs this step depends on")
    required_context: Optional[List[str]] = Field(default=None, description="Required context from previous steps")
    
    # Results
    execution_result: Optional[str] = Field(default=None, description="Result from executing this step")
    observations: Optional[List[str]] = Field(default=None, description="Observations made during execution")
    citations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Citations collected in this step")
    
    # Quality metrics
    confidence_score: Optional[float] = Field(default=None, description="Confidence in step completion (0-1)")
    grounding_score: Optional[float] = Field(default=None, description="How well grounded the results are (0-1)")
    
    # Timing
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    duration_seconds: Optional[float] = Field(default=None)
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "step_id": "step_001",
                    "title": "Market Analysis Research",
                    "description": "Gather comprehensive data on current AI market trends, size, and growth projections",
                    "step_type": "research",
                    "need_search": True,
                    "search_queries": [
                        "AI market size 2024",
                        "artificial intelligence industry growth rate",
                        "top AI companies market share"
                    ]
                }
            ]
        }


class PlanQuality(BaseModel):
    """Quality assessment of a research plan."""
    
    completeness_score: float = Field(description="How complete the plan is (0-1)")
    feasibility_score: float = Field(description="How feasible the plan is (0-1)")
    clarity_score: float = Field(description="How clear and well-defined the plan is (0-1)")
    coverage_score: float = Field(description="How well the plan covers the research topic (0-1)")
    
    overall_score: float = Field(description="Overall quality score (0-1)")
    issues: List[str] = Field(default_factory=list, description="Identified issues with the plan")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    
    def calculate_overall_score(self) -> float:
        """Calculate overall score from individual metrics."""
        scores = [
            self.completeness_score,
            self.feasibility_score,
            self.clarity_score,
            self.coverage_score
        ]
        return sum(scores) / len(scores)


class Plan(BaseModel):
    """Research plan with structured steps."""
    
    # Core plan information
    plan_id: str = Field(description="Unique identifier for the plan")
    title: str = Field(description="Title of the research plan")
    research_topic: str = Field(description="Original research topic/question")
    
    # Planning metadata
    thought: str = Field(description="Planning reasoning and approach")
    has_enough_context: bool = Field(default=False, description="Whether existing context is sufficient")
    needs_background_investigation: bool = Field(default=True, description="Whether background investigation is needed")
    
    # Steps
    steps: List[Step] = Field(default_factory=list, description="Ordered list of plan steps")
    
    # Plan metadata
    iteration: int = Field(default=0, description="Plan iteration number")
    quality_assessment: Optional[PlanQuality] = Field(default=None, description="Quality assessment of the plan")
    
    # Execution tracking
    current_step_index: int = Field(default=0, description="Index of currently executing step")
    completed_steps: int = Field(default=0, description="Number of completed steps")
    failed_steps: int = Field(default=0, description="Number of failed steps")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    
    def get_next_step(self) -> Optional[Step]:
        """Get the next pending step to execute."""
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                # Check if dependencies are met
                if step.depends_on:
                    deps_met = all(
                        self.get_step_by_id(dep_id).status == StepStatus.COMPLETED
                        for dep_id in step.depends_on
                        if self.get_step_by_id(dep_id)
                    )
                    if not deps_met:
                        continue
                return step
        return None
    
    def get_step_by_id(self, step_id: str) -> Optional[Step]:
        """Get a step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_completed_context(self) -> List[str]:
        """Get accumulated context from completed steps."""
        context = []
        for step in self.steps:
            if step.status == StepStatus.COMPLETED and step.observations:
                context.extend(step.observations)
        return context
    
    def is_complete(self) -> bool:
        """Check if the plan execution is complete."""
        return all(
            step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED]
            for step in self.steps
        )
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of plan execution."""
        return {
            "plan_id": self.plan_id,
            "title": self.title,
            "total_steps": len(self.steps),
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "completion_percentage": (self.completed_steps / len(self.steps) * 100) if self.steps else 0,
            "is_complete": self.is_complete(),
            "quality_score": self.quality_assessment.overall_score if self.quality_assessment else None
        }
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "plan_id": "plan_001",
                    "title": "Comprehensive AI Market Research",
                    "research_topic": "What are the current trends in the AI market?",
                    "thought": "To understand AI market trends, we need to gather data on market size, key players, emerging technologies, and future projections.",
                    "has_enough_context": False,
                    "steps": [
                        {
                            "step_id": "step_001",
                            "title": "Current Market Analysis",
                            "description": "Research current AI market size and growth",
                            "step_type": "research",
                            "need_search": True
                        },
                        {
                            "step_id": "step_002",
                            "title": "Technology Trends",
                            "description": "Identify emerging AI technologies and innovations",
                            "step_type": "research",
                            "need_search": True
                        }
                    ]
                }
            ]
        }


class PlanFeedback(BaseModel):
    """Human or automatic feedback on a plan."""
    
    feedback_type: str = Field(description="Type of feedback: human, automatic, quality_check")
    feedback: str = Field(description="The feedback content")
    suggestions: List[str] = Field(default_factory=list, description="Specific suggestions for improvement")
    requires_revision: bool = Field(default=False, description="Whether the plan needs revision")
    approved: bool = Field(default=False, description="Whether the plan is approved for execution")
    timestamp: datetime = Field(default_factory=datetime.now)