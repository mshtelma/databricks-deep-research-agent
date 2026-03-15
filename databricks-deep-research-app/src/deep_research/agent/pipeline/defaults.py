"""
Default Pipeline Configurations
================================

Pre-defined pipeline templates for common use cases.

Available pipelines:
- DEFAULT_DEEP_RESEARCH_PIPELINE: Full 5-agent research pipeline
- SIMPLE_RESEARCH_PIPELINE: Streamlined 3-agent pipeline
- REACT_LOOP_PIPELINE: Single-agent ReAct loop pattern
"""

from deep_research.agent.pipeline.config import AgentConfig, AgentType, PipelineConfig


# =============================================================================
# Default Deep Research Pipeline (5-agent architecture)
# =============================================================================

DEFAULT_DEEP_RESEARCH_PIPELINE = PipelineConfig(
    name="deep-research",
    description="Full 5-agent deep research pipeline with reflection",
    agents=[
        # Coordinator: Query classification and complexity assessment
        AgentConfig(
            agent_type=AgentType.COORDINATOR,
            name="coordinator",
            model_tier="analytical",
            next_on_success="background",
            config={"enable_clarification": True},
        ),
        # Background: Quick web search for initial context
        AgentConfig(
            agent_type=AgentType.BACKGROUND,
            name="background",
            model_tier="simple",
            next_on_success="planner",
            config={"max_search_queries": 3},
        ),
        # Planner: Create research plan
        AgentConfig(
            agent_type=AgentType.PLANNER,
            name="planner",
            model_tier="analytical",
            next_on_success="researcher",
            config={"max_plan_iterations": 3},
        ),
        # Researcher: Execute research steps
        AgentConfig(
            agent_type=AgentType.RESEARCHER,
            name="researcher",
            model_tier="analytical",
            next_on_success="reflector",
            config={
                "max_search_queries": 3,
                "max_urls_to_crawl": 5,
            },
        ),
        # Reflector: Evaluate and decide next action
        AgentConfig(
            agent_type=AgentType.REFLECTOR,
            name="reflector",
            model_tier="analytical",
            loop_condition="CONTINUE",
            loop_back_to="researcher",
            next_on_success="synthesizer",
            config={},
        ),
        # Synthesizer: Generate final report
        AgentConfig(
            agent_type=AgentType.SYNTHESIZER,
            name="synthesizer",
            model_tier="complex",
            next_on_success=None,  # End of pipeline
            config={},
        ),
    ],
    start_agent="coordinator",
    max_iterations=15,
    timeout_seconds=300,
)


# =============================================================================
# Simple Research Pipeline (3-agent streamlined)
# =============================================================================

SIMPLE_RESEARCH_PIPELINE = PipelineConfig(
    name="simple-research",
    description="Streamlined 3-agent pipeline without reflection",
    agents=[
        # Planner: Create focused research plan
        AgentConfig(
            agent_type=AgentType.PLANNER,
            name="planner",
            model_tier="analytical",
            next_on_success="researcher",
            config={"max_steps": 3},
        ),
        # Researcher: Execute all steps sequentially
        AgentConfig(
            agent_type=AgentType.RESEARCHER,
            name="researcher",
            model_tier="analytical",
            loop_condition="step_incomplete",
            loop_back_to="researcher",
            next_on_success="synthesizer",
            config={
                "max_search_queries": 2,
                "max_urls_to_crawl": 3,
            },
        ),
        # Synthesizer: Generate final report
        AgentConfig(
            agent_type=AgentType.SYNTHESIZER,
            name="synthesizer",
            model_tier="analytical",
            next_on_success=None,
            config={},
        ),
    ],
    start_agent="planner",
    max_iterations=10,
    timeout_seconds=180,
)


# =============================================================================
# ReAct Loop Pipeline (Single-agent pattern)
# =============================================================================

REACT_LOOP_PIPELINE = PipelineConfig(
    name="react-loop",
    description="Single researcher with ReAct loop pattern",
    agents=[
        # Researcher with ReAct loop - handles its own planning and reflection
        AgentConfig(
            agent_type=AgentType.RESEARCHER,
            name="researcher",
            model_tier="analytical",
            loop_condition="not_complete",
            loop_back_to="researcher",
            next_on_success="synthesizer",
            max_iterations=20,
            config={
                "mode": "react",
                "max_tool_calls": 15,
            },
        ),
        # Synthesizer: Generate final report
        AgentConfig(
            agent_type=AgentType.SYNTHESIZER,
            name="synthesizer",
            model_tier="analytical",
            next_on_success=None,
            config={},
        ),
    ],
    start_agent="researcher",
    max_iterations=25,
    timeout_seconds=240,
)


# =============================================================================
# Pipeline Registry
# =============================================================================

_PIPELINES: dict[str, PipelineConfig] = {
    "deep-research": DEFAULT_DEEP_RESEARCH_PIPELINE,
    "simple-research": SIMPLE_RESEARCH_PIPELINE,
    "react-loop": REACT_LOOP_PIPELINE,
}


def get_default_pipeline(name: str = "deep-research") -> PipelineConfig:
    """Get a default pipeline configuration by name.

    Args:
        name: Pipeline name. One of:
            - "deep-research": Full 5-agent pipeline (default)
            - "simple-research": Streamlined 3-agent pipeline
            - "react-loop": Single-agent ReAct pattern

    Returns:
        PipelineConfig for the requested pipeline

    Raises:
        ValueError: If pipeline name is not recognized
    """
    if name not in _PIPELINES:
        available = ", ".join(_PIPELINES.keys())
        raise ValueError(
            f"Unknown pipeline: '{name}'. Available pipelines: {available}"
        )
    return _PIPELINES[name]


def list_pipelines() -> list[str]:
    """Get list of available default pipeline names."""
    return list(_PIPELINES.keys())
