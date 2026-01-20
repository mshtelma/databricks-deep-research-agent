"""
Pipeline Module
===============

Provides declarative pipeline configuration and execution for research agents.

This module enables:
- Declarative pipeline configuration via PipelineConfig
- Customizable agent sequences with transitions
- Plugin-provided custom phases
- Multiple default pipeline templates

Example usage:
    from deep_research.agent.pipeline import (
        PipelineConfig,
        AgentConfig,
        AgentType,
        DEFAULT_DEEP_RESEARCH_PIPELINE,
    )

    # Create custom pipeline
    config = PipelineConfig(
        name="custom-research",
        description="Custom 3-agent pipeline",
        agents=[
            AgentConfig(agent_type=AgentType.PLANNER),
            AgentConfig(agent_type=AgentType.RESEARCHER),
            AgentConfig(agent_type=AgentType.SYNTHESIZER),
        ],
    )
"""

from deep_research.agent.pipeline.config import (
    AgentConfig,
    AgentType,
    PipelineConfig,
)
from deep_research.agent.pipeline.defaults import (
    DEFAULT_DEEP_RESEARCH_PIPELINE,
    SIMPLE_RESEARCH_PIPELINE,
    REACT_LOOP_PIPELINE,
    get_default_pipeline,
)
from deep_research.agent.pipeline.protocols import (
    CustomPhase,
    PipelineCustomization,
    PipelineCustomizer,
    PhaseInsertion,
    PhaseProvider,
)
from deep_research.agent.pipeline.executor import (
    ExecutionResult,
    PipelineExecutor,
    create_executor_for_pipeline,
)

__all__ = [
    # Configuration
    "AgentType",
    "AgentConfig",
    "PipelineConfig",
    # Default pipelines
    "DEFAULT_DEEP_RESEARCH_PIPELINE",
    "SIMPLE_RESEARCH_PIPELINE",
    "REACT_LOOP_PIPELINE",
    "get_default_pipeline",
    # Protocols and customization
    "CustomPhase",
    "PipelineCustomization",
    "PipelineCustomizer",
    "PhaseInsertion",
    "PhaseProvider",
    # Executor
    "ExecutionResult",
    "PipelineExecutor",
    "create_executor_for_pipeline",
]
