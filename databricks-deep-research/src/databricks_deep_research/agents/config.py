"""Agent node configuration models and subtype defaults."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Pool-related configs
# ---------------------------------------------------------------------------


class PoolWriteConfig(BaseModel):
    """Describes how an agent writes items to a shared pool."""

    model_config = ConfigDict(extra="forbid")
    pool: str
    extract: str  # Jinja / dot-path expression on agent output
    transform: str | None = None  # Optional transformation template


class PromptCompactionConfig(BaseModel):
    """Controls prompt-context compaction for injected pool content."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["none", "dedupe", "summarize", "auto"] = "none"
    max_total_chars: int = 0
    target_chars: int = 0
    summary_model_tier: str = "simple"
    dedupe_key: Literal["auto", "url", "title", "text"] = "auto"


class PoolInjectConfig(BaseModel):
    """Describes how pool contents are injected into an agent prompt."""

    model_config = ConfigDict(extra="forbid")
    pool: str
    threshold: float = 0.0  # BM25 / relevance threshold
    format: Literal["text", "json", "markdown"] = "text"
    max_items: int = 20  # max items to inject
    max_item_chars: int = 0  # 0 = unlimited; >0 truncates each item
    compaction: PromptCompactionConfig | None = None


class SynthesisContextFieldConfig(BaseModel):
    """Controls how one synthesizer context field is populated."""

    model_config = ConfigDict(extra="forbid")
    max_items: int = 20
    max_item_chars: int = 0
    compaction: PromptCompactionConfig | None = None


class SynthesisContextConfig(BaseModel):
    """Controls synthesizer-specific context materialization."""

    model_config = ConfigDict(extra="forbid")
    observations: SynthesisContextFieldConfig | None = None
    sources: SynthesisContextFieldConfig | None = None
    fallback_discovery_sources: SynthesisContextFieldConfig | None = None


# ---------------------------------------------------------------------------
# Primary agent node config
# ---------------------------------------------------------------------------


class AgentNodeConfig(BaseModel):
    """Full configuration for a single agent (LLM) node."""

    model_config = ConfigDict(extra="forbid")

    subtype: str  # coordinator, planner, researcher, reflector, synthesizer, evaluator
    model_tier: str = "analytical"
    system_prompt: str = ""
    user_prompt_template: str = ""
    input_keys: list[str] = Field(default_factory=list)
    output_key: str = "output"
    output_mode: str = "text"  # text, json, structured
    output_format: str = "text"  # text, markdown, json
    output_schema: dict[str, Any] | None = None
    grounding_mode: Literal["none", "classical_lite", "reclaim"] | None = None
    tools: list[str | dict[str, Any]] = Field(default_factory=list)
    pool_writes: list[PoolWriteConfig] = Field(default_factory=list)
    pool_tools: list[str] = Field(default_factory=list)
    max_tool_calls: int | None = None
    max_retries: int = 2
    max_result_chars: int = 4000  # 0=unlimited; >0 truncates old tool results
    compaction_strategy: Literal["truncate", "mask"] = "truncate"
    conversation_budget: int | None = None
    pool_inject: list[PoolInjectConfig] = Field(default_factory=list)
    synthesis_context: SynthesisContextConfig | None = None
    output_model: Any = None  # Pydantic model class for structured output


# ---------------------------------------------------------------------------
# Non-agent node configs
# ---------------------------------------------------------------------------


class ToolNodeConfig(BaseModel):
    """Configuration for a pure-tool (no LLM) node."""

    model_config = ConfigDict(extra="forbid")
    ref: dict[str, Any]  # tool reference descriptor
    input_mapping: dict[str, str] = Field(default_factory=dict)
    output_key: str = "tool_result"


class LoopNodeConfig(BaseModel):
    """Configuration for a loop control node."""

    model_config = ConfigDict(extra="forbid")
    until: dict[str, Any]  # Serialised Condition (StateCondition / LLMCondition / Composite)
    min_iterations: int = 1
    max_iterations: int = 10


class ConditionalNodeConfig(BaseModel):
    """Configuration for a conditional branching node."""

    model_config = ConfigDict(extra="forbid")
    conditions: list[dict[str, Any]]  # list of serialised ConditionBranch
    default_branch: int = 0


class SubworkflowNodeConfig(BaseModel):
    """Configuration for invoking a nested workflow."""

    model_config = ConfigDict(extra="forbid")
    ref: str  # workflow name or path
    params: dict[str, Any] = Field(default_factory=dict)
    input_mapping: dict[str, str] = Field(default_factory=dict)
    output_mapping: dict[str, str] = Field(default_factory=dict)
    output_key: str = "subworkflow_result"
    pool_mode: str = "inherit"  # inherit, isolate, merge


class PlanAndExecuteNodeConfig(BaseModel):
    """Configuration for the plan-and-execute meta-node."""

    model_config = ConfigDict(extra="forbid")
    planner: dict[str, Any]  # Serialised AgentNodeConfig for the planner agent
    items_path: str = "steps"  # dot-path into planner output for the iterable
    item_state_key: str = "current_step"
    body: dict[str, Any] = Field(default_factory=dict)  # Serialised child node(s) to run per item
    evaluator: dict[str, Any] | None = None  # Optional evaluator agent config
    max_iterations: int = 10
    min_iterations: int = 1
    max_replan_cycles: int = 3
    complete_on_exhaustion: bool = True
    planner_guidance: str = ""  # Free-text guidance injected into planner prompt
    synthesis_metadata: dict[str, str] = Field(default_factory=dict)  # Key-value pairs written to state for synthesizer


# ---------------------------------------------------------------------------
# Subtype defaults
#
# NOTE: ``input_keys`` values below are documentation only — at runtime,
# input_keys are auto-detected from prompt templates by the harness
# (``execute_agent()`` in ``harness.py``) using
# ``SafeTemplateRenderer.extract_variables()``.  Explicit ``input_keys``
# in YAML workflow definitions always override auto-detection.
# ---------------------------------------------------------------------------

SUBTYPE_DEFAULTS: dict[str, dict[str, Any]] = {
    "coordinator": {
        "model_tier": "simple",
        "output_key": "coordination",
        "output_format": "json",
        "input_keys": ["query"],
        "tools": [],
        "pool_writes": [],
        "pool_tools": [],
    },
    "planner": {
        "model_tier": "analytical",
        "output_key": "plan",
        "output_format": "json",
        "input_keys": ["query", "background"],
        "tools": [],
        "pool_writes": [],
        "pool_tools": [],
    },
    "researcher": {
        "model_tier": "analytical",
        "output_key": "findings",
        "output_format": "json",
        "input_keys": ["query", "current_step", "plan"],
        "tools": ["web_search", "web_crawl"],
        "pool_writes": [{"pool": "sources", "extract": "sources"}],
        "pool_tools": ["pool_search"],
    },
    "reflector": {
        "model_tier": "analytical",
        "output_key": "reflection",
        "output_format": "json",
        "input_keys": [
            "query", "plan_summary", "findings", "current_step",
            "remaining_steps", "total_steps", "steps_completed",
            "min_steps", "step_title", "iteration", "observation",
            "all_observations", "sources_count", "source_topics",
            "source_quality",
        ],
        "tools": [],
        "pool_writes": [],
        "pool_tools": [],
    },
    "synthesizer": {
        "model_tier": "complex",
        "output_key": "report",
        "output_format": "markdown",
        "input_keys": ["query", "plan"],
        "tools": [],
        "pool_writes": [],
        "pool_tools": ["pool_search"],
    },
    "evaluator": {
        "model_tier": "analytical",
        "output_key": "evaluation",
        "output_format": "json",
        "input_keys": [
            "query", "plan_summary", "findings", "current_step",
            "remaining_steps", "total_steps", "steps_completed",
            "min_steps", "step_title", "iteration", "observation",
            "all_observations", "sources_count", "source_topics",
            "source_quality",
        ],
        "tools": [],
        "pool_writes": [],
        "pool_tools": [],
    },
}
