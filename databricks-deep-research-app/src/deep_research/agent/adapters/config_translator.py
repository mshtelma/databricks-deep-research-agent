"""Config translator — OrchestrationConfig → WorkflowDefinition.

This is the highest-risk adapter: it translates the app's runtime
configuration into a framework workflow tree.  Every combination of
config flags must produce the correct tree shape.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)

from deep_research.agent.config import get_report_limits, get_research_type_config

if TYPE_CHECKING:
    from deep_research.agent.orchestration_config import OrchestrationConfig

logger = logging.getLogger(__name__)


def translate(
    config: OrchestrationConfig,
    available_tools: list[str] | None = None,
) -> WorkflowDefinition:
    """Translate an ``OrchestrationConfig`` into a ``WorkflowDefinition``.

    Routes to different workflow shapes based on ``query_mode``:

    * ``"simple"`` — coordinator only (direct response, no research).
    * ``"web_search"`` — coordinator → single researcher → synthesizer.
    * ``"deep_research"`` (default) — full pipeline with background,
      plan_and_execute, and synthesis.

    Args:
        config: The app's OrchestrationConfig dataclass instance.
        available_tools: Optional list of tool names registered in the
            ToolRegistry.  When provided, enterprise tools (non-web) are
            included in researcher/background node configs based on
            ``source_scope``.

    Returns:
        A WorkflowDefinition tree ready for executor.
    """
    query_mode = config.query_mode

    if query_mode == "simple":
        return _build_simple_workflow(config)
    elif query_mode == "web_search":
        return _build_web_search_workflow(config, available_tools)
    return _build_deep_research_workflow(config, available_tools)


def _build_simple_workflow(config: OrchestrationConfig) -> WorkflowDefinition:
    """Build a minimal workflow for simple queries (coordinator only)."""
    root = WorkflowNode(
        id="main",
        type=NodeType.sequence,
        label="Simple Query",
        children=[_build_coordinator(config)],
    )

    return WorkflowDefinition(
        id="simple",
        name="Simple Query",
        root=root,
        pools=[],
        required_inputs=["query"],
        output_keys=["report"],
    )


def _build_web_search_workflow(
    config: OrchestrationConfig,
    available_tools: list[str] | None = None,
) -> WorkflowDefinition:
    """Build a lightweight workflow for web search queries.

    Pipeline: coordinator → single researcher (max 5 tool calls) → synthesizer.
    No background investigation, no reflector, no planner loop.
    """
    tools = _resolve_search_tools(config, available_tools)

    researcher_config: dict[str, Any] = {
        "subtype": "researcher",
        "model_tier": "analytical",
        "input_keys": ["query"],
        "output_key": "findings",
        "tools": tools,
        "pool_writes": [
            {"pool": "observations", "extract": "findings"},
            {"pool": "sources", "extract": "sources"},
        ],
        "max_tool_calls": 5,
    }

    # Inject system instructions if present
    if config.system_instructions:
        researcher_config["system_prompt"] = config.system_instructions

    children: list[WorkflowNode] = [
        _build_coordinator(config),
        WorkflowNode(
            id="researcher",
            type=NodeType.agent,
            label="Web Researcher",
            config=researcher_config,
        ),
        _build_synthesizer(config),
    ]

    root = WorkflowNode(
        id="main",
        type=NodeType.sequence,
        label="Web Search Pipeline",
        children=children,
    )

    pools = _build_pools(config)

    return WorkflowDefinition(
        id="web_search",
        name="Web Search",
        root=root,
        pools=pools,
        required_inputs=["query"],
        output_keys=["report"],
    )


def _build_deep_research_workflow(
    config: OrchestrationConfig,
    available_tools: list[str] | None = None,
) -> WorkflowDefinition:
    """Build the full deep research pipeline.

    Pipeline: coordinator → background → plan_and_execute → synthesizer.
    """
    children: list[WorkflowNode] = []

    # 1. Coordinator
    children.append(_build_coordinator(config))

    # 2. Background (optional)
    if config.enable_background_investigation:
        children.append(_build_background(config, available_tools))

    # 3. Research cycle (plan_and_execute)
    children.append(_build_research_cycle(config, available_tools))

    # 4. Synthesizer
    children.append(_build_synthesizer(config))

    root = WorkflowNode(
        id="main",
        type=NodeType.sequence,
        label="Deep Research Pipeline",
        children=children,
    )

    pools = _build_pools(config)

    return WorkflowDefinition(
        id="deep_research",
        name="Deep Research",
        root=root,
        pools=pools,
        required_inputs=["query"],
        output_keys=["report"],
    )


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------


def _build_coordinator(_config: OrchestrationConfig) -> WorkflowNode:
    """Build coordinator node."""
    return WorkflowNode(
        id="coordinator",
        type=NodeType.agent,
        label="Query Classifier",
        config={
            "subtype": "coordinator",
            "model_tier": "simple",
            "input_keys": ["query"],
            "output_key": "coordination",
        },
    )


def _build_background(
    config: OrchestrationConfig,
    available_tools: list[str] | None = None,
) -> WorkflowNode:
    """Build background investigation node."""
    tools = _resolve_search_tools(config, available_tools)
    return WorkflowNode(
        id="background",
        type=NodeType.agent,
        label="Background Investigator",
        config={
            "subtype": "background",
            "model_tier": "simple",
            "input_keys": ["query"],
            "output_key": "background",
            "tools": tools,
            "pool_writes": [
                {"pool": "discovery_sources", "extract": "sources"},
            ],
            "max_tool_calls": 5,
        },
    )


def _build_research_cycle(
    config: OrchestrationConfig,
    available_tools: list[str] | None = None,
) -> WorkflowNode:
    """Build plan_and_execute node for the research cycle."""
    tools = _resolve_search_tools(config, available_tools)
    depth = config.research_depth
    min_steps, max_steps = _depth_to_step_limits(depth)

    # Planner config
    planner_config: dict[str, Any] = {
        "subtype": "planner",
        "model_tier": "analytical",
        "input_keys": ["query", "background"],
        "output_key": "plan",
    }

    # Body: researcher per step
    body_config: dict[str, Any] = {
        "subtype": "researcher",
        "model_tier": "analytical",
        "input_keys": ["query", "current_step", "plan"],
        "output_key": "findings",
        "tools": tools,
        "pool_writes": [
            {"pool": "observations", "extract": "findings"},
            {"pool": "sources", "extract": "sources"},
        ],
        "max_tool_calls": 15,
    }

    # Inject system instructions if present
    if config.system_instructions:
        planner_config["system_prompt"] = config.system_instructions
        body_config["system_prompt"] = config.system_instructions

    body_node: dict[str, Any] = {
        "id": "researcher",
        "type": "agent",
        "label": "Researcher",
        "config": body_config,
    }

    # Evaluator: reflector
    evaluator_config: dict[str, Any] = {
        "subtype": "reflector",
        "model_tier": "analytical",
        "input_keys": ["query", "plan", "findings", "current_step"],
        "output_key": "evaluation",
        "pool_inject": [{"pool": "observations", "threshold": 0}],
    }

    # Word limits by depth for synthesizer
    word_limits = {
        "light": (200, 500), "medium": (400, 1000),
        "extended": (800, 2000), "auto": (400, 1500),
    }
    min_w, max_w = word_limits.get(depth, (400, 1500))

    pe_config: dict[str, Any] = {
        "planner": planner_config,
        "items_path": "steps",
        "item_state_key": "current_step",
        "body": body_node,
        "evaluator": evaluator_config,
        "max_iterations": max_steps,
        "min_iterations": min_steps,
        "max_replan_cycles": config.max_plan_iterations,
        "synthesis_metadata": {
            "research_depth": depth,
            "min_words": str(min_w),
            "max_words": str(max_w),
        },
    }

    # Handle manual workflow mode
    if config.workflow_mode == "manual" and config.manual_steps:
        pe_config["evaluator"] = None  # No evaluator for manual mode

    return WorkflowNode(
        id="research_cycle",
        type=NodeType.plan_and_execute,
        label="Research Cycle",
        config=pe_config,
    )


def _build_synthesizer(config: OrchestrationConfig) -> WorkflowNode:
    """Build synthesizer node.

    Handles three synthesis modes:

    * ``simple`` (default) — plain report generation.
    * ``reclaim`` — citation-aware synthesis with claim extraction,
      citation grounding, and optional post-verification stages.

    The ``reclaim`` mode is activated when:
    1. ``config.synthesis_mode == "reclaim"``, **or**
    2. ``config.verify_sources`` is ``True`` (legacy toggle that implies
       citation-aware synthesis).

    When *reclaim* mode is active the node config includes extra keys that
    downstream agent wrappers use to enable citation pipeline stages.
    """
    synth_config: dict[str, Any] = {
        "subtype": "synthesizer",
        "model_tier": "complex",
        "input_keys": ["query", "plan"],
        "output_key": "report",
        "pool_inject": [
            {"pool": "observations", "threshold": 0},
            {"pool": "sources", "threshold": 0},
        ],
        "pool_tools": ["observations", "sources"],
        "max_tool_calls": 10,
    }

    # ------------------------------------------------------------------
    # Synthesis mode: reclaim (citation-aware) vs simple
    # ------------------------------------------------------------------
    synthesis_mode = config.synthesis_mode
    verify_sources = config.verify_sources

    # Citations are ALWAYS produced via a grounded synthesis mode (the cheap
    # "grounding-only" floor). ``verify_sources`` now controls ONLY the
    # expensive per-claim NLI verification overlay — NOT whether citations
    # exist:
    #   verify on  → grounding_mode=reclaim        (cite + NLI verify + disposition)
    #   verify off → grounding_mode=classical_lite (cite; skip NLI/correction/numeric)
    # Both grounded modes use the strict-cite prompt and parse ``[N]`` markers;
    # the framework synthesizer selects the prompt from ``grounding_mode``
    # (set on the node config below — the source of truth for cite-vs-verify).
    full_verify = synthesis_mode == "reclaim" or verify_sources

    # ``interleaved`` is the only generation strategy used here. The legacy
    # ``synthesis_mode="reclaim"`` alias was an invalid SynthesisMode that
    # silently fell back to interleaved; setting it explicitly drops that
    # warning with no behavior change.
    synth_schema: dict[str, Any] = {
        "synthesis_mode": "interleaved",
        "enable_citation_verification": True,
    }
    if full_verify:
        synth_config["grounding_mode"] = "reclaim"
        synth_schema["enable_isolated_verification"] = True
    else:
        synth_config["grounding_mode"] = "classical_lite"
        # Cheap grounding-only: generate + link + render citations, then skip
        # the expensive verification overlay (Stage 4 NLI / 5 correction /
        # 6 numeric) and Stage 8 disposition. Claims persist as resolvable-
        # but-unverified (a normal clickable citation, no verdict).
        synth_schema["enable_isolated_verification"] = False
        synth_schema["enable_citation_correction"] = False
        synth_schema["enable_numeric_qa_verification"] = False

    logger.debug(
        "SYNTHESIZER_GROUNDING_MODE grounding=%s verify_sources=%s",
        synth_config["grounding_mode"],
        verify_sources,
    )

    # ------------------------------------------------------------------
    # Report limits → framework synthesizer
    # ------------------------------------------------------------------
    # The framework's _get_reclaim_config() reads max_tokens and
    # target_word_count from output_schema.  Without these the pipeline
    # falls back to _RECLAIM_MAX_TOKENS — far too low for full reports.
    _valid_depths = {"light", "medium", "extended"}
    depth = config.research_depth if config.research_depth in _valid_depths else "medium"
    report_limits = get_report_limits(depth)
    synth_schema["max_tokens"] = report_limits.max_tokens
    synth_schema["target_word_count"] = report_limits.max_words

    # NLI verification concurrency and claim disposition from per-depth citation config
    citation_cfg = None
    try:
        depth_config = get_research_type_config(depth)
        citation_cfg = depth_config.citation_verification
        if citation_cfg and hasattr(citation_cfg, "max_concurrent_verifications"):
            synth_schema["max_concurrent_verifications"] = citation_cfg.max_concurrent_verifications
    except Exception:
        pass  # Fall back to framework default

    # Stage 8 claim disposition
    if citation_cfg:
        synth_schema["claim_disposition"] = citation_cfg.claim_disposition

    synth_config["output_schema"] = synth_schema

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------
    if config.output_format == "json":
        synth_config["output_format"] = "json"
        if config.output_schema:
            synth_config["output_model"] = config.output_schema

    # ------------------------------------------------------------------
    # Custom prompts
    # ------------------------------------------------------------------
    if config.structured_system_prompt:
        synth_config["system_prompt"] = config.structured_system_prompt
    if config.structured_user_prompt:
        synth_config["user_prompt_template"] = config.structured_user_prompt
    if config.system_instructions and "system_prompt" not in synth_config:
        synth_config["system_prompt"] = config.system_instructions

    # ------------------------------------------------------------------
    # Per-run report-style knobs (tone + output language). Prompts-over-knobs:
    # these reach the synthesizer's generation instructions via AgentNodeConfig.
    # An unrecognized tone name degrades to unset (None) rather than raising.
    # ------------------------------------------------------------------
    if config.tone:
        from databricks_deep_research.agents.config import Tone

        resolved_tone = Tone.from_name(config.tone)
        if resolved_tone is not None:
            synth_config["tone"] = resolved_tone
        else:
            logger.warning(
                "SYNTH_TONE_UNRECOGNIZED tone=%s (ignored; valid: %s)",
                config.tone,
                ", ".join(Tone.names()),
            )
    if config.output_language:
        synth_config["output_language"] = config.output_language

    return WorkflowNode(
        id="synthesizer",
        type=NodeType.agent,
        label="Report Synthesizer",
        config=synth_config,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_search_tools(
    config: OrchestrationConfig,
    available_tools: list[str] | None = None,
) -> list[str | dict[str, Any]]:
    """Build tool list based on source scope config.

    Returns a list of tool name strings (new format).  Legacy dict format
    is no longer emitted but is still accepted by the framework for
    backward compatibility.

    Args:
        config: Orchestration config with source_scope.
        available_tools: Tool names registered in the ToolRegistry.
            Enterprise tools (any name not in the web set) are included
            when source_scope allows enterprise access.
    """
    tools: list[str | dict[str, Any]] = []

    source_scope = config.source_scope
    web_allowed = source_scope in (None, "web_only", "all")
    enterprise_allowed = source_scope in (None, "enterprise_only", "all")

    if web_allowed:
        tools.append("web_search")
        tools.append("web_crawl")

    if enterprise_allowed and available_tools is not None:
        web_names = {"web_search", "web_crawl"}
        for name in available_tools:
            if name not in web_names:
                tools.append(name)

        enterprise_count = sum(1 for t in tools if t not in web_names)
        if enterprise_count == 0 and source_scope == "enterprise_only":
            logger.warning(
                "RESOLVE_SEARCH_TOOLS_NO_ENTERPRISE source_scope=%s "
                "available_tools=%s — no enterprise tools resolved",
                source_scope,
                available_tools,
            )

    logger.info(
        "RESOLVE_SEARCH_TOOLS source_scope=%s web_allowed=%s "
        "enterprise_allowed=%s tool_count=%d",
        source_scope,
        web_allowed,
        enterprise_allowed,
        len(tools),
    )

    return tools


def _depth_to_step_limits(depth: str) -> tuple[int, int]:
    """Map research depth to (min_steps, max_steps)."""
    limits = {
        "light": (1, 3),
        "medium": (2, 5),
        "extended": (3, 7),
        "auto": (2, 7),
    }
    return limits.get(depth, (2, 7))


def _build_pools(_config: OrchestrationConfig) -> list[dict[str, Any]]:
    """Build pool configuration."""
    return [
        {"name": "sources", "dedup_key": "url", "max_items": 200},
        {"name": "observations", "dedup_key": "content_hash", "max_items": 100},
        {"name": "discovery_sources", "dedup_key": "url", "max_items": 100},
    ]


__all__ = ["translate"]
