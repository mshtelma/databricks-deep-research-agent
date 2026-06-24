"""Display-metadata taxonomy for the Agent Designer inspector.

The Designer renders the *full* ``AgentNodeConfig`` JSON schema (see
``registry._agent_config_schema``); without grouping that is a flat wall of 46
fields. This module is the single source of truth that assigns each field to a
collapsible group + render order + (optional) widget/help, so the inspector can
present grouped, collapsible sections. It is decoration only — it never changes
validation or the persisted config.

A parity test (``tests/unit/agent_designer/test_field_groups.py``) asserts every
``AgentNodeConfig`` field is either grouped here or explicitly hidden, so a
newly-added framework knob can never silently fall out of the UI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldMeta:
    """Where/how one config field renders in the Designer inspector."""

    group: str
    order: int = 100
    # Overrides the JSON-schema-derived widget. Currently only ``"json"`` (a
    # textarea for free-form dict fields that have no ``properties`` to render).
    widget: str | None = None
    # Short inline description shown under the field (supplements the schema's).
    help: str | None = None


# Display order of groups in the inspector. Groups not listed sort last.
GROUP_ORDER: tuple[str, ...] = (
    "Basics",
    "Prompts",
    "Tools & Sources",
    "Output",
    "Context & Memory",
    "Execution",
    "Composition",
    "Data Flow",
    "Pools",
)

# Groups that start COLLAPSED in the inspector (advanced tuning).
ADVANCED_GROUPS: frozenset[str] = frozenset(
    {"Context & Memory", "Execution", "Composition", "Data Flow", "Pools"}
)

# Fields intentionally NOT shown in the config form:
#  - ``tools``: managed by the dedicated Tools tab (ToolsBindingForm), not here.
#  - ``output_model`` / ``extras``: internal (a live Pydantic model / free-form
#    bag set by the compiler), not user-editable.
HIDDEN_FIELDS: frozenset[str] = frozenset({"tools", "output_model", "extras"})


# Every ``AgentNodeConfig`` field maps to exactly one group (or HIDDEN_FIELDS).
FIELD_GROUPS: dict[str, FieldMeta] = {
    # -- Basics ------------------------------------------------------------
    "subtype": FieldMeta("Basics", 10, help="Agent role (researcher, planner, synthesizer, …)."),
    "model_tier": FieldMeta(
        "Basics", 20, help="LLM tier: simple (fast), analytical (balanced), complex (reasoning)."
    ),
    "model_family": FieldMeta("Basics", 30, help="Optional model-family override; blank = tier default."),
    "profile": FieldMeta(
        "Basics", 40, help="Execution profile; 'long_horizon' raises budgets for many-step work."
    ),
    "max_tool_calls": FieldMeta("Basics", 50, help="Max tool calls this agent may make (ReAct budget)."),
    # -- Prompts -----------------------------------------------------------
    "system_prompt": FieldMeta("Prompts", 10),
    "user_prompt_template": FieldMeta("Prompts", 20),
    # -- Tools & Sources ---------------------------------------------------
    "skills": FieldMeta(
        "Tools & Sources", 10, help="Attached skill packs (loaded on demand via read_skill)."
    ),
    "mcp_servers": FieldMeta(
        "Tools & Sources", 20, help="MCP servers this agent may call (declared at workflow level)."
    ),
    "allow_skill_scripts": FieldMeta(
        "Tools & Sources",
        30,
        help="Allow executing skill scripts. Also requires the global skills.allow_script_execution switch.",
    ),
    "grounding_mode": FieldMeta("Tools & Sources", 40),
    "hint_queries": FieldMeta(
        "Tools & Sources", 50, help="Seed search queries used to ground this agent before it plans."
    ),
    # -- Output ------------------------------------------------------------
    "tone": FieldMeta("Output", 10, help="Report writing tone."),
    "output_language": FieldMeta("Output", 20, help="Output language (blank = match the query)."),
    "synthesis_context": FieldMeta("Output", 30),
    # -- Context & Memory (advanced) --------------------------------------
    "conversation_budget": FieldMeta(
        "Context & Memory", 10, help="Approx. token budget for the running conversation."
    ),
    "max_result_chars": FieldMeta(
        "Context & Memory", 20, help="Per-tool-result character cap before compaction."
    ),
    "tool_output_offload": FieldMeta(
        "Context & Memory",
        30,
        help="Offload large tool outputs to the compute scratchpad. Needs a compute/code-action "
        "capable agent to dereference the handle, or data stays behind a preview.",
    ),
    "tool_output_budget": FieldMeta("Context & Memory", 40),
    "compaction_strategy": FieldMeta(
        "Context & Memory",
        50,
        help="How old tool results are compacted: truncate (clip) or mask (keep numeric/tabular lines).",
    ),
    "keep_intact_iterations": FieldMeta(
        "Context & Memory", 60, help="Recent tool-calling rounds kept uncompacted."
    ),
    "compaction_budget_chars": FieldMeta(
        "Context & Memory", 70, help="Char budget the evidence-rescue compactor targets (0 = off)."
    ),
    "evidence_rescue": FieldMeta(
        "Context & Memory",
        80,
        help="Compact lowest-value tool results first; never compact cited evidence.",
    ),
    "dedup_jaccard_threshold": FieldMeta("Context & Memory", 90),
    # -- Execution (advanced) ---------------------------------------------
    "action_mode": FieldMeta(
        "Execution",
        10,
        help="tools = classic tool calls; code/hybrid = MemEx code-action (compute closures + submit()).",
    ),
    "code_action_tools": FieldMeta(
        "Execution", 20, help="Tool names exposed as callable functions in code-action mode."
    ),
    "defer_tools": FieldMeta(
        "Execution",
        30,
        help="RAG-over-tools: defer full tool schemas until the model searches for them (large catalogs).",
    ),
    "defer_tool_threshold": FieldMeta(
        "Execution", 40, help="Catalog size above which tool schemas are deferred (0 = inert)."
    ),
    "max_retries": FieldMeta("Execution", 50),
    "per_tool_limits": FieldMeta(
        "Execution", 60, widget="json", help='Per-tool call-count caps, e.g. {"web_search": 5}.'
    ),
    "force_convergence": FieldMeta("Execution", 70),
    "convergence_rounds": FieldMeta("Execution", 80),
    # -- Composition (advanced) -------------------------------------------
    "spawnable_subagents": FieldMeta(
        "Composition",
        10,
        widget="json",
        help="Declared sub-agents this agent may spawn (governed spawn_agent).",
    ),
    "spawn_budget": FieldMeta("Composition", 20, help="Max spawn attempts (0 = spawning disabled)."),
    "max_concurrent_spawns": FieldMeta("Composition", 30),
    # -- Data Flow (advanced) ---------------------------------------------
    "input_keys": FieldMeta("Data Flow", 10, help="State keys this agent reads as input."),
    "output_key": FieldMeta("Data Flow", 20, help="State key this agent writes its result to."),
    "output_mode": FieldMeta("Data Flow", 30),
    "output_format": FieldMeta("Data Flow", 40),
    "output_schema": FieldMeta(
        "Data Flow", 50, widget="json", help="JSON schema for structured output (output_format=json)."
    ),
    # -- Pools (advanced) -------------------------------------------------
    "pool_writes": FieldMeta("Pools", 10),
    "pool_tools": FieldMeta("Pools", 20),
    "pool_inject": FieldMeta("Pools", 30),
}


def group_sort_key(group: str) -> tuple[int, str]:
    """Sort key for a group name: declared order first, then alphabetical."""
    try:
        return (GROUP_ORDER.index(group), "")
    except ValueError:
        return (len(GROUP_ORDER), group)
