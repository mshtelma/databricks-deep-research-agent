"""Workflow generation primitives for Agent Designer.

Designer-generated workflows must be executable and explainable as saved. This
module centralizes scaffold construction, fills builtin prompts at generation
time, and validates semantic data-flow invariants before the AST reaches the UI.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from databricks_deep_research.agents.config import SUBTYPE_DEFAULTS, AgentNodeConfig
from databricks_deep_research.templates.renderer import SafeTemplateRenderer
from databricks_deep_research.workflow.definition import NodeType
from databricks_deep_research.workflow.loader import load_workflow_from_dict

from deep_research.agent_designer.designer_architect import (
    compile_workflow_design_brief,
    format_workflow_design_brief,
)
from deep_research.agent_designer.designer_types import WorkflowDesignBrief

_RUNTIME_TEMPLATE_KEYS: frozenset[str] = frozenset(
    {
        "all_observations",
        "background",
        "completed_steps",
        "compute_namespace",
        "conversation_history",
        # Temporal anchor — auto-injected by the agent harness from
        # PromptTemporalContext.now() (or test fixtures fix the clock).
        # Every builtin system prompt now references {current_date} /
        # {current_timezone} via the shared TEMPORAL_ANCHOR_BLOCK.
        "current_date",
        "current_iso_datetime",
        "current_timezone",
        "fallback_discovery_sources",
        "file_context",
        "iteration",
        "max_steps",
        "max_words",
        "min_steps",
        "min_words",
        "observation",
        "page_contents",
        "plan_iterations",
        "plan_summary",
        "previous_observations",
        "reflector_feedback",
        "remaining_steps",
        "replan_budget",
        # Synthesizer revision block — auto-injected by the harness when
        # the workflow has both draft_report and a coverage_review with
        # decision='adjust' on the state log. Empty string otherwise.
        "revision_block_md",
        "research_depth",
        "search_results",
        "source_quality",
        "source_topics",
        "sources_count",
        "sources_list",
        "step_description",
        "step_prompt_guidance",
        "step_title",
        "step_type",
        "steps_completed",
        "steps_executed",
        "total_steps",
    }
)

_TEMPLATE_VAR_RE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")
_DESIGNER_LANE_TEMPLATE_ALLOWED_VARS: frozenset[str] = frozenset(
    {
        "query",
        "coordination",
        "current_step",
        "research_plan",
    }
) | _RUNTIME_TEMPLATE_KEYS


class WorkflowGenerationError(ValueError):
    """Raised when Designer generated a workflow with broken semantic contracts."""


def _coerce_unknown_template_variables_to_query(
    template: str,
    *,
    allowed_vars: frozenset[str],
) -> str:
    """Map Designer-authored placeholder aliases to the runtime query value.

    The saved workflow runtime has a small fixed state surface. Designers often
    author natural use-case aliases such as ``{company_or_ticker}``,
    ``{case_id}``, or ``{destination}``; those are semantically the user's
    request parameter, not separate state keys. Leaving them untouched makes the
    builder reject an otherwise good brief. Escaping them would leak raw
    placeholders to end users, so the executable default is to bind them to
    ``{query}``.
    """

    def replace(match: re.Match[str]) -> str:
        variable = match.group(1)
        if variable in allowed_vars:
            return match.group(0)
        return "{query}"

    return _TEMPLATE_VAR_RE.sub(replace, template)


def _default_prompts(subtype: str) -> tuple[str, str]:
    """Return builtin prompt defaults for an agent subtype."""
    if subtype == "coordinator":
        from databricks_deep_research.agents.prompts.coordinator import (
            COORDINATOR_SYSTEM_PROMPT,
            COORDINATOR_USER_PROMPT,
        )

        return COORDINATOR_SYSTEM_PROMPT, COORDINATOR_USER_PROMPT
    if subtype == "planner":
        from databricks_deep_research.agents.prompts.planner import (
            PLANNER_SYSTEM_PROMPT,
            PLANNER_USER_PROMPT,
        )

        return PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
    if subtype == "researcher":
        from databricks_deep_research.agents.prompts.researcher import (
            RESEARCHER_SYSTEM_PROMPT,
            RESEARCHER_USER_PROMPT,
        )

        return RESEARCHER_SYSTEM_PROMPT, RESEARCHER_USER_PROMPT
    if subtype in {"reflector", "evaluator"}:
        from databricks_deep_research.agents.prompts.reflector import (
            REFLECTOR_SYSTEM_PROMPT,
            REFLECTOR_USER_PROMPT,
        )

        return REFLECTOR_SYSTEM_PROMPT, REFLECTOR_USER_PROMPT
    if subtype == "synthesizer":
        from databricks_deep_research.agents.prompts.synthesizer import (
            SYNTHESIZER_SYSTEM_PROMPT,
            SYNTHESIZER_USER_PROMPT,
        )

        return SYNTHESIZER_SYSTEM_PROMPT, SYNTHESIZER_USER_PROMPT
    if subtype == "background":
        from databricks_deep_research.agents.prompts.background import (
            BACKGROUND_SYSTEM_PROMPT,
            BACKGROUND_USER_PROMPT,
        )

        return BACKGROUND_SYSTEM_PROMPT, BACKGROUND_USER_PROMPT
    return "", "{query}"


def _designer_synthesizer_system_prompt(system_prompt: str) -> str:
    """Return the builtin synthesizer prompt without fixed section-count caps.

    The framework's generic synthesizer prompt is optimized for ordinary chat
    research and says to use 2-3 main sections. Designer-generated workflows
    carry their own required outputs and coverage obligations, so a fixed
    section cap becomes a semantic conflict and can cause the critic to reject
    otherwise valid workflows.
    """
    replacements = {
        "- ## for main sections (2-3 max)": (
            "- ## for main sections required by this workflow; follow the "
            "workflow-specific output structure and do not impose a fixed "
            "section count"
        ),
        "- ## for 2-3 main sections": (
            "- ## for main sections required by this workflow; follow the "
            "workflow-specific output structure and do not impose a fixed "
            "section count"
        ),
    }
    cleaned = system_prompt
    for before, after in replacements.items():
        cleaned = cleaned.replace(before, after)
    return (
        cleaned
        + "\n\n## Designer Report Structure Contract\n\n"
        "The Designer workflow controls the required report headings, sections, "
        "deliverables, and coverage obligations. If any generic base prompt "
        "instruction conflicts with workflow-specific required outputs, follow "
        "the workflow-specific required outputs."
    )


def materialize_agent_config(config: dict[str, Any]) -> dict[str, Any]:
    """Apply subtype defaults and prompt templates to an agent config dict."""
    next_config = copy.deepcopy(config)
    subtype = str(next_config.get("subtype", "researcher"))
    defaults = copy.deepcopy(SUBTYPE_DEFAULTS.get(subtype, {}))
    for key, value in defaults.items():
        next_config.setdefault(key, value)

    system_prompt, user_prompt = _default_prompts(subtype)
    if not next_config.get("system_prompt") and system_prompt:
        next_config["system_prompt"] = system_prompt
    if not next_config.get("user_prompt_template") and user_prompt:
        next_config["user_prompt_template"] = user_prompt

    renderer = SafeTemplateRenderer()
    prompt_vars = renderer.extract_variables(
        str(next_config.get("system_prompt", ""))
    ) | renderer.extract_variables(str(next_config.get("user_prompt_template", "")))
    input_keys = list(next_config.get("input_keys") or [])
    for key in sorted(prompt_vars):
        if key != "query" and key not in input_keys:
            input_keys.append(key)
    if "query" in prompt_vars and "query" not in input_keys:
        input_keys.insert(0, "query")
    next_config["input_keys"] = input_keys

    AgentNodeConfig(**next_config)
    return next_config


def make_agent_node(
    *,
    node_id: str,
    label: str,
    subtype: str,
    output_key: str,
    input_keys: list[str] | None = None,
    model_tier: str | None = None,
    output_format: str | None = None,
    tools: list[str] | None = None,
    pool_writes: list[dict[str, Any]] | None = None,
    pool_inject: list[dict[str, Any]] | None = None,
    max_tool_calls: int | None = None,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an agent WorkflowNode as a JSON-compatible dict."""
    config: dict[str, Any] = {
        "subtype": subtype,
        "input_keys": input_keys or ["query"],
        "output_key": output_key,
    }
    if model_tier is not None:
        config["model_tier"] = model_tier
    if output_format is not None:
        config["output_format"] = output_format
    if tools is not None:
        config["tools"] = tools
    if pool_writes is not None:
        config["pool_writes"] = pool_writes
    if pool_inject is not None:
        config["pool_inject"] = pool_inject
    if max_tool_calls is not None:
        config["max_tool_calls"] = max_tool_calls
    if extra_config:
        config.update(copy.deepcopy(extra_config))

    return {
        "id": node_id,
        "type": NodeType.agent.value,
        "label": label,
        "config": materialize_agent_config(config),
        "children": [],
    }


def make_sequence(*, node_id: str, label: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a sequence WorkflowNode."""
    return {
        "id": node_id,
        "type": NodeType.sequence.value,
        "label": label,
        "config": {},
        "children": children,
    }


def make_parallel(*, node_id: str, label: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a parallel WorkflowNode.

    Children execute concurrently against the shared ``WorkflowState``. The
    engine's ``_exec_parallel`` honors per-child ``error_handling="continue"``
    semantics so a single failing child does not halt sibling children.
    Output flow piggybacks on shared state — typically via pool writes
    (children call ``pool_writes``) and a downstream consumer that reads
    via ``pool_inject``.
    """
    return {
        "id": node_id,
        "type": NodeType.parallel.value,
        "label": label,
        "config": {},
        "children": children,
    }


def make_conditional(
    *,
    node_id: str,
    label: str,
    conditions: list[dict[str, Any]],
    default_branch: int,
    children: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a conditional WorkflowNode."""
    return {
        "id": node_id,
        "type": NodeType.conditional.value,
        "label": label,
        "config": {
            "conditions": conditions,
            "default_branch": default_branch,
        },
        "children": children,
    }


def make_plan_and_execute(
    *,
    node_id: str,
    label: str,
    planner: dict[str, Any],
    body: dict[str, Any],
    evaluator: dict[str, Any] | None = None,
    items_path: str = "steps",
    item_state_key: str = "current_step",
    min_iterations: int = 1,
    max_iterations: int = 6,
    max_replan_cycles: int = 3,
    planner_guidance: str = "",
    synthesis_metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a plan-and-execute WorkflowNode."""
    config: dict[str, Any] = {
        "planner": materialize_agent_config(planner),
        "items_path": items_path,
        "item_state_key": item_state_key,
        "body": body,
        "min_iterations": min_iterations,
        "max_iterations": max_iterations,
        "max_replan_cycles": max_replan_cycles,
        "synthesis_metadata": synthesis_metadata or {},
    }
    if evaluator is not None:
        config["evaluator"] = materialize_agent_config(evaluator)
    if planner_guidance:
        config["planner_guidance"] = planner_guidance
    return {
        "id": node_id,
        "type": NodeType.plan_and_execute.value,
        "label": label,
        "config": config,
        "children": [],
    }


def make_tool_decl(
    *,
    name: str,
    kind: str,
    config: dict[str, Any] | None = None,
    description: str = "",
) -> dict[str, Any]:
    """Build a top-level workflow tool declaration."""
    return {
        "name": name,
        "kind": kind,
        "config": config or {},
        "description": description,
    }


def make_pool_decl(
    *,
    name: str,
    dedup_key: str | None = None,
    max_items: int = 100,
) -> dict[str, Any]:
    """Build a top-level workflow pool declaration."""
    pool: dict[str, Any] = {"name": name, "max_items": max_items}
    if dedup_key:
        pool["dedup_key"] = dedup_key
    return pool


def _bounded_intent(intent: str, *, max_length: int = 2000) -> str:
    """Return a compact single-source-of-truth designer intent for prompts."""
    normalized = " ".join(intent.strip().split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 15].rstrip() + " ...(truncated)"


def _bounded_multiline(value: str, *, max_length: int) -> str:
    """Return a length-bounded copy of ``value`` that PRESERVES newlines.

    Used for fields whose structural Markdown matters (the lane researcher's
    designer-authored ``user_prompt_template`` carries headings and numbered
    sub-question blocks that ``_bounded_intent``'s whitespace-collapse would
    destroy). Trailing whitespace per line is preserved; only the overall
    head/tail is stripped.
    """
    cleaned = str(value).strip()
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 15].rstrip() + " ...(truncated)"


def _designer_goal_block(
    intent: str,
    *,
    role: str,
    design_brief: WorkflowDesignBrief | None = None,
) -> str:
    """Prompt appendix that turns a Designer intent into executable behavior."""
    goal = _bounded_intent(intent)
    brief_text = format_workflow_design_brief(design_brief) if design_brief else ""
    if not goal:
        return ""
    role_guidance = {
        "coordinator": (
            "Classify terse user inputs as parameters for this designed task. "
            "Infer only the parameter role implied by the workflow goal and avoid "
            "substituting a generic interpretation."
        ),
        "planner": (
            "Create steps that satisfy the designed task and the design brief, "
            "not a generic overview. For terse user inputs, infer the missing "
            "parameter from the goal and make the plan goal-specific."
        ),
        "researcher": (
            "Execute each step through the lens of the designed task. Search for "
            "evidence that supports the goal-specific analysis, including any "
            "constraints, risks, tradeoffs, or decision criteria requested by the goal."
        ),
        "reflector": (
            "Evaluate coverage against the designed task. Do not mark complete while "
            "goal-specific outputs are missing, even if a generic research summary is covered."
        ),
        "synthesizer": (
            "Write the final answer for the designed task. Follow the workflow's "
            "requested output structure, analysis criteria, and tone. Do not omit "
            "requested domain-specific sections or add unrelated ones."
        ),
    }.get(role, "Honor the designed task in this workflow.")
    design_section = f"\n\n## Workflow Design Brief\n{brief_text}" if brief_text else ""
    return (
        "\n\n## Designer Goal\n"
        "This workflow was explicitly designed for the following task:\n"
        f"{goal}\n\n"
        "It was produced with an architect-and-critic design pass; treat the "
        "design brief below as required runtime behavior, not decoration."
        f"{design_section}\n\n"
        f"{role_guidance}"
    )


def _with_designer_goal(
    system_prompt: str,
    intent: str,
    *,
    role: str,
    design_brief: WorkflowDesignBrief | None = None,
) -> str:
    """Append the Designer goal to a built-in system prompt."""
    return f"{system_prompt}{_designer_goal_block(intent, role=role, design_brief=design_brief)}"


def _designer_planner_guidance(
    intent: str,
    design_brief: WorkflowDesignBrief | None = None,
) -> str:
    """Runtime planner guidance injected into {step_prompt_guidance}."""
    goal = _bounded_intent(intent)
    if not goal:
        return ""
    brief_text = format_workflow_design_brief(design_brief) if design_brief else ""
    brief_section = f"\n\nArchitect/critic design brief:\n{brief_text}" if brief_text else ""
    lane_specs = _lane_specs(design_brief, intent=goal) if design_brief else []
    if lane_specs:
        lane_lines = "\n".join(
            f"- {spec['id']}: {spec['description']}" for spec in lane_specs
        )
        lane_section = (
            "\n\nPlanner lane contract:\n"
            "Every step object MUST include a `lane` field. Use exactly one of "
            "the lane ids below so the workflow routes the step to the matching "
            "domain researcher branch. Use `cross_lane` only for final comparison, "
            "thesis, or synthesis steps that span multiple lanes.\n"
            f"{lane_lines}"
        )
    else:
        lane_section = ""
    return (
        "Designer task: "
        f"{goal}\n"
        "Plan for this designed task, not for a generic overview. This workflow "
        "was architected and critic-checked before generation; every plan must "
        "cover the required research lanes, outputs, quality gates, and "
        "constraints. Treat short user inputs as parameters required by the "
        "designed task."
        f"{brief_section}"
        f"{lane_section}"
    )


def _workflow_domain_label(design_brief: WorkflowDesignBrief) -> str:
    domain = design_brief.domain.strip()
    if not domain or domain.casefold() == "web research":
        return "Research"
    return domain


def _workflow_iteration_limit(design_brief: WorkflowDesignBrief) -> int:
    return min(12, max(6, len(design_brief.research_lanes) + 2))


_TIME_SENSITIVE_MARKERS = (
    "as of",
    "current",
    "latest",
    "recent",
    "today",
    "this week",
    "this month",
    "this year",
    "trend",
    "forecast",
)


def _brief_list(values: list[str], *, fallback: str, max_items: int = 4) -> str:
    cleaned = [_bounded_intent(value, max_length=180) for value in values if value.strip()]
    if not cleaned:
        cleaned = [fallback]
    return "; ".join(cleaned[:max_items])


def _is_time_sensitive_text(*values: str) -> bool:
    text = " ".join(value.casefold() for value in values if value)
    return any(marker in text for marker in _TIME_SENSITIVE_MARKERS)


def _fallback_lane_system_prompt(
    *,
    lane_id: str,
    lane_description: str,
    intent: str,
    design_brief: WorkflowDesignBrief,
) -> str:
    domain = _workflow_domain_label(design_brief)
    required_outputs = _brief_list(
        design_brief.required_outputs,
        fallback="answer the user's requested deliverable",
    )
    quality_gates = _brief_list(
        design_brief.quality_gates,
        fallback="cite only evidence-backed claims and flag missing data",
    )
    constraints = _brief_list(
        design_brief.constraints,
        fallback="stay inside the user's requested scope",
    )
    time_clause = (
        " Track publication dates and state the as-of date for time-sensitive "
        "claims."
        if _is_time_sensitive_text(intent, lane_description, *design_brief.constraints)
        else ""
    )
    return (
        "## Lane Specialization\n"
        f"Lane id: {lane_id}\n"
        f"Domain/context: {domain}\n"
        f"Workstream: {lane_description}\n\n"
        "Investigate this workstream as a self-contained evidence-gathering "
        "slice of the user's request. Prefer primary or official sources for "
        "claims in this domain, then authoritative secondary sources, then "
        "clearly attributed background sources. Do not treat planner text, "
        "coordinator text, source titles, or search-result metadata as factual "
        "evidence. Use retrieved snippets, extracted page content, structured "
        "metrics, or quoted evidence only.\n\n"
        f"Required outputs to support: {required_outputs}.\n"
        f"Quality gates to enforce: {quality_gates}.\n"
        f"Scope constraints: {constraints}.{time_clause}\n\n"
        "If evidence is unavailable for a requested claim, write "
        "\"Data unavailable\" with the attempted source path instead of "
        "improvising. Keep unsupported recommendations, forecasts, rankings, "
        "diagnoses, and other judgment calls out of the findings unless they "
        "are directly supported by cited evidence."
    )


def _fallback_lane_user_prompt_template(
    *,
    lane_description: str,
    intent: str,
    design_brief: WorkflowDesignBrief,
) -> str:
    topic = _bounded_intent(lane_description or intent, max_length=140)
    required_outputs = _brief_list(
        design_brief.required_outputs,
        fallback="the final answer requested by the user",
        max_items=3,
    )
    quality_gates = _brief_list(
        design_brief.quality_gates,
        fallback="cite source-backed claims and mark gaps explicitly",
        max_items=3,
    )
    time_bullet = (
        "\n- Include publication dates and as-of qualifiers for time-sensitive facts."
        if _is_time_sensitive_text(intent, lane_description, *design_brief.constraints)
        else ""
    )
    return (
        "## Investigation Brief\n\n"
        "You are investigating: **{query}**\n\n"
        f"Lane focus: {topic}\n\n"
        "### Sub-questions you MUST address (in this order)\n"
        f"1. What primary or authoritative sources directly address {topic} "
        "for the user's requested scope?\n"
        f"2. What concrete facts, dates, metrics, quotes, or structured data "
        f"from those sources support {topic}?\n"
        f"3. What relevant constraints, exceptions, tradeoffs, or risks change "
        f"the interpretation of {topic}?\n"
        f"4. Where do sources agree, conflict, or leave gaps about {topic}?\n"
        f"5. Which findings from {topic} are strong enough to support "
        "the final deliverable?\n\n"
        "### Required output structure\n"
        f"- **Evidence-backed findings**: facts that support {required_outputs}.\n"
        f"- **Coverage and conflicts**: source agreement, disagreement, and gaps.\n"
        f"- **Unsupported items**: claims blocked by {quality_gates}.\n\n"
        "### Search strategy\n"
        "- Run focused searches for each sub-question; refine with source names, "
        "official documents, or exact phrases found in promising results.\n"
        "- Crawl or retrieve source text before relying on a result; titles and "
        "metadata alone are not citeable evidence."
        f"{time_bullet}\n\n"
        "### Definition of done\n"
        "Each sub-question has a concise answer with citeable source text, OR "
        "is marked \"Data unavailable\" -- DO NOT improvise."
    )


def _lane_id(index: int) -> str:
    return f"lane_{index}"


def _lane_label(lane: str, index: int) -> str:
    cleaned = lane.rstrip(".")
    if len(cleaned) > 56:
        cleaned = cleaned[:53].rstrip() + "..."
    return f"Lane {index}: {cleaned}"


_SPECIALIZED_LANE_PREAMBLE = (
    "You are a research agent in a multi-agent workflow. Use the available "
    "tools to gather evidence for the lane workstream described below. "
    "Return JSON matching the output contract at the end of this prompt."
)


_INVESTIGATION_SCOPE_BLOCK = (
    "## Investigation Scope (from Coordinator)\n\n"
    "The Coordinator has resolved the user's query into structured scope. "
    "Read the ``extracted_scope`` field of the JSON below for canonical "
    "entities (preferred over informal references), time_window, and "
    "comparable peers. USE these resolved entities directly in your "
    "searches — do NOT spend tool calls re-extracting them from the raw "
    "query. If ``extracted_scope`` is null or absent, derive scope from "
    "the user's query as you would otherwise.\n\n"
    "{coordination}"
)


def _with_lane_user_prompt_contract(
    *,
    description: str,
    designer_template: str,
) -> str:
    """Wrap a Designer-authored lane template with the runtime prompt contract.

    The wrapper is structural, not domain-specific: semantic content still
    comes from the Designer's lane description and template, while the builder
    guarantees the prompt has the headings and runtime anchors that lane
    researchers need to execute without a second LLM repair pass.
    """
    template = _coerce_unknown_template_variables_to_query(
        designer_template.strip(),
        allowed_vars=_DESIGNER_LANE_TEMPLATE_ALLOWED_VARS,
    )
    if not template:
        return ""
    lane_focus = _bounded_intent(description, max_length=420)
    contract = (
        "## Investigation Brief\n\n"
        "You are investigating: **{query}**\n\n"
        f"Lane focus: {lane_focus}\n\n"
        "### Sub-questions you MUST address (in this order)\n"
        f"1. What are the most decision-relevant facts for this lane focus: {lane_focus}?\n"
        "2. Which current evidence supports or contradicts those facts?\n"
        "3. What metrics, events, entities, or comparisons materially change the interpretation?\n"
        "4. What uncertainties, data gaps, or conflicting signals remain?\n"
        "5. What bottom-line implications should the final report carry forward?\n\n"
        "### Required output structure\n"
        "- **Evidence summary**: cite the strongest findings and source context.\n"
        "- **Analysis and implications**: explain why the evidence matters for the user goal.\n"
        "- **Unknowns and caveats**: mark missing, stale, or conflicting evidence explicitly.\n\n"
        "### Search strategy\n"
        "- Start with queries that combine {query} with the lane focus terms above.\n"
        "- Prefer primary or high-authority sources, then refine searches around gaps.\n\n"
        "### Definition of done\n"
        "Each sub-question has a concise answer with citeable source text, OR "
        "is marked \"Data unavailable\" -- DO NOT improvise.\n\n"
        "### Designer-authored lane brief\n"
        f"{template}"
    )
    return _bounded_multiline(contract, max_length=4000)


def _assemble_lane_system_prompt(
    *,
    base_researcher_prompt: str,
    spec: dict[str, str],
    include_scope_block: bool = False,
) -> str:
    """Build a lane researcher's ``system_prompt``.

    Two paths:

    1. **Specialized path** — when ``spec['system_prompt']`` is non-empty
       (the Designer LLM populated ``research_lanes[].system_prompt`` at
       propose_workflow time, OR an ``update_block`` call patched this
       agent's prompt): the lane's system_prompt is DOMINATED by
       task-specific content. The generic researcher methodology
       (``RESEARCHER_DEFAULT_METHOD``) is REPLACED with a minimal preamble
       so >70% of the prompt is the LLM's task-specific specialization.
       The output contract (``RESEARCHER_OUTPUT_CONTRACT``) is always
       appended because downstream observation parsers depend on it.

    2. **Legacy path** — when ``spec['system_prompt']`` is empty (legacy
       ``list[str]`` briefs, or lanes the LLM did not specialize): behavior
       is BYTE-EQUAL to the prior implementation: the full base researcher
       prompt + the lane-focus footer.

    The split between methodology and output contract lives in
    ``researcher.py`` (``RESEARCHER_DEFAULT_METHOD`` + ``RESEARCHER_OUTPUT_CONTRACT``).

    When ``include_scope_block=True`` (parallel_lanes topology), a
    ``## Investigation Scope`` section is inserted referencing
    ``{coordination}``. The SafeTemplateRenderer fills this at runtime from
    ``state.get("coordination")``. The plan_and_execute path leaves this
    off because its lane researchers do not declare ``coordination`` in
    ``input_keys``.
    """
    specialization = (spec.get("system_prompt") or "").strip()
    focus_instruction = (
        "Run this static lane once. Keep the search strategy, evidence "
        "extraction, and findings focused on this workstream."
        if include_scope_block
        else "When current_step selects this lane, keep the search strategy, "
        "evidence extraction, and findings focused on this workstream."
    )
    if specialization:
        from databricks_deep_research.agents.prompts.researcher import (
            RESEARCHER_OUTPUT_CONTRACT,
        )

        sections: list[str] = [
            _SPECIALIZED_LANE_PREAMBLE,
            "",
            specialization,
            "",
        ]
        if include_scope_block:
            sections.extend([_INVESTIGATION_SCOPE_BLOCK, ""])
        sections.extend(
            [
                "## Required Lane Focus",
                f"Lane id: {spec['id']}",
                f"Lane workstream: {spec['description']}",
                "",
                focus_instruction,
                "",
                RESEARCHER_OUTPUT_CONTRACT,
            ]
        )
        return "\n".join(sections)
    # Legacy path: byte-equal to the prior implementation when no scope
    # block is requested. With scope, the block is inserted between the
    # base prompt and the lane-focus footer.
    sections = [base_researcher_prompt, ""]
    if include_scope_block:
        sections.extend([_INVESTIGATION_SCOPE_BLOCK, ""])
    sections.extend(
        [
            "## Required Lane Focus",
            f"Lane id: {spec['id']}",
            f"Lane workstream: {spec['description']}",
            "",
            focus_instruction,
        ]
    )
    return "\n".join(sections)


def _lane_extra_config(
    *,
    system_prompt: str,
    spec: dict[str, str] | None,
) -> dict[str, Any]:
    """Assemble the ``extra_config`` payload for a lane / single-agent researcher.

    Always includes the assembled ``system_prompt``. Adds
    ``user_prompt_template`` when the spec carries a non-empty
    designer-authored brief; the generic RESEARCHER_USER_PROMPT default is
    preserved (via ``materialize_agent_config``'s subtype fallback) when the
    spec is None or its template is empty. This is the single place where
    designer-authored per-lane user prompts enter the generated workflow.
    """
    extra: dict[str, Any] = {"system_prompt": system_prompt}
    if spec is not None:
        template = (spec.get("user_prompt_template") or "").strip()
        if template:
            extra["user_prompt_template"] = template
    return extra


def _lane_specs(
    design_brief: WorkflowDesignBrief,
    *,
    intent: str = "",
) -> list[dict[str, str]]:
    """Build builder-side lane spec dicts from the brief's LaneSpec list.

    Each entry includes the LLM-supplied specialized ``system_prompt`` exactly
    as supplied. When non-empty, the lane researcher's prompt assembly injects a
    ``## Lane Specialization`` block carrying this content — that is the single
    place where task-specific researcher guidance enters the generated workflow.

    The ``user_prompt_template`` field carries the LLM-supplied per-lane
    researcher user prompt (also possibly empty). When non-empty, the lane
    researcher's agent config sets ``user_prompt_template`` directly,
    replacing the generic RESEARCHER_USER_PROMPT default. The contract for
    this string is enforced upstream by the designer/validator (5 concrete
    sub-questions, 3 output sections, search strategy, unknowns handling).

    Do not synthesize prompt content here for normal lanes. If the designer LLM
    supplied only a lane description, the semantic gate must see that gap and
    send the LLM back to author the use-case-specific prompts.
    """
    del intent  # Retained for compatibility with existing call sites.
    specs: list[dict[str, str]] = []
    for index, lane in enumerate(design_brief.research_lanes, start=1):
        cleaned_description = _bounded_intent(lane.description, max_length=280)
        if not cleaned_description:
            continue
        lane_id = _lane_id(index)
        system_prompt = _bounded_intent(lane.system_prompt, max_length=3000)
        user_prompt_template = _with_lane_user_prompt_contract(
            description=cleaned_description,
            designer_template=_bounded_multiline(
                lane.user_prompt_template, max_length=2600
            ),
        )
        specs.append(
            {
                "id": lane_id,
                "label": _lane_label(cleaned_description, index),
                "description": cleaned_description,
                "system_prompt": system_prompt,
                "user_prompt_template": user_prompt_template,
            }
        )
    return specs


def build_web_research_workflow(
    intent: str,
    name: str,
    design_brief: WorkflowDesignBrief | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Designer research workflow.

    Dispatches on ``compiled_brief.topology``:

    * ``parallel_lanes`` (DEFAULT) — coordinator → parallel lane researchers →
      synthesizer. Each lane runs concurrently with its specialized prompt;
      output flows through shared pools. No planner, no router, no
      cross-lane fallback. The right shape for multi-aspect research.

    * ``plan_and_execute`` — coordinator → plan_and_execute (planner +
      router-over-lanes + reflector loop, with evaluator + replan) →
      synthesizer. Preserved verbatim for back-compat with existing saved
      agents and for the rare case that genuinely needs reflection-driven
      re-planning.

    * ``single_agent`` — coordinator → one specialized agent. Right for
      short factual questions.

    The default is ``parallel_lanes``; legacy briefs without a topology
    field coerce to it via :class:`WorkflowDesignBrief`'s validator.
    """
    compiled_brief = compile_workflow_design_brief(intent, design_brief)
    topology = compiled_brief.topology
    if topology == "plan_and_execute":
        return _build_plan_and_execute_workflow(intent, name, compiled_brief)
    if topology == "single_agent":
        return _build_single_agent_workflow(intent, name, compiled_brief)
    # Default: parallel_lanes
    return _build_parallel_lanes_workflow(intent, name, compiled_brief)


def _require_authored_lane_specs(
    lane_specs: list[dict[str, str]],
    *,
    topology: str,
) -> None:
    """Reject Designer-supplied semantic-empty workflows early.

    Legacy callers without a design_brief are still compiled through the YAML
    compatibility profile before this point. If lane_specs is empty here, the
    Designer supplied a brief but failed to author the lane shape required by
    the chosen topology. Do not invent generic semantic lanes in the builder.
    """
    if lane_specs:
        return
    raise ValueError(
        f"{topology} requires at least one Designer-authored research_lanes "
        "entry with a concrete LaneSpec. Add lane descriptions plus "
        "system_prompt and user_prompt_template instead of relying on a "
        "generic fallback."
    )


def _build_plan_and_execute_workflow(
    intent: str,
    name: str,
    compiled_brief: WorkflowDesignBrief,
) -> dict[str, Any]:
    """Plan-and-execute topology.

    Preserves the planner/router/evaluator shape for workflows that need
    sequential decomposition, while applying the same evidence contract and
    grounding defaults as the parallel topology.
    """
    domain_label = _workflow_domain_label(compiled_brief)
    max_iterations = _workflow_iteration_limit(compiled_brief)
    coordinator_system, _ = _default_prompts("coordinator")
    planner_system, _ = _default_prompts("planner")
    researcher_system, _ = _default_prompts("researcher")
    reflector_system, _ = _default_prompts("reflector")
    synthesizer_system, _ = _default_prompts("synthesizer")
    synthesizer_system = _designer_synthesizer_system_prompt(synthesizer_system)
    designer_goal = _bounded_intent(intent)
    lane_specs = _lane_specs(compiled_brief, intent=intent)
    _require_authored_lane_specs(lane_specs, topology="plan_and_execute")
    tools = [
        make_tool_decl(
            name="web_search",
            kind="web_search",
            config={"max_results": 10},
            description="Search the public web for relevant sources.",
        ),
        make_tool_decl(
            name="web_crawl",
            kind="web_crawl",
            description=(
                "Fetch and extract content from selected web pages. Requires "
                "a url_index from a prior web_search result in this workflow "
                "run; call web_search first when no valid index is available."
            ),
        ),
    ]
    coordinator = make_agent_node(
        node_id="coordinator",
        label="Coordinator",
        subtype="coordinator",
        input_keys=["query"],
        output_key="coordination",
        model_tier="simple",
        output_format="json",
        max_tool_calls=0,
        extra_config={
            "system_prompt": _with_designer_goal(
                coordinator_system,
                intent,
                role="coordinator",
                design_brief=compiled_brief,
            ),
        },
    )
    base_researcher_prompt = _with_designer_goal(
        researcher_system,
        intent,
        role="researcher",
        design_brief=compiled_brief,
    )
    lane_researchers = [
        make_agent_node(
            node_id=f"{spec['id']}-researcher",
            label=spec["label"],
            subtype="researcher",
            input_keys=["query", "current_step", "research_plan"],
            output_key="findings",
            model_tier="analytical",
            output_format="json",
            tools=["web_search", "web_crawl"],
            pool_writes=[
                {"pool": "observations", "extract": "findings"},
                {"pool": "sources", "extract": "sources"},
            ],
            max_tool_calls=8,
            extra_config=_lane_extra_config(
                system_prompt=_assemble_lane_system_prompt(
                    base_researcher_prompt=base_researcher_prompt,
                    spec=spec,
                ),
                spec=spec,
            ),
        )
        for spec in lane_specs
    ]
    cross_lane_spec = {
        "id": "cross_lane",
        "label": f"{domain_label} Cross-Lane Researcher",
        "description": (
            "Cross-lane evidence gathering for steps that compare, reconcile, "
            "or synthesize multiple workstreams."
        ),
        "system_prompt": _fallback_lane_system_prompt(
            lane_id="cross_lane",
            lane_description=(
                "Cross-lane evidence gathering for steps that compare, reconcile, "
                "or synthesize multiple workstreams."
            ),
            intent=intent,
            design_brief=compiled_brief,
        ),
        "user_prompt_template": _fallback_lane_user_prompt_template(
            lane_description=(
                "Cross-lane evidence gathering for steps that compare, reconcile, "
                "or synthesize multiple workstreams."
            ),
            intent=intent,
            design_brief=compiled_brief,
        ),
    }
    fallback_researcher = make_agent_node(
        node_id="cross-lane-researcher",
        label=f"{domain_label} Cross-Lane Researcher",
        subtype="researcher",
        input_keys=["query", "current_step", "research_plan"],
        output_key="findings",
        model_tier="analytical",
        output_format="json",
        tools=["web_search", "web_crawl"],
        pool_writes=[
            {"pool": "observations", "extract": "findings"},
            {"pool": "sources", "extract": "sources"},
        ],
        max_tool_calls=8,
        extra_config=_lane_extra_config(
            system_prompt=_assemble_lane_system_prompt(
                base_researcher_prompt=base_researcher_prompt,
                spec=cross_lane_spec,
            ),
            spec=cross_lane_spec,
        ),
    )
    lane_router = make_conditional(
        node_id="research-lane-router",
        label=f"{domain_label} Lane Router",
        conditions=[
            {
                "type": "state",
                "key": "current_step.lane",
                "operator": "eq",
                "value": spec["id"],
            }
            for spec in lane_specs
        ],
        default_branch=len(lane_researchers),
        children=[*lane_researchers, fallback_researcher],
    )
    reflector = make_agent_node(
        node_id="reflector",
        label=f"{domain_label} Critic",
        subtype="reflector",
        input_keys=["query", "current_step", "findings", "research_plan"],
        output_key="reflection",
        model_tier="analytical",
        output_format="json",
        pool_inject=[{"pool": "observations", "threshold": 0}],
        max_tool_calls=0,
        extra_config={
            "system_prompt": _with_designer_goal(
                _reflector_workflow_directive(compiled_brief)
                + "\n\n"
                + reflector_system,
                intent,
                role="reflector",
                design_brief=compiled_brief,
            ),
        },
    )
    body = make_sequence(
        node_id="research-body",
        label=f"{domain_label} Research Body",
        children=[lane_router, reflector],
    )
    plan_and_execute = make_plan_and_execute(
        node_id="plan-and-execute",
        label=f"Plan & Execute {domain_label}",
        planner={
            "subtype": "planner",
            "model_tier": "analytical",
            "input_keys": ["query", "coordination"],
            "output_key": "research_plan",
            "output_format": "json",
            "system_prompt": _with_designer_goal(
                planner_system,
                intent,
                role="planner",
                design_brief=compiled_brief,
            ),
        },
        body=body,
        evaluator={
            "subtype": "reflector",
            "model_tier": "analytical",
            "input_keys": ["query", "current_step", "findings", "research_plan"],
            "output_key": "evaluation",
            "output_format": "json",
            "pool_inject": [{"pool": "observations", "threshold": 0}],
            "max_tool_calls": 0,
            "system_prompt": _with_designer_goal(
                _reflector_workflow_directive(compiled_brief)
                + "\n\n"
                + reflector_system,
                intent,
                role="reflector",
                design_brief=compiled_brief,
            ),
        },
        items_path="steps",
        item_state_key="current_step",
        min_iterations=1,
        max_iterations=max_iterations,
        max_replan_cycles=3,
        planner_guidance=_designer_planner_guidance(intent, compiled_brief),
        synthesis_metadata={
            "research_depth": "medium",
            "min_words": "400",
            "max_words": "1200",
            "designer_goal": designer_goal,
            "designer_domain": compiled_brief.domain,
            "designer_research_lanes": "\n".join(
                lane.description for lane in compiled_brief.research_lanes
            ),
            "designer_lane_ids": "\n".join(
                f"{spec['id']}: {spec['description']}" for spec in lane_specs
            ),
            "designer_required_outputs": "\n".join(compiled_brief.required_outputs),
            "designer_quality_gates": "\n".join(compiled_brief.quality_gates),
        },
    )
    synthesizer = make_agent_node(
        node_id="synthesizer",
        label=f"{domain_label} Report Synthesizer",
        subtype="synthesizer",
        input_keys=["query", "research_plan", "findings", "reflection"],
        output_key="report",
        model_tier="complex",
        output_format="markdown",
        pool_inject=[
            {"pool": "observations", "threshold": 0},
            {"pool": "sources", "threshold": 0},
        ],
        max_tool_calls=0,
        extra_config={
            "system_prompt": _with_designer_goal(
                _plan_execute_synthesizer_directive(lane_specs)
                + "\n\n"
                + synthesizer_system,
                intent,
                role="synthesizer",
                design_brief=compiled_brief,
            ),
            "grounding_mode": compiled_brief.grounding_mode,
            "output_schema": _grounded_synthesizer_output_schema(compiled_brief),
        },
    )
    workflow = {
        "id": "designer-draft",
        "name": compiled_brief.workflow_name or name,
        "description": intent,
        "version": 1,
        "tools": tools,
        "pools": [
            make_pool_decl(name="sources", dedup_key="url", max_items=100),
            make_pool_decl(name="observations", dedup_key="content_hash", max_items=50),
        ],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["report"],
        "token_budget": 0,
        "timeout_seconds": 1800,
        "root": make_sequence(
            node_id="main",
            label=f"{domain_label} Pipeline",
            children=[coordinator, plan_and_execute, synthesizer],
        ),
    }
    validate_generated_workflow(workflow)
    return workflow


def _synthesizer_lane_coverage_directive(lane_specs: list[dict[str, str]]) -> str:
    """Build the lane-coverage directive prepended to the synthesizer prompt.

    Without this, the synthesizer happily synthesizes a section per lane even
    when a lane's researcher produced zero observations (e.g., when Brave
    returned 429 for every search) — confabulating figures, dates, and
    citations to fill the gap. The directive instructs the LLM to explicitly
    surface lane-level coverage gaps and refuse to invent content for empty
    lanes.

    Domain-agnostic: lists lanes by ID and description only; carries no
    use-case-specific guidance.
    """
    if not lane_specs:
        return ""
    lane_list = "\n".join(
        f"- {spec['id']}: {spec.get('description', '').strip() or '(no description)'}"
        for spec in lane_specs
    )
    return (
        "## Lane Reporting Status\n\n"
        "This workflow uses a parallel_lanes topology. Each lane researcher "
        "writes its findings to the shared observations and sources pools "
        "with its lane id as a prefix. If any lane produced NO observations "
        "(its prefix is absent from the observations pool), you MUST:\n\n"
        "1. Explicitly note 'Data unavailable for this lane' in the "
        "corresponding section of your report.\n"
        "2. NEVER invent figures, dates, quotations, recommendations, "
        "forecasts, rankings, or other unsupported judgment calls to fill "
        "that gap.\n"
        "3. NEVER cite a URL that did not appear in the sources pool.\n"
        "4. NEVER emit numerical claims (revenue, percentages, market shares, "
        "specific dates) without a direct supporting observation in the pool.\n"
        "5. When two observations conflict, surface the contradiction rather "
        "than picking one silently.\n\n"
        "Lanes in this workflow:\n"
        f"{lane_list}\n"
    )


def _final_coverage_reflector_directive(
    compiled_brief: WorkflowDesignBrief,
    lane_specs: list[dict[str, str]],
) -> str:
    coverage_obligations = _combined_coverage_obligations(compiled_brief, lane_specs)
    quality_gates = _brief_list(
        compiled_brief.quality_gates,
        fallback="evidence-backed claims, explicit missing-data notes, and no invented facts",
    )
    lane_list = "\n".join(
        f"- {spec['id']}: {spec.get('description', '').strip() or '(no description)'}"
        for spec in lane_specs
    )
    lane_section = f"\n\nStatic lanes to audit:\n{lane_list}" if lane_list else ""
    return (
        "## Final Coverage Review Contract\n\n"
        "You are the final coverage reflector for a static parallel-lanes "
        "workflow. Review the draft report against the shared observations "
        "and sources pools before the final answer is produced.\n\n"
        f"Required coverage obligations: {coverage_obligations}.\n"
        f"Quality gates: {quality_gates}.\n"
        "The coverage obligations above are combined from the user's requested "
        "outputs and every static lane. They are exhaustive for coverage: "
        "do not let a shorter Designer-authored required_outputs list narrow "
        "or replace the lane coverage obligations.\n"
        "Return the standard reflector JSON. Use decision='complete' only "
        "when the draft report covers every required output and every material "
        "factual claim is backed by usable observations/sources. Use "
        "decision='adjust' when sections are missing, evidence is partial, "
        "claims overreach the evidence, or a lane produced no usable data. "
        "Put concrete repair instructions in suggested_changes; the final "
        "synthesizer will use them to revise the report."
        f"{lane_section}"
    )


def _combined_coverage_obligations(
    compiled_brief: WorkflowDesignBrief,
    lane_specs: list[dict[str, str]],
) -> str:
    """Return a domain-neutral coverage list that cannot drop static lanes."""
    obligations: list[str] = []

    for item in compiled_brief.required_outputs:
        cleaned = str(item).strip()
        if cleaned:
            obligations.append(cleaned)

    for spec in lane_specs:
        description = str(spec.get("description") or "").strip()
        if description:
            obligations.append(f"Lane coverage - {description}")

    if not obligations:
        return "the user's requested deliverable"

    deduped: list[str] = []
    seen: set[str] = set()
    for obligation in obligations:
        key = " ".join(obligation.casefold().split())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(obligation)
    return "; ".join(deduped)


def _final_report_repair_directive(
    compiled_brief: WorkflowDesignBrief,
    lane_specs: list[dict[str, str]],
) -> str:
    coverage_obligations = _combined_coverage_obligations(compiled_brief, lane_specs)
    lane_list = "\n".join(
        f"- {spec['id']}: {spec.get('description', '').strip() or '(no description)'}"
        for spec in lane_specs
    )
    lane_section = f"\n\nStatic lanes:\n{lane_list}" if lane_list else ""
    return (
        "## Final Report Repair Contract\n\n"
        "You receive a draft_report plus coverage_review JSON from the final "
        "coverage reflector. Produce the user-facing final report only. Do not "
        "include reviewer JSON, internal workflow commentary, or planning text.\n\n"
        f"Required coverage obligations: {coverage_obligations}.\n"
        "These obligations combine the user's requested output sections with "
        "every static lane. If the draft report or coverage_review mentions a "
        "shorter required_outputs list, treat it as a hint, not a limit. "
        "Use clear user-facing headings and fold each lane's supported "
        "evidence into the most relevant section, or mark that lane's data "
        "unavailable when the evidence pool is empty.\n"
        "If coverage_review.decision is not 'complete', repair the draft by "
        "removing unsupported claims, marking missing data explicitly, and "
        "adding the required section headings that can be supported by the "
        "observations/sources pools. Never invent figures, citations, dates, "
        "rankings, forecasts, or recommendations to satisfy the checklist."
        f"{lane_section}"
    )


def _plan_execute_synthesizer_directive(lane_specs: list[dict[str, str]]) -> str:
    lane_list = "\n".join(
        f"- {spec['id']}: {spec.get('description', '').strip() or '(no description)'}"
        for spec in lane_specs
    )
    lane_section = f"\n\nPlanner lanes/workstreams:\n{lane_list}\n" if lane_list else ""
    return (
        "## Plan-And-Execute Evidence Contract\n\n"
        "This workflow uses a plan_and_execute topology. Planner steps, "
        "reflection text, routing decisions, and evaluator notes are control "
        "signals only; they are not evidence. Write the final report from the "
        "observations and sources pools. If a planned lane or step produced no "
        "substantive observation, mark that coverage gap rather than filling it "
        "from planner prose. Cite only URLs that appear in the sources pool and "
        "only when the source record includes usable snippet, content, "
        "structured metrics, or quoted evidence."
        f"{lane_section}"
    )


def _grounded_synthesizer_output_schema(
    compiled_brief: WorkflowDesignBrief,
) -> dict[str, Any]:
    """Strict verifier defaults for designer-generated grounded workflows."""
    if compiled_brief.grounding_mode == "none":
        return {}
    return {
        "claim_disposition": {"abstained": "remove"},
        "report_contract": {
            "domain": compiled_brief.domain,
            "user_goal": compiled_brief.user_goal,
            "required_outputs": list(compiled_brief.required_outputs),
            "quality_gates": list(compiled_brief.quality_gates),
            "constraints": list(compiled_brief.constraints),
        },
    }


def _reflector_workflow_directive(compiled_brief: WorkflowDesignBrief) -> str:
    required_outputs = _brief_list(
        compiled_brief.required_outputs,
        fallback="the user's requested deliverable",
    )
    quality_gates = _brief_list(
        compiled_brief.quality_gates,
        fallback="cite evidence-backed claims and flag missing data",
    )
    return (
        "## Workflow-Specific Review Contract\n\n"
        f"Required outputs: {required_outputs}.\n"
        f"Quality gates: {quality_gates}.\n"
        "Evaluate whether findings are backed by source text, not whether the "
        "planner described a plausible next step. Mark incomplete when evidence "
        "is missing, metadata-only, or too generic for the requested output."
    )


def _build_parallel_lanes_workflow(
    intent: str,
    name: str,
    compiled_brief: WorkflowDesignBrief,
) -> dict[str, Any]:
    """Parallel-lanes topology (NEW DEFAULT).

    Shape: ``sequence(coordinator, parallel(lane_1, ..., lane_N),
    draft_synthesizer, coverage_reflector, final_synthesizer)``.

    Each lane researcher runs concurrently against the shared workflow state,
    carries its LaneSpec.system_prompt specialization, and writes findings to
    the ``observations`` / ``sources`` pools via ``pool_writes``. The
    draft/final synthesizers and the coverage reflector read from those same
    pools via ``pool_inject`` — identical output-aggregation pattern to the
    legacy plan_and_execute topology, just without the planner/router/cross-
    lane-fallback brittleness that caused every step to fall through to the
    catch-all branch.

    Per-lane ``error_handling=skip`` keeps single-lane failures from halting
    the whole parallel; the engine emits a NodeSkippedEvent and other lanes
    continue.
    """
    domain_label = _workflow_domain_label(compiled_brief)
    coordinator_system, _ = _default_prompts("coordinator")
    researcher_system, _ = _default_prompts("researcher")
    reflector_system, _ = _default_prompts("reflector")
    synthesizer_system, _ = _default_prompts("synthesizer")
    synthesizer_system = _designer_synthesizer_system_prompt(synthesizer_system)
    lane_specs = _lane_specs(compiled_brief, intent=intent)
    _require_authored_lane_specs(lane_specs, topology="parallel_lanes")
    tools = [
        make_tool_decl(
            name="web_search",
            kind="web_search",
            config={"max_results": 10},
            description="Search the public web for relevant sources.",
        ),
        make_tool_decl(
            name="web_crawl",
            kind="web_crawl",
            description=(
                "Fetch and extract content from selected web pages. Requires "
                "a url_index from a prior web_search result in this workflow "
                "run; call web_search first when no valid index is available."
            ),
        ),
    ]
    coordinator = make_agent_node(
        node_id="coordinator",
        label="Coordinator",
        subtype="coordinator",
        input_keys=["query"],
        output_key="coordination",
        model_tier="simple",
        output_format="json",
        max_tool_calls=0,
        extra_config={
            "system_prompt": _with_designer_goal(
                coordinator_system,
                intent,
                role="coordinator",
                design_brief=compiled_brief,
            ),
        },
    )
    base_researcher_prompt = _with_designer_goal(
        researcher_system,
        intent,
        role="researcher",
        design_brief=compiled_brief,
    )
    # Each lane researcher is static. It does not depend on current_step,
    # research_plan, or planner lane routing; it consumes the resolved scope
    # from the coordinator and writes findings to a lane-specific state key.
    lane_researchers: list[dict[str, Any]] = []
    for spec in lane_specs:
        agent_node = make_agent_node(
            node_id=f"{spec['id']}-researcher",
            label=spec["label"],
            subtype="researcher",
            # ``coordination`` carries the coordinator's ExtractedScope
            # (entities, time_window, comparables) so each lane researcher
            # consumes resolved scope instead of re-extracting it from the
            # raw query. Generic state.get() fallback handles the lookup;
            # no resolver registration needed.
            input_keys=["query", "coordination"],
            output_key=f"findings_{spec['id']}",
            model_tier="analytical",
            output_format="json",
            tools=["web_search", "web_crawl"],
            pool_writes=[
                {"pool": "observations", "extract": f"findings_{spec['id']}"},
                {"pool": "sources", "extract": "sources"},
            ],
            # Initial search burst (5-6 calls) + room for follow-up crawls.
            # With Phase 1 Brave throttling in place, extra budget does not
            # worsen rate-limit pressure. Lower bound 12 keeps researchers
            # from truncating mid-thought ("Let me crawl...") as seen in
            # live planning-leak traces.
            max_tool_calls=12,
            extra_config=_lane_extra_config(
                system_prompt=_assemble_lane_system_prompt(
                    base_researcher_prompt=base_researcher_prompt,
                    spec=spec,
                    # parallel_lanes lanes consume coordination via input_keys
                    # → renderer fills {coordination} from state at runtime.
                    include_scope_block=True,
                ),
                spec=spec,
            ),
        )
        # error_handling sits at the WorkflowNode top level (not inside config).
        # See ``ErrorConfig`` in workflow/definition.py: on_error="skip" emits
        # a NodeSkippedEvent and lets sibling parallel children proceed.
        agent_node["error_handling"] = {
            "on_error": "skip",
            "max_retries": 1,
            "retry_delay_seconds": 1.0,
        }
        lane_researchers.append(agent_node)
    parallel_node = make_parallel(
        node_id="parallel-lanes",
        label=f"{domain_label} Parallel Research Lanes",
        children=lane_researchers,
    )
    synthesizer = make_agent_node(
        node_id="synthesizer",
        label=f"{domain_label} Draft Report Synthesizer",
        subtype="synthesizer",
        # input_keys drops plan_and_execute artifacts (research_plan,
        # reflection); the synthesizer here consumes everything through the
        # pools below.
        input_keys=["query", "coordination"],
        output_key="draft_report",
        model_tier="complex",
        output_format="markdown",
        pool_inject=[
            {"pool": "observations", "threshold": 0},
            {"pool": "sources", "threshold": 0},
        ],
        max_tool_calls=0,
        extra_config={
            "system_prompt": _with_designer_goal(
                _synthesizer_lane_coverage_directive(lane_specs)
                + "\n\n"
                + synthesizer_system,
                intent,
                role="synthesizer",
                design_brief=compiled_brief,
            ),
            # Engage the strict reclaim prompt (synthesizer.py:208
            # _build_reclaim_system_prompt) by default. Carried through from
            # the Designer brief (WorkflowDesignBrief.grounding_mode) so the
            # LLM can toggle to "classical_lite" (full citation pipeline) for
            # high-assurance workflows or "none" for speed-over-accuracy.
            # Default "reclaim" is the safe floor — strict anti-confabulation
            # prompt at zero extra LLM cost vs "none".
            "grounding_mode": compiled_brief.grounding_mode,
            "output_schema": _grounded_synthesizer_output_schema(compiled_brief),
            # synthesis_metadata cannot live on an agent node (AgentNodeConfig
            # extra=forbid); it is only valid on plan_and_execute. The
            # save-time critic gate now also reads ``workflow.description``
            # so it still recovers the user's intent for parallel_lanes
            # workflows. The brief fields the legacy block carried
            # (designer_research_lanes, designer_required_outputs, etc.) are
            # already embedded in this synthesizer's system_prompt by
            # ``_with_designer_goal`` above — no separate metadata needed.
        },
    )
    coverage_reflector = make_agent_node(
        node_id="coverage-reflector",
        label=f"{domain_label} Coverage Reflector",
        subtype="reflector",
        input_keys=["query", "coordination", "draft_report"],
        output_key="coverage_review",
        model_tier="analytical",
        output_format="json",
        pool_inject=[
            {"pool": "observations", "threshold": 0},
            {"pool": "sources", "threshold": 0},
        ],
        max_tool_calls=0,
        extra_config={
            "system_prompt": _with_designer_goal(
                _final_coverage_reflector_directive(compiled_brief, lane_specs)
                + "\n\n"
                + reflector_system,
                intent,
                role="reflector",
                design_brief=compiled_brief,
            ),
            "user_prompt_template": (
                "Review the draft report for {query}.\n\n"
                "## Draft Report\n{draft_report}\n\n"
                "Return reflector JSON with decision, reasoning, "
                "suggested_changes, evidence_sufficiency, and failure_mode."
            ),
        },
    )
    finalizer = make_agent_node(
        node_id="final-report-synthesizer",
        label=f"{domain_label} Final Report Synthesizer",
        subtype="synthesizer",
        input_keys=["query", "coordination", "draft_report", "coverage_review"],
        output_key="report",
        model_tier="complex",
        output_format="markdown",
        pool_inject=[
            {"pool": "observations", "threshold": 0},
            {"pool": "sources", "threshold": 0},
        ],
        max_tool_calls=0,
        extra_config={
            "system_prompt": _with_designer_goal(
                _final_report_repair_directive(compiled_brief, lane_specs)
                + "\n\n"
                + synthesizer_system,
                intent,
                role="synthesizer",
                design_brief=compiled_brief,
            ),
            "user_prompt_template": (
                "Produce the final report for {query}.\n\n"
                "## Draft Report\n{draft_report}\n\n"
                "## Coverage Review\n{coverage_review}\n\n"
                "Use only observations and sources available in the injected "
                "evidence pools."
            ),
            "grounding_mode": compiled_brief.grounding_mode,
            "output_schema": _grounded_synthesizer_output_schema(compiled_brief),
        },
    )
    workflow = {
        "id": "designer-draft",
        "name": compiled_brief.workflow_name or name,
        "description": intent,
        "version": 1,
        "tools": tools,
        "pools": [
            make_pool_decl(name="sources", dedup_key="url", max_items=100),
            make_pool_decl(name="observations", dedup_key="content_hash", max_items=50),
        ],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["report"],
        "token_budget": 0,
        "timeout_seconds": 1800,
        "root": make_sequence(
            node_id="main",
            label=f"{domain_label} Pipeline",
            children=[
                coordinator,
                parallel_node,
                synthesizer,
                coverage_reflector,
                finalizer,
            ],
        ),
    }
    validate_generated_workflow(workflow)
    return workflow


def _build_single_agent_workflow(
    intent: str,
    name: str,
    compiled_brief: WorkflowDesignBrief,
) -> dict[str, Any]:
    """Single-agent topology — for short factual questions.

    Shape: ``sequence(coordinator, agent)``. No lanes, no parallel, no
    plan_and_execute, no synthesizer. The agent answers the user's request
    directly with whatever specialization the brief / LaneSpec carries.
    """
    domain_label = _workflow_domain_label(compiled_brief)
    coordinator_system, _ = _default_prompts("coordinator")
    researcher_system, _ = _default_prompts("researcher")

    coordinator = make_agent_node(
        node_id="coordinator",
        label="Coordinator",
        subtype="coordinator",
        input_keys=["query"],
        output_key="coordination",
        model_tier="simple",
        output_format="json",
        max_tool_calls=0,
        extra_config={
            "system_prompt": _with_designer_goal(
                coordinator_system,
                intent,
                role="coordinator",
                design_brief=compiled_brief,
            ),
        },
    )

    # Use the first lane's specialization as the single-agent prompt contract.
    base_researcher_prompt = _with_designer_goal(
        researcher_system,
        intent,
        role="researcher",
        design_brief=compiled_brief,
    )
    lane_specs = _lane_specs(compiled_brief, intent=intent)
    _require_authored_lane_specs(lane_specs, topology="single_agent")
    single_spec: dict[str, str] = {
        **lane_specs[0],
        "id": "agent",
        "label": "Direct Answer Agent",
    }
    agent_prompt = _assemble_lane_system_prompt(
        base_researcher_prompt=base_researcher_prompt,
        spec=single_spec,
        # single_agent declares coordination in input_keys; the scope
        # block renders the same way as parallel_lanes.
        include_scope_block=True,
    )

    tools = [
        make_tool_decl(
            name="web_search",
            kind="web_search",
            config={"max_results": 5},
            description="Search the public web for relevant sources.",
        ),
    ]
    agent_node = make_agent_node(
        node_id="answer-agent",
        label=f"{domain_label} Answer Agent",
        subtype="researcher",
        input_keys=["query", "coordination"],
        output_key="report",
        model_tier="analytical",
        output_format="markdown",
        tools=["web_search"],
        max_tool_calls=4,
        extra_config=_lane_extra_config(
            system_prompt=agent_prompt,
            spec=single_spec,
        ),
    )
    workflow = {
        "id": "designer-draft",
        "name": compiled_brief.workflow_name or name,
        "description": intent,
        "version": 1,
        "tools": tools,
        "pools": [],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["report"],
        "token_budget": 0,
        "timeout_seconds": 600,
        "root": make_sequence(
            node_id="main",
            label=f"{domain_label} Single-Agent",
            children=[coordinator, agent_node],
        ),
    }
    validate_generated_workflow(workflow)
    return workflow


def build_direct_workflow(intent: str, name: str) -> dict[str, Any]:
    """Build a minimal direct-answer workflow with explicit prompts."""
    agent = make_agent_node(
        node_id="agent",
        label="Direct Answer Agent",
        subtype="custom",
        input_keys=["query"],
        output_key="output",
        model_tier="analytical",
        output_format="markdown",
        max_tool_calls=0,
        extra_config={
            "system_prompt": (
                "You answer the user's request directly and concisely. "
                "Use only the conversation context and the user query."
            ),
            "user_prompt_template": "{query}",
        },
    )
    workflow = {
        "id": "designer-draft",
        "name": name,
        "description": intent,
        "version": 1,
        "root": make_sequence(node_id="root", label="Workflow", children=[agent]),
        "tools": [],
        "pools": [],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["output"],
        "token_budget": 0,
        "timeout_seconds": 1800,
    }
    validate_generated_workflow(workflow)
    return workflow


def validate_generated_workflow(workflow: dict[str, Any]) -> None:
    """Validate structural and Designer semantic invariants for generated ASTs."""
    definition = load_workflow_from_dict(workflow)
    declared_tools = {tool.name for tool in definition.tools}
    terminal_outputs = set(definition.output_keys)
    produced_outputs: set[str] = set()
    errors: list[str] = []

    def walk(node: dict[str, Any], available_keys: set[str], in_loop: set[str]) -> set[str]:
        node_type = str(node.get("type", ""))
        node_id = str(node.get("id", "<unknown>"))
        raw_config = node.get("config")
        config = raw_config if isinstance(raw_config, dict) else {}
        next_keys = set(available_keys)
        if node_type == "agent":
            _validate_agent_semantics(
                node_id, config, declared_tools, available_keys | in_loop, errors
            )
            output_key = config.get("output_key")
            if isinstance(output_key, str) and output_key:
                next_keys.add(output_key)
                produced_outputs.add(output_key)
        elif node_type in {"sequence", "parallel", "loop", "conditional"}:
            for child in node.get("children") or []:
                if isinstance(child, dict):
                    next_keys.update(walk(child, next_keys, in_loop))
        elif node_type == "plan_and_execute":
            planner = config.get("planner")
            planner_output = None
            if isinstance(planner, dict):
                _validate_agent_semantics(
                    f"{node_id}.planner", planner, declared_tools, available_keys, errors
                )
                planner_output = planner.get("output_key")
                if isinstance(planner_output, str) and planner_output:
                    next_keys.add(planner_output)
                    produced_outputs.add(planner_output)
            item_key = str(config.get("item_state_key") or "current_step")
            body = config.get("body")
            if isinstance(body, dict):
                loop_keys = set(in_loop) | {item_key}
                if isinstance(planner_output, str):
                    loop_keys.add(planner_output)
                next_keys.update(walk(body, next_keys, loop_keys))
            evaluator = config.get("evaluator")
            if isinstance(evaluator, dict):
                eval_keys = set(next_keys) | {item_key}
                if isinstance(planner_output, str):
                    eval_keys.add(planner_output)
                _validate_agent_semantics(
                    f"{node_id}.evaluator", evaluator, declared_tools, eval_keys, errors
                )
        return next_keys

    walk(workflow["root"], set(definition.required_inputs), set())
    missing_outputs = terminal_outputs - produced_outputs
    if missing_outputs:
        errors.append(
            f"Workflow output_keys are not produced by any node: {sorted(missing_outputs)}"
        )
    if errors:
        raise WorkflowGenerationError("; ".join(errors))


def _validate_agent_semantics(
    node_id: str,
    config: dict[str, Any],
    declared_tools: set[str],
    available_keys: set[str],
    errors: list[str],
) -> None:
    system_prompt = str(config.get("system_prompt") or "")
    user_prompt = str(config.get("user_prompt_template") or "")
    if not system_prompt.strip():
        errors.append(f"Agent '{node_id}' has empty system_prompt")
    if not user_prompt.strip():
        errors.append(f"Agent '{node_id}' has empty user_prompt_template")

    renderer = SafeTemplateRenderer()
    prompt_vars = renderer.extract_variables(system_prompt) | renderer.extract_variables(
        user_prompt
    )
    configured_inputs = set(config.get("input_keys") or [])
    unavailable_inputs = configured_inputs - available_keys - _RUNTIME_TEMPLATE_KEYS
    if unavailable_inputs:
        errors.append(f"Agent '{node_id}' has unavailable input_keys: {sorted(unavailable_inputs)}")
    unresolved = prompt_vars - configured_inputs - available_keys - _RUNTIME_TEMPLATE_KEYS
    if unresolved:
        errors.append(f"Agent '{node_id}' has unresolved prompt variables: {sorted(unresolved)}")

    for tool_name in config.get("tools") or []:
        if isinstance(tool_name, str) and declared_tools and tool_name not in declared_tools:
            errors.append(f"Agent '{node_id}' references undeclared tool '{tool_name}'")
