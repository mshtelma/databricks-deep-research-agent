"""Schema-aware prompt injection (Part A) shared by app + shell-app.

An agent whose surface declares output slots should have its research AND its
synthesis targeted at those slots, so every declared section is fillable. This
module builds the two append-only, brace-sanitized contract blocks from the
declared slots and injects them into the designer-authored ``user_prompt_template``s
of a built :class:`WorkflowDefinition`:

- synthesizer: cover every slot with concrete, citable content (so the per-slot
  structuring wires become a near-mechanical projection);
- planner/researcher: gather evidence for every slot, so research is driven by
  the schema and slots don't render empty for want of evidence.

Only designer-authored templates are touched (builtin default prompts are left
as-is). Idempotent (markers guard re-injection) and fail-soft by contract.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from databricks_deep_research.surface.output_schema import SlotSpec, slot_docs
from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)

# Synthesizer contract: make the narrative report cover every slot.
STRUCTURED_CONTRACT_MARKER = "## Structured results coverage"

_STRUCTURED_CONTRACT_HEADER = (
    f"\n\n{STRUCTURED_CONTRACT_MARKER}\n"
    "A downstream step converts this report into structured UI data. Make "
    "sure the report explicitly provides, with concrete values and citation "
    "markers, every item below:\n"
)

_STRUCTURED_CONTRACT_FOOTER = (
    "\nCover EVERY item with specific figures/entries — compact markdown "
    "tables or lists are ideal. If information for an item is genuinely "
    "unavailable, state that explicitly instead of omitting it."
)

# Research-targeting variant: the planner/researchers see the SAME declared
# slots so their evidence-gathering is driven by the schema (not just the
# synthesizer's report). This is what makes every declared slot fillable —
# without it, research gathers generic evidence and data-specific slots
# (competitors, financials, …) come back empty for want of evidence.
RESEARCH_CONTRACT_MARKER = "## Required result sections"

_RESEARCH_CONTRACT_HEADER = (
    f"\n\n{RESEARCH_CONTRACT_MARKER}\n"
    "This agent's final report must fill the structured result sections below. "
    "Your research MUST gather concrete, citable evidence for EACH section — "
    "decompose the work so every section can be populated. Never invent data; "
    "if a section is genuinely unavailable after searching, note that.\n"
)

_RESEARCH_CONTRACT_FOOTER = (
    "\nPrioritize COVERAGE across all sections over exhaustive depth on any one."
)


def build_contracts(slots: dict[str, SlotSpec]) -> tuple[str, str]:
    """Return ``(research_contract, synth_contract)`` for *slots*.

    Both are brace-sanitized: designer-authored column labels must never become
    ``SafeTemplateRenderer`` placeholders in an injected prompt.
    """
    docs = slot_docs(slots)
    synth = (
        _STRUCTURED_CONTRACT_HEADER + docs + _STRUCTURED_CONTRACT_FOOTER
    ).replace("{", "(").replace("}", ")")
    research = (
        _RESEARCH_CONTRACT_HEADER + docs + _RESEARCH_CONTRACT_FOOTER
    ).replace("{", "(").replace("}", ")")
    return research, synth


def visit_agent_configs(
    workflow: WorkflowDefinition,
    visitor: Callable[[dict[str, Any]], None],
) -> None:
    """Visit agent configs, including agent configs nested in raw config bodies."""

    def visit_raw(raw_node: dict[str, Any]) -> None:
        config_dict = raw_node.get("config")
        if isinstance(config_dict, dict):
            if raw_node.get("type") == NodeType.agent.value:
                visitor(config_dict)
            elif raw_node.get("type") == NodeType.plan_and_execute.value:
                for nested_key in ("planner", "evaluator"):
                    nested = config_dict.get(nested_key)
                    if isinstance(nested, dict):
                        visitor(nested)
                body = config_dict.get("body")
                if isinstance(body, dict):
                    visit_raw(body)
        for child in raw_node.get("children") or []:
            if isinstance(child, dict):
                visit_raw(child)

    def visit_node(node: WorkflowNode) -> None:
        if node.type == NodeType.agent:
            visitor(node.config)
        elif node.type == NodeType.plan_and_execute:
            for nested_key in ("planner", "evaluator"):
                nested = node.config.get(nested_key)
                if isinstance(nested, dict):
                    visitor(nested)
            body = node.config.get("body")
            if isinstance(body, dict):
                visit_raw(body)
        for child in node.children:
            visit_node(child)

    visit_node(workflow.root)


def inject_structured_output_contract(
    workflow: WorkflowDefinition,
    slots: dict[str, SlotSpec],
) -> tuple[int, int]:
    """Append the schema contracts to designer-authored agent prompts.

    Mutates *workflow* in place. Returns ``(injected_synth, injected_research)``
    counts. No-op when *slots* is empty. Idempotent per template (markers guard).
    """
    if not slots:
        return (0, 0)
    research_contract, synth_contract = build_contracts(slots)
    injected_synth = 0
    injected_research = 0

    def inject(agent_config: dict[str, Any]) -> None:
        nonlocal injected_synth, injected_research
        subtype = agent_config.get("subtype")
        template = agent_config.get("user_prompt_template")
        if not isinstance(template, str) or not template.strip():
            return
        if subtype == "synthesizer":
            if STRUCTURED_CONTRACT_MARKER in template:
                return  # idempotent
            agent_config["user_prompt_template"] = template + synth_contract
            injected_synth += 1
        elif subtype in ("planner", "researcher"):
            if RESEARCH_CONTRACT_MARKER in template:
                return  # idempotent
            agent_config["user_prompt_template"] = template + research_contract
            injected_research += 1

    visit_agent_configs(workflow, inject)
    return (injected_synth, injected_research)


__all__ = [
    "RESEARCH_CONTRACT_MARKER",
    "STRUCTURED_CONTRACT_MARKER",
    "build_contracts",
    "inject_structured_output_contract",
    "visit_agent_configs",
]
