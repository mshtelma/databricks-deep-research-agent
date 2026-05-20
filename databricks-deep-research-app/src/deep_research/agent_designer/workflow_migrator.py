"""Detect saved agents that predate the per-node user_prompt_template
upgrade so the UI can surface a regeneration banner.

The Phase 1 wiring shipped a per-researcher ``user_prompt_template`` field;
agents created BEFORE that change have the field missing or set to the
generic ``RESEARCHER_USER_PROMPT`` builtin, which leaves the lane with no
concrete sub-questions and causes planning-text findings (the failure mode
the NVDA trace exposed).

This walker traverses any topology — single_agent, parallel_lanes,
sequence, loop, plan_and_execute — and returns a structured report listing
each researcher node that needs regeneration. The UI consumes the report
to surface a banner; the LLM-driven in-place upgrade path (Phase 4.2) uses
the same report to know which nodes to refresh.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Markers reused from the structural validator — keep in sync.
_GENERIC_USER_PROMPT_MARKERS = (
    "Execute the following research step:",
    "{step_title}",
)


@dataclass(frozen=True)
class MigratableResearcher:
    """One researcher node that needs a user_prompt_template upgrade."""

    node_id: str
    label: str
    reason: str  # "missing" | "generic_default"
    path: str


@dataclass(frozen=True)
class WorkflowMigrationReport:
    """Result of scanning a saved workflow for migration needs.

    Empty ``researchers`` means the workflow is up-to-date; any items mean
    the saved agent should surface a regeneration banner / be offered the
    in-place upgrade path. ``needs_regeneration`` is a convenience flag.
    """

    needs_regeneration: bool
    researchers: list[MigratableResearcher] = field(default_factory=list)


def _walk_agents(node: Any, path: str) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """Yield ``(node, config, path)`` for every agent in the workflow.

    Recurses into ``plan_and_execute`` planner/evaluator/body and any
    composite children. Same shape as
    ``semantic_validation._collect_agent_paths`` but kept local here so the
    migrator can ship/iterate independently.
    """
    collected: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    if not isinstance(node, dict):
        return collected
    config = node.get("config") or {}
    if not isinstance(config, dict):
        config = {}
    if node.get("type") == "agent":
        collected.append((node, config, path))
    if node.get("type") == "plan_and_execute":
        for nested_key in ("planner", "evaluator"):
            nested = config.get(nested_key)
            if isinstance(nested, dict):
                collected.append((nested, nested, f"{path}.config.{nested_key}"))
        body = config.get("body")
        if isinstance(body, dict):
            collected.extend(_walk_agents(body, f"{path}.config.body"))
    for idx, child in enumerate(node.get("children", []) or []):
        collected.extend(_walk_agents(child, f"{path}.children[{idx}]"))
    return collected


def _is_generic_default_template(template: str) -> bool:
    """Detect the generic ``RESEARCHER_USER_PROMPT`` builtin by marker text."""
    if not template:
        return True
    return any(marker in template for marker in _GENERIC_USER_PROMPT_MARKERS)


def scan_workflow_for_migration(definition: dict[str, Any]) -> WorkflowMigrationReport:
    """Walk the workflow graph; report researcher nodes needing the
    per-node user_prompt_template upgrade.

    Topology-agnostic: works on single_agent, parallel_lanes, sequence,
    loop, plan_and_execute, custom DAG. Only researcher subtype agents are
    candidates; planner/synthesizer/reflector nodes are intentionally
    ignored (they're upgraded separately when their prompts evolve).
    """
    needs: list[MigratableResearcher] = []
    for node, config, path in _walk_agents(definition.get("root"), "root"):
        subtype = str(config.get("subtype", "")).strip().lower()
        if subtype != "researcher":
            continue
        template = config.get("user_prompt_template") or ""
        if not isinstance(template, str):
            template = str(template or "")
        template_stripped = template.strip()
        if not template_stripped:
            reason = "missing"
        elif _is_generic_default_template(template_stripped):
            reason = "generic_default"
        else:
            continue
        needs.append(
            MigratableResearcher(
                node_id=str(node.get("id", "")),
                label=str(node.get("label") or node.get("id") or "researcher"),
                reason=reason,
                path=path,
            )
        )
    return WorkflowMigrationReport(
        needs_regeneration=bool(needs),
        researchers=needs,
    )
