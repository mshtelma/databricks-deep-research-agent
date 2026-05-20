"""Deployability checks for Agent Designer revision snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

DEFAULT_WORKFLOW_NAME = "Untitled Agent"
DEFAULT_SCAFFOLD_CHILD_IDS = ("coordinator", "plan-and-execute", "synthesizer")
DEFAULT_REVISION_ERROR_KIND = "default_revision_not_deployable"


@dataclass(frozen=True)
class RevisionDeployability:
    """Classification summary for a saved workflow revision."""

    deployable: bool
    classification: str
    workflow_name: str
    workflow_description: str
    root_child_ids: tuple[str, ...]
    root_child_summary: list[str]
    planner_guidance_present: bool


def _clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def root_child_summary(definition: dict[str, Any]) -> list[str]:
    """Return stable ``id:type:label`` strings for the root child nodes."""
    root = definition.get("root")
    if not isinstance(root, dict):
        return []
    children = root.get("children")
    if not isinstance(children, list):
        return []

    summary: list[str] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        node_id = _clean_text(child.get("id")) or "<unnamed>"
        node_type = _clean_text(child.get("type")) or "<unknown>"
        label = _clean_text(child.get("label"))
        summary.append(f"{node_id}:{node_type}:{label}")
    return summary


def root_child_ids(definition: dict[str, Any]) -> tuple[str, ...]:
    """Return root child ids, preserving order."""
    root = definition.get("root")
    if not isinstance(root, dict):
        return ()
    children = root.get("children")
    if not isinstance(children, list):
        return ()
    ids: list[str] = []
    for child in children:
        if isinstance(child, dict):
            ids.append(_clean_text(child.get("id")))
    return tuple(ids)


def has_planner_guidance(definition: dict[str, Any]) -> bool:
    """Return True when a plan-and-execute node carries explicit guidance."""

    def _walk(value: Any) -> bool:
        if isinstance(value, dict):
            for key in ("planner_guidance", "prompt_guidance"):
                guidance = value.get(key)
                if isinstance(guidance, str) and guidance.strip():
                    return True
            for child in value.values():
                if _walk(child):
                    return True
        elif isinstance(value, list):
            return any(_walk(child) for child in value)
        return False

    return _walk(definition)


def classify_revision_deployability(definition: dict[str, Any]) -> RevisionDeployability:
    """Classify whether a revision is deployable.

    Default policy: block the empty stock scaffold generated before the user
    applies a real design. The check is intentionally generic and only looks
    for default workflow identity plus the stock root child structure.
    """
    workflow_name = _clean_text(definition.get("name"))
    workflow_description = _clean_text(definition.get("description"))
    children = root_child_ids(definition)
    planner_guidance_present = has_planner_guidance(definition)

    is_default_name = workflow_name in {"", DEFAULT_WORKFLOW_NAME}
    is_default_scaffold = (
        is_default_name
        and not workflow_description
        and children == DEFAULT_SCAFFOLD_CHILD_IDS
        and not planner_guidance_present
    )

    return RevisionDeployability(
        deployable=not is_default_scaffold,
        classification="default_scaffold" if is_default_scaffold else "deployable",
        workflow_name=workflow_name,
        workflow_description=workflow_description,
        root_child_ids=children,
        root_child_summary=root_child_summary(definition),
        planner_guidance_present=planner_guidance_present,
    )


def default_revision_not_deployable_detail(
    *,
    agent_id: UUID,
    revision_id: UUID,
    deployability: RevisionDeployability,
) -> dict[str, Any]:
    """Build the HTTP 422 detail body for a blocked default revision."""
    return {
        "error_kind": DEFAULT_REVISION_ERROR_KIND,
        "agent_id": str(agent_id),
        "revision_id": str(revision_id),
        "workflow_name": deployability.workflow_name,
        "root_child_summary": deployability.root_child_summary,
        "message": "Save or select a designed workflow revision before deploying.",
    }
