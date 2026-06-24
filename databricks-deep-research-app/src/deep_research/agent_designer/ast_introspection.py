"""Shared, pure AST-introspection helpers for the Agent Designer.

These walkers were originally private to :mod:`probe.py`; they are promoted
here so the edit lane (``edit_planning``, the edit-diff guard, and
``signature_from_ast``) and the behavioral probe share ONE implementation
instead of duplicating tree-walking logic.

Everything here is pure (no LLM, no I/O) and operates on the workflow AST
dict shape: a top-level mapping with a ``root`` node, where every node has
``type``/``id``/``label``/``config`` and composite nodes carry ``children``
(and ``plan_and_execute`` bodies live under ``config.body``).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

# Builder-owned id convention for a tree_search level parallel
# (``_build_tree_search_workflow`` names them ``l{level}_research-level``). The
# topology walker keys off this structural marker to classify tree_search at the
# root BEFORE the generic parallel-recursion — otherwise its level parallels would
# make the walker infer ``parallel_lanes`` (including the depth-1 case, which is
# structurally a single parallel). This is a deterministic node-id convention
# emitted by the builder, not a content/domain keyword match.
_TREE_LEVEL_PARALLEL_ID_RE = re.compile(r"^l\d+_research-level$")


def config_of(node: Any) -> dict[str, Any]:
    """Return ``node['config']`` when it is a dict; otherwise an empty dict."""
    if not isinstance(node, dict):
        return {}
    raw = node.get("config")
    return raw if isinstance(raw, dict) else {}


def iter_agent_nodes(node: Any) -> Iterator[dict[str, Any]]:
    """Yield every agent node under *node*.

    Walks ``children`` AND ``config.body`` so ``plan_and_execute`` branches
    are covered.
    """
    if not isinstance(node, dict):
        return
    if node.get("type") == "agent":
        yield node
    for child in node.get("children") or []:
        yield from iter_agent_nodes(child)
    body = config_of(node).get("body")
    if isinstance(body, dict):
        yield from iter_agent_nodes(body)


def iter_all_nodes(node: Any) -> Iterator[dict[str, Any]]:
    """Yield every node (any type) under *node*.

    Walks ``children`` AND ``config.body`` — used to locate composite nodes
    (loop/conditional) that the agent-only walker skips.
    """
    if not isinstance(node, dict):
        return
    yield node
    for child in node.get("children") or []:
        yield from iter_all_nodes(child)
    body = config_of(node).get("body")
    if isinstance(body, dict):
        yield from iter_all_nodes(body)


def tool_kinds_for_lane(
    lane: dict[str, Any], ast_tools: list[dict[str, Any]]
) -> set[str]:
    """Return the set of tool kinds bound to *lane*, resolving by name."""
    tool_names = config_of(lane).get("tools") or []
    name_to_kind: dict[str, str] = {}
    for tool in ast_tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        kind = tool.get("kind")
        if isinstance(name, str) and isinstance(kind, str):
            name_to_kind[name] = kind
    return {name_to_kind[name] for name in tool_names if name in name_to_kind}


def is_lane_researcher(lane: dict[str, Any]) -> bool:
    """Heuristic: a researcher lane is an agent whose subtype is ``researcher``
    OR whose id starts with ``lane_``."""
    if not isinstance(lane, dict):
        return False
    subtype = config_of(lane).get("subtype")
    lane_id = str(lane.get("id") or "")
    return subtype == "researcher" or lane_id.startswith("lane_")


def topology_of_ast(ast: dict[str, Any]) -> str:
    """Best-effort topology classification by walking the root node tree."""
    root = ast.get("root") if isinstance(ast, dict) else None
    if not isinstance(root, dict):
        return "unknown"
    return topology_of_node(root)


def topology_of_node(node: dict[str, Any]) -> str:
    """Classify the topology rooted at *node*.

    Returns one of ``plan_and_execute``, ``router``, ``tree_search``,
    ``parallel_lanes``, ``single_agent``, or ``unknown``. ``best_of_n`` and other
    parallel fan-outs report as ``parallel_lanes`` (their family) — topology-specific
    shape is verified by dedicated invariants elsewhere.
    """
    if not isinstance(node, dict):
        return "unknown"
    if node.get("type") == "plan_and_execute":
        return "plan_and_execute"
    children = node.get("children") or []
    # router: a sequence whose child is a conditional (and which has no direct
    # parallel). Checked BEFORE the generic parallel-recursion — otherwise an
    # evidence parallel INSIDE a router branch would make the walker return
    # parallel_lanes and false-classify.
    if node.get("type") == "sequence":
        has_parallel = any(
            isinstance(c, dict) and c.get("type") == "parallel" for c in children
        )
        has_conditional = any(
            isinstance(c, dict) and c.get("type") == "conditional" for c in children
        )
        if has_conditional and not has_parallel:
            return "router"
        # tree_search: a sequence containing one or more level parallels with the
        # builder's ``l{N}_research-level`` id convention. Checked BEFORE the
        # generic parallel-recursion — otherwise its level parallels (including the
        # depth-1 single-parallel case) would make the walker return
        # parallel_lanes and false-classify. Mirrors the router precedence guard.
        if any(
            isinstance(c, dict)
            and c.get("type") == "parallel"
            and _TREE_LEVEL_PARALLEL_ID_RE.match(str(c.get("id") or ""))
            for c in children
        ):
            return "tree_search"
    # parallel_lanes: a sequence that contains a parallel node
    for child in children:
        if isinstance(child, dict) and child.get("type") == "parallel":
            return "parallel_lanes"
        nested = topology_of_node(child)
        if nested != "unknown":
            return nested
    # single_agent: root is a sequence whose children are all agents
    if node.get("type") == "sequence":
        agents = [c for c in children if isinstance(c, dict) and c.get("type") == "agent"]
        if agents and len(agents) == len(children):
            return "single_agent"
    return "unknown"


__all__ = [
    "config_of",
    "iter_agent_nodes",
    "iter_all_nodes",
    "tool_kinds_for_lane",
    "is_lane_researcher",
    "topology_of_ast",
    "topology_of_node",
]
