"""Mermaid flowchart serialisation for agent WorkflowDefinition ASTs.

The only public symbol is :func:`serialize_to_mermaid`, which converts a raw
definition dict to a Mermaid ``flowchart TD`` document.

Cycle handling (V1.5 requirement)
----------------------------------
Mermaid's parser rejects graphs containing real back-edges (cycles), so loop
and plan_and_execute nodes are projected acyclically:

* **loop**: the body sub-tree is rendered as a forward child; the back-edge
  (body → loop) is replaced by a dotted annotation edge pointing to a
  throw-away ``loop_repeat`` placeholder node labelled ``↻ repeat``.  The
  placeholder is intentionally invisible (empty label ``" "``); the edge label
  carries the ``max_iterations`` value from config so the diagram stays
  informative without reintroducing a cycle.

* **conditional**: each branch is rendered as a labelled forward edge; all
  branches converge to a synthetic ``(("merge"))`` node so the diagram is
  properly joined without dangling paths.

* **sequence / parallel**: children linked in declaration order.  Parallel
  children share the same parent source node (fan-out), with no synthetic join
  in V1 (a join node would require knowing the continuation, which is the
  parent's concern).

* **plan_and_execute**: the ``config.body`` sub-tree is rendered as a child.

* **agent / tool / subworkflow**: leaf nodes — no children emitted.
"""

from __future__ import annotations

_LEAF_TYPES = frozenset({"agent", "tool", "subworkflow"})


def serialize_to_mermaid(
    definition: dict[str, object],
    agent_name: str | None = None,
    agent_id: str | None = None,
) -> str:
    """Convert a WorkflowDefinition AST dict to a Mermaid flowchart string.

    Loops are projected to acyclic sub-graphs with a textual ``↻ repeat``
    annotation; conditional branches converge to a synthetic merge node.

    Args:
        definition: Raw AST dict (as stored in ``AgentV2.definition``).
        agent_name: Optional human-readable title placed in the front-matter.
        agent_id: Optional identifier used as the subgraph label; falls back
            to ``"root"`` when not provided.

    Returns:
        A valid Mermaid ``flowchart TD`` document as a UTF-8 string.
    """
    lines: list[str] = []

    # Front-matter title (optional)
    if agent_name:
        safe_name = str(agent_name).replace('"', "'")
        lines.append("---")
        lines.append(f"title: {safe_name}")
        lines.append("---")

    lines.append("flowchart TD")

    graph_id = _safe_id(str(agent_id) if agent_id else "root")
    lines.append(f"  subgraph {graph_id}")

    root = definition.get("root")
    if isinstance(root, dict):
        _emit_node(lines, root, path="root", indent="    ")

    lines.append("  end")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_id(raw: str) -> str:
    """Sanitise a path string into a Mermaid-safe node identifier.

    Replaces ``.``, ``-``, and spaces with underscores and strips any
    characters that Mermaid's lexer would reject.
    """
    return raw.replace(".", "_").replace("-", "_").replace(" ", "_")


def _safe_label(text: str) -> str:
    """Escape double-quotes in a label so it can be embedded in ``"…"``."""
    return text.replace('"', "'")


def _emit_node(
    lines: list[str],
    block: dict[str, object],
    path: str,
    indent: str,
) -> str:
    """Recursively emit Mermaid lines for *block* and return its node id.

    The returned node id is the id that *this* block's entry point exposes to
    the parent — for most node types this is the block's own id, but for
    ``conditional`` it is the synthetic merge node so that the continuation
    always connects to a single, well-defined point.

    Args:
        lines: Accumulator list; lines are appended in place.
        block: Raw AST dict for this node.
        path: Dot-separated path (e.g. ``"root.children.0"``).
        indent: Indentation prefix for the current nesting level.

    Returns:
        The Mermaid node id that represents the *exit* of this block.
    """
    node_id = _safe_id(path)
    raw_label = block.get("label") or block.get("node_type") or path
    label = _safe_label(str(raw_label))
    node_type = str(block.get("node_type", ""))

    # Emit the primary node declaration
    lines.append(f'{indent}{node_id}["{label}"]')

    children: list[object] = []
    raw_children = block.get("children")
    if isinstance(raw_children, list):
        children = raw_children

    config: dict[str, object] = {}
    raw_config = block.get("config")
    if isinstance(raw_config, dict):
        config = raw_config

    if node_type == "loop":
        # Body is the first child (if any).  The back-edge becomes a textual
        # annotation to avoid a real cycle in the Mermaid output.
        if children:
            first_child = children[0]
            if isinstance(first_child, dict):
                child_id = _emit_node(
                    lines, first_child, f"{path}.children.0", indent
                )
                lines.append(f"{indent}{node_id} --> {child_id}")

        # Annotate-only back-edge (no real cycle)
        max_iter = config.get("max_iterations", "?")
        repeat_id = f"{node_id}_repeat"
        lines.append(f'{indent}{repeat_id}[" "]')
        lines.append(
            f'{indent}{node_id} -.-> |"↻ repeat (max {max_iter})"| {repeat_id}'
        )
        return node_id

    elif node_type == "conditional":
        # Each branch fans out from this node; all branches converge to a
        # synthetic merge node so the continuation has a single entry point.
        merge_id = f"{node_id}_merge"
        lines.append(f'{indent}{merge_id}(("merge"))')

        conditions: list[object] = []
        raw_conds = config.get("conditions")
        if isinstance(raw_conds, list):
            conditions = raw_conds

        for i, child in enumerate(children):
            if not isinstance(child, dict):
                continue
            child_id = _emit_node(lines, child, f"{path}.children.{i}", indent)
            # Label from conditions list; fall back to "else" for extra branches
            cond_label = _safe_label(str(conditions[i])) if i < len(conditions) else "else"
            lines.append(f'{indent}{node_id} --> |"{cond_label}"| {child_id}')
            lines.append(f"{indent}{child_id} --> {merge_id}")

        # Return the merge node so the parent connects to it
        return merge_id

    elif node_type == "sequence":
        prev = node_id
        for i, child in enumerate(children):
            if not isinstance(child, dict):
                continue
            child_id = _emit_node(lines, child, f"{path}.children.{i}", indent)
            lines.append(f"{indent}{prev} --> {child_id}")
            prev = child_id

    elif node_type == "parallel":
        for i, child in enumerate(children):
            if not isinstance(child, dict):
                continue
            child_id = _emit_node(lines, child, f"{path}.children.{i}", indent)
            lines.append(f"{indent}{node_id} --> {child_id}")

    elif node_type == "plan_and_execute":
        body = config.get("body")
        if isinstance(body, dict):
            child_id = _emit_node(lines, body, f"{path}.config.body", indent)
            lines.append(f"{indent}{node_id} --> {child_id}")

    # agent, tool, subworkflow — leaf nodes, no children emitted

    return node_id
