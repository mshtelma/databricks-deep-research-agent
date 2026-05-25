"""Pure-function block-mutation primitives operating on workflow AST dicts.

All functions treat input ``ast`` as immutable and return a new dict.
Model-tier validation is derived from the same registry helpers used by the API.

Path format: dot-string with numeric indices for list elements.
  - ``"root"`` — the root WorkflowNode dict inside ``ast["root"]``
  - ``"root.children.0"`` — first child of root
  - ``"root.children.0.config.body"`` — body inside plan_and_execute config
"""

from __future__ import annotations

import copy
import uuid
from typing import Any

from deep_research.agent_designer.registry import model_tiers_payload

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BlockPathError(ValueError):
    """Invalid block path."""


class BlockMutationError(ValueError):
    """Mutation rejected by domain rules."""


# ---------------------------------------------------------------------------
# Valid model tiers
# ---------------------------------------------------------------------------


def _valid_tiers() -> frozenset[str]:
    """Return model tiers configured for this app deployment."""
    return frozenset(model_tiers_payload())

# Node types that carry children lists
_COMPOSITE_TYPES: frozenset[str] = frozenset(
    {"sequence", "parallel", "loop", "conditional", "plan_and_execute"}
)

# Patch keys that callers are allowed to supply to update_block
_ALLOWED_PATCH_KEYS: frozenset[str] = frozenset(
    {"label", "config", "error_handling", "budget_seconds"}
)

# Plan v2.1 PR-3 / codex CRITICAL-3 — when DESIGNER_DETERMINISTIC_BLUEPRINT
# is ON, ``update_block`` is allowed to patch only PROMPT-LEVEL fields
# under ``config``. Structural keys (``body``, ``evaluator``, ``children``,
# ``subtype``, ``type``, ``pools``, ``node_id``) become the deterministic
# blueprint builder's responsibility — the architect cannot mutate them
# via update_block, which closes the structural-bypass surface the codex
# review flagged. When the flag is OFF (PR-2 default), legacy semantics
# are preserved: any key the dict accepts is patchable.
_ALLOWED_CONFIG_PATCH_KEYS: frozenset[str] = frozenset(
    {
        "system_prompt",
        "user_prompt_template",
        "model_tier",
        "error_handling",
        "max_tool_calls",
    }
)
_FORBIDDEN_CONFIG_PATCH_KEYS: frozenset[str] = frozenset(
    {
        "body",
        "evaluator",
        "children",
        "subtype",
        "type",
        "pools",
        "node_id",
    }
)


# ---------------------------------------------------------------------------
# Internal path helpers
# ---------------------------------------------------------------------------


def _split_path(path: str) -> list[str | int]:
    """Split a dot-path string into a list of string/int segments.

    >>> _split_path("root.children.0.config.body")
    ['root', 'children', 0, 'config', 'body']
    """
    parts: list[str | int] = []
    for segment in path.split("."):
        try:
            parts.append(int(segment))
        except ValueError:
            parts.append(segment)
    return parts


def _get_at(ast: dict[str, Any], path: str) -> Any:
    """Return the value at *path* inside *ast*.

    The root path ``"root"`` returns ``ast["root"]``.

    Raises
    ------
    BlockPathError
        If any segment along the path is missing or the index is out of range.
    """
    segments = _split_path(path)
    current: Any = ast
    for i, seg in enumerate(segments):
        try:
            if isinstance(seg, int):
                if not isinstance(current, list):
                    raise BlockPathError(
                        f"Expected list at segment {i} of path '{path}', "
                        f"got {type(current).__name__}"
                    )
                current = current[seg]
            else:
                if not isinstance(current, dict):
                    raise BlockPathError(
                        f"Expected dict at segment {i} of path '{path}', "
                        f"got {type(current).__name__}"
                    )
                current = current[seg]
        except (KeyError, IndexError) as exc:
            raise BlockPathError(
                f"Path '{path}' not found: segment '{seg}' missing — {exc}"
            ) from exc
    return current


def _set_at(ast: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """Return a *new* ast with *value* set at *path* (immutable update).

    Deep-copies the full ast to guarantee immutability.  For small workflow
    ASTs this is acceptable per spec.

    Raises
    ------
    BlockPathError
        If any intermediate segment is missing.
    """
    new_ast: dict[str, Any] = copy.deepcopy(ast)
    segments = _split_path(path)
    current: Any = new_ast
    for seg in segments[:-1]:
        try:
            current = current[seg]
        except (KeyError, IndexError) as exc:
            raise BlockPathError(
                f"Path '{path}' not found at segment '{seg}': {exc}"
            ) from exc
    last = segments[-1]
    current[last] = value
    return new_ast


def _new_id() -> str:
    """Generate a fresh 8-char hex node id."""
    return uuid.uuid4().hex[:8]


def _is_descendant_path(ancestor: str, candidate: str) -> bool:
    """Return True if *candidate* is equal to or a descendant of *ancestor*."""
    if candidate == ancestor:
        return True
    return candidate.startswith(ancestor + ".")


def _default_condition() -> dict[str, Any]:
    return {"kind": "key_equals", "state_key": "intent", "value": "yes"}


# ---------------------------------------------------------------------------
# ID-based addressing (Option B) — accept node.id OR dot-notation path
# ---------------------------------------------------------------------------


def _collect_id_index(ast: dict[str, Any]) -> dict[str, str]:
    """Walk the AST and return a mapping of every node.id → indexed dot-path.

    Used by :func:`_resolve_node_ref` to support callers that pass a semantic
    node id (the natural choice for LLM-driven designers) instead of a brittle
    numeric path like ``root.children.1.children.0``.
    """
    out: dict[str, str] = {}

    def visit(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        node_id = node.get("id")
        if isinstance(node_id, str) and node_id:
            out.setdefault(node_id, path)
        for idx, child in enumerate(node.get("children") or []):
            visit(child, f"{path}.children.{idx}")
        config = node.get("config")
        if isinstance(config, dict):
            body = config.get("body")
            if isinstance(body, dict):
                visit(body, f"{path}.config.body")
            evaluator = config.get("evaluator")
            if isinstance(evaluator, dict):
                visit(evaluator, f"{path}.config.evaluator")

    root = ast.get("root")
    if isinstance(root, dict):
        visit(root, "root")
    return out


def _closest_id(ref: str, candidates: list[str]) -> str | None:
    """Return the closest candidate by lowercased substring overlap.

    Lightweight alternative to Levenshtein — sufficient for catching typos like
    ``"lane-fundamental"`` vs ``"lane-fundamentals"`` without pulling in a
    Levenshtein library. Returns ``None`` when nothing overlaps meaningfully.
    """
    if not ref or not candidates:
        return None
    lref = ref.lower()
    best: tuple[int, str] | None = None
    for cand in candidates:
        lcand = cand.lower()
        # Overlap score: longest common substring length, biased by prefix match.
        score = 0
        if lref == lcand:
            return cand
        if lref in lcand or lcand in lref:
            score = max(len(lref), len(lcand))
        else:
            # Crude n-gram overlap
            grams_ref = {lref[i : i + 3] for i in range(max(0, len(lref) - 2))}
            grams_cand = {lcand[i : i + 3] for i in range(max(0, len(lcand) - 2))}
            score = len(grams_ref & grams_cand)
        if score > 0 and (best is None or score > best[0]):
            best = (score, cand)
    return best[1] if best is not None else None


def _resolve_node_ref(ast: dict[str, Any], ref: str) -> str:
    """Resolve a user-supplied reference to an indexed dot-path.

    Accepts either:
    - A literal indexed dot-path (e.g. ``"root.children.1.children.0"``).
      Returned verbatim when it resolves.
    - A semantic node id (e.g. ``"lane-fundamentals"``). Mapped to its
      indexed path via :func:`_collect_id_index`.

    Raises ``BlockPathError`` with a structured hint listing available node
    ids when the reference resolves to neither. The hint is the part that
    teaches LLM designers what the right value should look like — see
    ``.omc/plans/`` for the rationale.
    """
    if not isinstance(ref, str) or not ref:
        raise BlockPathError("Empty path/ref supplied")
    # Fast path: a literal indexed path that resolves.
    try:
        _get_at(ast, ref)
        return ref
    except BlockPathError:
        pass
    # Try id lookup.
    id_index = _collect_id_index(ast)
    if ref in id_index:
        return id_index[ref]
    # Neither path nor id. Raise with a teaching hint.
    available = sorted(id_index.keys())
    hint_lines = [
        f"Path/id '{ref}' not found in AST.",
    ]
    if available:
        sample = available[:20]
        hint_lines.append(f"Available node ids: {sample}")
        suggestion = _closest_id(ref, available)
        if suggestion is not None:
            hint_lines.append(
                f"Did you mean '{suggestion}'? "
                f"(resolves to '{id_index[suggestion]}')"
            )
    else:
        hint_lines.append(
            "AST has no addressable nodes — did you call propose_workflow first?"
        )
    hint_lines.append(
        "Tip: pass the node 'id' directly (e.g. 'lane-fundamentals') OR a "
        "dot-notation indexed path (e.g. 'root.children.1.children.0')."
    )
    raise BlockPathError(" ".join(hint_lines))


# ---------------------------------------------------------------------------
# Public mutation primitives
# ---------------------------------------------------------------------------


def add_block(
    ast: dict[str, Any],
    parent_path: str,
    node_type: str,
    config: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], str]:
    """Append a new node to a composite node's ``children`` list.

    Parameters
    ----------
    ast:
        Current workflow AST dict (treated as immutable).
    parent_path:
        Dot-path to the parent node that will receive the new child.
        For ``plan_and_execute`` bodies use ``"…config.body"``; if ``body``
        is empty/``None`` or a non-sequence single node it is auto-wrapped in
        a sequence so that multiple children remain valid.
    node_type:
        The ``type`` field for the new node (e.g. ``"agent"``, ``"sequence"``).
    config:
        Free-form config dict for the new node.
    label:
        Human-readable label.

    Returns
    -------
    (new_ast, new_node_path)
        *new_ast* is the updated AST; *new_node_path* is the dot-path to the
        newly created node.
    """
    # Detect plan_and_execute body path: ends with "config.body"
    if parent_path.endswith("config.body"):
        return _add_to_plan_body(ast, parent_path, node_type, config, label)

    # Normal composite node — resolve either a semantic node id ("main") or
    # an indexed dot path ("root.children.1") to the parent node.
    parent_path = _resolve_node_ref(ast, parent_path)

    # Normal composite node — get or initialise children list
    try:
        parent_node = _get_at(ast, parent_path)
    except BlockPathError:
        raise

    if not isinstance(parent_node, dict):
        raise BlockPathError(
            f"Path '{parent_path}' does not point to a node dict"
        )

    children_path = parent_path + ".children"
    try:
        children = _get_at(ast, children_path)
        if not isinstance(children, list):
            raise BlockMutationError(
                f"Node at '{parent_path}' has a non-list 'children' field"
            )
    except BlockPathError:
        # children key doesn't exist yet — we'll create it
        children = []

    new_node: dict[str, Any] = {
        "id": _new_id(),
        "type": node_type,
        "label": label,
        "config": copy.deepcopy(config),
        "children": [],
    }

    new_children = list(children) + [new_node]
    new_ast = _set_at(ast, children_path, new_children)
    if parent_node.get("type") == "conditional":
        conditions = list(parent_node.get("config", {}).get("conditions", []))
        config = {
            **parent_node.get("config", {}),
            "conditions": conditions + [_default_condition()],
            "default_branch": len(new_children) - 1,
        }
        new_ast = _set_at(new_ast, parent_path + ".config", config)
    new_node_path = f"{children_path}.{len(children)}"
    return new_ast, new_node_path


def _add_to_plan_body(
    ast: dict[str, Any],
    body_path: str,
    node_type: str,
    config: dict[str, Any],
    label: str,
) -> tuple[dict[str, Any], str]:
    """Handle add_block for plan_and_execute body paths."""
    new_node: dict[str, Any] = {
        "id": _new_id(),
        "type": node_type,
        "label": label,
        "config": copy.deepcopy(config),
        "children": [],
    }

    # Resolve current body value (may be None or missing)
    try:
        current_body = _get_at(ast, body_path)
    except BlockPathError:
        current_body = None

    if current_body is None:
        # Empty body — set directly to the single new node
        new_ast = _set_at(ast, body_path, new_node)
        new_node_path = body_path
        return new_ast, new_node_path

    if isinstance(current_body, dict):
        body_type = current_body.get("type", "")
        if body_type == "sequence":
            # Already a sequence — append to its children
            existing_children: list[Any] = current_body.get("children", [])
            new_children = list(existing_children) + [new_node]
            new_ast = _set_at(
                ast, body_path + ".children", new_children
            )
            new_node_path = (
                f"{body_path}.children.{len(existing_children)}"
            )
            return new_ast, new_node_path
        else:
            # Single non-sequence node — wrap both in a new sequence
            wrapper_id = _new_id()
            wrapper: dict[str, Any] = {
                "id": wrapper_id,
                "type": "sequence",
                "label": "body",
                "config": {},
                "children": [current_body, new_node],
            }
            new_ast = _set_at(ast, body_path, wrapper)
            new_node_path = f"{body_path}.children.1"
            return new_ast, new_node_path

    raise BlockMutationError(
        f"Unexpected body value at '{body_path}': {type(current_body)}"
    )


def update_block(
    ast: dict[str, Any],
    path: str,
    patches: dict[str, Any],
) -> dict[str, Any]:
    """Return new AST with *patches* shallow-merged into the node at *path*.

    Allowed patch keys: ``label``, ``config``, ``error_handling``,
    ``budget_seconds``.  ``children`` and ``type`` are not patchable.

    Raises
    ------
    BlockPathError
        If *path* is invalid.
    BlockMutationError
        If *patches* contains ``"type"`` or ``"children"``.
    """
    # Validate patch keys
    forbidden = set(patches.keys()) - _ALLOWED_PATCH_KEYS
    if "type" in patches:
        raise BlockMutationError(
            "'type' is not patchable — it would invalidate the config schema"
        )
    if "children" in patches:
        raise BlockMutationError(
            "'children' is not patchable — use add/move/delete instead"
        )
    if forbidden:
        raise BlockMutationError(
            f"Disallowed patch keys: {sorted(forbidden)}"
        )

    # Plan v2.1 PR-3 — config-level allow-list (codex CRITICAL-3 fix).
    # When DESIGNER_DETERMINISTIC_BLUEPRINT is ON, the architect must
    # restrict config patches to prompt-only fields. Structural keys
    # (``body``, ``evaluator``, etc.) belong to the deterministic
    # blueprint builder; mutating them via update_block bypasses the
    # immutability fingerprint check that parse_architect_ast enforces
    # downstream. Flag OFF preserves legacy semantics so PR-2 tests and
    # the legacy architect flow keep working untouched.
    config_patch = patches.get("config")
    if isinstance(config_patch, dict):
        from deep_research.agent_designer.blueprint import (
            is_deterministic_blueprint_enabled,
        )

        if is_deterministic_blueprint_enabled():
            forbidden_config = (
                set(config_patch.keys()) & _FORBIDDEN_CONFIG_PATCH_KEYS
            )
            if forbidden_config:
                raise BlockMutationError(
                    f"Disallowed config patch keys: "
                    f"{sorted(forbidden_config)} (structural keys cannot be "
                    f"mutated when DESIGNER_DETERMINISTIC_BLUEPRINT is ON; "
                    f"use request_signature_revision instead)"
                )
            unknown_config = (
                set(config_patch.keys()) - _ALLOWED_CONFIG_PATCH_KEYS
            )
            if unknown_config:
                raise BlockMutationError(
                    f"Disallowed config patch keys: "
                    f"{sorted(unknown_config)} (allowed: "
                    f"{sorted(_ALLOWED_CONFIG_PATCH_KEYS)})"
                )

    # Resolve the user-supplied reference (id OR dot-path) to an indexed path.
    path = _resolve_node_ref(ast, path)

    # Verify path exists
    node = _get_at(ast, path)
    if not isinstance(node, dict):
        raise BlockPathError(f"Path '{path}' does not point to a node dict")

    new_ast = copy.deepcopy(ast)
    target = _get_at(new_ast, path)
    for key, val in patches.items():
        # Deep-merge for dict-valued patch keys (specifically `config`) so
        # callers can patch a single subfield without having to re-emit the
        # entire config. Otherwise patches={'config': {'system_prompt': X}}
        # would wipe out subtype/model_tier/tools and break validation.
        if (
            key == "config"
            and isinstance(val, dict)
            and isinstance(target.get(key), dict)
        ):
            existing_config = target[key]
            merged_config: dict[str, Any] = {**existing_config}
            for cfg_key, cfg_val in val.items():
                merged_config[cfg_key] = copy.deepcopy(cfg_val)
            target[key] = merged_config
        else:
            target[key] = copy.deepcopy(val)
    return new_ast


# Patch keys that callers are allowed to supply to update_pool. The framework
# defines pool fields as {name, dedup_key, max_items}; ``name`` is the lookup
# key and is not patchable, so only the latter two are patch-allowed.
_ALLOWED_POOL_PATCH_KEYS: frozenset[str] = frozenset({"dedup_key", "max_items"})


def update_pool(
    ast: dict[str, Any],
    pool_name: str,
    patches: dict[str, Any],
) -> dict[str, Any]:
    """Return a new AST with the top-level ``pools[*]`` entry patched.

    Allowed patch keys: ``dedup_key``, ``max_items``. ``name`` is the lookup
    key and not patchable; pools that don't exist by ``name`` raise.

    Raises
    ------
    BlockMutationError
        If *patches* contains a forbidden key or if no pool with the given
        ``pool_name`` exists.
    """
    if not isinstance(pool_name, str) or not pool_name:
        raise BlockMutationError("update_pool requires non-empty pool_name")
    if not isinstance(patches, dict) or not patches:
        raise BlockMutationError("update_pool requires non-empty patches dict")
    forbidden = set(patches.keys()) - _ALLOWED_POOL_PATCH_KEYS
    if forbidden:
        raise BlockMutationError(
            f"Disallowed pool patch keys: {sorted(forbidden)} "
            f"(allowed: {sorted(_ALLOWED_POOL_PATCH_KEYS)})"
        )
    pools = ast.get("pools")
    if not isinstance(pools, list):
        raise BlockMutationError(
            "AST has no top-level 'pools' list; cannot update pool"
        )
    target_index = None
    for i, pool in enumerate(pools):
        if isinstance(pool, dict) and pool.get("name") == pool_name:
            target_index = i
            break
    if target_index is None:
        existing_names = [
            p.get("name") for p in pools if isinstance(p, dict)
        ]
        raise BlockMutationError(
            f"No pool with name {pool_name!r}; existing: {existing_names}"
        )
    new_ast = copy.deepcopy(ast)
    new_pool = dict(new_ast["pools"][target_index])
    for key, val in patches.items():
        new_pool[key] = copy.deepcopy(val)
    new_ast["pools"][target_index] = new_pool
    return new_ast


def delete_block(ast: dict[str, Any], path: str) -> dict[str, Any]:
    """Return new AST with the node at *path* removed.

    Raises
    ------
    BlockMutationError
        If *path* == ``"root"`` (cannot delete the root node).
    BlockPathError
        If *path* is invalid.
    """
    if path == "root":
        raise BlockMutationError("Cannot delete the root node")

    # Resolve id-or-path before any path-shape assertions below.
    path = _resolve_node_ref(ast, path)

    # Verify path exists first
    _get_at(ast, path)

    segments = _split_path(path)
    # The last segment must be a list index — node lives inside a children list
    last = segments[-1]
    if not isinstance(last, int):
        raise BlockPathError(
            f"Path '{path}' does not point to a list element (last segment "
            f"'{last}' is not an integer index)"
        )

    parent_path = ".".join(str(s) for s in segments[:-1])
    parent_node_path = ".".join(str(s) for s in segments[:-2])
    parent_node = _get_at(ast, parent_node_path) if parent_node_path else None
    new_ast = copy.deepcopy(ast)
    parent_list = _get_at(new_ast, parent_path)
    if not isinstance(parent_list, list):
        raise BlockPathError(
            f"Parent at '{parent_path}' is not a list"
        )
    del parent_list[last]
    if isinstance(parent_node, dict) and parent_node.get("type") == "conditional":
        conditions = list(parent_node.get("config", {}).get("conditions", []))
        if last < len(conditions):
            del conditions[last]
        max_conditions = max(0, len(parent_list) - 1)
        previous_default = parent_node.get("config", {}).get("default_branch", len(parent_list))
        default_branch = previous_default
        if isinstance(default_branch, int) and default_branch > last:
            default_branch -= 1
        if not isinstance(default_branch, int):
            default_branch = len(parent_list) - 1
        default_branch = max(0, min(max(0, len(parent_list) - 1), default_branch))
        config = {
            **parent_node.get("config", {}),
            "conditions": conditions[:max_conditions],
            "default_branch": default_branch,
        }
        new_ast = _set_at(new_ast, parent_node_path + ".config", config)
    return new_ast


def move_block(
    ast: dict[str, Any],
    from_path: str,
    to_path: str,
    position: int | None = None,
) -> dict[str, Any]:
    """Move node at *from_path* into the children of the node at *to_path*.

    Parameters
    ----------
    ast:
        Current workflow AST dict.
    from_path:
        Dot-path to the node being moved.
    to_path:
        Dot-path to the destination composite node (must have ``children``).
    position:
        Index in the destination children list; ``None`` means append.

    Raises
    ------
    BlockMutationError
        If *to_path* is a descendant of *from_path* (would create a cycle).
    BlockPathError
        If either path is invalid.
    """
    # Accept id OR dot-path for both source and destination.
    from_path = _resolve_node_ref(ast, from_path)
    to_path = _resolve_node_ref(ast, to_path)

    # Validate paths exist
    node_to_move = copy.deepcopy(_get_at(ast, from_path))
    dest_node = _get_at(ast, to_path)

    if not isinstance(dest_node, dict):
        raise BlockPathError(
            f"Destination path '{to_path}' does not point to a node dict"
        )

    # Cycle check: to_path must not be equal to or descend from from_path
    if _is_descendant_path(from_path, to_path):
        raise BlockMutationError(
            f"Cannot move '{from_path}' into '{to_path}': "
            "destination is a descendant of the source (cycle)"
        )

    # Step 1: delete from source
    new_ast = delete_block(ast, from_path)

    # Step 2: recompute to_path after deletion (index shifts if needed)
    adjusted_to_path = _adjust_path_after_deletion(to_path, from_path)

    # Step 3: insert at destination
    dest_children_path = adjusted_to_path + ".children"
    try:
        dest_children = _get_at(new_ast, dest_children_path)
        if not isinstance(dest_children, list):
            dest_children = []
    except BlockPathError:
        dest_children = []

    new_children = list(dest_children)
    if position is None:
        new_children.append(node_to_move)
    else:
        new_children.insert(position, node_to_move)

    new_ast = _set_at(new_ast, dest_children_path, new_children)
    return new_ast


def _adjust_path_after_deletion(
    path_to_adjust: str, deleted_path: str
) -> str:
    """Adjust *path_to_adjust* after *deleted_path* was removed from a list.

    If the deleted node was a sibling with a smaller index in the same parent
    list, the index in *path_to_adjust* needs to be decremented by 1.
    """
    deleted_segs = _split_path(deleted_path)
    adjust_segs = list(_split_path(path_to_adjust))

    # They must share the same parent prefix for adjustment to apply
    if len(deleted_segs) == 0 or not isinstance(deleted_segs[-1], int):
        return path_to_adjust

    deleted_parent = deleted_segs[:-1]
    deleted_index = int(deleted_segs[-1])

    # Check if path_to_adjust has the same parent prefix at the right depth
    depth = len(deleted_parent)
    if len(adjust_segs) > depth and adjust_segs[:depth] == deleted_parent:
        seg_at_depth = adjust_segs[depth]
        if isinstance(seg_at_depth, int) and seg_at_depth > deleted_index:
            adjust_segs[depth] = seg_at_depth - 1

    return ".".join(str(s) for s in adjust_segs)


def declare_tool(
    ast: dict[str, Any],
    kind: str,
    name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Append a new tool declaration to ``ast['tools']``.

    Raises
    ------
    BlockMutationError
        If a tool with the same *name* already exists.
    """
    existing_tools: list[dict[str, Any]] = ast.get("tools", [])
    for tool in existing_tools:
        if tool.get("name") == name:
            raise BlockMutationError(
                f"Tool '{name}' is already declared in this workflow"
            )

    new_tool: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "config": copy.deepcopy(config),
        "description": "",
    }
    new_ast = copy.deepcopy(ast)
    new_ast.setdefault("tools", [])
    new_ast["tools"] = list(new_ast["tools"]) + [new_tool]
    return new_ast


def remove_tool(ast: dict[str, Any], name: str) -> dict[str, Any]:
    """Remove tool *name* from ``ast['tools']`` and node bindings.

    No-op (returns a shallow-copied ast) if the tool is not present.
    """
    new_ast = copy.deepcopy(ast)
    new_ast["tools"] = [t for t in new_ast.get("tools", []) if t.get("name") != name]

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        config = node.get("config")
        if isinstance(config, dict):
            tools = config.get("tools")
            if isinstance(tools, list):
                config["tools"] = [
                    item for item in tools if not (isinstance(item, str) and item == name)
                ]
            body = config.get("body")
            if isinstance(body, dict):
                visit(body)
            for nested_key in ("planner", "evaluator"):
                nested = config.get(nested_key)
                if isinstance(nested, dict):
                    visit({"config": nested})
        for child in node.get("children") or []:
            visit(child)

    visit(new_ast.get("root"))
    return new_ast


def bind_tool_to_block(
    ast: dict[str, Any],
    node_path: str,
    tool_name: str,
) -> dict[str, Any]:
    """Append *tool_name* to an agent node's ``config.tools`` list.

    Raises
    ------
    BlockMutationError
        If the node at *node_path* is not an agent node, or if *tool_name*
        is not declared in ``ast['tools']``.
    BlockPathError
        If *node_path* is invalid.
    """
    node_path = _resolve_node_ref(ast, node_path)
    node = _get_at(ast, node_path)
    if not isinstance(node, dict):
        raise BlockPathError(
            f"Path '{node_path}' does not point to a node dict"
        )

    if node.get("type") != "agent":
        raise BlockMutationError(
            f"Node at '{node_path}' is of type '{node.get('type')}', "
            "not 'agent'; only agent nodes can have tools bound"
        )

    declared_names = {t.get("name") for t in ast.get("tools", [])}
    if tool_name not in declared_names:
        raise BlockMutationError(
            f"Tool '{tool_name}' is not declared in ast['tools']; "
            "declare it first with declare_tool()"
        )

    current_tools: list[str] = node.get("config", {}).get("tools", [])
    if tool_name in current_tools:
        # Already bound — return unchanged copy
        return copy.deepcopy(ast)

    new_ast = copy.deepcopy(ast)
    target = _get_at(new_ast, node_path)
    target.setdefault("config", {})
    target["config"].setdefault("tools", [])
    target["config"]["tools"] = list(target["config"]["tools"]) + [tool_name]
    return new_ast


def set_model_tier(
    ast: dict[str, Any],
    node_path: str,
    tier: str,
) -> dict[str, Any]:
    """Set ``config.model_tier`` on an agent node.

    Raises
    ------
    BlockMutationError
        If the node is not an agent node or *tier* is not valid.
    BlockPathError
        If *node_path* is invalid.
    """
    node_path = _resolve_node_ref(ast, node_path)
    node = _get_at(ast, node_path)
    if not isinstance(node, dict):
        raise BlockPathError(
            f"Path '{node_path}' does not point to a node dict"
        )

    if node.get("type") != "agent":
        raise BlockMutationError(
            f"Node at '{node_path}' is of type '{node.get('type')}', "
            "not 'agent'; only agent nodes have a model_tier"
        )

    valid_tiers = _valid_tiers()
    if tier not in valid_tiers:
        raise BlockMutationError(
            f"Invalid model tier '{tier}'; must be one of {sorted(valid_tiers)}"
        )

    new_ast = copy.deepcopy(ast)
    target = _get_at(new_ast, node_path)
    target.setdefault("config", {})
    target["config"]["model_tier"] = tier
    return new_ast
