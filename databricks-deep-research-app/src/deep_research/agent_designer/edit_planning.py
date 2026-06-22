"""Edit-lane planning: scope classification, minimality guard, and the
topology-switch helpers (persisted-signature retrieval + prompt carry-over).

This is the brain of the level-aware EDIT path. ``classify_edit_scope`` is the
ONE LLM call that decides *at which level* a change must be made; everything
else here is pure and unit-testable:

* :class:`EditScope` — the classifier's structured output (route + levels +
  extractive target node ids + optional topology ``delta``).
* :func:`edit_diff_guard` — a server-side, field-level minimality check that
  flags out-of-scope node changes/additions/removals (never raises; surfaced
  to the user, never silent).
* :func:`stored_signature` / :func:`apply_signature_delta` /
  :func:`carry_over_prompts` — the topology lane: retrieve the signature that
  built the workflow (persisted by ``build_blueprint``), apply the requested
  delta, and best-effort carry the old per-node prompts onto the new scaffold.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from deep_research.agent_designer.ast_introspection import (
    config_of,
    iter_agent_nodes,
    iter_all_nodes,
)
from deep_research.services.llm.types import ModelTier

logger = logging.getLogger(__name__)

EditRoute = Literal["surgical", "topology", "rebuild", "unsupported"]
EditLevel = Literal["property", "prompt", "tool", "node"]

# Config fields each level may touch (used to DERIVE the allow-list; the guard
# enforces node SCOPE, these document intent + feed the FE summary).
_PROPERTY_FIELDS: frozenset[str] = frozenset(
    {
        "model_tier",
        "max_tool_calls",
        "error_handling",
        "budget_seconds",
        "output_format",
        "provider",
        "model",
        "model_family",
        "timeout_seconds",
        "max_results",
        "resolve_redirects",
    }
)
_PROMPT_FIELDS: frozenset[str] = frozenset({"system_prompt", "user_prompt_template"})

_MAX_INTENT_CHARS = 4000


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EditAllowList(BaseModel):
    """The minimality contract derived from an :class:`EditScope`.

    ``node_ids`` empty ⇒ the guard runs in advisory mode (count checks only),
    because the classifier did not pin specific targets.
    """

    model_config = ConfigDict(extra="ignore")

    node_ids: list[str] = Field(default_factory=list)
    allowed_fields: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    max_added: int = 0
    max_removed: int = 0


class EditScope(BaseModel):
    """Structured output of the edit-scope classifier (ONE fast LLM call).

    ``target_node_ids`` MUST be copied verbatim from the provided AST summary
    (extractive — never invented). ``delta`` is only meaningful for the
    ``topology`` route and carries the changed TaskSignature axes.
    """

    model_config = ConfigDict(extra="ignore")

    route: EditRoute = "surgical"
    levels: list[EditLevel] = Field(default_factory=list)
    target_node_ids: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    delta: dict[str, Any] = Field(default_factory=dict)
    change_summary: str = ""
    unsupported_reason: str | None = None

    def to_allow_list(self) -> EditAllowList:
        fields: set[str] = set()
        if "property" in self.levels:
            fields |= _PROPERTY_FIELDS
        if "prompt" in self.levels:
            fields |= _PROMPT_FIELDS
        if "tool" in self.levels:
            fields |= {"tools"}
        max_added = max_removed = 0
        if "node" in self.levels:
            # Generous bound — the guard FLAGS (never hard-blocks), and the user
            # approves the card. Scales with the number of named targets.
            max_added = len(self.target_node_ids) + 2
            max_removed = len(self.target_node_ids) + 2
        return EditAllowList(
            node_ids=list(self.target_node_ids),
            allowed_fields=sorted(fields),
            tool_names=list(self.tool_names),
            max_added=max_added,
            max_removed=max_removed,
        )


class GuardReport(BaseModel):
    """Result of :func:`edit_diff_guard`. ``ok`` is False when the proposed AST
    changed more than the EditScope authorized."""

    model_config = ConfigDict(extra="ignore")

    ok: bool = True
    violations: list[str] = Field(default_factory=list)
    added_node_ids: list[str] = Field(default_factory=list)
    removed_node_ids: list[str] = Field(default_factory=list)
    changed_node_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Scope classification (the one LLM call)
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = (
    "You are the Agent Designer EDIT-scope classifier. The user is editing an "
    "EXISTING workflow. Decide the MINIMAL level of change and return it as a "
    "structured object. Do not design the workflow; another agent applies the "
    "change. Be conservative: prefer the smallest route that satisfies the "
    "request."
)


def _classifier_prompt(intent: str, ast_summary: Any) -> str:
    summary_text = (
        ast_summary
        if isinstance(ast_summary, str)
        else json.dumps(ast_summary, ensure_ascii=True, indent=2, default=str)
    )
    return (
        "Classify this edit request against the workflow below.\n\n"
        "ROUTES (pick exactly one):\n"
        "- surgical: modify the EXISTING workflow in place — change a property "
        "(model tier, output format, search provider, limits), reword a prompt, "
        "add/remove/rebind a tool, or add/remove a SINGLE lane or candidate "
        "(cloned from an existing one). THIS IS THE DEFAULT for almost every "
        "edit, including changing how many best-of-N candidates or parallel "
        "lanes there are.\n"
        "- topology: a FUNDAMENTAL change to the workflow's shape — switching "
        "between patterns (e.g. best_of_n -> plan_and_execute, parallel_lanes -> "
        "router). Only when the requested shape cannot be reached by adding/"
        "removing/editing individual nodes. Put the changed signature axes in "
        "'delta' (e.g. {\"coordination_pattern\": null, "
        "\"step_dependencies_present\": true}).\n"
        "- rebuild: the user explicitly wants to start over / redesign from "
        "scratch / discard the current workflow.\n"
        "- unsupported: the request is not an actionable workflow edit (a "
        "question, impossible, or too ambiguous). Set 'unsupported_reason'.\n\n"
        "LEVELS (list all that apply for a surgical edit): property, prompt, "
        "tool, node.\n\n"
        "target_node_ids: copy the EXACT ids (verbatim) of the nodes this edit "
        "touches, from the summary below. Never invent ids. Leave empty only if "
        "truly workflow-wide.\n"
        "tool_names: tool kinds/names to add, remove, or bind (e.g. 'compute').\n"
        "change_summary: one user-facing sentence describing the intended change.\n\n"
        f"USER REQUEST:\n{intent[:_MAX_INTENT_CHARS]}\n\n"
        f"CURRENT WORKFLOW (node ids/types/subtypes/tools):\n{summary_text}\n"
    )


def _coerce_edit_scope(value: Any) -> EditScope | None:
    if isinstance(value, EditScope):
        return value
    if isinstance(value, dict):
        try:
            return EditScope.model_validate(value)
        except Exception:  # noqa: BLE001 - advisory; fall back
            return None
    if isinstance(value, str) and value.strip():
        try:
            return EditScope.model_validate_json(value)
        except Exception:  # noqa: BLE001
            return None
    return None


async def classify_edit_scope(
    *,
    llm: Any,
    intent: str,
    ast_summary: Any,
) -> EditScope:
    """Classify an edit request into a route + level + extractive targets.

    Fail-SAFE (never the original bug): on any classifier failure we return a
    ``surgical`` scope so the edit lane attempts a structure-preserving minimal
    edit rather than silently rebuilding. The edit agent's never-silent
    finalize still surfaces a no-op if nothing could be changed.
    """
    if llm is None or not callable(getattr(llm, "complete", None)):
        return EditScope(route="surgical", change_summary="")
    try:
        response = await llm.complete(
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user", "content": _classifier_prompt(intent, ast_summary)},
            ],
            tier=ModelTier.SIMPLE,
            temperature=0,
            max_tokens=900,
            structured_output=EditScope,
        )
    except Exception as exc:  # noqa: BLE001 - classification is best-effort
        logger.warning("DESIGNER_EDIT_SCOPE_CLASSIFY_FAILED: %s", exc)
        return EditScope(
            route="surgical",
            change_summary="(scope classification unavailable; attempting a minimal edit)",
        )
    scope = _coerce_edit_scope(
        getattr(response, "structured", None)
    ) or _coerce_edit_scope(getattr(response, "content", None))
    if scope is None:
        return EditScope(route="surgical", change_summary="")
    return scope


# ---------------------------------------------------------------------------
# Minimality guard (server-side, field/scope level)
# ---------------------------------------------------------------------------


def _index_nodes(ast: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = ast.get("root") if isinstance(ast, dict) else None
    out: dict[str, dict[str, Any]] = {}
    if isinstance(root, dict):
        for node in iter_all_nodes(root):
            node_id = node.get("id")
            if isinstance(node_id, str) and node_id:
                out.setdefault(node_id, node)
    return out


def _node_fingerprint(node: dict[str, Any]) -> str:
    """Stable projection of a node's OWN fields (type/label/config minus the
    structural-recursive keys, which are accounted for by add/remove)."""
    config = node.get("config")
    own_config = (
        {k: v for k, v in config.items() if k not in {"body", "evaluator", "children"}}
        if isinstance(config, dict)
        else {}
    )
    return json.dumps(
        {"type": node.get("type"), "label": node.get("label"), "config": own_config},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def edit_diff_guard(
    before_ast: dict[str, Any],
    after_ast: dict[str, Any],
    allow_list: EditAllowList,
) -> GuardReport:
    """Compare *before*/*after* and flag changes beyond what *allow_list*
    authorizes. Pure + total (never raises). The minimality contract:

    * node count may grow/shrink only within ``max_added``/``max_removed``;
    * when ``node_ids`` is non-empty, ONLY those nodes may change (catches the
      ReAct over-edit failure mode — touching unrelated nodes).

    Advisory when ``node_ids`` is empty (count checks only). Violations are
    surfaced to the user (never silent); the caller may also retry once.
    """
    before = _index_nodes(before_ast)
    after = _index_nodes(after_ast)
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(
        nid
        for nid in (set(before) & set(after))
        if _node_fingerprint(before[nid]) != _node_fingerprint(after[nid])
    )

    violations: list[str] = []
    if len(added) > allow_list.max_added:
        violations.append(
            f"added {len(added)} node(s) beyond the {allow_list.max_added} "
            f"authorized: {added}"
        )
    if len(removed) > allow_list.max_removed:
        violations.append(
            f"removed {len(removed)} node(s) beyond the {allow_list.max_removed} "
            f"authorized: {removed}"
        )
    if allow_list.node_ids:
        allowed = set(allow_list.node_ids)
        stray = [nid for nid in changed if nid not in allowed]
        if stray:
            violations.append(
                f"changed node(s) outside the requested scope "
                f"{sorted(allowed)}: {stray}"
            )

    return GuardReport(
        ok=not violations,
        violations=violations,
        added_node_ids=added,
        removed_node_ids=removed,
        changed_node_ids=changed,
    )


# ---------------------------------------------------------------------------
# Topology lane: persisted signature + best-effort prompt carry-over
# ---------------------------------------------------------------------------


def stored_signature(current_ast: Any) -> dict[str, Any] | None:
    """Return the TaskSignature persisted by ``build_blueprint`` (``ast
    ['designer_signature']``), or None for legacy ASTs built before persistence
    — the caller then degrades a topology edit to an explicit rebuild."""
    if not isinstance(current_ast, dict):
        return None
    sig = current_ast.get("designer_signature")
    return sig if isinstance(sig, dict) and sig else None


def apply_signature_delta(
    signature: dict[str, Any], delta: dict[str, Any] | None
) -> dict[str, Any]:
    """Shallow-merge *delta* (the requested structural change, as TaskSignature
    axes) onto the persisted signature. The classifier emits the axis changes
    that drive ``select_topology`` (e.g. a ``coordination_pattern`` flip)."""
    out = dict(signature)
    for key, val in (delta or {}).items():
        out[key] = val
    return out


_CARRY_FIELDS = ("system_prompt", "user_prompt_template", "model_tier")


def carry_over_prompts(
    old_ast: dict[str, Any], new_blueprint: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Best-effort copy of authored prompts from *old_ast* onto *new_blueprint*.

    Matches agent nodes by (subtype, ordinal-within-subtype) — a deliberately
    simple, topology-agnostic heuristic. Returns ``(node_patches, regenerated)``
    where ``node_patches`` is keyed by NEW node id (resolved by
    ``_apply_architect_patches``) and ``regenerated`` lists the new roles that
    had no old counterpart (surfaced to the user — tool/structural carry-over is
    intentionally out of scope here; prompts only).
    """

    def _by_subtype(ast: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        root = ast.get("root")
        if isinstance(root, dict):
            for node in iter_agent_nodes(root):
                cfg = config_of(node)
                subtype = str(cfg.get("subtype") or node.get("type") or "agent")
                groups.setdefault(subtype, []).append(node)
        return groups

    old_groups = _by_subtype(old_ast)
    patches: dict[str, dict[str, Any]] = {}
    regenerated: list[str] = []

    for subtype, new_nodes in _by_subtype(new_blueprint).items():
        old_nodes = old_groups.get(subtype, [])
        for ordinal, new_node in enumerate(new_nodes):
            new_id = str(new_node.get("id") or "")
            if not new_id:
                continue
            if ordinal < len(old_nodes):
                old_cfg = config_of(old_nodes[ordinal])
                patch = {f: old_cfg[f] for f in _CARRY_FIELDS if old_cfg.get(f)}
                if patch:
                    patches[new_id] = patch
                    continue
            regenerated.append(f"{subtype}#{ordinal}")

    return patches, regenerated


__all__ = [
    "EditRoute",
    "EditLevel",
    "EditScope",
    "EditAllowList",
    "GuardReport",
    "classify_edit_scope",
    "edit_diff_guard",
    "stored_signature",
    "apply_signature_delta",
    "carry_over_prompts",
]
