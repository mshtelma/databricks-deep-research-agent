"""Stateless chat session orchestrator for the Agent Designer.

Receives {messages, current_ast} per turn, runs an LLM with the CHAT_TOOLS,
dispatches each tool-call to the matching mutation primitive, validates the
resulting AST via load_workflow_from_dict, and yields SSE events.

Stateless invariant: no per-session storage on the orchestrator. The same
{messages, current_ast} input always produces the same output sequence
(modulo nondeterminism from the LLM).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from deep_research.agent_designer import mutations
from deep_research.agent_designer.assets import (
    DesignerAsset,
    asset_context_payload,
    assets_from_ast,
    inspect_assets,
    normalize_assets,
    recommend_tools_for_assets,
    resolve_default_table_warehouse_id,
)
from deep_research.agent_designer.ast_introspection import config_of, iter_all_nodes
from deep_research.agent_designer.ast_normalizer import normalize_ast
from deep_research.agent_designer.designer_architect import (
    WorkflowDesignBrief,
    designer_system_prompt,
)
from deep_research.agent_designer.discovery import (
    DesignerDiscoveryAdapter,
    SourceKind,
    _workspace_client_for_user_token,
)
from deep_research.agent_designer.edit_planning import (
    EditScope,
    apply_signature_delta,
    carry_over_prompts,
    classify_edit_scope,
    edit_diff_guard,
    stored_signature,
)
from deep_research.agent_designer.prompt_grounding import (
    ground_prompt,
    prompt_grounding_sse_result,
    sanitized_prompt_grounding_summary,
)
from deep_research.agent_designer.registry import (
    model_tiers_payload as _model_tiers_payload,
)
from deep_research.agent_designer.registry import (
    node_types_payload as _node_types_payload,
)
from deep_research.agent_designer.registry import (
    query_modes_payload as _query_modes_payload,
)
from deep_research.agent_designer.registry import (
    research_depths_payload as _research_depths_payload,
)
from deep_research.agent_designer.registry import (
    source_kinds_payload as _source_kinds_payload,
)
from deep_research.agent_designer.registry import (
    tool_kinds_payload as _tool_kinds_payload,
)
from deep_research.agent_designer.tool_contract import (
    extract_resource_semantics_structured,
    project_resolved_tool_contract,
    resolved_tool_contract_sse_result,
    resource_semantics_summary,
    sanitized_resolved_tool_contract_summary,
)
from deep_research.agent_designer.tools import (
    AddBlockArgs,
    BindToolArgs,
    DeclareToolArgs,
    DeleteBlockArgs,
    DiscoverSourcesArgs,
    InspectAssetsArgs,
    MoveBlockArgs,
    ProposeWorkflowArgs,
    RecommendToolsForAssetsArgs,
    RemoveToolArgs,
    SetModelTierArgs,
    UpdateBlockArgs,
    parse_tool_args,
)
from deep_research.agent_designer.workflow_builder import (
    build_direct_workflow,
    build_web_research_workflow,
)
from deep_research.core.auth import get_service_principal_workspace_client
from deep_research.observability.agent_designer_metrics import (
    log_chat_mutation,
    log_run_principal,
)

from .sse_events import (
    DesignerSSEEvent as DesignerSSEEvent,
)
from .sse_events import (
    DoneEvent as DoneEvent,
)
from .sse_events import (
    ErrorEvent as ErrorEvent,
)
from .sse_events import (
    MessageEvent as MessageEvent,
)
from .sse_events import (
    MutationProposedEvent as MutationProposedEvent,
)
from .sse_events import (
    ProgressEvent as ProgressEvent,
)
from .sse_events import (
    ToolCallEvent as ToolCallEvent,
)
from .sse_events import (
    ToolResultEvent as ToolResultEvent,
)

# SSE event types — moved to sse_events.py in the W5a refactor. Kept as
# re-exports so existing callers (tests, route handler) continue to
# import them from this module unchanged.
from .sse_events import (
    _SSEBase as _SSEBase,
)

# Public helpers — moved to validation_helpers.py in the W6 refactor. Kept
# as re-exports so existing callers (tests/complex/test_scaffold_and_run.py
# imports these by name from this module) continue to resolve.
from .validation_helpers import (
    _node_count as _node_count,
)
from .validation_helpers import (
    _quality_advice as _quality_advice,
)
from .validation_helpers import (
    _validate_ast as _validate_ast,
)
from .validation_helpers import (
    _validation_error as _validation_error,
)

logger = logging.getLogger(__name__)

# ---- Limits ----

# Frontend persists/resends up to designerChatPersistence.MAX_MESSAGES (60).
# Keep this aligned so a normal session never exceeds the cap; any overflow is
# trimmed oldest-first by _trim_conversation (graceful), NOT hard-rejected, so a
# Designer chat can never wedge on a long/heavily-retried conversation.
MAX_MESSAGES = 60
# Byte caps are env-overridable (per-workspace ops headroom) with higher defaults
# than the original 100 KB / 200 KB: a real corpus agent's AST is already ~71 KB
# and an 8-lane Best-of-N edit grows it further. Kept *modest* on purpose — the
# transcript becomes LLM context, so an oversized cap just defers failure to a
# context-window overflow downstream. Mirrors the AGENT_DESIGNER_YAML_MAX_BYTES
# env knob used by the import-yaml endpoint.
MAX_AST_BYTES = int(os.environ.get("DESIGNER_CHAT_MAX_AST_BYTES", str(256 * 1024)))
MAX_PAYLOAD_BYTES = int(
    os.environ.get("DESIGNER_CHAT_MAX_PAYLOAD_BYTES", str(512 * 1024))
)
MAX_DESIGNER_TOOL_ROUNDS = 4


def _payload_bytes(obj: Any) -> int:
    """UTF-8 byte length of a JSON-encoded payload (size-limit accounting)."""
    return len(json.dumps(obj, default=str).encode("utf-8"))


def _drop_leading_orphan_tools(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Never let a trimmed window START on a ``tool`` message: its producing
    ``assistant`` tool_calls turn was trimmed away, and the gateway rejects a
    ``tool_call_id`` with no preceding ``tool_calls`` (the multi-turn-400 class).
    Drop leading tool messages until the window opens on a user/assistant turn.
    """
    start = 0
    while start < len(messages) and messages[start].get("role") == "tool":
        start += 1
    return messages[start:]


# Keep the most recent messages verbatim (the current + immediately-prior turn);
# older tool/assistant payloads are LLM context only (run_turn consumes ONLY
# current_ast, never message history as a runtime input), so summarizing their
# oversized content keeps the request under budget without losing actionable
# detail. Threshold is per-message content bytes, not the whole payload.
_WIRE_KEEP_RECENT = 6
_TOOL_CONTENT_MAX_BYTES = 2 * 1024


def _summarize_oversized_tool_results(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Shrink the ``content`` of OLD oversized ``tool``/``assistant`` messages
    (beyond the recent window) to a compact placeholder. Preserves ``role``,
    ``tool_call_id`` and ``tool_calls`` so the wire shape (and the gateway's
    tool_call pairing) is untouched. Stale embedded AST snapshots / discovery
    dumps carry no runtime information (the live AST is sent separately as
    ``current_ast``). Idempotent — a summarized message is already small.
    """
    n = len(messages)
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        content = msg.get("content")
        if (
            i < n - _WIRE_KEEP_RECENT
            and msg.get("role") in ("tool", "assistant")
            and isinstance(content, str)
            and len(content.encode("utf-8")) > _TOOL_CONTENT_MAX_BYTES
        ):
            summarized = dict(msg)
            summarized["content"] = (
                f"[{msg.get('role')} content summarized: "
                f"{len(content)} chars omitted to fit the request budget]"
            )
            out.append(summarized)
        else:
            out.append(msg)
    return out


def _trim_conversation(
    messages: list[dict[str, Any]],
    current_ast: dict[str, Any] | None = None,
    assets: list[DesignerAsset] | list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Bound the conversation to the size budgets by trimming the OLDEST
    messages (graceful — never raises). Keep the last ``MAX_MESSAGES`` (the
    current user turn is last, so always retained), repair the wire shape, then
    drop further oldest messages until the whole payload fits
    ``MAX_PAYLOAD_BYTES``. The AST is a separate hard cap (:meth:`check_limits`),
    so only messages are trimmed here. Idempotent.
    """
    kept = (
        list(messages)
        if len(messages) <= MAX_MESSAGES
        else list(messages[-MAX_MESSAGES:])
    )
    kept = _drop_leading_orphan_tools(kept)
    kept = _summarize_oversized_tool_results(kept)
    budget = MAX_PAYLOAD_BYTES - _payload_bytes(current_ast or {}) - _payload_bytes(assets or [])
    while len(kept) > 1 and _payload_bytes(kept) > budget:
        kept = _drop_leading_orphan_tools(kept[1:])
    return kept

_READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "discover_sources",
        "inspect_assets",
        "recommend_tools_for_assets",
        "list_node_types",
        "list_tool_kinds",
        "list_modes",
        "validate",
    }
)

_MUTATING_DESIGNER_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "propose_workflow",
        "add_block",
        "update_block",
        "delete_block",
        "move_block",
        "declare_tool",
        "remove_tool",
        "bind_tool_to_block",
        "set_model_tier",
    }
)

# Valid model tier names per app.yaml. Any other value the architect emits
# is normalized to a sensible fallback so the runner doesn't crash on
# Opus-hallucinated tier names like "standard", "medium", "high", etc.
_VALID_MODEL_TIERS: frozenset[str] = frozenset(
    {"simple", "analytical", "complex", "bulk_analysis", "fast"}
)


_ARCHITECT_JSON_BLOCK_RE = __import__("re").compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", __import__("re").DOTALL
)


def _derive_normalization_fixes(
    state: Any, node_id: str
) -> list[dict[str, Any]]:
    """Re-derive Layer 2 fixes from the architect's raw message.

    The framework executor pushes only ``ToolResult.content`` to state, so the
    ``ParseArchitectAstTool``'s ``normalization_fixes`` payload doesn't reach
    the orchestrator. Running :func:`normalize_ast` on the un-normalized AST
    extracted from ``state['architect_message']`` reproduces the same
    deterministic fix list (both code paths use the exact same normalizer).

    Returns an empty list when this isn't the parse_architect_ast node, when
    the architect emitted no JSON block, or when the JSON failed to parse.
    """
    if node_id != "parse_architect_ast":
        return []
    try:
        raw_msg = state.get("architect_message")
    except Exception:
        return []
    if not isinstance(raw_msg, str) or not raw_msg:
        return []
    match = _ARCHITECT_JSON_BLOCK_RE.search(raw_msg)
    if match is None:
        return []
    try:
        raw_ast = json.loads(match.group(1))
    except (ValueError, TypeError):
        return []
    if not isinstance(raw_ast, dict):
        return []
    _, fixes = normalize_ast(raw_ast)
    return [f.to_dict() for f in fixes]


def _derive_workflow_llm_client(framework_llm: Any, workflow_def: Any) -> Any:
    """Apply workflow-local ``models:`` tiers to a framework LLM client.

    ``WorkflowRunner`` already layers YAML ``models:`` definitions onto the
    client before execution. The Designer shim builds ``WorkflowExecutor``
    directly so it can provide a designer-specific tool registry, so it must do
    the same layering here or workflow-local tiers such as ``critic`` fail at
    runtime.
    """
    models = getattr(workflow_def, "models", None)
    if not models:
        return framework_llm

    from databricks_deep_research.llm.client import parse_model_config

    yaml_mapping = parse_model_config(models)
    logger.info(
        "DESIGNER_APPLY_YAML_MODELS tiers=%s",
        list(yaml_mapping.keys()),
    )
    return framework_llm.derive(yaml_mapping)


def _coerce_ast_snapshot(raw: Any) -> dict[str, Any] | None:
    """Coerce a cached/state AST value into a non-empty dict snapshot."""
    if isinstance(raw, dict):
        return raw if raw else None
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if isinstance(decoded, dict) and decoded:
            return decoded
    return None


def _safe_state_get(state: Any, key: str) -> Any:
    """Read a state key, swallowing any backend error (best-effort)."""
    try:
        return state.get(key)
    except Exception:  # pragma: no cover - defensive
        return None


def _coerce_jsonish(value: Any) -> dict[str, Any] | None:
    """Coerce a state value (dict or JSON string) into a dict, else None."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            return None
        if isinstance(decoded, dict):
            return decoded
    return None


# Generic, topology-agnostic fallback shown when a designer turn produced no
# AST change. Best-of-N is named only as an *example* of a structural request,
# never as a hardcoded special case — keep this generic across topologies.
_GENERIC_NO_CHANGE_MESSAGE = (
    "I wasn't able to change the workflow for this request. It may need a "
    "structural change the designer can't express yet (for example a custom "
    "topology such as Best-of-N). Try rephrasing the request, or describe the "
    "change in terms of the existing building blocks (parallel lanes, "
    "plan-and-execute, or a single agent)."
)


def _terminal_feedback_message(state: Any) -> str:
    """Explain, in user-facing prose, why a designer turn produced no mutation.

    Pure and defensive: every source may be absent, a dict, or a JSON string.
    Priority — the architect's signature-revision reason, then the critic's
    rejection directives, then the structural-gate failures, then a generic
    capability message. The designer agents suppress their own prose
    (``suppress_planning_final_output``), so this is the only place a no-op turn
    can surface an explanation to the user.
    """
    revision = _coerce_jsonish(_safe_state_get(state, "revision_request"))
    if revision:
        reason = revision.get("reason") or revision.get("message")
        if isinstance(reason, str) and reason.strip():
            fields = revision.get("fields_to_reconsider") or revision.get("fields")
            suffix = (
                f" (revisiting: {', '.join(str(f) for f in fields)})"
                if isinstance(fields, list) and fields
                else ""
            )
            return (
                "I couldn't apply this change automatically — the designer needs "
                f"to reconsider the workflow structure: {reason.strip()}{suffix}"
            )

    verdict = _coerce_jsonish(_safe_state_get(state, "critic_verdict"))
    if verdict:
        directives = verdict.get("directives")
        issues = (
            [
                str(d.get("issue")).strip()
                for d in directives
                if isinstance(d, dict) and str(d.get("issue") or "").strip()
            ]
            if isinstance(directives, list)
            else []
        )
        if issues:
            return (
                "I reviewed the change but did not apply it — unresolved issues: "
                f"{'; '.join(issues[:3])}."
            )

    gate = _coerce_jsonish(_safe_state_get(state, "gate_result"))
    if gate and str(gate.get("status") or "").lower() == "fail":
        failures = gate.get("failures")
        items = (
            [str(f.get("message") or f).strip() for f in failures]
            if isinstance(failures, list)
            else []
        )
        items = [i for i in items if i]
        if items:
            return (
                "The proposed workflow didn't pass structural checks, so no "
                f"change was applied: {'; '.join(items[:3])}."
            )

    return _GENERIC_NO_CHANGE_MESSAGE


def _terminal_error_message(state: Any) -> str | None:
    """Return a terminal error message when the designer finished unapproved."""

    signature_loop = _coerce_jsonish(_safe_state_get(state, "signature_loop_done"))
    if not signature_loop:
        return None
    if signature_loop.get("critic_approved") is not False:
        return None

    feedback = _terminal_feedback_message(state)
    if signature_loop.get("exhausted") is True:
        return (
            "I couldn't create the agent because the workflow did not pass "
            f"designer review after the available revision attempts. {feedback}"
        )
    return (
        "I couldn't create the agent because the workflow did not pass "
        f"designer review. {feedback}"
    )


def _normalize_model_tiers(ast: dict[str, Any]) -> dict[str, Any]:
    """Walk the AST and replace any unknown ``model_tier`` value with a valid
    fallback. Lane researchers / coordinator default to analytical;
    synthesizers default to complex. Mutates a deep copy of the input."""
    import copy as _copy

    if not isinstance(ast, dict):
        return ast
    normalized = _copy.deepcopy(ast)

    def walk(n: Any) -> None:
        if not isinstance(n, dict):
            return
        cfg = n.get("config")
        if isinstance(cfg, dict):
            tier = cfg.get("model_tier")
            if isinstance(tier, str) and tier and tier not in _VALID_MODEL_TIERS:
                subtype = cfg.get("subtype") or ""
                cfg["model_tier"] = (
                    "complex" if subtype == "synthesizer" else "analytical"
                )
            body = cfg.get("body")
            if isinstance(body, dict):
                walk(body)
            evaluator = cfg.get("evaluator")
            if isinstance(evaluator, dict):
                walk(evaluator)
        for child in n.get("children") or []:
            walk(child)

    root = normalized.get("root") or normalized
    walk(root)
    return normalized


def _progress_event_for(event: Any) -> ProgressEvent | None:
    """Map a framework stream event to a transient UI ProgressEvent, or None.

    Only AGENT node starts (the slow Opus/GPT-5 steps that make the wire go
    silent for ~50s) and architect-critic loop iterations are surfaced — the
    many fast tool/gate/extract nodes are skipped to stay low-noise. Filtering
    by node *type* (not an id/topology allowlist) keeps this generic across
    every designer topology; the framework's own ``label`` is used verbatim.
    """
    evt_type = getattr(event, "event_type", None)
    if evt_type == "node_started" and getattr(event, "node_type", "") == "agent":
        return ProgressEvent(label=str(getattr(event, "label", "") or "Working"))
    if evt_type == "loop_iteration":
        return ProgressEvent(
            label="Refining",
            iteration=int(getattr(event, "iteration", 0) or 0),
            total=int(getattr(event, "max_iterations", 0) or 0),
        )
    return None


def _mutation_event_for_ast_change(
    *,
    tool_name: str,
    tool_call_id: str,
    raw_ast: Any,
    last_ast_seen: dict[str, Any],
    normalization_fixes: list[dict[str, Any]],
) -> MutationProposedEvent | None:
    """Build a mutation event when a cached/state AST differs from the last snapshot."""
    new_ast = _coerce_ast_snapshot(raw_ast)
    if new_ast is None:
        return None
    new_ast, event_fixes = normalize_ast(new_ast)
    if event_fixes:
        existing_fix_keys = {
            (
                fix.get("kind"),
                fix.get("path"),
                repr(fix.get("before")),
                repr(fix.get("after")),
            )
            for fix in normalization_fixes
            if isinstance(fix, dict)
        }
        for fix in event_fixes:
            payload = fix.to_dict()
            key = (
                payload.get("kind"),
                payload.get("path"),
                repr(payload.get("before")),
                repr(payload.get("after")),
            )
            if key not in existing_fix_keys:
                normalization_fixes.append(payload)
                existing_fix_keys.add(key)
    new_ast = _normalize_model_tiers(new_ast)
    if new_ast == last_ast_seen or not new_ast:
        return None
    errors, summary = _validate_ast(new_ast)
    return MutationProposedEvent(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        old_ast=last_ast_seen,
        new_ast=new_ast,
        validation_errors=errors,
        summary=summary,
        normalization_fixes=normalization_fixes,
    )


_BUILD_REQUEST_TERMS: frozenset[str] = frozenset(
    {
        "build",
        "create",
        "design",
        "generate",
        "make",
        "scaffold",
        "set up",
        "setup",
        "construct",
    }
)

_WORKFLOW_REQUEST_TERMS: frozenset[str] = frozenset(
    {
        "agent",
        "assistant",
        "workflow",
        "researcher",
        "pipeline",
    }
)


class RequestTooLargeError(ValueError):
    """Request exceeded one of the size limits."""


# ---- LLM client protocol ----


class LLMToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    arguments: dict[str, Any]


class LLMStreamChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str | None = None
    tool_call: LLMToolCall | None = None
    finish: bool = False


class LLMClientProto(Protocol):
    """Structural protocol for the chat-completion LLM client."""

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[LLMStreamChunk]: ...


# ---- AST helpers ----

_RESEARCH_INTENT_TERMS: frozenset[str] = frozenset(
    {
        "deep research",
        "research",
        "researcher",
        "web search",
        "search",
        "crawl",
        "summarize",
        "summarise",
        "summary",
        "report",
        "simple workflow",
        "research workflow",
        "investigate",
        "analyze",
        "analyse",
        "find sources",
        "sources",
    }
)


def _extract_required_outputs(ast: dict[str, Any]) -> list[str]:
    """Pull the brief's ``required_outputs`` back out of the generated AST.

    The Designer builder stores them under each plan_and_execute node's
    ``config.synthesis_metadata.designer_required_outputs`` (newline-joined).
    We walk the AST and concatenate any we find; deduplicated and order
    preserved. Returns an empty list when no metadata is present (legacy
    saved agents).
    """
    out: list[str] = []
    seen: set[str] = set()

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        config = node.get("config") or {}
        if isinstance(config, dict):
            meta = config.get("synthesis_metadata") or {}
            if isinstance(meta, dict):
                raw = meta.get("designer_required_outputs") or ""
                if isinstance(raw, str) and raw.strip():
                    for line in raw.splitlines():
                        cleaned = line.strip()
                        if cleaned and cleaned not in seen:
                            out.append(cleaned)
                            seen.add(cleaned)
            body = config.get("body")
            if isinstance(body, dict):
                walk(body)
        for child in node.get("children", []) or []:
            walk(child)

    walk(ast.get("root"))
    return out


async def _critique_ast(
    ast: dict[str, Any],
    intent: str,
    llm: LLMClientProto,
) -> dict[str, Any] | None:
    """Run the LLM-as-judge critic against the AST + the user's intent.

    Returns a plain-dict serialization of :class:`CritiqueResult` so the
    orchestrator can include it directly in the tool-result event. ``None``
    is returned when the inputs are insufficient (no intent or no AST) —
    callers should treat that as "skip critique" rather than as a failure.
    """
    if not ast or not intent or not intent.strip():
        return None
    # Local import to avoid a circular dependency at module load: this module
    # is the consumer of LLMClientProto; workflow_critic mirrors the same
    # protocol so the import direction stays orchestrator → workflow_critic.
    from deep_research.agent_designer.workflow_critic import (
        critique_workflow_against_intent,
    )

    required_outputs = _extract_required_outputs(ast)
    result = await critique_workflow_against_intent(
        definition=ast,
        intent=intent,
        required_outputs=required_outputs,
        llm=llm,
    )
    return result.model_dump()


def _is_research_intent(intent: str) -> bool:
    """Return whether the intent should start from a research workflow scaffold."""
    normalized = intent.casefold()
    return any(term in normalized for term in _RESEARCH_INTENT_TERMS)


def _workflow_name(intent: str) -> str:
    return intent[:80] if intent else "Untitled Agent"


def _propose_research_ast(
    intent: str,
    design_brief: WorkflowDesignBrief | None = None,
) -> dict[str, Any]:
    """Build a useful research workflow instead of a one-node placeholder."""
    return build_web_research_workflow(intent, _workflow_name(intent), design_brief)


def _propose_initial_ast(
    intent: str,
    design_brief: WorkflowDesignBrief | None = None,
) -> dict[str, Any]:
    """Skeleton workflow scaffold from intent. Caller refines via further tool calls."""
    if _is_research_intent(intent):
        return _propose_research_ast(intent, design_brief)

    return build_direct_workflow(intent, _workflow_name(intent))


def _with_architect_system_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepend the YAML-backed Designer architect/critic prompt for the LLM."""
    return [{"role": "system", "content": designer_system_prompt()}, *messages]


def _message_content_to_text(content: Any) -> str:
    """Return plain text from OpenAI-style message content shapes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


def _last_user_message_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_content_to_text(message.get("content"))
    return ""


def _looks_like_workflow_build_request(messages: list[dict[str, Any]]) -> bool:
    """Heuristic guard for requests that should result in a workflow mutation."""
    normalized = _last_user_message_text(messages).casefold()
    if not normalized:
        return False
    has_build_term = any(term in normalized for term in _BUILD_REQUEST_TERMS)
    has_workflow_term = any(term in normalized for term in _WORKFLOW_REQUEST_TERMS)
    if has_build_term and has_workflow_term:
        return True
    return any(
        phrase in normalized
        for phrase in (
            "agent that",
            "assistant that",
            "workflow that",
            "research assistant",
            "research agent",
        )
    )


def _assistant_tool_calls_message(
    tool_calls: list[LLMToolCall],
    *,
    content: str = "",
) -> dict[str, Any]:
    """Build the assistant transcript message that requested read-only tools."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, default=str),
                },
            }
            for tc in tool_calls
        ],
    }


def _tool_result_message(result: ToolResultEvent) -> dict[str, Any]:
    """Build an OpenAI-compatible tool result transcript message."""
    return {
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "name": result.tool_name,
        "content": json.dumps(result.result, default=str),
    }


def _patch_node_output_models(
    node: Any,
    mapping: dict[str, type],
) -> None:
    """Recursively patch agent nodes' ``output_model`` config entry.

    The framework's ``AgentNodeConfig`` accepts an ``output_model`` field
    (Pydantic class) for structured-output validation; YAML cannot easily
    encode a Python class reference, so the designer route shim assigns it
    programmatically after the workflow is loaded. The patch lives in the
    raw ``node.config`` dict so it survives the executor's later
    ``AgentNodeConfig(**node.config)`` reconstruction.
    """
    if node is None:
        return
    node_id = getattr(node, "id", None)
    config = getattr(node, "config", None)
    if node_id in mapping and isinstance(config, dict):
        config["output_model"] = mapping[node_id]
    for child in getattr(node, "children", None) or []:
        _patch_node_output_models(child, mapping)
    # plan_and_execute nests planner/evaluator inside config rather than children
    if isinstance(config, dict):
        body = config.get("body")
        if body is not None and hasattr(body, "id"):
            _patch_node_output_models(body, mapping)


# ---- Orchestrator ----


def _edit_stream_error_message(exc: Exception, *, lane: str) -> str:
    """Friendly, actionable failure text for a designer stream error.

    The edit lane previously surfaced the raw gateway exception (a wall of 400
    JSON) which reads to the user as "nothing happened, no error". For the known
    oversized / tool-transcript failure classes we explain it plainly and state
    the workflow was left unchanged; anything else gets a clean generic message.
    The build lane keeps the original (it has different recovery wording)."""
    detail = str(exc).lower()
    oversized = any(
        token in detail
        for token in (
            "thought_signature",
            "context_length",
            "context window",
            "maximum context",
            "too many tokens",
            "max_tokens",
            "badrequest",
            "bad_request",
            " 400",
            "code: 400",
        )
    )
    if lane == "edit" and oversized:
        return (
            "I couldn't apply that edit in a single pass — it touched too many "
            "nodes at once for the model to handle reliably. Your workflow was "
            "left unchanged. Try a narrower request (one tool, or a few nodes at "
            "a time) and I'll apply it cleanly."
        )
    if lane == "edit":
        return (
            "I hit an unexpected problem applying that edit, so I left the "
            "workflow unchanged. Please try rephrasing the change."
        )
    return (
        "The designer hit an unexpected error and couldn't update the "
        f"workflow: {exc}"
    )


def _is_meaningful_ast(current_ast: dict[str, Any] | None) -> bool:
    """True when *current_ast* is a real existing workflow (an EDIT context),
    not an empty/new canvas. Mirrors the frontend ``isWorkflowEmpty`` check: a
    root node with at least one child, OR a lone agent root (single_agent)."""
    if not isinstance(current_ast, dict):
        return False
    root = current_ast.get("root")
    if not isinstance(root, dict) or not root.get("type"):
        return False
    if root.get("children"):
        return True
    return root.get("type") == "agent"


def _compact_ast_summary(current_ast: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Compact, token-light projection of the workflow for the edit-scope
    classifier and the edit agent: one row per node with id/type/subtype/label
    and bound tool names. Never includes prompts (kept small + prompt-safe)."""
    out: list[dict[str, Any]] = []
    root = (current_ast or {}).get("root") if isinstance(current_ast, dict) else None
    if not isinstance(root, dict):
        return out
    for node in iter_all_nodes(root):
        cfg = config_of(node)
        row: dict[str, Any] = {"id": node.get("id"), "type": node.get("type")}
        if cfg.get("subtype"):
            row["subtype"] = cfg.get("subtype")
        if node.get("label"):
            row["label"] = node.get("label")
        tools = cfg.get("tools")
        if isinstance(tools, list) and tools:
            row["tools"] = tools
        out.append(row)
    return out


class DesignerChatOrchestrator:
    """Dispatches LLM tool-calls to mutation primitives and yields SSE events.

    Stateless: every run_turn invocation accepts {messages, current_ast} from the caller.
    """

    def __init__(
        self,
        llm: LLMClientProto,
        discovery: DesignerDiscoveryAdapter,
        *,
        workspace_client_factory: Callable[[str | None], Any] | None = None,
    ) -> None:
        self._llm = llm
        self._discovery = discovery
        # OBO workspace-client seam for the skill->workflow brief (mirrors the
        # discovery adapter's factory). Default builds an OBO client per token.
        self._ws_factory: Callable[[str | None], Any] = (
            workspace_client_factory or _workspace_client_for_user_token
        )

    async def run_turn(
        self,
        messages: list[dict[str, Any]],
        current_ast: dict[str, Any] | None,
        session_id: str | None,
        user_token: str,
        current_user_id: str = "",
        assets: list[DesignerAsset] | list[dict[str, Any]] | None = None,
        skill_names: list[str] | None = None,
    ) -> AsyncGenerator[DesignerSSEEvent, None]:
        """Drive the designer_workflow.yaml framework workflow and translate
        framework StreamEvents → DesignerSSEEvents for the frontend.

        This is the W5c thin shim that replaces the ~600 LOC hand-coded loop.
        It loads the YAML workflow, programmatically patches the architect /
        critic agent nodes with their structured-output models, builds a live
        WorkflowState seeded from the request, runs the workflow via
        WorkflowRunner.stream(), and emits a DesignerSSEEvent for every
        framework StreamEvent the frontend cares about.
        """
        import os
        from pathlib import Path

        from databricks_deep_research.workflow.loader import load_workflow
        from databricks_deep_research.workflow.state import WorkflowState

        from deep_research.agent.adapters.llm_adapter import (
            create_framework_llm_client,
        )
        from deep_research.agent_designer.critic_types import (
            CriticVerdict,
        )

        # Bound the conversation (graceful oldest-first trim) so a long or
        # heavily-retried session can never wedge, THEN enforce the byte caps.
        messages = self.prepare_messages(messages, current_ast, assets)
        self._check_limits(messages, current_ast, assets)
        # The client owns session persistence; the orchestrator stays stateless.
        _ = session_id

        # Resolve run_as from the AST and log the execution principal for audit.
        run_as_raw = (current_ast or {}).get("run_as", "caller")
        if isinstance(run_as_raw, dict) and "service_principal_id" in run_as_raw:
            sp_id: str = run_as_raw["service_principal_id"]
            from fastapi import HTTPException as _HTTPException

            try:
                await get_service_principal_workspace_client(
                    sp_id=sp_id,
                    requesting_user_id=current_user_id,
                )
            except _HTTPException:
                yield ErrorEvent(
                    message=f"Permission denied: missing CAN_USE_AS on service principal {sp_id!r}",
                )
                return
            log_run_principal(
                {
                    "requested_by_user_id": current_user_id,
                    "executed_as_sp_id": sp_id,
                    "run_kind": "sp",
                }
            )
        else:
            log_run_principal(
                {
                    "requested_by_user_id": current_user_id,
                    "executed_as_sp_id": None,
                    "run_kind": "caller",
                }
            )

        # Workflow selection + load happens AFTER grounding + routing (see the
        # "Route" block below): the resolved lane (build vs edit) determines
        # which YAML to load, so it cannot run before we know the route.

        # 3. Build the seeded WorkflowState. The most recent user message
        #    drives the architect; prior turns ride along as
        #    conversation_history (W1 framework primitive).
        user_intent = ""
        if messages:
            last = messages[-1]
            user_intent = _message_content_to_text(last.get("content"))

        prior_messages = messages[:-1] if len(messages) > 1 else []
        # Issue #2: on an EDIT, seed the existing workflow's corpus tools back
        # into grounding so the classifier doesn't see ``no_assets`` and rebuild
        # web-only — preserving vector/table/genie/knowledge tools by default.
        # UI-selected request assets take precedence over AST-derived ones
        # (normalize_assets dedups by (kind, identity), first-wins).
        normalized_request_assets = normalize_assets(
            [*normalize_assets(assets or []), *assets_from_ast(current_ast)]
        )
        prompt_grounding = await ground_prompt(
            intent=user_intent,
            existing_assets=normalized_request_assets,
            discovery=self._discovery,
            user_id=current_user_id,
            user_token=user_token,
            default_warehouse_id=resolve_default_table_warehouse_id(),
        )
        designer_assets = asset_context_payload(prompt_grounding.resolved_assets)
        prompt_grounding_payload = prompt_grounding.model_dump(mode="json")
        prompt_grounding_summary = sanitized_prompt_grounding_summary(prompt_grounding)
        semantic_llm = getattr(self._llm, "_llm", None) or self._llm
        resource_semantics, semantic_diagnostics = (
            await extract_resource_semantics_structured(
                llm=semantic_llm,
                intent=user_intent,
                grounding=prompt_grounding,
            )
        )
        resolved_tool_contract = project_resolved_tool_contract(
            prompt_grounding,
            intent=user_intent,
            semantics=resource_semantics,
        )
        if resolved_tool_contract is not None and semantic_diagnostics:
            resolved_tool_contract = resolved_tool_contract.model_copy(
                update={
                    "diagnostics": [
                        *resolved_tool_contract.diagnostics,
                        *semantic_diagnostics,
                    ]
                }
            )
        resource_semantics_payload = (
            resource_semantics.model_dump(mode="json", by_alias=True)
            if resource_semantics is not None
            else None
        )
        resolved_tool_contract_payload = (
            resolved_tool_contract.model_dump(mode="json", by_alias=True)
            if resolved_tool_contract is not None
            else None
        )
        resource_semantics_sse = resource_semantics_summary(resource_semantics)
        resolved_tool_contract_summary = sanitized_resolved_tool_contract_summary(
            resolved_tool_contract
        )

        # ---- Route: build (new/empty AST) vs edit (surgical | topology | rebuild) ----
        # The structure-deciding build stages are blind to the existing workflow,
        # so an EDIT must be classified HERE and dispatched to the right lane
        # rather than silently rebuilt (the root cause this plan fixes). Gated by
        # DESIGNER_EDIT_LANE (default-on; set to 0/false/off to force legacy build).
        edit_lane_enabled = os.environ.get(
            "DESIGNER_EDIT_LANE", "1"
        ).strip().lower() not in {"0", "false", "no", "off", ""}
        edit_scope: EditScope | None = None
        if not edit_lane_enabled or not _is_meaningful_ast(current_ast) or bool(skill_names):
            # Compiling a skill always (re)builds — never a surgical edit of an
            # existing canvas (Codex MED #6).
            route = "build"
        else:
            ast_summary = _compact_ast_summary(current_ast)
            edit_scope = await classify_edit_scope(
                llm=semantic_llm, intent=user_intent, ast_summary=ast_summary
            )
            route = edit_scope.route
        logger.info(
            "DESIGNER_TURN_ROUTE route=%s levels=%s targets=%s",
            route,
            (edit_scope.levels if edit_scope else []),
            (edit_scope.target_node_ids if edit_scope else []),
        )

        if route == "unsupported":
            reason = (edit_scope.unsupported_reason if edit_scope else "") or ""
            yield MessageEvent(
                content=(
                    "I couldn't apply this as a workflow edit"
                    + (f": {reason.strip()}" if reason.strip() else ".")
                    + " Try describing the change in terms of the existing nodes, "
                    "tools, or prompts — or ask me to rebuild the workflow."
                )
            )
            yield DoneEvent()
            return

        if route == "topology":
            async for _topo_ev in self._run_topology_edit(
                current_ast=current_ast or {},
                user_intent=user_intent,
                edit_scope=edit_scope,
                normalized_assets=normalized_request_assets,
                resolved_tool_contract=resolved_tool_contract,
            ):
                yield _topo_ev
            return

        # Build/rebuild → the deterministic-blueprint build workflow (UNCHANGED).
        # Surgical edit → the edit lane (mutation tools seeded with current_ast).
        lane = "edit" if route == "surgical" else "build"
        if lane == "edit":
            _wf_env = os.environ.get("DESIGNER_EDIT_WORKFLOW_YAML")
            wf_path = (
                Path(_wf_env)
                if _wf_env
                else Path(__file__).parent / "designer_edit_workflow.yaml"
            )
        else:
            _wf_env = os.environ.get("DESIGNER_WORKFLOW_YAML")
            wf_path = (
                Path(_wf_env)
                if _wf_env
                else Path(__file__).parent / "designer_workflow.yaml"
            )
        try:
            workflow_def = load_workflow(str(wf_path))
        except Exception as exc:
            logger.exception("DESIGNER_WORKFLOW_LOAD_FAILED")
            yield ErrorEvent(message=f"Designer workflow load failed: {exc}")
            yield DoneEvent()
            return
        # Patch the critic's structured-output model. The edit lane has no
        # "critic" node, so this is a no-op there (the walker matches by name).
        _patch_node_output_models(workflow_def.root, {"critic": CriticVerdict})

        state = WorkflowState(
            query=user_intent,
            conversation_history=prior_messages,
        )
        # Seed the keys the architect's user_prompt_template renders. Framework
        # template rendering stringifies values, so encode the current_ast as
        # JSON to round-trip cleanly.
        state.append("init", "user_intent", user_intent)
        state.append("init", "current_ast", json.dumps(current_ast or {}))
        state.append("init", "critic_verdict", "")
        state.append("init", "gate_result", "")
        state.append("init", "designer_assets", designer_assets)
        # Skill->Workflow brief: always seed (empty for non-skill turns so the
        # prompt var resolves) and override below when skills are attached.
        state.append("init", "skill_workflow_brief", "")
        state.append("init", "prompt_grounding", prompt_grounding_payload)
        state.append("init", "prompt_grounding_summary", prompt_grounding_summary)
        state.append("init", "resource_semantics", resource_semantics_payload)
        state.append("init", "resource_semantics_summary", resource_semantics_sse)
        state.append("init", "resolved_tool_contract", resolved_tool_contract_payload)
        state.append(
            "init",
            "resolved_tool_contract_summary",
            resolved_tool_contract_summary,
        )
        state.append(
            "init",
            "prompt_grounding_diagnostics",
            prompt_grounding_summary.get("diagnostics", []),
        )
        if lane == "edit":
            # Edit lane renders {edit_scope} + {current_ast_summary} in the
            # edit_agent's user prompt (build-lane keys above are ignored by it).
            state.append(
                "init",
                "edit_scope",
                edit_scope.model_dump(mode="json") if edit_scope else {},
            )
            state.append(
                "init", "current_ast_summary", _compact_ast_summary(current_ast)
            )
        yield ToolResultEvent(
            tool_call_id="prompt_grounding:init",
            tool_name="prompt_grounding",
            result=prompt_grounding_sse_result(prompt_grounding),
        )
        yield ToolResultEvent(
            tool_call_id="resource_semantics:init",
            tool_name="resource_semantics",
            result=resource_semantics_sse,
        )
        yield ToolResultEvent(
            tool_call_id="resolved_tool_contract:init",
            tool_name="resolved_tool_contract",
            result=resolved_tool_contract_sse_result(resolved_tool_contract),
        )

        # 4. Build the framework LLM client from the app's underlying LLM.
        #    The orchestrator's `self._llm` is the LLMClientProto adapter
        #    (AppLLMAdapter), which wraps the raw app LLMClient as `._llm`.
        #    Defensively fall back to the adapter itself when no wrapped
        #    client is exposed (test fakes may not declare ._llm).
        from typing import cast as _cast

        from deep_research.services.llm.client import LLMClient as _LLMClient

        underlying_app_llm = getattr(self._llm, "_llm", None) or self._llm
        try:
            framework_llm = create_framework_llm_client(
                _cast(_LLMClient, underlying_app_llm)
            )
            framework_llm = _derive_workflow_llm_client(framework_llm, workflow_def)
        except Exception as exc:
            logger.exception("DESIGNER_FRAMEWORK_LLM_BUILD_FAILED")
            yield ErrorEvent(message=f"Designer LLM init failed: {exc}")
            yield DoneEvent()
            return

        # Skill -> Workflow (P5): when skills are attached, summarize them (OBO,
        # fail-closed scanned) into a bounded brief and seed it so the classifier
        # emits a matching TaskSignature and the architect specializes each node's
        # prompt to a skill step. Fail-soft: any failure leaves the brief empty (a
        # normal build). Only the SUMMARY enters state — never the raw skill body.
        if skill_names:
            try:
                from deep_research.agent_designer.skill_brief import (
                    render_skill_brief,
                    summarize_skills_to_brief,
                )
                from deep_research.services.skill_runtime import (
                    build_runtime_skill_store,
                )

                _skill_store = build_runtime_skill_store(
                    llm_client=framework_llm,
                    workspace_client=self._ws_factory(user_token),
                    user_token=user_token,
                )
                _brief = await summarize_skills_to_brief(
                    skill_names, skill_store=_skill_store, llm=semantic_llm
                )
                _rendered = render_skill_brief(_brief)
                if _rendered:
                    state.append("skill_brief", "skill_workflow_brief", _rendered)
                logger.info(
                    "SKILL_TO_WORKFLOW_BRIEF_SEEDED skills=%d steps=%d",
                    len(skill_names),
                    len(_brief.steps),
                )
            except Exception:  # noqa: BLE001 — brief is best-effort; never break a turn
                logger.warning("SKILL_TO_WORKFLOW_BRIEF_FAILED", exc_info=True)

        # Build a WorkflowRunner via the app's shared factory so the Designer
        # follows the same construction pattern as every other entry point
        # (see workflow_runner_factory.py for the convention). The Designer
        # threads its own ToolRegistry via runner.stream(tool_registry=...)
        # so `type: tool` nodes (parse_architect_ast, structural_gate,
        # extract_critic_approved) resolve through the pre-populated registry.
        from databricks_deep_research.tools.registry import ToolRegistry

        from deep_research.agent.workflow_runner_factory import (
            build_app_workflow_runner,
        )
        from deep_research.agent_designer.framework_tools import (
            pop_compact_tool_results,
            push_compact_tool_results,
            register_designer_tools,
        )

        designer_registry = ToolRegistry()
        # Conversation-local AST cache. During the architect's ReAct loop the
        # workflow state's current_ast does NOT update between tool calls —
        # only the post-agent parse_architect_ast node writes back. So the
        # mutation tools maintain their own cache: propose_workflow seeds it,
        # subsequent update_block/etc. read+write it. Net effect: the LLM
        # doesn't have to echo the AST as an argument anymore, saving ~5K
        # output tokens/call and avoiding mid-call truncation.
        _ast_cache: list[Any] = [None]  # mutable single-cell holder
        if lane == "edit":
            # Seed the cache with the EXISTING workflow so the first mutation
            # tool builds on current_ast (a minimal delta), not an empty AST.
            _ast_cache[0] = current_ast or {}

        def _state_ast_getter() -> Any:
            if _ast_cache[0] is not None:
                return _ast_cache[0]
            try:
                return state.get("current_ast")
            except Exception:  # pragma: no cover — defensive
                return None

        def _state_ast_setter(new_ast: Any) -> None:
            _ast_cache[0] = new_ast

        register_designer_tools(
            designer_registry,
            discovery=self._discovery,
            state_getter=_state_ast_getter,
            state_setter=_state_ast_setter,
            asset_getter=lambda: state.get("designer_assets"),
            prompt_grounding_getter=lambda: state.get("prompt_grounding"),
            resolved_tool_contract_getter=lambda: state.get("resolved_tool_contract"),
            # Fix D — wire ParseArchitectAstTool patch mode. BuildBlueprintTool
            # writes ``initial_blueprint`` + ``blueprint_fingerprint`` to state;
            # without these getters, ParseArchitectAstTool falls back to
            # legacy AST parsing and the architect's node_patches JSON is
            # misinterpreted as a standalone AST (the bug observed in
            # event 13 of the failing investment_research scaffold-and-run).
            blueprint_getter=lambda: state.get("initial_blueprint"),
            fingerprint_getter=lambda: state.get("blueprint_fingerprint"),
            current_ast_summary_setter=lambda value: state.append(
                "parse_architect_ast", "current_ast_summary", value
            ),
            # Codex review #5 — surface the parse tool's REAL normalization
            # fixes (it computed them against the actual pre-normalized AST /
            # merged blueprint) instead of the orchestrator re-deriving them
            # from architect_message (a node_patches doc in patch mode).
            normalization_fixes_setter=lambda value: state.append(
                "parse_architect_ast", "designer_normalization_fixes", value
            ),
            # Fix (live run) — EmitTaskSignatureTool now writes its
            # validated payload directly to state.task_signature via this
            # setter. Without it, the classifier agent's free-form prose
            # would land in state.task_signature via the YAML output_key,
            # and downstream build_blueprint sees ``task_signature is
            # required (got None)`` because the prose isn't a dict.
            signature_setter=lambda value: state.append(
                "emit_task_signature", "task_signature", value
            ),
            # Plan v2.1 generic-robustness — expose blueprint lane_keys to
            # the architect prompt via the ``{lane_keys}`` template variable.
            # Without this, the architect cannot READ the content-derived
            # lane keys (computed in ``blueprint.compute_lane_key``) and
            # falls back to ordinal addressing for ``node_patches``, which
            # drifts across signature revisions (plan M7).
            lane_keys_setter=lambda value: state.append(
                "build_blueprint", "lane_keys", value
            ),
            # Plan v2.1 generic-robustness — surface the placeholder-pending
            # node id list as its own state key so the architect's
            # user_prompt_template can render ``{placeholder_pending_nodes}``
            # directly. Without this, the list is still in
            # ``initial_blueprint.placeholder_pending_nodes`` but buried
            # inside a multi-KB JSON blob the LLM tends to skim.
            placeholder_pending_setter=lambda value: state.append(
                "build_blueprint", "placeholder_pending_nodes", value
            ),
            # Plan v2.1 generic-robustness — wire RequestSignatureRevisionTool
            # state setters so the architect's escape valve persists. The
            # outer ``signature_loop`` (designer_workflow.yaml) reads these
            # to decide whether to re-run the classifier with the revision
            # hint. Without these closures, the tool was dead code (its
            # docstring at framework_tools.py:2255 explicitly flagged the
            # YAML loop-back as deferred PR-3 work).
            revision_count_getter=lambda: state.get("revision_count"),
            revision_count_setter=lambda value: state.append(
                "request_signature_revision", "revision_count", value
            ),
            revision_request_setter=lambda value: state.append(
                "request_signature_revision", "revision_request", value
            ),
            error_setter=lambda value: state.append(
                "request_signature_revision", "error", value
            ),
            # Intent-grounding writes back the merged designer_assets payload
            # (UI-selected ∪ LLM-grounded). All downstream consumers — the
            # classifier's user_prompt_template, build_blueprint, the architect's
            # inspect_assets / recommend_tools_for_assets tools — read
            # state.designer_assets via asset_getter, so a single in-place
            # write is sufficient.
            designer_assets_setter=lambda value: state.append(
                "emit_grounded_assets", "designer_assets", value
            ),
        )

        runner = build_app_workflow_runner(
            llm_client=framework_llm,
            # Designer tools don't go through ToolFactoryContext (they're
            # pre-registered in designer_registry); workspace/user_token
            # are unused by the Designer phase.
            workspace_client=None,
            user_token=None,
        )

        last_ast_seen: dict[str, Any] = current_ast or {}
        yielded_done = False
        emitted_mutation = False
        error_occurred = False

        # EDIT lane: scope mutation tool RESULTS to a bounded summary for the
        # duration of the ReAct stream, so a multi-node edit (e.g. "add a tool
        # to all candidates") can't balloon the transcript by echoing the full
        # AST per mutation (the 140K-token blowup that fell back to a gateway
        # 400). Build lane leaves it untouched. Reset in the finally below.
        _compact_token = (
            push_compact_tool_results(True) if lane == "edit" else None
        )
        try:
            async for event in runner.stream(
                workflow_def,
                state=state,
                tool_registry=designer_registry,
                strict_tool_resolution=True,
            ):
                evt_type = getattr(event, "event_type", None)

                if evt_type == "agent_stream_chunk":
                    content = getattr(event, "content", None) or ""
                    if content:
                        yield MessageEvent(content=content)

                elif evt_type == "tool_call":
                    yield ToolCallEvent(
                        tool_name=getattr(event, "tool_name", "unknown"),
                        tool_call_id=getattr(event, "tool_call_id", "") or "",
                        args=getattr(event, "arguments", {}) or {},
                    )

                elif evt_type in ("node_started", "loop_iteration"):
                    # Live progress so the chat shows activity (instead of a
                    # frozen spinner) during the long Opus/GPT-5 agent calls;
                    # also keeps the wire warm at node boundaries. Transient —
                    # never persisted to the transcript. See _progress_event_for.
                    progress_event = _progress_event_for(event)
                    if progress_event is not None:
                        yield progress_event

                elif evt_type == "tool_result" and lane != "edit":
                    # Build lane only. The edit lane SUPPRESSES intermediate
                    # mutation events (each carries a full old+new AST and the
                    # turn-registry buffers every event) and emits ONE net delta
                    # at finalize (Codex H7). Build-lane behavior is unchanged.
                    # The framework ReAct loop returns an agent's tool events
                    # after the agent finishes, while designer mutation tools
                    # update the conversation-local AST cache during execution.
                    # Emitting on mutating tool results prevents the UI/test
                    # harness from persisting an earlier node snapshot when the
                    # architect made later update_block calls in the same turn.
                    tool_name = str(getattr(event, "tool_name", "") or "")
                    if tool_name in _MUTATING_DESIGNER_TOOL_NAMES:
                        node_id = getattr(event, "node_id", "") or "unknown"
                        mutation_event = _mutation_event_for_ast_change(
                            tool_name=tool_name,
                            tool_call_id=f"tool_{tool_name}_{node_id}",
                            raw_ast=_ast_cache[0],
                            last_ast_seen=last_ast_seen,
                            normalization_fixes=[],
                        )
                        if mutation_event is not None:
                            yield mutation_event
                            last_ast_seen = mutation_event.new_ast
                            emitted_mutation = True

                elif evt_type == "node_completed" and lane != "edit":
                    # Build lane only (edit lane finalizes once — see above).
                    node_id = getattr(event, "node_id", "")
                    # After each node, surface a MutationProposedEvent if the
                    # AST has changed.
                    #
                    # AST sources (in priority order):
                    # 1. State.current_ast — populated by parse_architect_ast,
                    #    which runs Layer 2 normalize_ast (consolidations,
                    #    brace-escape, etc.). This is the CANONICAL form that
                    #    downstream nodes + the saved workflow should see.
                    # 2. _ast_cache — populated by architect's internal tool
                    #    calls (propose_workflow/update_block). Contains the
                    #    UN-normalized AST until parse_architect_ast runs.
                    #
                    # We prefer state when parse_architect_ast has produced
                    # something there, otherwise fall back to the in-flight
                    # cache (so partial-iteration emits during the architect's
                    # ReAct loop still surface).
                    state_raw: Any = None
                    try:
                        state_raw = state.get("current_ast")
                    except Exception:
                        state_raw = None
                    has_state = node_id == "parse_architect_ast" and (
                        isinstance(state_raw, dict)
                        or (
                            isinstance(state_raw, str)
                            and state_raw.strip()
                            and state_raw.strip() != "{}"
                        )
                    )
                    if has_state:
                        # parse_architect_ast wrote the normalized AST —
                        # this is canonical. Sync the cache so subsequent
                        # mutation tool calls (e.g. in the next loop iter)
                        # see the normalized version too.
                        raw: Any = state_raw
                        _ast_cache[0] = state_raw
                    else:
                        raw = _ast_cache[0] if _ast_cache[0] is not None else state_raw
                    if raw is not None:
                        # Layer 2 fix surfacing: prefer the REAL fixes the
                        # parse_architect_ast tool published to state (computed
                        # against the actual pre-normalized AST / merged
                        # blueprint — correct in patch mode where the raw
                        # architect_message is a node_patches doc). Fall back to
                        # re-deriving from architect_message only when the tool
                        # did not publish (older runs / non-parse nodes).
                        fixes_payload: list[dict[str, Any]] | None = None
                        try:
                            published = state.get("designer_normalization_fixes")
                        except Exception:
                            published = None
                        if isinstance(published, list):
                            fixes_payload = published
                        if fixes_payload is None:
                            fixes_payload = _derive_normalization_fixes(
                                state, node_id
                            )
                        mutation_event = _mutation_event_for_ast_change(
                            tool_name="propose_workflow",
                            tool_call_id=f"node_{node_id or 'unknown'}",
                            raw_ast=raw,
                            last_ast_seen=last_ast_seen,
                            normalization_fixes=fixes_payload,
                        )
                        if mutation_event is not None:
                            yield mutation_event
                            last_ast_seen = mutation_event.new_ast
                            emitted_mutation = True

                elif evt_type == "workflow_completed":
                    terminal_error = _terminal_error_message(state)
                    if terminal_error:
                        logger.info(
                            "DESIGNER_TURN_REVIEW_FAILED reason=critic_not_approved"
                        )
                        yield ErrorEvent(message=terminal_error)
                        yield DoneEvent()
                        yielded_done = True
                        continue
                    if lane == "edit":
                        # Emit the single net delta (+ guard) for the whole turn.
                        async for _fin in self._finalize_edit(
                            current_ast=current_ast or {},
                            final_ast=_ast_cache[0],
                            edit_scope=edit_scope,
                        ):
                            yield _fin
                    elif not emitted_mutation:
                        # Never-silent invariant (build lane): a turn that
                        # proposed no mutation must still tell the user why —
                        # the mutation card is the only surface otherwise.
                        logger.info(
                            "DESIGNER_TURN_NO_MUTATION reason=completed_without_ast_change"
                        )
                        yield MessageEvent(content=_terminal_feedback_message(state))
                    yield DoneEvent()
                    yielded_done = True
        except Exception as exc:
            logger.exception("DESIGNER_WORKFLOW_STREAM_FAILED")
            error_occurred = True
            yield ErrorEvent(message=_edit_stream_error_message(exc, lane=lane))
        finally:
            if _compact_token is not None:
                pop_compact_tool_results(_compact_token)

        if not yielded_done:
            # Stream ended without an explicit workflow_completed (early stop or
            # error). Guarantee a visible signal before closing the channel.
            if lane == "edit" and not error_occurred:
                async for _fin in self._finalize_edit(
                    current_ast=current_ast or {},
                    final_ast=_ast_cache[0],
                    edit_scope=edit_scope,
                ):
                    yield _fin
            elif not emitted_mutation and not error_occurred:
                logger.info(
                    "DESIGNER_TURN_NO_MUTATION reason=stream_ended_without_completion"
                )
                yield MessageEvent(content=_terminal_feedback_message(state))
            yield DoneEvent()

    async def _finalize_edit(
        self,
        *,
        current_ast: dict[str, Any],
        final_ast: Any,
        edit_scope: EditScope | None,
    ) -> AsyncGenerator[DesignerSSEEvent, None]:
        """Emit the single net result of a surgical edit (Codex H6/H7): one
        ``MutationProposedEvent`` for the cumulative delta plus a human-readable
        summary (with any minimality-guard warnings), OR a no-change message.
        Intermediate per-tool mutation events are suppressed in the stream loop,
        so this is the turn's only proposal. Diff baselines are normalized so a
        normalization-only difference never reads as a spurious change.
        """
        baseline, _ = normalize_ast(current_ast or {})
        event = _mutation_event_for_ast_change(
            tool_name="edit",
            tool_call_id="edit_finalize",
            raw_ast=final_ast,
            last_ast_seen=baseline,
            normalization_fixes=[],
        )
        if event is None:
            logger.info("DESIGNER_EDIT_RESULT result=no_change")
            summary = (
                edit_scope.change_summary
                if edit_scope and edit_scope.change_summary
                else ""
            )
            yield MessageEvent(
                content=(
                    "I didn't change the workflow"
                    + (f" — {summary}" if summary else " for this request.")
                    + " If you expected a change, name the node, tool, or property "
                    "to edit and I'll try again."
                )
            )
            return
        allow_list = (
            edit_scope.to_allow_list()
            if edit_scope is not None
            else EditScope(route="surgical").to_allow_list()
        )
        guard = edit_diff_guard(baseline, event.new_ast, allow_list)
        logger.info(
            "DESIGNER_EDIT_RESULT changed=%s added=%s removed=%s guard_ok=%s",
            guard.changed_node_ids,
            guard.added_node_ids,
            guard.removed_node_ids,
            guard.ok,
        )
        summary = (
            edit_scope.change_summary
            if edit_scope and edit_scope.change_summary
            else "Updated the workflow."
        )
        if not guard.ok:
            summary += (
                "\n\n⚠️ This proposal also changed parts of the workflow that "
                "weren't part of the request: " + "; ".join(guard.violations)
            )
        yield MessageEvent(content=summary)
        yield event

    async def _run_topology_edit(
        self,
        *,
        current_ast: dict[str, Any],
        user_intent: str,
        edit_scope: EditScope | None,
        normalized_assets: list[DesignerAsset],
        resolved_tool_contract: Any,
    ) -> AsyncGenerator[DesignerSSEEvent, None]:
        """Topology SWITCH: rebuild deterministically from the PERSISTED
        signature + the requested delta, carrying prompts over best-effort.

        Pure (no LLM architect pass): ``build_blueprint`` + ``carry_over_prompts``
        + prompt-merge. Degrades to a never-silent message for legacy ASTs with
        no persisted signature (a guessed rebuild would be lossy).
        """
        from deep_research.agent_designer.blueprint import build_blueprint

        yield ProgressEvent(label="Restructuring workflow")
        sig = stored_signature(current_ast)
        if sig is None:
            yield MessageEvent(
                content=(
                    "This workflow was built before structured editing, so I "
                    "can't change its topology without rebuilding it from scratch "
                    "(which could lose customizations). Ask me to rebuild it, or "
                    "make the change as smaller edits."
                )
            )
            yield DoneEvent()
            return
        new_sig = apply_signature_delta(sig, edit_scope.delta if edit_scope else {})
        assets_payload = [a.model_dump() for a in (normalized_assets or [])] or None
        try:
            blueprint = build_blueprint(
                new_sig,
                user_intent,
                assets=assets_payload,
                tool_contract=resolved_tool_contract,
            )
        except Exception as exc:  # noqa: BLE001 - surface, never silent
            logger.exception("DESIGNER_TOPOLOGY_BUILD_FAILED")
            yield MessageEvent(
                content=(
                    "I couldn't restructure the workflow for that request "
                    f"({exc}). Try rephrasing, or ask me to rebuild it."
                )
            )
            yield DoneEvent()
            return
        patches, regenerated = carry_over_prompts(current_ast, blueprint)
        for node_id, patch in patches.items():
            try:
                blueprint = mutations.update_block(
                    blueprint, node_id, {"config": patch}
                )
            except (mutations.BlockPathError, mutations.BlockMutationError):
                continue  # unmappable role → keep the blueprint's default prompt
        event = _mutation_event_for_ast_change(
            tool_name="topology_edit",
            tool_call_id="edit_topology",
            raw_ast=blueprint,
            last_ast_seen=normalize_ast(current_ast or {})[0],
            normalization_fixes=[],
        )
        summary = (
            edit_scope.change_summary
            if edit_scope and edit_scope.change_summary
            else "Restructured the workflow."
        )
        if regenerated:
            summary += (
                f"\n\n⚠️ I kept your prompts where roles matched; {len(regenerated)} "
                f"new role(s) use default prompts ({', '.join(regenerated[:6])}) and "
                "tool bindings were not carried over — review and refine them."
            )
        logger.info("DESIGNER_TOPOLOGY_RESULT regenerated=%s", regenerated)
        yield MessageEvent(content=summary)
        if event is not None:
            yield event
        else:
            yield MessageEvent(
                content="The requested change produced no net difference."
            )
        yield DoneEvent()

    def prepare_messages(
        self,
        messages: list[dict[str, Any]],
        current_ast: dict[str, Any] | None = None,
        assets: list[DesignerAsset] | list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Pre-flight: bound the conversation to the size budgets via graceful
        oldest-first trimming (never raises). Call in the route handler BEFORE
        :meth:`check_limits` and :meth:`run_turn` so both see the same bounded
        list. Idempotent — ``run_turn`` re-applies it defensively.
        """
        return _trim_conversation(messages, current_ast, assets)

    def check_limits(
        self,
        messages: list[dict[str, Any]],
        current_ast: dict[str, Any] | None,
        assets: list[DesignerAsset] | list[dict[str, Any]] | None = None,
    ) -> None:
        """Public alias for pre-flight size validation.

        Call this AFTER :meth:`prepare_messages` and before opening the SSE
        stream so a RequestTooLargeError can be surfaced as an HTTP 413 rather
        than as an SSE error event.
        """
        self._check_limits(messages, current_ast, assets)

    def _check_limits(
        self,
        messages: list[dict[str, Any]],
        current_ast: dict[str, Any] | None,
        assets: list[DesignerAsset] | list[dict[str, Any]] | None = None,
    ) -> None:
        """Hard size guard, applied AFTER :func:`_trim_conversation` has bounded
        the message count gracefully. The message-COUNT limit is intentionally
        not enforced here — trimming handles it, so a chat can never wedge on a
        long/retried conversation. Only genuinely unprocessable byte sizes raise
        (surfaced to the client as HTTP 413, never silent).
        """
        ast_size = _payload_bytes(current_ast or {})
        if ast_size > MAX_AST_BYTES:
            raise RequestTooLargeError(
                f"current_ast exceeds {MAX_AST_BYTES} bytes ({ast_size})"
            )
        total = ast_size + _payload_bytes(messages) + _payload_bytes(assets or [])
        if total > MAX_PAYLOAD_BYTES:
            raise RequestTooLargeError(
                f"total payload exceeds {MAX_PAYLOAD_BYTES} bytes ({total})"
            )

    async def _dispatch(
        self,
        tc: LLMToolCall,
        ast: dict[str, Any] | None,
        user_token: str,
        current_user_id: str,
        user_intent: str = "",
        assets: list[DesignerAsset] | list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[DesignerSSEEvent, None]:
        # Tag the active designer-chat trace with the tool name. MLflow's
        # update_current_trace appends, so successive tool calls accumulate
        # — searchable as ``tags.dr.designer_tool LIKE '%propose_workflow%'``
        # without needing per-tool spans. Failures are swallowed; tracing
        # is best-effort and must never break the chat.
        try:
            import mlflow

            mlflow.update_current_trace(tags={"dr.designer_tool.latest": tc.name})
        except Exception:  # pragma: no cover - defensive
            pass

        # Validate args
        try:
            parsed = parse_tool_args(tc.name, tc.arguments)
        except KeyError:
            log_chat_mutation(tc.name, {}, 0, "error")
            yield ErrorEvent(message=f"Unknown tool: {tc.name}", tool_call_id=tc.id)
            return
        except ValidationError as exc:
            log_chat_mutation(tc.name, {}, 0, "error")
            yield ErrorEvent(message=f"Invalid args for {tc.name}: {exc}", tool_call_id=tc.id)
            return

        # Build a small args summary (keys only, no full AST values)
        _args_summary: dict[str, object] = {k: str(v)[:80] for k, v in tc.arguments.items()}

        # Tools that don't mutate AST
        if tc.name == "discover_sources":
            assert isinstance(parsed, DiscoverSourcesArgs)
            valid_source_kinds = {item["kind"] for item in _source_kinds_payload()}
            kinds: list[SourceKind] | None = (
                [k for k in parsed.kinds if k in valid_source_kinds]  # type: ignore[misc]
                if parsed.kinds is not None
                else None
            )
            try:
                resources = await self._discovery.list_for_user(
                    user_token=user_token,
                    kinds=kinds,
                    user_id=current_user_id,
                )
                log_chat_mutation(tc.name, _args_summary, 0, "success")
                yield ToolResultEvent(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result={"resources": [r.model_dump() for r in resources]},
                )
            except Exception as exc:
                logger.exception("AGENT_DESIGNER_DISCOVER_SOURCES_FAILED")
                log_chat_mutation(tc.name, _args_summary, 0, "error")
                yield ErrorEvent(message=f"discover_sources failed: {exc}", tool_call_id=tc.id)
            return

        if tc.name == "inspect_assets":
            assert isinstance(parsed, InspectAssetsArgs)
            raw_assets = parsed.assets if parsed.assets is not None else asset_context_payload(assets)
            result = inspect_assets(raw_assets)
            log_chat_mutation(tc.name, _args_summary, 0, "success")
            yield ToolResultEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result=result,
            )
            return

        if tc.name == "recommend_tools_for_assets":
            assert isinstance(parsed, RecommendToolsForAssetsArgs)
            raw_assets = parsed.assets if parsed.assets is not None else asset_context_payload(assets)
            result = recommend_tools_for_assets(raw_assets, intent=parsed.intent or user_intent)
            log_chat_mutation(tc.name, _args_summary, 0, "success")
            yield ToolResultEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result=result,
            )
            return

        if tc.name == "list_node_types":
            log_chat_mutation(tc.name, _args_summary, 0, "success")
            yield ToolResultEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result={"node_types": _node_types_payload()},
            )
            return

        if tc.name == "list_tool_kinds":
            log_chat_mutation(tc.name, _args_summary, 0, "success")
            yield ToolResultEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result={"tool_kinds": _tool_kinds_payload()},
            )
            return

        if tc.name == "list_modes":
            log_chat_mutation(tc.name, _args_summary, 0, "success")
            yield ToolResultEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result={
                    "model_tiers": _model_tiers_payload(),
                    "query_modes": _query_modes_payload(),
                    "research_depths": _research_depths_payload(),
                    "source_kinds": _source_kinds_payload(),
                },
            )
            return

        if tc.name == "validate":
            errors, summary = _validate_ast(ast or {})
            advice = _quality_advice(ast or {}) if not errors else []
            # The LLM-as-judge critic runs only when structurally valid:
            # there is no point asking the critic about a broken AST.
            # The user's intent is taken from the most recent user message
            # (the same one the Designer LLM is currently responding to).
            critique: dict[str, Any] | None = None
            if not errors:
                try:
                    critique = await _critique_ast(ast or {}, user_intent, self._llm)
                except Exception as exc:  # noqa: BLE001 — critic is advisory
                    # Critic failure must not block validate. Log and continue.
                    logger.warning(
                        "validate-tool critique failed: %s",
                        exc,
                        exc_info=True,
                    )
                    critique = None
            log_chat_mutation(
                tc.name,
                _args_summary,
                len(errors),
                "success" if not errors else "validation_failed",
            )
            yield ToolResultEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result={
                    "valid": not errors,
                    "errors": errors,
                    "summary": summary,
                    # Quality advice: per-agent specialization gaps the LLM
                    # should address via update_block / bind_tool_to_block /
                    # set_model_tier. These do NOT affect ``valid``.
                    "advice": advice,
                    # LLM-as-judge critique: structured verdict on whether
                    # the workflow as built actually answers the user's
                    # intent. ``None`` when skipped (no intent or structural
                    # errors). Verdict ``fail`` blocks save (with override);
                    # ``needs_revision`` warns; ``pass`` is clean. Does NOT
                    # affect ``valid`` — that flag is structural only.
                    "critique": critique,
                },
            )
            return

        # Mutation tools
        if tc.name == "propose_workflow":
            assert isinstance(parsed, ProposeWorkflowArgs)
            new_ast = _propose_initial_ast(parsed.intent, parsed.design_brief)
        else:
            if ast is None:
                log_chat_mutation(tc.name, _args_summary, 0, "error")
                yield ErrorEvent(
                    message=f"Cannot run {tc.name}: no current_ast (call propose_workflow first)",
                    tool_call_id=tc.id,
                )
                return
            try:
                new_ast = self._apply_mutation(tc.name, parsed, ast)
            except (mutations.BlockPathError, mutations.BlockMutationError, ValueError) as exc:
                log_chat_mutation(tc.name, _args_summary, 0, "error")
                yield ErrorEvent(message=str(exc), tool_call_id=tc.id)
                return

        errors, summary = _validate_ast(new_ast)
        log_chat_mutation(
            tc.name,
            _args_summary,
            len(errors),
            "success" if not errors else "validation_failed",
        )
        yield MutationProposedEvent(
            tool_name=tc.name,
            tool_call_id=tc.id,
            old_ast=ast,
            new_ast=new_ast,
            validation_errors=errors,
            summary=summary,
        )

    def _apply_mutation(
        self,
        tool_name: str,
        parsed: BaseModel,
        ast: dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(parsed, AddBlockArgs):
            new_ast, _ = mutations.add_block(
                ast, parsed.parent_path, parsed.node_type, parsed.config, parsed.label
            )
            return new_ast
        if isinstance(parsed, UpdateBlockArgs):
            return mutations.update_block(ast, parsed.path, parsed.patches)
        if isinstance(parsed, DeleteBlockArgs):
            return mutations.delete_block(ast, parsed.path)
        if isinstance(parsed, MoveBlockArgs):
            return mutations.move_block(ast, parsed.from_path, parsed.to_path, parsed.position)
        if isinstance(parsed, DeclareToolArgs):
            return mutations.declare_tool(ast, parsed.kind, parsed.name, parsed.config)
        if isinstance(parsed, RemoveToolArgs):
            return mutations.remove_tool(ast, parsed.name)
        if isinstance(parsed, BindToolArgs):
            return mutations.bind_tool_to_block(ast, parsed.node_path, parsed.tool_name)
        if isinstance(parsed, SetModelTierArgs):
            return mutations.set_model_tier(ast, parsed.node_path, parsed.tier)
        raise ValueError(f"Unhandled mutation: {tool_name}")
