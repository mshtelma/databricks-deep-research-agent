"""Chat-message role vocabulary for the OpenAI-compatible LLM gateway.

The Databricks model-serving gateway (and the OpenAI Chat Completions API)
accept a fixed set of message roles. Callers that build conversation history
from their own domain models may use non-standard role names (e.g. an app that
stores assistant turns as ``"agent"``). Sending such a role yields an opaque
``400 BAD_REQUEST: Invalid role in the chat message``.

This module is the single source of truth for the valid role set and provides a
pure, dependency-free normalizer used at the message-assembly choke point so the
framework's contract — *emit OpenAI-format messages* — holds regardless of which
caller populated the history.
"""

from __future__ import annotations

from typing import Any

# The roles accepted by the OpenAI Chat Completions API / Databricks gateway.
OPENAI_CHAT_ROLES: frozenset[str] = frozenset(
    {"system", "developer", "user", "assistant", "tool", "function"}
)


def normalize_history_roles(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce conversation-history message roles to the OpenAI-valid set.

    Conversation-history turns are either the user's prior messages or the
    model's prior responses. Any role outside :data:`OPENAI_CHAT_ROLES` (e.g. an
    app-internal ``"agent"``) is a model turn and is mapped to ``"assistant"``.
    Valid roles (including ``tool``/``function`` carrying ``tool_call_id``/``name``)
    pass through untouched.

    The function is pure: rewritten messages are returned as NEW dicts and the
    input list and its dicts are never mutated (the framework's ``RuntimeState``
    is append-only and re-read per node, so in-place mutation would corrupt later
    nodes).

    Args:
        history: Conversation-history messages, each a dict with at least a
            ``role`` key.

    Returns:
        A new list with every message's ``role`` in :data:`OPENAI_CHAT_ROLES`.
    """
    normalized: list[dict[str, Any]] = []
    for msg in history:
        role = str(msg.get("role") or "")
        if role in OPENAI_CHAT_ROLES:
            normalized.append(msg)
        else:
            normalized.append({**msg, "role": "assistant"})
    return normalized


def sanitize_history_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce conversation history into a gateway-safe, tool-mechanics-free shape.

    The OpenAI-compatible Databricks gateway enforces TWO rules that
    :func:`normalize_history_roles` (role-only) does not cover:

    1. The message schema is ``extra="forbid"`` — a ``tool_calls`` key on a
       non-assistant message (or ``tool_call_id``/``name`` on the wrong role)
       yields ``400 ... messages.N.tool_calls: Extra inputs are not permitted``.
    2. Tool-call **pairing/adjacency** — an assistant ``tool_calls`` turn must be
       immediately followed by a ``tool`` message per call id, and a conversation
       may not end on an unmatched assistant tool-call turn.

    Cross-turn conversation history (e.g. prior Agent Designer chat turns) is
    reassembled client-side from streamed events, persisted/rehydrated, and then
    sliced (``messages[:-1]``) before being seeded as history. None of rules (1)
    or (2) can be guaranteed across that pipeline, so replaying tool mechanics is
    a recurring source of opaque 400s.

    This normalizer makes history UNCONDITIONALLY gateway-safe by FLATTENING tool
    mechanics: ``tool``/``function`` messages are dropped entirely, assistant
    ``tool_calls`` are stripped, and only ``{role, content}`` text turns
    (``system``/``user``/``assistant``) survive (non-standard roles such as an
    app's ``"agent"`` map to ``assistant``). An assistant turn that carried only
    tool calls (no textual content) is dropped — an empty assistant message is
    itself invalid and carries no forward conversational value.

    This is SAFE because in-turn tool calling is unaffected: the ReAct loop
    appends assistant ``tool_calls`` and their matching ``tool`` results inside
    ``ReactLoop.execute`` — those live in the per-turn message list, NOT in this
    cross-turn ``conversation_history``. Each agent turn re-runs its own tool
    loop, so prior-turn tool mechanics carry no forward value.

    Pure: returns NEW dicts; never mutates the input list or its dicts (the
    framework's ``RuntimeState`` is append-only and re-read per node).

    Args:
        history: Conversation-history messages, each a dict with at least a
            ``role`` key.

    Returns:
        A new list of ``{role, content}`` dicts with roles in
        :data:`OPENAI_CHAT_ROLES` and all tool mechanics removed.
    """
    sanitized: list[dict[str, Any]] = []
    for msg in history:
        role = str(msg.get("role") or "")
        # Drop cross-turn tool mechanics: tool/function results have no forward
        # value to the next turn and create orphan-message pairing 400s.
        if role in {"tool", "function"}:
            continue
        if role not in OPENAI_CHAT_ROLES:
            role = "assistant"
        content = msg.get("content")
        # An assistant turn with no textual content was a tool-call-only turn;
        # once we strip tool_calls it would be an (invalid) empty assistant
        # message, so drop it.
        if role == "assistant" and (
            content is None or (isinstance(content, str) and not content.strip())
        ):
            continue
        sanitized.append({"role": role, "content": "" if content is None else content})
    return sanitized
