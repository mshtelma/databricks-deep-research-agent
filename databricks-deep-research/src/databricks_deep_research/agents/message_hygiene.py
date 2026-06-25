"""Message-hygiene helpers shared by the agent harness and ReAct loop.

The OpenAI-compatible Databricks gateway rejects malformed tool-call message
sequences with opaque 400s. Two distinct hygiene operations protect against
that, previously inlined at two call sites:

1. **Dangling tool-result fill** (was inline in ``react_loop.ReactLoop.execute``):
   every assistant ``tool_calls`` turn must be followed by exactly one ``tool``
   message per call id, or the gateway rejects the turn with a "tool_use without
   tool_result" error. After dispatching a round of tool calls the loop appends
   a tool-result message for every call that has not already been answered.

2. **Cross-turn history sanitation** (the harness ``_build_messages`` replay-strip):
   replayed conversation history must not carry tool mechanics (assistant
   ``tool_calls`` keys, orphan ``tool``/``function`` messages), or the gateway
   400s with "Extra inputs are not permitted" / tool-call-pairing errors. The
   canonical implementation lives in :mod:`databricks_deep_research.llm.roles`
   (``sanitize_history_messages``) and is re-exported here as
   :func:`sanitize_conversation_history` so both hygiene operations have a single
   import home.

Consolidating these into one module gives the two callers a shared, documented
contract. The behavior of each helper is byte-identical to the inline code it
replaces (pinned by golden tests in ``tests/unit/agents/test_message_hygiene.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from databricks_deep_research.llm.client import ToolCall
from databricks_deep_research.llm.roles import sanitize_history_messages

# Re-export under a name that reads as the consolidated message-hygiene entry
# point. The implementation is unchanged — see the module docstring.
sanitize_conversation_history = sanitize_history_messages

__all__ = [
    "fill_missing_tool_results",
    "sanitize_conversation_history",
]


def fill_missing_tool_results(
    messages: list[dict[str, Any]],
    tool_calls: Iterable[ToolCall],
    responded_ids: set[str],
    tool_message_builder: Callable[[str, str], dict[str, Any]],
) -> None:
    """Append an empty tool-result message for every unanswered tool call.

    Ensures ALL ``tool_calls`` in an assistant turn have a matching ``tool``
    result message, preventing "tool_use without tool_result" 400s from the
    Anthropic/Databricks gateway. A call id already present in *responded_ids*
    (because it was executed, served from cache, or budget-rejected) is left
    untouched; any remaining id gets an empty-content result via
    *tool_message_builder*.

    This is behavior-identical to the inline loop it replaces in
    ``react_loop.ReactLoop.execute``::

        for tc in response.tool_calls:
            if tc.id not in responded_tc_ids:
                messages.append(self._tool_msg(tc.id, ""))

    Args:
        messages: The in-progress message list. Mutated in place (matching the
            ReAct loop's existing append-as-it-goes pattern).
        tool_calls: The tool calls from the current assistant turn, in order.
        responded_ids: Call ids that already have a tool-result message.
        tool_message_builder: Builds a ``{role: "tool", ...}`` dict from
            ``(tool_call_id, content)`` — passed in so the exact message shape
            stays owned by the caller (``ReactLoop._tool_msg``).
    """
    for tc in tool_calls:
        if tc.id not in responded_ids:
            messages.append(tool_message_builder(tc.id, ""))
