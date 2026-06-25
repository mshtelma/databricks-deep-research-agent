"""Detect provider safety/content-policy terminations from an LLM response.

When a model-serving provider truncates a generation for a safety or
content-policy reason, the partial response can contain *dangling* tool calls —
an assistant turn that requested a tool but was cut off before completing. If
those partial calls are dispatched and their results replayed, the next request
to the provider 400s ("tool_use without tool_result" / malformed tool sequence).
We already fill dangling tool-result slots (see
:func:`databricks_deep_research.agents.message_hygiene.fill_missing_tool_results`),
but that does not help when the *cause* of the truncation is a safety stop: the
right behavior is to SUPPRESS the partial calls and surface a terminal status
(``RunStatus`` ``safety_termination``) rather than to keep looping.

This module is a plain-Python strategy map (no LangChain / provider SDKs) from a
provider's finish/stop reason to a normalized verdict. The framework reaches
these providers through the Databricks OpenAI-compatible gateway, which usually
normalizes to OpenAI-style ``finish_reason`` values but can leak the provider's
native reason — so the map covers BOTH the OpenAI-normalized and the native
Anthropic / Gemini vocabularies. Matching is case-insensitive.

A BENIGN exclusion set guarantees that normal completion (``stop`` / ``end_turn``)
and output-length truncation (``length`` / ``max_tokens`` / ``MAX_TOKENS``) are
NEVER classified as safety terminations — a budget cut-off is recoverable and
must not suppress tool calls.

Verified provider vocabularies (sources: OpenAI Chat Completions, Anthropic
Messages, Gemini ``generateContent``):

- OpenAI ``finish_reason``: ``stop``, ``length``, ``tool_calls``,
  ``content_filter`` (safety), ``function_call``.
- Anthropic ``stop_reason``: ``end_turn``, ``max_tokens``, ``stop_sequence``,
  ``tool_use``, ``pause_turn``, ``refusal`` (safety).
- Gemini ``finishReason``: ``STOP``, ``MAX_TOKENS``, ``SAFETY``, ``RECITATION``,
  ``BLOCKLIST``, ``PROHIBITED_CONTENT``, ``SPII`` (all but the first two are
  safety/content-policy).
"""

from __future__ import annotations

from typing import Protocol


class _HasFinishReason(Protocol):
    """Structural type for anything carrying a finish/stop reason.

    ``LLMResponse`` (``llm.client``) satisfies this via its ``finish_reason``
    field. Kept as a Protocol so the detector does not import the concrete
    response type and stays unit-testable with a tiny stub. Declared as a
    read-only ``property`` so a frozen dataclass (``LLMResponse``) — whose
    attributes are read-only — structurally matches.
    """

    @property
    def finish_reason(self) -> str: ...


# Provider safety/content-policy reasons → the normalized verdict string.
# Keys are lower-cased; lookups lower-case the response value first.
#
# This is the strategy map: each entry maps a provider's native (or
# OpenAI-normalized) terminal reason to the single normalized verdict
# ``"safety_terminated"``. Grouped by provider for auditability; the lookup is
# provider-agnostic (the gateway does not reliably tell us which provider
# produced a given reason).
_SAFETY_REASONS: dict[str, str] = {
    # OpenAI (and the gateway's OpenAI-normalized form for every family).
    "content_filter": "safety_terminated",
    # Anthropic native.
    "refusal": "safety_terminated",
    # Gemini native.
    "safety": "safety_terminated",
    "recitation": "safety_terminated",
    "blocklist": "safety_terminated",
    "prohibited_content": "safety_terminated",
    "spii": "safety_terminated",
}


# Reasons that look terminal but are explicitly NOT safety — normal completion
# and output-length truncation. Anything in this set short-circuits to "benign"
# so a budget cut-off can never be misclassified as a safety stop. Lower-cased.
_BENIGN_REASONS: frozenset[str] = frozenset(
    {
        # Normal completion.
        "stop",          # OpenAI
        "end_turn",      # Anthropic
        "stop_sequence",  # Anthropic (hit a stop string)
        # Output-length truncation (recoverable — must NOT be suppressed).
        "length",        # OpenAI
        "max_tokens",    # Anthropic + Gemini (gemini uppercases; normalized here)
        # Benign tool / control flow.
        "tool_calls",    # OpenAI (model wants a tool)
        "tool_use",      # Anthropic
        "function_call",  # OpenAI (legacy)
        "pause_turn",    # Anthropic (resumable server-tool pause)
        # Empty / unknown reason — treated as benign (fail-open: never suppress
        # tool calls on an ambiguous signal). Gemini's catch-all ``OTHER`` is
        # likewise not a confirmed safety stop, so it is benign here.
        "",
        "other",
    }
)


def classify_termination_reason(reason: str | None) -> str | None:
    """Classify a raw finish/stop *reason* string.

    Returns the normalized verdict (``"safety_terminated"``) when the reason is a
    known safety/content-policy stop, or ``None`` for benign / unknown reasons.

    The BENIGN set is checked first so a value that somehow appears in both
    tables (it never does today) cannot be classified as safety. An unknown
    reason returns ``None`` (fail-open): we only suppress on a *confirmed* safety
    signal, never on an unrecognized one.
    """
    if reason is None:
        return None
    normalized = reason.strip().lower()
    if normalized in _BENIGN_REASONS:
        return None
    return _SAFETY_REASONS.get(normalized)


def safety_termination_reason(response: _HasFinishReason) -> str | None:
    """Return the normalized safety verdict for *response*, else ``None``.

    Reads ``response.finish_reason`` — the only terminal signal the framework's
    ``LLMResponse`` carries. Streaming responses built without a finish reason
    default to ``"stop"`` (benign), so this is safely a no-op on the streaming
    path (which therefore never suppresses; documented limitation).
    """
    return classify_termination_reason(getattr(response, "finish_reason", None))


def is_safety_terminated(response: _HasFinishReason) -> bool:
    """Whether *response* was terminated for a safety/content-policy reason.

    Convenience boolean wrapper over :func:`safety_termination_reason`. Benign
    reasons (normal stop, ``length`` / ``max_tokens``) return ``False``.
    """
    return safety_termination_reason(response) is not None
