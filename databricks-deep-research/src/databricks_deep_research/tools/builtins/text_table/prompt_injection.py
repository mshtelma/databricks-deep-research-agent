"""Render BOUND text-table bindings into a deterministic prompt block.

The block is intended for prepending to a system prompt so the LLM knows
which tables it can address by ``binding`` name (rather than fully-qualified
name) when calling the ``table_*`` tools. Properties:

- **BOUND-only.** ``DISCOVERED`` bindings are runtime-discovered and not
  yet validated for prompt inclusion. They never appear here.
- **Deterministic order.** Sorted by binding name so prompt caches stay
  warm regardless of registration order.
- **PII redaction.** Free-form ``description`` text is scrubbed for emails,
  phone numbers, and SSN-shaped sequences before emission. Identifiers
  (binding name, FQN, role columns) are NOT redacted — they are
  structurally validated upstream and must round-trip verbatim.
- **Token budget cap.** Output is truncated when an estimated token count
  would exceed ``max_tokens``. The default estimator uses a 4-chars-per-
  token heuristic; callers can pass a custom ``token_estimator`` (e.g.
  a real BPE-aware counter).

The function is pure and stateless: it takes a registry snapshot and
emits a ``str``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .binding import BindingInfo, BindingSource
from .registry import TableBindingRegistry

__all__ = ["render_table_bindings_prompt"]

# ---------------------------------------------------------------------------
# PII redaction patterns. Order matters: SSN BEFORE phone, because the SSN
# pattern is more specific and would otherwise be partially consumed by the
# phone matcher.
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Phone matches optional + and country code, then 10+ digits split by spaces,
# parens, dots, or dashes. Conservative — won't catch all formats, but covers
# the common US/E.164-ish shapes.
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?"  # country code
    r"(?:\(?\d{3}\)?[\s.-]?)"  # area code
    r"\d{3}[\s.-]?\d{4}"  # local
)


def _redact(text: str) -> str:
    """Scrub emails, SSNs, and phone numbers from free-form text."""
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _SSN_RE.sub("[redacted-ssn]", text)
    text = _PHONE_RE.sub("[redacted-phone]", text)
    return text


# ---------------------------------------------------------------------------
# Token estimation. Default to a 4-char-per-token heuristic so the framework
# stays free of a heavy BPE dependency. Callers with access to a real
# tokenizer can pass one through ``token_estimator``.
# ---------------------------------------------------------------------------

_DEFAULT_CHARS_PER_TOKEN = 4


def _default_estimator(text: str) -> int:
    if not text:
        return 0
    # Round up so short strings are not zero.
    return (len(text) + _DEFAULT_CHARS_PER_TOKEN - 1) // _DEFAULT_CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Per-binding rendering.
# ---------------------------------------------------------------------------


def _render_binding_block(info: BindingInfo) -> str:
    """Render one binding as a markdown-ish block.

    Format::

        - binding: <name>
          fqn: <fqn>
          description: <redacted-description>
          roles:
            id: <id_column>
            content: <content_column>
            partition: <partition_column>     # only when set
            order: <order_column>             # only when set
            label: <label_column>             # only when set
          numeric_columns: [c1, c2]            # only when non-empty
    """
    lines: list[str] = []
    lines.append(f"- binding: {info.name}")
    lines.append(f"  fqn: {info.fqn}")
    if info.description:
        lines.append(f"  description: {_redact(info.description)}")
    if info.roles is not None:
        roles = info.roles
        role_lines: list[str] = []
        role_lines.append(f"    id: {roles.id_column}")
        role_lines.append(f"    content: {roles.content_column}")
        if roles.partition_column:
            role_lines.append(f"    partition: {roles.partition_column}")
        if roles.order_column:
            role_lines.append(f"    order: {roles.order_column}")
        if roles.label_column:
            role_lines.append(f"    label: {roles.label_column}")
        if roles.type_column:
            role_lines.append(f"    type: {roles.type_column}")
        if roles.date_column:
            role_lines.append(f"    date: {roles.date_column}")
        if role_lines:
            lines.append("  roles:")
            lines.extend(role_lines)
    if info.numeric_columns:
        cols = ", ".join(info.numeric_columns)
        lines.append(f"  numeric_columns: [{cols}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


_HEADER = "## Available text tables"
_FOOTER_TRUNCATED = "[truncated: token budget exceeded]"


def render_table_bindings_prompt(
    registry: TableBindingRegistry,
    *,
    max_tokens: int = 1024,
    token_estimator: Callable[[str], int] | None = None,
) -> str:
    """Render BOUND bindings into a deterministic prompt block.

    Parameters
    ----------
    registry:
        The :class:`TableBindingRegistry` to read from.
    max_tokens:
        Hard cap on the estimated token count of the rendered string.
        When the running total would exceed this, rendering stops and a
        ``[truncated: ...]`` marker is appended. Must be a positive int.
    token_estimator:
        Callable that maps a string to an integer token count. Defaults
        to a 4-chars-per-token heuristic. Pass a real tokenizer (e.g.
        ``len(enc.encode(text))`` from ``tiktoken``) for tighter packing.

    Returns
    -------
    str
        The rendered block, or an empty string when no BOUND bindings
        are registered.
    """
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError(
            f"max_tokens must be a positive integer; got {max_tokens!r}"
        )

    estimator = token_estimator or _default_estimator

    snapshot = registry.metadata_snapshot()
    bound_infos = sorted(
        (info for info in snapshot.values() if info.source is BindingSource.BOUND),
        key=lambda info: info.name,
    )
    if not bound_infos:
        return ""

    pieces: list[str] = [_HEADER]
    running = estimator(_HEADER)
    truncated = False

    for info in bound_infos:
        block = _render_binding_block(info)
        block_with_sep = "\n" + block
        cost = estimator(block_with_sep)
        # Reserve room for the truncation footer in the budget.
        footer_cost = estimator("\n" + _FOOTER_TRUNCATED)
        if running + cost + footer_cost > max_tokens:
            truncated = True
            break
        pieces.append(block)
        running += cost

    if truncated:
        pieces.append(_FOOTER_TRUNCATED)

    return "\n".join(pieces)
