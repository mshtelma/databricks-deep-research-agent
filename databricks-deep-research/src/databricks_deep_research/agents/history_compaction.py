"""Value-ranked history compaction for the ReAct loop (spec §1.3).

Pure, fully-typed helpers (no LLM, no LangChain, no imports from ``react_loop``
to avoid an import cycle). When the conversation grows past budget, the ReAct
loop must shed bytes — but it should shed *low-value* tool results (those that
produced no accepted sources) before *high-value* ones (accepted/cited). This
module ranks the old tool messages lowest-value-first and produces the compacted
form of a single message.

The two non-offload compaction strategies (``line_preserving_truncate`` /
``hard_clip``) live in :mod:`databricks_deep_research.agents.tool_offload` and are
the single source of truth — this module imports them rather than re-deriving the
rung logic. An *offloaded* message (whose full result already lives in the
compute scratchpad under a handle) compacts to a one-line pointer instead.

Value tiers (lowest → highest), mirroring ``_tool_result_cacheable``:

* ``unaccepted`` — ``accepted_source_count == 0`` and not a builtin tool.
* ``builtin`` — ``evidence_quality == "builtin"`` (compute/namespace tools).
* ``accepted`` — ``accepted_source_count > 0`` (top-protected tier mid-loop;
  "cited" is a post-synthesis concept and is unavailable here, so ACCEPTED is the
  highest tier we can rank by).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from databricks_deep_research.agents.tool_offload import (
    hard_clip,
    line_preserving_truncate,
)

# Value-tier ranks. Lower rank == lower value == compacted first.
RANK_UNACCEPTED = 0
RANK_BUILTIN = 1
RANK_ACCEPTED = 2


@dataclass(frozen=True)
class RankedToolResult:
    """One old tool message ranked for compaction, lowest-value-first."""

    index: int  # position in the ``messages`` list
    tool_call_id: str
    rank: int  # one of RANK_UNACCEPTED / RANK_BUILTIN / RANK_ACCEPTED
    value: dict[str, Any]  # the side-table value metadata for this message


def value_rank(value: dict[str, Any]) -> int:
    """Return the value tier of a tool result from its side-table metadata.

    Accepted (``accepted_source_count > 0``) outranks builtin, which outranks
    unaccepted. Mirrors the value signals in ``react_loop._tool_result_cacheable``.
    """
    try:
        accepted = int(value.get("accepted_source_count", 0) or 0)
    except (TypeError, ValueError):
        accepted = 0
    if accepted > 0:
        return RANK_ACCEPTED
    if str(value.get("evidence_quality", "")) == "builtin":
        return RANK_BUILTIN
    return RANK_UNACCEPTED


def rank_tool_results(
    messages: list[dict[str, Any]],
    value_by_tcid: dict[str, dict[str, Any]],
    *,
    upto: int | None = None,
) -> list[RankedToolResult]:
    """Rank old tool messages lowest-value-first for compaction.

    Only ``role == "tool"`` messages whose ``tool_call_id`` has value metadata in
    ``value_by_tcid`` are considered. When ``upto`` is given, only messages with
    ``index < upto`` are ranked (the caller keeps recent iterations intact).

    The result is sorted by ``(rank, index)`` so that, within a tier, earlier
    (older) messages are compacted before later ones — a stable, deterministic
    order. Lowest-value messages come first.
    """
    ranked: list[RankedToolResult] = []
    for i, msg in enumerate(messages):
        if upto is not None and i >= upto:
            break
        if msg.get("role") != "tool":
            continue
        tc_id = msg.get("tool_call_id")
        if not isinstance(tc_id, str):
            continue
        value = value_by_tcid.get(tc_id)
        if value is None:
            continue
        ranked.append(
            RankedToolResult(
                index=i,
                tool_call_id=tc_id,
                rank=value_rank(value),
                value=value,
            )
        )
    ranked.sort(key=lambda r: (r.rank, r.index))
    return ranked


def compact_tool_message(
    content: str,
    value: dict[str, Any],
    *,
    max_chars: int,
    line_preserving: bool,
) -> str:
    """Return the compacted form of one tool message.

    * Offloaded result (``offload_handle`` set in ``value``) => a one-line pointer
      ``"[result in compute var <handle>]"``; the full result already lives in the
      compute scratchpad, so the model needs only the handle.
    * Otherwise delegate to the §1.2 rungs (single source of truth in
      ``tool_offload``): ``line_preserving_truncate`` when ``line_preserving`` is
      True, else ``hard_clip``.
    """
    handle = value.get("offload_handle")
    if isinstance(handle, str) and handle:
        return f"[result in compute var {handle}]"
    if line_preserving:
        return line_preserving_truncate(content, max_chars=max_chars)
    return hard_clip(content, max_chars)
