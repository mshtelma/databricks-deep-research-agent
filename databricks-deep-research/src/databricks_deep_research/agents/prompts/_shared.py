"""Shared prompt fragments used across multiple builtin agents.

Putting the temporal anchor (and any future cross-cutting prompt fragments)
in one place prevents per-agent drift. If we improve the wording, it's a
single edit.

The fragment is concatenated into each agent's system prompt via
``from ._shared import TEMPORAL_ANCHOR_BLOCK`` at module load time, so
adding new agents that need temporal grounding only requires importing the
block — no per-agent re-wording.
"""

from __future__ import annotations

# Referenced template variables (substituted by ``SafeTemplateRenderer``):
#   {current_date}      — ISO date, e.g. "2026-05-19"
#   {current_timezone}  — IANA-style timezone name, e.g. "UTC"
#
# Both are auto-injected by the agent harness via
# ``PromptTemporalContext.now()``. Tests can override either value by
# pre-populating ``context["current_date"]`` / ``context["current_timezone"]``
# before the harness runs.
TEMPORAL_ANCHOR_BLOCK: str = """## Temporal Anchor
Today's date: {current_date} ({current_timezone}). When sources reference fiscal periods, quarters, or dated events, anchor every interpretation against this date. If a source's own date is missing or ambiguous, flag the uncertainty in your output rather than silently treating it as current."""


__all__ = ["TEMPORAL_ANCHOR_BLOCK"]
