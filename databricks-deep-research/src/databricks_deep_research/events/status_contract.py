"""Typed terminal-status contract for async units (node / subagent / tool).

A single typed status enum, pinned to BOTH the backend (this module + the
event fields in ``types.py``) and the frontend (``activityLabels.ts``) by ONE
shared fixture: ``databricks-deep-research-app/contracts/run_status_contract.json``.

This kills the recurring "frontend-gate-vs-backend-prose" status drift: a
framework pytest asserts the JSON enum equals ``get_args(RunStatus)``, and a
frontend vitest asserts the TS label map keys equal the same JSON — so either
side drifting fails CI.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

# The single typed terminal-status enum for every async unit.
#
# ``safety_termination`` is included so feature 2.3's safety detector (US-107)
# has a terminal status to stamp without re-touching this contract.
RunStatus = Literal[
    "running",
    "completed",
    "failed",
    "cancelled",
    "timed_out",
    "polling_timed_out",
    "budget_exceeded",
    "skipped",
    "safety_termination",
]


def make_status_kwargs(status: RunStatus, *, error: str | None = None) -> dict[str, Any]:
    """Build the status (and optional error) kwargs for a terminal event.

    Loud-fails on an out-of-enum status (the DeerFlow status_contract pattern)
    so a typo cannot silently produce an unrecognized terminal status.

    Args:
        status: One of the ``RunStatus`` literal values.
        error: Optional error string to attach (omitted when falsy).

    Returns:
        A kwargs dict containing ``status`` and, if provided, ``error``.

    Raises:
        ValueError: If ``status`` is not a member of ``RunStatus``.
    """
    if status not in get_args(RunStatus):
        raise ValueError(f"out-of-enum status {status!r}")
    out: dict[str, Any] = {"status": status}
    if error:
        out["error"] = error
    return out
