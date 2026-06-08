"""Temporal grounding for agent prompts.

Encapsulates the "what is today?" injection so that:
  * Production callers get UTC now() without any prompt touching ``datetime``.
  * Tests can supply a fixed-clock context to make prompt rendering and LLM
    interactions deterministic.
  * Future timezone configurability (per-user, per-deployment) is a single
    constructor argument away.

Used by the agent harness at the top of every builtin agent invocation; the
emitted keys are consumed by the per-agent prompt templates via the
``{current_date}`` / ``{current_iso_datetime}`` / ``{current_timezone}``
``SafeTemplateRenderer`` slots.

Example
-------
Production (UTC):

    from databricks_deep_research.agents.temporal import PromptTemporalContext

    ctx = PromptTemporalContext.now()
    # → PromptTemporalContext(current_date="2026-05-19", ...)

Tests (fixed clock):

    ctx = PromptTemporalContext(
        current_date="2024-01-15",
        current_iso_datetime="2024-01-15T00:00:00+00:00",
        timezone_name="UTC",
    )
"""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo
from typing import Any

from pydantic import BaseModel, Field


class PromptTemporalContext(BaseModel):
    """Time-of-run values that every agent prompt may reference.

    Construction is via :meth:`now` (production) or direct instantiation
    (tests). The :meth:`as_context_keys` method emits a dict suitable for
    splatting into the harness's ``context`` dict.

    Attributes
    ----------
    current_date : str
        ISO-8601 date, e.g. ``"2026-05-19"``. 90% of prompts only need this.
    current_iso_datetime : str
        ISO-8601 datetime with seconds precision and explicit offset, e.g.
        ``"2026-05-19T13:36:25+00:00"``.
    timezone_name : str
        IANA-style name (``UTC``, ``America/New_York``, ...). Lets prompts
        disambiguate when the time matters more than the date.
    """

    current_date: str = Field(min_length=10, max_length=10)
    current_iso_datetime: str = Field(min_length=19)
    timezone_name: str = Field(default="UTC", min_length=1)

    model_config = {"frozen": True}

    @classmethod
    def now(cls, *, tz: tzinfo = UTC, tz_name: str = "UTC") -> "PromptTemporalContext":
        """Construct from the system clock.

        Defaults to UTC for cross-deployment consistency. Callers (e.g., an
        app that wants user-local time) can pass a different timezone.

        Parameters
        ----------
        tz : tzinfo
            Timezone to anchor the clock against. Defaults to UTC.
        tz_name : str
            Human-readable name of the timezone, e.g. ``"UTC"`` or
            ``"America/New_York"``. The ``tzinfo`` object's own name may be
            an opaque identifier (e.g., on Linux for system-tz objects), so
            we accept the human-readable form separately.
        """
        now = datetime.now(tz=tz)
        return cls(
            current_date=now.strftime("%Y-%m-%d"),
            current_iso_datetime=now.isoformat(timespec="seconds"),
            timezone_name=tz_name,
        )

    def as_context_keys(self) -> dict[str, Any]:
        """Return a dict ready to splat into the harness's ``context`` dict.

        The shape matches the template-variable names referenced by the
        ``TEMPORAL_ANCHOR_BLOCK`` shared prompt constant.
        """
        return {
            "current_date": self.current_date,
            "current_iso_datetime": self.current_iso_datetime,
            "current_timezone": self.timezone_name,
        }


__all__ = ["PromptTemporalContext"]
