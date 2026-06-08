"""Gate (HITL) events emitted by the ReactLoop when ``@requires_approval``
tools fire and an :class:`ApprovalBroker` is attached.

Discriminator: ``event_type``. Joined into :class:`FrameworkEvent` via
``events/types.py``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from databricks_deep_research.events.types import StreamEvent


class GateWaitingEvent(StreamEvent):
    """A tool call is waiting on an external approval decision."""

    event_type: Literal["gate_waiting"] = "gate_waiting"
    node_id: str = ""
    timestamp: str = ""
    request_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class GateResumedEvent(StreamEvent):
    """An approval decision arrived and the tool is being invoked."""

    event_type: Literal["gate_resumed"] = "gate_resumed"
    node_id: str = ""
    timestamp: str = ""
    request_id: str
    approver: str | None = None


class GateDeniedEvent(StreamEvent):
    """The approval was denied; the tool returns a denial marker."""

    event_type: Literal["gate_denied"] = "gate_denied"
    node_id: str = ""
    timestamp: str = ""
    request_id: str
    reason: str | None = None
    approver: str | None = None


class GateTimeoutEvent(StreamEvent):
    """The approval window elapsed without a decision; treated as denial."""

    event_type: Literal["gate_timeout"] = "gate_timeout"
    node_id: str = ""
    timestamp: str = ""
    request_id: str
    elapsed_seconds: float = 0.0


__all__ = [
    "GateWaitingEvent",
    "GateResumedEvent",
    "GateDeniedEvent",
    "GateTimeoutEvent",
]
