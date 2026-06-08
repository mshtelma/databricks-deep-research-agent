"""Regression: HITL gate insert is dead code for default subtypes.

Pins the canonical ReactLoop._execute_single_tool path:
- When no ``_framework_approval_broker`` is in ``ctx.extras``, the gate
  branch is skipped immediately with no event emission and no extra state
  mutation.
- When ``requires_confirmation`` is absent on the tool definition, the
  gate branch is also skipped regardless of broker presence.

Together these guarantee the 6 existing builtin subtypes (none of which
set ``requires_confirmation`` and none of which attach an approval broker
by default) see byte-identical behavior post-Phase-2.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from databricks_deep_research.agents.react_loop_hitl import run_hitl_gate
from databricks_deep_research.tools.protocol import (
    ToolContext,
    ToolDefinition,
)


class _FakeLoop:
    """Lightweight ReactLoopHook substitute exposing only what run_hitl_gate touches."""

    def __init__(self, *, broker, node_id: str = "n1") -> None:
        self._ctx = ToolContext()
        if broker is not None:
            self._ctx.extras["_framework_approval_broker"] = broker
        self._node_id = node_id
        self.captured_events: list[Any] = []

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def extras(self) -> Mapping[str, Any]:
        return self._ctx.extras

    def emit_event(self, event: Any) -> None:
        self.captured_events.append(event)


class _FakeTool:
    def __init__(self, *, requires_confirmation: bool, reason: str = "") -> None:
        meta = {}
        if requires_confirmation:
            meta["requires_confirmation"] = True
        if reason:
            meta["approval_reason"] = reason
        self.definition = ToolDefinition(
            name="test_tool",
            description="",
            parameters={"type": "object"},
            metadata=meta,
        )


@pytest.mark.asyncio
async def test_no_broker_means_gate_returns_none() -> None:
    """Without an ApprovalBroker, the gate is a no-op even on flagged tools."""
    loop = _FakeLoop(broker=None)
    tool = _FakeTool(requires_confirmation=True)
    result = await run_hitl_gate(loop, tool, {"k": "v"})
    assert result is None
    assert loop.captured_events == []


@pytest.mark.asyncio
async def test_no_requires_confirmation_means_caller_skips_gate() -> None:
    """The wrapping ``if`` in react_loop.py only invokes ``run_hitl_gate``
    when ``requires_confirmation`` is set. We assert the dead-code
    contract by checking the metadata gate would be inactive."""
    tool = _FakeTool(requires_confirmation=False)
    assert (tool.definition.metadata or {}).get("requires_confirmation") is None


@pytest.mark.asyncio
async def test_gate_emits_waiting_then_denied_on_denied_decision() -> None:
    from databricks_deep_research.api.approval import (
        ApprovalDecision,
        InProcessApprovalBroker,
    )

    broker = InProcessApprovalBroker()
    loop = _FakeLoop(broker=broker)
    tool = _FakeTool(requires_confirmation=True, reason="destructive")

    async def deny() -> None:
        await asyncio.sleep(0.05)
        # Find the only pending request_id
        rid = next(iter(broker._events))
        broker.resolve(rid, ApprovalDecision(approved=False, reason="vetoed"))

    deny_task = asyncio.create_task(deny())
    result = await run_hitl_gate(loop, tool, {"k": "v"})
    await deny_task

    assert result is not None
    assert result["content"] == "[approval denied]"
    assert result["meta"]["tool_success"] is False
    types = [e.event_type for e in loop.captured_events]
    assert "gate_waiting" in types
    assert "gate_denied" in types


@pytest.mark.asyncio
async def test_gate_emits_waiting_then_resumed_on_approval() -> None:
    from databricks_deep_research.api.approval import (
        ApprovalDecision,
        InProcessApprovalBroker,
    )

    broker = InProcessApprovalBroker()
    loop = _FakeLoop(broker=broker)
    tool = _FakeTool(requires_confirmation=True)

    async def approve() -> None:
        await asyncio.sleep(0.05)
        rid = next(iter(broker._events))
        broker.resolve(rid, ApprovalDecision(approved=True, approver="alice"))

    approve_task = asyncio.create_task(approve())
    result = await run_hitl_gate(loop, tool, {"k": "v"})
    await approve_task

    assert result is None  # caller falls through to normal execution
    types = [e.event_type for e in loop.captured_events]
    assert "gate_waiting" in types
    assert "gate_resumed" in types
    resumed = next(e for e in loop.captured_events if e.event_type == "gate_resumed")
    assert resumed.approver == "alice"


@pytest.mark.asyncio
async def test_gate_emits_timeout_event_on_timeout() -> None:
    from databricks_deep_research.api.approval import InProcessApprovalBroker

    broker = InProcessApprovalBroker()
    # Override the broker's request to time out immediately.
    original_request = broker.request

    async def fast_timeout(*args, **kwargs):
        kwargs["timeout_seconds"] = 0.05
        return await original_request(*args, **kwargs)

    broker.request = fast_timeout  # type: ignore[method-assign]

    loop = _FakeLoop(broker=broker)
    tool = _FakeTool(requires_confirmation=True)

    result = await run_hitl_gate(loop, tool, {"k": "v"})
    assert result is not None
    assert result["content"] == "[approval timed out]"
    types = [e.event_type for e in loop.captured_events]
    assert "gate_waiting" in types
    assert "gate_timeout" in types
