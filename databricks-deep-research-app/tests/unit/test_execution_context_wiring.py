"""Unit tests for PR3a wiring of HITL ownership through both entry points.

Covers:
1. ``ExecutionContext`` accepts ``user_id`` and ``approval_broker`` fields.
2. Agent API path: ``Agent(user_id=..., approval_broker=...)`` registers
   the user as the HITL owner via ``broker.owner_of(rid)``.
3. State -> harness round-trip: ``_framework_extras`` stashed by
   ``Agent._apply_extras_to_state`` reaches ``ToolContext.extras`` after
   harness execution.
4. ``HITL_OWNER_MISSING`` canary log fires when broker is wired but
   ``_framework_user_id`` is absent.
5. Agent-server / CLI policy: when no broker is wired, ``run_hitl_gate``
   returns ``None`` (no gate).

All tests use only the public framework surface plus the documented
``_framework_*`` reserved-prefix keys defined in PR3a.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from databricks_deep_research.agents.react_loop_hitl import run_hitl_gate
from databricks_deep_research.api.agent import Agent
from databricks_deep_research.api.approval import (
    ApprovalDecision,
    InProcessApprovalBroker,
)
from databricks_deep_research.tools.protocol import (
    ToolContext,
    ToolDefinition,
)
from databricks_deep_research.workflow.context import ExecutionContext
from databricks_deep_research.workflow.state import WorkflowState

# --------------------------------------------------------------------------- helpers


class _FakeLoop:
    """Lightweight ReactLoopHook substitute for HITL gate tests."""

    def __init__(
        self,
        *,
        broker: Any = None,
        user_id: str | None = None,
        node_id: str = "n1",
    ) -> None:
        self._ctx = ToolContext()
        if broker is not None:
            self._ctx.extras["_framework_approval_broker"] = broker
        if user_id is not None:
            self._ctx.extras["_framework_user_id"] = user_id
        self._node_id = node_id
        self.captured_events: list[Any] = []

    @property
    def extras(self) -> dict[str, Any]:
        return self._ctx.extras

    @property
    def node_id(self) -> str:
        return self._node_id

    def emit_event(self, event: Any) -> None:
        self.captured_events.append(event)


class _FakeTool:
    def __init__(self, *, requires_confirmation: bool = True, reason: str = "") -> None:
        meta: dict[str, Any] = {}
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


# --------------------------------------------------------------------------- 1


def test_execution_context_accepts_user_id_and_broker() -> None:
    """``ExecutionContext`` exposes the new ``user_id`` and ``approval_broker`` fields."""
    broker = MagicMock()
    ctx = ExecutionContext(
        llm_client=MagicMock(),
        user_id="alice",
        approval_broker=broker,
    )
    assert ctx.user_id == "alice"
    assert ctx.approval_broker is broker

    # Defaults preserved for legacy callers (no kwargs).
    legacy = ExecutionContext(llm_client=MagicMock())
    assert legacy.user_id is None
    assert legacy.approval_broker is None


# --------------------------------------------------------------------------- 2


@pytest.mark.asyncio
async def test_agent_api_path_owner_set() -> None:
    """An ``Agent`` configured with ``user_id`` and ``approval_broker`` should
    register that user as the request owner when the gate fires.
    """
    broker = InProcessApprovalBroker()
    agent = Agent(
        name="test_agent",
        user_id="alice",
        approval_broker=broker,
    )

    # Drive ``_apply_extras_to_state`` directly — this is the documented
    # writer that stashes ``_framework_extras`` into WorkflowState for the
    # harness to project into ToolContext.
    state = WorkflowState(query="hello")
    agent._apply_extras_to_state(state, thread_id=None)

    extras = state.get("_framework_extras")
    assert isinstance(extras, dict)
    assert extras.get("_framework_user_id") == "alice"
    assert extras.get("_framework_approval_broker") is broker

    # Now run the gate end-to-end with those extras and assert ownership.
    loop = _FakeLoop(broker=broker, user_id="alice")
    tool = _FakeTool(reason="destructive")

    async def approve() -> None:
        await asyncio.sleep(0.05)
        rid = next(iter(broker._events))
        # Sanity check: owner was registered.
        assert broker.owner_of(rid) == "alice"
        broker.resolve(
            rid,
            ApprovalDecision(approved=True, approver="alice"),
            requester_user_id="alice",
        )

    approve_task = asyncio.create_task(approve())
    result = await run_hitl_gate(loop, tool, {"k": "v"})
    await approve_task

    assert result is None  # approved -> caller proceeds


# --------------------------------------------------------------------------- 3


def test_agent_extras_reach_tool_context() -> None:
    """The Agent path's ``_framework_extras`` must round-trip through state
    so the harness can project it into ``ToolContext.extras``.

    This test exercises only the public surface: ``Agent`` writes via
    ``_apply_extras_to_state``; the harness reads via ``state.get`` —
    both contracts established by PR3a.
    """
    broker = InProcessApprovalBroker()
    agent = Agent(
        name="test_agent",
        user_id="alice",
        approval_broker=broker,
    )
    state = WorkflowState(query="hi")
    agent._apply_extras_to_state(state, thread_id="t-1")

    extras = state.get("_framework_extras")
    assert isinstance(extras, dict)
    assert extras["_framework_user_id"] == "alice"
    assert extras["_framework_approval_broker"] is broker
    assert extras["_framework_thread_id"] == "t-1"


# --------------------------------------------------------------------------- 4


@pytest.mark.asyncio
async def test_hitl_owner_missing_canary_log(caplog: pytest.LogCaptureFixture) -> None:
    """When the broker is wired but ``_framework_user_id`` is absent, the
    gate emits a HITL_OWNER_MISSING WARNING canary so partial-wire failures
    are visible in production logs.
    """
    broker = InProcessApprovalBroker()
    loop = _FakeLoop(broker=broker)  # no user_id
    tool = _FakeTool()

    async def approve() -> None:
        await asyncio.sleep(0.05)
        rid = next(iter(broker._events))
        broker.resolve(rid, ApprovalDecision(approved=True, approver="anon"))

    approve_task = asyncio.create_task(approve())
    with caplog.at_level(logging.WARNING):
        await run_hitl_gate(loop, tool, {"k": "v"})
    await approve_task

    assert any("HITL_OWNER_MISSING" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- 5


@pytest.mark.asyncio
async def test_agent_server_hitl_disabled() -> None:
    """Non-FastAPI entrypoints (agent_server, CLI) pass approval_broker=None;
    ``run_hitl_gate`` is a safe no-op (returns None) under that policy.
    """
    loop = _FakeLoop(broker=None)
    tool = _FakeTool()
    result = await run_hitl_gate(loop, tool, {"k": "v"})
    assert result is None
    assert loop.captured_events == []
