"""HITL approval gate tests for the ReactLoop integration."""

from __future__ import annotations

import asyncio

import pytest

from databricks_deep_research.api.approval import (
    ApprovalDecision,
    InProcessApprovalBroker,
    requires_approval,
)
from databricks_deep_research.tools.api import tool


@pytest.mark.asyncio
async def test_broker_round_trip() -> None:
    broker = InProcessApprovalBroker()
    task = asyncio.create_task(
        broker.request("r1", "delete_record", {"id": "abc"}, timeout_seconds=2)
    )
    await asyncio.sleep(0.05)
    assert broker.is_pending("r1")
    ok = broker.resolve(
        "r1", ApprovalDecision(approved=True, approver="alice"),
    )
    assert ok is True
    decision = await task
    assert decision.approved is True
    assert decision.approver == "alice"


@pytest.mark.asyncio
async def test_broker_denied() -> None:
    broker = InProcessApprovalBroker()
    task = asyncio.create_task(
        broker.request("r2", "tool", {}, timeout_seconds=2)
    )
    await asyncio.sleep(0.05)
    broker.resolve("r2", ApprovalDecision(approved=False, reason="not approved"))
    decision = await task
    assert decision.approved is False
    assert decision.reason == "not approved"


@pytest.mark.asyncio
async def test_broker_timeout() -> None:
    broker = InProcessApprovalBroker()
    decision = await broker.request("r3", "tool", {}, timeout_seconds=0.1)
    assert decision.approved is False
    assert decision.reason == "timeout"


@pytest.mark.asyncio
async def test_broker_double_resolve_returns_false() -> None:
    broker = InProcessApprovalBroker()
    task = asyncio.create_task(
        broker.request("r4", "tool", {}, timeout_seconds=2)
    )
    await asyncio.sleep(0.05)
    assert broker.resolve("r4", ApprovalDecision(approved=True)) is True
    # second resolve returns False (409 Conflict)
    assert broker.resolve("r4", ApprovalDecision(approved=True)) is False
    await task


def test_broker_unknown_request_returns_false() -> None:
    broker = InProcessApprovalBroker()
    assert broker.resolve("nope", ApprovalDecision(approved=True)) is False


@pytest.mark.asyncio
async def test_request_id_collision_raises() -> None:
    broker = InProcessApprovalBroker()
    asyncio.create_task(broker.request("rX", "tool", {}, timeout_seconds=2))
    await asyncio.sleep(0.05)
    with pytest.raises(ValueError, match="request_id collision"):
        await broker.request("rX", "tool", {}, timeout_seconds=2)


def test_requires_approval_inner_then_outer() -> None:
    @tool
    @requires_approval(reason="destructive")
    def dangerous(x: str) -> str:
        """Dangerous op."""
        return x

    assert dangerous.definition.metadata["requires_confirmation"] is True
    assert dangerous.definition.metadata["approval_reason"] == "destructive"


def test_requires_approval_outer_then_inner() -> None:
    @requires_approval(reason="also destructive")
    @tool
    def dangerous(x: str) -> str:
        """Dangerous op."""
        return x

    assert dangerous.definition.metadata["requires_confirmation"] is True
    assert dangerous.definition.metadata["approval_reason"] == "also destructive"


def test_requires_confirmation_kwarg_path() -> None:
    @tool(requires_confirmation=True)
    def dangerous(x: str) -> str:
        """Dangerous op."""
        return x

    assert dangerous.definition.metadata["requires_confirmation"] is True


def test_default_tool_has_no_confirmation_flag() -> None:
    @tool
    def normal(x: str) -> str:
        """Normal op."""
        return x

    assert normal.definition.metadata.get("requires_confirmation") is None
