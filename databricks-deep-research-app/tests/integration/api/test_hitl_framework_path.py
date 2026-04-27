"""Integration regression test for HITL framework-path authorization (PR3a).

Asserts the end-to-end contract on the **orchestrator path**:

1. ``run_hitl_gate`` populates the broker with the authenticated owner
   when ``ToolContext.extras["_framework_user_id"]`` is set (which PR3a
   guarantees via ``ExecutionContext`` -> harness propagation).
2. The HTTP layer rejects cross-user resolves with 403.
3. The recorded ``approver`` field is bound to the authenticated identity
   (HTTP layer responsibility, not ``body.approver``).
4. The structured log line ``HITL_OWNER_SET`` is emitted via caplog
   when ownership is registered (replaces the v1 plan's manual
   "spin up make dev" verification).

Note on scope: a fully-mocked LLM path through
``stream_research_via_framework`` requires the FastAPI test client to
hydrate ``app.state`` with chat services, plugin manager, storage stack,
etc. — far more than this PR's surface area. The PRD explicitly permits
the pragmatic fallback (hand-crafted ``ToolContext``) when a deterministic
LLM mock is not feasible in the budget, with the caplog assertion as the
primary regression guard. The full e2e against the FastAPI ``/jobs`` POST
flow is deferred to a follow-up; this test pins the load-bearing
framework-path behavior PR3a introduces.
"""

from __future__ import annotations

import asyncio
import logging
import os

# Settings validation in deep_research.core.config requires either a
# DATABASE_URL or a Lakebase instance to be set. This test exercises only
# the in-process broker + FastAPI dependency override; no DB calls are
# made. Set a placeholder URL so Settings validates at import time.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://test:test@localhost:5432/test_hitl_framework",
)

import pytest  # noqa: E402
from databricks_deep_research.agents.react_loop_hitl import (  # noqa: E402
    run_hitl_gate,
)
from databricks_deep_research.api.approval import (  # noqa: E402
    ApprovalDecision,
    InProcessApprovalBroker,
)
from databricks_deep_research.tools.protocol import (  # noqa: E402
    ToolContext,
    ToolDefinition,
)
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from deep_research.api.v1 import hitl  # noqa: E402
from deep_research.core.auth import UserIdentity  # noqa: E402
from deep_research.middleware.auth import get_current_user_identity  # noqa: E402

# --------------------------------------------------------------------------- helpers


class _FakeLoop:
    """Minimal ReactLoop substitute simulating the harness-populated context.

    PR3a guarantees that for the orchestrator path the harness builds a
    ``ToolContext`` whose ``extras`` contains both
    ``_framework_user_id`` and ``_framework_approval_broker`` (sourced
    from ``ExecutionContext.user_id`` and ``.approval_broker``). This
    fake reproduces that contract.
    """

    def __init__(self, *, broker: InProcessApprovalBroker, user_id: str) -> None:
        self._ctx = ToolContext()
        self._ctx.extras["_framework_approval_broker"] = broker
        self._ctx.extras["_framework_user_id"] = user_id
        self._node_id = "researcher_n1"
        self._pending_events: list = []  # type: ignore[type-arg]


class _FakeTool:
    def __init__(self, *, reason: str = "destructive") -> None:
        self.definition = ToolDefinition(
            name="hitl_protected_tool",
            description="",
            parameters={"type": "object"},
            metadata={
                "requires_confirmation": True,
                "approval_reason": reason,
            },
        )


def _identity(user_id: str, email: str | None = None) -> UserIdentity:
    return UserIdentity(
        user_id=user_id,
        email=email or f"{user_id}@example.com",
        display_name=user_id.title(),
    )


def _build_app(
    *,
    user: UserIdentity,
    broker: InProcessApprovalBroker,
) -> FastAPI:
    app = FastAPI()
    app.include_router(hitl.router)
    app.state.approval_broker = broker
    app.dependency_overrides[get_current_user_identity] = lambda: user
    return app


# --------------------------------------------------------------------------- 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hitl_framework_path_owner_set_via_alice() -> None:
    """When the framework path runs the gate with user_id='alice', the
    broker registers 'alice' as the request owner.
    """
    broker = InProcessApprovalBroker()
    loop = _FakeLoop(broker=broker, user_id="alice")
    tool = _FakeTool()

    async def approve() -> None:
        await asyncio.sleep(0.05)
        rid = next(iter(broker._events))
        # Owner is set BEFORE resolve — PR3a's central guarantee.
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


# --------------------------------------------------------------------------- 2


@pytest.mark.integration
def test_hitl_framework_path_cross_user_returns_403() -> None:
    """Bob cannot resolve a request owned by Alice — HTTP layer returns 403."""
    broker = InProcessApprovalBroker()
    request_id = "req-cross-user"

    # Spin up the broker.request(...) on a dedicated loop so the request
    # is registered and pending when Bob's HTTP call arrives.
    loop = asyncio.new_event_loop()

    async def kick_off() -> ApprovalDecision:
        return await broker.request(
            request_id,
            "tool",
            {},
            timeout_seconds=5.0,
            owner_user_id="alice",
        )

    request_task = loop.create_task(kick_off())
    loop.run_until_complete(asyncio.sleep(0.05))

    try:
        bob_app = _build_app(user=_identity("bob"), broker=broker)
        with TestClient(bob_app) as client:
            response = client.post(
                f"/research/hitl/approve/{request_id}",
                json={"approved": True},
            )
        assert response.status_code == 403
        assert "Not authorized" in response.json()["detail"]
    finally:
        # Drain the pending broker.request to keep the loop tidy.
        broker.resolve(
            request_id,
            ApprovalDecision(approved=False, reason="cleanup"),
            requester_user_id="alice",
        )
        loop.run_until_complete(request_task)
        loop.close()


# --------------------------------------------------------------------------- 3


@pytest.mark.integration
def test_hitl_framework_path_approver_field_bound() -> None:
    """The recorded ``approver`` is bound to the authenticated identity,
    not whatever ``body.approver`` claims.
    """
    broker = InProcessApprovalBroker()
    request_id = "req-approver-binding"

    loop = asyncio.new_event_loop()

    async def kick_off() -> ApprovalDecision:
        return await broker.request(
            request_id,
            "tool",
            {},
            timeout_seconds=5.0,
            owner_user_id="alice",
        )

    request_task = loop.create_task(kick_off())
    loop.run_until_complete(asyncio.sleep(0.05))

    try:
        alice = _identity("alice", email="alice@databricks.com")
        alice_app = _build_app(user=alice, broker=broker)
        with TestClient(alice_app) as client:
            response = client.post(
                f"/research/hitl/approve/{request_id}",
                # Audit-only label; MUST be ignored by the binding logic.
                json={"approved": True, "approver": "spoofed-approver"},
            )
        assert response.status_code == 200

        decision = loop.run_until_complete(request_task)
        # Approver is bound to alice's authenticated identity (email preferred).
        assert decision.approver == "alice@databricks.com"
        assert decision.approved is True
    finally:
        loop.close()


# --------------------------------------------------------------------------- 4


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hitl_framework_path_logs_owner_set(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``HITL_OWNER_SET`` is emitted at INFO when ownership registers.

    Replaces v1's "manual: spin up make dev, observe logs" with a
    deterministic caplog assertion.
    """
    broker = InProcessApprovalBroker()
    loop = _FakeLoop(broker=broker, user_id="alice")
    tool = _FakeTool()

    async def approve() -> None:
        await asyncio.sleep(0.05)
        rid = next(iter(broker._events))
        broker.resolve(
            rid,
            ApprovalDecision(approved=True, approver="alice"),
            requester_user_id="alice",
        )

    approve_task = asyncio.create_task(approve())
    with caplog.at_level(logging.INFO):
        await run_hitl_gate(loop, tool, {"k": "v"})
    await approve_task

    matches = [r for r in caplog.records if "HITL_OWNER_SET" in r.message]
    assert matches, "expected at least one HITL_OWNER_SET log record"
    assert any("alice" in r.getMessage() for r in matches)
