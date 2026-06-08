"""HITL endpoint contract tests.

Builds a minimal FastAPI app with the ``hitl.router`` mounted plus a
broker on ``app.state``. Asserts the 200/401/403/409/422/503 status
code mapping and the C2 post-timeout decision-loss fix.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from databricks_deep_research.api.approval import (
    ApprovalDecision,
    InProcessApprovalBroker,
)
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from deep_research.api.v1 import hitl
from deep_research.core.auth import UserIdentity
from deep_research.middleware.auth import get_current_user_identity

# =============================================================================
# Fixtures
# =============================================================================


def _identity(user_id: str = "alice", email: str = "alice@example.com") -> UserIdentity:
    return UserIdentity(user_id=user_id, email=email, display_name=user_id.title())


def _build_app(
    *,
    user: UserIdentity | None = None,
    broker: InProcessApprovalBroker | None = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(hitl.router)
    if broker is not None:
        app.state.approval_broker = broker
    if user is not None:
        app.dependency_overrides[get_current_user_identity] = lambda: user
    return app


@pytest.fixture
def broker() -> InProcessApprovalBroker:
    return InProcessApprovalBroker()


@pytest.fixture
def app_with_broker(broker: InProcessApprovalBroker) -> FastAPI:
    return _build_app(user=_identity("alice"), broker=broker)


@pytest.fixture
def app_without_broker() -> FastAPI:
    return _build_app(user=_identity("alice"))


# =============================================================================
# Helpers
# =============================================================================


def _seed_pending_request(
    broker: InProcessApprovalBroker,
    request_id: str,
    *,
    owner_user_id: str | None = None,
    timeout_seconds: float = 5.0,
) -> tuple[asyncio.AbstractEventLoop, asyncio.Task[ApprovalDecision]]:
    """Spin up a separate event loop and start ``broker.request(...)``.

    Returns the loop + task so callers can drain after running their
    assertions. The loop must be closed by the caller via
    ``loop.run_until_complete(task); loop.close()``.
    """
    loop = asyncio.new_event_loop()

    async def kick_off() -> ApprovalDecision:
        return await broker.request(
            request_id,
            "tool",
            {},
            timeout_seconds=timeout_seconds,
            owner_user_id=owner_user_id,
        )

    task = loop.create_task(kick_off())
    # Give the coroutine a chance to register the request_id before tests run.
    loop.run_until_complete(asyncio.sleep(0.05))
    return loop, task


# =============================================================================
# Existing happy-path tests (preserved + auth fixture added)
# =============================================================================


def test_approve_returns_200_on_first_resolve(
    app_with_broker: FastAPI, broker: InProcessApprovalBroker
) -> None:
    loop, task = _seed_pending_request(broker, "rid-1")

    client = TestClient(app_with_broker)
    response = client.post(
        "/research/hitl/approve/rid-1",
        json={"approved": True, "approver": "alice"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "resolved"
    assert data["request_id"] == "rid-1"

    loop.run_until_complete(task)
    loop.close()


def test_approve_returns_409_on_double_resolve(
    app_with_broker: FastAPI, broker: InProcessApprovalBroker
) -> None:
    loop, task = _seed_pending_request(broker, "rid-2")

    client = TestClient(app_with_broker)
    first = client.post(
        "/research/hitl/approve/rid-2",
        json={"approved": True},
    )
    assert first.status_code == 200
    second = client.post(
        "/research/hitl/approve/rid-2",
        json={"approved": True},
    )
    assert second.status_code == 409

    loop.run_until_complete(task)
    loop.close()


def test_approve_returns_409_on_unknown_request(app_with_broker: FastAPI) -> None:
    client = TestClient(app_with_broker)
    response = client.post(
        "/research/hitl/approve/never-existed",
        json={"approved": True},
    )
    assert response.status_code == 409


def test_approve_returns_422_on_malformed_body(app_with_broker: FastAPI) -> None:
    client = TestClient(app_with_broker)
    response = client.post(
        "/research/hitl/approve/rid-3",
        json={"foo": "bar"},  # Missing required ``approved``
    )
    assert response.status_code == 422


def test_approve_returns_503_when_broker_missing(app_without_broker: FastAPI) -> None:
    client = TestClient(app_without_broker)
    response = client.post(
        "/research/hitl/approve/rid-4",
        json={"approved": True},
    )
    assert response.status_code == 503


# =============================================================================
# US-006: New regression tests for C1 (auth+authz), C2 (post-timeout),
# C4 (concurrent resolve), and the bound-approver behavior.
# =============================================================================


def test_approve_unauthenticated_returns_401(
    broker: InProcessApprovalBroker,
) -> None:
    """C1: when the auth dep rejects (production behavior), endpoint returns 401.

    We simulate the "auth rejected" path by overriding ``get_current_user_identity``
    to raise HTTPException(401). This proves the endpoint REQUIRES the auth dep
    to succeed — there is no path in the endpoint code that bypasses auth.
    """

    def _strict_auth() -> UserIdentity:
        raise HTTPException(status_code=401, detail="Not authenticated")

    app = FastAPI()
    app.include_router(hitl.router)
    app.state.approval_broker = broker
    app.dependency_overrides[get_current_user_identity] = _strict_auth

    client = TestClient(app)
    response = client.post(
        "/research/hitl/approve/rid-unauth",
        json={"approved": True},
    )
    assert response.status_code == 401, (
        f"Endpoint must surface auth-dep 401; got {response.status_code}"
    )


def test_approve_cross_user_returns_403(
    broker: InProcessApprovalBroker,
) -> None:
    """C1: user B cannot resolve user A's request."""
    loop, task = _seed_pending_request(broker, "rid-cross", owner_user_id="alice")

    # User Bob tries to approve Alice's request.
    app = _build_app(user=_identity("bob", "bob@example.com"), broker=broker)
    client = TestClient(app)
    response = client.post(
        "/research/hitl/approve/rid-cross",
        json={"approved": True},
    )
    assert response.status_code == 403, (
        f"Cross-user resolve must be 403; got {response.status_code}"
    )

    # The pending request should still be pending (Bob's failed attempt did
    # not resolve it). Drain by having Alice approve.
    app2 = _build_app(user=_identity("alice"), broker=broker)
    client2 = TestClient(app2)
    response2 = client2.post(
        "/research/hitl/approve/rid-cross",
        json={"approved": True},
    )
    assert response2.status_code == 200

    loop.run_until_complete(task)
    loop.close()


def test_approve_same_user_after_owner_set_returns_200(
    broker: InProcessApprovalBroker,
) -> None:
    """C1: when owner is registered, the same user can resolve."""
    loop, task = _seed_pending_request(broker, "rid-same", owner_user_id="alice")

    app = _build_app(user=_identity("alice"), broker=broker)
    client = TestClient(app)
    response = client.post(
        "/research/hitl/approve/rid-same",
        json={"approved": True},
    )
    assert response.status_code == 200

    loop.run_until_complete(task)
    loop.close()


def test_approver_field_bound_to_authenticated_identity(
    broker: InProcessApprovalBroker,
) -> None:
    """C1: client-supplied body.approver must NOT override the auth identity."""
    loop, task = _seed_pending_request(broker, "rid-bind", owner_user_id="alice")

    app = _build_app(user=_identity("alice", "alice@example.com"), broker=broker)
    client = TestClient(app)
    response = client.post(
        "/research/hitl/approve/rid-bind",
        json={"approved": True, "approver": "ATTACKER"},
    )
    assert response.status_code == 200

    decision = loop.run_until_complete(task)
    loop.close()
    assert decision.approver == "alice@example.com", (
        f"Approver should be bound to user.email, not client value; "
        f"got {decision.approver!r}"
    )


def test_resolve_after_timeout_returns_409(
    broker: InProcessApprovalBroker,
) -> None:
    """C2: late HTTP resolve after the agent has timed out must return 409."""
    loop, task = _seed_pending_request(
        broker, "rid-timeout", owner_user_id="alice", timeout_seconds=0.1
    )
    # Let the request time out.
    decision = loop.run_until_complete(task)
    assert decision.approved is False
    assert decision.reason == "timeout"

    # Now POST a late approval. The C2 fix MUST cause this to 409, not 200.
    app = _build_app(user=_identity("alice"), broker=broker)
    client = TestClient(app)
    response = client.post(
        "/research/hitl/approve/rid-timeout",
        json={"approved": True},
    )
    assert response.status_code == 409, (
        f"Late resolve after timeout must 409 (C2 fix); got {response.status_code}"
    )
    loop.close()


def test_request_returns_timeout_decision_after_late_resolve(
    broker: InProcessApprovalBroker,
) -> None:
    """C2 / Architect requirement: assert the request() coroutine return is
    the timeout decision, not the late approval. This proves the agent
    received the correct outcome (denied via timeout), not silently the
    user's late approval.
    """
    loop, task = _seed_pending_request(
        broker, "rid-coro-timeout", owner_user_id="alice", timeout_seconds=0.1
    )

    # Wait for the timeout to fire on the request() coroutine.
    decision = loop.run_until_complete(task)

    # Assert request() returned the timeout decision.
    assert decision.approved is False
    assert decision.reason == "timeout"

    # Try a late resolve via HTTP — this should fail (409 per C2).
    app = _build_app(user=_identity("alice"), broker=broker)
    client = TestClient(app)
    response = client.post(
        "/research/hitl/approve/rid-coro-timeout",
        json={"approved": True},
    )
    assert response.status_code == 409

    # Re-confirm the agent's decision is unchanged.
    assert decision.approved is False
    assert decision.reason == "timeout"

    loop.close()


def test_concurrent_resolve_only_one_wins(
    broker: InProcessApprovalBroker,
) -> None:
    """C4: two concurrent resolve() calls — exactly one returns True."""
    loop, task = _seed_pending_request(broker, "rid-race", owner_user_id="alice")

    results: list[bool] = []
    barrier = threading.Barrier(parties=2)

    def attempt() -> None:
        barrier.wait()  # synchronize start
        ok = broker.resolve(
            "rid-race",
            ApprovalDecision(approved=True, approver="alice@example.com"),
            requester_user_id="alice",
        )
        results.append(ok)

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    # Exactly one must win.
    assert sum(results) == 1, f"Expected exactly one True; got {results}"

    loop.run_until_complete(task)
    loop.close()


def test_owner_of_returns_registered_owner(
    broker: InProcessApprovalBroker,
) -> None:
    """C1: broker exposes ownership for the HTTP layer's pre-resolve check."""
    loop, task = _seed_pending_request(
        broker, "rid-owner", owner_user_id="alice"
    )

    assert broker.owner_of("rid-owner") == "alice"
    assert broker.owner_of("nonexistent") is None

    # Drain.
    broker.resolve(
        "rid-owner", ApprovalDecision(approved=True), requester_user_id="alice"
    )
    loop.run_until_complete(task)
    loop.close()


def test_resolve_without_requester_is_backward_compat(
    broker: InProcessApprovalBroker,
) -> None:
    """C1: legacy callers that pass no requester_user_id still work even
    when an owner is registered (the broker only enforces when BOTH sides
    provide a value). The HTTP layer is responsible for always passing the
    user_id; this preserves backward compat for non-HTTP callers.
    """
    loop, task = _seed_pending_request(
        broker, "rid-bc", owner_user_id="alice"
    )

    ok = broker.resolve("rid-bc", ApprovalDecision(approved=True))
    assert ok is True

    loop.run_until_complete(task)
    loop.close()
