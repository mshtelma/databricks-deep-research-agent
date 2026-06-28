"""HITL approval HTTP endpoint.

Pairs with :class:`InProcessApprovalBroker` from the framework. The broker
must be attached to ``app.state`` (see ``deep_research.core.app_state``)
so the endpoint can resolve it across requests.

Usage flow::

    1. Frontend renders a "approve" / "deny" prompt when it receives a
       ``GateWaitingEvent`` over SSE.
    2. User clicks → frontend POSTs ``/api/v1/research/hitl/approve/{request_id}``
       with ``{"approved": true, "approver": "alice"}``.
    3. The endpoint resolves the broker, which unblocks the agent.
    4. Subsequent calls with the same request_id return ``409 Conflict``.

Security model (PR1 + PR3a):

- **Authentication**: required via ``CurrentUser`` dependency (401 if absent).
- **Authorization**: when the broker has a registered ``owner_user_id`` for
  the request, only that user may resolve (403 otherwise). Ownership is
  set by ``react_loop_hitl.run_hitl_gate`` from
  ``ToolContext.extras["_framework_user_id"]``. Two code paths populate
  this key: (1) the orchestrator path
  (``framework_orchestrator.py`` -> ``ExecutionContext`` -> harness
  -> ``ToolContext.extras``) and (2) the public Agent API path
  (``Agent.user_id`` -> ``_apply_extras_to_state`` -> ``WorkflowState``
  -> harness -> ``ToolContext.extras``). Both wired in PR3a.
- **Non-FastAPI entrypoints** (agent_server, CLI) intentionally pass
  ``approval_broker=None``; HITL gating is disabled on those paths.
- **Approver binding**: the recorded ``approver`` field is bound to the
  authenticated identity (``user.email`` or ``user.user_id``);
  ``body.approver`` is treated as an audit-only label and does NOT
  override the bound value.

Tested at
``databricks-deep-research-app/tests/integration/api/test_hitl_framework_path.py``.
"""

from __future__ import annotations

import logging
from typing import cast

from databricks_deep_research.api.approval import (
    ApprovalDecision,
    InProcessApprovalBroker,
)
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from deep_research.middleware.auth import CurrentUser

logger = logging.getLogger(__name__)
router = APIRouter()


class ApproveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool = Field(..., description="True to approve, False to deny.")
    reason: str | None = Field(default=None, description="Optional rationale for the decision.")
    approver: str | None = Field(default=None, description="User-facing approver identifier.")


class ApproveResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    request_id: str


def _resolve_broker(request: Request) -> InProcessApprovalBroker:
    """Fetch the broker from FastAPI app state.

    The broker must be created at startup and stored on
    ``app.state.approval_broker``. If absent, raise 503 so the frontend
    knows HITL is not configured.
    """
    broker = getattr(request.app.state, "approval_broker", None)
    if broker is None:
        raise HTTPException(
            status_code=503,
            detail="HITL approval broker not configured on the server.",
        )
    # Custom brokers must implement the protocol; accept duck-typed.
    if not isinstance(broker, InProcessApprovalBroker) and not hasattr(
        broker, "resolve"
    ):
        raise HTTPException(
            status_code=503,
            detail="Configured approval_broker does not implement resolve().",
        )
    return cast(InProcessApprovalBroker, broker)


@router.post(
    "/research/hitl/approve/{request_id}",
    response_model=ApproveResponse,
    tags=["HITL"],
)
async def approve(
    request_id: str,
    body: ApproveBody,
    request: Request,
    user: CurrentUser,
) -> ApproveResponse:
    """Resolve an HITL approval request.

    Authentication: requires ``CurrentUser`` (401 if missing).
    Authorization: when the broker has a registered ``owner_user_id`` for
    this request_id, only that user may resolve it (403 otherwise). The
    ``approver`` field on the resolved record is bound to the authenticated
    identity (``user.email`` or ``user.user_id``); ``body.approver`` is
    treated as an audit-only label and does NOT override the bound value.

    Returns:
        200 OK on the first valid resolution.
        403 Forbidden if the requester is not the registered owner.
        409 Conflict if the request has already been resolved or doesn't exist.
    """
    broker = _resolve_broker(request)

    # Pre-check ownership so the HTTP layer can distinguish 403 (authz)
    # from 409 (already-resolved). The broker also defensively rejects
    # mismatches inside resolve(), but that is reported as 409.
    owner_of = getattr(broker, "owner_of", None)
    if callable(owner_of):
        owner = owner_of(request_id)
        if owner is not None and owner != user.user_id:
            logger.info(
                "HITL_AUTHZ_DENIED request_id=%s owner=%s requester=%s",
                request_id, owner, user.user_id,
            )
            raise HTTPException(
                status_code=403,
                detail="Not authorized to resolve this approval request.",
            )

    bound_approver = user.email or user.user_id
    decision = ApprovalDecision(
        approved=body.approved,
        reason=body.reason,
        approver=bound_approver,
    )
    ok = broker.resolve(
        request_id, decision, requester_user_id=user.user_id
    )
    if not ok:
        logger.info(
            "HITL_RESOLVE_CONFLICT request_id=%s already_resolved_or_unknown=true",
            request_id,
        )
        raise HTTPException(
            status_code=409,
            detail="Request already resolved or unknown.",
        )
    logger.info(
        "HITL_RESOLVED request_id=%s approved=%s approver=%s",
        request_id, body.approved, bound_approver,
    )
    return ApproveResponse(status="resolved", request_id=request_id)
