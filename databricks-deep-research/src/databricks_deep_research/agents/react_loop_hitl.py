"""HITL approval gate logic invoked from :func:`ReactLoop._execute_single_tool`.

Kept in its own module so the in-place diff to ``react_loop.py`` stays small
(per the strategy spec's ≤ 13-LoC harness budget). The function returns
``None`` when the tool was approved (caller falls through to normal
execution); otherwise returns a denial metadata dict that the caller
returns up the stack.

Depends only on the typed
:class:`databricks_deep_research.agents.protocol.ReactLoopHook` Protocol —
the concrete :class:`ReactLoop` satisfies it structurally. This decoupling
keeps the gate free of runtime introspection on framework objects
(Constitution #4).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from databricks_deep_research.events.hitl import (
    GateDeniedEvent,
    GateResumedEvent,
    GateTimeoutEvent,
    GateWaitingEvent,
)

if TYPE_CHECKING:
    from databricks_deep_research.agents.protocol import ReactLoopHook
    from databricks_deep_research.tools.protocol import ResearchTool

logger = logging.getLogger(__name__)


async def run_hitl_gate(
    loop: ReactLoopHook,
    tool: ResearchTool,
    args: dict[str, Any],
) -> dict[str, Any] | None:
    """Run the HITL approval gate for *tool*.

    Returns:
        ``None`` if approved (caller proceeds with normal execution).
        A ``{"content", "meta"}`` dict on denial / timeout that the caller
        returns up the stack.
    """
    broker = loop.extras.get("_framework_approval_broker")
    if broker is None:
        return None

    # Owner is the authenticated user_id of the session originating this
    # tool execution. Sourced from extras under the reserved
    # "_framework_user_id" key. Populated by two paths (PR3a): the
    # orchestrator path (framework_orchestrator -> ExecutionContext ->
    # harness) and the public Agent API path (Agent.user_id ->
    # _apply_extras_to_state -> harness). When absent (CLI/agent_server),
    # ownership stays None and the HTTP layer falls back to
    # authenticated-only (no per-request authz).
    # By contract (PR3a), the writers in framework_orchestrator.py and
    # api/agent.py only ever stash a ``str`` (or never set the key at all)
    # under "_framework_user_id". We accept the contract and rely on
    # ``mypy --strict`` at the call sites; runtime narrowing here would
    # violate Constitution #4.
    owner_user_id: str | None = cast(
        "str | None", loop.extras.get("_framework_user_id")
    )

    if owner_user_id is None:
        logger.warning(
            "HITL_OWNER_MISSING node=%s tool=%s — broker is wired but no "
            "_framework_user_id; owner-authz disabled, falling back to "
            "authenticated-only",
            loop.node_id,
            tool.definition.name,
        )

    metadata = tool.definition.metadata or {}
    reason = str(metadata.get("approval_reason", ""))
    request_id = str(uuid4())
    timestamp = datetime.now(tz=UTC).isoformat()

    waiting = GateWaitingEvent(
        node_id=loop.node_id,
        timestamp=timestamp,
        request_id=request_id,
        tool_name=tool.definition.name,
        arguments=dict(args),
        reason=reason,
    )
    loop.emit_event(waiting)

    decision = await broker.request(
        request_id, tool.definition.name, args, reason=reason,
        owner_user_id=owner_user_id,
    )
    if owner_user_id is not None:
        logger.info(
            "HITL_OWNER_SET request_id=%s owner=%s",
            request_id,
            owner_user_id,
        )
    decision_ts = datetime.now(tz=UTC).isoformat()

    if decision.approved:
        resumed = GateResumedEvent(
            node_id=loop.node_id,
            timestamp=decision_ts,
            request_id=request_id,
            approver=decision.approver,
        )
        loop.emit_event(resumed)
        return None

    is_timeout = (decision.reason or "").lower() == "timeout"
    if is_timeout:
        event: Any = GateTimeoutEvent(
            node_id=loop.node_id,
            timestamp=decision_ts,
            request_id=request_id,
        )
    else:
        event = GateDeniedEvent(
            node_id=loop.node_id,
            timestamp=decision_ts,
            request_id=request_id,
            reason=decision.reason,
            approver=decision.approver,
        )
    loop.emit_event(event)

    denial_marker = "approval denied" if not is_timeout else "approval timed out"
    return {
        "content": f"[{denial_marker}]",
        "meta": {
            "tool_success": False,
            "tool_error": f"approval_denied:{tool.definition.name}",
            "raw_source_count": 0,
            "accepted_source_count": 0,
            "rejected_source_count": 0,
        },
    }


__all__ = ["run_hitl_gate"]
