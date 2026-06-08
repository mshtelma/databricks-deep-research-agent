"""Human-in-the-loop approval primitives.

A :class:`ResearchTool` can be flagged ``requires_confirmation=True`` (or
decorated with :func:`requires_approval`). When the ReactLoop encounters
such a tool AND an :class:`ApprovalBroker` is attached via
``ctx.extras["_framework_approval_broker"]``, it pauses execution, emits a
:class:`GateWaitingEvent`, and awaits ``broker.request(...)``.

The broker is a Protocol — applications can plug in their own (HTTP-driven,
queue-driven, etc.). :class:`InProcessApprovalBroker` is the default
in-memory implementation; pair it with the FastAPI route at
``deep_research/api/v1/hitl.py`` to ship an end-to-end approval UX.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ApprovalDecision:
    """Decision returned by an :class:`ApprovalBroker`."""

    approved: bool
    reason: str | None = None
    approver: str | None = None


class ApprovalBroker(Protocol):
    """Protocol for HITL approval brokers."""

    async def request(
        self,
        request_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        reason: str = "",
        timeout_seconds: float = 300.0,
        owner_user_id: str | None = None,
    ) -> ApprovalDecision:
        """Block until an approval decision arrives or the timeout elapses.

        ``owner_user_id`` (optional) tags this request with an authorizing
        user so :meth:`resolve` can reject mismatched callers (HTTP 403).
        """
        ...


class InProcessApprovalBroker:
    """In-memory broker resolved via :class:`asyncio.Event`.

    Applications signal a decision via :meth:`resolve`. Idempotent against
    double-submits: once the event is set, subsequent calls return ``False``
    so the HTTP layer can return ``409 Conflict``. A 60-second post-resolve
    grace window keeps the request_id mapped (so a retry submits cleanly).

    Concurrency: an internal :class:`threading.Lock` guards mutations to
    the backing dictionaries. The lock is NEVER held while awaiting
    ``evt.wait()`` (the resolve gate); only registration, ownership writes,
    and decision-store updates run under the lock. A threading lock (rather
    than ``asyncio.Lock``) is used so the broker can be exercised from
    callers running on different event loops (test fixtures, multi-worker
    deployments) without cross-loop binding errors.
    """

    def __init__(self, *, grace_seconds: float = 60.0) -> None:
        self._events: dict[str, asyncio.Event] = {}
        self._decisions: dict[str, ApprovalDecision] = {}
        self._resolved_at: dict[str, float] = {}
        self._owners: dict[str, str] = {}
        self._lock = threading.Lock()
        self._grace_seconds = grace_seconds

    async def request(
        self,
        request_id: str,
        tool_name: str,
        arguments: dict[str, Any],  # noqa: ARG002 — protocol surface
        reason: str = "",  # noqa: ARG002 — protocol surface
        timeout_seconds: float = 300.0,
        owner_user_id: str | None = None,
    ) -> ApprovalDecision:
        # Register under the lock so concurrent registrations cannot race.
        with self._lock:
            if request_id in self._events:
                raise ValueError(f"request_id collision: {request_id!r}")
            evt = asyncio.Event()
            self._events[request_id] = evt
            if owner_user_id is not None:
                self._owners[request_id] = owner_user_id

        try:
            # Wait OUTSIDE the lock to avoid deadlock with concurrent resolve().
            await asyncio.wait_for(evt.wait(), timeout=timeout_seconds)
        except TimeoutError:
            logger.info(
                "APPROVAL_TIMEOUT request_id=%s tool=%s elapsed=%s",
                request_id, tool_name, timeout_seconds,
            )
            # Atomically transition to a resolved-with-timeout state so any
            # late HTTP resolve() short-circuits at the is_set() check below
            # and the caller (HTTP layer) sees 409 Conflict instead of a
            # silent decision-loss (Architect S4: do not orphan _decisions
            # on the normal-path pop).
            with self._lock:
                if not evt.is_set():
                    evt.set()
                    self._resolved_at[request_id] = time.monotonic()
            return ApprovalDecision(approved=False, reason="timeout")

        with self._lock:
            decision = self._decisions.pop(
                request_id, ApprovalDecision(approved=False)
            )
        return decision

    def resolve(
        self,
        request_id: str,
        decision: ApprovalDecision,
        requester_user_id: str | None = None,
    ) -> bool:
        """Signal a decision for the given request_id.

        Args:
            request_id: The pending approval to resolve.
            decision: The :class:`ApprovalDecision` to deliver.
            requester_user_id: When provided, must match the
                ``owner_user_id`` registered via :meth:`request` (if any).
                Mismatch causes the call to be rejected as a defensive
                gate (returns ``False``); HTTP layers SHOULD also call
                :meth:`owner_of` and return 403 explicitly before invoking
                resolve to distinguish authz-denied from already-resolved.

        Returns:
            ``True`` if this caller resolved the gate; ``False`` if the gate
            was already resolved, the request_id is unknown, or
            ``requester_user_id`` did not match the registered owner.
        """
        with self._lock:
            evt = self._events.get(request_id)
            if evt is None:
                return False
            if evt.is_set():
                return False
            owner = self._owners.get(request_id)
            if (
                owner is not None
                and requester_user_id is not None
                and owner != requester_user_id
            ):
                logger.info(
                    "HITL_AUTHZ_DENIED request_id=%s owner=%s requester=%s",
                    request_id, owner, requester_user_id,
                )
                return False
            self._decisions[request_id] = decision
            evt.set()
            self._resolved_at[request_id] = time.monotonic()
        return True

    def owner_of(self, request_id: str) -> str | None:
        """Return the registered owner_user_id for *request_id*, or None.

        HTTP layers can call this BEFORE :meth:`resolve` to distinguish
        an authz-denied 403 from an already-resolved 409.
        """
        return self._owners.get(request_id)

    def is_pending(self, request_id: str) -> bool:
        evt = self._events.get(request_id)
        return evt is not None and not evt.is_set()

    def cleanup(self, *, now: float | None = None) -> int:
        """Remove resolved request entries past the grace window.

        Returns the number of entries reclaimed. Safe to call periodically
        from a background task; not invoked automatically (PR2 wires this).
        """
        with self._lock:
            current = (
                now if now is not None else time.monotonic()
            )
            stale: list[str] = [
                rid
                for rid, t in self._resolved_at.items()
                if current - t > self._grace_seconds
            ]
            for rid in stale:
                self._events.pop(rid, None)
                self._decisions.pop(rid, None)
                self._resolved_at.pop(rid, None)
                self._owners.pop(rid, None)
            return len(stale)


_REQUIRES_APPROVAL_ATTR = "_dr_requires_approval"
_APPROVAL_REASON_ATTR = "_dr_approval_reason"


def requires_approval(reason: str = "") -> Any:
    """Decorator marking a ``@tool`` as gated by HITL approval.

    Works regardless of stacking order::

        # Inner-then-outer (recommended): tool() picks up the marker
        @tool
        @requires_approval(reason="destructive")
        def commit_to_delta(...): ...

        # Outer-then-inner: requires_approval edits the metadata
        @requires_approval(reason="destructive")
        @tool
        def commit_to_delta(...): ...

    The flag flows into ``ToolDefinition.metadata["requires_confirmation"]``
    which the ReactLoop consults at execution time.
    """

    def _wrap(target: Any) -> Any:
        # Case 1: target is a raw callable — attach markers; @tool will read them.
        if callable(target) and not hasattr(target, "_definition"):
            setattr(target, _REQUIRES_APPROVAL_ATTR, True)
            if reason:
                setattr(target, _APPROVAL_REASON_ATTR, reason)
            return target

        # Case 2: target is already a _DecoratedTool — patch the definition.
        if hasattr(target, "_definition"):
            old = target._definition
            meta = dict(old.metadata or {})
            meta["requires_confirmation"] = True
            if reason:
                meta["approval_reason"] = reason
            target._definition = old.__class__(
                name=old.name,
                description=old.description,
                parameters=old.parameters,
                source_type=old.source_type,
                source_kind=old.source_kind,
                metadata=meta,
            )
            return target

        raise TypeError(
            "@requires_approval must wrap a callable or @tool-decorated callable; "
            f"got {type(target).__name__}"
        )

    return _wrap


__all__ = [
    "ApprovalBroker",
    "ApprovalDecision",
    "InProcessApprovalBroker",
    "requires_approval",
]
