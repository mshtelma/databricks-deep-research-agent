"""In-memory registry of in-flight designer chat turns (async dispatch + resume).

Why this exists
---------------
A Best-of-N designer turn runs for minutes. The Databricks Apps gateway imposes
an *absolute* wall-clock cap (~4 min, observed on AIS) on a single streamed HTTP
response that a keepalive cannot defeat. If the turn is consumed *by* the request
(the old ``_sse_stream_with_heartbeat`` consumed ``run_turn`` directly and
cancelled it in ``finally``), the gateway cut kills the whole turn.

This registry **decouples the turn from the HTTP request**: ``run_turn`` is drained
by a registry-owned ``asyncio.Task`` into an append-only in-memory buffer, and HTTP
readers (the initial POST and any resume GET) stream *from* the buffer. A reader
disconnect cancels only the reader; the producer keeps running and stays resumable.
The producer makes only *outbound* LLM calls and holds no inbound HTTP connection,
so it is structurally immune to the inbound-response cap. The frontend reconnects
across connections (``GET /chat/{turn_id}/events?since=N``) and resumes from the
last delivered event; terminal events live in the buffer, so a late reconnect
always receives the final ``mutation_proposed`` + ``done``.

Scope / tech-debt
-----------------
**Per-process, in-memory.** This is correct because Databricks Apps runs a single
uvicorn worker (``entrypoint.sh`` has no ``--workers``) and a turn is ephemeral
(<=~10 min). If Apps ever runs >1 replica or needs restart-durability, migrate to a
DB-backed event log mirroring ``research_sessions`` (see ``services/job_manager.py``
+ ``api/v1/jobs.py``) — a resume would then survive process boundaries instead of
returning 404 → "please resend".
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from deep_research.agent_designer.sse_events import (
    DesignerSSEEvent,
    DoneEvent,
    ErrorEvent,
)

logger = logging.getLogger(__name__)

TurnStatus = Literal["running", "done", "error", "cancelled"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class StreamChunk:
    """One item a reader yields: a real event (with its sequence) or a keepalive.

    ``event is None`` means "no event arrived within the heartbeat window — emit a
    wire keepalive to keep the socket warm". The route translates this to an SSE
    comment frame; a real event becomes an ``id:``-tagged SSE event frame.
    """

    event: DesignerSSEEvent | None
    seq: int | None


@dataclass
class BufferedTurn:
    """A single in-flight (or finished) designer turn and its event buffer.

    ``events`` is append-only; an event's **sequence number is its list index**.
    ``_cond`` broadcasts to waiting readers whenever ``events`` grows or ``status``
    changes. Created on the running event loop (never at import) so the asyncio
    primitives bind to the correct loop.
    """

    turn_id: str
    owner_user_id: str
    session_id: str | None
    events: list[DesignerSSEEvent] = field(default_factory=list)
    status: TurnStatus = "running"
    created_at: datetime = field(default_factory=_utcnow)
    last_access: datetime = field(default_factory=_utcnow)
    _cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    _task: asyncio.Task[None] | None = None
    _readers: int = 0
    _grace: asyncio.TimerHandle | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status != "running"

    async def append(self, event: DesignerSSEEvent) -> None:
        async with self._cond:
            self.events.append(event)
            self._cond.notify_all()

    async def set_status(self, status: TurnStatus) -> None:
        async with self._cond:
            self.status = status
            self._cond.notify_all()


class DesignerTurnRegistry:
    """Owns the producer tasks + buffers for all in-flight designer turns.

    Config knobs (env, mirroring the existing ``DESIGNER_CHAT_*`` precedent) are
    read once at construction; pass explicit values in tests for fast timers.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float | None = None,
        max_active: int | None = None,
        per_user_max: int | None = None,
        idle_grace_seconds: float | None = None,
    ) -> None:
        self._turns: dict[str, BufferedTurn] = {}
        self._ttl_seconds = (
            ttl_seconds if ttl_seconds is not None else _env_float("DESIGNER_TURN_TTL_SECONDS", 900.0)
        )
        self._max_active = (
            max_active if max_active is not None else _env_int("DESIGNER_TURN_MAX_ACTIVE", 32)
        )
        self._per_user_max = (
            per_user_max if per_user_max is not None else _env_int("DESIGNER_TURN_PER_USER_MAX", 2)
        )
        self._idle_grace_seconds = (
            idle_grace_seconds
            if idle_grace_seconds is not None
            else _env_float("DESIGNER_TURN_IDLE_GRACE_SECONDS", 60.0)
        )

    # -- lifecycle ---------------------------------------------------------

    @property
    def per_user_max(self) -> int:
        return self._per_user_max

    def active_count_for_user(self, user_id: str) -> int:
        return sum(
            1 for t in self._turns.values() if t.owner_user_id == user_id and not t.is_terminal
        )

    def start(
        self,
        *,
        owner_user_id: str,
        session_id: str | None,
        producer: AsyncIterator[DesignerSSEEvent],
        on_error: Callable[[BaseException], None],
    ) -> BufferedTurn:
        """Register a turn and start draining ``producer`` in a background task.

        The task is owned here (not by any HTTP request), so a reader disconnect
        never cancels it. Opportunistically sweeps terminal turns first.
        """
        self.sweep()
        turn = BufferedTurn(
            turn_id=uuid.uuid4().hex,
            owner_user_id=owner_user_id,
            session_id=session_id,
        )
        self._turns[turn.turn_id] = turn
        turn._task = asyncio.create_task(self._drain(turn, producer, on_error))
        logger.info(
            "DESIGNER_TURN_STARTED turn_id=%s user=%s session=%s active=%d",
            turn.turn_id,
            owner_user_id,
            session_id,
            len(self._turns),
        )
        return turn

    def get(self, turn_id: str, *, owner_user_id: str) -> BufferedTurn | None:
        """Return the turn iff it exists AND is owned by ``owner_user_id``.

        Foreign / unknown ids both return ``None`` so the caller can answer 404
        without leaking existence (a ``turn_id`` is an unguessable capability, but
        the owner check is the real authorization boundary).
        """
        turn = self._turns.get(turn_id)
        if turn is None or turn.owner_user_id != owner_user_id:
            return None
        return turn

    async def _drain(
        self,
        turn: BufferedTurn,
        producer: AsyncIterator[DesignerSSEEvent],
        on_error: Callable[[BaseException], None],
    ) -> None:
        """Pump every event from ``producer`` into the buffer until terminal.

        ``run_turn`` yields its own ``DoneEvent`` last on the happy path. On a
        producer exception we surface a *sanitized* ``error`` + ``done`` into the
        buffer (so a reconnecting client sees a clean terminal, never raw upstream
        detail). On a deliberate idle-grace cancel we mark ``cancelled`` and
        re-raise (there are no readers to notify by construction).
        """
        try:
            async for event in producer:
                await turn.append(event)
            await turn.set_status("done")
            logger.info(
                "DESIGNER_TURN_COMPLETED turn_id=%s events=%d status=done",
                turn.turn_id,
                len(turn.events),
            )
        except asyncio.CancelledError:
            # Idle-grace reclaim: no active readers, so a plain field set is enough
            # (a late reader re-evaluates is_terminal under the lock before waiting).
            turn.status = "cancelled"
            logger.info(
                "DESIGNER_TURN_CANCELLED turn_id=%s events=%d (no active reader)",
                turn.turn_id,
                len(turn.events),
            )
            raise
        except Exception as exc:  # noqa: BLE001 — surfaced as a sanitized terminal
            on_error(exc)
            await turn.append(
                ErrorEvent(message="The designer chat failed. See server logs for details.")
            )
            await turn.append(DoneEvent())
            await turn.set_status("error")
            logger.info(
                "DESIGNER_TURN_COMPLETED turn_id=%s events=%d status=error",
                turn.turn_id,
                len(turn.events),
            )

    # -- reading -----------------------------------------------------------

    async def stream(
        self,
        turn: BufferedTurn,
        *,
        start_seq: int,
        heartbeat_seconds: float,
    ) -> AsyncIterator[StreamChunk]:
        """Yield buffered events from ``start_seq``, with idle keepalives.

        Resumable: pass ``start_seq=N`` to replay ``events[N:]`` (a reconnect). The
        producer is **never** cancelled here — a reader detach only decrements the
        refcount (which may schedule an idle-grace reclaim of a reader-less turn).
        Two terminal exits: (a) just yielded a ``done`` event, or (b) the turn is
        terminal and the buffer is fully drained (covers the cancel path, which
        appends no ``done``).
        """
        self._attach(turn)
        seq = max(start_seq, 0)
        try:
            while True:
                async with turn._cond:
                    ready = await self._wait_ready(turn, seq, heartbeat_seconds)
                if not ready:
                    yield StreamChunk(event=None, seq=None)  # keepalive
                    continue
                while seq < len(turn.events):
                    event = turn.events[seq]
                    yield StreamChunk(event=event, seq=seq)
                    seq += 1
                    if event.type == "done":
                        return
                turn.last_access = _utcnow()
                if turn.is_terminal and seq >= len(turn.events):
                    return
        finally:
            self._detach(turn)

    @staticmethod
    async def _wait_ready(turn: BufferedTurn, seq: int, timeout: float) -> bool:
        """Wait (holding ``turn._cond``) until an event at ``seq`` exists or the turn
        is terminal. Returns ``True`` if ready, ``False`` on heartbeat timeout.

        ``asyncio.wait_for`` cancels the inner ``cond.wait_for`` on timeout;
        ``Condition.wait`` re-acquires the lock before propagating ``CancelledError``,
        so the enclosing ``async with`` stays consistent.
        """
        try:
            await asyncio.wait_for(
                turn._cond.wait_for(lambda: len(turn.events) > seq or turn.is_terminal),
                timeout=timeout,
            )
            return True
        except TimeoutError:  # asyncio.TimeoutError is an alias of builtin TimeoutError (3.11+)
            return False

    # -- reader refcount + idle-grace reclaim ------------------------------

    def _attach(self, turn: BufferedTurn) -> None:
        turn._readers += 1
        if turn._grace is not None:
            turn._grace.cancel()
            turn._grace = None

    def _detach(self, turn: BufferedTurn) -> None:
        turn._readers = max(turn._readers - 1, 0)
        if turn._readers == 0 and not turn.is_terminal and self._idle_grace_seconds >= 0:
            loop = asyncio.get_running_loop()
            turn._grace = loop.call_later(self._idle_grace_seconds, self._reclaim, turn)

    def _reclaim(self, turn: BufferedTurn) -> None:
        """Cancel a still-running, reader-less turn to stop orphaned token burn."""
        turn._grace = None
        if turn._readers == 0 and not turn.is_terminal and turn._task is not None:
            logger.info("DESIGNER_TURN_RECLAIM turn_id=%s (idle, no readers)", turn.turn_id)
            turn._task.cancel()

    # -- cleanup -----------------------------------------------------------

    def sweep(self) -> int:
        """Evict terminal, reader-less turns older than the TTL; cap total turns.

        Returns the number evicted. Never evicts a running turn or one with an
        active reader. When over ``max_active``, evicts the oldest *terminal*
        reader-less turns (LRU by ``last_access``).
        """
        now = _utcnow()
        evicted = 0
        for turn_id in list(self._turns):
            turn = self._turns[turn_id]
            if (
                turn.is_terminal
                and turn._readers == 0
                and (now - turn.last_access).total_seconds() > self._ttl_seconds
            ):
                del self._turns[turn_id]
                evicted += 1

        if len(self._turns) > self._max_active:
            evictable = sorted(
                (t for t in self._turns.values() if t.is_terminal and t._readers == 0),
                key=lambda t: t.last_access,
            )
            overflow = len(self._turns) - self._max_active
            for turn in evictable[:overflow]:
                del self._turns[turn.turn_id]
                evicted += 1

        if evicted:
            logger.info("DESIGNER_TURN_EVICTED count=%d remaining=%d", evicted, len(self._turns))
        return evicted


# Module-level singleton — one per process (single uvicorn worker on Apps).
designer_turn_registry = DesignerTurnRegistry()
