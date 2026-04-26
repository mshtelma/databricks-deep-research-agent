"""`HydrationProgress` — SSE-aware wrapper around `ChatStateCache.get`.

Implements plan §R-5: first-turn hydration with partial timeouts that emit
progressive SSE events so the UI can indicate context loading. If hydration
runs past the absolute cap, raises `HydrationTimeoutError` (routes translate
to HTTP 503 with `Retry-After`).

Deadlines default to 1 s → `context_loading`, 3 s → `context_slow`, 10 s → cap.
All values are configurable so tests can drive the boundaries with a fake
clock-free approach (`asyncio.wait_for` with sub-second timeouts).

The counter `storage_first_turn_outcome_total{outcome=warm|slow|cold|capped,backend=…}`
lands on every call regardless of outcome, so SLO dashboards can compute the
warm/cold hit-rate.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from deep_research.storage.observability import get_sink

if TYPE_CHECKING:  # pragma: no cover
    from deep_research.storage.backend import StorageBackend
    from deep_research.storage.cache import ChatStateCache
    from deep_research.storage.documents import ChatDocument

logger = logging.getLogger(__name__)

# Signature: `(event_name, data_dict_or_none) -> None`. Intentionally generic
# so the route can plumb any SSE framework (FastAPI-sse, sse-starlette, raw).
SseEmit = Callable[[str, dict | None], None]

Deadline = tuple[float, str | None]


class HydrationTimeoutError(Exception):
    """Hydration exceeded the absolute cap. Route handler maps to HTTP 503."""

    def __init__(self, chat_id: UUID, cap_seconds: float) -> None:
        super().__init__(
            f"chat {chat_id} hydration exceeded {cap_seconds}s cap"
        )
        self.chat_id = chat_id
        self.cap_seconds = cap_seconds
        self.retry_after = 2  # seconds; caller can echo in `Retry-After` header.


@dataclass
class HydrationProgress:
    """Per-request helper. Construct fresh on each `POST /research` call."""

    cache: "ChatStateCache"
    sse_emit: SseEmit | None = None
    backend_label: str = "unknown"
    deadlines: tuple[Deadline, ...] = (
        (1.0, "context_loading"),
        (3.0, "context_slow"),
        (10.0, None),
    )

    async def hydrate(
        self,
        chat_id: UUID,
        *,
        user_id: str | None = None,
        title_hint: str = "",
    ) -> "ChatDocument":
        """Await cache hydration, emitting SSE events at each progressive deadline.

        Raises `HydrationTimeoutError` if the final (cap) deadline passes.
        Emits `storage_first_turn_outcome_total` exactly once per call.
        """
        # A single background task we `shield` on successive `wait_for`s so a
        # preempting timeout doesn't cancel the underlying hydration.
        task = asyncio.ensure_future(
            self.cache.get(chat_id, user_id=user_id, title_hint=title_hint)
        )
        sink = get_sink()
        elapsed = 0.0
        outcomes: list[str] = []  # labels we've emitted at each boundary

        try:
            for deadline, event_name in self.deadlines:
                remaining = max(0.0, deadline - elapsed)
                try:
                    doc = await asyncio.wait_for(
                        asyncio.shield(task), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    elapsed = deadline
                    # Final deadline with no event → hard cap.
                    if event_name is None:
                        task.cancel()
                        sink.counter(
                            "storage_first_turn_outcome_total",
                            outcome="capped",
                            backend=self.backend_label,
                        )
                        raise HydrationTimeoutError(chat_id, deadline)
                    if self.sse_emit is not None:
                        try:
                            self.sse_emit(event_name, None)
                        except Exception:  # noqa: BLE001 — SSE errors mustn't fail hydrate
                            logger.exception("SSE emit failed for %s", event_name)
                    outcomes.append(event_name)
                    continue
                else:
                    outcome = _classify_outcome(outcomes)
                    sink.counter(
                        "storage_first_turn_outcome_total",
                        outcome=outcome,
                        backend=self.backend_label,
                    )
                    return doc
            # If we fell out of the loop without returning, that's a bug — the
            # last deadline has `event_name=None` which always raises above.
            raise AssertionError("unreachable: final deadline must be None")
        finally:
            # If the task is still pending (e.g. raise happened before it
            # resolved), make sure we don't leak it. `gather` with
            # return_exceptions swallows the CancelledError cleanly.
            if not task.done():
                task.cancel()
                try:
                    await asyncio.gather(task, return_exceptions=True)
                except Exception:  # noqa: BLE001 — defensive only
                    pass


def _classify_outcome(emitted_events: list[str]) -> str:
    """Map emitted boundary events to an SLO outcome label.

    * No events emitted → `warm` (completed before first deadline).
    * Only `context_loading` → `slow` (past 1 s, before 3 s).
    * `context_slow` also emitted → `cold` (past 3 s, before cap).
    """
    if not emitted_events:
        return "warm"
    if "context_slow" in emitted_events:
        return "cold"
    return "slow"
