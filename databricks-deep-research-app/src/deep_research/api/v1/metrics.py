"""Client metrics ingest endpoint.

POST /api/v1/metrics/client

Accepts a batch of client-side metric events emitted by the React frontend
(block_render_count, dnd_drop_failed, widget_fallback, token_refresh_attempts,
token_refresh_failures, revisions_tab_opened, agent_run_clicked,
agent_visibility_changed) and forwards them to the shared MetricsSink so they
land in the same structured log stream as server-side agent_designer.* signals.

Security controls:
  - Allowlist of signal names (unknown names are dropped silently).
  - Per-user in-memory rate limit: 60 batches/min, 10 batches/sec.
  - Body size hard-cap: 1 KiB.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from time import time
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from deep_research.middleware.auth import CurrentUser
from deep_research.storage.observability import get_sink

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_SIGNALS: frozenset[str] = frozenset(
    {
        "block_render_count",
        "dnd_drop_failed",
        "widget_fallback",
        "token_refresh_attempts",
        "token_refresh_failures",
        "revisions_tab_opened",
        "agent_run_clicked",
        "agent_visibility_changed",
    }
)


class ClientMetricEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float | None = None
    labels: dict[str, str] | None = None
    timestamp_ms: int


class ClientMetricsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: Annotated[list[ClientMetricEvent], Field(min_length=1, max_length=64)]


# ---------------------------------------------------------------------------
# Per-user rate limit state (in-memory; for V1.5 single-replica is fine).
# ---------------------------------------------------------------------------
_user_rate: dict[str, deque[float]] = defaultdict(deque)
RATE_LIMIT_PER_MIN = 60
RATE_LIMIT_PER_SEC = 10
MAX_BODY_BYTES = 1024  # 1 KiB


@router.post("/client")
async def ingest_client_metrics(
    request: Request,
    body: ClientMetricsRequest,
    user: CurrentUser,
) -> dict[str, int]:
    """Ingest a batch of client-side metric events.

    Args:
        request: Raw FastAPI request (used for body-size check).
        body: Validated batch of metric events.
        user: Authenticated user identity (injected by auth middleware).

    Returns:
        A dict with the count of accepted events.

    Raises:
        HTTPException 413: if the raw request body exceeds MAX_BODY_BYTES.
        HTTPException 429: if the user has exceeded the per-minute or
            per-second rate limit.
    """
    # Body size check — Pydantic has already parsed the body, but we verify
    # the raw size to enforce the 1 KiB hard-cap.
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="batch exceeds 1 KiB")

    # Rate limiting — sliding-window over the last 60 seconds.
    now = time()
    times = _user_rate[user.user_id]
    while times and now - times[0] > 60:
        times.popleft()
    per_sec = sum(1 for t in times if now - t < 1)
    if len(times) >= RATE_LIMIT_PER_MIN or per_sec >= RATE_LIMIT_PER_SEC:
        raise HTTPException(status_code=429, detail="rate limit")
    times.append(now)

    # Emit allowed signals to the shared MetricsSink.
    sink = get_sink()
    for event in body.events:
        if event.name not in ALLOWED_SIGNALS:
            logger.debug("client_metrics: dropping unknown signal %r", event.name)
            continue
        full_name = f"agent_designer.{event.name}"
        labels: dict[str, str] = event.labels or {}
        if event.value is not None:
            sink.histogram(full_name, event.value, **labels)
        else:
            sink.counter(full_name, 1, **labels)

    return {"accepted": len(body.events)}
