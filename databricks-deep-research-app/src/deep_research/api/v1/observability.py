"""Client-side error report ingest endpoint.

POST /api/v1/observability/client-errors

Receives uncaught client-side errors (React error-boundary catches, window.onerror,
unhandledrejection) from the SPA so they land in the SAME structured server log stream as
backend logs — otherwise a browser crash (e.g. the `[App]` blank screen) is invisible to
``make logs``. The highest-signal field is ``boundary_name`` (the ErrorBoundary ``name``,
e.g. "Surface"/"App"), which is meaningful even without sourcemaps.

Security controls (mirror ``metrics.py``):
  - Pydantic model ``extra="forbid"`` + per-field length caps.
  - Raw body hard-cap: 24 KiB (exceeds the summed field caps).
  - Per-user in-memory rate limit: 30 reports/min, 5/sec.
  - Auth via ``CurrentUser`` (OAuth-gated); the global Origin-checking ``CSRFMiddleware``
    auto-covers this state-changing route.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from time import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from deep_research.middleware.auth import CurrentUser
from deep_research.storage.observability import get_sink

logger = logging.getLogger(__name__)

router = APIRouter()


class ClientErrorReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["render", "window_error", "unhandled_rejection"]
    message: str = Field(max_length=2048)
    stack: str | None = Field(default=None, max_length=8192)
    component_stack: str | None = Field(default=None, max_length=8192)
    boundary_name: str | None = Field(default=None, max_length=128)
    route: str | None = Field(default=None, max_length=512)
    bundle_id: str | None = Field(default=None, max_length=64)
    user_agent: str | None = Field(default=None, max_length=256)


# ---------------------------------------------------------------------------
# Per-user rate-limit state (in-memory; single-replica is fine, matches metrics.py).
# ---------------------------------------------------------------------------
_user_rate: dict[str, deque[float]] = defaultdict(deque)
RATE_LIMIT_PER_MIN = 30
RATE_LIMIT_PER_SEC = 5
MAX_BODY_BYTES = 24 * 1024  # 24 KiB — must exceed the summed per-field caps
_LOG_TRUNC = 4096  # keep logged stacks bounded


def _truncate(value: str | None, limit: int = _LOG_TRUNC) -> str:
    if not value:
        return "-"
    return value if len(value) <= limit else value[:limit] + "…[truncated]"


@router.post("/client-errors")
async def ingest_client_error(
    request: Request,
    body: ClientErrorReport,
    user: CurrentUser,
) -> dict[str, int]:
    """Ingest one client-side error report -> WARNING log + ``client_error`` counter.

    Raises:
        HTTPException 413: raw body exceeds ``MAX_BODY_BYTES``.
        HTTPException 429: per-user per-minute or per-second rate limit exceeded.
    """
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="report exceeds 24 KiB")

    now = time()
    times = _user_rate[user.user_id]
    while times and now - times[0] > 60:
        times.popleft()
    per_sec = sum(1 for t in times if now - t < 1)
    if len(times) >= RATE_LIMIT_PER_MIN or per_sec >= RATE_LIMIT_PER_SEC:
        raise HTTPException(status_code=429, detail="rate limit")
    times.append(now)

    logger.warning(
        "client_error | kind=%s | boundary=%s | route=%s | bundle=%s | user=%s "
        "| msg=%s | stack=%s | component_stack=%s",
        body.kind,
        body.boundary_name or "-",
        body.route or "-",
        body.bundle_id or "-",
        user.user_id,
        _truncate(body.message, 512),
        _truncate(body.stack),
        _truncate(body.component_stack),
    )
    get_sink().counter(
        "client_error",
        1,
        kind=body.kind,
        boundary=body.boundary_name or "unknown",
    )

    return {"accepted": 1}
