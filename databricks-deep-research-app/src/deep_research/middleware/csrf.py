"""CSRF protection middleware using Origin header validation.

Defence-in-depth layer for state-changing requests. The primary auth mechanism
is OBO tokens (header-based, not automatically sent cross-origin), so CSRF risk
is limited to the incognito cookie flow which already uses SameSite=strict.

This middleware adds Origin validation as an additional defence layer per OWASP
recommendations.
"""

import json
import logging
from urllib.parse import urlparse

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# HTTP methods that never change server state
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class CSRFMiddleware:
    """Validate the Origin header on state-changing requests.

    Algorithm:
    1. Non-HTTP scopes and safe methods pass through unconditionally.
    2. If the Origin header is absent, empty, or ``"null"`` the request is
       treated as same-origin (or a non-browser client) and allowed.
    3. Otherwise the origin is normalised (lowercase, trailing-slash stripped)
       and checked against the allow-set.  HTTP origins are rejected when
       *enforce_https* is ``True``, except for localhost addresses.
    """

    def __init__(
        self,
        app: ASGIApp,
        allowed_origins: set[str],
        enforce_https: bool = True,
    ) -> None:
        self.app = app
        self._allowed_origins = allowed_origins
        self._enforce_https = enforce_https

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method: str = scope.get("method", "GET")
        if method in _SAFE_METHODS:
            await self.app(scope, receive, send)
            return

        origin: str | None = Headers(scope=scope).get("origin")

        # No Origin header → same-origin request or non-browser client
        if not origin or origin == "null":
            await self.app(scope, receive, send)
            return

        normalized = origin.strip().lower().rstrip("/")

        # Enforce HTTPS in production (allow localhost for dev)
        if self._enforce_https:
            parsed = urlparse(normalized)
            if parsed.scheme == "http" and parsed.hostname not in (
                "localhost",
                "127.0.0.1",
                "::1",
            ):
                logger.warning(
                    "CSRF_REJECT_HTTP_ORIGIN origin=%s method=%s path=%s",
                    origin,
                    method,
                    scope.get("path", ""),
                )
                await _send_403(scope, receive, send, "Origin must use HTTPS")
                return

        # In development, allow any localhost origin (matches HTTPS exception pattern)
        if not self._enforce_https:
            parsed = urlparse(normalized)
            if parsed.hostname in ("localhost", "127.0.0.1", "::1"):
                await self.app(scope, receive, send)
                return

        if normalized not in self._allowed_origins:
            logger.warning(
                "CSRF_REJECT_ORIGIN origin=%s method=%s path=%s",
                origin,
                method,
                scope.get("path", ""),
            )
            await _send_403(scope, receive, send, "Origin not allowed")
            return

        await self.app(scope, receive, send)


async def _send_403(_scope: Scope, _receive: Receive, send: Send, detail: str) -> None:
    """Send a 403 JSON response without importing FastAPI."""
    body = json.dumps({"detail": detail}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 403,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
