"""Tests for security response headers middleware."""

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from deep_research.middleware.security import SecurityHeadersMiddleware


async def _ok_app(scope: Scope, receive: Receive, send: Send) -> None:
    """Minimal ASGI app returning 200."""
    response = JSONResponse({"ok": True}, status_code=200)
    await response(scope, receive, send)


async def _custom_header_app(scope: Scope, receive: Receive, send: Send) -> None:
    """ASGI app that sets a custom header to verify preservation."""
    response = JSONResponse({"ok": True}, status_code=200, headers={"X-Custom": "keep"})
    await response(scope, receive, send)


def _make_client(
    csp_policy: str | None = "default-src 'self'",
    report_only: bool = False,
    enable_hsts: bool = False,
    inner_app: ASGIApp = _ok_app,
) -> AsyncClient:
    app: ASGIApp = SecurityHeadersMiddleware(
        inner_app, csp_policy=csp_policy, report_only=report_only, enable_hsts=enable_hsts,
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


@pytest.mark.asyncio
async def test_all_security_headers_present() -> None:
    async with _make_client() as client:
        resp = await client.get("/any")
    assert resp.status_code == 200
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["X-XSS-Protection"] == "0"
    assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "Content-Security-Policy" in resp.headers


@pytest.mark.asyncio
async def test_csp_enforced_header_name() -> None:
    async with _make_client(csp_policy="default-src 'self'", report_only=False) as client:
        resp = await client.get("/any")
    assert "Content-Security-Policy" in resp.headers
    assert resp.headers["Content-Security-Policy"] == "default-src 'self'"


@pytest.mark.asyncio
async def test_csp_report_only_header_name() -> None:
    async with _make_client(csp_policy="default-src 'self'", report_only=True) as client:
        resp = await client.get("/any")
    assert "Content-Security-Policy-Report-Only" in resp.headers
    assert resp.headers["Content-Security-Policy-Report-Only"] == "default-src 'self'"


@pytest.mark.asyncio
async def test_no_csp_when_policy_is_none() -> None:
    async with _make_client(csp_policy=None) as client:
        resp = await client.get("/any")
    assert "Content-Security-Policy" not in resp.headers
    assert "Content-Security-Policy-Report-Only" not in resp.headers
    # Other headers should still be present
    assert resp.headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.asyncio
async def test_existing_headers_preserved() -> None:
    async with _make_client(inner_app=_custom_header_app) as client:
        resp = await client.get("/any")
    assert resp.headers["X-Custom"] == "keep"
    assert resp.headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.asyncio
async def test_non_http_passthrough() -> None:
    """Verify lifespan scope passes through without error."""
    calls: list[str] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        calls.append(scope["type"])

    middleware = SecurityHeadersMiddleware(inner, csp_policy="default-src 'self'")
    await middleware({"type": "lifespan"}, None, None)  # type: ignore[arg-type]
    assert calls == ["lifespan"]


# ── HSTS ──


@pytest.mark.asyncio
async def test_hsts_header_present_when_enabled() -> None:
    async with _make_client(enable_hsts=True) as client:
        resp = await client.get("/any")
    assert resp.status_code == 200
    assert resp.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"


@pytest.mark.asyncio
async def test_hsts_header_absent_when_disabled() -> None:
    async with _make_client(enable_hsts=False) as client:
        resp = await client.get("/any")
    assert "Strict-Transport-Security" not in resp.headers


@pytest.mark.asyncio
async def test_hsts_default_is_disabled() -> None:
    """Default factory (no enable_hsts kwarg) must not inject HSTS."""
    async with _make_client() as client:
        resp = await client.get("/any")
    assert "Strict-Transport-Security" not in resp.headers


# ── Production CSP directive coverage ──

_PRODUCTION_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: https:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

_REQUIRED_DIRECTIVES = [
    "default-src 'self'",
    "script-src 'self'",
    "frame-ancestors 'none'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
]


@pytest.mark.asyncio
async def test_production_csp_contains_all_hardened_directives() -> None:
    """Regression guard: all security-critical directives must be present."""
    async with _make_client(csp_policy=_PRODUCTION_CSP) as client:
        resp = await client.get("/any")
    csp = resp.headers["Content-Security-Policy"]
    for directive in _REQUIRED_DIRECTIVES:
        assert directive in csp, f"Missing CSP directive: {directive}"
