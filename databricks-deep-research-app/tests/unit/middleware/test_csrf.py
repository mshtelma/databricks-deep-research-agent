"""Tests for CSRF middleware Origin header validation."""

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from deep_research.middleware.csrf import CSRFMiddleware


async def _echo_app(scope: Scope, receive: Receive, send: Send) -> None:
    """Minimal ASGI app that returns 200 with the request method."""
    request = Request(scope, receive)
    response = JSONResponse({"method": request.method}, status_code=200)
    await response(scope, receive, send)


def _make_client(
    allowed_origins: set[str] | None = None,
    enforce_https: bool = True,
) -> AsyncClient:
    """Create an httpx AsyncClient wrapping the CSRF middleware."""
    origins = allowed_origins if allowed_origins is not None else {"https://app.com"}
    app: ASGIApp = CSRFMiddleware(
        _echo_app,
        allowed_origins=origins,
        enforce_https=enforce_https,
    )
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


# ── Safe methods always pass ──


@pytest.mark.asyncio
async def test_get_passes_with_evil_origin() -> None:
    async with _make_client() as client:
        resp = await client.get("/any", headers={"Origin": "http://evil.com"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_head_passes_with_evil_origin() -> None:
    async with _make_client() as client:
        resp = await client.head("/any", headers={"Origin": "http://evil.com"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_options_passes_with_evil_origin() -> None:
    async with _make_client() as client:
        resp = await client.options("/any", headers={"Origin": "http://evil.com"})
    assert resp.status_code == 200


# ── No Origin / empty / null → pass through ──


@pytest.mark.asyncio
async def test_post_no_origin_passes() -> None:
    async with _make_client() as client:
        resp = await client.post("/any")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_post_empty_origin_passes() -> None:
    async with _make_client() as client:
        resp = await client.post("/any", headers={"Origin": ""})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_post_null_origin_passes() -> None:
    async with _make_client() as client:
        resp = await client.post("/any", headers={"Origin": "null"})
    assert resp.status_code == 200


# ── Valid origin passes ──


@pytest.mark.asyncio
async def test_post_valid_origin_passes() -> None:
    async with _make_client({"https://app.com"}) as client:
        resp = await client.post("/any", headers={"Origin": "https://app.com"})
    assert resp.status_code == 200


# ── Invalid origin rejected for state-changing methods ──


@pytest.mark.asyncio
async def test_post_invalid_origin_rejected() -> None:
    async with _make_client({"https://app.com"}) as client:
        resp = await client.post("/any", headers={"Origin": "https://evil.com"})
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Origin not allowed"


@pytest.mark.asyncio
async def test_put_invalid_origin_rejected() -> None:
    async with _make_client({"https://app.com"}) as client:
        resp = await client.put("/any", headers={"Origin": "https://evil.com"})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_delete_invalid_origin_rejected() -> None:
    async with _make_client({"https://app.com"}) as client:
        resp = await client.delete("/any", headers={"Origin": "https://evil.com"})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_patch_invalid_origin_rejected() -> None:
    async with _make_client({"https://app.com"}) as client:
        resp = await client.patch("/any", headers={"Origin": "https://evil.com"})
    assert resp.status_code == 403


# ── HTTPS enforcement ──


@pytest.mark.asyncio
async def test_http_origin_rejected_in_prod() -> None:
    async with _make_client({"http://app.com"}, enforce_https=True) as client:
        resp = await client.post("/any", headers={"Origin": "http://app.com"})
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Origin must use HTTPS"


@pytest.mark.asyncio
async def test_http_localhost_allowed_in_prod() -> None:
    async with _make_client({"http://localhost:5173"}, enforce_https=True) as client:
        resp = await client.post("/any", headers={"Origin": "http://localhost:5173"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_http_origin_allowed_in_dev() -> None:
    async with _make_client({"http://app.com"}, enforce_https=False) as client:
        resp = await client.post("/any", headers={"Origin": "http://app.com"})
    assert resp.status_code == 200


# ── Normalisation ──


@pytest.mark.asyncio
async def test_case_insensitive_matching() -> None:
    async with _make_client({"https://app.com"}) as client:
        resp = await client.post("/any", headers={"Origin": "HTTPS://APP.COM"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_trailing_slash_normalised() -> None:
    async with _make_client({"https://app.com"}) as client:
        resp = await client.post("/any", headers={"Origin": "https://app.com/"})
    assert resp.status_code == 200


# ── Subdomain not matched (exact set lookup) ──


@pytest.mark.asyncio
async def test_subdomain_not_matched() -> None:
    async with _make_client({"https://app.com"}) as client:
        resp = await client.post("/any", headers={"Origin": "https://app.com.evil.com"})
    assert resp.status_code == 403


# ── Empty allowed_origins set ──


@pytest.mark.asyncio
async def test_empty_allowed_origins_rejects_all() -> None:
    async with _make_client(set()) as client:
        resp = await client.post("/any", headers={"Origin": "https://any.com"})
    assert resp.status_code == 403
