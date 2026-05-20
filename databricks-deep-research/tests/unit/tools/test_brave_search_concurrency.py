"""Tests for the process-wide semaphore and 429 retry in BraveSearchAdapter.

Covers the Phase 1 fixes for the NVDA-trace 429 cascade defect: when 7 lanes
burst concurrent searches, requests must serialize through the semaphore and
transient 429s must retry with exponential backoff instead of killing the
lane.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import pytest

from databricks_deep_research.tools.builtins import brave_search


@pytest.fixture(autouse=True)
def _reset_semaphore_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets a clean module-level semaphore + isolated env."""
    brave_search._reset_semaphore_for_tests()
    monkeypatch.delenv("BRAVE_MAX_CONCURRENCY", raising=False)
    monkeypatch.delenv("BRAVE_MAX_RETRIES", raising=False)
    yield
    brave_search._reset_semaphore_for_tests()


class _RecordingTransport(httpx.AsyncBaseTransport):
    """Test transport that lets us inspect in-flight concurrency."""

    def __init__(self, responder: Any) -> None:
        self._responder = responder
        self.in_flight = 0
        self.max_in_flight = 0
        self.call_count = 0
        self._lock = asyncio.Lock()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.call_count += 1
        try:
            await asyncio.sleep(0.05)  # simulate latency so concurrency is observable
            return await self._responder(request)
        finally:
            async with self._lock:
                self.in_flight -= 1


def _json_response(status: int = 200, body: dict | None = None, headers: dict | None = None) -> httpx.Response:
    if body is None:
        body = {"web": {"results": []}}
    return httpx.Response(status, json=body, headers=headers or {})


async def test_semaphore_limits_concurrent_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """20 concurrent searches must never exceed the configured concurrency cap."""
    monkeypatch.setenv("BRAVE_MAX_CONCURRENCY", "3")

    async def responder(_request: httpx.Request) -> httpx.Response:
        return _json_response()

    transport = _RecordingTransport(responder)
    adapter = brave_search.BraveSearchAdapter(api_key="test")
    # Pin the test transport into the adapter's client.
    adapter._client = httpx.AsyncClient(transport=transport, timeout=30.0)

    await asyncio.gather(*(adapter.search(f"q{i}", count=1) for i in range(20)))
    await adapter.aclose()

    assert transport.max_in_flight <= 3, (
        f"semaphore breached: max_in_flight={transport.max_in_flight}"
    )
    assert transport.call_count == 20


async def test_429_retry_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two 429s followed by a 200 must succeed without exception."""
    monkeypatch.setenv("BRAVE_MAX_RETRIES", "3")

    call_count = {"n": 0}

    async def responder(_request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] < 3:
            return _json_response(status=429, headers={"Retry-After": "0.01"})
        return _json_response(body={"web": {"results": [{"url": "https://a", "title": "A", "description": "snip"}]}})

    transport = _RecordingTransport(responder)
    adapter = brave_search.BraveSearchAdapter(api_key="test")
    adapter._client = httpx.AsyncClient(transport=transport, timeout=30.0)

    results = await adapter.search("query", count=1)
    await adapter.aclose()

    assert call_count["n"] == 3
    assert len(results) == 1
    assert results[0].url == "https://a"


async def test_429_exhausted_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If every attempt returns 429, the final attempt surfaces an HTTPStatusError."""
    monkeypatch.setenv("BRAVE_MAX_RETRIES", "2")

    async def responder(_request: httpx.Request) -> httpx.Response:
        return _json_response(status=429, headers={"Retry-After": "0.01"})

    transport = _RecordingTransport(responder)
    adapter = brave_search.BraveSearchAdapter(api_key="test")
    adapter._client = httpx.AsyncClient(transport=transport, timeout=30.0)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await adapter.search("query", count=1)
    await adapter.aclose()

    assert exc_info.value.response.status_code == 429
    assert transport.call_count == 2  # exact max_retries attempts


async def test_retry_after_header_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """If 429 carries Retry-After, the sleep duration must use that value."""
    monkeypatch.setenv("BRAVE_MAX_RETRIES", "2")

    sleeps: list[float] = []
    original_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await original_sleep(0)

    monkeypatch.setattr(brave_search.asyncio, "sleep", fake_sleep)

    call_count = {"n": 0}

    async def responder(_request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _json_response(status=429, headers={"Retry-After": "5"})
        return _json_response()

    transport = _RecordingTransport(responder)
    adapter = brave_search.BraveSearchAdapter(api_key="test")
    adapter._client = httpx.AsyncClient(transport=transport, timeout=30.0)

    await adapter.search("query", count=1)
    await adapter.aclose()

    # The retry-backoff sleep must honor Retry-After (5s) plus jitter ≤0.5.
    # Other small sleeps from the transport simulation are filtered out.
    retry_sleeps = [s for s in sleeps if s >= 1.0]
    assert len(retry_sleeps) == 1, f"unexpected retry sleep count: {sleeps}"
    assert 5.0 <= retry_sleeps[0] <= 5.5


async def test_reusable_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single adapter must reuse its httpx.AsyncClient across calls."""
    async def responder(_request: httpx.Request) -> httpx.Response:
        return _json_response()

    transport = _RecordingTransport(responder)
    adapter = brave_search.BraveSearchAdapter(api_key="test")
    adapter._client = httpx.AsyncClient(transport=transport, timeout=30.0)

    client_a = adapter._get_client()
    await adapter.search("q1", count=1)
    client_b = adapter._get_client()
    await adapter.search("q2", count=1)
    client_c = adapter._get_client()

    assert client_a is client_b is client_c
    await adapter.aclose()


def test_semaphore_default_when_env_unset() -> None:
    """With no env var, semaphore uses the paid-tier default (10).

    Free-tier deployments should set ``BRAVE_MAX_CONCURRENCY=1`` in the
    environment.
    """
    sem = brave_search._get_semaphore()
    assert sem._value == 10


def test_semaphore_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """BRAVE_MAX_CONCURRENCY env var configures the limit."""
    monkeypatch.setenv("BRAVE_MAX_CONCURRENCY", "8")
    brave_search._reset_semaphore_for_tests()
    sem = brave_search._get_semaphore()
    assert sem._value == 8


def test_semaphore_invalid_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-integer env var falls back to default rather than crashing."""
    monkeypatch.setenv("BRAVE_MAX_CONCURRENCY", "not-a-number")
    brave_search._reset_semaphore_for_tests()
    sem = brave_search._get_semaphore()
    assert sem._value == 10
