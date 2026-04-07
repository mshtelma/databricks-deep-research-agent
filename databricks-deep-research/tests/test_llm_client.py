"""Tests for the FrameworkLLMClient retry behavior."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import BadRequestError, InternalServerError, PermissionDeniedError, RateLimitError

from databricks_deep_research.llm.client import FrameworkLLMClient


def _make_client() -> tuple[FrameworkLLMClient, AsyncMock]:
    """Create a FrameworkLLMClient with a mocked AsyncOpenAI."""
    mock_openai = AsyncMock()
    client = FrameworkLLMClient(
        openai_client=mock_openai,
        model_mapping={"analytical": "test-model"},
    )
    return client, mock_openai


def _make_bad_request_error() -> BadRequestError:
    """Create a BadRequestError with required fields."""
    response = httpx.Response(
        status_code=400,
        request=httpx.Request("POST", "https://api.example.com/chat/completions"),
        json={"error": {"message": "tool_use without tool_result", "type": "invalid_request_error"}},
    )
    return BadRequestError(
        message="tool_use without tool_result",
        response=response,
        body={"error": {"message": "tool_use without tool_result"}},
    )


def _make_internal_server_error() -> InternalServerError:
    """Create an InternalServerError with required fields."""
    response = httpx.Response(
        status_code=500,
        request=httpx.Request("POST", "https://api.example.com/chat/completions"),
        json={"error": {"message": "internal error", "type": "server_error"}},
    )
    return InternalServerError(
        message="internal error",
        response=response,
        body={"error": {"message": "internal error"}},
    )


def _make_permission_denied_error() -> PermissionDeniedError:
    response = httpx.Response(
        status_code=403,
        request=httpx.Request("POST", "https://api.example.com/chat/completions"),
        json={"error": {"message": "forbidden", "type": "permission_denied"}},
    )
    return PermissionDeniedError(
        message="forbidden",
        response=response,
        body={"error": {"message": "forbidden"}},
    )


@pytest.mark.asyncio
async def test_bad_request_not_retried() -> None:
    """400 BadRequest should raise immediately without retrying."""
    client, mock_openai = _make_client()

    # Make the underlying call raise BadRequestError
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=_make_bad_request_error()
    )

    with pytest.raises(BadRequestError):
        await client.complete(
            [{"role": "user", "content": "test"}],
            "analytical",
        )

    # Should have been called only once (no retries)
    assert mock_openai.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_server_error_retried() -> None:
    """500 InternalServerError should be retried."""
    client, mock_openai = _make_client()

    # First call fails with 500, subsequent calls also fail
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=_make_internal_server_error()
    )

    with pytest.raises(InternalServerError):
        await client.complete(
            [{"role": "user", "content": "test"}],
            "analytical",
        )

    # Should have been called 3 times (1 initial + 2 retries)
    assert mock_openai.chat.completions.create.await_count == 3


def test_from_databricks_direct_token_no_client_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[tuple[str, str]] = []

    class DummyOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            self.api_key = api_key
            self.base_url = base_url
            created.append((api_key, base_url))

    monkeypatch.setenv("DATABRICKS_HOST", "https://dbc.example.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token-a")
    monkeypatch.setattr("databricks_deep_research.llm.client.AsyncOpenAI", DummyOpenAI)

    client = FrameworkLLMClient.from_databricks(model="test-model")

    # Direct-token path: no provider, static client, single instance
    assert client._client_provider is None
    assert client._client.api_key == "token-a"
    assert client._client.base_url == "https://dbc.example.com/serving-endpoints"
    assert len(created) == 1


@pytest.mark.asyncio
async def test_permission_denied_refreshes_client_once() -> None:
    first_client = AsyncMock()
    second_client = AsyncMock()
    first_client.chat.completions.create = AsyncMock(side_effect=_make_permission_denied_error())
    second_client.chat.completions.create = AsyncMock(return_value=_make_success_response())

    provider_calls = 0

    def provider() -> AsyncMock:
        nonlocal provider_calls
        provider_calls += 1
        return second_client

    client = FrameworkLLMClient(
        openai_client=first_client,
        model_mapping={"analytical": "test-model"},
        client_provider=provider,
    )

    result = await client.complete(
        [{"role": "user", "content": "test"}],
        "analytical",
    )

    assert provider_calls == 1
    assert result.content == '{"steps": [{"id": "s1"}]}'
    assert first_client.chat.completions.create.await_count == 1
    assert second_client.chat.completions.create.await_count == 1

async def test_client_provider_is_not_called_for_every_request() -> None:
    """Provider should only be used for explicit refreshes, not normal calls."""
    base_client = AsyncMock()
    base_client.chat.completions.create = AsyncMock(return_value=_make_success_response())

    refreshed_client = AsyncMock()
    refreshed_client.chat.completions.create = AsyncMock(return_value=_make_success_response())

    provider = MagicMock(return_value=refreshed_client)

    client = FrameworkLLMClient(
        openai_client=base_client,
        model_mapping={"analytical": "test-model"},
        client_provider=provider,
    )

    result = await client.complete(
        [{"role": "user", "content": "test"}],
        "analytical",
    )

    assert result.content == '{"steps": [{"id": "s1"}]}'
    assert base_client.chat.completions.create.await_count == 1
    assert provider.call_count == 0
    assert refreshed_client.chat.completions.create.await_count == 0



# ---------------------------------------------------------------------------
# Structured output fallback tests
# ---------------------------------------------------------------------------


class _TestValidationError(Exception):
    """Name contains 'ValidationError' for client fallback detection."""


def _make_json_unsupported_error() -> BadRequestError:
    """Create a BadRequestError for json_object not supported."""
    response = httpx.Response(
        status_code=400,
        request=httpx.Request("POST", "https://api.example.com/chat/completions"),
        json={
            "error_code": "INVALID_PARAMETER_VALUE",
            "message": "Response format type json_object is not supported",
        },
    )
    return BadRequestError(
        message="Response format type json_object is not supported",
        response=response,
        body={"error_code": "INVALID_PARAMETER_VALUE"},
    )


def _make_success_response() -> AsyncMock:
    """Create a mock successful completion response."""
    resp = AsyncMock()
    resp.choices = [AsyncMock()]
    resp.choices[0].message.content = '{"steps": [{"id": "s1"}]}'
    resp.choices[0].message.tool_calls = None
    resp.choices[0].finish_reason = "stop"
    resp.model = "test-model"
    resp.usage = AsyncMock()
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.total_tokens = 15
    return resp


@pytest.mark.asyncio
async def test_structured_validation_fallback_json_unsupported() -> None:
    """When structured output fails validation and json_object is unsupported,
    gracefully fall through to standard completion without response_format."""
    from pydantic import BaseModel

    class DummyOutput(BaseModel):
        value: int

    client, mock_openai = _make_client()

    # 1. beta.parse raises ValidationError
    mock_openai.beta.chat.completions.parse = AsyncMock(
        side_effect=_TestValidationError("value: input should be int")
    )

    # 2. Track create calls to verify json_object attempted then dropped
    create_calls: list[dict[str, Any]] = []

    async def mock_create(**kwargs: Any) -> Any:
        create_calls.append(dict(kwargs))
        if kwargs.get("response_format") == {"type": "json_object"}:
            raise _make_json_unsupported_error()
        return _make_success_response()

    mock_openai.chat.completions.create = AsyncMock(side_effect=mock_create)

    result = await client.complete(
        [{"role": "user", "content": "test"}],
        "analytical",
        structured_output=DummyOutput,
    )

    # Assertions
    assert result.content == '{"steps": [{"id": "s1"}]}'
    assert mock_openai.beta.chat.completions.parse.await_count == 1
    assert len(create_calls) == 2

    # First create call had json_object
    assert create_calls[0]["response_format"] == {"type": "json_object"}
    # Second create call had no response_format
    assert "response_format" not in create_calls[1]


@pytest.mark.asyncio
async def test_structured_validation_fallback_json_supported() -> None:
    """When structured output fails validation and json_object IS supported,
    return the json_object response directly."""
    from pydantic import BaseModel

    class DummyOutput(BaseModel):
        value: int

    client, mock_openai = _make_client()

    # beta.parse raises ValidationError
    mock_openai.beta.chat.completions.parse = AsyncMock(
        side_effect=_TestValidationError("value: input should be int")
    )

    # create succeeds with json_object (model supports it)
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_make_success_response()
    )

    result = await client.complete(
        [{"role": "user", "content": "test"}],
        "analytical",
        structured_output=DummyOutput,
    )

    assert result.content == '{"steps": [{"id": "s1"}]}'
    # Only one create call needed (json_object succeeded)
    assert mock_openai.chat.completions.create.await_count == 1


# ---------------------------------------------------------------------------
# Rate-limit retry tests
# ---------------------------------------------------------------------------


def _make_rate_limit_error(retry_after: str | None = None) -> RateLimitError:
    """Create a RateLimitError with optional Retry-After header."""
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(
        status_code=429,
        headers=headers,
        request=httpx.Request("POST", "https://api.example.com/chat/completions"),
        json={"error": {"message": "rate limited", "type": "rate_limit_error"}},
    )
    return RateLimitError(
        message="rate limited",
        response=response,
        body={"error": {"message": "rate limited"}},
    )


@pytest.mark.asyncio
async def test_rate_limit_retried_when_retry_rate_limit_true() -> None:
    """429 with retry_rate_limit=True: first attempt fails, second succeeds."""
    client, mock_openai = _make_client()

    call_count = 0

    async def mock_func() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_rate_limit_error()
        return "success"

    with patch("databricks_deep_research.llm.client.asyncio.sleep", new_callable=AsyncMock):
        result = await client._retry_with_backoff(mock_func, retry_rate_limit=True)

    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_rate_limit_not_retried_when_retry_rate_limit_false() -> None:
    """429 with retry_rate_limit=False: raises immediately (backward compat)."""
    client, mock_openai = _make_client()

    call_count = 0

    async def mock_func() -> str:
        nonlocal call_count
        call_count += 1
        raise _make_rate_limit_error()

    with pytest.raises(RateLimitError):
        await client._retry_with_backoff(mock_func, retry_rate_limit=False)

    assert call_count == 1


@pytest.mark.asyncio
async def test_rate_limit_respects_retry_after_header() -> None:
    """429 with Retry-After: 2 header → verify sleep called with capped value."""
    client, mock_openai = _make_client()

    call_count = 0

    async def mock_func() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _make_rate_limit_error(retry_after="2")
        return "success"

    with patch("databricks_deep_research.llm.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await client._retry_with_backoff(mock_func, retry_rate_limit=True)

    assert result == "success"
    assert call_count == 2
    # Sleep should have been called once with the retry-after value (2.0)
    mock_sleep.assert_awaited_once()
    actual_backoff = mock_sleep.call_args[0][0]
    assert actual_backoff == 2.0


@pytest.mark.asyncio
async def test_aclose_swallows_loop_closed_runtime_error() -> None:
    mock_openai = AsyncMock()
    mock_openai.aclose = AsyncMock(side_effect=RuntimeError("Event loop is closed"))
    client = FrameworkLLMClient(
        openai_client=mock_openai,
        model_mapping={"analytical": "test-model"},
    )

    await client.aclose()

    assert client._closed is True
