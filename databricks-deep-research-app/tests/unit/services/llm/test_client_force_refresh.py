"""Unit tests for LLMClient.force_refresh_client + the shared refresh helper.

Exercises:
  * passive _ensure_fresh_client (delegates to _refresh_client_sync(force=False))
  * active force_refresh_client (delegates to _refresh_client_sync(force=True))
  * async _force_refresh_token (locked wrapper around the same helper)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from deep_research.core.databricks_auth import clear_databricks_auth


@pytest.fixture(autouse=True)
def _reset_auth_singleton() -> None:
    """Each test starts with a fresh DatabricksAuth singleton."""
    clear_databricks_auth()
    yield
    clear_databricks_auth()


def _profile_settings(mock_settings: MagicMock) -> None:
    mock_settings.return_value.databricks_token = None
    mock_settings.return_value.databricks_host = None
    mock_settings.return_value.databricks_config_profile = "p"
    mock_settings.return_value.is_databricks_app = False


def _direct_settings(mock_settings: MagicMock) -> None:
    mock_settings.return_value.databricks_token = "static-pat"
    mock_settings.return_value.databricks_host = "https://h.databricks.com"
    mock_settings.return_value.databricks_config_profile = None
    mock_settings.return_value.is_databricks_app = False


@patch("deep_research.core.databricks_auth.get_settings")
@patch("deep_research.core.databricks_auth.WorkspaceClient")
@patch("deep_research.services.llm.client.AsyncOpenAI")
@patch("deep_research.services.llm.client.ModelConfig")
def test_force_refresh_in_oauth_mode_invalidates_and_rebuilds(
    _mock_model_config: MagicMock,
    mock_openai: MagicMock,
    mock_wc_class: MagicMock,
    mock_settings: MagicMock,
) -> None:
    """force_refresh_client invalidates the auth cache AND rebuilds AsyncOpenAI
    even when the locally-cached token would still be considered fresh."""
    _profile_settings(mock_settings)

    mock_wc = MagicMock()
    mock_wc.config.host = "https://w.databricks.com"
    mock_wc.config.authenticate.return_value = {"Authorization": "Bearer tok-A"}
    mock_wc_class.return_value = mock_wc

    from deep_research.services.llm.client import LLMClient

    client = LLMClient()
    client._ensure_fresh_client()
    assert client._current_token == "tok-A"
    assert mock_openai.call_count == 1

    # Rotate the SDK response (simulating a fresh OAuth round trip)
    mock_wc.config.authenticate.return_value = {"Authorization": "Bearer tok-B"}

    # Active refresh: even though _credential is still locally "valid"
    # (1h window), invalidate() drops it and force_refresh=True regenerates.
    client.force_refresh_client()

    assert client._current_token == "tok-B"
    assert mock_openai.call_count == 2
    mock_openai.assert_called_with(
        api_key="tok-B",
        base_url="https://w.databricks.com/serving-endpoints",
    )


@patch("deep_research.core.databricks_auth.get_settings")
@patch("deep_research.services.llm.client.AsyncOpenAI")
@patch("deep_research.services.llm.client.ModelConfig")
def test_force_refresh_in_direct_token_mode_lazy_inits_only(
    _mock_model_config: MagicMock,
    mock_openai: MagicMock,
    mock_settings: MagicMock,
) -> None:
    """PAT mode: force_refresh_client still lazily initialises (if needed)
    but is otherwise a no-op — direct tokens cannot be refreshed."""
    _direct_settings(mock_settings)

    from deep_research.services.llm.client import LLMClient

    client = LLMClient()
    first = client.force_refresh_client()  # triggers lazy init
    second = client.force_refresh_client()  # is_oauth=False → no rebuild

    assert first is second
    # AsyncOpenAI built once during lazy init; PAT mode doesn't rebuild
    assert mock_openai.call_count == 1
    assert client._current_token == "static-pat"


@patch("deep_research.core.databricks_auth.get_settings")
@patch("deep_research.core.databricks_auth.WorkspaceClient")
@patch("deep_research.services.llm.client.AsyncOpenAI")
@patch("deep_research.services.llm.client.ModelConfig")
def test_force_refresh_propagates_sdk_failure(
    _mock_model_config: MagicMock,
    _mock_openai: MagicMock,
    mock_wc_class: MagicMock,
    mock_settings: MagicMock,
) -> None:
    """If get_token(force_refresh=True) raises, force_refresh_client propagates —
    never silently substitutes a stale client."""
    _profile_settings(mock_settings)

    mock_wc = MagicMock()
    mock_wc.config.host = "https://w.databricks.com"
    # First authenticate() succeeds; the second (after invalidate) blows up
    mock_wc.config.authenticate.side_effect = [
        {"Authorization": "Bearer tok-A"},
        RuntimeError("refresh token revoked"),
    ]
    mock_wc_class.return_value = mock_wc

    from deep_research.services.llm.client import LLMClient

    client = LLMClient()
    client._ensure_fresh_client()  # primes the cache with tok-A

    with pytest.raises(RuntimeError, match="refresh token revoked"):
        client.force_refresh_client()


@patch("deep_research.core.databricks_auth.get_settings")
@patch("deep_research.core.databricks_auth.WorkspaceClient")
@patch("deep_research.services.llm.client.AsyncOpenAI")
@patch("deep_research.services.llm.client.ModelConfig")
def test_force_refresh_lazy_init_without_prior_ensure(
    _mock_model_config: MagicMock,
    mock_openai: MagicMock,
    mock_wc_class: MagicMock,
    mock_settings: MagicMock,
) -> None:
    """force_refresh_client works as the very first call on a fresh client
    (the framework's adapter calls _ensure_fresh_client first, but the contract
    must work either way)."""
    _profile_settings(mock_settings)

    mock_wc = MagicMock()
    mock_wc.config.host = "https://w.databricks.com"
    mock_wc.config.authenticate.return_value = {"Authorization": "Bearer tok-X"}
    mock_wc_class.return_value = mock_wc

    from deep_research.services.llm.client import LLMClient

    client = LLMClient()
    # Skip _ensure_fresh_client entirely
    result = client.force_refresh_client()

    assert result is client._client
    assert client._current_token == "tok-X"
    mock_openai.assert_called_once_with(
        api_key="tok-X",
        base_url="https://w.databricks.com/serving-endpoints",
    )


@pytest.mark.asyncio
@patch("deep_research.core.databricks_auth.get_settings")
@patch("deep_research.core.databricks_auth.WorkspaceClient")
@patch("deep_research.services.llm.client.AsyncOpenAI")
@patch("deep_research.services.llm.client.ModelConfig")
async def test_async_force_refresh_token_uses_shared_helper(
    _mock_model_config: MagicMock,
    mock_openai: MagicMock,
    mock_wc_class: MagicMock,
    mock_settings: MagicMock,
) -> None:
    """The async _force_refresh_token wrapper goes through the same
    _refresh_client_sync helper as force_refresh_client (no duplicate
    AsyncOpenAI(api_key=..., base_url=...) construction site)."""
    _profile_settings(mock_settings)

    mock_wc = MagicMock()
    mock_wc.config.host = "https://w.databricks.com"
    mock_wc.config.authenticate.return_value = {"Authorization": "Bearer tok-1"}
    mock_wc_class.return_value = mock_wc

    from deep_research.services.llm.client import LLMClient

    client = LLMClient()
    client._ensure_fresh_client()

    mock_wc.config.authenticate.return_value = {"Authorization": "Bearer tok-2"}
    refreshed = await client._force_refresh_token()

    assert refreshed is True
    assert client._current_token == "tok-2"
    # Retry counter advanced (capped at 2 in the wrapper)
    assert client._auth_retry_count == 1


@pytest.mark.asyncio
@patch("deep_research.core.databricks_auth.get_settings")
@patch("deep_research.services.llm.client.AsyncOpenAI")
@patch("deep_research.services.llm.client.ModelConfig")
async def test_async_force_refresh_token_returns_false_in_pat_mode(
    _mock_model_config: MagicMock,
    _mock_openai: MagicMock,
    mock_settings: MagicMock,
) -> None:
    """_force_refresh_token short-circuits in PAT mode (no OAuth → can't refresh)."""
    _direct_settings(mock_settings)

    from deep_research.services.llm.client import LLMClient

    client = LLMClient()
    refreshed = await client._force_refresh_token()
    assert refreshed is False


@pytest.mark.asyncio
@patch("deep_research.core.databricks_auth.get_settings")
@patch("deep_research.core.databricks_auth.WorkspaceClient")
@patch("deep_research.services.llm.client.AsyncOpenAI")
@patch("deep_research.services.llm.client.ModelConfig")
async def test_async_force_refresh_token_respects_retry_cap(
    _mock_model_config: MagicMock,
    _mock_openai: MagicMock,
    mock_wc_class: MagicMock,
    mock_settings: MagicMock,
) -> None:
    """After 2 successive refresh attempts the wrapper bails out — protects
    the app's own 403 path from infinite auth storms."""
    _profile_settings(mock_settings)

    mock_wc = MagicMock()
    mock_wc.config.host = "https://w.databricks.com"
    mock_wc.config.authenticate.return_value = {"Authorization": "Bearer tok"}
    mock_wc_class.return_value = mock_wc

    from deep_research.services.llm.client import LLMClient

    client = LLMClient()
    client._ensure_fresh_client()

    # 1st attempt: counter -> 1, refresh runs
    assert await client._force_refresh_token() is True
    # 2nd attempt: counter -> 2, refresh runs
    assert await client._force_refresh_token() is True
    # 3rd attempt: counter -> 3, exceeds cap → returns False AND resets counter
    assert await client._force_refresh_token() is False
    assert client._auth_retry_count == 0


@patch("deep_research.core.databricks_auth.get_settings")
@patch("deep_research.core.databricks_auth.WorkspaceClient")
@patch("deep_research.services.llm.client.AsyncOpenAI")
@patch("deep_research.services.llm.client.ModelConfig")
def test_passive_ensure_fresh_does_not_invalidate(
    _mock_model_config: MagicMock,
    mock_openai: MagicMock,
    mock_wc_class: MagicMock,
    mock_settings: MagicMock,
) -> None:
    """The passive path must NOT touch the SDK cache — that would defeat
    the purpose of separating passive from active refresh."""
    _profile_settings(mock_settings)

    mock_wc = MagicMock()
    mock_wc.config.host = "https://w.databricks.com"
    mock_wc.config.authenticate.return_value = {"Authorization": "Bearer tok"}
    mock_wc_class.return_value = mock_wc

    from deep_research.services.llm.client import LLMClient

    client = LLMClient()
    client._ensure_fresh_client()
    # Spy on invalidate() to prove it is NOT called by the passive path
    client._auth.invalidate = MagicMock(wraps=client._auth.invalidate)

    client._ensure_fresh_client()
    client._ensure_fresh_client()

    client._auth.invalidate.assert_not_called()
    assert mock_openai.call_count == 1


@patch("deep_research.core.databricks_auth.get_settings")
@patch("deep_research.core.databricks_auth.WorkspaceClient")
@patch("deep_research.services.llm.client.AsyncOpenAI")
@patch("deep_research.services.llm.client.ModelConfig")
def test_force_refresh_calls_invalidate(
    _mock_model_config: MagicMock,
    _mock_openai: MagicMock,
    mock_wc_class: MagicMock,
    mock_settings: MagicMock,
) -> None:
    """force_refresh_client must call DatabricksAuth.invalidate() exactly once
    per call (defends against SDK-side token cache poisoning)."""
    _profile_settings(mock_settings)

    mock_wc = MagicMock()
    mock_wc.config.host = "https://w.databricks.com"
    mock_wc.config.authenticate.return_value = {"Authorization": "Bearer tok"}
    mock_wc_class.return_value = mock_wc

    from deep_research.services.llm.client import LLMClient

    client = LLMClient()
    client._ensure_fresh_client()
    client._auth.invalidate = MagicMock(wraps=client._auth.invalidate)

    client.force_refresh_client()

    client._auth.invalidate.assert_called_once()
