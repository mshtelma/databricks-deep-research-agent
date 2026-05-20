"""Adapter wiring tests for the framework's auth-refresh hook.

The framework's ``FrameworkLLMClient`` accepts a ``client_provider`` callback
that it invokes on 403 to mint a fresh client. Wiring this to the *passive*
``_ensure_fresh_client`` (as the adapter previously did) was the root cause of
the persistent "Invalid Token" 403s — passive refresh returns the same cached
client. These tests pin the wiring to the *active* ``force_refresh_client``
method instead.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from deep_research.services.llm.types import ModelEndpoint, ModelRole, SelectionStrategy


def _make_endpoint(eid: str, ident: str) -> ModelEndpoint:
    return ModelEndpoint(
        id=eid,
        endpoint_identifier=ident,
        max_context_window=128_000,
        tokens_per_minute=200_000,
    )


def _make_role(name: str, eps: list[str]) -> ModelRole:
    return ModelRole(
        name=name,
        endpoints=eps,
        fallback_on_429=True,
        rotation_strategy=SelectionStrategy.PRIORITY,
    )


def _make_mock_llm(
    *,
    auth_mode: str = "profile",
    token: str = "tok-prefix-abcdef",
) -> MagicMock:
    """Construct a mock app LLMClient with both refresh methods exposed.

    Mirrors the surface area the adapter touches:
      * ``_config.get_role`` / ``_config.get_endpoint`` for model mapping,
      * ``_ensure_fresh_client`` for initial client extraction,
      * ``force_refresh_client`` for the framework's client_provider hook,
      * ``_auth.auth_mode`` for the diagnostic log.
    """
    fake_openai = MagicMock(name="AsyncOpenAI")
    fake_openai.base_url = "https://w.databricks.com/serving-endpoints"
    fake_openai.api_key = token

    ep = _make_endpoint("ep-a", "model-a")
    roles = {"analytical": _make_role("analytical", ["ep-a"])}
    endpoints = {"ep-a": ep}
    config = MagicMock()
    config.get_role = MagicMock(side_effect=lambda n: roles.get(n) or _raise(KeyError(n)))
    config.get_endpoint = MagicMock(side_effect=lambda i: endpoints[i])

    llm = MagicMock()
    llm._config = config
    llm._ensure_fresh_client = MagicMock(return_value=fake_openai)
    llm.force_refresh_client = MagicMock(return_value=fake_openai)
    llm._auth = MagicMock()
    llm._auth.auth_mode = auth_mode
    return llm


def _raise(exc: BaseException) -> None:
    raise exc


@patch("deep_research.agent.adapters.llm_adapter.FrameworkLLMClient")
def test_client_provider_is_force_refresh_client(mock_fw_cls: MagicMock) -> None:
    """Regression: client_provider MUST be ``force_refresh_client``, not the
    passive ``_ensure_fresh_client``. Identity check, not equality."""
    from deep_research.agent.adapters.llm_adapter import create_framework_llm_client

    llm = _make_mock_llm()

    create_framework_llm_client(llm)

    call_kwargs = mock_fw_cls.call_args.kwargs
    assert call_kwargs["client_provider"] is llm.force_refresh_client
    assert call_kwargs["client_provider"] is not llm._ensure_fresh_client


@patch("deep_research.agent.adapters.llm_adapter.FrameworkLLMClient")
def test_initial_client_extracted_via_passive_path(mock_fw_cls: MagicMock) -> None:
    """The initial AsyncOpenAI is taken from the cheap passive path —
    only the refresh hook is active. Saves an unnecessary OAuth round trip
    on the happy path."""
    from deep_research.agent.adapters.llm_adapter import create_framework_llm_client

    llm = _make_mock_llm()

    create_framework_llm_client(llm)

    llm._ensure_fresh_client.assert_called_once()
    llm.force_refresh_client.assert_not_called()


@patch("deep_research.agent.adapters.llm_adapter.FrameworkLLMClient")
def test_emits_bind_log_for_observability(
    _mock_fw_cls: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A single ``FWK_LLM_ADAPTER_BIND`` log line is emitted so the binding
    of (auth_mode, base_url, token_prefix) is visible at research startup.
    Makes "which token is the framework using?" answerable from logs alone."""
    from deep_research.agent.adapters.llm_adapter import create_framework_llm_client

    llm = _make_mock_llm(auth_mode="profile", token="abcdefgh-rest-of-token")
    caplog.set_level(logging.INFO, logger="deep_research.agent.adapters.llm_adapter")

    create_framework_llm_client(llm)

    msgs = [r.getMessage() for r in caplog.records]
    assert any("FWK_LLM_ADAPTER_BIND" in m for m in msgs)
    bind = next(m for m in msgs if "FWK_LLM_ADAPTER_BIND" in m)
    # Token must be prefix-masked (never log the full bearer)
    assert "abcdefgh***" in bind
    assert "abcdefgh-rest-of-token" not in bind
    assert "auth_mode=profile" in bind


@patch("deep_research.agent.adapters.llm_adapter.FrameworkLLMClient")
def test_bind_log_masks_token_in_automatic_mode(
    _mock_fw_cls: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Token prefix masking works for the Databricks Apps automatic-OAuth case too."""
    from deep_research.agent.adapters.llm_adapter import create_framework_llm_client

    llm = _make_mock_llm(auth_mode="automatic", token="zzzzzzzz-supersecret")
    caplog.set_level(logging.INFO, logger="deep_research.agent.adapters.llm_adapter")

    create_framework_llm_client(llm)

    bind = next(
        r.getMessage()
        for r in caplog.records
        if "FWK_LLM_ADAPTER_BIND" in r.getMessage()
    )
    assert "zzzzzzzz***" in bind
    assert "supersecret" not in bind
    assert "auth_mode=automatic" in bind
