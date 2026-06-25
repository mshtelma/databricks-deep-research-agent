"""Default web-search backend wired into the runner's ToolFactoryContext.

``_apply_default_search_client`` makes the INHERITED web backend (tools with no
per-tool ``config.provider``) follow the workspace ``search.provider`` — Databricks
built-in search by default — so per-agent web tools that omit a provider use
Databricks with no Brave key. This is the linchpin of the provider rollout: the
framework factory's no-provider branch reads ``ctx.search_client``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from databricks_deep_research.tools.builtins.databricks_web_search import (
    DatabricksWebSearchAdapter,
)
from databricks_deep_research.tools.factory import ToolFactoryContext

from deep_research.agent.workflow_runner_factory import (
    _apply_default_search_client,
    _apply_serving_client_provider,
)
from deep_research.core.app_config import DatabricksSearchConfig, SearchConfig


def _patch_search(monkeypatch: pytest.MonkeyPatch, cfg: SearchConfig) -> None:
    # _apply_default_search_client imports get_app_config from core.app_config.
    monkeypatch.setattr(
        "deep_research.core.app_config.get_app_config",
        lambda: SimpleNamespace(search=cfg),
    )


def _fake_llm() -> Any:
    # The databricks adapter is built with client_provider=lambda: llm.openai_client
    # (called lazily at search time, not at construction), so a placeholder is enough.
    return SimpleNamespace(openai_client=SimpleNamespace())


def _dbx_cfg() -> SearchConfig:
    return SearchConfig(
        provider="databricks",
        databricks=DatabricksSearchConfig(endpoint="databricks-gpt-5"),
    )


class TestApplyDefaultSearchClient:
    def test_databricks_default_sets_databricks_adapter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_search(monkeypatch, _dbx_cfg())
        ctx = ToolFactoryContext(search_client="seeded-by-from_defaults")
        _apply_default_search_client(ctx, _fake_llm())
        assert isinstance(ctx.search_client, DatabricksWebSearchAdapter)

    def test_brave_default_leaves_search_client_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Explicit global brave (opt-in) → keep whatever from_defaults seeded.
        _patch_search(monkeypatch, SearchConfig(provider="brave"))
        sentinel = object()
        ctx = ToolFactoryContext(search_client=sentinel)
        _apply_default_search_client(ctx, _fake_llm())
        assert ctx.search_client is sentinel

    def test_databricks_default_without_llm_client_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No llm_client → adapter cannot be built → leave the from_defaults value
        # (Brave iff key, else None). Must NOT raise (don't-assume-brave safety).
        _patch_search(monkeypatch, _dbx_cfg())
        sentinel = object()
        ctx = ToolFactoryContext(search_client=sentinel)
        _apply_default_search_client(ctx, None)
        assert ctx.search_client is sentinel

    def test_unreadable_config_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom() -> Any:
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(
            "deep_research.core.app_config.get_app_config", _boom
        )
        sentinel = object()
        ctx = ToolFactoryContext(search_client=sentinel)
        _apply_default_search_client(ctx, _fake_llm())  # swallowed, no raise
        assert ctx.search_client is sentinel


class TestApplyServingClientProvider:
    """Built-in web search runs as the app/SP serving client — the same client
    the LLM calls use — so ``serving_client_provider`` returns it. Set provider-
    agnostically (the framework explicit-``provider: databricks`` path uses it),
    never the OBO user token."""

    def test_sets_sp_serving_client_provider(self) -> None:
        llm = _fake_llm()
        ctx = ToolFactoryContext()
        _apply_serving_client_provider(ctx, llm)
        assert ctx.serving_client_provider is not None
        assert ctx.serving_client_provider() is llm.openai_client

    def test_no_llm_client_is_noop(self) -> None:
        ctx = ToolFactoryContext()
        _apply_serving_client_provider(ctx, None)
        assert ctx.serving_client_provider is None
