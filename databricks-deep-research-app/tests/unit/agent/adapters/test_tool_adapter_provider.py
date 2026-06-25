"""Tests for web-search provider selection in tool_adapter.

Covers _build_web_search_client (brave / databricks / fallback) and the
_domain_predicate helper used to thread per-agent domain filters into the
Databricks built-in-search adapter.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from databricks_deep_research.tools.builtins.databricks_web_search import (
    DatabricksWebSearchAdapter,
)

from deep_research.agent.adapters.tool_adapter import (
    BraveSearchAdapter,
    _allowlist_patterns,
    _build_web_search_client,
    _domain_predicate,
)
from deep_research.core.app_config import (
    DatabricksSearchConfig,
    DomainFilterConfig,
    DomainFilterMode,
)


def _dbx_cfg() -> DatabricksSearchConfig:
    return DatabricksSearchConfig(endpoint="databricks-gpt-5", max_results=7)


class TestBuildWebSearchClient:
    def test_brave_provider_builds_brave_adapter(self) -> None:
        client = _build_web_search_client(
            search_provider="brave",
            brave_client=MagicMock(),
            domain_filter_config=None,
            llm_client=None,
            databricks_search_cfg=None,
        )
        assert isinstance(client, BraveSearchAdapter)

    def test_databricks_provider_builds_databricks_adapter(self) -> None:
        llm = SimpleNamespace(openai_client=SimpleNamespace(base_url="https://h/serving-endpoints", api_key="t"))
        client = _build_web_search_client(
            search_provider="databricks",
            brave_client=None,
            domain_filter_config=None,
            llm_client=llm,
            databricks_search_cfg=_dbx_cfg(),
        )
        assert isinstance(client, DatabricksWebSearchAdapter)

    def test_databricks_misconfigured_falls_back_to_brave(self) -> None:
        # provider=databricks but no llm_client -> fall back to Brave when available
        client = _build_web_search_client(
            search_provider="databricks",
            brave_client=MagicMock(),
            domain_filter_config=None,
            llm_client=None,
            databricks_search_cfg=_dbx_cfg(),
        )
        assert isinstance(client, BraveSearchAdapter)

    def test_no_backend_available_returns_none(self) -> None:
        client = _build_web_search_client(
            search_provider="databricks",
            brave_client=None,
            domain_filter_config=None,
            llm_client=None,
            databricks_search_cfg=None,
        )
        assert client is None

    def test_databricks_provider_pushes_include_allowlist(self) -> None:
        llm = SimpleNamespace(openai_client=SimpleNamespace(base_url="https://h/serving-endpoints", api_key="t"))
        cfg = DomainFilterConfig(mode=DomainFilterMode.INCLUDE, include_domains=["*.reuters.com"])
        client = _build_web_search_client(
            search_provider="databricks",
            brave_client=None,
            domain_filter_config=cfg,
            llm_client=llm,
            databricks_search_cfg=_dbx_cfg(),  # databricks-gpt-5 → OpenAI backend
        )
        assert isinstance(client, DatabricksWebSearchAdapter)
        assert client._backend._allowed_domains == ["reuters.com"]  # type: ignore[union-attr]

    def test_databricks_push_flag_false_skips_filter(self) -> None:
        llm = SimpleNamespace(openai_client=SimpleNamespace(base_url="https://h/serving-endpoints", api_key="t"))
        cfg = DomainFilterConfig(mode=DomainFilterMode.INCLUDE, include_domains=["reuters.com"])
        client = _build_web_search_client(
            search_provider="databricks",
            brave_client=None,
            domain_filter_config=cfg,
            llm_client=llm,
            databricks_search_cfg=DatabricksSearchConfig(
                endpoint="databricks-gpt-5", push_allowed_domains=False
            ),
        )
        assert isinstance(client, DatabricksWebSearchAdapter)
        assert client._backend._allowed_domains == []  # type: ignore[union-attr]
        assert client._scope_clause != ""  # hint still applied

    def test_databricks_no_filter_no_clause(self) -> None:
        llm = SimpleNamespace(openai_client=SimpleNamespace(base_url="https://h/serving-endpoints", api_key="t"))
        client = _build_web_search_client(
            search_provider="databricks",
            brave_client=None,
            domain_filter_config=None,
            llm_client=llm,
            databricks_search_cfg=_dbx_cfg(),
        )
        assert isinstance(client, DatabricksWebSearchAdapter)
        assert client._scope_clause == ""


class TestAllowlistPatterns:
    def test_none_returns_empty(self) -> None:
        assert _allowlist_patterns(None) == []

    def test_exclude_mode_returns_empty(self) -> None:
        cfg = DomainFilterConfig(
            mode=DomainFilterMode.EXCLUDE,
            exclude_domains=["x.com"],
            include_domains=["y.com"],  # ignored in EXCLUDE mode
        )
        assert _allowlist_patterns(cfg) == []

    def test_include_mode_returns_includes(self) -> None:
        cfg = DomainFilterConfig(
            mode=DomainFilterMode.INCLUDE, include_domains=["*.reuters.com", "bbc.com"]
        )
        assert _allowlist_patterns(cfg) == ["*.reuters.com", "bbc.com"]

    def test_both_mode_returns_includes(self) -> None:
        cfg = DomainFilterConfig(
            mode=DomainFilterMode.BOTH, include_domains=["a.com"], exclude_domains=["b.com"]
        )
        assert _allowlist_patterns(cfg) == ["a.com"]


class TestDomainPredicate:
    def test_none_config_returns_none(self) -> None:
        assert _domain_predicate(None) is None

    def test_inactive_filter_returns_none(self) -> None:
        cfg = DomainFilterConfig(mode=DomainFilterMode.INCLUDE, include_domains=[])
        assert _domain_predicate(cfg) is None

    def test_active_filter_returns_predicate(self) -> None:
        cfg = DomainFilterConfig(mode=DomainFilterMode.INCLUDE, include_domains=["example.com"])
        pred = _domain_predicate(cfg)
        assert pred is not None
        assert pred("https://example.com/page") is True
        assert pred("https://other.com/page") is False
