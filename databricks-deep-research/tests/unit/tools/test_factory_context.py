"""Unit tests for ToolFactoryContext.from_defaults()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from databricks_deep_research.tools.factory import ToolFactoryContext


class TestFromDefaults:
    """ToolFactoryContext.from_defaults() auto-detection tests."""

    def test_no_env_vars_returns_none_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no env vars or deps, all optional fields are None."""
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        # Patch WorkspaceClient import to fail (simulate missing creds)
        with (
            patch(
                "databricks_deep_research.tools.factory.ToolFactoryContext.from_defaults",
                wraps=ToolFactoryContext.from_defaults,
            ),
            patch.dict("sys.modules", {"databricks.sdk": None}),
        ):
                ctx = ToolFactoryContext.from_defaults()

        assert ctx.crawler is None
        # workspace_client and search_client may be None depending on env

    def test_brave_api_key_param_creates_adapter(self) -> None:
        """Explicit brave_api_key creates a BraveSearchAdapter."""
        ctx = ToolFactoryContext.from_defaults(brave_api_key="test-key-123")

        assert ctx.search_client is not None
        from databricks_deep_research.tools.builtins.brave_search import BraveSearchAdapter

        assert isinstance(ctx.search_client, BraveSearchAdapter)

    def test_brave_api_key_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """BRAVE_API_KEY env var is picked up when no explicit param."""
        monkeypatch.setenv("BRAVE_API_KEY", "env-key-456")
        ctx = ToolFactoryContext.from_defaults()

        assert ctx.search_client is not None
        from databricks_deep_research.tools.builtins.brave_search import BraveSearchAdapter

        assert isinstance(ctx.search_client, BraveSearchAdapter)

    def test_explicit_workspace_client_used(self) -> None:
        """Provided workspace_client is used directly without auto-detect."""
        mock_ws = MagicMock()
        ctx = ToolFactoryContext.from_defaults(workspace_client=mock_ws)

        assert ctx.workspace_client is mock_ws

    def test_crawler_always_none(self) -> None:
        """crawler is always None — WebCrawlTool uses defaults."""
        ctx = ToolFactoryContext.from_defaults(brave_api_key="key")
        assert ctx.crawler is None

    def test_extras_passed_through(self) -> None:
        """extras dict is forwarded to the context."""
        ctx = ToolFactoryContext.from_defaults(extras={"custom": "value"})
        assert ctx.extras == {"custom": "value"}

    def test_extras_default_empty(self) -> None:
        """extras defaults to empty dict when not provided."""
        ctx = ToolFactoryContext.from_defaults()
        assert ctx.extras == {}

    def test_user_token_passed_through(self) -> None:
        """user_token is forwarded to the context."""
        ctx = ToolFactoryContext.from_defaults(user_token="tok-123")
        assert ctx.user_token == "tok-123"


class TestApiKeys:
    """api_keys field on ToolFactoryContext."""

    def test_default_empty(self) -> None:
        ctx = ToolFactoryContext()
        assert ctx.api_keys == {}

    def test_brave_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAVE_API_KEY", "brave-123")
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        ctx = ToolFactoryContext.from_defaults()
        assert ctx.api_keys.get("brave") == "brave-123"
        assert "jina" not in ctx.api_keys

    def test_jina_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        monkeypatch.setenv("JINA_API_KEY", "jina-456")
        ctx = ToolFactoryContext.from_defaults()
        assert ctx.api_keys.get("jina") == "jina-456"

    def test_both_keys_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAVE_API_KEY", "b")
        monkeypatch.setenv("JINA_API_KEY", "j")
        ctx = ToolFactoryContext.from_defaults()
        assert ctx.api_keys == {"brave": "b", "jina": "j"}

    def test_no_env_vars_empty_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        monkeypatch.delenv("JINA_API_KEY", raising=False)
        with patch.dict("sys.modules", {"databricks.sdk": None}):
            ctx = ToolFactoryContext.from_defaults()
        assert ctx.api_keys == {}

    def test_brave_param_populates_api_keys(self) -> None:
        ctx = ToolFactoryContext.from_defaults(brave_api_key="explicit-key")
        assert ctx.api_keys.get("brave") == "explicit-key"
