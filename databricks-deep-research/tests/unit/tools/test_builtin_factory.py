"""Unit tests for BuiltinToolFactory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.workflow.definition import ToolDeclaration


class TestBuiltinFactoryWebCrawl:
    """BuiltinToolFactory wires web_crawl with or without an injected crawler."""

    @pytest.mark.asyncio
    async def test_web_crawl_allows_default_crawler(self) -> None:
        """web_crawl creation should succeed when ctx.crawler is None."""
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="web_crawl", kind="web_crawl", config={})
        ctx = ToolFactoryContext()  # crawler=None by default

        tool = await factory.create(decl, ctx)

        assert tool.definition.name == "web_crawl"
        assert tool._crawler is None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_web_crawl_passes_config(self) -> None:
        """timeout and max_content_length from decl.config are forwarded."""
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_crawl",
            kind="web_crawl",
            config={"timeout": 15.0, "max_content_length": 10_000},
        )
        ctx = ToolFactoryContext(crawler=MagicMock())

        tool = await factory.create(decl, ctx)

        # Access internals to verify config was passed through
        assert tool._timeout == 15.0  # type: ignore[attr-defined]
        assert tool._max_content_length == 10_000  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_web_crawl_default_config(self) -> None:
        """Default timeout/max_content_length when not in decl.config."""
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="web_crawl", kind="web_crawl", config={})
        ctx = ToolFactoryContext(crawler=MagicMock())

        tool = await factory.create(decl, ctx)

        assert tool._timeout == 30.0  # type: ignore[attr-defined]
        assert tool._max_content_length == 50_000  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_web_search_still_requires_search_client(self) -> None:
        """web_search still raises ValueError when search_client is None."""
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="web_search", kind="web_search", config={})
        ctx = ToolFactoryContext()

        with pytest.raises(ValueError, match="search_client required"):
            await factory.create(decl, ctx)


class TestProviderResolution:
    """Provider-based tool creation via config.provider."""

    @pytest.mark.asyncio
    async def test_jina_search_provider(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_search", kind="web_search",
            config={"provider": "jina"},
        )
        ctx = ToolFactoryContext(api_keys={"jina": "test-key"})

        tool = await factory.create(decl, ctx)

        assert tool.definition.name == "web_search"
        from databricks_deep_research.tools.builtins.jina_search import JinaSearchAdapter
        assert isinstance(tool._client, JinaSearchAdapter)  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_jina_search_no_key_ok(self) -> None:
        """Jina works without API key."""
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_search", kind="web_search",
            config={"provider": "jina"},
        )
        ctx = ToolFactoryContext()

        tool = await factory.create(decl, ctx)
        assert tool.definition.name == "web_search"

    @pytest.mark.asyncio
    async def test_brave_provider_with_api_keys(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_search", kind="web_search",
            config={"provider": "brave"},
        )
        ctx = ToolFactoryContext(api_keys={"brave": "brave-key-123"})

        tool = await factory.create(decl, ctx)

        assert tool.definition.name == "web_search"
        from databricks_deep_research.tools.builtins.brave_search import BraveSearchAdapter
        assert isinstance(tool._client, BraveSearchAdapter)  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_brave_provider_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_search", kind="web_search",
            config={"provider": "brave"},
        )
        ctx = ToolFactoryContext()

        with pytest.raises(ValueError, match="BRAVE_API_KEY"):
            await factory.create(decl, ctx)

    @pytest.mark.asyncio
    async def test_unknown_search_provider_raises(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_search", kind="web_search",
            config={"provider": "google"},
        )
        ctx = ToolFactoryContext()

        with pytest.raises(ValueError, match="Unknown search provider.*google"):
            await factory.create(decl, ctx)

    @pytest.mark.asyncio
    async def test_no_provider_uses_legacy_search_client(self) -> None:
        """No provider key → legacy path using ctx.search_client."""
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="web_search", kind="web_search", config={})
        mock_client = MagicMock()
        ctx = ToolFactoryContext(search_client=mock_client)

        tool = await factory.create(decl, ctx)

        assert tool._client is mock_client  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_max_content_per_result_passed(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_search", kind="web_search",
            config={"provider": "jina", "max_content_per_result": 2000},
        )
        ctx = ToolFactoryContext()

        tool = await factory.create(decl, ctx)
        assert tool._max_content_per_result == 2000  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_jina_crawl_provider(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_crawl", kind="web_crawl",
            config={"provider": "jina"},
        )
        ctx = ToolFactoryContext(api_keys={"jina": "key"})

        tool = await factory.create(decl, ctx)

        assert tool.definition.name == "web_crawl"
        from databricks_deep_research.tools.builtins.jina_crawl import JinaCrawlAdapter
        assert isinstance(tool._crawler, JinaCrawlAdapter)  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_web_crawl_no_provider_uses_ctx_crawler(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="web_crawl", kind="web_crawl", config={})
        mock_crawler = MagicMock()
        ctx = ToolFactoryContext(crawler=mock_crawler)

        tool = await factory.create(decl, ctx)
        assert tool._crawler is mock_crawler  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_unknown_crawl_provider_raises(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="web_crawl", kind="web_crawl",
            config={"provider": "bing"},
        )
        ctx = ToolFactoryContext()

        with pytest.raises(ValueError, match="Unknown crawl provider.*bing"):
            await factory.create(decl, ctx)
