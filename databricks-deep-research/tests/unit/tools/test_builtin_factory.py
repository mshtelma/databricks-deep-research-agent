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
