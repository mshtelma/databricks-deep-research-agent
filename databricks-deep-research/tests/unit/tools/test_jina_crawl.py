"""Unit tests for JinaCrawlAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from databricks_deep_research.tools.builtins.jina_crawl import JinaCrawlAdapter


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


class TestJinaCrawlAdapter:
    @pytest.mark.asyncio
    async def test_returns_content_and_title(self) -> None:
        crawler = JinaCrawlAdapter(api_key="test-key")
        mock_resp = _mock_response({
            "data": {
                "url": "https://example.com",
                "title": "Example Page",
                "content": "Full extracted text of the page.",
            }
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            content, title = await crawler("https://example.com")

        assert content == "Full extracted text of the page."
        assert title == "Example Page"

    @pytest.mark.asyncio
    async def test_unwrapped_response(self) -> None:
        """Handle direct {url, title, content} without data wrapper."""
        crawler = JinaCrawlAdapter()
        mock_resp = _mock_response({
            "url": "https://example.com",
            "title": "Direct",
            "content": "Direct content.",
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            content, title = await crawler("https://example.com")

        assert content == "Direct content."
        assert title == "Direct"

    @pytest.mark.asyncio
    async def test_authorization_header(self) -> None:
        crawler = JinaCrawlAdapter(api_key="jina_xxx")
        mock_resp = _mock_response({
            "data": {"url": "u", "title": "t", "content": "c"}
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await crawler("https://example.com")

        call_kwargs = mock_client.get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer jina_xxx"

    @pytest.mark.asyncio
    async def test_empty_content_raises(self) -> None:
        crawler = JinaCrawlAdapter()
        mock_resp = _mock_response({
            "data": {"url": "u", "title": "t", "content": ""}
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ValueError, match="empty content"):
                await crawler("https://example.com")

    @pytest.mark.asyncio
    async def test_no_title_returns_none(self) -> None:
        crawler = JinaCrawlAdapter()
        mock_resp = _mock_response({
            "data": {"url": "u", "title": "", "content": "some text"}
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            content, title = await crawler("https://example.com")

        assert content == "some text"
        assert title is None

    @pytest.mark.asyncio
    async def test_url_passed_in_path(self) -> None:
        crawler = JinaCrawlAdapter()
        mock_resp = _mock_response({
            "data": {"url": "u", "title": "t", "content": "c"}
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await crawler("https://example.com/page")

        call_args = mock_client.get.call_args
        assert call_args.args[0] == "https://r.jina.ai/https://example.com/page"
