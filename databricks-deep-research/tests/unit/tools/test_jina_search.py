"""Unit tests for JinaSearchAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from databricks_deep_research.tools.builtins.jina_search import JinaSearchAdapter


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture()
def jina_results() -> dict:
    return {
        "data": [
            {
                "url": "https://example.com/a",
                "title": "Article A",
                "description": "Short description of A",
                "content": "Full page content of article A " * 50,
            },
            {
                "url": "https://example.com/b",
                "title": "Article B",
                "content": "Full page content of article B " * 30,
            },
        ]
    }


class TestJinaSearchAdapter:
    @pytest.mark.asyncio
    async def test_maps_results_with_content(self, jina_results: dict) -> None:
        adapter = JinaSearchAdapter(api_key="test-key")
        mock_resp = _mock_response(jina_results)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = await adapter.search("test query", count=5)

        assert len(results) == 2
        # First result: has description → used as snippet
        assert results[0].url == "https://example.com/a"
        assert results[0].title == "Article A"
        assert results[0].snippet == "Short description of A"
        assert results[0].content is not None
        assert results[0].content.startswith("Full page content of article A")
        # Second result: no description → snippet = content[:200]
        assert results[1].url == "https://example.com/b"
        assert results[1].snippet == results[1].content[:200]

    @pytest.mark.asyncio
    async def test_authorization_header_with_key(self) -> None:
        adapter = JinaSearchAdapter(api_key="jina_xxx")
        mock_resp = _mock_response({"data": []})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await adapter.search("q")

        call_kwargs = mock_client.get.call_args
        assert call_kwargs.kwargs["headers"]["Authorization"] == "Bearer jina_xxx"

    @pytest.mark.asyncio
    async def test_no_authorization_without_key(self) -> None:
        adapter = JinaSearchAdapter()
        mock_resp = _mock_response({"data": []})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await adapter.search("q")

        call_kwargs = mock_client.get.call_args
        assert "Authorization" not in call_kwargs.kwargs["headers"]

    @pytest.mark.asyncio
    async def test_count_maps_to_num_param(self) -> None:
        adapter = JinaSearchAdapter()
        mock_resp = _mock_response({"data": []})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await adapter.search("q", count=3)

        call_kwargs = mock_client.get.call_args
        assert call_kwargs.kwargs["params"]["num"] == 3

    @pytest.mark.asyncio
    async def test_empty_response(self) -> None:
        adapter = JinaSearchAdapter()
        mock_resp = _mock_response({"data": []})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = await adapter.search("q")

        assert results == []

    @pytest.mark.asyncio
    async def test_freshness_accepted_without_error(self) -> None:
        """freshness is accepted for protocol conformance but ignored."""
        adapter = JinaSearchAdapter()
        mock_resp = _mock_response({"data": []})

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = await adapter.search("q", freshness="pd")

        assert results == []

    @pytest.mark.asyncio
    async def test_empty_content_treated_as_none(self) -> None:
        adapter = JinaSearchAdapter()
        mock_resp = _mock_response({
            "data": [{"url": "https://x.com", "title": "X", "content": ""}]
        })

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            results = await adapter.search("q")

        assert len(results) == 1
        assert results[0].content is None

    @pytest.mark.asyncio
    async def test_http_error_propagates(self) -> None:
        import httpx

        adapter = JinaSearchAdapter()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429 Too Many Requests",
            request=MagicMock(),
            response=MagicMock(status_code=429),
        )

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await adapter.search("q")
