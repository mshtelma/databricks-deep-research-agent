"""Tests for web_crawl content-type routing, the JS-rendering option, and
per-domain rate limiting (spec §4.6, Tier 3 — JS scraping via the integrated
Jina path).

These cover the opt-in capabilities added to :class:`WebCrawlTool`:

* **Content-type URL routing** — a ``.pdf`` URL routes to the injected PDF
  parser instead of the HTML crawler.
* **JS-rendering option** — when ``js_render`` is enabled, pages route through
  the JS-capable backend (the already-integrated Jina ``ContentCrawler``)
  rather than httpx + trafilatura. Playwright is intentionally NOT shipped.
* **Per-domain rate limiting** — same-domain bursts are throttled (algorithm
  ported from gpt-researcher's ``GlobalRateLimiter``, keyed per-domain).
* **Byte-identical default** — with every new knob at its default the crawl
  path is unchanged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from databricks_deep_research.tools.builtins.web_crawl import (
    WebCrawlTool,
    _DomainRateLimiter,
    _is_pdf_url,
)
from databricks_deep_research.tools.protocol import ToolContext, UrlRegistry

# Long enough to clear the _MIN_USEFUL_CONTENT_LENGTH quality gate.
_LONG_TEXT = "This is a real article with meaningful content. " * 10


@pytest.fixture(autouse=True)
def _mock_dns_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decouple tests from real DNS. Fail-closed DNS makes this load-bearing."""
    monkeypatch.setattr(
        "databricks_deep_research.tools.builtins.web_crawl._resolve_and_validate_ip",
        lambda _hostname: (True, "93.184.216.34"),
    )


def _make_context(urls: list[str] | None = None) -> ToolContext:
    registry = UrlRegistry()
    for url in urls or []:
        registry.register(url)
    return ToolContext(url_registry=registry)


# ---------------------------------------------------------------------------
# .pdf detector
# ---------------------------------------------------------------------------


class TestIsPdfUrl:
    """Unit tests for the .pdf content-type detector (ported routing heuristic)."""

    def test_plain_pdf(self) -> None:
        assert _is_pdf_url("https://example.com/report.pdf") is True

    def test_uppercase_extension(self) -> None:
        assert _is_pdf_url("https://example.com/REPORT.PDF") is True

    def test_pdf_with_query_string(self) -> None:
        # Query strings / fragments must not defeat routing.
        assert _is_pdf_url("https://example.com/doc.pdf?dl=1#page=2") is True

    def test_html_url_is_not_pdf(self) -> None:
        assert _is_pdf_url("https://example.com/article.html") is False

    def test_pdf_substring_not_extension(self) -> None:
        assert _is_pdf_url("https://example.com/pdf-viewer/index.html") is False


# ---------------------------------------------------------------------------
# Content-type URL routing
# ---------------------------------------------------------------------------


class TestContentTypeRouting:
    """A .pdf URL routes to the PDF parser; non-PDF URLs do not."""

    @pytest.mark.asyncio
    async def test_pdf_url_routes_to_pdf_parser(self) -> None:
        pdf_parser = AsyncMock(return_value=(_LONG_TEXT, "PDF Title"))
        html_crawler = AsyncMock(return_value=(_LONG_TEXT, "HTML Title"))
        tool = WebCrawlTool(crawler=html_crawler, pdf_parser=pdf_parser)
        ctx = _make_context(["https://example.com/report.pdf"])

        result = await tool.execute({"url_index": 0}, ctx)

        assert result.success
        pdf_parser.assert_awaited_once_with("https://example.com/report.pdf")
        html_crawler.assert_not_awaited()
        assert result.sources[0].title == "PDF Title"

    @pytest.mark.asyncio
    async def test_non_pdf_url_does_not_route_to_pdf_parser(self) -> None:
        pdf_parser = AsyncMock(return_value=(_LONG_TEXT, "PDF Title"))
        html_crawler = AsyncMock(return_value=(_LONG_TEXT, "HTML Title"))
        tool = WebCrawlTool(crawler=html_crawler, pdf_parser=pdf_parser)
        ctx = _make_context(["https://example.com/article.html"])

        result = await tool.execute({"url_index": 0}, ctx)

        assert result.success
        html_crawler.assert_awaited_once_with("https://example.com/article.html")
        pdf_parser.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pdf_url_without_parser_uses_default_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No pdf_parser configured → .pdf URL follows the normal (default) path."""
        mock_crawl = AsyncMock(return_value=(_LONG_TEXT, "Title"))
        monkeypatch.setattr(
            "databricks_deep_research.tools.builtins.web_crawl._default_crawl",
            mock_crawl,
        )
        tool = WebCrawlTool()  # no crawler, no pdf_parser
        ctx = _make_context(["https://example.com/report.pdf"])

        await tool.execute({"url_index": 0}, ctx)

        mock_crawl.assert_awaited_once()


# ---------------------------------------------------------------------------
# JS-rendering option (Tier 3, via the integrated Jina backend)
# ---------------------------------------------------------------------------


class TestJsRenderingOption:
    """A JS-flagged URL routes through the (stubbed) Jina JS-capable backend."""

    @pytest.mark.asyncio
    async def test_js_flag_routes_to_js_crawler(self) -> None:
        # Stub the Jina backend — no network. Any ContentCrawler-shaped mock works.
        js_crawler = AsyncMock(return_value=(_LONG_TEXT, "JS Title"))
        default_crawler = AsyncMock(return_value=(_LONG_TEXT, "Default Title"))
        tool = WebCrawlTool(
            crawler=default_crawler, js_render=True, js_crawler=js_crawler
        )
        ctx = _make_context(["https://spa.example.com/app"])

        result = await tool.execute({"url_index": 0}, ctx)

        assert result.success
        js_crawler.assert_awaited_once_with("https://spa.example.com/app")
        default_crawler.assert_not_awaited()
        assert result.sources[0].title == "JS Title"

    @pytest.mark.asyncio
    async def test_js_render_uses_real_jina_adapter_shape(self) -> None:
        """js_crawler accepts a real JinaCrawlAdapter (stubbed transport, no network)."""
        from databricks_deep_research.tools.builtins.jina_crawl import JinaCrawlAdapter

        adapter = JinaCrawlAdapter(api_key="test-key")
        tool = WebCrawlTool(js_render=True, js_crawler=adapter)
        ctx = _make_context(["https://spa.example.com/app"])

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "url": "https://spa.example.com/app",
                "title": "Rendered SPA",
                "content": _LONG_TEXT,
            }
        }
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await tool.execute({"url_index": 0}, ctx)

        assert result.success
        assert result.sources[0].title == "Rendered SPA"
        # Routed through Jina Reader (r.jina.ai), not httpx+trafilatura.
        assert mock_client.get.call_args.args[0].startswith("https://r.jina.ai/")

    @pytest.mark.asyncio
    async def test_js_render_off_is_default_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """js_render defaults OFF: the default httpx+trafilatura path is used."""
        mock_crawl = AsyncMock(return_value=(_LONG_TEXT, "Title"))
        monkeypatch.setattr(
            "databricks_deep_research.tools.builtins.web_crawl._default_crawl",
            mock_crawl,
        )
        js_crawler = AsyncMock(return_value=(_LONG_TEXT, "JS Title"))
        # js_crawler supplied but js_render is False → must NOT be used.
        tool = WebCrawlTool(js_crawler=js_crawler)
        ctx = _make_context(["https://example.com/page"])

        await tool.execute({"url_index": 0}, ctx)

        mock_crawl.assert_awaited_once()
        js_crawler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pdf_routing_takes_precedence_over_js(self) -> None:
        """A .pdf URL routes to the PDF parser even when JS rendering is on."""
        pdf_parser = AsyncMock(return_value=(_LONG_TEXT, "PDF"))
        js_crawler = AsyncMock(return_value=(_LONG_TEXT, "JS"))
        tool = WebCrawlTool(
            js_render=True, js_crawler=js_crawler, pdf_parser=pdf_parser
        )
        ctx = _make_context(["https://example.com/report.pdf"])

        result = await tool.execute({"url_index": 0}, ctx)

        assert result.success
        pdf_parser.assert_awaited_once()
        js_crawler.assert_not_awaited()


# ---------------------------------------------------------------------------
# Per-domain rate limiting
# ---------------------------------------------------------------------------


class TestDomainRateLimiter:
    """Per-domain throttling (algorithm ported from gpt-researcher)."""

    @pytest.mark.asyncio
    async def test_disabled_when_interval_zero(self) -> None:
        """min_interval == 0 → no sleeping at all (byte-identical default)."""
        limiter = _DomainRateLimiter(0.0)
        slept: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            slept.append(secs)

        with patch("asyncio.sleep", _fake_sleep):
            await limiter.acquire("https://example.com/a")
            await limiter.acquire("https://example.com/b")

        assert slept == []

    @pytest.mark.asyncio
    async def test_throttles_same_domain_burst(self) -> None:
        """A same-domain second request sleeps the remaining interval."""
        limiter = _DomainRateLimiter(1.0)
        slept: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            slept.append(secs)

        # First call records the timestamp; the immediate second call (same domain)
        # must sleep close to the full interval (monotonic gap is ~0).
        with patch("asyncio.sleep", _fake_sleep):
            await limiter.acquire("https://example.com/a")
            await limiter.acquire("https://example.com/b")

        assert len(slept) == 1
        assert 0.0 < slept[0] <= 1.0

    @pytest.mark.asyncio
    async def test_different_domains_not_throttled(self) -> None:
        """Bursts across distinct domains are independent — no throttling."""
        limiter = _DomainRateLimiter(1.0)
        slept: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            slept.append(secs)

        with patch("asyncio.sleep", _fake_sleep):
            await limiter.acquire("https://a.example.com/x")
            await limiter.acquire("https://b.example.com/y")

        assert slept == []

    @pytest.mark.asyncio
    async def test_tool_applies_rate_limiter_on_same_domain(self) -> None:
        """WebCrawlTool throttles a same-domain burst of two crawls."""
        crawler = AsyncMock(return_value=(_LONG_TEXT, "Page"))
        tool = WebCrawlTool(crawler=crawler, min_domain_interval=1.0)
        ctx = _make_context(["https://example.com/one", "https://example.com/two"])
        slept: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            slept.append(secs)

        with patch("asyncio.sleep", _fake_sleep):
            await tool.execute({"url_index": 0}, ctx)
            await tool.execute({"url_index": 1}, ctx)

        assert crawler.await_count == 2
        assert len(slept) == 1  # second same-domain crawl was throttled

    @pytest.mark.asyncio
    async def test_tool_default_no_rate_limiting(self) -> None:
        """Default tool (min_domain_interval=0) never sleeps — byte-identical."""
        crawler = AsyncMock(return_value=(_LONG_TEXT, "Page"))
        tool = WebCrawlTool(crawler=crawler)
        ctx = _make_context(["https://example.com/one", "https://example.com/two"])
        slept: list[float] = []

        async def _fake_sleep(secs: float) -> None:
            slept.append(secs)

        with patch("asyncio.sleep", _fake_sleep):
            await tool.execute({"url_index": 0}, ctx)
            await tool.execute({"url_index": 1}, ctx)

        assert slept == []


# ---------------------------------------------------------------------------
# Default path unchanged
# ---------------------------------------------------------------------------


class TestDefaultPathUnchanged:
    """The default crawl path (all new options off) is unchanged."""

    @pytest.mark.asyncio
    async def test_default_crawler_path_intact(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No crawler + no new knobs → _default_crawl with IP-pinning kwargs."""
        mock_crawl = AsyncMock(return_value=(_LONG_TEXT, "Title"))
        monkeypatch.setattr(
            "databricks_deep_research.tools.builtins.web_crawl._default_crawl",
            mock_crawl,
        )
        tool = WebCrawlTool()
        ctx = _make_context(["https://example.com/page"])

        await tool.execute({"url_index": 0}, ctx)

        mock_crawl.assert_awaited_once()
        call = mock_crawl.call_args
        assert call.kwargs.get("resolved_ip") == "93.184.216.34"
        assert call.kwargs.get("timeout") == 30.0
        assert call.kwargs.get("max_content_length") == 50_000

    @pytest.mark.asyncio
    async def test_injected_crawler_still_gets_plain_url(self) -> None:
        """A plain injected crawler (no new knobs) is called with the raw URL."""
        crawler = AsyncMock(return_value=(_LONG_TEXT, "Page"))
        tool = WebCrawlTool(crawler=crawler)
        ctx = _make_context(["https://example.com/page"])

        await tool.execute({"url_index": 0}, ctx)

        crawler.assert_awaited_once_with("https://example.com/page")
