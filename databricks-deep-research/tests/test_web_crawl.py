"""Tests for the web crawl tool."""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from databricks_deep_research.tools.builtins.web_crawl import (
    WebCrawlTool,
    _check_redirect_ssrf,
    _resolve_and_validate_ip,
)
from databricks_deep_research.tools.protocol import TableRegistry, ToolContext, UrlRegistry


@pytest.fixture(autouse=True)
def _mock_dns_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decouple tests from real DNS. Fail-closed DNS makes this load-bearing."""
    monkeypatch.setattr(
        "databricks_deep_research.tools.builtins.web_crawl._resolve_and_validate_ip",
        lambda hostname: (True, "93.184.216.34"),
    )


def _make_context(
    urls: list[str] | None = None,
    *,
    table_registry: TableRegistry | None = None,
) -> ToolContext:
    registry = UrlRegistry()
    for url in urls or []:
        registry.register(url)
    return ToolContext(url_registry=registry, table_registry=table_registry)


def _mock_crawler(text: str, title: str | None = None) -> AsyncMock:
    return AsyncMock(return_value=(text, title))


def _mock_http_error(status_code: int) -> AsyncMock:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(status_code=status_code, request=request)
    return AsyncMock(side_effect=httpx.HTTPStatusError("boom", request=request, response=response))


# ---------------------------------------------------------------------------
# Quality gate: short content rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_short_content_rejected() -> None:
    """Pages with very short content are rejected as low quality."""
    crawler = _mock_crawler("Access Denied", "Error")
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"])

    result = await tool.execute({"url_index": 0}, ctx)

    assert not result.success
    assert "too short" in result.content
    assert result.sources == []


@pytest.mark.asyncio
async def test_empty_content_rejected() -> None:
    """Pages with empty content are rejected."""
    crawler = _mock_crawler("", None)
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"])

    result = await tool.execute({"url_index": 0}, ctx)

    assert not result.success
    assert "no readable text" in result.content


@pytest.mark.asyncio
async def test_sufficient_content_accepted() -> None:
    """Pages with enough content pass the quality gate."""
    long_text = "This is a real article with meaningful content. " * 10
    crawler = _mock_crawler(long_text, "Good Article")
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"])

    result = await tool.execute({"url_index": 0}, ctx)

    assert result.success
    assert len(result.sources) == 1
    assert result.sources[0].url == "https://example.com"


@pytest.mark.asyncio
async def test_whitespace_only_short_content_rejected() -> None:
    """Pages with mostly whitespace and short text are rejected."""
    crawler = _mock_crawler("   Forbidden   \n\n  ", "403")
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"])

    result = await tool.execute({"url_index": 0}, ctx)

    assert not result.success
    assert "too short" in result.content


@pytest.mark.asyncio
async def test_repeated_non_retryable_url_failure_is_suppressed() -> None:
    crawler = _mock_http_error(403)
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"])

    first = await tool.execute({"url_index": 0}, ctx)
    second = await tool.execute({"url_index": 0}, ctx)

    assert not first.success
    assert not second.success
    assert second.error == "Suppressed repeated crawl failure"
    assert crawler.await_count == 1


@pytest.mark.asyncio
async def test_domain_suppression_blocks_target_after_threshold_failures() -> None:
    """After ``_DOMAIN_SUPPRESSION_THRESHOLD`` distinct non-retryable failures on a
    single domain, the next distinct URL on that domain is suppressed without a crawl.

    Kept threshold-relative (derived from the constant rather than hardcoded) so
    tuning ``_DOMAIN_SUPPRESSION_THRESHOLD`` — raised 2->4 in cfe0b2e so a couple
    of transient failures don't disable a whole domain — doesn't silently re-break
    this test.
    """
    threshold = UrlRegistry._DOMAIN_SUPPRESSION_THRESHOLD
    crawler = _mock_http_error(403)
    tool = WebCrawlTool(crawler=crawler)
    # ``threshold`` distinct URLs that each fail-and-crawl, plus one more on the
    # same domain that should be suppressed once the threshold is reached.
    urls = [f"https://example.com/p{i}" for i in range(threshold + 1)]
    ctx = _make_context(urls)

    results = [await tool.execute({"url_index": i}, ctx) for i in range(threshold + 1)]

    # Every distinct URL fails; only the first ``threshold`` are actually crawled.
    assert all(not r.success for r in results)
    assert crawler.await_count == threshold

    # The URL past the threshold is suppressed at domain scope without a crawl.
    suppressed = results[-1]
    assert suppressed.error == "Suppressed repeated crawl failure"
    assert suppressed.data["suppression_scope"] == "domain"


# ---------------------------------------------------------------------------
# Table extraction in crawled content
# ---------------------------------------------------------------------------

_TABLE_TEXT = (
    "Some intro text about GDP data.\n\n"
    "| Country | GDP |\n"
    "|---|---|\n"
    "| US | 25T |\n"
    "| China | 18T |\n\n"
    "Some trailing text."
)


@pytest.mark.asyncio
async def test_table_detected_and_registered() -> None:
    """Markdown tables in crawled content are detected and registered."""
    crawler = _mock_crawler(_TABLE_TEXT, "GDP Page")
    reg = TableRegistry()
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"], table_registry=reg)

    result = await tool.execute({"url_index": 0}, ctx)

    assert result.success
    assert "table_idx=" in result.content
    assert result.data.get("table_count", 0) >= 1
    assert len(reg) >= 1

    entry = reg.resolve(0)
    assert entry is not None
    assert entry.source_kind == "web"
    assert "example.com" in entry.source_label


@pytest.mark.asyncio
async def test_extract_tables_disabled() -> None:
    """extract_tables=False suppresses table detection."""
    crawler = _mock_crawler(_TABLE_TEXT, "GDP Page")
    reg = TableRegistry()
    tool = WebCrawlTool(crawler=crawler, extract_tables=False)
    ctx = _make_context(["https://example.com"], table_registry=reg)

    result = await tool.execute({"url_index": 0}, ctx)

    assert result.success
    assert len(reg) == 0
    assert "table_idx" not in result.content


@pytest.mark.asyncio
async def test_table_no_registry_no_crash() -> None:
    """Table detection works even when table_registry is None (no registration)."""
    crawler = _mock_crawler(_TABLE_TEXT, "GDP Page")
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"])  # no table_registry

    result = await tool.execute({"url_index": 0}, ctx)

    assert result.success
    # Tables detected in data but not registered
    assert result.data.get("table_count", 0) >= 1


@pytest.mark.asyncio
async def test_table_registry_capacity_overflow() -> None:
    """When registry is full, tables are skipped gracefully."""
    crawler = _mock_crawler(_TABLE_TEXT, "GDP Page")
    reg = TableRegistry(max_tables=0)  # already full
    tool = WebCrawlTool(crawler=crawler)
    ctx = _make_context(["https://example.com"], table_registry=reg)

    result = await tool.execute({"url_index": 0}, ctx)

    assert result.success
    assert len(reg) == 0  # nothing registered


# ---------------------------------------------------------------------------
# DNS resolution and validation (_resolve_and_validate_ip)
# ---------------------------------------------------------------------------


class TestResolveAndValidateIp:
    """Unit tests for the fail-closed DNS resolver."""

    def test_empty_hostname(self) -> None:
        is_safe, ip = _resolve_and_validate_ip("")
        assert is_safe is False
        assert ip is None

    def test_blocked_hostname(self) -> None:
        is_safe, ip = _resolve_and_validate_ip("metadata.google.internal")
        assert is_safe is False
        assert ip is None

    @patch("socket.getaddrinfo")
    def test_public_ipv4(self, mock_getaddrinfo: MagicMock) -> None:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
        ]
        is_safe, ip = _resolve_and_validate_ip("example.com")
        assert is_safe is True
        assert ip == "93.184.216.34"

    @patch("socket.getaddrinfo")
    def test_private_ipv4(self, mock_getaddrinfo: MagicMock) -> None:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.1", 0)),
        ]
        is_safe, ip = _resolve_and_validate_ip("evil.internal")
        assert is_safe is False
        assert ip is None

    @patch("socket.getaddrinfo")
    def test_loopback(self, mock_getaddrinfo: MagicMock) -> None:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        ]
        is_safe, ip = _resolve_and_validate_ip("localhost")
        assert is_safe is False
        assert ip is None

    @patch("socket.getaddrinfo")
    def test_link_local(self, mock_getaddrinfo: MagicMock) -> None:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0)),
        ]
        is_safe, ip = _resolve_and_validate_ip("metadata.cloud")
        assert is_safe is False
        assert ip is None

    @patch("socket.getaddrinfo")
    def test_dns_failure_fail_closed(self, mock_getaddrinfo: MagicMock) -> None:
        mock_getaddrinfo.side_effect = socket.gaierror("Name resolution failed")
        is_safe, ip = _resolve_and_validate_ip("nonexistent.example")
        assert is_safe is False
        assert ip is None

    @patch("socket.getaddrinfo")
    def test_mixed_public_private(self, mock_getaddrinfo: MagicMock) -> None:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.1", 0)),
        ]
        is_safe, ip = _resolve_and_validate_ip("dual-homed.example")
        assert is_safe is False
        assert ip is None


# ---------------------------------------------------------------------------
# SSRF protection integration tests
# ---------------------------------------------------------------------------


class TestSSRFProtection:
    """Integration tests for SSRF protection through execute()."""

    @pytest.mark.asyncio
    async def test_private_ip_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """URLs resolving to private IPs are blocked before crawling."""
        monkeypatch.setattr(
            "databricks_deep_research.tools.builtins.web_crawl._resolve_and_validate_ip",
            lambda hostname: (False, None),
        )
        crawler = _mock_crawler("should not be called", "Nope")
        tool = WebCrawlTool(crawler=crawler)
        ctx = _make_context(["https://example.com"])

        result = await tool.execute({"url_index": 0}, ctx)

        assert not result.success
        assert "private/internal" in result.content
        crawler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resolved_ip_passed_to_default_crawl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The resolved IP is forwarded to _default_crawl."""
        monkeypatch.setattr(
            "databricks_deep_research.tools.builtins.web_crawl._resolve_and_validate_ip",
            lambda hostname: (True, "93.184.216.34"),
        )
        mock_crawl = AsyncMock(return_value=("Some content " * 20, "Title"))
        monkeypatch.setattr(
            "databricks_deep_research.tools.builtins.web_crawl._default_crawl",
            mock_crawl,
        )
        tool = WebCrawlTool()  # no custom crawler → uses _default_crawl
        ctx = _make_context(["https://example.com"])

        await tool.execute({"url_index": 0}, ctx)

        mock_crawl.assert_awaited_once()
        call_kwargs = mock_crawl.call_args
        assert call_kwargs.kwargs.get("resolved_ip") == "93.184.216.34"

    @pytest.mark.asyncio
    async def test_custom_crawler_gets_original_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Custom crawlers receive the original URL, not an IP-pinned one."""
        monkeypatch.setattr(
            "databricks_deep_research.tools.builtins.web_crawl._resolve_and_validate_ip",
            lambda hostname: (True, "93.184.216.34"),
        )
        crawler = _mock_crawler("Real content for testing. " * 10, "Page")
        tool = WebCrawlTool(crawler=crawler)
        ctx = _make_context(["https://example.com/page"])

        await tool.execute({"url_index": 0}, ctx)

        crawler.assert_awaited_once_with("https://example.com/page")


# ---------------------------------------------------------------------------
# Redirect SSRF tests
# ---------------------------------------------------------------------------


class TestRedirectSSRF:
    """Unit tests for _check_redirect_ssrf."""

    @pytest.mark.asyncio
    async def test_file_scheme_blocked(self) -> None:
        """Redirects to file:// scheme are blocked."""
        response = MagicMock()
        response.is_redirect = True
        response.headers = {"location": "file:///etc/passwd"}
        response.request = httpx.Request("GET", "https://evil.com")

        with pytest.raises(httpx.TooManyRedirects, match="disallowed scheme"):
            await _check_redirect_ssrf(response)

    @pytest.mark.asyncio
    async def test_gopher_scheme_blocked(self) -> None:
        """Redirects to gopher:// scheme are blocked."""
        response = MagicMock()
        response.is_redirect = True
        response.headers = {"location": "gopher://evil:25/"}
        response.request = httpx.Request("GET", "https://evil.com")

        with pytest.raises(httpx.TooManyRedirects, match="disallowed scheme"):
            await _check_redirect_ssrf(response)

    @pytest.mark.asyncio
    async def test_https_redirect_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Redirects to https:// with public IPs are allowed."""
        monkeypatch.setattr(
            "databricks_deep_research.tools.builtins.web_crawl._is_private_url",
            lambda hostname: False,
        )
        response = MagicMock()
        response.is_redirect = True
        response.headers = {"location": "https://safe.example.com/"}
        response.request = httpx.Request("GET", "https://origin.com")

        # Should not raise
        await _check_redirect_ssrf(response)

    @pytest.mark.asyncio
    async def test_redirect_to_private_ip_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Redirects to hosts resolving to private IPs are blocked."""
        monkeypatch.setattr(
            "databricks_deep_research.tools.builtins.web_crawl._is_private_url",
            lambda hostname: True,
        )
        response = MagicMock()
        response.is_redirect = True
        response.headers = {"location": "http://evil.com/"}
        response.request = httpx.Request("GET", "https://origin.com")

        with pytest.raises(httpx.TooManyRedirects, match="private IP"):
            await _check_redirect_ssrf(response)
