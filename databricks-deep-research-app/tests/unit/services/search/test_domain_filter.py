"""Unit tests for domain filtering with wildcard patterns."""

import pytest

from deep_research.core.app_config import DomainFilterConfig, DomainFilterMode
from deep_research.services.search.domain_filter import (
    DomainFilter,
    extract_domain,
    match_domain_pattern,
)


class TestExtractDomain:
    """Tests for domain extraction from URLs."""

    def test_simple_url(self) -> None:
        assert extract_domain("https://example.com/path") == "example.com"

    def test_subdomain(self) -> None:
        assert extract_domain("https://www.example.com") == "www.example.com"

    def test_with_port(self) -> None:
        assert extract_domain("https://example.com:8080/path") == "example.com"

    def test_with_query_params(self) -> None:
        assert extract_domain("https://example.com/path?q=test") == "example.com"

    def test_http_url(self) -> None:
        assert extract_domain("http://example.com") == "example.com"

    def test_uppercase_domain(self) -> None:
        assert extract_domain("https://EXAMPLE.COM/path") == "example.com"

    def test_invalid_url(self) -> None:
        assert extract_domain("not-a-url") == ""

    def test_empty_url(self) -> None:
        assert extract_domain("") == ""


class TestMatchDomainPattern:
    """Tests for wildcard pattern matching."""

    def test_exact_match(self) -> None:
        assert match_domain_pattern("example.com", "example.com") is True
        assert match_domain_pattern("other.com", "example.com") is False

    def test_case_insensitive(self) -> None:
        assert match_domain_pattern("EXAMPLE.COM", "example.com") is True
        assert match_domain_pattern("example.com", "EXAMPLE.COM") is True

    def test_suffix_wildcard_basic(self) -> None:
        """*.gov should match any .gov domain."""
        assert match_domain_pattern("cdc.gov", "*.gov") is True
        assert match_domain_pattern("nasa.gov", "*.gov") is True

    def test_suffix_wildcard_with_subdomain(self) -> None:
        """*.gov should match subdomains too."""
        assert match_domain_pattern("www.cdc.gov", "*.gov") is True
        assert match_domain_pattern("api.data.gov", "*.gov") is True

    def test_suffix_wildcard_no_false_positive(self) -> None:
        """*.gov should NOT match gov appearing elsewhere."""
        assert match_domain_pattern("gov.example.com", "*.gov") is False
        assert match_domain_pattern("govexample.com", "*.gov") is False

    def test_suffix_wildcard_edu(self) -> None:
        """*.edu should match educational domains."""
        assert match_domain_pattern("mit.edu", "*.edu") is True
        assert match_domain_pattern("www.stanford.edu", "*.edu") is True
        assert match_domain_pattern("edu.example.com", "*.edu") is False

    def test_prefix_wildcard_basic(self) -> None:
        """news.* should match news with any TLD."""
        assert match_domain_pattern("news.com", "news.*") is True
        assert match_domain_pattern("news.org", "news.*") is True

    def test_prefix_wildcard_with_subdomain(self) -> None:
        """news.* should match news as a subdomain prefix."""
        assert match_domain_pattern("news.bbc.com", "news.*") is True

    def test_prefix_wildcard_no_false_positive(self) -> None:
        """news.* should NOT match fakenews.com."""
        assert match_domain_pattern("fakenews.com", "news.*") is False
        assert match_domain_pattern("oldnews.com", "news.*") is False

    def test_complex_wildcard(self) -> None:
        """*.example.* should match subdomains with any TLD."""
        assert match_domain_pattern("sub.example.com", "*.example.*") is True
        assert match_domain_pattern("api.example.org", "*.example.*") is True

    def test_wikipedia_pattern(self) -> None:
        """*.wikipedia.org should match all Wikipedia subdomains."""
        assert match_domain_pattern("en.wikipedia.org", "*.wikipedia.org") is True
        assert match_domain_pattern("de.wikipedia.org", "*.wikipedia.org") is True
        assert match_domain_pattern("wikipedia.org", "*.wikipedia.org") is True

    def test_whitespace_handling(self) -> None:
        """Patterns with whitespace should be trimmed."""
        assert match_domain_pattern("example.com", " example.com ") is True
        assert match_domain_pattern(" example.com ", "example.com") is True


class TestDomainFilterExcludeMode:
    """Tests for DomainFilter in exclude (blacklist) mode."""

    def test_no_filters_allows_all(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.EXCLUDE,
            exclude_domains=[],
        )
        filter = DomainFilter(config)

        result = filter.is_allowed("https://example.com")
        assert result.allowed is True

    def test_exclude_exact_domain(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.EXCLUDE,
            exclude_domains=["spam.com"],
        )
        filter = DomainFilter(config)

        assert filter.is_allowed("https://spam.com").allowed is False
        assert filter.is_allowed("https://notspam.com").allowed is True

    def test_exclude_wildcard_pattern(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.EXCLUDE,
            exclude_domains=["*.ru", "*.cn"],
        )
        filter = DomainFilter(config)

        assert filter.is_allowed("https://evil.ru").allowed is False
        assert filter.is_allowed("https://malware.cn").allowed is False
        assert filter.is_allowed("https://example.com").allowed is True

    def test_exclude_multiple_patterns(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.EXCLUDE,
            exclude_domains=["spam.com", "*.ru", "malware.*"],
        )
        filter = DomainFilter(config)

        assert filter.is_allowed("https://spam.com").allowed is False
        assert filter.is_allowed("https://evil.ru").allowed is False
        assert filter.is_allowed("https://malware.net").allowed is False
        assert filter.is_allowed("https://safe.com").allowed is True


class TestDomainFilterIncludeMode:
    """Tests for DomainFilter in include (whitelist) mode."""

    def test_empty_include_list(self) -> None:
        """Empty include list means nothing is allowed."""
        config = DomainFilterConfig(
            mode=DomainFilterMode.INCLUDE,
            include_domains=[],
        )
        filter = DomainFilter(config)

        # With empty include list, filter is not active
        assert filter.is_active is False

    def test_include_exact_domain(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.INCLUDE,
            include_domains=["trusted.com"],
        )
        filter = DomainFilter(config)

        assert filter.is_allowed("https://trusted.com").allowed is True
        assert filter.is_allowed("https://untrusted.com").allowed is False

    def test_include_wildcard_pattern(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.INCLUDE,
            include_domains=["*.gov", "*.edu"],
        )
        filter = DomainFilter(config)

        assert filter.is_allowed("https://cdc.gov").allowed is True
        assert filter.is_allowed("https://mit.edu").allowed is True
        assert filter.is_allowed("https://example.com").allowed is False

    def test_include_multiple_patterns(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.INCLUDE,
            include_domains=["reuters.com", "bbc.com", "*.gov"],
        )
        filter = DomainFilter(config)

        assert filter.is_allowed("https://reuters.com").allowed is True
        assert filter.is_allowed("https://bbc.com").allowed is True
        assert filter.is_allowed("https://cdc.gov").allowed is True
        assert filter.is_allowed("https://random.com").allowed is False


class TestDomainFilterBothMode:
    """Tests for DomainFilter in 'both' mode (whitelist + blacklist)."""

    def test_both_mode_include_then_exclude(self) -> None:
        """In 'both' mode, exclude takes precedence."""
        config = DomainFilterConfig(
            mode=DomainFilterMode.BOTH,
            include_domains=["*.gov", "*.edu", "*.com"],
            exclude_domains=["spam.com"],
        )
        filter = DomainFilter(config)

        assert filter.is_allowed("https://cdc.gov").allowed is True
        assert filter.is_allowed("https://example.com").allowed is True
        assert filter.is_allowed("https://spam.com").allowed is False  # Excluded

    def test_both_mode_must_match_include(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.BOTH,
            include_domains=["*.gov"],
            exclude_domains=["spam.gov"],
        )
        filter = DomainFilter(config)

        assert filter.is_allowed("https://cdc.gov").allowed is True
        assert filter.is_allowed("https://spam.gov").allowed is False  # Excluded
        assert filter.is_allowed("https://example.com").allowed is False  # Not in include


class TestDomainFilterResults:
    """Tests for filtering search result lists."""

    def test_filter_results_removes_blocked(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.EXCLUDE,
            exclude_domains=["blocked.com"],
        )
        filter = DomainFilter(config)

        results = [
            {"url": "https://allowed.com", "title": "Allowed"},
            {"url": "https://blocked.com", "title": "Blocked"},
            {"url": "https://also-allowed.org", "title": "Also Allowed"},
        ]

        filtered = filter.filter_results(results)

        assert len(filtered) == 2
        assert all(r["url"] != "https://blocked.com" for r in filtered)

    def test_filter_results_preserves_order(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.INCLUDE,
            include_domains=["first.com", "third.com"],
        )
        filter = DomainFilter(config)

        results = [
            {"url": "https://first.com", "title": "First"},
            {"url": "https://second.com", "title": "Second"},
            {"url": "https://third.com", "title": "Third"},
        ]

        filtered = filter.filter_results(results)

        assert len(filtered) == 2
        assert filtered[0]["title"] == "First"
        assert filtered[1]["title"] == "Third"

    def test_filter_results_custom_url_key(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.EXCLUDE,
            exclude_domains=["bad.com"],
        )
        filter = DomainFilter(config)

        results = [
            {"link": "https://good.com", "name": "Good"},
            {"link": "https://bad.com", "name": "Bad"},
        ]

        filtered = filter.filter_results(results, url_key="link")

        assert len(filtered) == 1
        assert filtered[0]["name"] == "Good"


class TestDomainFilterIsActive:
    """Tests for is_active property."""

    def test_exclude_mode_active_with_excludes(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.EXCLUDE,
            exclude_domains=["*.ru"],
        )
        assert DomainFilter(config).is_active is True

    def test_exclude_mode_inactive_without_excludes(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.EXCLUDE,
            exclude_domains=[],
        )
        assert DomainFilter(config).is_active is False

    def test_include_mode_active_with_includes(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.INCLUDE,
            include_domains=["*.gov"],
        )
        assert DomainFilter(config).is_active is True

    def test_include_mode_inactive_without_includes(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.INCLUDE,
            include_domains=[],
        )
        assert DomainFilter(config).is_active is False

    def test_both_mode_active_with_either(self) -> None:
        # Active with just includes
        config1 = DomainFilterConfig(
            mode=DomainFilterMode.BOTH,
            include_domains=["*.gov"],
            exclude_domains=[],
        )
        assert DomainFilter(config1).is_active is True

        # Active with just excludes
        config2 = DomainFilterConfig(
            mode=DomainFilterMode.BOTH,
            include_domains=[],
            exclude_domains=["*.ru"],
        )
        assert DomainFilter(config2).is_active is True


class TestDomainFilterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_url_not_allowed(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.EXCLUDE,
            exclude_domains=[],
        )
        filter = DomainFilter(config)

        result = filter.is_allowed("not-a-valid-url")
        assert result.allowed is False
        assert "Could not extract domain" in result.reason

    def test_empty_url_not_allowed(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.EXCLUDE,
            exclude_domains=[],
        )
        filter = DomainFilter(config)

        result = filter.is_allowed("")
        assert result.allowed is False

    def test_match_result_contains_pattern(self) -> None:
        config = DomainFilterConfig(
            mode=DomainFilterMode.EXCLUDE,
            exclude_domains=["*.ru"],
        )
        filter = DomainFilter(config)

        result = filter.is_allowed("https://evil.ru")
        assert result.allowed is False
        assert result.matched_pattern == "*.ru"
        assert "*.ru" in result.reason
