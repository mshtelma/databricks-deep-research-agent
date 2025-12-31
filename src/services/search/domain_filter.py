"""Domain filtering with wildcard pattern support.

Provides URL filtering based on domain whitelist/blacklist configuration.
Supports wildcard patterns for flexible domain matching.
"""

import fnmatch
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from src.core.app_config import DomainFilterConfig, DomainFilterMode

logger = logging.getLogger(__name__)


@dataclass
class DomainMatchResult:
    """Result of domain matching operation."""

    allowed: bool
    matched_pattern: str | None = None
    reason: str = ""


def extract_domain(url: str) -> str:
    """Extract domain from URL, handling edge cases.

    Args:
        url: Full URL string

    Returns:
        Domain/hostname extracted from URL, lowercase
    """
    try:
        parsed = urlparse(url)
        domain = parsed.hostname or ""
        return domain.lower()
    except Exception:
        return ""


def match_domain_pattern(domain: str, pattern: str) -> bool:
    """Match a domain against a wildcard pattern.

    Supports patterns like:
    - "*.gov" - matches any .gov domain (e.g., cdc.gov, www.nasa.gov)
    - "*.edu" - matches any .edu domain
    - "news.*" - matches news.com, news.org, etc.
    - "*.example.*" - matches any subdomain and TLD of example
    - "exact.com" - exact match only

    Args:
        domain: Domain to check (e.g., "www.cdc.gov")
        pattern: Pattern with optional wildcards (e.g., "*.gov")

    Returns:
        True if domain matches pattern
    """
    domain = domain.lower().strip()
    pattern = pattern.lower().strip()

    # Handle exact match first
    if domain == pattern:
        return True

    # For suffix patterns (*.gov), match domain suffix
    # *.gov → matches cdc.gov, www.cdc.gov
    if pattern.startswith("*."):
        suffix = pattern[2:]  # Remove "*."
        if domain == suffix or domain.endswith("." + suffix):
            return True

    # For prefix patterns (news.*), match domain prefix
    # news.* → matches news.com, news.org
    if pattern.endswith(".*"):
        prefix = pattern[:-2]  # Remove ".*"
        if domain == prefix or domain.startswith(prefix + "."):
            return True

    # General fnmatch for complex patterns like *.example.*
    return fnmatch.fnmatch(domain, pattern)


class DomainFilter:
    """Filters domains based on whitelist/blacklist configuration.

    Thread-safe domain filtering with support for three modes:
    - INCLUDE: Only domains in include_domains are allowed
    - EXCLUDE: Domains in exclude_domains are blocked
    - BOTH: Must be in include_domains AND not in exclude_domains
    """

    def __init__(self, config: DomainFilterConfig) -> None:
        """Initialize domain filter.

        Args:
            config: Domain filter configuration
        """
        self._config = config
        self._include_patterns = [p.lower() for p in config.include_domains]
        self._exclude_patterns = [p.lower() for p in config.exclude_domains]

    @property
    def is_active(self) -> bool:
        """Check if any filtering is configured."""
        if self._config.mode == DomainFilterMode.INCLUDE:
            return bool(self._include_patterns)
        elif self._config.mode == DomainFilterMode.EXCLUDE:
            return bool(self._exclude_patterns)
        else:  # BOTH
            return bool(self._include_patterns) or bool(self._exclude_patterns)

    def is_allowed(self, url: str) -> DomainMatchResult:
        """Check if a URL's domain is allowed.

        Args:
            url: URL to check

        Returns:
            DomainMatchResult with allowed status and reason
        """
        domain = extract_domain(url)
        if not domain:
            return DomainMatchResult(
                allowed=False, reason="Could not extract domain from URL"
            )

        mode = self._config.mode

        # Check exclude list first (always blocks if matched)
        if mode in (DomainFilterMode.EXCLUDE, DomainFilterMode.BOTH):
            for pattern in self._exclude_patterns:
                if match_domain_pattern(domain, pattern):
                    if self._config.log_filtered:
                        logger.info(
                            "DOMAIN_BLOCKED",
                            extra={
                                "domain": domain,
                                "pattern": pattern,
                                "url": url[:100],
                            },
                        )
                    return DomainMatchResult(
                        allowed=False,
                        matched_pattern=pattern,
                        reason=f"Domain blocked by exclude pattern: {pattern}",
                    )

        # Check include list (if mode requires it)
        if mode in (DomainFilterMode.INCLUDE, DomainFilterMode.BOTH):
            if self._include_patterns:
                for pattern in self._include_patterns:
                    if match_domain_pattern(domain, pattern):
                        return DomainMatchResult(
                            allowed=True,
                            matched_pattern=pattern,
                            reason=f"Domain allowed by include pattern: {pattern}",
                        )

                # Not in include list
                if self._config.log_filtered:
                    logger.info(
                        "DOMAIN_NOT_WHITELISTED",
                        extra={
                            "domain": domain,
                            "url": url[:100],
                        },
                    )
                return DomainMatchResult(allowed=False, reason="Domain not in include list")

        # If we get here, domain is allowed (exclude-only mode, not in blacklist)
        return DomainMatchResult(allowed=True, reason="Domain allowed")

    def filter_results(
        self, results: list[dict[str, Any]], url_key: str = "url"
    ) -> list[dict[str, Any]]:
        """Filter a list of search results by domain.

        Args:
            results: List of result dictionaries
            url_key: Key in result dict containing URL

        Returns:
            Filtered list with blocked domains removed
        """
        if not self.is_active:
            return results

        filtered = []
        for result in results:
            url = result.get(url_key, "")
            match_result = self.is_allowed(url)
            if match_result.allowed:
                filtered.append(result)
        return filtered
