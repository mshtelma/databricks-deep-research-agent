"""
Web content fetcher using trafilatura for clean text extraction.

This module provides high-quality web content extraction with caching,
error handling, and retry logic for robust research operations.
"""

import logging
import requests
import hashlib
import time
import asyncio
from typing import Optional, Dict, List, Tuple, Any
from urllib.parse import urlparse
from dataclasses import dataclass
from .types import FetchingConfig

logger = logging.getLogger(__name__)


@dataclass
class FetchAttempt:
    """Record of a fetch attempt."""
    url: str
    success: bool
    status_code: Optional[int]
    error: Optional[str]
    retry_count: int
    duration: float
    content_length: int = 0

# Try to import trafilatura
try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    logger.warning(
        "trafilatura not available. Web content fetching will be limited. "
        "Install with: pip install trafilatura"
    )


class WebContentFetcher:
    """
    Fetches and extracts clean text from web pages using trafilatura.

    Features:
    - Clean text extraction with table preservation
    - Caching to avoid re-fetching
    - Retry logic with exponential backoff
    - 403 error handling with user-agent rotation
    - Domain blocking after repeated failures
    - Content validation

    Args:
        config: FetchingConfig with timeout, retry, and cache settings
    """

    # Rotate user agents to avoid bot detection
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"
    ]

    def __init__(self, config: Optional[FetchingConfig] = None):
        """Initialize fetcher with configuration."""
        self.config = config or FetchingConfig()
        self._cache: Dict[str, str] = {}  # url_hash -> content
        self._cache_timestamps: Dict[str, float] = {}  # url_hash -> timestamp
        self._user_agent_index = 0
        self.fetch_history: List[FetchAttempt] = []
        self.blocked_domains: set = set()
        self.session = requests.Session()

        if not TRAFILATURA_AVAILABLE:
            logger.error(
                "⚠️ trafilatura not installed! Web content fetching will fail. "
                "Install with: pip install trafilatura"
            )

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with rotating user agent."""
        return {
            "User-Agent": self.USER_AGENTS[self._user_agent_index],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

    def _rotate_user_agent(self):
        """Rotate to next user agent."""
        self._user_agent_index = (self._user_agent_index + 1) % len(self.USER_AGENTS)

    def fetch_with_fallback(
        self,
        url: str,
        alternatives: Optional[List[str]] = None,
        timeout: Optional[int] = None
    ) -> Tuple[Optional[str], Optional[FetchAttempt]]:
        """
        Fetch with retry logic and alternative URLs.

        Args:
            url: Primary URL to fetch
            alternatives: List of alternative URLs to try if primary fails
            timeout: Optional override for timeout

        Returns:
            Tuple of (content, fetch_attempt) or (None, last_attempt)
        """
        urls_to_try = [url]
        if alternatives:
            urls_to_try.extend(alternatives)

        for target_url in urls_to_try:
            # Skip known blocked domains
            domain = urlparse(target_url).netloc
            if domain in self.blocked_domains:
                logger.debug(f"WEB_FETCH: Skipping blocked domain: {domain}")
                continue

            # Try fetching this URL
            content = self.fetch_content(target_url, timeout)

            # Get last attempt from history
            last_attempt = self.fetch_history[-1] if self.fetch_history else None

            if content:
                return content, last_attempt

            # Mark domain as blocked if consistent 403s
            if last_attempt and last_attempt.status_code == 403 and last_attempt.retry_count >= 2:
                self.blocked_domains.add(domain)
                logger.warning(f"WEB_FETCH: Domain blocked after repeated 403s: {domain}")

        # All attempts failed
        return None, self.fetch_history[-1] if self.fetch_history else None

    def fetch_content(
        self,
        url: str,
        timeout: Optional[int] = None
    ) -> Optional[str]:
        """
        Fetch and extract clean text content from URL.

        Args:
            url: URL to fetch
            timeout: Optional override for timeout (uses config default if not provided)

        Returns:
            Extracted text content, or None if fetch/extraction failed
        """

        if not TRAFILATURA_AVAILABLE:
            logger.error(f"Cannot fetch {url}: trafilatura not available")
            return None

        # Check cache first
        cache_key = self._get_cache_key(url)
        if self.config.cache_enabled and cache_key in self._cache:
            # Check if cache is still valid (TTL)
            cache_age = time.time() - self._cache_timestamps.get(cache_key, 0)
            if cache_age < self.config.cache_ttl_seconds:
                logger.info(f"WEB_FETCH: Cache hit for {url[:60]}... (age={cache_age:.0f}s)")
                return self._cache[cache_key]
            else:
                # Cache expired
                logger.debug(f"WEB_FETCH: Cache expired for {url[:60]}... (age={cache_age:.0f}s)")
                del self._cache[cache_key]
                del self._cache_timestamps[cache_key]

        logger.info(f"WEB_FETCH: Fetching {url[:80]}...")

        timeout_val = timeout or self.config.timeout_seconds
        start_time = time.time()

        for attempt in range(self.config.max_retries):
            try:
                # Add exponential backoff delay between retries
                if attempt > 0:
                    delay = min(2 ** attempt, 30)  # Max 30 seconds
                    logger.debug(f"WEB_FETCH: Retry #{attempt}, waiting {delay}s...")
                    time.sleep(delay)

                # Fetch HTML with rotating user agent
                response = self.session.get(
                    url,
                    timeout=timeout_val + (attempt * 5),  # Increase timeout on retries
                    headers=self._get_headers()
                )

                # Check for 403 specifically
                if response.status_code == 403:
                    logger.warning(
                        f"WEB_FETCH: 403 Forbidden (attempt {attempt+1}/{self.config.max_retries}): {url[:60]}"
                    )
                    # Rotate user agent for next attempt
                    self._rotate_user_agent()

                    # Record failed attempt
                    fetch_attempt = FetchAttempt(
                        url=url,
                        success=False,
                        status_code=403,
                        error="403 Forbidden",
                        retry_count=attempt,
                        duration=time.time() - start_time
                    )
                    self.fetch_history.append(fetch_attempt)

                    if attempt < self.config.max_retries - 1:
                        continue  # Try again with different user agent
                    else:
                        return None  # All retries exhausted

                response.raise_for_status()

                html = response.text
                logger.debug(f"WEB_FETCH: Downloaded {len(html)} bytes HTML")

                # Extract clean text using trafilatura
                extracted_text = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,  # CRITICAL: Keep tables for data extraction
                    include_links=False,
                    no_fallback=False,  # Use fallback extraction if needed
                )

                if not extracted_text:
                    logger.warning(f"WEB_FETCH: No content extracted from {url[:60]}")
                    return None

                # Validate content length
                if len(extracted_text) < self.config.min_content_length:
                    logger.warning(
                        f"WEB_FETCH: Content too short ({len(extracted_text)} chars) "
                        f"from {url[:60]}, min={self.config.min_content_length}"
                    )
                    return None

                # Truncate if too long
                if len(extracted_text) > self.config.max_content_length:
                    logger.info(
                        f"WEB_FETCH: Truncating content from {len(extracted_text)} "
                        f"to {self.config.max_content_length} chars"
                    )
                    extracted_text = extracted_text[:self.config.max_content_length]

                logger.info(
                    f"WEB_FETCH: ✅ Successfully extracted {len(extracted_text)} chars "
                    f"from {url[:60]}"
                )

                # Record successful attempt
                fetch_attempt = FetchAttempt(
                    url=url,
                    success=True,
                    status_code=response.status_code,
                    error=None,
                    retry_count=attempt,
                    duration=time.time() - start_time,
                    content_length=len(extracted_text)
                )
                self.fetch_history.append(fetch_attempt)

                # Cache result
                if self.config.cache_enabled:
                    self._cache[cache_key] = extracted_text
                    self._cache_timestamps[cache_key] = time.time()

                return extracted_text

            except requests.exceptions.Timeout:
                logger.warning(
                    f"WEB_FETCH: Timeout on attempt {attempt+1}/{self.config.max_retries} "
                    f"for {url[:60]}"
                )
                # Record timeout attempt
                fetch_attempt = FetchAttempt(
                    url=url,
                    success=False,
                    status_code=None,
                    error="Timeout",
                    retry_count=attempt,
                    duration=time.time() - start_time
                )
                self.fetch_history.append(fetch_attempt)

                if attempt < self.config.max_retries - 1:
                    continue

            except requests.exceptions.RequestException as e:
                status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
                logger.warning(
                    f"WEB_FETCH: Request failed on attempt {attempt+1}/{self.config.max_retries}: "
                    f"{type(e).__name__}: {e}"
                )
                # Record failed attempt
                fetch_attempt = FetchAttempt(
                    url=url,
                    success=False,
                    status_code=status_code,
                    error=f"{type(e).__name__}: {str(e)}",
                    retry_count=attempt,
                    duration=time.time() - start_time
                )
                self.fetch_history.append(fetch_attempt)

                if attempt < self.config.max_retries - 1:
                    continue

            except Exception as e:
                logger.error(
                    f"WEB_FETCH: Unexpected error extracting content: "
                    f"{type(e).__name__}: {e}"
                )
                # Record unexpected error
                fetch_attempt = FetchAttempt(
                    url=url,
                    success=False,
                    status_code=None,
                    error=f"Unexpected: {type(e).__name__}: {str(e)}",
                    retry_count=attempt,
                    duration=time.time() - start_time
                )
                self.fetch_history.append(fetch_attempt)
                break

        logger.error(f"WEB_FETCH: ❌ Failed to fetch content from {url[:60]}")
        return None

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def clear_cache(self):
        """Clear the content cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("WEB_FETCH: Cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total_chars = sum(len(content) for content in self._cache.values())
        return {
            "cached_urls": len(self._cache),
            "total_cached_chars": total_chars,
            "avg_content_length": total_chars // len(self._cache) if self._cache else 0
        }

    def get_fetch_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about fetch attempts.

        Returns:
            Dictionary with fetch statistics including success rates,
            error counts, blocked domains, and performance metrics
        """
        if not self.fetch_history:
            return {
                "total_attempts": 0,
                "successful_fetches": 0,
                "failed_fetches": 0,
                "success_rate": 0.0,
                "error_403_count": 0,
                "timeout_count": 0,
                "blocked_domains": [],
                "avg_retry_count": 0.0,
                "avg_duration_seconds": 0.0,
                "total_content_fetched_chars": 0
            }

        successful = [a for a in self.fetch_history if a.success]
        failed = [a for a in self.fetch_history if not a.success]
        error_403 = [a for a in failed if a.status_code == 403]
        timeouts = [a for a in failed if a.error and "Timeout" in a.error]

        total_retries = sum(a.retry_count for a in self.fetch_history)
        total_duration = sum(a.duration for a in self.fetch_history)
        total_content = sum(a.content_length for a in successful)

        return {
            "total_attempts": len(self.fetch_history),
            "successful_fetches": len(successful),
            "failed_fetches": len(failed),
            "success_rate": len(successful) / len(self.fetch_history) if self.fetch_history else 0.0,
            "error_403_count": len(error_403),
            "timeout_count": len(timeouts),
            "blocked_domains": list(self.blocked_domains),
            "avg_retry_count": total_retries / len(self.fetch_history) if self.fetch_history else 0.0,
            "avg_duration_seconds": round(total_duration / len(self.fetch_history), 2) if self.fetch_history else 0.0,
            "total_content_fetched_chars": total_content
        }
