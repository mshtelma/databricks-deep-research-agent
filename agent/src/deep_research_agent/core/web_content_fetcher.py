"""
Web content fetcher using trafilatura for clean text extraction.

This module provides high-quality web content extraction with caching,
error handling, and retry logic for robust research operations.
"""

import logging
import requests
import hashlib
import time
from typing import Optional, Dict
from .types import FetchingConfig

logger = logging.getLogger(__name__)

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
    - Retry logic and timeout handling
    - Content validation

    Args:
        config: FetchingConfig with timeout, retry, and cache settings
    """

    def __init__(self, config: Optional[FetchingConfig] = None):
        """Initialize fetcher with configuration."""
        self.config = config or FetchingConfig()
        self._cache: Dict[str, str] = {}  # url_hash -> content
        self._cache_timestamps: Dict[str, float] = {}  # url_hash -> timestamp

        if not TRAFILATURA_AVAILABLE:
            logger.error(
                "⚠️ trafilatura not installed! Web content fetching will fail. "
                "Install with: pip install trafilatura"
            )

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

        for attempt in range(self.config.max_retries):
            try:
                # Fetch HTML
                response = requests.get(
                    url,
                    timeout=timeout_val,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Research Bot; +https://databricks.com)"
                    }
                )
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
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)  # Brief delay before retry
                    continue

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"WEB_FETCH: Request failed on attempt {attempt+1}/{self.config.max_retries}: "
                    f"{type(e).__name__}: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)
                    continue

            except Exception as e:
                logger.error(
                    f"WEB_FETCH: Unexpected error extracting content: "
                    f"{type(e).__name__}: {e}"
                )
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
