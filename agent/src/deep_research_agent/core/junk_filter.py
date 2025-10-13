"""
Simple junk filter for search results - filters OBVIOUS junk only.

This module provides conservative pre-fetch filtering to remove URLs that are
clearly worthless (social media, paywalls, error pages, etc.).

Philosophy: Better to fetch and validate than to miss good content.
Only filter OBVIOUS junk - when in doubt, fetch it.
"""

import re
import logging
from typing import Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class JunkFilter:
    """
    Filters OBVIOUS junk URLs before fetching content.

    Conservative approach: Only removes clear junk (social media, error pages, etc.).
    When uncertain, allows the URL through for fetching and post-fetch validation.
    """

    # Social media and user-generated content domains (rarely useful for research)
    JUNK_DOMAINS = {
        'twitter.com', 'x.com',
        'facebook.com', 'fb.com',
        'instagram.com',
        'pinterest.com', 'pin.it',
        'reddit.com/user',
        'linkedin.com/posts',
        'tiktok.com',
        'snapchat.com',
        'youtube.com/shorts',  # Shorts are too brief
        'amazon.com/gp/product',  # Product pages
        'amazon.com/dp/',
        'ebay.com/itm',
        'aliexpress.com/item',
        'etsy.com/listing',
    }

    # URL patterns that indicate junk content
    JUNK_PATTERNS = [
        # Error pages
        r'/404\.', r'/error\.', r'/not-found', r'/page-not-found',
        r'/oops\.', r'/whoops\.', r'/missing\.', r'/gone\.',

        # Authentication/paywalls
        r'/login\b', r'/signin\b', r'/sign-in\b', r'/log-in\b',
        r'/register\b', r'/signup\b', r'/sign-up\b',
        r'/paywall\b', r'/subscribe\b', r'/subscription\b',
        r'/member-only\b', r'/premium-content\b',
        r'/checkout\b', r'/cart\b', r'/basket\b',

        # Commercial/transactional
        r'/buy-now\b', r'/purchase\b', r'/pricing\b', r'/plans\b',
        r'/add-to-cart\b', r'/order\b', r'/payment\b',

        # Navigation/utility pages
        r'/search\?', r'/search/', r'/results\?',
        r'/sitemap\.', r'/archive\.', r'/tag/',
        r'/category/', r'/author/', r'/user/',

        # Media fragments (not full articles)
        r'/gallery/', r'/photo/', r'/image/', r'/video/',
        r'/slideshow/', r'/photos/',

        # Temporary/dynamic content
        r'/tmp/', r'/temp/', r'/cache/',
        r'/session/', r'/download\.php',
    ]

    # Title patterns that indicate junk
    JUNK_TITLE_PATTERNS = [
        r'^404\b', r'^error\b', r'^page not found',
        r'^access denied\b', r'^forbidden\b',
        r'^cookie (consent|policy|notice)',
        r'^privacy (policy|notice|settings)',
        r'^terms (of service|and conditions)',
        r'^subscribe to', r'^sign up', r'^log in',
        r'^buy now\b', r'^shop now\b', r'^add to cart',
    ]

    def __init__(self):
        """Initialize junk filter with compiled patterns."""
        self.url_patterns = [re.compile(p, re.I) for p in self.JUNK_PATTERNS]
        self.title_patterns = [re.compile(p, re.I) for p in self.JUNK_TITLE_PATTERNS]

    def is_junk(self, url: str, title: str = "") -> Tuple[bool, str]:
        """
        Check if URL/title represents OBVIOUS junk content.

        Args:
            url: URL to check
            title: Optional title/snippet to check

        Returns:
            Tuple of (is_junk: bool, reason: str)
            - is_junk: True if this is obvious junk, False otherwise
            - reason: Explanation of why it's junk (empty if not junk)
        """
        if not url:
            return (True, "Empty URL")

        # Parse URL components
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
        except Exception as e:
            logger.warning(f"Failed to parse URL {url}: {e}")
            return (False, "")  # When uncertain, allow it through

        # Check domain blacklist
        for junk_domain in self.JUNK_DOMAINS:
            if junk_domain in domain:
                return (True, f"Junk domain: {junk_domain}")

        # Check URL patterns
        full_url = url.lower()
        for pattern in self.url_patterns:
            if pattern.search(full_url):
                return (True, f"Junk URL pattern: {pattern.pattern}")

        # Check title patterns (if title provided)
        if title:
            title_lower = title.lower()
            for pattern in self.title_patterns:
                if pattern.search(title_lower):
                    return (True, f"Junk title pattern: {pattern.pattern}")

        # If we get here, it's not obvious junk
        return (False, "")

    def filter_results(self, results: list, get_url_fn=None, get_title_fn=None) -> Tuple[list, list]:
        """
        Filter a list of results, removing obvious junk.

        Args:
            results: List of result objects/dicts
            get_url_fn: Optional function to extract URL from result (default: r.url or r['url'])
            get_title_fn: Optional function to extract title from result (default: r.title or r['title'])

        Returns:
            Tuple of (non_junk_results, junk_results) for logging/stats
        """
        non_junk = []
        junk = []

        for result in results:
            # Extract URL
            if get_url_fn:
                url = get_url_fn(result)
            elif isinstance(result, dict):
                url = result.get('url', '')
            else:
                url = getattr(result, 'url', '')

            # Extract title
            if get_title_fn:
                title = get_title_fn(result)
            elif isinstance(result, dict):
                title = result.get('title', '')
            else:
                title = getattr(result, 'title', '')

            # Check if junk
            is_junk, reason = self.is_junk(url, title)

            if is_junk:
                junk.append((result, reason))
                logger.debug(f"⏭️ FILTERED: {reason} - {url[:60]}")
            else:
                non_junk.append(result)

        if junk:
            logger.info(f"🗑️ Filtered {len(junk)}/{len(results)} obvious junk URLs")

        return non_junk, junk
