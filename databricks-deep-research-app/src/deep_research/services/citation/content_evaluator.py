"""Content Quality Evaluator - Filters low-quality sources before citation.

Evaluates crawled content to identify:
- Abstract-only pages (no full text)
- Paywalled content
- Navigation/menu text
- Empty or minimal content

High-quality sources have:
- Specific facts and quotes
- Numeric data
- Sufficient word count
"""

import re
from dataclasses import dataclass

from deep_research.core.logging_utils import get_logger, truncate

logger = get_logger(__name__)


# Paywall indicators - phrases that suggest content is behind a paywall
PAYWALL_INDICATORS = [
    "subscribe to read",
    "sign in to continue",
    "create an account",
    "login required",
    "premium content",
    "members only",
    "subscription required",
    "please log in",
    "access denied",
    "register to view",
    "unlock this article",
    "start your free trial",
    "become a member",
]

# Abstract-only indicators - phrases suggesting only abstract is available
ABSTRACT_INDICATORS = [
    "abstract only",
    "full text not available",
    "access the full article",
    "view full text",
    "read the full paper",
    "download the full pdf",
    "click to view full text",
    "to read the full-text",
    "available in pdf",
    "purchase this article",
]

# Navigation/boilerplate indicators - common website chrome
NAVIGATION_INDICATORS = [
    "skip to main content",
    "javascript is required",
    "cookies policy",
    "privacy policy",
    "terms of service",
    "about us",
    "contact us",
    "advertisement",
    "sponsored content",
    "related articles",
    "you might also like",
    "share this article",
    "print this page",
    "email this article",
]


@dataclass
class ContentQuality:
    """Result of content quality evaluation."""

    score: float  # 0.0 - 1.0, higher is better
    has_specific_facts: bool
    has_numeric_data: bool
    is_abstract_only: bool
    is_paywall: bool
    is_navigation_heavy: bool
    word_count: int
    reason: str  # Human-readable explanation


def evaluate_content_quality(content: str, query: str) -> ContentQuality:
    """Evaluate if crawled content is useful for citation.

    Uses heuristics to detect:
    - Paywall pages
    - Abstract-only academic content
    - Navigation-heavy pages
    - Pages with specific facts and numbers

    Args:
        content: The crawled page content.
        query: The research query (for relevance context).

    Returns:
        ContentQuality with score and flags.
    """
    if not content or len(content.strip()) < 100:
        return ContentQuality(
            score=0.0,
            has_specific_facts=False,
            has_numeric_data=False,
            is_abstract_only=False,
            is_paywall=False,
            is_navigation_heavy=False,
            word_count=0,
            reason="Empty or minimal content",
        )

    content_lower = content.lower()
    words = content.split()
    word_count = len(words)

    # Check for paywall
    paywall_matches = sum(1 for p in PAYWALL_INDICATORS if p in content_lower)
    is_paywall = paywall_matches >= 2

    # Check for abstract-only
    abstract_matches = sum(1 for p in ABSTRACT_INDICATORS if p in content_lower)
    is_abstract_only = abstract_matches >= 2

    # Check for navigation-heavy content
    nav_matches = sum(1 for p in NAVIGATION_INDICATORS if p in content_lower)
    # If more than 20% of content seems to be navigation
    is_navigation_heavy = nav_matches >= 4 or (word_count < 300 and nav_matches >= 2)

    # Check for numeric data (specific numbers, percentages, currencies)
    numeric_patterns = [
        r"\$[\d,]+(?:\.\d+)?",  # Currency
        r"\d+(?:\.\d+)?%",  # Percentages
        r"\b\d{4}\b",  # Years
        r"\b\d+(?:,\d{3})+(?:\.\d+)?\b",  # Large numbers with commas
        r"\b\d+\.\d+\b",  # Decimal numbers
        r"(?:million|billion|trillion|thousand)\s+(?:dollars|euros|pounds)?",  # Written numbers
    ]
    numeric_matches = sum(len(re.findall(p, content, re.I)) for p in numeric_patterns)
    has_numeric_data = numeric_matches >= 3

    # Check for specific facts (quotes, specific statements)
    # Look for patterns that indicate specific information
    fact_patterns = [
        r'"[^"]{20,200}"',  # Quoted statements
        r"according to\s+[\w\s]+,",  # Attributed statements
        r"(?:reported|announced|stated|said)\s+that",  # Reported statements
        r"the\s+(?:study|research|report|survey)\s+(?:found|shows|indicates)",  # Research findings
    ]
    fact_matches = sum(len(re.findall(p, content, re.I)) for p in fact_patterns)
    has_specific_facts = fact_matches >= 2 or word_count > 500

    # Calculate quality score
    score = 0.5  # Base score

    # Penalties
    if is_paywall:
        score -= 0.4
    if is_abstract_only:
        score -= 0.3
    if is_navigation_heavy:
        score -= 0.2
    if word_count < 200:
        score -= 0.2
    elif word_count < 100:
        score -= 0.4

    # Bonuses
    if has_numeric_data:
        score += 0.2
    if has_specific_facts:
        score += 0.15
    if word_count > 500:
        score += 0.1
    if word_count > 1000:
        score += 0.1

    # Clamp score to [0, 1]
    score = max(0.0, min(1.0, score))

    # Determine reason
    if is_paywall:
        reason = "Paywall detected"
    elif is_abstract_only:
        reason = "Abstract only, no full text"
    elif is_navigation_heavy:
        reason = "Too much navigation content"
    elif word_count < 200:
        reason = f"Insufficient content ({word_count} words)"
    elif score >= 0.7:
        reason = "High quality content"
    elif score >= 0.5:
        reason = "Acceptable quality"
    else:
        reason = "Low quality content"

    result = ContentQuality(
        score=score,
        has_specific_facts=has_specific_facts,
        has_numeric_data=has_numeric_data,
        is_abstract_only=is_abstract_only,
        is_paywall=is_paywall,
        is_navigation_heavy=is_navigation_heavy,
        word_count=word_count,
        reason=reason,
    )

    logger.debug(
        "CONTENT_QUALITY_EVALUATED",
        score=round(score, 2),
        word_count=word_count,
        has_numeric=has_numeric_data,
        has_facts=has_specific_facts,
        is_paywall=is_paywall,
        is_abstract=is_abstract_only,
        content_preview=truncate(content, 50),
    )

    return result


def filter_high_quality_sources(
    sources: list[dict[str, str]],
    min_score: float = 0.5,
    query: str = "",
) -> list[dict[str, str]]:
    """Filter sources to keep only high-quality content.

    Args:
        sources: List of source dicts with 'url', 'title', 'content' keys.
        min_score: Minimum quality score (0.0-1.0) to keep.
        query: Research query for relevance context.

    Returns:
        Filtered list of high-quality sources.
    """
    high_quality = []
    for source in sources:
        content = source.get("content", "")
        if not content:
            continue

        quality = evaluate_content_quality(content, query)

        if quality.score >= min_score and not quality.is_abstract_only:
            high_quality.append(source)
            logger.debug(
                "SOURCE_ACCEPTED",
                url=source.get("url", "")[:50],
                score=round(quality.score, 2),
                reason=quality.reason,
            )
        else:
            logger.debug(
                "SOURCE_REJECTED",
                url=source.get("url", "")[:50],
                score=round(quality.score, 2),
                reason=quality.reason,
            )

    logger.info(
        "CONTENT_QUALITY_FILTER",
        total_sources=len(sources),
        high_quality_count=len(high_quality),
        filtered_out=len(sources) - len(high_quality),
    )

    return high_quality
