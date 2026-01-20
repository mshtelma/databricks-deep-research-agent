"""Human-readable citation key generation utilities.

This module generates citation keys from source metadata, replacing
numeric markers [0], [1], [2] with human-readable keys like [Arxiv], [Zhipu].

Key generation priority:
1. Domain-based for web sources: arxiv.org → "Arxiv"
2. Title abbreviation: "GLM-4.7 Technical Report" → "GLM47"
3. Fallback: "Source"

Collision handling: If key exists, append discriminator: Arxiv → Arxiv-2 → Arxiv-3
"""

import logging
import re
from urllib.parse import urlparse

from deep_research.services.citation.evidence_selector import RankedEvidence

logger = logging.getLogger(__name__)


def extract_domain_key(url: str) -> str:
    """Extract domain-based citation key from URL.

    Args:
        url: Source URL.

    Returns:
        Capitalized domain name without TLD.

    Examples:
        "https://arxiv.org/abs/123" → "Arxiv"
        "https://docs.databricks.com/x" → "Docs"
        "https://www.github.com/repo" → "Github"
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc

        # Handle empty netloc (invalid URL)
        if not domain:
            return "Web"

        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Take first part before TLD
        # "arxiv.org" → "arxiv", "docs.databricks.com" → "docs"
        base = domain.split(".")[0]

        # Handle empty base after split
        if not base:
            return "Web"

        # Capitalize for consistency
        return base.capitalize()
    except Exception:
        return "Web"


def abbreviate_title(title: str) -> str:
    """Abbreviate document title to a short citation key.

    Args:
        title: Document or source title.

    Returns:
        Abbreviated key combining first word and version number if present.

    Examples:
        "GLM-4.7 Technical Report" → "GLM47"
        "Qwen2 Model Card" → "Qwen2"
        "Deep Learning for NLP" → "Deep"
    """
    if not title:
        return "Doc"

    # Extract words and numbers
    words = re.findall(r"[A-Za-z]+|\d+\.?\d*", title)
    if not words:
        return "Doc"

    # Take first word (max 6 chars)
    first_word: str = words[0]
    key: str = first_word[:6]

    # Append version number if found (max 2 digits)
    for word in words[1:]:
        if re.match(r"^\d", word):
            # Remove decimal point and take first 2 chars
            version = word.replace(".", "")[:2]
            key = key + version
            break

    return key


def build_citation_key_map(evidence_pool: list[RankedEvidence]) -> dict[int, str]:
    """Build mapping from evidence index to human-readable citation key.

    Args:
        evidence_pool: List of ranked evidence spans with source metadata.

    Returns:
        Dictionary mapping evidence index to citation key.

    Example:
        evidence_pool = [
            RankedEvidence(source_url="arxiv.org/..."),
            RankedEvidence(source_url="arxiv.org/..."),
            RankedEvidence(source_url="github.com/..."),
        ]
        → {0: "Arxiv", 1: "Arxiv-2", 2: "Github"}
    """
    key_map: dict[int, str] = {}
    used_keys: set[str] = set()

    for idx, evidence in enumerate(evidence_pool):
        # Priority: domain > title > fallback
        if evidence.source_url:
            base_key = extract_domain_key(evidence.source_url)
        elif evidence.source_title:
            base_key = abbreviate_title(evidence.source_title)
        else:
            base_key = "Source"

        # Handle collisions with discriminator suffix
        key = base_key
        counter = 2
        while key in used_keys:
            key = f"{base_key}-{counter}"
            counter += 1

        key_map[idx] = key
        used_keys.add(key)

        logger.debug(
            "CITATION_KEY_GENERATED evidence_index=%d citation_key=%s source_url=%s source_title=%s",
            idx,
            key,
            evidence.source_url[:60] if evidence.source_url else None,
            evidence.source_title[:40] if evidence.source_title else None,
        )

    return key_map


def replace_numeric_markers(content: str, key_map: dict[int, str]) -> str:
    """Replace numeric citation markers with human-readable keys.

    Args:
        content: Text content with [0], [1], [2] markers.
        key_map: Mapping from index to citation key.

    Returns:
        Content with [Arxiv], [Zhipu] style markers.

    Example:
        content = "Claim one [0]. Claim two [1]."
        key_map = {0: "Arxiv", 1: "Zhipu"}
        → "Claim one [Arxiv]. Claim two [Zhipu]."
    """

    def replacer(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        key = key_map.get(idx)
        if key:
            return f"[{key}]"
        # Keep original if no mapping found
        return match.group(0)

    return re.sub(r"\[(\d+)\]", replacer, content)


def parse_citation_key(marker: str) -> str | None:
    """Extract citation key from a marker string.

    Args:
        marker: Citation marker like "[Arxiv]" or "[Zhipu-2]".

    Returns:
        The key without brackets, or None if invalid.

    Example:
        "[Arxiv]" → "Arxiv"
        "[Zhipu-2]" → "Zhipu-2"
        "[0]" → None (numeric markers not valid)
    """
    match = re.match(r"^\[([A-Za-z][A-Za-z0-9-]*(?:-\d+)?)\]$", marker)
    if match:
        return match.group(1)
    return None
