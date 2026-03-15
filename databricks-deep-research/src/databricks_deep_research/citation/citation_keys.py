"""Human-readable citation key generation utilities.

Generates citation keys from source metadata, replacing numeric markers
``[0]``, ``[1]`` with human-readable keys like ``[Arxiv]``, ``[Github]``.

Key generation priority:
1. Domain-based for web sources: ``arxiv.org`` -> ``"Arxiv"``
2. Title abbreviation: ``"GLM-4.7 Technical Report"`` -> ``"GLM47"``
3. Fallback: ``"Source"``

Collision handling: append discriminator ``Arxiv`` -> ``Arxiv-2`` -> ``Arxiv-3``.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from databricks_deep_research.citation.types import RankedEvidence

logger = logging.getLogger(__name__)

# URL patterns identifying Databricks enterprise resources.
# Enterprise sources use title-based citation keys because their
# URL domains (workspace hostnames) are not meaningful to readers.
_ENTERPRISE_URL_PATTERNS = (
    "genie://",
    "vs://",
    "ka://",
    "enterprise://",
    "/sql/genie/spaces/",
    "/ml/endpoints/",
    "/explore/data/",
    "/compute/vector-search",
)


def _is_enterprise_url(url: str) -> bool:
    """Check if *url* points to a Databricks enterprise resource."""
    return any(pattern in url for pattern in _ENTERPRISE_URL_PATTERNS)


def extract_domain_key(url: str) -> str:
    """Extract a short citation key from a URL's domain.

    Examples::

        "https://arxiv.org/abs/123"   -> "Arxiv"
        "https://www.github.com/repo" -> "Github"
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        if not domain:
            return "Web"
        if domain.startswith("www."):
            domain = domain[4:]
        base = domain.split(".")[0]
        if not base:
            return "Web"
        return base.capitalize()
    except Exception:
        return "Web"


def abbreviate_title(title: str) -> str:
    """Abbreviate a document title to a short citation key.

    Examples::

        "GLM-4.7 Technical Report" -> "GLM47"
        "Qwen2 Model Card"        -> "Qwen2"
    """
    if not title:
        return "Doc"
    words = re.findall(r"[A-Za-z]+|\d+\.?\d*", title)
    if not words:
        return "Doc"
    key: str = words[0][:6]
    for word in words[1:]:
        if re.match(r"^\d", word):
            version = word.replace(".", "")[:2]
            key = key + version
            break
    return key


def build_citation_key_map(
    evidence_pool: list[RankedEvidence],
) -> dict[int, str]:
    """Build ``{evidence_index: human_readable_key}`` mapping.

    Uses domain-based keys for web sources and title-based keys for
    enterprise sources.  Collisions are resolved with ``-2``, ``-3``, etc.
    """
    key_map: dict[int, str] = {}
    used_keys: set[str] = set()

    for idx, evidence in enumerate(evidence_pool):
        if evidence.source_url and not _is_enterprise_url(evidence.source_url):
            base_key = extract_domain_key(evidence.source_url)
        elif evidence.source_title:
            base_key = abbreviate_title(evidence.source_title)
        else:
            base_key = "Source"

        key = base_key
        counter = 2
        while key in used_keys:
            key = f"{base_key}-{counter}"
            counter += 1

        key_map[idx] = key
        used_keys.add(key)

        logger.debug(
            "CITATION_KEY_GENERATED evidence_index=%d key=%s url=%s title=%s",
            idx,
            key,
            evidence.source_url[:60] if evidence.source_url else None,
            evidence.source_title[:40] if evidence.source_title else None,
        )

    return key_map


def replace_numeric_markers(content: str, key_map: dict[int, str]) -> str:
    """Replace ``[0]``, ``[1]`` markers with human-readable ``[Arxiv]`` keys."""

    def _replacer(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        key = key_map.get(idx)
        return f"[{key}]" if key else match.group(0)

    return re.sub(r"\[(\d+)\]", _replacer, content)


def parse_citation_key(marker: str) -> str | None:
    """Extract citation key from a marker string.

    Returns the key without brackets, or ``None`` if the marker is invalid
    (e.g. a numeric ``[0]`` marker).

    Examples::

        "[Arxiv]"   -> "Arxiv"
        "[Zhipu-2]" -> "Zhipu-2"
        "[0]"       -> None
    """
    match = re.match(r"^\[([A-Za-z][A-Za-z0-9-]*(?:-\d+)?)\]$", marker)
    if match:
        return match.group(1)
    return None
