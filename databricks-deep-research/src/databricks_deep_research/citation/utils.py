"""Shared leaf utilities for the citation verification pipeline.

This module has **zero** imports from other ``citation.*`` modules to
prevent import cycles.  It provides deduplication targets for helpers
that were previously copy-pasted across 6+ citation modules.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Text truncation (replaces 6 private _truncate copies)
# ---------------------------------------------------------------------------


def truncate(text: str | None, max_len: int = 200) -> str:
    """Truncate *text* to *max_len* characters with ellipsis.

    Handles ``None`` (returns ``"<none>"``) for callers that pass
    optional strings (e.g. numeric_verifier, citation_corrector).
    Default 200 is the most common default across consumers.
    """
    if text is None:
        return "<none>"
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Numeric / temporal detection (replaces 3 _has_numeric_content copies)
# ---------------------------------------------------------------------------

_NUMERIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\$[\d,.]+[BMK]?", re.IGNORECASE),
    re.compile(r"\d+(?:\.\d+)?%"),
    re.compile(r"\d+(?:,\d{3})+"),
    re.compile(r"\d+\s*(?:billion|million|thousand)", re.IGNORECASE),
    re.compile(r"[$\u20ac\u00a3]\s*\d"),
)

NUMERIC_PATTERN = re.compile(
    r"[$\u20ac\u00a3\u00a5]?\(?\d[\d,]*(?:\.\d+)?(?:\s*(?:%|million|billion|m|b|k|x))?\)?",
    re.IGNORECASE,
)

TEMPORAL_PATTERN = re.compile(
    r"\b(?:q[1-4]|20\d{2}|first quarter|second quarter|third quarter|"
    r"fourth quarter|full[- ]year|year[- ]to[- ]date)\b",
    re.IGNORECASE,
)


def has_numeric_content(text: str) -> bool:
    """Return ``True`` when *text* contains numbers or statistics."""
    return any(p.search(text) for p in _NUMERIC_PATTERNS)
