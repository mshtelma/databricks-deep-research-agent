"""Unit tests for source-identity helpers (URL canonicalization + content hash).

These back the cross-turn dedup + citation-integrity invariants of the unified
chat-memory write path: two URLs that differ only by tracking params / case /
default port / trailing slash must collapse to ONE canonical identity, and a
source's content must hash stably for drift detection.
"""

from __future__ import annotations

import pytest

from deep_research.agent.url_canonical import canonicalize_url, content_sha256

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://Example.com/Path/", "https://example.com/Path"),
        ("https://example.com/p?utm_source=x&id=5&fbclid=z", "https://example.com/p?id=5"),
        ("http://example.com:80/a", "http://example.com/a"),
        ("https://example.com:443/a#frag", "https://example.com/a"),
        ("https://example.com", "https://example.com"),
        ("https://example.com/", "https://example.com/"),
    ],
)
def test_canonicalize_url(raw: str, expected: str) -> None:
    assert canonicalize_url(raw) == expected


def test_canonicalize_url_collapses_tracking_variants() -> None:
    a = canonicalize_url("https://x.com/a?utm_source=google&utm_medium=cpc")
    b = canonicalize_url("https://x.com/a")
    assert a == b == "https://x.com/a"


def test_content_sha256_stable_and_null_safe() -> None:
    assert content_sha256(None) is None
    assert content_sha256("abc") == content_sha256("abc")
    assert content_sha256("abc") != content_sha256("abd")
    assert content_sha256("") is not None  # empty string still hashes
