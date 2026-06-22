"""Pure helpers for source identity: URL canonicalization + content hashing.

Cross-turn source dedup and citation integrity both need a stable identity for
a source. ``canonicalize_url`` collapses cosmetic variants (case, default port,
tracking params, fragment, trailing slash) to one form; ``content_sha256``
hashes fetched content so later turns can detect drift.

No I/O. Pure functions only.
"""

from __future__ import annotations

import hashlib
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_PREFIXES: tuple[str, ...] = ("utm_",)
_TRACKING_KEYS: frozenset[str] = frozenset(
    {"fbclid", "gclid", "ref", "ref_src", "mc_cid", "mc_eid"}
)
_DEFAULT_PORTS: dict[str, str] = {"http": "80", "https": "443"}


def canonicalize_url(url: str) -> str:
    """Return a stable canonical form of *url* for dedup.

    Lowercases scheme + host, strips the default port, removes tracking query
    params (``utm_*``, ``fbclid``, ...), drops the fragment, and removes a
    trailing slash on the path (except a bare root path).
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    netloc = host
    if parts.port is not None and str(parts.port) != _DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{parts.port}"
    path = parts.path.rstrip("/") or parts.path
    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith(_TRACKING_PREFIXES)
        and key.lower() not in _TRACKING_KEYS
    ]
    return urlunsplit((scheme, netloc, path, urlencode(kept), ""))


def content_sha256(text: str | None) -> str | None:
    """SHA-256 hex digest of *text* for content-drift detection; None-safe."""
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
