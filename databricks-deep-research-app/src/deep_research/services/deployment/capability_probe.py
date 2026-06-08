"""TTL'd capability-probe cache for the Deploy Here feature (Section S2).

Encapsulates the ``(user_id, host) → ProbeResult`` cache so that the
``deploy_here_action`` endpoint and the new ``can_deploy_here`` endpoint
share a single authoritative cache rather than each maintaining their own
module-level dict.

Only confirmed permission failures and successes are cached.  Transient
network errors (``probe_error``) are *not* cached so the next call retries.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal

from deep_research.core.config import get_settings

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

ProbeReason = Literal[
    "missing_workspace_permission",
    "probe_error",
    "missing_obo_token",
]


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    reason: ProbeReason | None  # None when ok=True
    checked_at: float  # time.monotonic()


# ---------------------------------------------------------------------------
# Cache implementation
# ---------------------------------------------------------------------------


class CapabilityProbeCache:
    """Thread-safe TTL cache for workspace capability probes.

    Keyed on ``(user_id, host)`` so the same user in two different workspaces
    is two independent entries and a bad-permission workspace never poisons
    another.

    TTL is supplied at construction time.  The module-level singleton reads it
    from ``settings.deploy_here_probe_ttl_seconds`` at first import; tests can
    inject a short TTL via ``CapabilityProbeCache(ttl_seconds=0)``.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[tuple[str, str], ProbeResult] = {}
        self._lock = threading.Lock()

    def get(self, user_id: str, host: str) -> ProbeResult | None:
        """Return a cached result if still within TTL, otherwise None.

        Reads are done without the lock — dict.get is atomic in CPython and
        the worst case is a stale read that triggers a fresh probe.
        """
        entry = self._cache.get((user_id, host))
        if entry is None:
            return None
        if time.monotonic() - entry.checked_at > self._ttl:
            return None
        return entry

    def set(
        self,
        user_id: str,
        host: str,
        *,
        ok: bool,
        reason: ProbeReason | None = None,
    ) -> None:
        """Cache a probe result.

        Only ``ok=True`` and ``reason='missing_workspace_permission'`` entries
        are persisted; ``probe_error`` results are intentionally skipped so
        transient failures auto-retry on the next request.
        """
        # Do not cache transient errors — caller still stores a result for the
        # *response*, but we skip the cache write so the next request retries.
        if not ok and reason == "probe_error":
            return
        result = ProbeResult(ok=ok, reason=reason, checked_at=time.monotonic())
        with self._lock:
            self._cache[(user_id, host)] = result

    def invalidate(self, user_id: str, host: str) -> None:
        """Remove a single entry from the cache (used by the refresh endpoint)."""
        with self._lock:
            self._cache.pop((user_id, host), None)

    def clear(self) -> None:
        """Remove all entries — test helper."""
        with self._lock:
            self._cache.clear()


# ---------------------------------------------------------------------------
# Error classifier (shared between cache users)
# ---------------------------------------------------------------------------


def _classify_probe_error(exc: Exception) -> tuple[bool, ProbeReason | None]:
    """Map an exception caught during a deploy capability probe to ``(ok, reason)``.

    Returns ``(False, reason)`` where ``reason`` is one of the ``ProbeReason``
    values, or ``(False, "probe_error")`` for everything else.  Never raises.
    """
    cls_name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if (
        "permissiondenied" in cls_name
        or "forbidden" in cls_name
        or "unauthorized" in cls_name
        or "permission" in msg
        or "forbidden" in msg
        or "403" in msg
    ):
        return False, "missing_workspace_permission"
    return False, "probe_error"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_cache: CapabilityProbeCache | None = None
_default_cache_lock = threading.Lock()


def get_default_cache() -> CapabilityProbeCache:
    """Return the module-level singleton, creating it on first call."""
    global _default_cache  # noqa: PLW0603
    if _default_cache is None:
        with _default_cache_lock:
            if _default_cache is None:
                ttl = get_settings().deploy_here_probe_ttl_seconds
                _default_cache = CapabilityProbeCache(ttl_seconds=ttl)
    return _default_cache
