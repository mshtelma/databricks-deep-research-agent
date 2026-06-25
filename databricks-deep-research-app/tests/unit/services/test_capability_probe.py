"""Unit tests for services/deployment/capability_probe.py (Section S2).

Covers:
- TTL expiry: cached entry is returned within TTL, stale entry is re-fetched.
- set/get/invalidate round-trips.
- probe_error results are NOT cached.
- _classify_probe_error triage matrix.
"""
from __future__ import annotations

import time

from deep_research.services.deployment.capability_probe import (
    CapabilityProbeCache,
    _classify_probe_error,
    get_default_cache,
)

# ---------------------------------------------------------------------------
# CapabilityProbeCache
# ---------------------------------------------------------------------------


class TestCapabilityProbeCache:
    def test_get_returns_none_when_empty(self) -> None:
        cache = CapabilityProbeCache(ttl_seconds=60.0)
        assert cache.get("u1", "host1") is None

    def test_set_then_get_within_ttl(self) -> None:
        cache = CapabilityProbeCache(ttl_seconds=60.0)
        cache.set("u1", "host1", ok=True)
        result = cache.get("u1", "host1")
        assert result is not None
        assert result.ok is True
        assert result.reason is None

    def test_set_failure_then_get(self) -> None:
        cache = CapabilityProbeCache(ttl_seconds=60.0)
        cache.set("u2", "host2", ok=False, reason="missing_workspace_permission")
        result = cache.get("u2", "host2")
        assert result is not None
        assert result.ok is False
        assert result.reason == "missing_workspace_permission"

    def test_expired_entry_returns_none(self) -> None:
        cache = CapabilityProbeCache(ttl_seconds=0.0)  # expires immediately
        cache.set("u3", "host3", ok=True)
        # Sleep a tiny bit to ensure monotonic time has advanced past TTL=0.
        time.sleep(0.01)
        assert cache.get("u3", "host3") is None

    def test_probe_error_not_cached(self) -> None:
        """probe_error results must NOT be written to the cache."""
        cache = CapabilityProbeCache(ttl_seconds=60.0)
        cache.set("u4", "host4", ok=False, reason="probe_error")
        assert cache.get("u4", "host4") is None

    def test_invalidate_removes_entry(self) -> None:
        cache = CapabilityProbeCache(ttl_seconds=60.0)
        cache.set("u5", "host5", ok=True)
        cache.invalidate("u5", "host5")
        assert cache.get("u5", "host5") is None

    def test_invalidate_missing_key_is_noop(self) -> None:
        cache = CapabilityProbeCache(ttl_seconds=60.0)
        # Should not raise.
        cache.invalidate("nonexistent", "nonexistent-host")

    def test_clear_removes_all(self) -> None:
        cache = CapabilityProbeCache(ttl_seconds=60.0)
        cache.set("u1", "h1", ok=True)
        cache.set("u2", "h2", ok=True)
        cache.clear()
        assert cache.get("u1", "h1") is None
        assert cache.get("u2", "h2") is None

    def test_different_hosts_are_independent(self) -> None:
        cache = CapabilityProbeCache(ttl_seconds=60.0)
        cache.set("u1", "host-a", ok=True)
        cache.set("u1", "host-b", ok=False, reason="missing_workspace_permission")
        a = cache.get("u1", "host-a")
        b = cache.get("u1", "host-b")
        assert a is not None and a.ok is True
        assert b is not None and b.ok is False

    def test_different_users_are_independent(self) -> None:
        cache = CapabilityProbeCache(ttl_seconds=60.0)
        cache.set("alice", "host", ok=True)
        cache.set("bob", "host", ok=False, reason="missing_workspace_permission")
        alice = cache.get("alice", "host")
        bob = cache.get("bob", "host")
        assert alice is not None and alice.ok is True
        assert bob is not None and bob.ok is False


# ---------------------------------------------------------------------------
# _classify_probe_error
# ---------------------------------------------------------------------------


class TestClassifyProbeError:
    def _exc(self, name: str, msg: str = "") -> Exception:
        """Create an exception with a custom class name."""
        cls = type(name, (Exception,), {})
        return cls(msg)

    def test_permission_denied_by_class_name(self) -> None:
        exc = self._exc("PermissionDenied", "you lack permission")
        ok, reason = _classify_probe_error(exc)
        assert ok is False
        assert reason == "missing_workspace_permission"

    def test_forbidden_by_class_name(self) -> None:
        exc = self._exc("ForbiddenError", "forbidden")
        ok, reason = _classify_probe_error(exc)
        assert ok is False
        assert reason == "missing_workspace_permission"

    def test_403_in_message(self) -> None:
        exc = self._exc("SomeError", "HTTP 403 access denied")
        ok, reason = _classify_probe_error(exc)
        assert ok is False
        assert reason == "missing_workspace_permission"

    def test_not_found_is_probe_error(self) -> None:
        exc = self._exc("ResourceNotFoundException", "resource not found")
        ok, reason = _classify_probe_error(exc)
        assert ok is False
        assert reason == "probe_error"

    def test_connection_error_is_probe_error(self) -> None:
        exc = ConnectionError("connection refused")
        ok, reason = _classify_probe_error(exc)
        assert ok is False
        assert reason == "probe_error"

    def test_generic_exception_is_probe_error(self) -> None:
        exc = RuntimeError("something unexpected happened")
        ok, reason = _classify_probe_error(exc)
        assert ok is False
        assert reason == "probe_error"

    def test_permission_in_message_lowercase(self) -> None:
        exc = self._exc("ApiError", "permission denied for this resource")
        ok, reason = _classify_probe_error(exc)
        assert ok is False
        assert reason == "missing_workspace_permission"


# ---------------------------------------------------------------------------
# get_default_cache singleton
# ---------------------------------------------------------------------------


class TestGetDefaultCache:
    def test_returns_same_instance(self) -> None:
        c1 = get_default_cache()
        c2 = get_default_cache()
        assert c1 is c2

    def test_is_capability_probe_cache(self) -> None:
        assert isinstance(get_default_cache(), CapabilityProbeCache)
