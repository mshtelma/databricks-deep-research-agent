"""Unit tests for the OBO token refresh service (US-601).

Tests cover:
- detect_expiring() threshold logic
- refresh() happy path and failure paths
- Per-token-hash mutex coalescing concurrent calls
- Rotation cache invalidation
- Metric emissions

SECURITY NOTE: raw tokens never appear in log lines — only hash prefixes.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers to build minimal fake JWTs (no signature needed for unit tests)
# ---------------------------------------------------------------------------


def _make_jwt(exp: int) -> str:
    """Build a minimal unsigned JWT with the given 'exp' claim."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload_bytes = json.dumps({"exp": exp, "sub": "test"}).encode()
    payload = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
    return f"{header}.{payload}.fakesig"


def _token_hash_prefix(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Module-level state cleanup between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_obo_module_state() -> None:
    """Clear module-level caches and locks between tests."""
    import deep_research.services.obo_client as obo_mod

    obo_mod._refresh_locks.clear()
    obo_mod._rotation_cache.clear()
    yield
    obo_mod._refresh_locks.clear()
    obo_mod._rotation_cache.clear()


# ---------------------------------------------------------------------------
# detect_expiring
# ---------------------------------------------------------------------------


class TestDetectExpiring:
    def test_detect_expiring_within_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Token expiring in <300 s is detected as True."""
        monkeypatch.setenv("AGENT_DESIGNER_OBO_REFRESH_ENABLED", "1")
        exp = int(time.time()) + 100  # expires in 100 s — within 300 s threshold
        token = _make_jwt(exp)

        from deep_research.services.obo_client import detect_expiring

        assert detect_expiring(token) is True

    def test_detect_expiring_outside_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Token expiring in >300 s is detected as False."""
        monkeypatch.setenv("AGENT_DESIGNER_OBO_REFRESH_ENABLED", "1")
        exp = int(time.time()) + 600  # expires in 600 s — outside threshold
        token = _make_jwt(exp)

        from deep_research.services.obo_client import detect_expiring

        assert detect_expiring(token) is False

    def test_detect_expiring_disabled_always_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When flag is off, detect_expiring always returns False."""
        monkeypatch.delenv("AGENT_DESIGNER_OBO_REFRESH_ENABLED", raising=False)
        exp = int(time.time()) + 10  # would normally be expiring
        token = _make_jwt(exp)

        from deep_research.services.obo_client import detect_expiring

        assert detect_expiring(token) is False


# ---------------------------------------------------------------------------
# refresh — happy path
# ---------------------------------------------------------------------------


class TestRefreshSuccess:
    async def test_refresh_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy-path: refresh returns new token, records success metric exactly once."""
        monkeypatch.setenv("AGENT_DESIGNER_OBO_REFRESH_ENABLED", "1")

        # Build a token that is NOT already expired (has future exp).
        exp = int(time.time()) + 60
        old_token = _make_jwt(exp)
        new_token_value = _make_jwt(int(time.time()) + 3600)

        # Mock the workspace client token exchange.
        mock_response = MagicMock()
        mock_response.token_value = new_token_value
        mock_wc = MagicMock()
        mock_wc.token_management.create_obo_token.return_value = mock_response

        from deep_research.services.obo_client import refresh
        from deep_research.storage.observability import RecordingSink, use_sink

        sink = RecordingSink()
        with use_sink(sink):
            result = await refresh(old_token, mock_wc)

        assert result == new_token_value
        # Exactly one success metric emission.
        assert sink.count("agent_designer.token_refresh_attempt", outcome="success") == 1.0
        assert sink.count("agent_designer.token_refresh_attempt", outcome="failure") == 0.0


# ---------------------------------------------------------------------------
# refresh — failure paths
# ---------------------------------------------------------------------------


class TestRefreshFailures:
    async def test_refresh_failure_network(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Network error increments record_token_refresh_failure(error_kind='network')."""
        monkeypatch.setenv("AGENT_DESIGNER_OBO_REFRESH_ENABLED", "1")

        exp = int(time.time()) + 60
        old_token = _make_jwt(exp)

        mock_wc = MagicMock()
        mock_wc.token_management.create_obo_token.side_effect = ConnectionError("timeout")

        from deep_research.services.obo_client import refresh
        from deep_research.storage.observability import RecordingSink, use_sink

        sink = RecordingSink()
        with use_sink(sink), pytest.raises(ConnectionError):
            await refresh(old_token, mock_wc)

        assert sink.count("agent_designer.token_refresh_failure", error_kind="network") == 1.0
        assert sink.count("agent_designer.token_refresh_attempt", outcome="failure") == 1.0

    async def test_refresh_failure_permission(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Permission error increments record_token_refresh_failure(error_kind='permission')."""
        monkeypatch.setenv("AGENT_DESIGNER_OBO_REFRESH_ENABLED", "1")

        exp = int(time.time()) + 60
        old_token = _make_jwt(exp)

        mock_wc = MagicMock()
        mock_wc.token_management.create_obo_token.side_effect = Exception("403 PERMISSION_DENIED")

        from deep_research.services.obo_client import refresh
        from deep_research.storage.observability import RecordingSink, use_sink

        sink = RecordingSink()
        with use_sink(sink), pytest.raises(Exception, match="403"):
            await refresh(old_token, mock_wc)

        assert sink.count("agent_designer.token_refresh_failure", error_kind="permission") == 1.0
        assert sink.count("agent_designer.token_refresh_attempt", outcome="failure") == 1.0


# ---------------------------------------------------------------------------
# Concurrent coalescing
# ---------------------------------------------------------------------------


class TestConcurrentRefreshCoalesces:
    async def test_concurrent_refresh_coalesces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """5 concurrent refresh() calls for the same token result in ONE upstream call."""
        monkeypatch.setenv("AGENT_DESIGNER_OBO_REFRESH_ENABLED", "1")

        exp = int(time.time()) + 60
        old_token = _make_jwt(exp)
        new_token_value = _make_jwt(int(time.time()) + 3600)

        call_count = 0

        def _create_obo_token(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.token_value = new_token_value
            return resp

        mock_wc = MagicMock()
        mock_wc.token_management.create_obo_token.side_effect = _create_obo_token

        from deep_research.services.obo_client import refresh

        results = await asyncio.gather(*[refresh(old_token, mock_wc) for _ in range(5)])

        # All callers must receive the new token.
        assert all(r == new_token_value for r in results)
        # Upstream called exactly once due to lock + cache coalescing.
        assert call_count == 1


# ---------------------------------------------------------------------------
# Hash uniqueness after rotation
# ---------------------------------------------------------------------------


class TestRefreshProducesNewHash:
    async def test_refresh_produces_new_sha256_hash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The new token's hash prefix must differ from the old one."""
        monkeypatch.setenv("AGENT_DESIGNER_OBO_REFRESH_ENABLED", "1")

        exp = int(time.time()) + 60
        old_token = _make_jwt(exp)
        # Make new token payload different enough to produce a distinct hash.
        new_token_value = _make_jwt(int(time.time()) + 9999) + "_unique"

        mock_response = MagicMock()
        mock_response.token_value = new_token_value
        mock_wc = MagicMock()
        mock_wc.token_management.create_obo_token.return_value = mock_response

        from deep_research.services.obo_client import refresh
        from deep_research.storage.observability import RecordingSink, use_sink

        sink = RecordingSink()
        with use_sink(sink):
            result = await refresh(old_token, mock_wc)

        assert result is not None
        assert _token_hash_prefix(old_token) != _token_hash_prefix(result)


# ---------------------------------------------------------------------------
# Rotation cache invalidation
# ---------------------------------------------------------------------------


class TestCacheInvalidationOnRotation:
    async def test_cache_invalidation_on_rotation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After rotation, get_rotated_token(old_token) returns the new token."""
        monkeypatch.setenv("AGENT_DESIGNER_OBO_REFRESH_ENABLED", "1")

        exp = int(time.time()) + 60
        old_token = _make_jwt(exp)
        new_token_value = _make_jwt(int(time.time()) + 3600)

        mock_response = MagicMock()
        mock_response.token_value = new_token_value
        mock_wc = MagicMock()
        mock_wc.token_management.create_obo_token.return_value = mock_response

        from deep_research.services.obo_client import get_rotated_token, refresh
        from deep_research.storage.observability import RecordingSink, use_sink

        sink = RecordingSink()
        with use_sink(sink):
            await refresh(old_token, mock_wc)

        # Rotation cache must map old token -> new token.
        rotated = get_rotated_token(old_token)
        assert rotated == new_token_value

        # A different (unrelated) token returns None.
        unrelated = _make_jwt(int(time.time()) + 100) + "_other"
        assert get_rotated_token(unrelated) is None
