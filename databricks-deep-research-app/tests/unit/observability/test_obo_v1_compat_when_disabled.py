"""V1 compat test: OBO refresh is a no-op when the feature flag is unset.

Requirement: when AGENT_DESIGNER_OBO_REFRESH_ENABLED is unset (or != "1"),
  - detect_expiring() always returns False
  - refresh() returns None without touching the upstream SDK
  - No metric emissions of any kind (RecordingSink stays empty)
"""
from __future__ import annotations

import base64
import json
import time
from unittest.mock import MagicMock

import pytest


def _make_jwt(exp: int) -> str:
    """Build a minimal fake JWT with the given 'exp' claim."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload_bytes = json.dumps({"exp": exp, "sub": "test"}).encode()
    payload = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
    return f"{header}.{payload}.fakesig"


@pytest.fixture(autouse=True)
def reset_obo_module_state() -> None:
    """Clear module-level caches between tests."""
    import deep_research.services.obo_client as obo_mod

    obo_mod._refresh_locks.clear()
    obo_mod._rotation_cache.clear()
    yield
    obo_mod._refresh_locks.clear()
    obo_mod._rotation_cache.clear()


async def test_no_refresh_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """detect_expiring returns False; refresh is a no-op; no metrics emitted."""
    # Ensure the flag is absent.
    monkeypatch.delenv("AGENT_DESIGNER_OBO_REFRESH_ENABLED", raising=False)

    # Token that looks expired (exp in the past).
    exp = int(time.time()) - 60
    expired_token = _make_jwt(exp)

    mock_wc = MagicMock()

    from deep_research.services.obo_client import detect_expiring, refresh
    from deep_research.storage.observability import RecordingSink, use_sink

    sink = RecordingSink()
    with use_sink(sink):
        is_expiring = detect_expiring(expired_token)
        result = await refresh(expired_token, mock_wc)

    # Feature off → detect_expiring always False.
    assert is_expiring is False

    # Feature off → refresh is a no-op.
    assert result is None

    # Upstream SDK never called.
    mock_wc.token_management.create_obo_token.assert_not_called()

    # Zero metric emissions.
    assert len(sink.emissions) == 0, (
        f"Expected no metric emissions when flag is off, got: {sink.emissions}"
    )
