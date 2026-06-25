"""Tests for per-run override of global flags from the chat Options pane (P2).

The chat Options pane lets a user override ``cross_session_memory_enabled`` and
``followup_live_search_enabled`` for a single run via ``OrchestrationConfig``.
These tests pin the override precedence on the cross-session-memory wrapper
(``None`` => inherit global; ``False`` => off even when global is on).
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
)

import pytest  # noqa: E402

from deep_research.agent.framework_orchestrator import (  # noqa: E402
    _maybe_prepend_cross_session_memory,
)
from deep_research.agent.orchestration_config import OrchestrationConfig  # noqa: E402

pytestmark = pytest.mark.unit


class _SettingsOn:
    cross_session_memory_enabled = True
    cross_session_memory_min_confidence = "medium"
    cross_session_memory_max_facts = 10
    cross_session_memory_timeout_sec = 5.0


class _SettingsOff:
    cross_session_memory_enabled = False
    cross_session_memory_min_confidence = "medium"
    cross_session_memory_max_facts = 10
    cross_session_memory_timeout_sec = 5.0


async def test_per_run_disable_overrides_global_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """enable_cross_session_memory=False disables memory even when global is ON."""
    import deep_research.core.config as cfgmod

    monkeypatch.setattr(cfgmod, "get_settings", lambda: _SettingsOn())
    history = [{"role": "user", "content": "hi"}]
    cfg = OrchestrationConfig(enable_cross_session_memory=False)
    out = await _maybe_prepend_cross_session_memory(
        history, cfg, db=None, user_id="u1", chat_id=None
    )
    assert out == history  # no injection: per-run override wins over global ON


async def test_none_inherits_global_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """enable_cross_session_memory=None inherits the (off) global flag."""
    import deep_research.core.config as cfgmod

    monkeypatch.setattr(cfgmod, "get_settings", lambda: _SettingsOff())
    history = [{"role": "user", "content": "hi"}]
    cfg = OrchestrationConfig()  # enable_cross_session_memory defaults to None
    out = await _maybe_prepend_cross_session_memory(
        history, cfg, db=None, user_id="u1", chat_id=None
    )
    assert out == history  # inherits global OFF


def test_orchestration_config_override_fields_default_none() -> None:
    """Both per-run override fields default to None (inherit global)."""
    cfg = OrchestrationConfig()
    assert cfg.enable_cross_session_memory is None
    assert cfg.allow_live_search is None
