"""Tests for the Designer /capabilities endpoint (global feature-gate states)."""

from __future__ import annotations

import os

# Importing the api.v1 package constructs Settings() at import time; ensure a
# storage backend is satisfiable in an isolated test env (mirrors other api tests).
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
)

from typing import Any  # noqa: E402

from deep_research.api.v1.agent_designer import (  # noqa: E402
    CapabilitiesResponse,
    get_capabilities,
)


async def test_capabilities_reports_bool_gate_states() -> None:
    """The endpoint returns boolean gate states (defaults from app.yaml/Settings)."""
    resp = await get_capabilities()
    assert isinstance(resp, CapabilitiesResponse)
    assert isinstance(resp.skill_scripts_global, bool)
    assert isinstance(resp.cross_session_memory_global, bool)
    assert isinstance(resp.live_search_global, bool)


async def test_skill_scripts_gate_follows_app_config(monkeypatch: Any) -> None:
    """skill_scripts_global mirrors app_config.skills.allow_script_execution."""
    import deep_research.api.v1.agent_designer as mod

    class _Skills:
        allow_script_execution = True

    class _Cfg:
        skills = _Skills()

    monkeypatch.setattr(mod, "get_app_config", lambda: _Cfg())
    resp = await get_capabilities()
    assert resp.skill_scripts_global is True
