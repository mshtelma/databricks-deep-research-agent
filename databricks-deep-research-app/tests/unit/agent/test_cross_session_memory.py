"""Unit tests for the app-side cross-session memory READ path.

Covers the fail-soft + hard-timeout wrapper (``inject_cross_session_memory``):
a stubbed store's facts are injected as a spotlighted role=user message; a
backend that raises or is slow degrades to no-memory (no exception propagates);
the confidence threshold + max-cap are honored; and the empty/no-store path
returns ``None`` (default behavior byte-identical).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from databricks_deep_research.memory import ConfidenceLabel, CrossSessionFact

from deep_research.agent.cross_session_memory import inject_cross_session_memory

pytestmark = pytest.mark.unit


class _StubStore:
    """Returns a fixed fact list, recording the keying args it was called with."""

    def __init__(self, facts: list[CrossSessionFact]) -> None:
        self._facts = facts
        self.calls: list[dict[str, object]] = []

    async def read_facts(
        self,
        *,
        user_id: str,
        agent_id: str | None,
        exclude_chat_id: object = None,
        min_confidence: ConfidenceLabel = "medium",
        limit: int = 20,
    ) -> list[CrossSessionFact]:
        self.calls.append(
            {
                "user_id": user_id,
                "agent_id": agent_id,
                "exclude_chat_id": exclude_chat_id,
                "min_confidence": min_confidence,
                "limit": limit,
            }
        )
        return list(self._facts)


class _RaisingStore:
    async def read_facts(self, **_kwargs: object) -> list[CrossSessionFact]:
        raise RuntimeError("backend exploded")


class _SlowStore:
    def __init__(self, delay: float) -> None:
        self._delay = delay

    async def read_facts(self, **_kwargs: object) -> list[CrossSessionFact]:
        await asyncio.sleep(self._delay)
        return [CrossSessionFact(content="too late", confidence="high")]


def _fact(content: str, confidence: str = "high", days_ago: int = 0) -> CrossSessionFact:
    return CrossSessionFact(
        content=content,
        confidence=confidence,  # type: ignore[arg-type]
        updated_at=datetime.now(UTC) - timedelta(days=days_ago),
    )


class TestInjects:
    async def test_injects_facts_as_spotlighted_user_message(self) -> None:
        store = _StubStore([_fact("user prefers tables over prose", "high")])
        msg = await inject_cross_session_memory(
            store=store, user_id="u1", agent_id="a1"
        )
        assert msg is not None
        assert msg["role"] == "user"
        assert "<attached_context" in msg["content"]  # spotlighted (untrusted)
        assert "user prefers tables over prose" in msg["content"]

    async def test_keying_args_passed_through(self) -> None:
        store = _StubStore([_fact("x", "high")])
        chat_id = uuid4()
        await inject_cross_session_memory(
            store=store, user_id="u1", agent_id="a1", exclude_chat_id=chat_id
        )
        assert store.calls[0]["user_id"] == "u1"
        assert store.calls[0]["agent_id"] == "a1"
        assert store.calls[0]["exclude_chat_id"] == chat_id


class TestFailSoft:
    async def test_raising_store_degrades_to_none(self) -> None:
        # The #1 invariant: a backend error NEVER propagates — request succeeds
        # with no memory.
        msg = await inject_cross_session_memory(
            store=_RaisingStore(), user_id="u1", agent_id="a1"
        )
        assert msg is None

    async def test_slow_read_is_bounded_and_degrades(self) -> None:
        msg = await inject_cross_session_memory(
            store=_SlowStore(delay=0.5),
            user_id="u1",
            agent_id="a1",
            timeout_seconds=0.05,
        )
        assert msg is None

    async def test_no_store_returns_none(self) -> None:
        assert await inject_cross_session_memory(
            store=None, user_id="u1", agent_id="a1"
        ) is None

    async def test_no_user_returns_none(self) -> None:
        store = _StubStore([_fact("x", "high")])
        assert await inject_cross_session_memory(
            store=store, user_id=None, agent_id="a1"
        ) is None


class TestPolicyHonored:
    async def test_confidence_threshold_drops_low(self) -> None:
        store = _StubStore([_fact("low-noise", "low")])
        msg = await inject_cross_session_memory(
            store=store, user_id="u1", agent_id="a1", min_confidence="medium"
        )
        # All facts below threshold => nothing to inject.
        assert msg is None

    async def test_max_cap_honored(self) -> None:
        store = _StubStore([_fact(f"f{i}", "high", days_ago=i) for i in range(10)])
        msg = await inject_cross_session_memory(
            store=store, user_id="u1", agent_id="a1", max_facts=2
        )
        assert msg is not None
        # Exactly 2 fact bullet lines survive the cap.
        assert msg["content"].count("[high]") == 2

    async def test_empty_store_returns_none(self) -> None:
        store = _StubStore([])
        assert await inject_cross_session_memory(
            store=store, user_id="u1", agent_id="a1"
        ) is None


class _Settings:
    """Minimal Settings stand-in for the orchestrator wiring test."""

    def __init__(self, *, enabled: bool) -> None:
        self.cross_session_memory_enabled = enabled
        self.cross_session_memory_min_confidence = "medium"
        self.cross_session_memory_max_facts = 20
        self.cross_session_memory_timeout_sec = 3.0


class TestOrchestratorWiring:
    """``_maybe_prepend_cross_session_memory`` is the orchestrator seam."""

    async def test_flag_off_returns_history_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deep_research.agent import framework_orchestrator as fo

        monkeypatch.setattr(
            "deep_research.core.config.get_settings",
            lambda: _Settings(enabled=False),
        )

        class _Cfg:
            agent_id = "a1"

        history = [{"role": "user", "content": "hi"}]
        result = await fo._maybe_prepend_cross_session_memory(
            history, _Cfg(), object(), "u1", str(uuid4())
        )
        # Byte-identical: same list object, untouched.
        assert result is history

    async def test_flag_off_with_none_history_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deep_research.agent import framework_orchestrator as fo

        monkeypatch.setattr(
            "deep_research.core.config.get_settings",
            lambda: _Settings(enabled=False),
        )

        class _Cfg:
            agent_id = None

        result = await fo._maybe_prepend_cross_session_memory(
            None, _Cfg(), object(), "u1", None
        )
        assert result is None

    async def test_flag_on_no_session_degrades_to_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Flag on but db=None (cached path) => no-memory, history unchanged.
        from deep_research.agent import framework_orchestrator as fo

        monkeypatch.setattr(
            "deep_research.core.config.get_settings",
            lambda: _Settings(enabled=True),
        )

        class _Cfg:
            agent_id = "a1"

        history = [{"role": "user", "content": "hi"}]
        result = await fo._maybe_prepend_cross_session_memory(
            history, _Cfg(), None, "u1", str(uuid4())
        )
        assert result is history
