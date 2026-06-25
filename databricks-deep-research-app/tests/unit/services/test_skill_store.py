"""Unit tests for the app-side skill store, scanner, and seed-sync.

All mocked — no real LLM and no real database. The LakebaseSkillStore's
fail-closed gate and the seed-sync orchestration are exercised via spies; the
SQL itself is covered by integration/migration tests, not here.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from databricks_deep_research.skills import (
    FilesystemSkillStore,
    Skill,
    SkillStoreError,
)

from deep_research.services.skill_store import (
    LakebaseSkillStore,
    LLMSkillSecurityScanner,
    sync_seed_skills,
)


def _skill() -> Skill:
    return Skill(name="demo", description="a demo skill", body="# Demo\n\nbody")


def _llm_returning(content: str) -> AsyncMock:
    """A mock FrameworkLLMClient whose complete() returns *content*."""
    llm = AsyncMock()
    response = AsyncMock()
    response.content = content
    llm.complete = AsyncMock(return_value=response)
    return llm


# ---------------------------------------------------------------------------
# LLMSkillSecurityScanner — fail-closed
# ---------------------------------------------------------------------------


class TestScanner:
    async def test_safe_verdict(self) -> None:
        scanner = LLMSkillSecurityScanner(_llm_returning("SAFE\nlooks fine"))
        result = await scanner.scan(_skill())
        assert result.safe is True
        assert result.reason == "looks fine"

    async def test_unsafe_verdict(self) -> None:
        scanner = LLMSkillSecurityScanner(
            _llm_returning("UNSAFE\nprompt injection")
        )
        result = await scanner.scan(_skill())
        assert result.safe is False
        assert "prompt injection" in result.reason

    async def test_empty_verdict_is_unsafe(self) -> None:
        scanner = LLMSkillSecurityScanner(_llm_returning("   "))
        assert (await scanner.scan(_skill())).safe is False

    async def test_garbage_verdict_is_unsafe(self) -> None:
        scanner = LLMSkillSecurityScanner(_llm_returning("maybe? not sure"))
        assert (await scanner.scan(_skill())).safe is False

    async def test_llm_exception_is_unsafe(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("gateway 500"))
        scanner = LLMSkillSecurityScanner(llm)
        result = await scanner.scan(_skill())
        assert result.safe is False
        assert "scan error" in result.reason


# ---------------------------------------------------------------------------
# LakebaseSkillStore.put_skill — fail-closed gate
# ---------------------------------------------------------------------------


class _ScanResultStub:
    def __init__(self, safe: bool, reason: str = "") -> None:
        self.safe = safe
        self.reason = reason


class _Scanner:
    def __init__(self, result: Any = None, *, raises: bool = False) -> None:
        self._result = result
        self._raises = raises

    async def scan(self, skill: Skill) -> Any:
        if self._raises:
            raise RuntimeError("scanner exploded")
        return self._result


class TestPutSkillGate:
    async def test_unsafe_scan_does_not_persist(
        self, mock_db_session: AsyncMock
    ) -> None:
        store = LakebaseSkillStore(mock_db_session)
        store._upsert = AsyncMock()  # type: ignore[method-assign]
        with pytest.raises(SkillStoreError, match="rejected"):
            await store.put_skill(
                _skill(), scan=_Scanner(_ScanResultStub(safe=False, reason="bad"))
            )
        store._upsert.assert_not_awaited()
        mock_db_session.add.assert_not_called()

    async def test_scanner_exception_does_not_persist(
        self, mock_db_session: AsyncMock
    ) -> None:
        store = LakebaseSkillStore(mock_db_session)
        store._upsert = AsyncMock()  # type: ignore[method-assign]
        with pytest.raises(SkillStoreError, match="scan failed"):
            await store.put_skill(_skill(), scan=_Scanner(raises=True))
        store._upsert.assert_not_awaited()

    async def test_safe_scan_persists(self, mock_db_session: AsyncMock) -> None:
        store = LakebaseSkillStore(mock_db_session)
        store._upsert = AsyncMock()  # type: ignore[method-assign]
        await store.put_skill(
            _skill(), scan=_Scanner(_ScanResultStub(safe=True, reason="ok"))
        )
        store._upsert.assert_awaited_once()
        # verdict threaded through to the upsert
        _, kwargs = store._upsert.call_args
        assert kwargs["security_verdict"] == "ok"
        assert kwargs["is_seed"] is False


# ---------------------------------------------------------------------------
# seed-sync — idempotent orchestration
# ---------------------------------------------------------------------------


class TestSeedSync:
    async def test_syncs_all_bundled_seeds(
        self, mock_db_session: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[str] = []

        async def _fake_upsert_seed(self: Any, skill: Skill) -> None:
            captured.append(skill.name)

        monkeypatch.setattr(
            LakebaseSkillStore, "upsert_seed", _fake_upsert_seed
        )
        count = await sync_seed_skills(
            mock_db_session, seed_store=FilesystemSkillStore()
        )
        assert count == 3
        assert set(captured) == {"deep-research", "data-analysis", "chart"}

    async def test_idempotent_rerun(
        self, mock_db_session: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        async def _fake_upsert_seed(self: Any, skill: Skill) -> None:
            calls["n"] += 1

        monkeypatch.setattr(
            LakebaseSkillStore, "upsert_seed", _fake_upsert_seed
        )
        store = FilesystemSkillStore()
        await sync_seed_skills(mock_db_session, seed_store=store)
        await sync_seed_skills(mock_db_session, seed_store=store)
        # Running twice upserts the same 3 seeds each time (no duplication logic
        # needed — upsert-by-name is the idempotency mechanism).
        assert calls["n"] == 6
