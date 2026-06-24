"""Unit tests for the fail-closed security-scan gate in put_skill.

The framework's :class:`SkillStore` Protocol mandates that ``put_skill``
persists ONLY after a safe verdict and is fail-closed (a non-safe verdict or a
scan exception must block the write). These tests use a reference in-memory
store that implements that gate so the contract is verified independently of
the app's Lakebase backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from databricks_deep_research.skills.models import Skill, SkillMeta
from databricks_deep_research.skills.store import (
    SkillScanResult,
    SkillSecurityScanner,
    SkillStoreError,
)


@dataclass(frozen=True)
class _ScanResult:
    safe: bool
    reason: str = ""


class _SafeScanner:
    async def scan(self, skill: Skill) -> SkillScanResult:
        return _ScanResult(safe=True, reason="ok")


class _UnsafeScanner:
    async def scan(self, skill: Skill) -> SkillScanResult:
        return _ScanResult(safe=False, reason="prompt injection detected")


class _RaisingScanner:
    async def scan(self, skill: Skill) -> SkillScanResult:
        raise RuntimeError("scanner backend exploded")


class _GateStore:
    """Reference SkillStore implementing the fail-closed put_skill gate."""

    def __init__(self) -> None:
        self._by_name: dict[str, Skill] = {}

    async def list_skills(self) -> list[SkillMeta]:
        return [s.meta for s in self._by_name.values()]

    async def get_skill(self, name: str) -> Skill | None:
        return self._by_name.get(name)

    async def put_skill(
        self, skill: Skill, *, scan: SkillSecurityScanner
    ) -> None:
        # Fail-closed: any scan exception denies the write.
        try:
            result = await scan.scan(skill)
        except Exception as exc:  # noqa: BLE001 - intentional fail-closed catch
            raise SkillStoreError(f"security scan failed: {exc}") from exc
        if not result.safe:
            raise SkillStoreError(
                f"security scan rejected skill {skill.name!r}: {result.reason}"
            )
        self._by_name[skill.name] = skill


def _skill() -> Skill:
    return Skill(name="x", description="a skill", body="body")


class TestFailClosedGate:
    async def test_safe_verdict_persists(self) -> None:
        store = _GateStore()
        await store.put_skill(_skill(), scan=_SafeScanner())
        assert await store.get_skill("x") is not None

    async def test_unsafe_verdict_does_not_persist(self) -> None:
        store = _GateStore()
        with pytest.raises(SkillStoreError, match="rejected"):
            await store.put_skill(_skill(), scan=_UnsafeScanner())
        assert await store.get_skill("x") is None

    async def test_scanner_exception_does_not_persist(self) -> None:
        store = _GateStore()
        with pytest.raises(SkillStoreError, match="scan failed"):
            await store.put_skill(_skill(), scan=_RaisingScanner())
        assert await store.get_skill("x") is None
