"""Skill storage protocols + the bundled read-only filesystem store.

The framework defines **backend-agnostic protocols** so hosts can plug in any
persistence (the app ships a Lakebase-backed store).  It also bundles a
read-only :class:`FilesystemSkillStore` that loads the seed skills from package
data — available even with no app database, keeping the framework standalone.

Two protocols:

* :class:`SkillStore` — ``list_skills`` / ``get_skill`` / ``put_skill``.
  ``put_skill`` takes a :class:`SkillSecurityScanner` and persists **only**
  after a PASS (the gate lives in the writing store, not the caller).
* :class:`SkillSecurityScanner` — judges whether a skill is safe to persist.
  **Fail-closed**: a scan error or any non-safe verdict must block the write.

``FilesystemSkillStore.put_skill`` raises — bundled seeds are immutable.
"""

from __future__ import annotations

import importlib.resources
import logging
from typing import Protocol, runtime_checkable

from databricks_deep_research.skills.models import Skill, SkillMeta
from databricks_deep_research.skills.parser import SkillParseError, parse_skill

__all__ = [
    "FilesystemSkillStore",
    "SkillScanResult",
    "SkillSecurityScanner",
    "SkillStore",
    "SkillStoreError",
]

logger = logging.getLogger(__name__)

# Package subdirectory (relative to this package) holding the seed ``*.md``.
_SEEDS_PACKAGE = "databricks_deep_research.skills.seeds"


class SkillStoreError(RuntimeError):
    """Raised when a store operation is unsupported or fails."""


class SkillScanResult(Protocol):
    """Result of a security scan.

    A scanner returns any object satisfying this protocol.  ``safe`` is the
    decisive field: only ``safe is True`` permits a persist.  ``reason`` is an
    optional human-readable explanation recorded with the (rejected) write.
    """

    @property
    def safe(self) -> bool:
        """``True`` only when the skill is cleared to persist."""
        ...

    @property
    def reason(self) -> str:
        """Human-readable rationale for the verdict (may be empty)."""
        ...


@runtime_checkable
class SkillSecurityScanner(Protocol):
    """Judges whether a skill is safe to persist (fail-closed contract).

    Implementations MUST treat their own internal errors as *unsafe*: callers
    rely on ``scan`` either returning a non-safe result or having the writing
    store deny on exception.  Never return ``safe=True`` on uncertainty.
    """

    async def scan(self, skill: Skill) -> SkillScanResult:
        """Return a :class:`SkillScanResult` for *skill*."""
        ...


@runtime_checkable
class SkillStore(Protocol):
    """Backend-agnostic skill persistence.

    The dominant access pattern is "list metadata cheaply, fetch a body on
    demand", so :meth:`list_skills` returns lightweight :class:`SkillMeta` and
    :meth:`get_skill` returns the full :class:`Skill`.
    """

    async def list_skills(self) -> list[SkillMeta]:
        """Return metadata for all available skills (cheap; no bodies)."""
        ...

    async def get_skill(self, name: str) -> Skill | None:
        """Return the full skill named *name*, or ``None`` if absent."""
        ...

    async def put_skill(
        self, skill: Skill, *, scan: SkillSecurityScanner
    ) -> None:
        """Persist *skill* only after *scan* returns a safe verdict.

        Implementations MUST be fail-closed: if the scan raises or returns a
        non-safe result, the skill is NOT persisted.

        Raises:
            SkillStoreError: If the store is read-only or the scan blocks the
                write.
        """
        ...


class FilesystemSkillStore:
    """Read-only :class:`SkillStore` over bundled seed skills (package data).

    Seeds live under ``databricks_deep_research/skills/seeds/*.md`` and are
    parsed lazily on first access, then cached for the store's lifetime.
    Available with no database — the framework's standalone default.
    """

    def __init__(self, *, seeds_package: str = _SEEDS_PACKAGE) -> None:
        self._seeds_package = seeds_package
        self._cache: dict[str, Skill] | None = None

    # -- loading -------------------------------------------------------------

    def _load(self) -> dict[str, Skill]:
        """Parse all bundled seed files (cached).

        A seed that fails to parse is logged and skipped — one malformed bundled
        file must not break listing of the rest.
        """
        if self._cache is not None:
            return self._cache

        skills: dict[str, Skill] = {}
        anchor = importlib.resources.files(self._seeds_package)
        for entry in anchor.iterdir():
            if not entry.name.endswith(".md"):
                continue
            text = entry.read_text(encoding="utf-8")
            try:
                skill = parse_skill(text)
            except SkillParseError:
                logger.exception("SKILL_SEED_PARSE_FAILED file=%s", entry.name)
                continue
            skills[skill.name] = skill

        self._cache = skills
        return skills

    # -- SkillStore ----------------------------------------------------------

    async def list_skills(self) -> list[SkillMeta]:
        """Return metadata for all bundled seeds, sorted by name."""
        skills = self._load()
        return [skills[name].meta for name in sorted(skills)]

    async def get_skill(self, name: str) -> Skill | None:
        """Return the bundled seed named *name*, or ``None`` if absent."""
        return self._load().get(name)

    async def put_skill(
        self, skill: Skill, *, scan: SkillSecurityScanner
    ) -> None:
        """Always raises — the bundled filesystem store is read-only."""
        del skill, scan  # part of the SkillStore protocol; read-only store
        raise SkillStoreError(
            "FilesystemSkillStore is read-only; use a persistent SkillStore "
            "(e.g. the app's LakebaseSkillStore) to author skills"
        )
