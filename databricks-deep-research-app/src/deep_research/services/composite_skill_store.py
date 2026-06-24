"""Composite skill store — a read aggregator over multiple :class:`SkillStore`s.

The runtime needs ONE :class:`SkillStore` (wired into
``ToolFactoryContext.extras["_skill_store"]``) that transparently spans every
configured source: the user's workspace-FS folders, the governed Lakebase
store, and the framework's bundled seeds. :class:`CompositeSkillStore` composes
them in precedence order and deduplicates by name.

Read-only by design: authoring goes to a concrete writable store (the app's
``LakebaseSkillStore``); this aggregator only powers the runtime read paths
(``read_skill``, the prompt skills-section, and discovery). ``put_skill`` raises
``SkillStoreError`` accordingly — which the protocol explicitly permits.

Fail-soft: a single failing backend (an unreachable workspace, a transient
Lakebase error) is logged and skipped so the remaining sources still resolve —
a skill source being momentarily unavailable must never break a research run.
"""

from __future__ import annotations

import logging

from databricks_deep_research.skills import (
    Skill,
    SkillMeta,
    SkillSecurityScanner,
    SkillStore,
    SkillStoreError,
)

logger = logging.getLogger(__name__)

__all__ = ["CompositeSkillStore"]


class CompositeSkillStore:
    """A :class:`SkillStore` that reads across several backends by precedence.

    Args:
        stores: Backends in precedence order (highest first). On a name
            collision the earlier store wins for both listing and fetching.
    """

    def __init__(self, stores: list[SkillStore]) -> None:
        self._stores = list(stores)

    async def list_skills(self) -> list[SkillMeta]:
        """Merge metadata across backends; first occurrence of a name wins."""
        seen: dict[str, SkillMeta] = {}
        for store in self._stores:
            try:
                metas = await store.list_skills()
            except Exception:  # noqa: BLE001 — one bad backend must not break listing
                logger.exception(
                    "COMPOSITE_SKILL_LIST_FAILED store=%s", type(store).__name__
                )
                continue
            for meta in metas:
                seen.setdefault(meta.name, meta)
        return list(seen.values())

    async def get_skill(self, name: str) -> Skill | None:
        """Return the skill from the highest-precedence backend that has it."""
        for store in self._stores:
            try:
                skill = await store.get_skill(name)
            except Exception:  # noqa: BLE001 — try the next backend on error
                logger.exception(
                    "COMPOSITE_SKILL_GET_FAILED store=%s name=%s",
                    type(store).__name__,
                    name,
                )
                continue
            if skill is not None:
                return skill
        return None

    async def put_skill(self, skill: Skill, *, scan: SkillSecurityScanner) -> None:
        """Always raises — the composite is a read aggregator.

        Authoring must target a concrete writable store (e.g. the app's
        ``LakebaseSkillStore``) so the fail-closed scan + audit happen in one
        well-defined place rather than ambiguously across backends.
        """
        del skill, scan  # part of the SkillStore protocol; read-only aggregator
        raise SkillStoreError(
            "CompositeSkillStore is read-only (a runtime read aggregator); "
            "author skills via a concrete writable store (e.g. LakebaseSkillStore)."
        )
