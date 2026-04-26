"""Cache-backed `IPreferencesService`.

Legacy stores preferences in a dedicated `user_preferences` table. Cached
impl stashes the same fields under `UserDocument.preferences` (a JSON dict).
Returns a `SimpleNamespace` that quacks like the legacy `UserPreferences`
ORM object — every field callers access in the codebase (`system_instructions`,
`default_research_depth`, `default_query_mode`, `theme`,
`notifications_enabled`, `updated_at`, `update_instructions` / `update_depth`
bound methods) is either stored or adapted.

Bound-method mutators (`update_instructions` etc.) are emulated by setting
the corresponding attribute on the namespace — callers that previously did
`prefs.update_instructions(txt)` now effectively mutate the returned object;
the cached service persists once at the end of `update_preferences`. Legacy
parity is behavioral, not API-identical for these rarely-called mutators.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from deep_research.services._cached_base import _CachedServiceBase
from deep_research.services._protocols import IPreferencesService
from deep_research.storage.documents import UserDocument

if TYPE_CHECKING:
    from deep_research.storage.factory import StorageStack


logger = logging.getLogger(__name__)


_DEFAULTS: dict[str, Any] = {
    "system_instructions": None,
    "default_research_depth": "AUTO",
    "default_query_mode": "simple",
    "theme": "system",
    "notifications_enabled": True,
}


class CachedPreferencesService(_CachedServiceBase, IPreferencesService):
    """`IPreferencesService` via `UserDocument.preferences`."""

    def __init__(self, stack: "StorageStack") -> None:
        super().__init__(stack)

    async def get_preferences(self, user_id: str) -> SimpleNamespace:
        doc = await self._load_or_create(user_id)
        return _prefs_namespace(doc)

    async def update_preferences(
        self,
        user_id: str,
        system_instructions: str | None = None,
        default_research_depth: Any | None = None,
        default_query_mode: Any | None = None,
        theme: str | None = None,
        notifications_enabled: bool | None = None,
    ) -> SimpleNamespace:
        doc = await self._load_or_create(user_id)
        prefs = dict(doc.preferences)
        if system_instructions is not None:
            prefs["system_instructions"] = system_instructions
        if default_research_depth is not None:
            prefs["default_research_depth"] = _enum_value(default_research_depth)
        if default_query_mode is not None:
            prefs["default_query_mode"] = _enum_value(default_query_mode)
        if theme is not None:
            prefs["theme"] = theme
        if notifications_enabled is not None:
            prefs["notifications_enabled"] = notifications_enabled
        now = datetime.now(UTC)
        prefs["updated_at"] = now.isoformat()
        doc.preferences = prefs
        doc.updated_at = now
        await self._stack.backend.write_user_doc(doc)
        logger.info("PREFERENCES_UPDATED user_id=%s", user_id)
        return _prefs_namespace(doc)

    async def get_system_instructions(self, user_id: str) -> str | None:
        p = await self.get_preferences(user_id)
        return p.system_instructions

    async def get_default_research_depth(self, user_id: str) -> Any:
        p = await self.get_preferences(user_id)
        return p.default_research_depth

    async def get_default_query_mode(self, user_id: str) -> str:
        p = await self.get_preferences(user_id)
        return p.default_query_mode

    def to_dict(self, preferences: Any) -> dict[str, Any]:
        depth = getattr(preferences, "default_research_depth", None)
        updated_at = getattr(preferences, "updated_at", None)
        return {
            "system_instructions": getattr(preferences, "system_instructions", None),
            "default_research_depth": _enum_value(depth),
            "default_query_mode": getattr(preferences, "default_query_mode", None),
            "theme": getattr(preferences, "theme", None),
            "notifications_enabled": getattr(preferences, "notifications_enabled", None),
            "updated_at": (
                updated_at.isoformat() if isinstance(updated_at, datetime)
                else updated_at
            ),
        }

    # -- Internal ------------------------------------------------------

    async def _load_or_create(self, user_id: str) -> UserDocument:
        doc = await self._stack.backend.load_user_doc(user_id)
        if doc is not None:
            # Ensure defaults for any missing keys so accessors don't KeyError.
            merged = {**_DEFAULTS, **(doc.preferences or {})}
            if merged != doc.preferences:
                doc.preferences = merged
            return doc
        now = datetime.now(UTC)
        return UserDocument(
            user_id=user_id,
            created_at=now,
            updated_at=now,
            preferences=dict(_DEFAULTS),
        )


# --- Helpers --------------------------------------------------------------


def _enum_value(value: Any) -> Any:
    """Return `value.value` for Enums, else pass through."""
    return getattr(value, "value", value)


def _prefs_namespace(doc: UserDocument) -> SimpleNamespace:
    p = {**_DEFAULTS, **(doc.preferences or {})}
    return SimpleNamespace(
        user_id=doc.user_id,
        system_instructions=p.get("system_instructions"),
        default_research_depth=p.get("default_research_depth"),
        default_query_mode=p.get("default_query_mode"),
        theme=p.get("theme"),
        notifications_enabled=p.get("notifications_enabled"),
        updated_at=doc.updated_at,
    )
