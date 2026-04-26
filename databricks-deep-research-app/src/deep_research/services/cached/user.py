"""Cache-backed `IUserService` — routes user identity through `UserDocument`.

Legacy `UserService.upsert(user_id, email, display_name)` writes to a `users`
table with `last_seen_at = now()`. The cached impl stores the same fields in
`UserDocument.profile` alongside any other per-user state. `resolve_user_ids`
fan-outs across many users so we issue them in parallel via `asyncio.gather`
to keep p95 sane.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING

from deep_research.services._cached_base import _CachedServiceBase
from deep_research.services._protocols import IUserService
from deep_research.storage.documents import UserDocument

if TYPE_CHECKING:
    from deep_research.storage.factory import StorageStack

logger = logging.getLogger(__name__)


class CachedUserService(_CachedServiceBase, IUserService):
    """`IUserService` over `UserDocument`."""

    def __init__(self, stack: "StorageStack") -> None:
        super().__init__(stack)

    async def upsert(
        self,
        user_id: str,
        email: str | None,
        display_name: str | None,
    ) -> None:
        now = datetime.now(UTC)
        existing = await self._stack.backend.load_user_doc(user_id)
        if existing is None:
            doc = UserDocument(
                user_id=user_id,
                created_at=now,
                updated_at=now,
                profile={
                    "email": email,
                    "display_name": display_name,
                    "last_seen_at": now.isoformat(),
                },
            )
        else:
            doc = existing
            profile = dict(doc.profile)
            profile["email"] = email
            profile["display_name"] = display_name
            profile["last_seen_at"] = now.isoformat()
            doc.profile = profile
            doc.updated_at = now
        await self._stack.backend.write_user_doc(doc)

    async def resolve_user_ids(
        self,
        user_ids: list[str],
    ) -> dict[str, tuple[str | None, str | None]]:
        if not user_ids:
            return {}
        docs = await asyncio.gather(
            *(self._stack.backend.load_user_doc(uid) for uid in user_ids),
            return_exceptions=True,
        )
        resolved: dict[str, tuple[str | None, str | None]] = {}
        for uid, doc in zip(user_ids, docs, strict=False):
            if isinstance(doc, BaseException):
                logger.warning("load_user_doc failed for %s: %s", uid, doc)
                continue
            if doc is None:
                continue
            profile = doc.profile or {}
            resolved[uid] = (profile.get("email"), profile.get("display_name"))
        return resolved
