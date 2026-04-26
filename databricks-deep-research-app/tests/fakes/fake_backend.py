"""In-memory `StorageBackend` implementation for unit tests.

Goals:

* **Full Protocol coverage.** Every method on `StorageBackend` is implemented;
  tests never need conditional logic for "backend doesn't support X."
* **Aliasing safety.** Deep-copies on both read and write, so a test that
  mutates a returned object cannot pollute backend state. Catches production
  bugs where callers forget to snapshot before mutating.
* **Version-gated semantics.** Mirrors the real version-guard behavior of
  `LakebaseBackend.write_chat_state` so `ConflictError` can be exercised.
* **Projection sync.** `write_chat_state` keeps `chat_deleted_files` in sync
  with `ChatState.live_file_ids()` via full-overwrite per chat, exactly as
  the production backends do.
* **Latency injection.** Optional `latency_ms` simulates slow backends;
  integration-style tests can exercise timeouts without touching a real DB.
* **Failure injection.** Optional callbacks per-method so chaos tests can
  wire in `TransientError`s at will.
"""

from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID

from deep_research.storage.backend import (
    ConflictError,
    PermanentError,
    SchemaError,
    StorageBackend,
)
from deep_research.storage.documents import (
    ChatDocument,
    ChatMeta,
    ChatState,
    PrepJobDocument,
    UserDocument,
)

# Typed alias for the failure-injection callbacks.
FailHook = Callable[[str, tuple[Any, ...]], None]


class FakeBackend(StorageBackend):
    """In-memory backend for tests. Not thread-safe; asyncio-only."""

    def __init__(
        self,
        *,
        latency_ms: float = 0.0,
        fail_hook: FailHook | None = None,
    ) -> None:
        # Document tables — keyed by PK.
        self._chat_meta: dict[UUID, ChatMeta] = {}
        self._chat_state: dict[UUID, ChatState] = {}
        self._user_docs: dict[str, UserDocument] = {}
        self._prep_jobs: dict[UUID, PrepJobDocument] = {}

        # List tables — (table_name, pk_value) → row dict.
        self._list_rows: dict[tuple[str, Any], dict[str, Any]] = {}

        # Append-only tables — list of rows per table.
        self._events: dict[str, list[dict[str, Any]]] = {}

        # Knobs for tests.
        self._latency_ms = latency_ms
        self._fail_hook = fail_hook

        self._migrated = False
        self._closed = False

    # -- Test helpers ---------------------------------------------------

    @property
    def migrated(self) -> bool:
        return self._migrated

    @property
    def closed(self) -> bool:
        return self._closed

    def raw_chat_state(self, chat_id: UUID) -> ChatState | None:
        """Direct (non-copied) read, for test assertions."""
        return self._chat_state.get(chat_id)

    def raw_chat_meta(self, chat_id: UUID) -> ChatMeta | None:
        return self._chat_meta.get(chat_id)

    def raw_events(self, table: str) -> list[dict[str, Any]]:
        return self._events.get(table, [])

    def raw_list_rows(self, table: str) -> list[dict[str, Any]]:
        return [row for (t, _), row in self._list_rows.items() if t == table]

    def _log(self, method: str, args: tuple[Any, ...]) -> None:
        if self._fail_hook is not None:
            self._fail_hook(method, args)

    async def _tick(self) -> None:
        if self._latency_ms:
            await asyncio.sleep(self._latency_ms / 1000.0)

    def _assert_open(self) -> None:
        if self._closed:
            raise PermanentError("FakeBackend is closed")

    # -- Chat document -------------------------------------------------

    async def load_chat(self, chat_id: UUID) -> ChatDocument | None:
        self._log("load_chat", (chat_id,))
        await self._tick()
        self._assert_open()
        meta = self._chat_meta.get(chat_id)
        if meta is None or meta.deleted_at is not None:
            return None
        state = self._chat_state.get(chat_id) or ChatState()
        # Deep-copy so tests can mutate returned objects safely.
        return ChatDocument(meta=meta.model_copy(deep=True), state=state.model_copy(deep=True))

    async def write_chat(
        self,
        doc: ChatDocument,
        *,
        expected_version: int,
    ) -> int:
        self._log("write_chat", (doc.meta.chat_id, expected_version))
        await self._tick()
        self._assert_open()

        chat_id = doc.meta.chat_id
        existing_meta = self._chat_meta.get(chat_id)
        current_version = existing_meta.version if existing_meta else 0
        if current_version != expected_version:
            raise ConflictError(
                f"version mismatch on chat {chat_id}: "
                f"expected {expected_version}, found {current_version}"
            )

        # Upsert meta (keeping the caller's promoted columns).
        new_meta = doc.meta.model_copy(deep=True)
        new_meta.version = current_version + 1
        new_meta.updated_at = datetime.now(tz=timezone.utc)
        self._chat_meta[chat_id] = new_meta

        # Upsert state.
        self._chat_state[chat_id] = doc.state.model_copy(deep=True)

        # Rebuild chat_deleted_files projection.
        table = "chat_deleted_files"
        existing = self._events.get(table, [])
        self._events[table] = [
            row for row in existing if row.get("chat_id") != chat_id
        ]
        for file_id in doc.state.live_file_ids():
            self._events.setdefault(table, []).append(
                {"chat_id": chat_id, "file_id": file_id}
            )

        return new_meta.version

    async def list_chat_metas(
        self,
        user_id: str,
        *,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
        status: str | None = None,
    ) -> list[ChatMeta]:
        self._log("list_chat_metas", (user_id, include_deleted, limit, offset, search, status))
        await self._tick()
        self._assert_open()
        rows = [
            m.model_copy(deep=True)
            for m in self._chat_meta.values()
            if m.user_id == user_id and (include_deleted or m.deleted_at is None)
        ]
        if search:
            lo = search.lower()
            rows = [m for m in rows if lo in (m.title or "").lower()]
        if status is not None:
            rows = [m for m in rows if getattr(m, "status", None) == status]
        rows.sort(key=lambda m: m.updated_at, reverse=True)
        return rows[offset : offset + limit]

    async def load_chats_for_user(
        self,
        user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        search: str | None = None,
    ) -> list[ChatDocument]:
        self._log("load_chats_for_user", (user_id, limit, offset, status, search))
        await self._tick()
        self._assert_open()
        metas = await self.list_chat_metas(
            user_id,
            include_deleted=False,
            limit=limit,
            offset=offset,
            search=search,
            status=status,
        )
        docs = []
        for meta in metas:
            state = self._chat_state.get(meta.chat_id) or ChatState()
            docs.append(
                ChatDocument(meta=meta.model_copy(deep=True), state=state.model_copy(deep=True))
            )
        return docs

    # -- User document -------------------------------------------------

    async def load_user_doc(self, user_id: str) -> UserDocument | None:
        self._log("load_user_doc", (user_id,))
        await self._tick()
        self._assert_open()
        doc = self._user_docs.get(user_id)
        return doc.model_copy(deep=True) if doc else None

    async def write_user_doc(self, doc: UserDocument) -> None:
        self._log("write_user_doc", (doc.user_id,))
        await self._tick()
        self._assert_open()
        self._user_docs[doc.user_id] = doc.model_copy(deep=True)

    # -- Prep-job document --------------------------------------------

    async def load_prep_job(self, job_id: UUID) -> PrepJobDocument | None:
        self._log("load_prep_job", (job_id,))
        await self._tick()
        self._assert_open()
        doc = self._prep_jobs.get(job_id)
        return doc.model_copy(deep=True) if doc else None

    async def write_prep_job(self, doc: PrepJobDocument) -> None:
        self._log("write_prep_job", (doc.prep_job_id,))
        await self._tick()
        self._assert_open()
        self._prep_jobs[doc.prep_job_id] = doc.model_copy(deep=True)

    async def write_prep_heartbeat(self, job_id: UUID, ts: datetime) -> None:
        self._log("write_prep_heartbeat", (job_id, ts))
        await self._tick()
        self._assert_open()
        doc = self._prep_jobs.get(job_id)
        if doc is None:
            raise PermanentError(f"prep_job {job_id} does not exist")
        doc.heartbeat = ts

    # -- List tables ---------------------------------------------------

    async def list_rows(
        self,
        table: str,
        where: dict[str, Any],
        *,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self._log("list_rows", (table, where, order_by, limit))
        await self._tick()
        self._assert_open()
        # Real backends treat every table uniformly. The Fake keeps two
        # internal buckets (list tables and event tables) to mirror the
        # Protocol's two *write* paths, but from the caller's perspective
        # the *read* path doesn't care. Check both.
        list_matches = [
            copy.deepcopy(row)
            for (t, _), row in self._list_rows.items()
            if t == table and all(row.get(k) == v for k, v in where.items())
        ]
        event_matches = [
            copy.deepcopy(row)
            for row in self._events.get(table, [])
            if all(row.get(k) == v for k, v in where.items())
        ]
        matched = list_matches + event_matches
        if order_by is not None:
            key = order_by.lstrip("-")
            reverse = order_by.startswith("-")
            matched.sort(key=lambda r: r.get(key), reverse=reverse)
        if limit is not None:
            matched = matched[:limit]
        return matched

    async def upsert_row(
        self,
        table: str,
        row: dict[str, Any],
        *,
        pk: str,
    ) -> None:
        self._log("upsert_row", (table, pk, row.get(pk)))
        await self._tick()
        self._assert_open()
        pk_value = row[pk]
        self._list_rows[(table, pk_value)] = copy.deepcopy(row)

    async def delete_row(
        self,
        table: str,
        pk_value: Any,
        *,
        pk: str,
    ) -> None:
        self._log("delete_row", (table, pk_value))
        await self._tick()
        self._assert_open()
        self._list_rows.pop((table, pk_value), None)

    # -- Append-only tables -------------------------------------------

    async def append_events(
        self,
        table: str,
        rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> None:
        self._log("append_events", (table, len(rows)))
        await self._tick()
        self._assert_open()
        buf = self._events.setdefault(table, [])
        for row in rows:
            buf.append(copy.deepcopy(row))

    async def read_chunk(
        self,
        file_id: UUID,
        chunk_index: int | None = None,
    ) -> list[dict[str, Any]]:
        self._log("read_chunk", (file_id, chunk_index))
        await self._tick()
        self._assert_open()
        rows = self._events.get("file_chunks", [])
        matched = [
            copy.deepcopy(r)
            for r in rows
            if r.get("file_id") == file_id
            and (chunk_index is None or r.get("chunk_index") == chunk_index)
        ]
        matched.sort(key=lambda r: r.get("chunk_index", 0))
        return matched

    # -- Lifecycle -----------------------------------------------------

    async def migrate(self) -> None:
        self._log("migrate", ())
        await self._tick()
        if self._closed:
            raise SchemaError("FakeBackend is closed; cannot migrate")
        self._migrated = True

    async def close(self) -> None:
        self._log("close", ())
        self._closed = True
