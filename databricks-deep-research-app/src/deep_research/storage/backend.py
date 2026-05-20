"""Storage backend Protocol and error types.

Both production backends (`LakebaseBackend`, `SQLWarehouseBackend`) and the
test `FakeBackend` implement this Protocol. Higher layers talk exclusively to
the Protocol — no SQLAlchemy, no Databricks SDK calls leak upward.

The Protocol deliberately exposes one method per on-wire statement shape. Every
operation that higher layers need is either:

  * A document round-trip (`load_chat` / `write_chat_state` / `write_chat_meta`)
  * A cold list or upsert (`list_rows` / `upsert_row` / `delete_row`)
  * A user / prep-job document round-trip
  * An append-only event batch
  * A file-chunk point read
  * A bounded projection read (`list_chat_metas`)

No multi-statement "use case" method belongs here; orchestration of multiple
backend calls lives in the cache/queue layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from deep_research.storage.documents import (
        ChatDocument,
        ChatMeta,
        PrepJobDocument,
        UserDocument,
    )


# --- Error hierarchy --------------------------------------------------------


class BackendError(Exception):
    """Base class for every error raised by a `StorageBackend`.

    Callers distinguish three broad categories:

    * `TransientError` — retryable (network timeout, warehouse contention,
      momentary auth-token refresh). The WriteQueue retries with backoff.
    * `PermanentError` — not retryable (bad input, PK violation, schema
      mismatch). The WriteQueue logs and drops.
    * `SchemaError` — raised from `migrate()` at startup; considered fatal.
    """


class TransientError(BackendError):
    """Retryable failure; WriteQueue schedules another attempt."""


class PermanentError(BackendError):
    """Non-retryable failure; WriteQueue drops the op and logs."""


class SchemaError(BackendError):
    """DDL problem surfaced at startup; fatal."""


class ConflictError(BackendError):
    """Version-gated `write_chat_state` found a newer version on disk.

    The caller (WriteQueue) should re-read the row via `load_chat`, reconcile
    the in-memory delta, bump the cached version, and re-enqueue. In a
    single-worker deployment this only fires when a migration script wrote
    behind our back.
    """


class MigrationInProgressError(BackendError):
    """Signalled when `STORAGE_MIGRATION_MODE=1` is active.

    Cache translates this to HTTP 503 with `Retry-After`. The WriteQueue's
    flush loop raises it internally to pause without spinning.
    """


# --- Ancillary value types --------------------------------------------------


class WriteOutcome:
    """Per-op outcome returned by `flush_batch`-style methods.

    Small immutable record; intentionally not a pydantic model to keep the
    `BatchResult` path allocation-free under load.
    """

    __slots__ = ("key", "ok", "error")

    def __init__(self, key: str, ok: bool, error: BaseException | None = None) -> None:
        self.key = key
        self.ok = ok
        self.error = error

    def __repr__(self) -> str:  # pragma: no cover — debug only
        tag = "ok" if self.ok else f"err({self.error!r})"
        return f"WriteOutcome({self.key!r}, {tag})"


class BatchResult:
    """Aggregate of `WriteOutcome`s for a batch flush.

    The WriteQueue consults this to decide per-key retry vs. drop.
    """

    __slots__ = ("outcomes",)

    def __init__(self, outcomes: list[WriteOutcome] | None = None) -> None:
        self.outcomes = outcomes if outcomes is not None else []

    def add(self, key: str, *, ok: bool, error: BaseException | None = None) -> None:
        self.outcomes.append(WriteOutcome(key, ok, error))

    @property
    def failed(self) -> list[WriteOutcome]:
        return [o for o in self.outcomes if not o.ok]

    def __len__(self) -> int:
        return len(self.outcomes)


# --- The Protocol -----------------------------------------------------------


@runtime_checkable
class StorageBackend(Protocol):
    """Pluggable storage Protocol.

    Every implementation must be safe for concurrent use from within a single
    asyncio event loop. Blocking wire calls (Databricks SDK) should be wrapped
    in `asyncio.to_thread`; asyncpg-native calls are already non-blocking.
    """

    # -- Hot path: split chat document ----------------------------------

    async def load_chat(self, chat_id: UUID) -> ChatDocument | None:
        """Return the chat document by PK, or None if absent / soft-deleted.

        Implementations MUST issue exactly one statement (JOIN chat_meta +
        chat_state on Postgres, LEFT JOIN on Delta).
        """
        ...

    async def write_chat(
        self,
        doc: ChatDocument,
        *,
        expected_version: int,
    ) -> int:
        """Atomic, version-gated upsert of the entire chat document.

        Writes the meta row, the state row, and the `chat_deleted_files`
        projection in a single logical operation. Returns the new version.
        Raises `ConflictError` if the on-disk version does not equal
        `expected_version`.

        Atomicity on Lakebase is a single transaction (ACID). On the SQL
        Warehouse it is a sequence of statements with a documented race
        window that matters only in the multi-worker future (see plan
        follow-up F1); single-worker deployments are safe today.
        """
        ...

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
        """Projection-only list for `GET /api/v1/chats`.

        Never touches the `chat_state` payload. Used by the ColdReadCache for
        the chat-list UI and by the Hydrator for prefetch.

        Args:
            user_id: Filter to this user's chats.
            include_deleted: When False (default), excludes soft-deleted chats.
            limit: Maximum rows returned.
            offset: Number of rows to skip (for pagination).
            search: Case-insensitive substring match on ``title``.
                Postgres: ``ILIKE '%<escaped>%' ESCAPE '\\'``.
                Delta: ``LOWER(title) LIKE LOWER('%<pattern>%')``.
                Special chars ``%``, ``_``, ``\\`` in the search term are
                escaped before interpolation.
            status: Exact status filter (e.g. ``"active"``, ``"archived"``).
        """
        ...

    async def load_chats_for_user(
        self,
        user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        search: str | None = None,
    ) -> list[ChatDocument]:
        """Single-round-trip batch load of full chat documents for a user.

        Issues ONE statement (JOIN chat_meta + chat_state on both backends).
        Applies the same filters as ``list_chat_metas`` — use this when callers
        need the full ``ChatDocument`` (e.g. for a batch-prefetch of state).

        Thread-safety: no per-call serialization; relies on backend atomicity.
        """
        ...

    # -- User document --------------------------------------------------

    async def load_user_doc(self, user_id: str) -> UserDocument | None: ...
    async def write_user_doc(self, doc: UserDocument) -> None: ...

    # -- Prep-job document ---------------------------------------------

    async def load_prep_job(self, job_id: UUID) -> PrepJobDocument | None: ...
    async def write_prep_job(self, doc: PrepJobDocument) -> None: ...
    async def write_prep_heartbeat(self, job_id: UUID, ts: datetime) -> None:
        """Direct single-statement heartbeat update; bypasses the WriteQueue.

        Heartbeats must never be deferred by flush backlog or the zombie
        detector will false-positive. See plan section "Concurrency model".
        """
        ...

    # -- List tables (templates, custom_agents) ------------------------

    async def list_rows(
        self,
        table: str,
        where: Mapping[str, Any],
        *,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    async def upsert_row(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: str,
    ) -> None: ...

    async def delete_row(
        self,
        table: str,
        pk_value: Any,
        *,
        pk: str,
    ) -> None: ...

    # -- Append-only tables --------------------------------------------

    async def append_events(
        self,
        table: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        """Bulk insert for `research_events`, `audit_log`, `file_chunks`,
        `message_feedback`, `chat_deleted_files`.

        Implementations issue ONE statement per call regardless of batch size.
        """
        ...

    async def read_chunk(
        self,
        file_id: UUID,
        chunk_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """Read one or all chunks for an uploaded file."""
        ...

    # -- Lifecycle ------------------------------------------------------

    async def migrate(self) -> None:
        """Apply idempotent DDL for the backend.

        Raises `SchemaError` if versions are incompatible.
        """
        ...

    async def close(self) -> None:
        """Release all backend resources. Called on app shutdown after the
        WriteQueue has drained.
        """
        ...
