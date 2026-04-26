"""StorageBackend implementation over Postgres / Lakebase.

Uses the existing async SQLAlchemy engine (`deep_research.db.session`) to get
OAuth token rotation, PgBouncer-safe connection config, and pool recycling for
free. Every method issues raw SQL via `text(...)` — no ORM, no ad-hoc query
building — so the SQL shape is explicit and matches the plan document.

Transactional semantics: `write_chat` runs meta upsert + state upsert +
`chat_deleted_files` rebuild in a single transaction. If any step fails the
whole thing rolls back and the WriteQueue's retry logic kicks in.
"""

from __future__ import annotations

import importlib.resources
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import sqlalchemy.exc
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, AsyncSession

from deep_research.storage.backend import (
    ConflictError,
    PermanentError,
    SchemaError,
    TransientError,
)
from deep_research.storage.documents import (
    ChatDocument,
    ChatMeta,
    ChatState,
    PrepJobDocument,
    UserDocument,
)

logger = logging.getLogger(__name__)


_TRANSIENT_MARKERS = (
    # asyncpg / network
    "connection is closed",
    "connection was closed",
    "connection is not open",
    "connection is lost",
    "another operation is in progress",
    "server closed the connection",
    # Postgres codes that correspond to transient conditions
    "deadlock detected",
    "serialization failure",
    "could not serialize access",
    "canceling statement due to statement timeout",
)


def _wrap_error(exc: BaseException) -> Exception:
    """Map a raw SQLAlchemy error into our retry-aware hierarchy."""
    message = str(exc).lower()
    if isinstance(exc, (sqlalchemy.exc.InterfaceError, sqlalchemy.exc.OperationalError)):
        return TransientError(str(exc))
    if isinstance(exc, sqlalchemy.exc.TimeoutError):
        return TransientError(str(exc))
    if any(marker in message for marker in _TRANSIENT_MARKERS):
        return TransientError(str(exc))
    return PermanentError(str(exc))


def _dumps(payload: Any) -> str:
    """Render a pydantic model or dict to a JSON string for ::jsonb casts."""
    if hasattr(payload, "model_dump_json"):
        return payload.model_dump_json()
    return json.dumps(payload, default=str, sort_keys=True)


def _coerce_jsonish(value: Any) -> Any:
    """JSON-encode dict/list/tuple/Pydantic binds for raw ``text()`` statements.

    ``text()`` bypasses SQLAlchemy's type system, so the JSONB bind
    processor never runs and asyncpg's registered jsonb codec receives a
    raw Python object — failing with ``AttributeError: 'dict' object has
    no attribute 'encode'``. This restores parity with
    ``param_codec._encode_value`` in ``storage/sql_warehouse.py``, which
    performs the analogous coercion on the Databricks SDK side. Scalars
    pass through; already-serialized JSON strings pass through (not
    dict/list).
    """
    if isinstance(value, dict | list | tuple) or hasattr(value, "model_dump_json"):
        return _dumps(value)
    return value


def _scoped_sm(
    inner: "async_sessionmaker[AsyncSession]",
    schema: str,
) -> "async_sessionmaker[AsyncSession]":
    """Schema-scoping wrapper — now an identity passthrough.

    Historical versions issued ``SET search_path …; COMMIT;`` on every
    session entry, and a later iteration moved the same GUC to
    asyncpg's ``server_settings`` at connect time. Both approaches are
    unsafe because the SQLAlchemy async engine is shared with the
    legacy SQLAlchemy ORM: pinning the pool-wide search_path to
    ``deep_research_state,public`` shadows the legacy ``public``
    tables whose same-named twins in ``deep_research_state`` have
    different columns — producing ``UndefinedColumnError`` on legacy
    queries.

    Schema scoping is now applied inline by every SQL statement in
    this module (``f"{self._ns}.chat_meta"`` etc.). That matches the
    ``${ns}.table`` style that ``SQLWarehouseBackend`` already uses
    and leaves legacy ORM references to resolve via the server
    default search_path.

    Retained as a passthrough so existing call sites (and future
    tests that mock it) keep working. The ``schema`` argument is
    still validated for defense in depth.
    """
    _assert_safe_identifier(schema)
    return inner


class LakebaseBackend:
    """Postgres / Lakebase backend. Protocol-compliant with `StorageBackend`."""

    def __init__(
        self,
        *,
        engine: AsyncEngine | None = None,
        session_maker: async_sessionmaker[AsyncSession] | None = None,
        ddl_path: Path | None = None,
        schema: str | None = None,
    ) -> None:
        """
        Args:
            engine: Optional pre-built AsyncEngine. If omitted, the shared
                engine from `deep_research.db.session.get_engine()` is used.
            session_maker: Optional pre-built session factory. Overrides
                `engine` if both are provided.
            ddl_path: Optional override for the DDL file path (tests).
            schema: Postgres schema that holds the chat-document tables. When
                set, every session issued by this backend runs
                ``SET LOCAL search_path TO <schema>, public`` so unqualified
                table names resolve to this schema first. Legacy Alembic
                tables (in ``public``) stay untouched. Must match a schema
                that exists (auto-created by ``migrate()``).
        """
        raw_sm: async_sessionmaker[AsyncSession]
        if session_maker is not None:
            raw_sm = session_maker
        elif engine is not None:
            raw_sm = async_sessionmaker(engine, expire_on_commit=False)
        else:
            # Import lazily so tests that never call `migrate()` don't need
            # the settings module or lakebase config to be importable.
            from deep_research.db.session import get_session_maker

            raw_sm = get_session_maker()

        self._schema = schema
        self._sm = _scoped_sm(raw_sm, schema) if schema else raw_sm

        self._ddl_path = ddl_path or (
            Path(__file__).with_name("lakebase_ddl.sql")
        )
        self._closed = False

    # -- Lifecycle -----------------------------------------------------

    async def migrate(self) -> None:
        """Apply idempotent DDL.

        When ``schema`` is set, create the schema first (outside the DDL
        transaction — CREATE SCHEMA takes a global lock that conflicts with
        the subsequent CREATE TABLE statements on Lakebase autoscaling),
        then run the DDL with every table reference fully qualified.

        The DDL file uses a ``{ns}.`` placeholder before every table name.
        We substitute it with the configured schema (e.g. ``"deep_research_state".``)
        before executing, so the DDL never relies on server-side ``search_path``
        resolution — that would be unsafe anyway because the SQLAlchemy engine
        is shared with the legacy ORM that reads unqualified ``public.*``
        tables with the same names but different column layouts. This matches
        the fully-qualified query style used everywhere else in the storage
        engine (``f"{self._ns}.chat_meta"`` in this module, ``${ns}.table`` in
        ``sql_warehouse.py``). A previous iteration used ``SET LOCAL
        search_path`` to scope the transaction, but the ``SET LOCAL`` didn't
        reliably propagate through the PgBouncer-pooled asyncpg pipeline for
        the subsequent ``CREATE TABLE``, causing Postgres to resolve unqualified
        names against the connection default (``public``) where the app
        service principal lacks CREATE privilege.
        """
        try:
            raw_sql = self._ddl_path.read_text()
        except OSError as exc:
            raise SchemaError(f"cannot read {self._ddl_path}: {exc}") from exc

        if self._schema is not None:
            _assert_safe_identifier(self._schema)
            ns_prefix = f'"{self._schema}".'
        else:
            # No schema configured: let unqualified names resolve via the
            # connection's default search_path (typically ``public``). This
            # path is only exercised by tests and pathological configs.
            ns_prefix = ""
        ddl_sql = raw_sql.replace("{ns}.", ns_prefix)
        statements = _split_sql(ddl_sql)

        try:
            if self._schema is not None:
                # Use the raw (unscoped) session_maker for this CREATE — the
                # scoped one would try to SET search_path to a schema that
                # doesn't exist yet (historical artifact; scoped is now a
                # passthrough).
                from deep_research.db.session import get_session_maker

                raw_sm = get_session_maker()
                async with raw_sm() as session:
                    async with session.begin():
                        await session.execute(
                            text(f'CREATE SCHEMA IF NOT EXISTS "{self._schema}"')
                        )
            async with self._sm() as session:
                async with session.begin():
                    for stmt in statements:
                        await session.execute(text(stmt))
        except Exception as exc:
            raise SchemaError(f"DDL application failed: {exc}") from exc

    async def close(self) -> None:
        self._closed = True
        # The engine is shared with the rest of the app; it is disposed by
        # the app's lifespan, not here.

    def _ensure_open(self) -> None:
        if self._closed:
            raise PermanentError("LakebaseBackend is closed")

    @property
    def _ns(self) -> str:
        """Schema-qualifier prefix for every SQL statement.

        All queries fully qualify their tables (``f"{self._ns}.chat_meta"``)
        so the new storage engine never depends on ``search_path``. Legacy
        ORM unqualified references resolve to the server default
        (``public``) untouched. Falls back to ``public`` only when no
        schema was configured — a pathological test-only state.
        """
        return self._schema or "public"

    # -- Hot path: chat document --------------------------------------

    async def load_chat(self, chat_id: UUID) -> ChatDocument | None:
        self._ensure_open()
        try:
            async with self._sm() as session:
                row = (
                    await session.execute(
                        text(
                            f"SELECT m.chat_id, m.user_id, m.title, m.preview, "
                            f"       m.created_at, m.updated_at, m.deleted_at, m.version, "
                            f"       s.state "
                            f"FROM {self._ns}.chat_meta m "
                            f"LEFT JOIN {self._ns}.chat_state s USING (chat_id) "
                            f"WHERE m.chat_id = :cid AND m.deleted_at IS NULL"
                        ),
                        {"cid": chat_id},
                    )
                ).first()
        except Exception as exc:
            raise _wrap_error(exc) from exc

        if row is None:
            return None

        meta = ChatMeta(
            chat_id=row.chat_id,
            user_id=row.user_id,
            title=row.title,
            preview=row.preview,
            created_at=row.created_at,
            updated_at=row.updated_at,
            deleted_at=row.deleted_at,
            version=row.version,
        )
        raw_state = row.state if row.state is not None else {"schema_version": 1}
        state = ChatState.model_validate(raw_state)
        return ChatDocument(meta=meta, state=state)

    async def write_chat(
        self,
        doc: ChatDocument,
        *,
        expected_version: int,
    ) -> int:
        self._ensure_open()
        cid = doc.meta.chat_id
        state_json = doc.state.model_dump_json()

        try:
            async with self._sm() as session:
                async with session.begin():
                    # Version-gated meta upsert. INSERT for new chats,
                    # conditional UPDATE for existing ones. RETURNING
                    # version lets us distinguish success from conflict.
                    result = await session.execute(
                        text(
                            f"INSERT INTO {self._ns}.chat_meta "
                            f"  (chat_id, user_id, title, preview, created_at, "
                            f"   updated_at, deleted_at, version) "
                            f"VALUES "
                            f"  (:cid, :uid, :title, :preview, :created_at, "
                            f"   now(), :deleted_at, 1) "
                            f"ON CONFLICT (chat_id) DO UPDATE SET "
                            f"  title = EXCLUDED.title, "
                            f"  preview = EXCLUDED.preview, "
                            f"  updated_at = now(), "
                            f"  deleted_at = EXCLUDED.deleted_at, "
                            f"  version = {self._ns}.chat_meta.version + 1 "
                            f"WHERE {self._ns}.chat_meta.version = :expected "
                            f"RETURNING version"
                        ),
                        {
                            "cid": cid,
                            "uid": doc.meta.user_id,
                            "title": doc.meta.title,
                            "preview": doc.meta.preview,
                            "created_at": doc.meta.created_at,
                            "deleted_at": doc.meta.deleted_at,
                            "expected": expected_version,
                        },
                    )
                    row = result.first()
                    if row is None:
                        # Either the row exists with a different version, or
                        # a concurrent INSERT won the race (not possible in
                        # single-worker, but defensively handle).
                        raise ConflictError(
                            f"version conflict on chat {cid}: expected "
                            f"{expected_version}"
                        )
                    new_version = int(row.version)

                    # State upsert.
                    await session.execute(
                        text(
                            f"INSERT INTO {self._ns}.chat_state (chat_id, state) "
                            f"VALUES (:cid, CAST(:state AS jsonb)) "
                            f"ON CONFLICT (chat_id) DO UPDATE SET "
                            f"  state = EXCLUDED.state"
                        ),
                        {"cid": cid, "state": state_json},
                    )

                    # Rebuild chat_deleted_files projection for this chat.
                    await session.execute(
                        text(
                            f"DELETE FROM {self._ns}.chat_deleted_files WHERE chat_id = :cid"
                        ),
                        {"cid": cid},
                    )
                    file_ids = doc.state.live_file_ids()
                    if file_ids:
                        await session.execute(
                            text(
                                f"INSERT INTO {self._ns}.chat_deleted_files "
                                f"  (chat_id, file_id) VALUES (:cid, :fid)"
                            ),
                            [{"cid": cid, "fid": fid} for fid in file_ids],
                        )
            return new_version
        except ConflictError:
            raise
        except Exception as exc:
            raise _wrap_error(exc) from exc

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
        self._ensure_open()
        conditions = ["user_id = :uid"]
        params: dict = {"uid": user_id, "lim": limit, "off": offset}
        if not include_deleted:
            conditions.append("deleted_at IS NULL")
        if search:
            escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append("title ILIKE :search ESCAPE '\\'")
            params["search"] = f"%{escaped}%"
        if status is not None:
            conditions.append("status = :status")
            params["status"] = status
        where = " AND ".join(conditions)
        try:
            async with self._sm() as session:
                rows = (
                    await session.execute(
                        text(
                            f"SELECT chat_id, user_id, title, preview, "
                            f"       created_at, updated_at, deleted_at, version "
                            f"FROM {self._ns}.chat_meta "
                            f"WHERE {where} "
                            f"ORDER BY updated_at DESC "
                            f"LIMIT :lim OFFSET :off"
                        ),
                        params,
                    )
                ).all()
        except Exception as exc:
            raise _wrap_error(exc) from exc
        return [
            ChatMeta(
                chat_id=r.chat_id,
                user_id=r.user_id,
                title=r.title,
                preview=r.preview,
                created_at=r.created_at,
                updated_at=r.updated_at,
                deleted_at=r.deleted_at,
                version=r.version,
            )
            for r in rows
        ]

    async def load_chats_for_user(
        self,
        user_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        search: str | None = None,
    ) -> list[ChatDocument]:
        """Single JOIN query returning full ChatDocuments for a user."""
        self._ensure_open()
        conditions = ["m.user_id = :uid", "m.deleted_at IS NULL"]
        params: dict = {"uid": user_id, "lim": limit, "off": offset}
        if search:
            escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append("m.title ILIKE :search ESCAPE '\\'")
            params["search"] = f"%{escaped}%"
        if status is not None:
            conditions.append("m.status = :status")
            params["status"] = status
        where = " AND ".join(conditions)
        try:
            async with self._sm() as session:
                rows = (
                    await session.execute(
                        text(
                            f"SELECT m.chat_id, m.user_id, m.title, m.preview, "
                            f"       m.created_at, m.updated_at, m.deleted_at, m.version, "
                            f"       s.state "
                            f"FROM {self._ns}.chat_meta m "
                            f"LEFT JOIN {self._ns}.chat_state s USING (chat_id) "
                            f"WHERE {where} "
                            f"ORDER BY m.updated_at DESC "
                            f"LIMIT :lim OFFSET :off"
                        ),
                        params,
                    )
                ).all()
        except Exception as exc:
            raise _wrap_error(exc) from exc
        result = []
        for r in rows:
            meta = ChatMeta(
                chat_id=r.chat_id,
                user_id=r.user_id,
                title=r.title,
                preview=r.preview,
                created_at=r.created_at,
                updated_at=r.updated_at,
                deleted_at=r.deleted_at,
                version=r.version,
            )
            raw_state = r.state if r.state is not None else {"schema_version": 1}
            state = ChatState.model_validate(raw_state)
            result.append(ChatDocument(meta=meta, state=state))
        return result

    # -- User document -----------------------------------------------

    async def load_user_doc(self, user_id: str) -> UserDocument | None:
        self._ensure_open()
        try:
            async with self._sm() as session:
                row = (
                    await session.execute(
                        text(
                            f"SELECT user_id, created_at, updated_at, state "
                            f"FROM {self._ns}.user_documents WHERE user_id = :uid"
                        ),
                        {"uid": user_id},
                    )
                ).first()
        except Exception as exc:
            raise _wrap_error(exc) from exc

        if row is None:
            return None
        data = row.state or {}
        return UserDocument(
            user_id=row.user_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
            profile=data.get("profile", {}),
            preferences=data.get("preferences", {}),
            data_sources=data.get("data_sources", []),
        )

    async def write_user_doc(self, doc: UserDocument) -> None:
        self._ensure_open()
        payload = _dumps(
            {
                "profile": doc.profile,
                "preferences": doc.preferences,
                "data_sources": doc.data_sources,
            }
        )
        try:
            async with self._sm() as session:
                async with session.begin():
                    await session.execute(
                        text(
                            f"INSERT INTO {self._ns}.user_documents "
                            f"  (user_id, created_at, updated_at, state) "
                            f"VALUES (:uid, :ca, now(), CAST(:state AS jsonb)) "
                            f"ON CONFLICT (user_id) DO UPDATE SET "
                            f"  updated_at = now(), "
                            f"  state = EXCLUDED.state"
                        ),
                        {"uid": doc.user_id, "ca": doc.created_at, "state": payload},
                    )
        except Exception as exc:
            raise _wrap_error(exc) from exc

    # -- Prep-job document -------------------------------------------

    async def load_prep_job(self, job_id: UUID) -> PrepJobDocument | None:
        self._ensure_open()
        try:
            async with self._sm() as session:
                row = (
                    await session.execute(
                        text(
                            f"SELECT prep_job_id, account_id, status, heartbeat, "
                            f"       created_at, updated_at, state "
                            f"FROM {self._ns}.prep_job_documents WHERE prep_job_id = :jid"
                        ),
                        {"jid": job_id},
                    )
                ).first()
        except Exception as exc:
            raise _wrap_error(exc) from exc

        if row is None:
            return None
        data = row.state or {}
        return PrepJobDocument(
            prep_job_id=row.prep_job_id,
            account_id=row.account_id,
            status=row.status,
            heartbeat=row.heartbeat,
            created_at=row.created_at,
            updated_at=row.updated_at,
            query=data.get("query", ""),
            result=data.get("result", {}),
            worker=data.get("worker"),
            timings=data.get("timings", {}),
        )

    async def write_prep_job(self, doc: PrepJobDocument) -> None:
        self._ensure_open()
        payload = _dumps(
            {
                "query": doc.query,
                "result": doc.result,
                "worker": doc.worker,
                "timings": doc.timings,
            }
        )
        try:
            async with self._sm() as session:
                async with session.begin():
                    await session.execute(
                        text(
                            f"INSERT INTO {self._ns}.prep_job_documents "
                            f"  (prep_job_id, account_id, status, heartbeat, "
                            f"   created_at, updated_at, state) "
                            f"VALUES (:jid, :acc, :st, :hb, :ca, now(), CAST(:state AS jsonb)) "
                            f"ON CONFLICT (prep_job_id) DO UPDATE SET "
                            f"  account_id = EXCLUDED.account_id, "
                            f"  status = EXCLUDED.status, "
                            f"  heartbeat = EXCLUDED.heartbeat, "
                            f"  updated_at = now(), "
                            f"  state = EXCLUDED.state"
                        ),
                        {
                            "jid": doc.prep_job_id,
                            "acc": doc.account_id,
                            "st": doc.status,
                            "hb": doc.heartbeat,
                            "ca": doc.created_at,
                            "state": payload,
                        },
                    )
        except Exception as exc:
            raise _wrap_error(exc) from exc

    async def write_prep_heartbeat(self, job_id: UUID, ts: datetime) -> None:
        self._ensure_open()
        try:
            async with self._sm() as session:
                async with session.begin():
                    result = await session.execute(
                        text(
                            f"UPDATE {self._ns}.prep_job_documents SET heartbeat = :ts "
                            f"WHERE prep_job_id = :jid"
                        ),
                        {"ts": ts, "jid": job_id},
                    )
                    if result.rowcount == 0:
                        raise PermanentError(
                            f"prep_job {job_id} does not exist"
                        )
        except PermanentError:
            raise
        except Exception as exc:
            raise _wrap_error(exc) from exc

    # -- List tables --------------------------------------------------

    async def list_rows(
        self,
        table: str,
        where: Mapping[str, Any],
        *,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_open()
        _assert_safe_identifier(table)
        where_sql, params = _compile_where(where)
        order_sql = ""
        if order_by is not None:
            col = order_by.lstrip("-")
            _assert_safe_identifier(col)
            direction = "DESC" if order_by.startswith("-") else "ASC"
            order_sql = f" ORDER BY {col} {direction}"
        limit_sql = ""
        if limit is not None:
            params["_limit"] = int(limit)
            limit_sql = " LIMIT :_limit"
        sql = f"SELECT * FROM {self._ns}.{table}{where_sql}{order_sql}{limit_sql}"
        try:
            async with self._sm() as session:
                rows = (await session.execute(text(sql), params)).mappings().all()
        except Exception as exc:
            raise _wrap_error(exc) from exc
        return [dict(r) for r in rows]

    async def upsert_row(
        self,
        table: str,
        row: Mapping[str, Any],
        *,
        pk: str,
    ) -> None:
        self._ensure_open()
        _assert_safe_identifier(table)
        _assert_safe_identifier(pk)
        cols = list(row.keys())
        for c in cols:
            _assert_safe_identifier(c)
        cols_sql = ", ".join(cols)
        placeholders = ", ".join(f":{c}" for c in cols)
        update_sql = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in cols if c != pk
        )
        sql = (
            f"INSERT INTO {self._ns}.{table} ({cols_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT ({pk}) DO UPDATE SET {update_sql}"
            if update_sql
            else (
                f"INSERT INTO {self._ns}.{table} ({cols_sql}) VALUES ({placeholders}) "
                f"ON CONFLICT ({pk}) DO NOTHING"
            )
        )
        try:
            async with self._sm() as session:
                async with session.begin():
                    await session.execute(
                        text(sql),
                        {k: _coerce_jsonish(v) for k, v in row.items()},
                    )
        except Exception as exc:
            raise _wrap_error(exc) from exc

    async def delete_row(
        self,
        table: str,
        pk_value: Any,
        *,
        pk: str,
    ) -> None:
        self._ensure_open()
        _assert_safe_identifier(table)
        _assert_safe_identifier(pk)
        try:
            async with self._sm() as session:
                async with session.begin():
                    await session.execute(
                        text(f"DELETE FROM {self._ns}.{table} WHERE {pk} = :pk"),
                        {"pk": pk_value},
                    )
        except Exception as exc:
            raise _wrap_error(exc) from exc

    # -- Append-only tables -----------------------------------------

    async def append_events(
        self,
        table: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self._ensure_open()
        if not rows:
            return
        _assert_safe_identifier(table)
        # Collect the column union across all rows (events may carry slightly
        # different fields). Missing columns resolve to SQL NULL.
        cols: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    _assert_safe_identifier(k)
                    cols.append(k)
        placeholders = ", ".join(f":{c}" for c in cols)
        sql = (
            f"INSERT INTO {self._ns}.{table} ({', '.join(cols)}) "
            f"VALUES ({placeholders})"
        )
        try:
            async with self._sm() as session:
                async with session.begin():
                    await session.execute(
                        text(sql),
                        [{c: _coerce_jsonish(r.get(c)) for c in cols} for r in rows],
                    )
        except Exception as exc:
            raise _wrap_error(exc) from exc

    # -- Cleanup extension (optional; used by CleanupLoop) --------------

    async def cleanup_soft_deleted(
        self,
        *,
        chat_retention_days: int,
    ) -> "CleanupStats":
        """Delete rows for chats soft-deleted past the retention window.

        Order of operations:
        1. `file_chunks` via `chat_deleted_files` projection (for chats
           soft-deleted ≥ 1 day ago — short window so the UX "undo delete"
           still has chunks available for a day).
        2. `chat_state` / `chat_deleted_files` / `chat_meta` for chats soft-
           deleted ≥ `chat_retention_days` days ago (full purge).
        """
        from deep_research.storage.cleanup import CleanupStats

        stats = CleanupStats()
        self._ensure_open()
        async with self._sm() as session:
            async with session.begin():
                # 1) Remove orphaned file_chunks for chats that have been
                # soft-deleted ≥ 1 day. Keep chunks within the 1-day window
                # so accidental deletes can be reverted client-side.
                r = await session.execute(
                    text(
                        f"DELETE FROM {self._ns}.file_chunks "
                        f"WHERE file_id IN ( "
                        f"  SELECT file_id FROM {self._ns}.chat_deleted_files "
                        f"  WHERE chat_id IN ( "
                        f"    SELECT chat_id FROM {self._ns}.chat_meta "
                        f"    WHERE deleted_at IS NOT NULL "
                        f"      AND deleted_at < now() - INTERVAL '1 day' "
                        f"  ) "
                        f")"
                    )
                )
                stats.file_chunks_deleted = r.rowcount or 0

                # 2) Full purge past retention window.
                r = await session.execute(
                    text(
                        f"DELETE FROM {self._ns}.chat_deleted_files WHERE chat_id IN ("
                        f"  SELECT chat_id FROM {self._ns}.chat_meta "
                        f"  WHERE deleted_at IS NOT NULL "
                        f"    AND deleted_at < now() - make_interval(days => :days)"
                        f")"
                    ),
                    {"days": chat_retention_days},
                )
                stats.chat_deleted_files_rows_deleted = r.rowcount or 0

                r = await session.execute(
                    text(
                        f"DELETE FROM {self._ns}.chat_state WHERE chat_id IN ("
                        f"  SELECT chat_id FROM {self._ns}.chat_meta "
                        f"  WHERE deleted_at IS NOT NULL "
                        f"    AND deleted_at < now() - make_interval(days => :days)"
                        f")"
                    ),
                    {"days": chat_retention_days},
                )
                stats.chat_state_rows_deleted = r.rowcount or 0

                r = await session.execute(
                    text(
                        f"DELETE FROM {self._ns}.chat_meta "
                        f"WHERE deleted_at IS NOT NULL "
                        f"  AND deleted_at < now() - make_interval(days => :days)"
                    ),
                    {"days": chat_retention_days},
                )
                stats.chat_meta_rows_deleted = r.rowcount or 0
        return stats

    async def read_chunk(
        self,
        file_id: UUID,
        chunk_index: int | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_open()
        params: dict[str, Any] = {"fid": file_id}
        filter_sql = "file_id = :fid"
        if chunk_index is not None:
            filter_sql += " AND chunk_index = :idx"
            params["idx"] = chunk_index
        try:
            async with self._sm() as session:
                rows = (
                    await session.execute(
                        text(
                            f"SELECT * FROM {self._ns}.file_chunks "
                            f"WHERE {filter_sql} ORDER BY chunk_index"
                        ),
                        params,
                    )
                ).mappings().all()
        except Exception as exc:
            raise _wrap_error(exc) from exc
        return [dict(r) for r in rows]


# --- helpers ----------------------------------------------------------------


_SAFE_IDENT = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _assert_safe_identifier(ident: str) -> None:
    """Reject identifiers that aren't plain snake_case/alphanumeric.

    Column and table names go into SQL without parameter binding. This guard
    prevents the generic list/upsert helpers from being turned into an
    injection vector if a caller passes untrusted input.
    """
    if not ident or not all(c in _SAFE_IDENT for c in ident):
        raise PermanentError(f"unsafe SQL identifier: {ident!r}")


def _compile_where(where: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Compile a simple equality map into a `WHERE` clause + bound params."""
    if not where:
        return "", {}
    clauses: list[str] = []
    params: dict[str, Any] = {}
    for i, (k, v) in enumerate(where.items()):
        _assert_safe_identifier(k)
        pname = f"w{i}"
        clauses.append(f"{k} = :{pname}")
        params[pname] = v
    return " WHERE " + " AND ".join(clauses), params


def _split_sql(sql: str) -> list[str]:
    """Split a DDL file into individual statements.

    Handles simple `;`-terminated statements and ignores blank lines and
    `--` single-line comments. Does not attempt to handle `$$`-quoted
    blocks — the DDL file sticks to plain statements.
    """
    statements: list[str] = []
    buf: list[str] = []
    for raw_line in sql.splitlines():
        line = raw_line.split("--", 1)[0].rstrip()
        if not line.strip():
            continue
        buf.append(line)
        if line.endswith(";"):
            statement = " ".join(buf).rstrip(";").strip()
            if statement:
                statements.append(statement)
            buf.clear()
    tail = " ".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements
