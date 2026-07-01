"""StorageBackend over the Databricks SDK StatementExecution API.

Single-worker assumption: the two-statement version-gated write path (SELECT
then UPDATE/INSERT) is **not** atomic on Delta — see ADR follow-up F1. For
the current single-worker deployment shape this is safe; a future multi-worker
rollout must either add an advisory-lock table or move the version column
into `chat_state`.

Every SQL literal flows through `param_codec`; there is zero string
interpolation of user-controlled data. The only templated substring is
`${ns}` which resolves to the fully-qualified catalog.schema namespace.

Blocking SDK calls run under `asyncio.to_thread` so the event loop is never
blocked by the 1–3 s warehouse round-trip.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

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
    ResearchSessionState,
    UserDocument,
)
from deep_research.storage.observability import get_sink
from deep_research.storage.param_codec import params as codec_params

if TYPE_CHECKING:  # pragma: no cover
    from databricks.sdk import WorkspaceClient

    from deep_research.storage.cleanup import CleanupStats

logger = logging.getLogger(__name__)


_TRANSIENT_SDK_ERROR_CODES = frozenset(
    {
        "WAREHOUSE_UNAVAILABLE",
        "WAREHOUSE_NOT_FOUND",  # can be transient during warehouse start
        "WAREHOUSE_NOT_READY",
        "TIMEOUT",
        "REQUEST_TIMEOUT",
        "THROTTLED",
        "TOO_MANY_REQUESTS",
        "TEMPORARILY_UNAVAILABLE",
        "INTERNAL_ERROR",
        "SERVICE_UNAVAILABLE",
    }
)

_TRANSIENT_MESSAGE_MARKERS = (
    "warehouse is stopped",
    "warehouse starting",
    "timeout",
    "temporarily unavailable",
    "internal error",
    "service unavailable",
    "connection reset",
    "connection refused",
    "statement execution timed out",
)

# `_SAFE_IDENT` restricts user-supplied identifiers to plain snake_case.
_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _assert_safe_identifier(name: str) -> None:
    if not name or not _SAFE_IDENT.match(name):
        raise PermanentError(f"unsafe SQL identifier: {name!r}")


# --- Polling config ---------------------------------------------------------


@dataclass(frozen=True)
class _PollConfig:
    initial_wait_sec: float = 10.0     # sent to SDK's wait_timeout
    poll_interval_sec: float = 1.0      # between get_statement polls
    max_total_sec: float = 30.0         # overall timeout including poll


# --- Result wrappers --------------------------------------------------------


@dataclass
class _ExecResult:
    """Parsed result of a successful statement execution."""

    columns: list[str]
    rows: list[dict[str, Any]]
    num_affected_rows: int | None = None


# --- Backend ----------------------------------------------------------------


class SQLWarehouseBackend:
    """Databricks SQL Warehouse backend — Protocol-compliant with `StorageBackend`."""

    def __init__(
        self,
        *,
        client: WorkspaceClient | None = None,
        warehouse_id: str,
        catalog: str,
        schema: str,
        timeout_sec: float = 30.0,
        poll_interval_sec: float = 1.0,
        ddl_path: Path | None = None,
    ) -> None:
        _assert_safe_identifier(catalog)
        _assert_safe_identifier(schema)
        self._client = client  # lazy: constructed on first use if None
        self._warehouse_id = warehouse_id
        self._catalog = catalog
        self._schema = schema
        self._ns = f"{catalog}.{schema}"
        self._poll = _PollConfig(
            poll_interval_sec=poll_interval_sec, max_total_sec=timeout_sec
        )
        self._ddl_path = ddl_path or Path(__file__).with_name("sql_warehouse_ddl.sql")
        self._closed = False
        self._labels = {"backend": "sql_warehouse"}

    # -- Lifecycle -----------------------------------------------------

    async def migrate(self) -> None:
        try:
            ddl = self._ddl_path.read_text()
        except OSError as exc:
            raise SchemaError(f"cannot read {self._ddl_path}: {exc}") from exc

        for stmt in _split_sql(ddl):
            resolved = self._substitute_ns(stmt)
            try:
                await self._execute(resolved)
            except PermanentError as exc:
                raise SchemaError(f"DDL failed: {stmt[:120]}…: {exc}") from exc
            except TransientError as exc:
                raise SchemaError(f"DDL transient failure: {exc}") from exc

    async def close(self) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise PermanentError("SQLWarehouseBackend is closed")

    # -- Chat document ------------------------------------------------

    async def load_chat(self, chat_id: UUID) -> ChatDocument | None:
        self._ensure_open()
        result = await self._execute(
            "SELECT m.chat_id, m.user_id, m.title, m.preview, "
            "       m.created_at, m.updated_at, m.deleted_at, m.version, "
            "       s.state "
            "FROM ${ns}.chat_meta m "
            "LEFT JOIN ${ns}.chat_state s USING (chat_id) "
            "WHERE m.chat_id = :cid AND m.deleted_at IS NULL "
            "LIMIT 1",
            {"cid": str(chat_id)},
        )
        if not result.rows:
            return None
        r = result.rows[0]
        meta = ChatMeta(
            chat_id=UUID(r["chat_id"]),
            user_id=r["user_id"],
            title=r.get("title") or "",
            preview=r.get("preview") or "",
            created_at=_as_datetime(r["created_at"]),
            updated_at=_as_datetime(r["updated_at"]),
            deleted_at=_as_datetime(r["deleted_at"]) if r.get("deleted_at") else None,
            version=int(r["version"]),
        )
        raw_state = r.get("state")
        state = ChatState.model_validate_json(raw_state) if raw_state else ChatState()
        return ChatDocument(meta=meta, state=state)

    async def write_chat(
        self,
        doc: ChatDocument,
        *,
        expected_version: int,
    ) -> int:
        self._ensure_open()
        cid = str(doc.meta.chat_id)

        # Step 1: Version check. SELECT-then-act is not atomic on Delta; see
        # module docstring on the single-worker assumption.
        check = await self._execute(
            "SELECT version FROM ${ns}.chat_meta WHERE chat_id = :cid LIMIT 1",
            {"cid": cid},
        )
        exists = bool(check.rows)
        current_version = int(check.rows[0]["version"]) if exists else 0
        if current_version != expected_version:
            raise ConflictError(
                f"version mismatch on chat {cid}: "
                f"expected {expected_version}, found {current_version}"
            )
        new_version = current_version + 1

        # Step 2: MERGE chat_state (idempotent upsert).
        await self._execute(
            "MERGE INTO ${ns}.chat_state t "
            "USING (SELECT :cid AS chat_id, :state AS state) s "
            "ON t.chat_id = s.chat_id "
            "WHEN MATCHED THEN UPDATE SET state = s.state "
            "WHEN NOT MATCHED THEN INSERT (chat_id, state) VALUES (s.chat_id, s.state)",
            {"cid": cid, "state": doc.state.model_dump_json()},
        )

        # Step 3: Upsert chat_meta — UPDATE for existing, INSERT for fresh.
        if exists:
            await self._execute(
                "UPDATE ${ns}.chat_meta "
                "SET user_id = :uid, title = :title, preview = :preview, "
                "    updated_at = :updated_at, deleted_at = :deleted_at, "
                "    version = :new_version "
                "WHERE chat_id = :cid",
                {
                    "cid": cid,
                    "uid": doc.meta.user_id,
                    "title": doc.meta.title,
                    "preview": doc.meta.preview,
                    "updated_at": doc.meta.updated_at,
                    "deleted_at": doc.meta.deleted_at,
                    "new_version": new_version,
                },
            )
        else:
            await self._execute(
                "INSERT INTO ${ns}.chat_meta "
                "  (chat_id, user_id, title, preview, created_at, updated_at, "
                "   deleted_at, version) "
                "VALUES "
                "  (:cid, :uid, :title, :preview, :created_at, :updated_at, "
                "   :deleted_at, :version)",
                {
                    "cid": cid,
                    "uid": doc.meta.user_id,
                    "title": doc.meta.title,
                    "preview": doc.meta.preview,
                    "created_at": doc.meta.created_at,
                    "updated_at": doc.meta.updated_at,
                    "deleted_at": doc.meta.deleted_at,
                    "version": new_version,
                },
            )

        # Step 4: Rebuild chat_deleted_files projection (full-overwrite semantics).
        await self._execute(
            "DELETE FROM ${ns}.chat_deleted_files WHERE chat_id = :cid",
            {"cid": cid},
        )
        file_ids = doc.state.live_file_ids()
        if file_ids:
            placeholders = ", ".join(f"(:cid, :fid{i})" for i in range(len(file_ids)))
            params: dict[str, Any] = {"cid": cid}
            for i, fid in enumerate(file_ids):
                params[f"fid{i}"] = str(fid)
            await self._execute(
                f"INSERT INTO ${{ns}}.chat_deleted_files (chat_id, file_id) VALUES {placeholders}",
                params,
            )
        return new_version

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
        params: dict[str, Any] = {"uid": user_id, "lim": limit, "off": offset}
        if not include_deleted:
            conditions.append("deleted_at IS NULL")
        if search:
            # Delta: case-insensitive via LOWER() on both sides
            escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append("LOWER(title) LIKE LOWER(:search)")
            params["search"] = f"%{escaped}%"
        if status is not None:
            conditions.append("status = :status")
            params["status"] = status
        where = " AND ".join(conditions)
        result = await self._execute(
            f"SELECT chat_id, user_id, title, preview, created_at, updated_at, "
            f"       deleted_at, version "
            f"FROM ${{ns}}.chat_meta "
            f"WHERE {where} "
            f"ORDER BY updated_at DESC "
            f"LIMIT :lim OFFSET :off",
            params,
        )
        return [
            ChatMeta(
                chat_id=UUID(r["chat_id"]),
                user_id=r["user_id"],
                title=r.get("title") or "",
                preview=r.get("preview") or "",
                created_at=_as_datetime(r["created_at"]),
                updated_at=_as_datetime(r["updated_at"]),
                deleted_at=_as_datetime(r["deleted_at"]) if r.get("deleted_at") else None,
                version=int(r["version"]),
            )
            for r in result.rows
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
        params: dict[str, Any] = {"uid": user_id, "lim": limit, "off": offset}
        if search:
            escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append("LOWER(m.title) LIKE LOWER(:search)")
            params["search"] = f"%{escaped}%"
        if status is not None:
            conditions.append("m.status = :status")
            params["status"] = status
        where = " AND ".join(conditions)
        result = await self._execute(
            f"SELECT m.chat_id, m.user_id, m.title, m.preview, "
            f"       m.created_at, m.updated_at, m.deleted_at, m.version, "
            f"       s.state "
            f"FROM ${{ns}}.chat_meta m "
            f"LEFT JOIN ${{ns}}.chat_state s USING (chat_id) "
            f"WHERE {where} "
            f"ORDER BY m.updated_at DESC "
            f"LIMIT :lim OFFSET :off",
            params,
        )
        docs = []
        for r in result.rows:
            meta = ChatMeta(
                chat_id=UUID(r["chat_id"]),
                user_id=r["user_id"],
                title=r.get("title") or "",
                preview=r.get("preview") or "",
                created_at=_as_datetime(r["created_at"]),
                updated_at=_as_datetime(r["updated_at"]),
                deleted_at=_as_datetime(r["deleted_at"]) if r.get("deleted_at") else None,
                version=int(r["version"]),
            )
            raw_state = r.get("state")
            state = ChatState.model_validate_json(raw_state) if raw_state else ChatState()
            docs.append(ChatDocument(meta=meta, state=state))
        return docs

    # -- User document ------------------------------------------------

    async def load_user_doc(self, user_id: str) -> UserDocument | None:
        self._ensure_open()
        result = await self._execute(
            "SELECT user_id, created_at, updated_at, state "
            "FROM ${ns}.user_documents WHERE user_id = :uid LIMIT 1",
            {"uid": user_id},
        )
        if not result.rows:
            return None
        r = result.rows[0]
        data = _json_or_empty(r.get("state"))
        return UserDocument(
            user_id=r["user_id"],
            created_at=_as_datetime(r["created_at"]),
            updated_at=_as_datetime(r["updated_at"]),
            profile=data.get("profile", {}),
            preferences=data.get("preferences", {}),
            data_sources=data.get("data_sources", []),
        )

    async def write_user_doc(self, doc: UserDocument) -> None:
        self._ensure_open()
        payload = json.dumps(
            {
                "profile": doc.profile,
                "preferences": doc.preferences,
                "data_sources": doc.data_sources,
            },
            default=_json_default,
            sort_keys=True,
        )
        await self._execute(
            "MERGE INTO ${ns}.user_documents t "
            "USING (SELECT :uid AS user_id, :ca AS created_at, "
            "              :ua AS updated_at, :state AS state) s "
            "ON t.user_id = s.user_id "
            "WHEN MATCHED THEN UPDATE SET "
            "  updated_at = s.updated_at, state = s.state "
            "WHEN NOT MATCHED THEN INSERT "
            "  (user_id, created_at, updated_at, state) "
            "  VALUES (s.user_id, s.created_at, s.updated_at, s.state)",
            {
                "uid": doc.user_id,
                "ca": doc.created_at,
                "ua": doc.updated_at,
                "state": payload,
            },
        )

    # -- Prep-job document --------------------------------------------

    async def load_prep_job(self, job_id: UUID) -> PrepJobDocument | None:
        self._ensure_open()
        result = await self._execute(
            "SELECT prep_job_id, account_id, status, heartbeat, created_at, "
            "       updated_at, state "
            "FROM ${ns}.prep_job_documents WHERE prep_job_id = :jid LIMIT 1",
            {"jid": str(job_id)},
        )
        if not result.rows:
            return None
        r = result.rows[0]
        data = _json_or_empty(r.get("state"))
        return PrepJobDocument(
            prep_job_id=UUID(r["prep_job_id"]),
            account_id=r["account_id"],
            status=r["status"],
            heartbeat=_as_datetime(r["heartbeat"]),
            created_at=_as_datetime(r["created_at"]),
            updated_at=_as_datetime(r["updated_at"]),
            query=data.get("query", ""),
            result=data.get("result", {}),
            worker=data.get("worker"),
            timings=data.get("timings", {}),
        )

    async def write_prep_job(self, doc: PrepJobDocument) -> None:
        self._ensure_open()
        payload = json.dumps(
            {
                "query": doc.query,
                "result": doc.result,
                "worker": doc.worker,
                "timings": doc.timings,
            },
            default=_json_default,
            sort_keys=True,
        )
        await self._execute(
            "MERGE INTO ${ns}.prep_job_documents t "
            "USING (SELECT :jid AS prep_job_id, :acc AS account_id, :st AS status, "
            "              :hb AS heartbeat, :ca AS created_at, :ua AS updated_at, "
            "              :state AS state) s "
            "ON t.prep_job_id = s.prep_job_id "
            "WHEN MATCHED THEN UPDATE SET "
            "  account_id = s.account_id, status = s.status, "
            "  heartbeat = s.heartbeat, updated_at = s.updated_at, state = s.state "
            "WHEN NOT MATCHED THEN INSERT "
            "  (prep_job_id, account_id, status, heartbeat, created_at, "
            "   updated_at, state) "
            "  VALUES (s.prep_job_id, s.account_id, s.status, s.heartbeat, "
            "          s.created_at, s.updated_at, s.state)",
            {
                "jid": str(doc.prep_job_id),
                "acc": doc.account_id,
                "st": doc.status,
                "hb": doc.heartbeat,
                "ca": doc.created_at,
                "ua": doc.updated_at,
                "state": payload,
            },
        )

    async def write_prep_heartbeat(self, job_id: UUID, ts: datetime) -> None:
        self._ensure_open()
        result = await self._execute(
            "UPDATE ${ns}.prep_job_documents SET heartbeat = :ts "
            "WHERE prep_job_id = :jid",
            {"ts": ts, "jid": str(job_id)},
        )
        if result.num_affected_rows is not None and result.num_affected_rows == 0:
            raise PermanentError(f"prep_job {job_id} does not exist")

    # -- Research session queries ------------------------------------
    #
    # Backed by ``chat_state.state.research_sessions`` JSON array (the
    # ``state`` column on Delta is stored as TEXT/JSON-string). All five
    # methods follow a read-transform pattern: fetch the candidate rows
    # for the user / chat, parse the ``state`` JSON via
    # ``ChatState.model_validate_json``, compute the result in Python.
    # Update methods (``mark_stale_research_sessions_failed``,
    # ``write_research_session_heartbeat``) then serialize back and
    # ``MERGE INTO`` ``chat_state``. **Non-atomic for array element
    # mutation**: a concurrent writer landing between the SELECT and
    # MERGE may have its update overwritten. Mitigated for cleanup by
    # the cutoff predicate (only stale sessions are touched) and the
    # exclude-list (only sessions unknown to this worker). For
    # heartbeat writes, the race window is short and the heartbeat is
    # idempotent. See plan v4 ADR pre-mortem S3 + ``project_dual_backend``
    # memory.

    async def count_active_research_sessions(self, user_id: str) -> int:
        self._ensure_open()
        result = await self._execute(
            "SELECT s.state "
            "FROM ${ns}.chat_meta m "
            "JOIN ${ns}.chat_state s USING (chat_id) "
            "WHERE m.user_id = :uid AND m.deleted_at IS NULL",
            {"uid": user_id},
        )
        count = 0
        for row in result.rows:
            raw_state = row.get("state")
            if not raw_state:
                continue
            try:
                state = ChatState.model_validate_json(raw_state)
            except Exception:
                # Defensive: malformed state should not block the count;
                # the WriteQueue's repair path handles re-serialization.
                continue
            if any(rs.status == "in_progress" for rs in state.research_sessions):
                count += 1
        return count

    async def mark_stale_research_sessions_failed(
        self,
        cutoff: datetime,
        exclude_session_ids: Sequence[UUID],
    ) -> int:
        self._ensure_open()
        exclude_ids = {str(sid) for sid in exclude_session_ids}
        # Step 1: fetch candidate rows (any chat with at least one
        # in_progress session); inexpensive heuristic via LIKE keeps the
        # candidate set small without requiring server-side JSON parsing.
        candidates = await self._execute(
            "SELECT m.chat_id, s.state "
            "FROM ${ns}.chat_meta m "
            "JOIN ${ns}.chat_state s USING (chat_id) "
            "WHERE m.deleted_at IS NULL "
            "  AND s.state LIKE '%\"status\":\"in_progress\"%'",
        )
        chats_modified = 0
        # Step 2: per-chat parse + mutate + write-back. Each chat is its
        # own MERGE; failure of one does not block the others.
        for row in candidates.rows:
            cid = row.get("chat_id")
            raw_state = row.get("state")
            if not raw_state or cid is None:
                continue
            try:
                state = ChatState.model_validate_json(raw_state)
            except Exception:
                continue
            mutated = False
            for rs in state.research_sessions:
                if rs.status != "in_progress":
                    continue
                if str(rs.id) in exclude_ids:
                    continue
                if rs.last_heartbeat is not None and rs.last_heartbeat >= cutoff:
                    continue
                rs.status = "failed"
                rs.completed_at = datetime.now(rs.completed_at.tzinfo if rs.completed_at else None)
                mutated = True
            if not mutated:
                continue
            await self._execute(
                "MERGE INTO ${ns}.chat_state t "
                "USING (SELECT :cid AS chat_id, :state AS state) s "
                "ON t.chat_id = s.chat_id "
                "WHEN MATCHED THEN UPDATE SET state = s.state",
                {"cid": str(cid), "state": state.model_dump_json()},
            )
            chats_modified += 1
        return chats_modified

    async def list_user_jobs(
        self,
        user_id: str,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[tuple[UUID, ResearchSessionState]]:
        self._ensure_open()
        result = await self._execute(
            "SELECT m.chat_id, s.state "
            "FROM ${ns}.chat_meta m "
            "JOIN ${ns}.chat_state s USING (chat_id) "
            "WHERE m.user_id = :uid AND m.deleted_at IS NULL",
            {"uid": user_id},
        )
        rows: list[tuple[UUID, ResearchSessionState]] = []
        for row in result.rows:
            cid_raw = row.get("chat_id")
            raw_state = row.get("state")
            if not raw_state or cid_raw is None:
                continue
            try:
                state = ChatState.model_validate_json(raw_state)
            except Exception:
                continue
            cid = UUID(cid_raw) if isinstance(cid_raw, str) else cid_raw
            for rs in state.research_sessions:
                if status is not None and rs.status != status:
                    continue
                rows.append((cid, rs))
        rows.sort(key=lambda pair: pair[1].started_at, reverse=True)
        return rows[: int(limit)]

    async def get_active_session_for_chat(
        self,
        chat_id: UUID,
        user_id: str,
    ) -> ResearchSessionState | None:
        self._ensure_open()
        result = await self._execute(
            "SELECT s.state "
            "FROM ${ns}.chat_meta m "
            "JOIN ${ns}.chat_state s USING (chat_id) "
            "WHERE m.chat_id = :cid AND m.user_id = :uid AND m.deleted_at IS NULL "
            "LIMIT 1",
            {"cid": str(chat_id), "uid": user_id},
        )
        if not result.rows:
            return None
        raw_state = result.rows[0].get("state")
        if not raw_state:
            return None
        try:
            state = ChatState.model_validate_json(raw_state)
        except Exception:
            return None
        active = [rs for rs in state.research_sessions if rs.status == "in_progress"]
        if not active:
            return None
        active.sort(key=lambda rs: rs.started_at, reverse=True)
        return active[0]

    async def write_research_session_heartbeat(
        self,
        chat_id: UUID,
        session_id: UUID,
        ts: datetime,
    ) -> None:
        self._ensure_open()
        # Read-transform-write: the SQL Warehouse equivalent of Lakebase's
        # in-place jsonb_set. Single-worker race window is acceptable; a
        # concurrent legitimate writer would land its own write either
        # before or after this one and the last write wins. Heartbeats
        # are idempotent — a missed heartbeat is reaped by the next
        # cleanup cycle.
        result = await self._execute(
            "SELECT state FROM ${ns}.chat_state WHERE chat_id = :cid LIMIT 1",
            {"cid": str(chat_id)},
        )
        if not result.rows:
            return
        raw_state = result.rows[0].get("state")
        if not raw_state:
            return
        try:
            state = ChatState.model_validate_json(raw_state)
        except Exception:
            return
        sid_str = str(session_id)
        mutated = False
        for rs in state.research_sessions:
            if str(rs.id) == sid_str:
                rs.last_heartbeat = ts
                mutated = True
                break
        if not mutated:
            return
        await self._execute(
            "MERGE INTO ${ns}.chat_state t "
            "USING (SELECT :cid AS chat_id, :state AS state) s "
            "ON t.chat_id = s.chat_id "
            "WHEN MATCHED THEN UPDATE SET state = s.state",
            {"cid": str(chat_id), "state": state.model_dump_json()},
        )

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
            limit_sql = f" LIMIT {int(limit)}"
        sql = f"SELECT * FROM ${{ns}}.{table}{where_sql}{order_sql}{limit_sql}"
        result = await self._execute(sql, params)
        return [dict(r) for r in result.rows]

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
        using_select = ", ".join(f":{c} AS {c}" for c in cols)
        set_clause = ", ".join(f"{c} = s.{c}" for c in cols if c != pk)
        insert_cols = ", ".join(cols)
        insert_vals = ", ".join(f"s.{c}" for c in cols)
        sql = (
            f"MERGE INTO ${{ns}}.{table} t USING (SELECT {using_select}) s "
            f"ON t.{pk} = s.{pk} "
            f"WHEN MATCHED THEN UPDATE SET {set_clause} "
            f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})"
        )
        await self._execute(sql, dict(row))

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
        await self._execute(
            f"DELETE FROM ${{ns}}.{table} WHERE {pk} = :pk",
            {"pk": pk_value},
        )

    # -- Append-only tables -------------------------------------------

    async def append_events(
        self,
        table: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self._ensure_open()
        if not rows:
            return
        _assert_safe_identifier(table)
        # Union of columns across rows — missing values bind to NULL.
        cols: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    _assert_safe_identifier(k)
                    cols.append(k)
        # One multi-row INSERT statement regardless of batch size.
        values_parts: list[str] = []
        params: dict[str, Any] = {}
        for i, r in enumerate(rows):
            row_placeholders = ", ".join(f":r{i}_{c}" for c in cols)
            values_parts.append(f"({row_placeholders})")
            for c in cols:
                params[f"r{i}_{c}"] = r.get(c)
        values_sql = ", ".join(values_parts)
        sql = (
            f"INSERT INTO ${{ns}}.{table} ({', '.join(cols)}) VALUES {values_sql}"
        )
        await self._execute(sql, params)

    # -- Cleanup extension (optional; used by CleanupLoop) --------------

    async def cleanup_soft_deleted(
        self,
        *,
        chat_retention_days: int,
    ) -> CleanupStats:
        """Delta equivalent of the Lakebase cleanup cascade.

        Passes cutoff timestamps as parameters to stay dialect-agnostic and
        avoid Delta interval-literal quirks.
        """
        from datetime import datetime, timedelta

        from deep_research.storage.cleanup import CleanupStats

        self._ensure_open()
        stats = CleanupStats()
        now = datetime.now(tz=UTC)
        short_cutoff = now - timedelta(days=1)
        long_cutoff = now - timedelta(days=chat_retention_days)

        r = await self._execute(
            "DELETE FROM ${ns}.file_chunks WHERE file_id IN ("
            "  SELECT file_id FROM ${ns}.chat_deleted_files WHERE chat_id IN ("
            "    SELECT chat_id FROM ${ns}.chat_meta "
            "    WHERE deleted_at IS NOT NULL AND deleted_at < :short "
            "  )"
            ")",
            {"short": short_cutoff},
        )
        stats.file_chunks_deleted = r.num_affected_rows or 0

        r = await self._execute(
            "DELETE FROM ${ns}.chat_deleted_files WHERE chat_id IN ("
            "  SELECT chat_id FROM ${ns}.chat_meta "
            "  WHERE deleted_at IS NOT NULL AND deleted_at < :longc "
            ")",
            {"longc": long_cutoff},
        )
        stats.chat_deleted_files_rows_deleted = r.num_affected_rows or 0

        r = await self._execute(
            "DELETE FROM ${ns}.chat_state WHERE chat_id IN ("
            "  SELECT chat_id FROM ${ns}.chat_meta "
            "  WHERE deleted_at IS NOT NULL AND deleted_at < :longc "
            ")",
            {"longc": long_cutoff},
        )
        stats.chat_state_rows_deleted = r.num_affected_rows or 0

        r = await self._execute(
            "DELETE FROM ${ns}.chat_meta "
            "WHERE deleted_at IS NOT NULL AND deleted_at < :longc",
            {"longc": long_cutoff},
        )
        stats.chat_meta_rows_deleted = r.num_affected_rows or 0

        return stats

    async def read_chunk(
        self,
        file_id: UUID,
        chunk_index: int | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_open()
        where = "file_id = :fid"
        params: dict[str, Any] = {"fid": str(file_id)}
        if chunk_index is not None:
            where += " AND chunk_index = :idx"
            params["idx"] = int(chunk_index)
        result = await self._execute(
            f"SELECT * FROM ${{ns}}.file_chunks WHERE {where} ORDER BY chunk_index",
            params,
        )
        return [dict(r) for r in result.rows]

    # -- Execute / substitution ---------------------------------------

    def _substitute_ns(self, sql: str) -> str:
        return sql.replace("${ns}", self._ns)

    def _get_client(self) -> WorkspaceClient:
        if self._client is None:
            from databricks.sdk import WorkspaceClient as _WC

            self._client = _WC()
        return self._client

    async def _execute(
        self,
        sql: str,
        params: Mapping[str, Any] | None = None,
    ) -> _ExecResult:
        """Issue a single statement and wait for completion.

        Returns an `_ExecResult` on success. Raises `TransientError` on
        retryable SDK failures and `PermanentError` otherwise. The WriteQueue
        / caller decides whether to retry.
        """
        resolved_sql = self._substitute_ns(sql)
        sink = get_sink()
        started = asyncio.get_event_loop().time()

        def _sync_call() -> Any:
            client = self._get_client()
            sdk_params = codec_params(dict(params)) if params else None
            # Initial wait_timeout caps at 50s per SDK docs; we use a smaller
            # value so asyncio.wait_for can preempt.
            initial = min(self._poll.initial_wait_sec, 50.0)
            response = client.statement_execution.execute_statement(
                statement=resolved_sql,
                warehouse_id=self._warehouse_id,
                parameters=sdk_params,
                wait_timeout=f"{int(initial)}s",
            )
            return response

        try:
            response = await asyncio.to_thread(_sync_call)
        except Exception as exc:  # noqa: BLE001
            raise _wrap_sdk_exception(exc) from exc

        # Poll until terminal state — yielding to the event loop between polls.
        total_elapsed = asyncio.get_event_loop().time() - started
        while True:
            state = _state_name(response)
            if state in ("SUCCEEDED",):
                break
            if state in ("FAILED", "CANCELED", "CLOSED"):
                raise _wrap_status_error(response)
            # PENDING or RUNNING
            if total_elapsed >= self._poll.max_total_sec:
                # Best-effort cancel.
                await asyncio.to_thread(_maybe_cancel, self._client, response.statement_id)
                raise TransientError(
                    f"statement timed out after {self._poll.max_total_sec}s"
                )
            await asyncio.sleep(self._poll.poll_interval_sec)
            try:
                response = await asyncio.to_thread(
                    self._get_client().statement_execution.get_statement,
                    response.statement_id,
                )
            except Exception as exc:  # noqa: BLE001
                raise _wrap_sdk_exception(exc) from exc
            total_elapsed = asyncio.get_event_loop().time() - started

        parsed = _parse_response(response)
        duration = asyncio.get_event_loop().time() - started
        sink.histogram(
            "storage_sql_statement_seconds",
            duration,
            backend=self._labels["backend"],
        )
        return parsed


# --- Helpers ---------------------------------------------------------------


def _split_sql(sql: str) -> list[str]:
    """Split a DDL file into individual statements (semicolon-terminated)."""
    statements: list[str] = []
    buf: list[str] = []
    for raw_line in sql.splitlines():
        line = raw_line.split("--", 1)[0].rstrip()
        if not line.strip():
            continue
        buf.append(line)
        if line.endswith(";"):
            stmt = " ".join(buf).rstrip(";").strip()
            if stmt:
                statements.append(stmt)
            buf.clear()
    tail = " ".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _compile_where(where: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
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


def _state_name(response: Any) -> str:
    """Return the SDK statement state as a plain uppercase string."""
    if response is None or response.status is None:
        return "UNKNOWN"
    state = response.status.state
    if state is None:
        return "UNKNOWN"
    return getattr(state, "value", None) or getattr(state, "name", None) or str(state)


def _wrap_sdk_exception(exc: BaseException) -> Exception:
    msg = str(exc).lower()
    # Transient markers take precedence.
    if any(marker in msg for marker in _TRANSIENT_MESSAGE_MARKERS):
        return TransientError(str(exc))
    # Fall back: treat unknown connection / HTTP errors as transient.
    exc_type = type(exc).__name__.lower()
    if any(kw in exc_type for kw in ("timeout", "connection", "network")):
        return TransientError(str(exc))
    return PermanentError(str(exc))


def _wrap_status_error(response: Any) -> Exception:
    status = getattr(response, "status", None)
    err = getattr(status, "error", None)
    code = getattr(err, "error_code", None)
    message = getattr(err, "message", None) or "statement execution failed"
    if code and str(code) in _TRANSIENT_SDK_ERROR_CODES:
        return TransientError(f"{code}: {message}")
    if any(marker in message.lower() for marker in _TRANSIENT_MESSAGE_MARKERS):
        return TransientError(message)
    return PermanentError(f"{code or 'SQL_ERROR'}: {message}")


def _maybe_cancel(client: WorkspaceClient | None, statement_id: str) -> None:
    if client is None:
        return
    try:
        client.statement_execution.cancel_execution(statement_id=statement_id)
    except Exception:  # noqa: BLE001 — best-effort
        logger.debug("cancel failed", exc_info=True)


def _parse_response(response: Any) -> _ExecResult:
    """Parse an SDK response into `_ExecResult`. Shape is defensive because
    the SDK result structure has shifted across versions."""
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    num_affected: int | None = None

    manifest = getattr(response, "manifest", None)
    if manifest is not None:
        schema = getattr(manifest, "schema", None)
        if schema is not None:
            schema_cols = getattr(schema, "columns", None) or []
            for c in schema_cols:
                name = getattr(c, "name", None)
                if name is not None:
                    columns.append(name)
        num_affected = (
            getattr(manifest, "total_affected_rows", None)
            if hasattr(manifest, "total_affected_rows")
            else None
        )

    result = getattr(response, "result", None)
    if result is not None:
        data_array = getattr(result, "data_array", None) or []
        for row in data_array:
            if columns:
                rows.append({columns[i]: row[i] for i in range(min(len(columns), len(row)))})
            else:
                rows.append({str(i): v for i, v in enumerate(row)})
        # Some SDK responses put affected rows on the top-level result.
        if num_affected is None:
            num_affected = getattr(result, "num_affected_rows", None)

    return _ExecResult(columns=columns, rows=rows, num_affected_rows=num_affected)


def _as_datetime(value: Any) -> datetime:
    """Coerce a wire value to `datetime`. SDK may deliver ISO strings."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise PermanentError(f"expected datetime, got {type(value).__name__}")


def _json_or_empty(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PermanentError(f"invalid JSON in document column: {exc}") from exc
        return parsed if isinstance(parsed, dict) else {}
    raise PermanentError(f"unexpected state column type: {type(value).__name__}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"cannot JSON-encode {type(obj).__name__}")
