"""Cache-backed `IFileUploadService` — routes file metadata + chunks through `StorageStack`.

Metadata rows live in the ``uploaded_files`` list table
(``_cold_upsert_row`` / ``_cold_list_rows`` / ``_cold_delete_row``).
File chunks are stored in the ``file_chunks`` append-only table via
``backend.append_events``, batched at most 1000 rows per call.
Chunk reads go through ``backend.read_chunk``.

File content is still written to the local filesystem (or passed as bytes by
the caller) — the cached service delegates storage-path logic to the same
filesystem helper the legacy service uses, keeping the on-disk layout
identical.

Return shape: every method returns a lightweight DTO that mirrors the legacy
``UploadedFile`` / ``FileChunk`` ORM attribute surface so all 8 call sites
work without modification.
"""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, BinaryIO
from uuid import UUID, uuid4

from deep_research.services._cached_base import _CachedServiceBase
from deep_research.services._protocols import IFileUploadService
from deep_research.services.file_upload_service import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILES_PER_SESSION,
    MAX_TOTAL_SESSION_SIZE_BYTES,
    SESSION_FILE_TTL_HOURS,
    FileType,
    get_file_type_from_extension,
    get_file_type_from_mime,
)

if TYPE_CHECKING:
    from deep_research.storage.factory import StorageStack

logger = logging.getLogger(__name__)

_FILE_TABLE = "uploaded_files"
_CHUNK_TABLE = "file_chunks"
_CHUNK_BATCH_SIZE = 1000


# ---------------------------------------------------------------------------
# View objects (legacy-compatible DTOs)
# ---------------------------------------------------------------------------


class _FileChunkView:
    """Lightweight DTO mirroring ``FileChunk`` ORM attribute surface."""

    __slots__ = ("file_id", "chunk_index", "content", "metadata_", "id")

    def __init__(
        self,
        file_id: UUID,
        chunk_index: int,
        content: str,
        metadata_: dict[str, Any] | None = None,
    ) -> None:
        self.id: UUID = uuid4()
        self.file_id = file_id
        self.chunk_index = chunk_index
        self.content = content
        self.metadata_ = metadata_ or {}


class _UploadedFileView:
    """Lightweight DTO mirroring ``UploadedFile`` ORM attribute surface."""

    __slots__ = (
        "id", "owner_id", "session_id", "filename", "file_type",
        "file_size", "storage_path", "processing_status",
        "chunk_count", "expires_at", "metadata_",
        "created_at", "updated_at",
        # computed
        "_total_extracted_chars",
    )

    def __init__(
        self,
        id: UUID,
        owner_id: str,
        filename: str,
        file_type: str,
        file_size: int,
        storage_path: str,
        processing_status: str = "pending",
        session_id: UUID | None = None,
        chunk_count: int = 0,
        expires_at: datetime | None = None,
        metadata_: dict[str, Any] | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        self.id = id
        self.owner_id = owner_id
        self.session_id = session_id
        self.filename = filename
        self.file_type = file_type
        self.file_size = file_size
        self.storage_path = storage_path
        self.processing_status = processing_status
        self.chunk_count = chunk_count
        self.expires_at = expires_at
        self.metadata_: dict[str, Any] = metadata_ or {}
        now = datetime.now(UTC)
        self.created_at = created_at or now
        self.updated_at = updated_at or now
        self._total_extracted_chars = 0

    @property
    def type(self) -> FileType:
        return FileType(self.file_type)

    @property
    def is_ready(self) -> bool:
        return self.processing_status == "ready"

    @property
    def is_failed(self) -> bool:
        return self.processing_status == "failed"

    def mark_processing(self) -> None:
        self.processing_status = "processing"
        self.updated_at = datetime.now(UTC)

    def mark_ready(
        self,
        chunk_count: int,
        total_extracted_chars: int = 0,
    ) -> None:
        self.processing_status = "ready"
        self.chunk_count = chunk_count
        self._total_extracted_chars = total_extracted_chars
        self.updated_at = datetime.now(UTC)

    def mark_failed(self, error: str) -> None:
        self.processing_status = "failed"
        self.metadata_["error"] = error
        self.updated_at = datetime.now(UTC)


# ---------------------------------------------------------------------------
# Row serialisation helpers
# ---------------------------------------------------------------------------


def _row_to_file_view(row: dict[str, Any]) -> _UploadedFileView:
    raw_id = row["id"]
    file_id = raw_id if isinstance(raw_id, UUID) else UUID(str(raw_id))
    raw_session = row.get("session_id")
    session_id = None
    if raw_session is not None:
        session_id = raw_session if isinstance(raw_session, UUID) else UUID(str(raw_session))
    raw_expires = row.get("expires_at")
    expires_at: datetime | None = None
    if raw_expires:
        expires_at = (
            raw_expires
            if isinstance(raw_expires, datetime)
            else datetime.fromisoformat(str(raw_expires))
        )
    raw_created = row.get("created_at")
    created_at: datetime | None = None
    if raw_created:
        created_at = (
            raw_created
            if isinstance(raw_created, datetime)
            else datetime.fromisoformat(str(raw_created))
        )
    raw_updated = row.get("updated_at")
    updated_at: datetime | None = None
    if raw_updated:
        updated_at = (
            raw_updated
            if isinstance(raw_updated, datetime)
            else datetime.fromisoformat(str(raw_updated))
        )
    return _UploadedFileView(
        id=file_id,
        owner_id=str(row["owner_id"]),
        session_id=session_id,
        filename=str(row.get("filename", "")),
        file_type=str(row.get("file_type", "txt")),
        file_size=int(row.get("file_size", 0)),
        storage_path=str(row.get("storage_path", "")),
        processing_status=str(row.get("processing_status", "pending")),
        chunk_count=int(row.get("chunk_count", 0)),
        expires_at=expires_at,
        metadata_=row.get("metadata_") or {},
        created_at=created_at,
        updated_at=updated_at,
    )


def _file_view_to_row(f: _UploadedFileView) -> dict[str, Any]:
    # Pass datetime objects through unchanged. Both backends handle them:
    # - Lakebase (asyncpg) binds them directly to TIMESTAMPTZ columns
    # - SQL Warehouse's param_codec._encode_value isinstance-checks datetime
    #   and serializes with the TIMESTAMP type hint
    # Pre-serializing via isoformat() breaks asyncpg's timestamptz codec.
    return {
        "id": str(f.id),
        "owner_id": f.owner_id,
        "session_id": str(f.session_id) if f.session_id else None,
        "filename": f.filename,
        "file_type": f.file_type,
        "file_size": f.file_size,
        "storage_path": f.storage_path,
        "processing_status": f.processing_status,
        "chunk_count": f.chunk_count,
        "expires_at": f.expires_at,
        "metadata_": f.metadata_,
        "created_at": f.created_at,
        "updated_at": f.updated_at,
    }


def _chunk_row_to_view(row: dict[str, Any]) -> _FileChunkView:
    raw_id = row.get("file_id")
    file_id = raw_id if isinstance(raw_id, UUID) else UUID(str(raw_id))
    # DB column is `metadata` (no underscore). The DTO attribute keeps the
    # trailing-underscore form for parity with the legacy ORM surface.
    view = _FileChunkView(
        file_id=file_id,
        chunk_index=int(row.get("chunk_index", 0)),
        content=str(row.get("content", "")),
        metadata_=row.get("metadata") or {},
    )
    return view


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CachedFileUploadService(_CachedServiceBase, IFileUploadService):
    """``IFileUploadService`` backed by ``StorageStack`` list + append tables."""

    _service_name = "file_upload"

    def __init__(
        self,
        stack: StorageStack,
        storage_path: str | None = None,
    ) -> None:
        super().__init__(stack)
        self._storage_path = storage_path or tempfile.gettempdir()

    # -- Validation (mirrors legacy service) ---------------------------------

    def validate_file(
        self,
        filename: str,
        file_size: int,
        content_type: str | None = None,
    ) -> tuple[bool, str | None, FileType | None]:
        if file_size > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
            return False, f"File exceeds maximum size of {max_mb}MB", None
        if file_size == 0:
            return False, "File is empty", None
        file_type = get_file_type_from_extension(filename)
        if file_type is None and content_type:
            file_type = get_file_type_from_mime(content_type)
        if file_type is None:
            from deep_research.services.file_upload_service import SUPPORTED_FILE_TYPES
            supported = ", ".join(
                ext for exts in SUPPORTED_FILE_TYPES.values() for ext in exts
            )
            return False, f"Unsupported file type. Supported: {supported}", None
        return True, None, file_type

    async def validate_session_quota(
        self,
        owner_id: str,
        session_id: UUID | None,
        new_file_size: int,
    ) -> tuple[bool, str | None]:
        if session_id is None:
            return True, None
        rows = await self._cold_list_rows(
            _FILE_TABLE, {"owner_id": owner_id, "session_id": str(session_id)}
        )
        file_count = len(rows)
        if file_count >= MAX_FILES_PER_SESSION:
            return False, f"Maximum {MAX_FILES_PER_SESSION} files per session"
        total_size = sum(int(r.get("file_size", 0)) for r in rows)
        if total_size + new_file_size > MAX_TOTAL_SESSION_SIZE_BYTES:
            max_mb = MAX_TOTAL_SESSION_SIZE_BYTES / (1024 * 1024)
            return False, f"Total session files exceed {max_mb}MB limit"
        return True, None

    # -- Upload and Storage --------------------------------------------------

    async def upload_file(
        self,
        owner_id: str,
        filename: str,
        file_content: BinaryIO | bytes,
        file_size: int,
        session_id: UUID | None = None,
        content_type: str | None = None,
    ) -> tuple[_UploadedFileView | None, str | None]:
        is_valid, error, file_type = self.validate_file(filename, file_size, content_type)
        if not is_valid or file_type is None:
            return None, error

        is_valid, error = await self.validate_session_quota(owner_id, session_id, file_size)
        if not is_valid:
            return None, error

        file_id = uuid4()
        storage_dir = Path(self._storage_path) / "uploads" / owner_id
        storage_dir.mkdir(parents=True, exist_ok=True)

        base_name = PurePosixPath(filename).name
        sanitized = re.sub(r"[^\w.\-]", "_", base_name)[:200]
        if not sanitized or sanitized.startswith("."):
            sanitized = f"upload{sanitized}"
        safe_filename = f"{file_id}_{sanitized}"
        storage_path = storage_dir / safe_filename

        try:
            if isinstance(file_content, bytes):
                storage_path.write_bytes(file_content)
            else:
                with storage_path.open("wb") as f:
                    shutil.copyfileobj(file_content, f)
        except OSError as e:
            logger.error("Failed to write file: %s", str(e))
            return None, f"Failed to store file: {e}"

        expires_at = None
        if session_id is not None:
            expires_at = datetime.now(UTC) + timedelta(hours=SESSION_FILE_TTL_HOURS)

        view = _UploadedFileView(
            id=file_id,
            owner_id=owner_id,
            session_id=session_id,
            filename=filename,
            file_type=file_type.value,
            file_size=file_size,
            storage_path=str(storage_path),
            processing_status="pending",
            expires_at=expires_at,
        )
        await self._cold_upsert_row(_FILE_TABLE, _file_view_to_row(view), pk="id")
        logger.info(
            "File uploaded id=%s owner=%s name=%s", file_id, owner_id, filename
        )
        return view, None

    # -- Processing (chunking) -----------------------------------------------

    async def process_file(
        self, file_id: UUID
    ) -> tuple[bool, str | None]:
        rows = await self._cold_list_rows(_FILE_TABLE, {"id": str(file_id)})
        if not rows:
            return False, "File not found"
        view = _row_to_file_view(rows[0])

        view.mark_processing()
        await self._cold_upsert_row(_FILE_TABLE, _file_view_to_row(view), pk="id")

        try:
            from deep_research.services.file_upload_service import FileUploadService

            _svc = FileUploadService.__new__(FileUploadService)
            _svc._storage_path = self._storage_path

            content = _svc._read_file_content(view)  # type: ignore[arg-type]
            if content is None:
                view.mark_failed("Failed to read file content")
                await self._cold_upsert_row(_FILE_TABLE, _file_view_to_row(view), pk="id")
                return False, "Failed to read file content"

            content = _svc._sanitize_text_content(content)
            if not content:
                view.mark_failed("No extractable text content found")
                await self._cold_upsert_row(_FILE_TABLE, _file_view_to_row(view), pk="id")
                return False, "No extractable text content found"

            total_extracted_chars = len(content)
            chunks_data = _svc._chunk_content(content, view.type)

            # Write chunks in batches of up to _CHUNK_BATCH_SIZE.
            # Column name is `metadata` (no underscore) — matches DDL in
            # lakebase_ddl.sql / sql_warehouse_ddl.sql. The trailing-underscore
            # form is only used for uploaded_files (legacy ORM attribute).
            chunk_rows = []
            for i, chunk_d in enumerate(chunks_data):
                chunk_rows.append({
                    "file_id": str(file_id),
                    "chunk_index": i,
                    "content": chunk_d["content"],
                    "metadata": chunk_d.get("metadata", {}),
                })

            for batch_start in range(0, len(chunk_rows), _CHUNK_BATCH_SIZE):
                batch = chunk_rows[batch_start: batch_start + _CHUNK_BATCH_SIZE]
                await self._stack.backend.append_events(_CHUNK_TABLE, batch)

            view.mark_ready(len(chunks_data), total_extracted_chars=total_extracted_chars)
            await self._cold_upsert_row(_FILE_TABLE, _file_view_to_row(view), pk="id")
            logger.info(
                "File processed id=%s chunks=%d", file_id, len(chunks_data)
            )
            return True, None

        except Exception as e:
            logger.error("File processing failed: %s", str(e))
            try:
                view.mark_failed(str(e))
                await self._cold_upsert_row(_FILE_TABLE, _file_view_to_row(view), pk="id")
            except Exception:
                pass
            return False, str(e)

    # -- Reads ---------------------------------------------------------------

    async def get_session_files(
        self,
        owner_id: str,
        session_id: UUID | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[_UploadedFileView], int]:
        where: dict[str, Any] = {"owner_id": owner_id}
        if session_id is not None:
            where["session_id"] = str(session_id)
        rows = await self._cold_list_rows(_FILE_TABLE, where, order_by="-created_at")
        total = len(rows)
        page = rows[offset: offset + limit]
        return [_row_to_file_view(r) for r in page], total

    async def get_for_user(
        self, file_id: UUID, owner_id: str
    ) -> _UploadedFileView | None:
        rows = await self._cold_list_rows(
            _FILE_TABLE, {"id": str(file_id), "owner_id": owner_id}
        )
        if not rows:
            return None
        return _row_to_file_view(rows[0])

    async def get(self, file_id: UUID) -> _UploadedFileView | None:
        rows = await self._cold_list_rows(_FILE_TABLE, {"id": str(file_id)})
        if not rows:
            return None
        return _row_to_file_view(rows[0])

    async def get_file_chunks(
        self,
        file_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[_FileChunkView]:
        rows = await self._stack.backend.read_chunk(file_id)
        page = rows[offset: offset + limit]
        return [_chunk_row_to_view(r) for r in page]

    async def get_first_chunk(self, file_id: UUID) -> _FileChunkView | None:
        rows = await self._stack.backend.read_chunk(file_id, chunk_index=0)
        if not rows:
            return None
        return _chunk_row_to_view(rows[0])

    # -- Writes / deletes ----------------------------------------------------

    async def update(self, uploaded_file: Any) -> Any:
        """Persist in-place mutations to an _UploadedFileView."""
        if isinstance(uploaded_file, _UploadedFileView):
            uploaded_file.updated_at = datetime.now(UTC)
            await self._cold_upsert_row(
                _FILE_TABLE, _file_view_to_row(uploaded_file), pk="id"
            )
        return uploaded_file

    async def delete_file(
        self, file_id: UUID, owner_id: str
    ) -> bool:
        view = await self.get_for_user(file_id, owner_id)
        if not view:
            return False
        try:
            storage_path = Path(view.storage_path)
            if storage_path.exists():
                storage_path.unlink()
        except OSError as e:
            logger.warning("Failed to delete storage file: %s", e)
        await self._cold_delete_row(_FILE_TABLE, str(file_id), pk="id")
        return True

    async def delete_expired_files(self) -> int:
        now = datetime.now(UTC)
        all_rows = await self._cold_list_rows(_FILE_TABLE)
        deleted = 0
        for row in all_rows:
            raw_exp = row.get("expires_at")
            if not raw_exp:
                continue
            exp = (
                raw_exp
                if isinstance(raw_exp, datetime)
                else datetime.fromisoformat(str(raw_exp))
            )
            if exp < now:
                view = _row_to_file_view(row)
                try:
                    p = Path(view.storage_path)
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass
                await self._cold_delete_row(_FILE_TABLE, str(view.id), pk="id")
                deleted += 1
        logger.info("Deleted %d expired files", deleted)
        return deleted
