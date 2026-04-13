"""FileUploadService - CRUD operations and processing for uploaded files.

Manages user-uploaded files for research:
- File validation (type, size limits)
- Chunking with paragraph boundaries
- Storage to local filesystem (or Databricks Volumes)
- Session-scoped file management

Part of 007-enterprise-data-sources feature (T088).
"""

import logging
import re
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID, uuid4

from sqlalchemy import and_, func, select

from deep_research.models.uploaded_file import (
    FileChunk,
    FileProcessingStatus,
    FileType,
    UploadedFile,
)
from deep_research.services.base import BaseRepository

logger = logging.getLogger(__name__)

# =============================================================================
# Constants (FR-059)
# =============================================================================

# Maximum file size: 10MB per file
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

# Maximum total session files: 50MB
MAX_TOTAL_SESSION_SIZE_BYTES = 50 * 1024 * 1024  # 50MB

# Maximum files per session
MAX_FILES_PER_SESSION = 20

# Supported file types (no OCR - FR-059)
SUPPORTED_FILE_TYPES = {
    FileType.PDF: [".pdf"],
    FileType.TXT: [".txt", ".text"],
    FileType.MD: [".md", ".markdown"],
    FileType.DOCX: [".docx"],
}

# File type MIME mappings
MIME_TYPE_MAPPING = {
    "application/pdf": FileType.PDF,
    "text/plain": FileType.TXT,
    "text/markdown": FileType.MD,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": FileType.DOCX,
}

# Chunk size target (characters)
CHUNK_SIZE_TARGET = 1500
CHUNK_SIZE_MIN = 500
CHUNK_SIZE_MAX = 3000

# Session file expiration (hours)
SESSION_FILE_TTL_HOURS = 4


def get_file_type_from_extension(filename: str) -> FileType | None:
    """Get file type from filename extension.

    Args:
        filename: Original filename.

    Returns:
        FileType if supported, None otherwise.
    """
    ext = Path(filename).suffix.lower()
    for file_type, extensions in SUPPORTED_FILE_TYPES.items():
        if ext in extensions:
            return file_type
    return None


def get_file_type_from_mime(content_type: str) -> FileType | None:
    """Get file type from MIME content type.

    Args:
        content_type: MIME type string.

    Returns:
        FileType if supported, None otherwise.
    """
    # Normalize content type (remove charset, etc.)
    content_type = content_type.split(";")[0].strip().lower()
    return MIME_TYPE_MAPPING.get(content_type)


class FileUploadService(BaseRepository[UploadedFile]):
    """Service for managing uploaded files.

    Extends BaseRepository[UploadedFile] for standard CRUD operations.
    Provides specialized methods for:
    - File validation and upload
    - Content chunking
    - Session file management
    """

    model = UploadedFile

    def __init__(
        self,
        session: Any,
        storage_path: str | None = None,
    ) -> None:
        """Initialize service with database session.

        Args:
            session: Async SQLAlchemy session.
            storage_path: Base path for file storage. Defaults to temp directory.
        """
        super().__init__(session)
        self._storage_path = storage_path or tempfile.gettempdir()

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_file(
        self,
        filename: str,
        file_size: int,
        content_type: str | None = None,
    ) -> tuple[bool, str | None, FileType | None]:
        """Validate a file for upload.

        Args:
            filename: Original filename.
            file_size: File size in bytes.
            content_type: Optional MIME content type.

        Returns:
            Tuple of (is_valid, error_message, file_type).
        """
        # Check file size
        if file_size > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
            return False, f"File exceeds maximum size of {max_mb}MB", None

        if file_size == 0:
            return False, "File is empty", None

        # Determine file type
        file_type = get_file_type_from_extension(filename)
        if file_type is None and content_type:
            file_type = get_file_type_from_mime(content_type)

        if file_type is None:
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
        """Validate session quota for new file.

        Args:
            owner_id: User ID.
            session_id: Session ID (None for permanent files).
            new_file_size: Size of new file in bytes.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if session_id is None:
            # No quota for permanent files (yet)
            return True, None

        # Count existing files
        count_result = await self._session.execute(
            select(func.count(UploadedFile.id)).where(
                and_(
                    UploadedFile.owner_id == owner_id,
                    UploadedFile.session_id == session_id,
                )
            )
        )
        file_count = count_result.scalar() or 0

        if file_count >= MAX_FILES_PER_SESSION:
            return False, f"Maximum {MAX_FILES_PER_SESSION} files per session"

        # Calculate total size
        size_result = await self._session.execute(
            select(func.sum(UploadedFile.file_size)).where(
                and_(
                    UploadedFile.owner_id == owner_id,
                    UploadedFile.session_id == session_id,
                )
            )
        )
        total_size = size_result.scalar() or 0

        if total_size + new_file_size > MAX_TOTAL_SESSION_SIZE_BYTES:
            max_mb = MAX_TOTAL_SESSION_SIZE_BYTES / (1024 * 1024)
            return False, f"Total session files exceed {max_mb}MB limit"

        return True, None

    # =========================================================================
    # Upload and Storage
    # =========================================================================

    async def upload_file(
        self,
        owner_id: str,
        filename: str,
        file_content: BinaryIO | bytes,
        file_size: int,
        session_id: UUID | None = None,
        content_type: str | None = None,
    ) -> tuple[UploadedFile | None, str | None]:
        """Upload and store a file.

        Args:
            owner_id: User ID.
            filename: Original filename.
            file_content: File content as bytes or file-like object.
            file_size: File size in bytes.
            session_id: Optional session ID for session-scoped files.
            content_type: Optional MIME content type.

        Returns:
            Tuple of (uploaded_file, error_message).
        """
        # Validate file
        is_valid, error, file_type = self.validate_file(
            filename, file_size, content_type
        )
        if not is_valid or file_type is None:
            return None, error

        # Validate quota
        is_valid, error = await self.validate_session_quota(
            owner_id, session_id, file_size
        )
        if not is_valid:
            return None, error

        # Generate storage path
        file_id = uuid4()
        storage_dir = Path(self._storage_path) / "uploads" / owner_id
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename: strip directory components, allow only safe characters
        import re
        from pathlib import PurePosixPath

        base_name = PurePosixPath(filename).name
        sanitized = re.sub(r'[^\w.\-]', '_', base_name)[:200]
        if not sanitized or sanitized.startswith('.'):
            sanitized = f"upload{sanitized}"
        safe_filename = f"{file_id}_{sanitized}"
        storage_path = storage_dir / safe_filename

        # Write file to storage
        try:
            if isinstance(file_content, bytes):
                storage_path.write_bytes(file_content)
            else:
                with storage_path.open("wb") as f:
                    shutil.copyfileobj(file_content, f)
        except OSError as e:
            logger.error("Failed to write file: %s", str(e))
            return None, f"Failed to store file: {e}"

        # Calculate expiration for session-scoped files
        expires_at = None
        if session_id is not None:
            expires_at = datetime.now(UTC) + timedelta(hours=SESSION_FILE_TTL_HOURS)

        # Create database record
        uploaded_file = UploadedFile(
            id=file_id,
            owner_id=owner_id,
            session_id=session_id,
            filename=filename,
            file_type=file_type.value,
            file_size=file_size,
            storage_path=str(storage_path),
            processing_status=FileProcessingStatus.PENDING.value,
            expires_at=expires_at,
        )

        uploaded_file = await self.add(uploaded_file)
        logger.info(
            "File uploaded",
            extra={
                "file_id": str(uploaded_file.id),
                "owner_id": owner_id,
                "file_name": filename,
                "file_type": file_type.value,
                "file_size": file_size,
            },
        )

        return uploaded_file, None

    # =========================================================================
    # Chunking
    # =========================================================================

    async def process_file(self, file_id: UUID) -> tuple[bool, str | None]:
        """Process (chunk) a file for search.

        Args:
            file_id: File ID to process.

        Returns:
            Tuple of (success, error_message).
        """
        # Get file
        uploaded_file = await self.get(file_id)
        if not uploaded_file:
            return False, "File not found"

        # Mark as processing
        uploaded_file.mark_processing()
        await self.update(uploaded_file)

        try:
            # Read file content
            content = self._read_file_content(uploaded_file)
            if content is None:
                uploaded_file.mark_failed("Failed to read file content")
                await self.update(uploaded_file)
                return False, "Failed to read file content"
            content = self._sanitize_text_content(content)
            if not content:
                uploaded_file.mark_failed("No extractable text content found")
                await self.update(uploaded_file)
                return False, "No extractable text content found"

            total_extracted_chars = len(content)

            # Chunk content
            chunks = self._chunk_content(content, uploaded_file.type)

            # Create chunk records
            chunk_entities = []
            for i, chunk_data in enumerate(chunks):
                chunk = FileChunk(
                    file_id=file_id,
                    chunk_index=i,
                    content=chunk_data["content"],
                    metadata_=chunk_data.get("metadata", {}),
                )
                chunk_entities.append(chunk)

            # Bulk insert chunks within savepoint so that a failed insert
            # does not corrupt the outer transaction or detach the parent entity.
            async with self._session.begin_nested():
                for chunk in chunk_entities:
                    self._session.add(chunk)
                await self._session.flush()

            # Mark as ready (outside savepoint — savepoint already released)
            uploaded_file.mark_ready(len(chunks), total_extracted_chars=total_extracted_chars)
            await self.update(uploaded_file)

            logger.info(
                "File processed",
                extra={
                    "file_id": str(file_id),
                    "chunk_count": len(chunks),
                    "file_size_bytes": uploaded_file.file_size,
                    "total_extracted_chars": total_extracted_chars,
                },
            )
            return True, None

        except Exception as e:
            logger.error("File processing failed: %s", str(e))
            try:
                uploaded_file.mark_failed(str(e))
                await self.update(uploaded_file)
            except Exception as update_err:
                logger.error(
                    "Could not mark file as failed: %s", str(update_err)
                )
            return False, str(e)

    def _read_file_content(self, uploaded_file: UploadedFile) -> str | None:
        """Read and extract text content from a file.

        Args:
            uploaded_file: File to read.

        Returns:
            Text content or None if failed.
        """
        storage_path = Path(uploaded_file.storage_path)
        if not storage_path.exists():
            return None

        file_type = uploaded_file.type

        try:
            if file_type == FileType.TXT or file_type == FileType.MD:
                return storage_path.read_text(encoding="utf-8")

            elif file_type == FileType.PDF:
                return self._extract_pdf_text(storage_path)

            elif file_type == FileType.DOCX:
                return self._extract_docx_text(storage_path)

            else:
                logger.warning(f"Unsupported file type: {file_type}")
                return None

        except Exception as e:
            logger.error("Failed to read file content: %s", str(e))
            return None

    def _extract_pdf_text(self, path: Path) -> str | None:
        """Extract text from PDF using Docling.

        Returns markdown text when conversion succeeds.
        """
        try:
            from docling.document_converter import (  # type: ignore[import-not-found]
                DocumentConverter,
            )
        except ImportError:
            logger.warning("Docling not available, cannot extract PDF text")
            return None

        try:
            converter = DocumentConverter()
            result = converter.convert(str(path))
            document = getattr(result, "document", None)
            if document is None:
                logger.warning("Docling conversion returned no document object")
                return None

            markdown: str = document.export_to_markdown()
            if not markdown or not markdown.strip():
                logger.warning("Docling extracted no text from PDF")
                return None

            return markdown
        except Exception as e:
            logger.error("Docling PDF extraction failed: %s", str(e))
            return None

    def _sanitize_text_content(self, content: str) -> str:
        """Sanitize extracted text for PostgreSQL-safe storage.

        PostgreSQL text/varchar cannot contain null bytes.
        """
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        # Keep newline and tab; remove other ASCII control characters.
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", normalized)
        return sanitized.strip()

    def _extract_docx_text(self, path: Path) -> str | None:
        """Extract text from DOCX file.

        Uses python-docx if available.
        """
        try:
            from docx import Document  # type: ignore[import-not-found]

            doc = Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)

        except ImportError:
            logger.warning("python-docx not available, DOCX text extraction limited")
            # Fallback: try to read as XML
            import zipfile

            try:
                with zipfile.ZipFile(path) as z:
                    xml_content = z.read("word/document.xml").decode("utf-8")
                    # Very basic text extraction from XML
                    text = re.sub(r"<[^>]+>", " ", xml_content)
                    text = re.sub(r"\s+", " ", text)
                    return text.strip()
            except Exception:
                return None

    def _chunk_content(
        self,
        content: str,
        _file_type: FileType,
    ) -> list[dict[str, Any]]:
        """Chunk content with paragraph boundaries.

        Uses naive chunking with paragraph boundaries for now.

        Args:
            content: Text content to chunk.
            _file_type: File type for format-specific handling (reserved for future use).

        Returns:
            List of chunk dictionaries with content and metadata.
        """
        chunks: list[dict[str, Any]] = []

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\n+", content)

        current_chunk = ""
        current_page = 1

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check for page markers (PDF)
            page_match = re.match(r"^\[Page (\d+)\]", para)
            if page_match:
                current_page = int(page_match.group(1))
                para = re.sub(r"^\[Page \d+\]\s*", "", para)
                if not para:
                    continue

            # Would adding this paragraph exceed max chunk size?
            potential_chunk = current_chunk + "\n\n" + para if current_chunk else para

            if len(potential_chunk) > CHUNK_SIZE_MAX:
                # Save current chunk if it meets minimum
                if len(current_chunk) >= CHUNK_SIZE_MIN:
                    chunks.append({
                        "content": current_chunk,
                        "metadata": {"page_number": current_page},
                    })
                    current_chunk = para
                else:
                    # Current chunk too small, force merge
                    current_chunk = potential_chunk

            elif len(potential_chunk) >= CHUNK_SIZE_TARGET:
                # Good size, save it
                chunks.append({
                    "content": potential_chunk,
                    "metadata": {"page_number": current_page},
                })
                current_chunk = ""
            else:
                # Keep building
                current_chunk = potential_chunk

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk,
                "metadata": {"page_number": current_page},
            })

        return chunks

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def get_session_files(
        self,
        owner_id: str,
        session_id: UUID | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[UploadedFile], int]:
        """Get files for a user/session.

        Args:
            owner_id: User ID.
            session_id: Optional session ID. If None, returns all user files.
            limit: Maximum files to return.
            offset: Number of files to skip.

        Returns:
            Tuple of (files, total_count).
        """
        conditions = [UploadedFile.owner_id == owner_id]
        if session_id is not None:
            conditions.append(UploadedFile.session_id == session_id)

        # Get total count
        count_result = await self._session.execute(
            select(func.count(UploadedFile.id)).where(and_(*conditions))
        )
        total = count_result.scalar() or 0

        # Get files
        query = (
            select(UploadedFile)
            .where(and_(*conditions))
            .order_by(UploadedFile.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(query)
        files = list(result.scalars().all())

        return files, total

    async def get_for_user(
        self,
        file_id: UUID,
        owner_id: str,
    ) -> UploadedFile | None:
        """Get a file by ID with ownership check.

        Args:
            file_id: File ID.
            owner_id: User ID.

        Returns:
            File if found and owned by user, None otherwise.
        """
        result = await self._session.execute(
            select(UploadedFile).where(
                and_(
                    UploadedFile.id == file_id,
                    UploadedFile.owner_id == owner_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_file_chunks(
        self,
        file_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FileChunk]:
        """Get chunks for a file.

        Args:
            file_id: File ID.
            limit: Maximum chunks to return.
            offset: Number of chunks to skip.

        Returns:
            List of chunks ordered by index.
        """
        result = await self._session.execute(
            select(FileChunk)
            .where(FileChunk.file_id == file_id)
            .order_by(FileChunk.chunk_index)
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def get_first_chunk(self, file_id: UUID) -> FileChunk | None:
        """Get first chunk of a file for preview.

        Args:
            file_id: File ID.

        Returns:
            First chunk or None.
        """
        result = await self._session.execute(
            select(FileChunk)
            .where(FileChunk.file_id == file_id)
            .order_by(FileChunk.chunk_index)
            .limit(1)
        )
        return result.scalar_one_or_none()

    # =========================================================================
    # Cleanup Methods
    # =========================================================================

    async def delete_expired_files(self) -> int:
        """Delete expired session files.

        Returns:
            Number of files deleted.
        """
        now = datetime.now(UTC)

        # Find expired files
        result = await self._session.execute(
            select(UploadedFile).where(
                and_(
                    UploadedFile.expires_at.isnot(None),
                    UploadedFile.expires_at < now,
                )
            )
        )
        expired_files = list(result.scalars().all())

        if not expired_files:
            return 0

        # Delete storage files and database records
        deleted_count = 0
        for uploaded_file in expired_files:
            try:
                # Delete storage file
                storage_path = Path(uploaded_file.storage_path)
                if storage_path.exists():
                    storage_path.unlink()

                # Delete database record (cascades to chunks)
                await self.delete(uploaded_file)
                deleted_count += 1

            except Exception as e:
                logger.error(
                    "Failed to delete expired file %s: %s",
                    str(uploaded_file.id),
                    str(e),
                )

        logger.info(f"Deleted {deleted_count} expired files")
        return deleted_count

    async def delete_file(self, file_id: UUID, owner_id: str) -> bool:
        """Delete a file by ID with ownership check.

        Args:
            file_id: File ID.
            owner_id: User ID.

        Returns:
            True if deleted, False if not found.
        """
        uploaded_file = await self.get_for_user(file_id, owner_id)
        if not uploaded_file:
            return False

        # Delete storage file
        try:
            storage_path = Path(uploaded_file.storage_path)
            if storage_path.exists():
                storage_path.unlink()
        except OSError as e:
            logger.warning(f"Failed to delete storage file: {e}")

        # Delete database record (cascades to chunks)
        await self.delete(uploaded_file)
        return True
