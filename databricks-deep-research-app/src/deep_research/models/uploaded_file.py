"""UploadedFile and FileChunk SQLAlchemy models for user file uploads.

Supports file upload functionality (US7) for research:
- File metadata tracking (name, type, size, storage path)
- Processing status for chunking pipeline
- Session-scoped files with optional expiration
- File chunks for content-based search

Part of 007-enterprise-data-sources feature (T085, T086).
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from deep_research.db.base import BaseModel


class FileType(str, Enum):
    """Supported file types for upload.

    Supported types (FR-059: no OCR required):
    - PDF: Portable Document Format
    - TXT: Plain text
    - MD: Markdown
    - DOCX: Microsoft Word (OpenXML)
    """

    PDF = "pdf"
    TXT = "txt"
    MD = "md"
    DOCX = "docx"


class FileProcessingStatus(str, Enum):
    """Processing status for uploaded files.

    Tracks the chunking pipeline state:
    - PENDING: File uploaded, awaiting processing
    - PROCESSING: Currently being chunked
    - READY: Processing complete, searchable
    - FAILED: Processing failed (see error in metadata)
    """

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class UploadedFile(BaseModel):
    """User-uploaded file for research.

    Stores metadata about uploaded files including processing status
    and chunk count. Files can be session-scoped (ephemeral) or
    permanent depending on session_id.

    Attributes:
        owner_id: Databricks workspace user ID who uploaded this file.
        session_id: Optional session ID for session-scoped (ephemeral) files.
        filename: Original filename (for display).
        file_type: Type of file (pdf, txt, md, docx).
        file_size: Size in bytes.
        storage_path: Path to file in storage (local or Databricks Volumes).
        processing_status: Current processing state.
        chunk_count: Number of chunks created from this file.
        expires_at: Optional expiration time for session-scoped files.
        metadata_: Additional metadata (JSONB for flexibility).
    """

    __tablename__ = "uploaded_files"

    # Owner identification (Databricks workspace user ID)
    owner_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Session scope (nullable for permanent files)
    session_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    # File metadata
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)

    # Processing status
    processing_status: Mapped[str] = mapped_column(
        String(20),
        default=FileProcessingStatus.PENDING.value,
        nullable=False,
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    # Optional expiration for session-scoped files
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Additional metadata (error messages, page count, etc.)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # Relationships
    chunks: Mapped[list["FileChunk"]] = relationship(
        "FileChunk",
        back_populates="file",
        cascade="all, delete-orphan",
        order_by="FileChunk.chunk_index",
    )

    # Indexes
    __table_args__ = (
        # Fast lookup by owner
        Index("idx_uploaded_files_owner", "owner_id"),
        # Fast lookup by session
        Index("idx_uploaded_files_session", "session_id"),
        # Composite for listing owner's session files
        Index("idx_uploaded_files_owner_session", "owner_id", "session_id"),
        # For cleanup of expired files
        Index("idx_uploaded_files_expires_at", "expires_at"),
        # For processing queue
        Index("idx_uploaded_files_status", "processing_status"),
    )

    @property
    def type(self) -> FileType:
        """Get file type as enum."""
        return FileType(self.file_type)

    @property
    def status(self) -> FileProcessingStatus:
        """Get processing status as enum."""
        return FileProcessingStatus(self.processing_status)

    @property
    def is_ready(self) -> bool:
        """Check if file is ready for search."""
        return self.processing_status == FileProcessingStatus.READY.value

    @property
    def is_session_scoped(self) -> bool:
        """Check if file is session-scoped (ephemeral)."""
        return self.session_id is not None

    @property
    def is_expired(self) -> bool:
        """Check if file has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at

    @property
    def total_extracted_chars(self) -> int | None:
        """Total character count of extracted text, if available.

        Stored during processing for efficient strategy selection.
        Returns None for legacy files processed before this optimization.
        """
        if self.metadata_:
            val = self.metadata_.get("total_extracted_chars")
            if val is not None:
                return int(val)
        return None

    def mark_processing(self) -> None:
        """Mark file as being processed."""
        self.processing_status = FileProcessingStatus.PROCESSING.value

    def mark_ready(self, chunk_count: int, total_extracted_chars: int | None = None) -> None:
        """Mark file as ready for search.

        Args:
            chunk_count: Number of chunks created.
            total_extracted_chars: Total character count of extracted text
                (measured after sanitization, before chunking). Stored in
                metadata_ for efficient strategy selection at query time.
        """
        self.processing_status = FileProcessingStatus.READY.value
        self.chunk_count = chunk_count
        if total_extracted_chars is not None:
            self.metadata_ = {**(self.metadata_ or {}), "total_extracted_chars": total_extracted_chars}

    def mark_failed(self, error: str) -> None:
        """Mark file processing as failed.

        Args:
            error: Error message describing the failure.
        """
        self.processing_status = FileProcessingStatus.FAILED.value
        self.metadata_ = {**(self.metadata_ or {}), "error": error}


class FileChunk(BaseModel):
    """Chunk of an uploaded file for search.

    Stores content chunks extracted from uploaded files with
    location metadata for citation.

    Attributes:
        file_id: FK to parent uploaded file.
        chunk_index: Position of this chunk in the file (0-indexed).
        content: Text content of this chunk.
        metadata_: Location metadata (page_number, section, etc.).
    """

    __tablename__ = "file_chunks"

    # Foreign key to file
    file_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("uploaded_files.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Chunk position
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Location metadata (page_number, section, etc.)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )

    # Relationships
    file: Mapped["UploadedFile"] = relationship(
        "UploadedFile",
        back_populates="chunks",
    )

    # Indexes
    __table_args__ = (
        # Fast lookup by file
        Index("idx_file_chunks_file", "file_id"),
        # Fast lookup by file and position
        Index("idx_file_chunks_file_index", "file_id", "chunk_index"),
    )

    @property
    def page_number(self) -> int | None:
        """Get page number from metadata if available."""
        return self.metadata_.get("page_number")

    @property
    def section(self) -> str | None:
        """Get section heading from metadata if available."""
        return self.metadata_.get("section")
