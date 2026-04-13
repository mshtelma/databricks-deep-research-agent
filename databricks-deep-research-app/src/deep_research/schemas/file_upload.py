"""File upload Pydantic schemas.

Schemas for file upload API (US7):
- Request/response schemas for upload, listing, preview
- Processing status enum
- File type validation

Part of 007-enterprise-data-sources feature (T091).
"""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import Field

from deep_research.schemas.common import BaseSchema, TimestampMixin


class FileProcessingStatus(StrEnum):
    """Processing status for uploaded files."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class FileType(StrEnum):
    """Supported file types for upload."""

    PDF = "pdf"
    TXT = "txt"
    MD = "md"
    DOCX = "docx"


# =============================================================================
# Chunk Schemas
# =============================================================================


class FileChunkResponse(BaseSchema):
    """Response schema for a file chunk."""

    id: UUID
    file_id: UUID
    chunk_index: int
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict, alias="metadata")
    page_number: int | None = None
    section: str | None = None


# =============================================================================
# File Schemas
# =============================================================================


class UploadedFileBase(BaseSchema):
    """Base schema for uploaded files."""

    filename: str = Field(..., max_length=255)
    file_type: FileType


class UploadedFileResponse(UploadedFileBase, TimestampMixin):
    """Response schema for an uploaded file."""

    id: UUID
    owner_id: str
    session_id: UUID | None = None
    file_size: int
    processing_status: FileProcessingStatus
    chunk_count: int = 0
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict, alias="metadata")


class UploadedFileListResponse(BaseSchema):
    """Paginated list of uploaded files."""

    items: list[UploadedFileResponse]
    total: int
    limit: int
    offset: int


# =============================================================================
# Preview Schema
# =============================================================================


class FilePreviewResponse(BaseSchema):
    """Response schema for file preview (first chunk)."""

    file_id: UUID
    filename: str
    file_type: FileType
    file_size: int
    processing_status: FileProcessingStatus
    chunk_count: int
    preview_content: str | None = None
    preview_chunk_index: int | None = None
    error: str | None = None


# =============================================================================
# Search Result Schema
# =============================================================================


class FileSearchResult(BaseSchema):
    """Search result from file content search."""

    file_id: UUID
    filename: str
    chunk_id: UUID
    chunk_index: int
    content: str
    score: float = 0.0
    page_number: int | None = None
    section: str | None = None
    highlight: str | None = None


class FileSearchResponse(BaseSchema):
    """Response schema for file search."""

    query: str
    results: list[FileSearchResult]
    total_results: int
