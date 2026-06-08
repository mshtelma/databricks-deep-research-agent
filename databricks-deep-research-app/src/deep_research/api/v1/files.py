"""File upload endpoints.

API endpoints for user file upload functionality (US7):
- POST /files/upload - Upload file(s) multipart
- GET /files - List session files
- GET /files/{id} - Get file details
- GET /files/{id}/preview - Preview file content (first chunk)
- DELETE /files/{id} - Remove file

Part of 007-enterprise-data-sources feature (T090).
"""

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile

from deep_research.core.deps import get_file_upload_service
from deep_research.core.exceptions import NotFoundError
from deep_research.middleware.auth import CurrentUser
from deep_research.models.uploaded_file import FileProcessingStatus
from deep_research.schemas.file_upload import (
    FilePreviewResponse,
    UploadedFileListResponse,
    UploadedFileResponse,
)
from deep_research.schemas.file_upload import (
    FileProcessingStatus as FileProcessingStatusSchema,
)
from deep_research.schemas.file_upload import (
    FileType as FileTypeSchema,
)
from deep_research.services._protocols import IFileUploadService

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files")


def _file_to_response(file: Any) -> UploadedFileResponse:
    """Convert UploadedFile model or view to response schema."""

    return UploadedFileResponse(
        id=file.id,
        owner_id=file.owner_id,
        session_id=file.session_id,
        filename=file.filename,
        file_type=FileTypeSchema(file.file_type),
        file_size=file.file_size,
        storage_path=file.storage_path,
        processing_status=FileProcessingStatusSchema(file.processing_status),
        chunk_count=file.chunk_count,
        expires_at=file.expires_at,
        metadata=file.metadata_,
        created_at=file.created_at,
        updated_at=file.updated_at,
    )


# =============================================================================
# Upload Endpoint
# =============================================================================


@router.post("/upload", response_model=list[UploadedFileResponse], status_code=201)
async def upload_files(
    _request: Request,
    user: CurrentUser,
    files: list[UploadFile] = File(..., description="File(s) to upload"),
    session_id: UUID | None = Query(None, description="Optional session ID for session-scoped files"),
    service: IFileUploadService = Depends(get_file_upload_service),
) -> list[UploadedFileResponse]:
    """Upload one or more files.

    Accepts PDF, TXT, MD, and DOCX files.
    Maximum 10MB per file, 50MB total per session.

    Files are automatically processed (chunked) for search after upload.
    """
    uploaded_files: list[UploadedFileResponse] = []
    errors: list[str] = []

    for upload in files:
        # Get file size
        content = await upload.read()
        file_size = len(content)

        # Upload file
        uploaded_file, error = await service.upload_file(
            owner_id=user.user_id,
            filename=upload.filename or "unnamed",
            file_content=content,
            file_size=file_size,
            session_id=session_id,
            content_type=upload.content_type,
        )

        if error:
            errors.append(f"{upload.filename}: {error}")
            continue

        if uploaded_file:
            # Process file (chunking) - run synchronously for now
            # TODO: Move to background job for large files
            success, process_error = await service.process_file(uploaded_file.id)
            if not success:
                logger.warning(
                    f"File processing failed: {process_error}",
                    extra={"file_id": str(uploaded_file.id)},
                )

            # Refresh to get updated processing status
            uploaded_file = await service.get(uploaded_file.id)
            if uploaded_file:
                uploaded_files.append(_file_to_response(uploaded_file))

    # If all uploads failed, raise error
    if not uploaded_files and errors:
        raise HTTPException(
            status_code=400,
            detail={"message": "All file uploads failed", "errors": errors},
        )

    # If some uploads failed, include errors in response metadata
    # (but still return successful uploads)
    if errors:
        logger.warning(f"Some file uploads failed: {errors}")

    return uploaded_files


# =============================================================================
# List Endpoint
# =============================================================================


@router.get("", response_model=UploadedFileListResponse)
async def list_files(
    _request: Request,
    user: CurrentUser,
    session_id: UUID | None = Query(None, description="Filter by session ID"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    service: IFileUploadService = Depends(get_file_upload_service),
) -> UploadedFileListResponse:
    """List uploaded files.

    Returns paginated list of user's uploaded files.
    Optionally filter by session ID for session-scoped files.
    """
    files, total = await service.get_session_files(
        owner_id=user.user_id,
        session_id=session_id,
        limit=limit,
        offset=offset,
    )

    return UploadedFileListResponse(
        items=[_file_to_response(f) for f in files],
        total=total,
        limit=limit,
        offset=offset,
    )


# =============================================================================
# Get Endpoint
# =============================================================================


@router.get("/{file_id}", response_model=UploadedFileResponse)
async def get_file(
    _request: Request,
    file_id: UUID,
    user: CurrentUser,
    service: IFileUploadService = Depends(get_file_upload_service),
) -> UploadedFileResponse:
    """Get file details by ID."""
    uploaded_file = await service.get_for_user(file_id, user.user_id)

    if not uploaded_file:
        raise NotFoundError("File", str(file_id))

    return _file_to_response(uploaded_file)


# =============================================================================
# Preview Endpoint
# =============================================================================


@router.get("/{file_id}/preview", response_model=FilePreviewResponse)
async def preview_file(
    _request: Request,
    file_id: UUID,
    user: CurrentUser,
    service: IFileUploadService = Depends(get_file_upload_service),
) -> FilePreviewResponse:
    """Preview file content (first chunk).

    Returns file metadata and first chunk content for quick preview.
    """
    uploaded_file = await service.get_for_user(file_id, user.user_id)

    if not uploaded_file:
        raise NotFoundError("File", str(file_id))

    # Get first chunk for preview
    preview_content = None
    preview_chunk_index = None
    error = None

    if uploaded_file.processing_status == FileProcessingStatus.READY.value:
        first_chunk = await service.get_first_chunk(file_id)
        if first_chunk:
            preview_content = first_chunk.content
            preview_chunk_index = first_chunk.chunk_index
    elif uploaded_file.processing_status == FileProcessingStatus.FAILED.value:
        error = uploaded_file.metadata_.get("error", "Processing failed")

    return FilePreviewResponse(
        file_id=uploaded_file.id,
        filename=uploaded_file.filename,
        file_type=FileTypeSchema(uploaded_file.file_type),
        file_size=uploaded_file.file_size,
        processing_status=FileProcessingStatusSchema(uploaded_file.processing_status),
        chunk_count=uploaded_file.chunk_count,
        preview_content=preview_content,
        preview_chunk_index=preview_chunk_index,
        error=error,
    )


# =============================================================================
# Delete Endpoint
# =============================================================================


@router.delete("/{file_id}", status_code=204)
async def delete_file(
    _request: Request,
    file_id: UUID,
    user: CurrentUser,
    service: IFileUploadService = Depends(get_file_upload_service),
) -> None:
    """Delete an uploaded file.

    Removes the file from storage and database.
    """
    deleted = await service.delete_file(file_id, user.user_id)

    if not deleted:
        raise NotFoundError("File", str(file_id))
