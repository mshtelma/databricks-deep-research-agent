"""User skill-folder endpoints (Feature 2.2 / A3).

CRUD for the per-user skill-folder roots that the runtime scans (under the
user's OBO identity) on top of the built-in ``~/.skills`` / ``~/.assistant/skills``
roots. All operations are scoped to the authenticated user.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.db.session import get_db
from deep_research.middleware.auth import CurrentUser
from deep_research.schemas.skill_folder import (
    AddSkillFolderRequest,
    SkillFolderListResponse,
    SkillFolderResponse,
    normalize_skill_folder_path,
)
from deep_research.services.skill_folder_store import SkillFolderStore

router = APIRouter()


def _to_response(row: object) -> SkillFolderResponse:
    return SkillFolderResponse(
        id=row.id,  # type: ignore[attr-defined]
        path=row.path,  # type: ignore[attr-defined]
        kind=row.kind,  # type: ignore[attr-defined]
        created_at=row.created_at,  # type: ignore[attr-defined]
    )


@router.get("/skill-folders", response_model=SkillFolderListResponse)
async def list_skill_folders(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> SkillFolderListResponse:
    """List the current user's registered skill folders."""
    store = SkillFolderStore(db)
    folders = await store.list_for_user(user.user_id)
    return SkillFolderListResponse(folders=[_to_response(f) for f in folders])


@router.post(
    "/skill-folders",
    response_model=SkillFolderResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_skill_folder(
    request: AddSkillFolderRequest,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> SkillFolderResponse:
    """Register a skill folder for the current user (idempotent)."""
    # Re-validate with kind-specific rules (the request validator only enforces
    # absoluteness, since ``kind`` isn't bound during field validation).
    try:
        normalize_skill_folder_path(request.path, request.kind)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    store = SkillFolderStore(db)
    try:
        row = await store.add(user.user_id, request.path, request.kind)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    await db.commit()
    return _to_response(row)


@router.delete(
    "/skill-folders/{folder_id}", status_code=status.HTTP_204_NO_CONTENT
)
async def delete_skill_folder(
    folder_id: UUID,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Remove one of the current user's skill folders."""
    store = SkillFolderStore(db)
    removed = await store.remove(user.user_id, folder_id)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="skill folder not found"
        )
    await db.commit()
