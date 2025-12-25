"""User preferences endpoints."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.middleware.auth import CurrentUser
from src.models.research_session import ResearchDepth
from src.schemas.preferences import (
    UpdatePreferencesRequest,
    UserPreferencesResponse,
)

router = APIRouter()


@router.get("", response_model=UserPreferencesResponse)
async def get_preferences(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> UserPreferencesResponse:
    """Get user preferences."""
    # TODO: Implement with PreferencesService
    return UserPreferencesResponse(
        system_instructions=None,
        default_research_depth=ResearchDepth.AUTO,
        theme="system",
        notifications_enabled=True,
        updated_at=datetime.now(UTC),
    )


@router.put("", response_model=UserPreferencesResponse)
async def update_preferences(
    request: UpdatePreferencesRequest,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> UserPreferencesResponse:
    """Update user preferences."""
    # TODO: Implement with PreferencesService
    return UserPreferencesResponse(
        system_instructions=request.system_instructions,
        default_research_depth=request.default_research_depth or ResearchDepth.AUTO,
        theme=request.theme or "system",
        notifications_enabled=request.notifications_enabled if request.notifications_enabled is not None else True,
        updated_at=datetime.now(UTC),
    )
