"""User preferences endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_db
from src.middleware.auth import CurrentUser
from src.schemas.preferences import (
    UpdatePreferencesRequest,
    UserPreferencesResponse,
)
from src.services.preferences_service import PreferencesService

router = APIRouter()


@router.get("", response_model=UserPreferencesResponse)
async def get_preferences(
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> UserPreferencesResponse:
    """Get user preferences."""
    service = PreferencesService(db)
    preferences = await service.get_preferences(user.user_id)
    return UserPreferencesResponse(
        system_instructions=preferences.system_instructions,
        default_research_depth=preferences.default_research_depth,
        default_query_mode=preferences.default_query_mode,
        theme=preferences.theme,
        notifications_enabled=preferences.notifications_enabled,
        updated_at=preferences.updated_at,
    )


@router.put("", response_model=UserPreferencesResponse)
async def update_preferences(
    request: UpdatePreferencesRequest,
    user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> UserPreferencesResponse:
    """Update user preferences."""
    service = PreferencesService(db)
    preferences = await service.update_preferences(
        user_id=user.user_id,
        system_instructions=request.system_instructions,
        default_research_depth=request.default_research_depth,
        default_query_mode=request.default_query_mode,
        theme=request.theme,
        notifications_enabled=request.notifications_enabled,
    )
    await db.commit()
    return UserPreferencesResponse(
        system_instructions=preferences.system_instructions,
        default_research_depth=preferences.default_research_depth,
        default_query_mode=preferences.default_query_mode,
        theme=preferences.theme,
        notifications_enabled=preferences.notifications_enabled,
        updated_at=preferences.updated_at,
    )
