"""User profile API endpoints."""

from fastapi import APIRouter

from deep_research.middleware.auth import CurrentUser
from deep_research.schemas.user import UserProfileResponse

router = APIRouter(prefix="/user", tags=["user"])


@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(user: CurrentUser) -> UserProfileResponse:
    """Get current user's profile information.

    Returns the authenticated user's identity details including
    user ID, email, and display name for UI display.
    """
    return UserProfileResponse(
        user_id=user.user_id,
        email=user.email,
        display_name=user.display_name,
        workspace=None,  # Can be extracted from workspace client if needed
    )
