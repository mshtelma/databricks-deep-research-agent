"""User-related Pydantic schemas."""

from deep_research.schemas.common import BaseSchema


class UserProfileResponse(BaseSchema):
    """User profile information response.

    Contains the authenticated user's identity information
    for display in the UI.
    """

    user_id: str
    email: str
    display_name: str
    workspace: str | None = None
