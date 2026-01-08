"""User preferences Pydantic schemas."""

from datetime import datetime

from pydantic import Field

from src.models.enums import QueryMode
from src.models.research_session import ResearchDepth
from src.schemas.common import BaseSchema


class UpdatePreferencesRequest(BaseSchema):
    """Request schema for updating preferences."""

    system_instructions: str | None = Field(None, max_length=10000)
    default_research_depth: ResearchDepth | None = None
    default_query_mode: QueryMode | None = None
    theme: str | None = Field(None, max_length=20)
    notifications_enabled: bool | None = None


class UserPreferencesResponse(BaseSchema):
    """Response schema for user preferences."""

    system_instructions: str | None = None
    default_research_depth: ResearchDepth
    default_query_mode: QueryMode
    theme: str
    notifications_enabled: bool
    updated_at: datetime
