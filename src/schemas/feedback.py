"""Feedback-related Pydantic schemas."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from src.schemas.common import BaseSchema


class FeedbackRequest(BaseSchema):
    """Request schema for submitting feedback."""

    rating: Literal["positive", "negative"]
    feedback_text: str | None = Field(None, max_length=5000)
    feedback_category: str | None = Field(None, max_length=50)


class FeedbackResponse(BaseSchema):
    """Response schema for feedback."""

    id: UUID
    message_id: UUID
    rating: str
    feedback_text: str | None = None
    feedback_category: str | None = None
    created_at: datetime
