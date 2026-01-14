"""Common Pydantic schemas."""

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

T = TypeVar("T")


class BaseSchema(BaseModel):
    """Base schema with common configuration.

    Uses camelCase aliases for JSON serialization (frontend compatibility).
    - alias_generator: generates camelCase aliases from snake_case field names
    - populate_by_name: accepts both snake_case and camelCase on input
    - serialize_by_alias: outputs camelCase (uses aliases) by default
    """

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        alias_generator=to_camel,
        serialize_by_alias=True,
    )


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""

    created_at: datetime
    updated_at: datetime


class ErrorResponse(BaseModel):
    """Standard error response."""

    code: str
    message: str
    details: dict[str, Any] | None = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""

    items: list[T]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    database: str = "connected"
    version: str = "1.0.0"
