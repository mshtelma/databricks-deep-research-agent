"""Pagination utilities."""

from collections.abc import Sequence
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Pagination parameters from query string."""

    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response."""

    items: Sequence[T]
    total: int
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        """Check if there are more items."""
        return self.offset + len(self.items) < self.total

    @property
    def page(self) -> int:
        """Current page number (1-indexed)."""
        if self.limit == 0:
            return 1
        return (self.offset // self.limit) + 1

    @property
    def total_pages(self) -> int:
        """Total number of pages."""
        if self.limit == 0:
            return 1
        return (self.total + self.limit - 1) // self.limit

    @classmethod
    def create(
        cls,
        items: Sequence[T],
        total: int,
        params: PaginationParams,
    ) -> "PaginatedResponse[T]":
        """Create paginated response from items and params."""
        return cls(
            items=items,
            total=total,
            limit=params.limit,
            offset=params.offset,
        )


def paginate_query(
    query,
    limit: int,
    offset: int,
):
    """Apply pagination to SQLAlchemy query.

    Args:
        query: SQLAlchemy select query.
        limit: Maximum number of items.
        offset: Number of items to skip.

    Returns:
        Modified query with limit and offset applied.
    """
    return query.limit(limit).offset(offset)
