"""Base repository pattern for async SQLAlchemy services.

This module provides a generic BaseRepository class that eliminates
the ~95 occurrences of `add() + flush() + refresh()` boilerplate
across CRUD services.

Usage:
    class ClaimService(BaseRepository[Claim]):
        model = Claim

        async def get_with_citations(self, claim_id: UUID) -> Claim | None:
            # Custom method with eager loading
            ...
"""

import logging
from typing import Generic, TypeVar
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.core.exceptions import NotFoundError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT")


class BaseRepository(Generic[ModelT]):
    """Base repository with common CRUD operations.

    Provides:
    - add(): Add and persist a new entity
    - add_many(): Add multiple entities
    - get(): Get entity by ID (returns None if not found)
    - get_or_raise(): Get entity by ID (raises NotFoundError if not found)
    - update(): Persist changes to an entity
    - delete(): Delete an entity
    - delete_by_id(): Delete entity by ID

    Subclasses should set the `model` class attribute to the SQLAlchemy model.

    Attributes:
        model: The SQLAlchemy model class (must be set by subclass).
    """

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self._session = session

    async def add(self, entity: ModelT) -> ModelT:
        """Add and persist a new entity.

        Args:
            entity: Entity instance to add.

        Returns:
            The persisted entity with ID populated.
        """
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        return entity

    async def add_many(self, entities: list[ModelT]) -> list[ModelT]:
        """Add and persist multiple entities.

        More efficient than calling add() in a loop as it batches
        the flush operation.

        Args:
            entities: List of entity instances to add.

        Returns:
            List of persisted entities with IDs populated.
        """
        for entity in entities:
            self._session.add(entity)
        await self._session.flush()
        for entity in entities:
            await self._session.refresh(entity)
        return entities

    async def get(self, entity_id: UUID) -> ModelT | None:
        """Get entity by primary key.

        Args:
            entity_id: Entity UUID.

        Returns:
            Entity if found, None otherwise.
        """
        result = await self._session.execute(
            select(self.model).where(self.model.id == entity_id)  # type: ignore[attr-defined]
        )
        return result.scalar_one_or_none()

    async def get_or_raise(self, entity_id: UUID) -> ModelT:
        """Get entity by ID or raise NotFoundError.

        Args:
            entity_id: Entity UUID.

        Returns:
            Entity instance.

        Raises:
            NotFoundError: If entity not found.
        """
        entity = await self.get(entity_id)
        if not entity:
            raise NotFoundError(self.model.__name__, str(entity_id))
        return entity

    async def update(self, entity: ModelT) -> ModelT:
        """Persist changes to an entity.

        Call this after modifying an entity's attributes to flush
        changes to the database.

        Args:
            entity: Entity with modifications.

        Returns:
            The updated entity.
        """
        await self._session.flush()
        await self._session.refresh(entity)
        return entity

    async def delete(self, entity: ModelT) -> None:
        """Delete an entity.

        Args:
            entity: Entity to delete.
        """
        await self._session.delete(entity)
        await self._session.flush()

    async def delete_by_id(self, entity_id: UUID) -> bool:
        """Delete entity by ID.

        Args:
            entity_id: Entity UUID.

        Returns:
            True if deleted, False if not found.
        """
        entity = await self.get(entity_id)
        if not entity:
            return False
        await self.delete(entity)
        return True
