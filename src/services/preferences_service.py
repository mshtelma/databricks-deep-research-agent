"""Preferences service for managing user preferences."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging_utils import get_logger
from src.models.research_session import ResearchDepth
from src.models.user_preferences import UserPreferences

logger = get_logger(__name__)


class PreferencesService:
    """Service for managing user preferences."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize preferences service.

        Args:
            session: Database session.
        """
        self._session = session

    async def get_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences, creating defaults if not exists.

        Args:
            user_id: User ID.

        Returns:
            UserPreferences instance.
        """
        result = await self._session.execute(
            select(UserPreferences).where(UserPreferences.user_id == user_id)
        )
        preferences = result.scalar_one_or_none()

        if not preferences:
            # Create default preferences for new user
            preferences = UserPreferences(
                user_id=user_id,
                default_research_depth=ResearchDepth.AUTO,
                system_instructions=None,
                theme="system",
                notifications_enabled=True,
            )
            self._session.add(preferences)
            await self._session.flush()
            await self._session.refresh(preferences)

            logger.info(
                "PREFERENCES_CREATED",
                user_id=user_id,
            )

        return preferences

    async def update_preferences(
        self,
        user_id: str,
        system_instructions: str | None = None,
        default_research_depth: ResearchDepth | None = None,
        theme: str | None = None,
        notifications_enabled: bool | None = None,
    ) -> UserPreferences:
        """Update user preferences.

        Args:
            user_id: User ID.
            system_instructions: System instructions for all chats.
            default_research_depth: Default research depth.
            theme: UI theme preference.
            notifications_enabled: Whether notifications are enabled.

        Returns:
            Updated UserPreferences instance.
        """
        preferences = await self.get_preferences(user_id)

        # Update fields if provided
        if system_instructions is not None:
            preferences.update_instructions(system_instructions)

        if default_research_depth is not None:
            preferences.update_depth(default_research_depth)

        if theme is not None:
            preferences.update_theme(theme)

        if notifications_enabled is not None:
            preferences.update_notifications(notifications_enabled)

        preferences.updated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(preferences)

        logger.info(
            "PREFERENCES_UPDATED",
            user_id=user_id,
        )

        return preferences

    async def get_system_instructions(self, user_id: str) -> str | None:
        """Get user's system instructions.

        Args:
            user_id: User ID.

        Returns:
            System instructions or None.
        """
        preferences = await self.get_preferences(user_id)
        return preferences.system_instructions

    async def get_default_research_depth(self, user_id: str) -> ResearchDepth:
        """Get user's default research depth.

        Args:
            user_id: User ID.

        Returns:
            Default research depth.
        """
        preferences = await self.get_preferences(user_id)
        return preferences.default_research_depth

    def to_dict(self, preferences: UserPreferences) -> dict[str, Any]:
        """Convert preferences to dictionary.

        Args:
            preferences: UserPreferences instance.

        Returns:
            Dictionary representation.
        """
        return {
            "system_instructions": preferences.system_instructions,
            "default_research_depth": preferences.default_research_depth.value,
            "theme": preferences.theme,
            "notifications_enabled": preferences.notifications_enabled,
            "updated_at": preferences.updated_at.isoformat() if preferences.updated_at else None,
        }
