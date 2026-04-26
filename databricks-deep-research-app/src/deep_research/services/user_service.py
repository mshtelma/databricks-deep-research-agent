"""User identity service — upsert and batch lookup."""

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.models.user import User


class UserService:
    """Service for managing user identity records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert(
        self,
        user_id: str,
        email: str | None,
        display_name: str | None,
    ) -> None:
        """Create or update user record from auth identity."""
        stmt = (
            pg_insert(User)
            .values(
                user_id=user_id,
                email=email,
                display_name=display_name,
            )
            .on_conflict_do_update(
                index_elements=["user_id"],
                set_={
                    "email": email,
                    "display_name": display_name,
                    "last_seen_at": func.now(),
                },
            )
        )
        await self._session.execute(stmt)

    async def resolve_user_ids(
        self,
        user_ids: list[str],
    ) -> dict[str, tuple[str | None, str | None]]:
        """Batch resolve user_ids to (email, display_name) pairs."""
        if not user_ids:
            return {}
        result = await self._session.execute(
            select(User.user_id, User.email, User.display_name).where(
                User.user_id.in_(user_ids)
            )
        )
        return {
            row.user_id: (row.email, row.display_name) for row in result.all()
        }
