"""User identity model — synced from Databricks auth on login."""

from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from deep_research.db.base import Base


class User(Base):
    """Databricks workspace user, synced on authentication.

    NOT an ownership authority — Databricks IAM is.
    This is a lookup table for resolving user_id → email/display_name.

    FK constraints from user_id/owner_id columns in other tables exist at the
    database level (migration 021) but are intentionally not declared as ORM
    relationships to avoid eager-loading side effects.
    """

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )
    email: Mapped[str | None] = mapped_column(
        String(320),
        nullable=True,
        index=True,
    )
    display_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
