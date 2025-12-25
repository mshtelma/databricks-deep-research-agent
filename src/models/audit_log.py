"""AuditLog SQLAlchemy model."""

from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base, UUIDMixin


class AuditAction(str, Enum):
    """Audit action types."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"
    ARCHIVE = "archive"
    RESTORE = "restore"


class AuditLog(Base, UUIDMixin):
    """Audit log model.

    Records user actions for security and compliance.
    Append-only table with minimum 1 year retention.
    """

    __tablename__ = "audit_logs"

    # User who performed the action
    user_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    # Action details (matches migration column names)
    action: Mapped[AuditAction] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    resource_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    resource_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # Additional context (matches migration column name)
    details: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )

    # Request context (matches migration - String not INET)
    ip_address: Mapped[str | None] = mapped_column(
        String(45),
        nullable=True,
    )
    user_agent: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    @classmethod
    def create_log(
        cls,
        user_id: str,
        action: AuditAction,
        resource_type: str,
        resource_id: str | None = None,
        details: dict | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> "AuditLog":
        """Factory method to create an audit log entry."""
        return cls(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
        )
