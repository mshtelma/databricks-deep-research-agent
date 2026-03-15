"""Audit logging middleware."""

from typing import Annotated, Any

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.core.auth import UserIdentity
from deep_research.db.session import get_db
from deep_research.middleware.auth import CurrentUser
from deep_research.models.audit_log import AuditAction, AuditLog


class AuditLogger:
    """Audit logger for tracking user actions."""

    def __init__(self, db: AsyncSession, user: UserIdentity, request: Request):
        self.db = db
        self.user = user
        self.request = request

    def _get_ip_address(self) -> str | None:
        """Extract client IP address from request."""
        # Check for forwarded headers (behind proxy)
        forwarded = self.request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Check real IP header
        real_ip = self.request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        if self.request.client:
            return self.request.client.host

        return None

    def _get_user_agent(self) -> str | None:
        """Extract user agent from request."""
        return self.request.headers.get("user-agent")

    async def log(
        self,
        action: AuditAction,
        resource_type: str,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditLog:
        """Create an audit log entry.

        Args:
            action: Type of action performed.
            resource_type: Entity type affected (e.g., "chat", "message").
            resource_id: ID of the affected entity.
            details: Additional context about the action.

        Returns:
            Created AuditLog entry.
        """
        log_entry = AuditLog.create_log(
            user_id=self.user.user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=self._get_ip_address(),
            user_agent=self._get_user_agent(),
        )

        self.db.add(log_entry)
        # Note: commit happens at request end via session dependency
        return log_entry

    async def log_create(
        self,
        entity: str,
        entity_id: str,
        details: dict[str, Any] | None = None,
    ) -> AuditLog:
        """Log a create action."""
        return await self.log(AuditAction.CREATE, entity, entity_id, details)

    async def log_read(
        self,
        entity: str,
        entity_id: str,
        details: dict[str, Any] | None = None,
    ) -> AuditLog:
        """Log a read action."""
        return await self.log(AuditAction.READ, entity, entity_id, details)

    async def log_update(
        self,
        entity: str,
        entity_id: str,
        details: dict[str, Any] | None = None,
    ) -> AuditLog:
        """Log an update action."""
        return await self.log(AuditAction.UPDATE, entity, entity_id, details)

    async def log_delete(
        self,
        entity: str,
        entity_id: str,
        details: dict[str, Any] | None = None,
    ) -> AuditLog:
        """Log a delete action."""
        return await self.log(AuditAction.DELETE, entity, entity_id, details)

    async def log_export(
        self,
        entity: str,
        entity_id: str,
        format: str,
        details: dict[str, Any] | None = None,
    ) -> AuditLog:
        """Log an export action."""
        meta = details or {}
        meta["export_format"] = format
        return await self.log(AuditAction.EXPORT, entity, entity_id, meta)


async def get_audit_logger(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    user: CurrentUser,
) -> AuditLogger:
    """FastAPI dependency for audit logger."""
    return AuditLogger(db=db, user=user, request=request)


# Type alias for dependency injection
Auditor = Annotated[AuditLogger, Depends(get_audit_logger)]
