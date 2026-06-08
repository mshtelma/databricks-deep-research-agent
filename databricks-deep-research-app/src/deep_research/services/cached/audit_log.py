"""Cache-backed ``IAuditLogService`` — append-only audit log via ``StorageStack``.

Audit log entries are written to the ``audit_log`` append-only table using
``_append_event``. This is fire-and-forget; the ``WriteQueue`` batches and
flushes the rows on its next tick.

DDL columns (both backends):
    user_id    TEXT
    action     TEXT
    target_id  TEXT  (nullable UUID as string)
    details    TEXT  (JSON blob)
    ts         TIMESTAMP

The ``log`` method never raises on transient errors — it suppresses and logs
any exception so audit failures never break the user-facing request.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from deep_research.services._cached_base import _CachedServiceBase
from deep_research.services._protocols import IAuditLogService

if TYPE_CHECKING:
    from deep_research.storage.factory import StorageStack

logger = logging.getLogger(__name__)

_TABLE = "audit_log"


class CachedAuditLogService(_CachedServiceBase, IAuditLogService):
    """``IAuditLogService`` backed by ``StorageStack`` append-only table.

    Thread-safety: ``_append_event`` is non-blocking and enqueues the row into
    the ``WriteQueue`` buffer. Concurrent calls are safe; ordering within the
    same process tick is FIFO by enqueue order.
    """

    _service_name = "audit_log"

    def __init__(self, stack: StorageStack) -> None:
        super().__init__(stack)

    async def log(
        self,
        user_id: str,
        action: str,
        target_id: UUID | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append one audit entry. Never raises on transient errors.

        Thread-safety: non-blocking enqueue; concurrent calls are safe.
        """
        try:
            row: dict[str, Any] = {
                "user_id": user_id,
                "action": action,
                "target_id": str(target_id) if target_id is not None else None,
                "details": json.dumps(details or {}, default=str),
                "ts": datetime.now(UTC).isoformat(),
            }
            self._append_event(_TABLE, row)
        except Exception:
            logger.warning(
                "AUDIT_LOG_ENQUEUE_FAILED user_id=%s action=%s",
                user_id,
                action,
                exc_info=True,
            )
