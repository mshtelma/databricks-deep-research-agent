"""Cache-backed `IFeedbackService`.

Persistence model: `message_feedback` is a list-table (CRUD on individual
rows), not a chat-scoped append-only event buffer — the legacy service
supports update/delete by feedback_id. We therefore route through the
backend's `list_rows` / `upsert_row` / `delete_row` APIs rather than the
event queue.

Behavioral differences from `SQLAlchemyFeedbackService` (documented):

* **No message-existence check.** The legacy impl reads the `messages` row
  to verify the referenced message exists before inserting feedback. Under
  the cached model, messages live inside `ChatState.messages[]` — there's
  no global `messages` table to query, and the caller would need to supply
  a `chat_id` to look them up. The cached impl trusts the caller; orphan
  feedback rows are harmless.
* **MLflow span best-effort.** Preserved verbatim from the legacy impl —
  feedback persistence still succeeds if MLflow logging raises.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from deep_research.services._cached_base import _CachedServiceBase
from deep_research.services._protocols import IFeedbackService

if TYPE_CHECKING:
    from deep_research.storage.factory import StorageStack

logger = logging.getLogger(__name__)


_VALID_RATINGS = {"positive", "negative"}

_TABLE = "message_feedback"


class CachedFeedbackService(_CachedServiceBase, IFeedbackService):
    """`IFeedbackService` implementation over a `StorageStack` list-table."""

    def __init__(self, stack: "StorageStack") -> None:
        super().__init__(stack)

    async def create_feedback(
        self,
        message_id: UUID,
        user_id: str,
        rating: str,
        feedback_text: str | None = None,
        feedback_category: str | None = None,
    ) -> SimpleNamespace:
        if rating not in _VALID_RATINGS:
            raise ValueError(f"Invalid rating: {rating}")

        existing = await self._stack.backend.list_rows(
            _TABLE,
            where={"message_id": message_id, "user_id": user_id},
            limit=1,
        )
        if existing:
            raise ValueError("Feedback already exists for this message")

        feedback_id = uuid4()
        now = datetime.now(UTC)
        row = {
            "feedback_id": feedback_id,
            "message_id": message_id,
            "user_id": user_id,
            "ts": now,
            "rating": rating,
            "feedback_text": feedback_text,
            "feedback_category": feedback_category,
            # `feedback` column on the Delta DDL is a STRING-JSON blob; carry
            # both the flat fields (Lakebase list-table) and the composite
            # JSON so the Warehouse append_events path stays lossless.
            "feedback": {
                "rating": rating,
                "feedback_text": feedback_text,
                "feedback_category": feedback_category,
            },
        }
        await self._stack.backend.upsert_row(
            _TABLE, row, pk="feedback_id"
        )
        self._log_mlflow_best_effort(message_id, rating, feedback_category)
        return SimpleNamespace(id=feedback_id, **{
            k: v for k, v in row.items() if k != "feedback_id"
        })

    async def get_feedback(
        self,
        message_id: UUID,
        user_id: str,
    ) -> SimpleNamespace | None:
        rows = await self._stack.backend.list_rows(
            _TABLE,
            where={"message_id": message_id, "user_id": user_id},
            limit=1,
        )
        if not rows:
            return None
        return _row_to_namespace(rows[0])

    async def get_feedback_for_message(
        self, message_id: UUID
    ) -> list[SimpleNamespace]:
        rows = await self._stack.backend.list_rows(
            _TABLE, where={"message_id": message_id}
        )
        return [_row_to_namespace(r) for r in rows]

    async def update_feedback(
        self,
        feedback_id: UUID,
        user_id: str,
        rating: str | None = None,
        feedback_text: str | None = None,
        feedback_category: str | None = None,
    ) -> SimpleNamespace | None:
        rows = await self._stack.backend.list_rows(
            _TABLE, where={"feedback_id": feedback_id, "user_id": user_id}, limit=1
        )
        if not rows:
            return None
        row = dict(rows[0])
        if rating is not None:
            if rating not in _VALID_RATINGS:
                raise ValueError(f"Invalid rating: {rating}")
            row["rating"] = rating
        if feedback_text is not None:
            row["feedback_text"] = feedback_text
        if feedback_category is not None:
            row["feedback_category"] = feedback_category
        row["feedback"] = {
            "rating": row.get("rating"),
            "feedback_text": row.get("feedback_text"),
            "feedback_category": row.get("feedback_category"),
        }
        await self._stack.backend.upsert_row(_TABLE, row, pk="feedback_id")
        return _row_to_namespace(row)

    async def delete_feedback(
        self,
        feedback_id: UUID,
        user_id: str,
    ) -> bool:
        rows = await self._stack.backend.list_rows(
            _TABLE, where={"feedback_id": feedback_id, "user_id": user_id}, limit=1
        )
        if not rows:
            return False
        await self._stack.backend.delete_row(
            _TABLE, feedback_id, pk="feedback_id"
        )
        return True

    async def get_message_feedback_stats(
        self,
        message_id: UUID,
    ) -> dict[str, Any]:
        rows = await self._stack.backend.list_rows(
            _TABLE, where={"message_id": message_id}
        )
        pos = sum(1 for r in rows if r.get("rating") == "positive")
        neg = sum(1 for r in rows if r.get("rating") == "negative")
        return {"positive_count": pos, "negative_count": neg, "total": pos + neg}

    def _log_mlflow_best_effort(
        self, message_id: UUID, rating: str, category: str | None
    ) -> None:
        try:
            import mlflow  # deferred — optional dep

            with mlflow.start_span(name="user_feedback", span_type="UNKNOWN") as span:
                span.set_attributes({
                    "feedback.message_id": str(message_id),
                    "feedback.rating": rating,
                    "feedback.category": category or "none",
                    "feedback.timestamp": datetime.now(UTC).isoformat(),
                })
        except Exception as exc:  # noqa: BLE001 — MLflow must never fail the write
            logger.warning("MLFLOW_FEEDBACK_LOG_FAILED error=%s", exc)


def _row_to_namespace(row: dict[str, Any]) -> SimpleNamespace:
    """Map a stored row back to an ORM-shape attribute namespace."""
    return SimpleNamespace(
        id=row.get("feedback_id"),
        message_id=row.get("message_id"),
        user_id=row.get("user_id"),
        rating=row.get("rating"),
        feedback_text=row.get("feedback_text"),
        feedback_category=row.get("feedback_category"),
        created_at=row.get("ts"),
    )
