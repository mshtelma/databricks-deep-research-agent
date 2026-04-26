"""ChatMemoryPluginExt SQLAlchemy model — namespaced plugin extensions to chat memory."""

from typing import Any
from uuid import UUID

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from deep_research.db.base import BaseModel


class ChatMemoryPluginExt(BaseModel):
    """Per-plugin, per-chat JSONB payload written by `ContextEnricher`.

    Size-capped at `ChatMemoryService.PAYLOAD_MAX_BYTES` (64 KB) at the
    service layer — the DB column is JSONB and unbounded, so enforcement
    lives above the ORM.
    """

    __tablename__ = "chat_memory_plugin_ext"

    chat_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    plugin_name: Mapped[str] = mapped_column(String(128), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    __table_args__ = (
        Index("idx_cmpe_chat", "chat_id"),
        Index("idx_cmpe_chat_plugin", "chat_id", "plugin_name", unique=True),
    )
