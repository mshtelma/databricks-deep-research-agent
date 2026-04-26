"""ChatMemoryEntity SQLAlchemy model — per-chat entity registry."""

from uuid import UUID

from sqlalchemy import ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from deep_research.db.base import BaseModel
from deep_research.models.enums import EntityType


class ChatMemoryEntity(BaseModel):
    """Canonical entity record tracked across the whole chat."""

    __tablename__ = "chat_memory_entities"

    chat_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    entity_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=EntityType.OTHER.value,
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    aliases: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )
    supporting_finding_ids: Mapped[list[UUID]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    __table_args__ = (
        Index("idx_cme_chat", "chat_id"),
        Index("idx_cme_chat_name", "chat_id", "name", unique=True),
        Index("idx_cme_chat_type", "chat_id", "entity_type"),
    )
