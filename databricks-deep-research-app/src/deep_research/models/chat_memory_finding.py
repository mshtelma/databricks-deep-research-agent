"""ChatMemoryFinding SQLAlchemy model — durable per-chat research finding."""

from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from deep_research.db.base import BaseModel
from deep_research.models.enums import Confidence, FindingOrigin


class ChatMemoryFinding(BaseModel):
    """A structured research finding persisted with the conversation.

    Findings unify file-derived knowledge (source_step=0, origin=FILE) and
    research-derived knowledge (source_step>=1, origin in {WEB, ENTERPRISE,
    COMPUTE, PLUGIN}). Dedup via (chat_id, content_hash) unique index makes
    upserts idempotent.
    """

    __tablename__ = "chat_memory_findings"

    chat_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    research_session_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("research_sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_step: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    origin: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=FindingOrigin.WEB.value,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default=Confidence.MEDIUM.value,
    )
    entity_ids: Mapped[list[UUID]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )
    supersedes_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chat_memory_findings.id", ondelete="SET NULL"),
        nullable=True,
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        Index("idx_cmf_chat", "chat_id"),
        Index("idx_cmf_chat_hash", "chat_id", "content_hash", unique=True),
        Index("idx_cmf_chat_step", "chat_id", "source_step"),
    )
