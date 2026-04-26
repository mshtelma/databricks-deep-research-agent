"""ChatMemoryCoverage SQLAlchemy model — what's been explored and how deeply."""

from uuid import UUID

from sqlalchemy import ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from deep_research.db.base import BaseModel
from deep_research.models.enums import CoverageDepth, CoverageStatus


class ChatMemoryCoverage(BaseModel):
    """Per-chat coverage map used by the reflector to assess research depth."""

    __tablename__ = "chat_memory_coverage"

    chat_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=CoverageStatus.GAP.value,
    )
    depth: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=CoverageDepth.SURFACE.value,
    )

    __table_args__ = (
        Index("idx_cmc_chat", "chat_id"),
        Index("idx_cmc_chat_topic", "chat_id", "topic", unique=True),
    )
