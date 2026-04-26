"""ChatMemoryFile SQLAlchemy model — per-chat preprocessing metadata for uploaded files."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from deep_research.db.base import BaseModel


class ChatMemoryFile(BaseModel):
    """Preprocessing state + one-line summary for an uploaded file, per chat.

    Staleness check: if `uploaded_files.updated_at > preprocessed_at`, the
    file should be re-preprocessed.
    """

    __tablename__ = "chat_memory_files"

    chat_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("chats.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("uploaded_files.id", ondelete="CASCADE"),
        nullable=False,
    )
    one_line_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    entity_ids: Mapped[list[UUID]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )
    preprocessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("idx_cmfi_chat", "chat_id"),
        Index("idx_cmfi_chat_file", "chat_id", "file_id", unique=True),
    )
