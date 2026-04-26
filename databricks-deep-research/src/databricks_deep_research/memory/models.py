"""Pydantic projections + configuration models for the research memory layer.

These are the shapes agents and the framework see at runtime. Persistence
(SQLAlchemy ``ChatMemoryFinding`` etc.) lives in the app repo; the service
converts ORM rows <-> these projections at the hydrate/upsert boundary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class FileRef(BaseModel):
    """One row in the files_index — a lightweight reference to an uploaded file."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    filename: str
    file_type: str
    size: int
    chunk_count: int
    status: str  # matches UploadedFile.processing_status (pending/processing/ready/failed)
    one_line_summary: str = ""


class EntityRecord(BaseModel):
    """Canonical entity with aliases + back-pointers to the findings that mention it."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    name: str
    entity_type: str  # StrEnum values from deep_research.models.enums.EntityType
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)
    supporting_finding_ids: list[UUID] = Field(default_factory=list)


class KnowledgeFinding(BaseModel):
    """A durable research finding with provenance and optional supersedes link.

    PR 1 populates only file-derived findings (``source_step=0``,
    ``origin="file"``). PR 2 adds web / enterprise / compute / plugin
    findings produced by the consolidate_from_pool update loop.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID
    content: str
    confidence: Literal["high", "medium", "low"]
    source_step: int = 0
    origin: str = "web"  # StrEnum values from FindingOrigin
    entity_ids: list[UUID] = Field(default_factory=list)
    supersedes_id: UUID | None = None
    content_hash: str = ""
    created_at: datetime | None = None


class CoverageEntry(BaseModel):
    """One row in the coverage map used by the reflector."""

    model_config = ConfigDict(frozen=True)

    topic: str
    status: Literal["covered", "partial", "gap"]
    depth: Literal["surface", "moderate", "deep"]


class ChatMemorySnapshot(BaseModel):
    """Immutable projection of a chat's memory state at a point in time.

    Returned by ``ChatMemoryService.hydrate``. The service rebuilds this
    from ORM rows every turn start; agents never mutate it directly — they
    request renders via ``render(agent_type, max_chars)``.
    """

    model_config = ConfigDict(frozen=True)

    chat_id: UUID
    files: list[FileRef] = Field(default_factory=list)
    entities: list[EntityRecord] = Field(default_factory=list)
    findings: list[KnowledgeFinding] = Field(default_factory=list)
    coverage: list[CoverageEntry] = Field(default_factory=list)
    plugin_extensions: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not (self.files or self.entities or self.findings or self.coverage)


class RateLimitConfig(BaseModel):
    """Per-run token-bucket for memory LLM updates (PR 2)."""

    max_memory_updates_per_minute: int = 30
    retry_backoff_seconds: float = 1.0
    retry_max_attempts: int = 3
    circuit_breaker_threshold: int = 2


class MemoryConfig(BaseModel):
    """Workflow-level memory configuration, parsed from the ``memory:`` YAML block.

    PR 1 ships with ``enabled`` defaulted off at the workflow level; plugins
    can set ``enabled=True`` in their YAML once they reference
    ``{chat_memory_render}`` in a prompt (PR 2 feature). File-side hydration
    runs unconditionally when the chat has uploaded files — it does not
    require ``enabled=True`` because the universal spotlighted appendix is
    injected whenever memory has content.
    """

    enabled: bool = False
    update_model_tier: str = "simple"
    max_findings: int = 30
    max_ekd_chars: int = 4000
    update_after: Literal["researcher", "reflector"] = "researcher"
    finding_merge_threshold: float = 0.8
    spotlighting_mode: Literal["delimit", "datamark"] = "datamark"
    session_persistence: bool = True
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
