"""Pydantic models that define the chat-document schema.

These are the single source of truth for the on-wire JSON shape, regardless of
backend (Postgres JSONB, Delta STRING JSON, or future VARIANT). Mutation
helpers (`upsert_finding`, `add_message`, …) live on `ChatState` as plain
methods so that callers at the cache layer never touch the internal shape of
any sub-object.

Schema-evolution safety (plan section "Document pydantic models"):

* Every document-level model sets `model_config = ConfigDict(extra="ignore")`
  so unknown fields written by a future schema version do not crash today's
  hydrate. Aliasing between old and new names is expected to be rare and
  handled by `_migrate` validators.
* `ChatState._migrate` runs as a `mode="before"` validator. It reads the
  incoming `schema_version` and applies any number of registered transforms
  before Pydantic validation proper. Adding a new version is: bump
  `CURRENT_SCHEMA_VERSION`, add a branch in `_migrate`, write a unit test.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from deep_research.models.enums import (
    Confidence,
    CoverageStatus,
    EntityType,
    FindingOrigin,
)

# Incremented whenever the JSON shape on the wire changes. Old rows carry an
# older number and are upgraded lazily on read.
CURRENT_SCHEMA_VERSION: int = 1

# Compaction thresholds — enforced by `ChatState.compact_if_needed()`.
MAX_MESSAGES: int = 50
MAX_SOURCES: int = 200
MAX_RESEARCH_SESSIONS: int = 10
PLUGIN_EXT_PAYLOAD_MAX_BYTES: int = 64 * 1024  # 64 KiB per plugin
STATE_SOFT_BUDGET_BYTES: int = 1 * 1024 * 1024  # 1 MiB warning
STATE_HARD_BUDGET_BYTES: int = 4 * 1024 * 1024  # 4 MiB reject


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


# --- Leaf-node models -------------------------------------------------------


class ChatMetaEmbed(BaseModel):
    """Denormalized chat-level metadata echoed inside the document.

    The authoritative copy for list/filter queries lives in the `chat_meta`
    table (see `ChatMeta` below). This embed lets the agent see the fields
    without a second lookup.
    """

    model_config = ConfigDict(extra="ignore")

    type: str = "native"
    title: str = ""
    incognito_session_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: UUID = Field(default_factory=uuid4)
    role: str
    content: str
    ts: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Finding(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: UUID = Field(default_factory=uuid4)
    content_hash: str
    content: str
    summary: str | None = None
    step: int = 0
    origin: str = FindingOrigin.WEB.value
    confidence: str = Confidence.MEDIUM.value
    entity_ids: list[UUID] = Field(default_factory=list)
    source_ids: list[UUID] = Field(default_factory=list)
    research_session_id: UUID | None = None
    supersedes_id: UUID | None = None


class Entity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: UUID = Field(default_factory=uuid4)
    name: str
    type: str = EntityType.OTHER.value
    aliases: list[str] = Field(default_factory=list)
    supporting_finding_ids: list[UUID] = Field(default_factory=list)


class Coverage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: UUID = Field(default_factory=uuid4)
    topic: str
    status: str = CoverageStatus.GAP.value
    depth: str | None = None
    # Freshness stamps (Phase 2e) — the memory-aware routing gate (Phase 3a)
    # reads these to decide answer-from-memory vs re-research. Default-safe:
    # old documents without these keys validate to (0, None) via extra="ignore".
    as_of_turn: int = 0
    updated_at: datetime | None = None


class FileMemo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: UUID  # file_id
    name: str
    status: str = "processed"
    entity_ids: list[UUID] = Field(default_factory=list)
    summary: str | None = None


class PluginExtEntry(BaseModel):
    """Plugin-authored payload under `memory.plugin_ext[<plugin_name>]`.

    Extra fields are allowed so plugins can evolve their structured brief
    independently of framework releases.
    """

    model_config = ConfigDict(extra="allow")

    payload: dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    model_config = ConfigDict(extra="ignore")

    findings: list[Finding] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    coverage: list[Coverage] = Field(default_factory=list)
    files: list[FileMemo] = Field(default_factory=list)
    plugin_ext: dict[str, PluginExtEntry] = Field(default_factory=dict)


class Source(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: UUID = Field(default_factory=uuid4)
    url: str
    title: str | None = None
    last_used_step: int = 0
    source_type: str = "web"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchSessionState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: UUID = Field(default_factory=uuid4)
    message_id: UUID | None = None
    status: str = "in_progress"
    plan: dict[str, Any] = Field(default_factory=dict)
    observations: dict[str, Any] = Field(default_factory=dict)
    query_classification: dict[str, Any] = Field(default_factory=dict)
    execution_state: dict[str, Any] = Field(default_factory=dict)
    verification_data: dict[str, Any] = Field(default_factory=dict)
    # Value-free promotion trace (spec 6.1); None = not promotable (kept nullable,
    # intentionally NOT in the _none_to_empty_dict coercion list).
    promotion_trace: dict[str, Any] | None = None
    current_step: int = 0
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    last_heartbeat: datetime | None = None

    # Legacy chats were persisted with plan/verification_data=None because the
    # SQL ResearchSession model columns are nullable and the complete-path
    # writer passed None through. Coerce back to {} on read so hydrating a
    # pre-fix document doesn't blow up ChatState.model_validate.
    @field_validator(
        "plan",
        "observations",
        "query_classification",
        "execution_state",
        "verification_data",
        mode="before",
    )
    @classmethod
    def _none_to_empty_dict(cls, v: Any) -> Any:
        return {} if v is None else v


class UploadedFileMeta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: UUID  # file_id
    name: str
    size: int
    mime: str | None = None
    status: str = "processed"
    summary_ref: UUID | None = None  # points into Memory.files[i].id


class MessagesSummary(BaseModel):
    """Summary stub replacing messages older than `MAX_MESSAGES`."""

    model_config = ConfigDict(extra="ignore")

    content: str = ""
    covers_through_ts: datetime | None = None


# --- Top-level models -------------------------------------------------------


class ChatState(BaseModel):
    """JSON payload stored in `chat_state.state`.

    See the plan document for the canonical schema description.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: int = CURRENT_SCHEMA_VERSION
    chat: ChatMetaEmbed = Field(default_factory=ChatMetaEmbed)
    messages: list[Message] = Field(default_factory=list)
    messages_summary: MessagesSummary | None = None
    memory: Memory = Field(default_factory=Memory)
    sources: list[Source] = Field(default_factory=list)
    research_sessions: list[ResearchSessionState] = Field(default_factory=list)
    uploaded_files: list[UploadedFileMeta] = Field(default_factory=list)

    # -- Lazy-migration hook ------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _migrate(cls, data: Any) -> Any:
        """Apply any needed schema upgrades before Pydantic validation.

        Called on every load from storage. Keep migrations side-effect-free
        and fast (no network, no DB, no LLM calls).
        """
        if not isinstance(data, dict):
            return data
        version = int(data.get("schema_version", 0))
        if version < 1:
            data = _migrate_0_to_1(data)
        # Future:
        #   if version < 2: data = _migrate_1_to_2(data)
        return data

    # -- Size helpers -------------------------------------------------------

    def byte_size(self) -> int:
        """Serialized size in bytes — used to enforce soft/hard budgets."""
        return len(self.model_dump_json().encode("utf-8"))

    def is_over_hard_budget(self) -> bool:
        return self.byte_size() > STATE_HARD_BUDGET_BYTES

    def is_over_soft_budget(self) -> bool:
        return self.byte_size() > STATE_SOFT_BUDGET_BYTES

    # -- Mutation helpers (natural-key dedup + LRU eviction) ----------------

    def add_message(self, message: Message) -> None:
        """Append a message; compact older messages when over threshold."""
        self.messages.append(message)
        self.compact_messages_if_needed()

    def compact_messages_if_needed(self) -> None:
        """When `len(messages) > MAX_MESSAGES`, drop oldest and summarize.

        The summarizer is a simple "first sentence per message" fallback; a
        real summarizer can be plugged in by a future refactor — the callsite
        does not care.
        """
        if len(self.messages) <= MAX_MESSAGES:
            return
        overflow = self.messages[: len(self.messages) - MAX_MESSAGES]
        self.messages = self.messages[-MAX_MESSAGES:]
        summary_text = " ".join(_first_sentence(m.content) for m in overflow)
        covers_through = overflow[-1].ts if overflow else None
        if self.messages_summary is None:
            self.messages_summary = MessagesSummary(
                content=summary_text, covers_through_ts=covers_through
            )
        else:
            existing = self.messages_summary.content
            self.messages_summary = MessagesSummary(
                content=(existing + " " + summary_text).strip(),
                covers_through_ts=covers_through,
            )

    def upsert_finding(self, finding: Finding) -> None:
        """Replace by `content_hash`; append if absent."""
        for i, existing in enumerate(self.memory.findings):
            if existing.content_hash == finding.content_hash:
                finding.id = existing.id
                self.memory.findings[i] = finding
                return
        self.memory.findings.append(finding)

    def upsert_entity(self, entity: Entity) -> None:
        """Replace by `name`; append if absent."""
        for i, existing in enumerate(self.memory.entities):
            if existing.name == entity.name:
                entity.id = existing.id
                self.memory.entities[i] = entity
                return
        self.memory.entities.append(entity)

    def upsert_coverage(self, coverage: Coverage) -> None:
        """Replace by `topic`; append if absent."""
        for i, existing in enumerate(self.memory.coverage):
            if existing.topic == coverage.topic:
                coverage.id = existing.id
                self.memory.coverage[i] = coverage
                return
        self.memory.coverage.append(coverage)

    def upsert_file_memo(self, memo: FileMemo) -> None:
        """Replace by `id` (file_id)."""
        for i, existing in enumerate(self.memory.files):
            if existing.id == memo.id:
                self.memory.files[i] = memo
                return
        self.memory.files.append(memo)

    def upsert_plugin_ext(
        self,
        plugin_name: str,
        payload: dict[str, Any],
    ) -> None:
        """Replace the plugin's whole payload; enforces the 64 KiB cap."""
        encoded = PluginExtEntry(payload=payload).model_dump_json().encode("utf-8")
        if len(encoded) > PLUGIN_EXT_PAYLOAD_MAX_BYTES:
            raise ValueError(
                f"plugin_ext[{plugin_name!r}] payload is "
                f"{len(encoded)} bytes, exceeds "
                f"{PLUGIN_EXT_PAYLOAD_MAX_BYTES} byte cap"
            )
        self.memory.plugin_ext[plugin_name] = PluginExtEntry(payload=payload)

    def add_source(self, source: Source) -> None:
        """Dedup by URL; LRU-evict when over threshold."""
        for existing in self.sources:
            if existing.url == source.url:
                existing.last_used_step = max(
                    existing.last_used_step, source.last_used_step
                )
                existing.title = existing.title or source.title
                return
        self.sources.append(source)
        if len(self.sources) > MAX_SOURCES:
            self.sources.sort(key=lambda s: s.last_used_step, reverse=True)
            self.sources = self.sources[:MAX_SOURCES]

    def upsert_research_session(self, session: ResearchSessionState) -> None:
        """Replace by `id`; trim to `MAX_RESEARCH_SESSIONS` most recent."""
        for i, existing in enumerate(self.research_sessions):
            if existing.id == session.id:
                self.research_sessions[i] = session
                break
        else:
            self.research_sessions.append(session)
        if len(self.research_sessions) > MAX_RESEARCH_SESSIONS:
            self.research_sessions.sort(key=lambda s: s.started_at, reverse=True)
            self.research_sessions = self.research_sessions[:MAX_RESEARCH_SESSIONS]

    def get_research_session(
        self, session_id: UUID
    ) -> ResearchSessionState | None:
        """Find a research session by id. O(N) with N ≤ MAX_RESEARCH_SESSIONS."""
        for rs in self.research_sessions:
            if rs.id == session_id:
                return rs
        return None

    def upsert_uploaded_file(self, meta: UploadedFileMeta) -> None:
        for i, existing in enumerate(self.uploaded_files):
            if existing.id == meta.id:
                self.uploaded_files[i] = meta
                return
        self.uploaded_files.append(meta)

    def live_file_ids(self) -> list[UUID]:
        """List of file_ids currently referenced by the document.

        Written to the `chat_deleted_files` projection at flush time so the
        hourly cleanup job can delete orphaned `file_chunks`.
        """
        return [f.id for f in self.uploaded_files]


class ChatMeta(BaseModel):
    """Promoted-column row stored in `chat_meta`.

    Read on its own for `GET /api/v1/chats`, the chat-list UI, and the
    Hydrator's prefetch. Never joined against `chat_state` for that path.
    """

    model_config = ConfigDict(extra="ignore")

    PREVIEW_MAX_CHARS: ClassVar[int] = 120

    chat_id: UUID
    user_id: str
    title: str = ""
    preview: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    deleted_at: datetime | None = None
    version: int = 1

    @classmethod
    def preview_from_state(cls, state: ChatState) -> str:
        """Materialize the list-UI preview at flush time.

        Plan: "first 120 chars of first message, materialized at flush."
        Falls back to the messages summary if we've already compacted past
        the first message.
        """
        if state.messages:
            return state.messages[0].content[: cls.PREVIEW_MAX_CHARS]
        if state.messages_summary is not None:
            return state.messages_summary.content[: cls.PREVIEW_MAX_CHARS]
        return ""


class ChatDocument(BaseModel):
    """Unified view returned from `StorageBackend.load_chat` — the caller
    gets meta + state together even though they live in two tables.
    """

    model_config = ConfigDict(extra="ignore")

    meta: ChatMeta
    state: ChatState

    @classmethod
    def new(cls, chat_id: UUID, user_id: str, *, title: str = "") -> ChatDocument:
        """Construct a fresh document for a chat that doesn't exist yet.

        Used when `load_chat` returns None and the caller wants an empty
        in-memory snapshot to start mutating.
        """
        now = _utcnow()
        return cls(
            meta=ChatMeta(
                chat_id=chat_id,
                user_id=user_id,
                title=title,
                created_at=now,
                updated_at=now,
                version=0,  # 0 until first successful flush
            ),
            state=ChatState(chat=ChatMetaEmbed(title=title)),
        )


class UserDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    profile: dict[str, Any] = Field(default_factory=dict)
    preferences: dict[str, Any] = Field(default_factory=dict)
    data_sources: list[dict[str, Any]] = Field(default_factory=list)


class PrepJobDocument(BaseModel):
    """Plugin-owned (sapresalesbot) background-job document.

    `heartbeat` is a top-level column on the underlying table (not inside the
    `state` payload) so heartbeat updates bypass the WriteQueue — see plan
    section "PrepJob heartbeat".
    """

    model_config = ConfigDict(extra="ignore")

    prep_job_id: UUID
    account_id: str
    status: str = "queued"
    heartbeat: datetime = Field(default_factory=_utcnow)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    query: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    worker: str | None = None
    timings: dict[str, Any] = Field(default_factory=dict)


# --- Migration registry -----------------------------------------------------


def _migrate_0_to_1(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade pre-versioned documents to schema_version=1.

    Pre-v1 rows have no `schema_version` key; we simply stamp it. All other
    fields map straight through because v1 is the first versioned shape.
    """
    data = dict(data)
    data["schema_version"] = 1
    return data


# --- Private helpers --------------------------------------------------------


def _first_sentence(text: str) -> str:
    """Return the first sentence, stripped. Bounded to 200 chars.

    Deliberately naive: splits on the first `. `, `? `, or `! `. Only used
    for the fallback message summarizer in `ChatState.compact_messages_if_needed`.
    """
    text = text.strip()
    if not text:
        return ""
    for terminator in (". ", "? ", "! "):
        idx = text.find(terminator)
        if 0 < idx < 200:
            return text[: idx + 1]
    return text[:200]
