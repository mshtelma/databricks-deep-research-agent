"""Typed Protocols for every service that touches persistence.

Wave 4 of the storage cutover (see the finalization plan). Consumers depend on
the `I*` Protocols, not on concrete classes. Both the legacy SQLAlchemy
implementations and the new cache-backed implementations satisfy the
Protocol so we can flip `settings.storage_service_impl` without any caller
churn. Runtime checking is enabled so tests can `isinstance(x, IFoo)`.

Service Protocols in this file are the **minimum stable contract** that
callers rely on. Internal helpers (e.g. `_upsert_finding` private methods)
are not part of the Protocol — they live on concrete classes only.

Porting status (as of 2026-04-22):

* Landed: `IFeedbackService`, `IAuditLogService`, `IResearchEventService`.
* Stubbed (see docstring TODOs): the 12 other service interfaces. Each is a
  follow-up PR that extracts the concrete class's public methods, drops the
  signatures here with `...` bodies, renames the class to `SQLAlchemy*`, and
  adds a back-compat alias.

The pattern for extracting a new Protocol:

1. Read the target service's public methods (anything not `_`-prefixed).
2. Copy the signature (types unchanged) into a `Protocol` class here.
3. Drop the body; add a one-line docstring.
4. Rename the concrete class `XService` → `SQLAlchemyXService`.
5. At the bottom of the concrete module, add `XService = SQLAlchemyXService`
   so existing imports still work.
6. Update consumer type hints from `XService` → `IXService`.
"""

from __future__ import annotations

from builtins import list as builtin_list
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

# --- Append-only Protocols (simplest — start here) ------------------------


@runtime_checkable
class IFeedbackService(Protocol):
    """User feedback on agent messages.

    Cache-backed implementations treat this as a list-table (CRUD on rows)
    rather than append-only — the legacy surface supports update/delete.
    """

    async def create_feedback(
        self,
        message_id: UUID,
        user_id: str,
        rating: str,
        feedback_text: str | None = None,
        feedback_category: str | None = None,
    ) -> Any: ...
    async def get_feedback(self, message_id: UUID, user_id: str) -> Any | None: ...
    async def get_feedback_for_message(
        self, message_id: UUID
    ) -> list[Any]: ...
    async def update_feedback(
        self,
        feedback_id: UUID,
        user_id: str,
        rating: str | None = None,
        feedback_text: str | None = None,
        feedback_category: str | None = None,
    ) -> Any | None: ...
    async def delete_feedback(
        self, feedback_id: UUID, user_id: str
    ) -> bool: ...
    async def get_message_feedback_stats(
        self, message_id: UUID
    ) -> dict[str, Any]: ...


@runtime_checkable
class IAuditLogService(Protocol):
    """Security / compliance audit log. Append-only."""

    async def log(
        self,
        user_id: str,
        action: str,
        target_id: UUID | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append one audit entry. Never raises on transient errors."""
        ...


@runtime_checkable
class IResearchEventService(Protocol):
    """Streaming research-event persistence (SSE audit log)."""

    async def save_event(
        self,
        research_session_id: UUID,
        event_type: str,
        payload: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> Any: ...
    async def save_events_batch(
        self,
        research_session_id: UUID,
        events: Sequence[dict[str, Any]],
    ) -> int: ...
    async def get_events_for_session(
        self,
        research_session_id: UUID,
        *,
        event_types: list[str] | None = None,
        limit: int | None = None,
    ) -> list[Any]: ...
    async def get_events_since_sequence(
        self,
        research_session_id: UUID,
        since_sequence: int,
        limit: int = 100,
    ) -> list[Any]: ...
    def event_to_dict(self, event: Any) -> dict[str, Any]: ...
    def events_to_list(self, events: list[Any]) -> list[dict[str, Any]]: ...


# --- Cold-path list-table Protocols ----------------------------------------


@runtime_checkable
class ITemplateService(Protocol):
    """Prompt template CRUD."""

    async def create_template(
        self,
        owner_id: str,
        name: str,
        template_type: Any,
        content: str,
        description: str | None = None,
        variables: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        visibility: Any = "private",
        is_default: bool = False,
    ) -> Any: ...

    async def get_for_user(self, template_id: UUID, user_id: str) -> Any | None: ...
    async def get_accessible(self, template_id: UUID, user_id: str) -> Any | None: ...
    async def get_by_name(self, owner_id: str, name: str) -> Any | None: ...
    async def get_accessible_templates(
        self,
        user_id: str,
        template_type: Any | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Any], int]: ...
    async def search_by_tags(
        self, user_id: str, tags: list[str], limit: int = 50
    ) -> list[Any]: ...
    async def get_default_template(
        self, owner_id: str, template_type: Any
    ) -> Any | None: ...
    async def set_default_template(
        self, template_id: UUID, owner_id: str
    ) -> Any | None: ...
    async def set_as_default(
        self, template_id: UUID, owner_id: str, type_: Any
    ) -> None:
        """Atomically set ``template_id`` as the unique default for
        ``(owner_id, type_)``. Flips exactly one row's ``is_default`` to True
        and unsets any others in the same operation. Race-free in legacy
        (single SQL UPDATE); best-effort in cached (sequential cold-path
        upserts under the same process)."""
        ...
    async def update_template(self, template: Any) -> Any: ...
    async def update(self, template: Any) -> Any: ...
    async def delete_template(self, template: Any) -> None: ...
    async def delete(self, template: Any) -> None: ...
    async def commit(self) -> None:
        """Durably persist any pending writes for this service.

        Legacy impl flushes + commits the underlying SQLAlchemy session.
        Cached impl is a no-op because cold-path list-table writes are
        already synchronous via ``_cold_upsert_row`` / ``_cold_delete_row``.

        Raises whatever the underlying transaction raises; callers must
        not silently swallow these errors.
        """
        ...
    def render_template(
        self, template: Any, variables: dict[str, Any]
    ) -> tuple[str, list[str], list[str]]: ...
    def validate_variables(
        self, template: Any, variables: dict[str, Any]
    ) -> list[str]: ...


@runtime_checkable
class ICustomAgentService(Protocol):
    """Custom agent CRUD + preset-step management.

    Cached impl stores agents in the ``custom_agents`` list table
    (``_cold_list_rows`` / ``_cold_upsert_row`` / ``_cold_delete_row``) and
    denormalizes preset steps into ``custom_agents.steps`` (JSONB array).
    Legacy impl uses the ``custom_agents`` + ``agent_preset_steps`` ORM tables.
    """

    async def create_agent(
        self,
        owner_id: str,
        name: str,
        description: str | None = None,
        avatar_url: str | None = None,
        system_prompt_template_id: UUID | None = None,
        synthesis_template_id: UUID | None = None,
        source_scope: str = "all",
        enabled_sources: list[str] | None = None,
        disabled_sources: list[str] | None = None,
        use_planner: bool = True,
        default_depth: str = "medium",
        default_mode: str = "planner",
        enable_clarification: bool = True,
        output_format: str = "markdown",
        output_schema: dict[str, Any] | None = None,
        visibility: str = "private",
        preset_steps: list[dict[str, Any]] | None = None,
        model_overrides: dict[str, str] | None = None,
        domain_filter_mode: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> Any: ...

    async def get_for_user(self, agent_id: UUID, user_id: str) -> Any | None: ...
    async def get_accessible(self, agent_id: UUID, user_id: str) -> Any | None: ...
    async def get_by_name(self, owner_id: str, name: str) -> Any | None: ...
    async def get_system_agents(self) -> list[Any]: ...
    async def get_workspace_agents(self) -> list[Any]: ...
    async def resolve_agent_for_request(
        self,
        user_id: str,
        agent_id: UUID | None = None,
        agent_name: str | None = None,
    ) -> Any | None: ...

    async def get_accessible_agents(
        self,
        user_id: str,
        visibility: str | None = None,
        source_scope: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Any], int]: ...

    async def update(self, agent: Any) -> Any: ...
    async def delete(self, agent: Any) -> None: ...

    # -- Preset step management ---
    async def get_agent_preset_steps(self, agent_id: UUID) -> list[Any]: ...
    async def get_preset_step(self, step_id: UUID, agent_id: UUID) -> Any | None: ...
    async def create_preset_step(
        self,
        agent_id: UUID,
        title: str,
        description: str | None = None,
        order: int = 1,
        is_required: bool = True,
        source_hints: dict[str, Any] | None = None,
        source_scope: str | None = None,
    ) -> Any: ...
    async def update_preset_step(self, step: Any) -> Any: ...
    async def delete_preset_step(self, step: Any) -> None: ...
    async def reorder_preset_steps(
        self,
        agent_id: UUID,
        step_order: list[UUID],
    ) -> list[Any]: ...


@runtime_checkable
class IPreferencesService(Protocol):
    """User preferences (tucked inside `UserDocument.preferences` in the
    cached impl; lives in its own table in legacy).
    """

    async def get_preferences(self, user_id: str) -> Any: ...
    async def update_preferences(
        self,
        user_id: str,
        system_instructions: str | None = None,
        default_research_depth: Any | None = None,
        default_query_mode: Any | None = None,
        theme: str | None = None,
        notifications_enabled: bool | None = None,
    ) -> Any: ...
    async def get_system_instructions(self, user_id: str) -> str | None: ...
    async def get_default_research_depth(self, user_id: str) -> Any: ...
    async def get_default_query_mode(self, user_id: str) -> str: ...
    def to_dict(self, preferences: Any) -> dict[str, Any]: ...


@runtime_checkable
class IDataSourceService(Protocol):
    """User data sources — CRUD + OBO validation for enterprise data sources."""

    async def create_vector_search_source(
        self,
        owner_id: str,
        user_token: str,
        name: str,
        endpoint_name: str,
        index_name: str,
        description: str | None = None,
        visibility: Any = None,
        enable_hybrid: bool = True,
        enable_reranking: bool = True,
        num_results: int = 10,
    ) -> tuple[Any, str | None]: ...

    async def create_genie_source(
        self,
        owner_id: str,
        user_token: str,
        name: str,
        space_id: str,
        description: str | None = None,
        example_questions: list[str] | None = None,
        visibility: Any = None,
    ) -> tuple[Any, str | None]: ...

    async def create_assistant_source(
        self,
        owner_id: str,
        user_token: str,
        name: str,
        endpoint_name: str,
        description: str | None = None,
        pass_context: bool = True,
        visibility: Any = None,
    ) -> tuple[Any, str | None]: ...

    async def get_accessible_sources(
        self,
        user_id: str,
        source_type: Any = None,
        only_valid: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Any], int]: ...

    async def get_for_user(self, source_id: UUID, user_id: str) -> Any | None: ...

    async def get_accessible(self, source_id: UUID, user_id: str) -> Any | None: ...

    async def revalidate_source(
        self,
        source: Any,
        user_token: str,
    ) -> tuple[bool, str | None]: ...

    async def mark_sources_expired(self, owner_id: str) -> int: ...

    async def update(self, source: Any) -> Any: ...

    async def delete(self, source: Any) -> None: ...


@runtime_checkable
class IUserService(Protocol):
    """User identity CRUD."""

    async def upsert(
        self,
        user_id: str,
        email: str | None,
        display_name: str | None,
    ) -> None: ...
    async def resolve_user_ids(
        self,
        user_ids: list[str],
    ) -> dict[str, tuple[str | None, str | None]]: ...


# --- Chat dataclass -------------------------------------------------------


@dataclass
class ChatFullView:
    """Read-only bundle of a chat with all related objects.

    Returned by `IChatService.get_full`. Callers build the full response
    from these objects without additional round-trips.

    Attributes:
        chat: A chat-like object exposing ``id``, ``user_id``, ``title``,
            ``status``, ``chat_type``, ``created_at``, ``updated_at``,
            ``deleted_at``. May be a legacy ORM Chat or a cached ChatView.
        messages: All messages in the chat, ordered by creation time.
        research_sessions: Research sessions linked to this chat.
        sources: All sources associated with this chat.
    """

    chat: Any
    messages: list[Any] = field(default_factory=list)
    research_sessions: list[Any] = field(default_factory=list)
    sources: list[Any] = field(default_factory=list)


@runtime_checkable
class IChatService(Protocol):
    """Chat conversation CRUD.

    Cached impl stores the canonical chat doc in `ChatDocument` (meta + state).
    The legacy impl reads from the `public.chats` ORM table.

    All returned `Chat`-like objects expose at minimum: ``id``, ``user_id``,
    ``title``, ``status``, ``chat_type``, ``created_at``, ``updated_at``,
    ``deleted_at``. The cached impl synthesizes a `ChatView` that mirrors
    the legacy `Chat` ORM attribute surface so callers are unaffected.

    Thread-safety: concurrent calls on different `chat_id` values are safe.
    Calls sharing a `chat_id` are serialized by the per-chat lock inside
    `_mutate_chat`.
    """

    async def create(self, user_id: str, title: str | None = None) -> Any:
        """Create a new chat and return a Chat-like object."""
        ...

    async def get_for_user(self, chat_id: UUID, user_id: str) -> Any | None:
        """Return chat owned by user, or None if not found / not owned."""
        ...

    async def get_by_id(self, chat_id: UUID) -> Any | None:
        """Return chat by PK without ownership check; None if absent."""
        ...

    async def get_full(self, chat_id: UUID, user_id: str) -> Any | None:
        """Return fully loaded chat (messages + sessions + sources).

        Legacy impl returns a `Chat` ORM object with relationships loaded.
        Cached impl returns a `ChatFullView` bundle.
        """
        ...

    async def list(
        self,
        user_id: str,
        status: Any | None = None,
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
    ) -> tuple[list[Any], int]:
        """Return ``(chats, total_count)`` for the user.

        ``status`` accepts ``ChatStatus`` enum values or string equivalents.
        ``search`` is case-insensitive substring match on title.
        """
        ...

    async def update_chat(
        self,
        chat_id: UUID,
        user_id: str,
        title: str | None = None,
        status: Any | None = None,
        surface_state_patch: dict[str, Any] | None = None,
    ) -> Any | None:
        """Update title and/or status; return updated chat or None.

        ``surface_state_patch`` is a per-agent shallow merge into
        ``metadata["surface_state"]``. Each agent entry is shallow-merged;
        ``action_runs`` within each entry uses newest-updated_at-wins with
        idempotent-replay semantics. See
        ``services.storage.surface_state.merge_surface_state`` for details.
        """
        ...

    async def soft_delete(self, chat_id: UUID, user_id: str) -> bool:
        """Soft-delete chat; return True if deleted, False if not found."""
        ...

    async def restore(self, chat_id: UUID, user_id: str) -> Any | None:
        """Restore soft-deleted chat; return restored chat or None."""
        ...

    async def update_title_from_message(
        self, chat_id: UUID, message_content: str
    ) -> None:
        """Set title from message body if title is currently empty."""
        ...

    async def purge_deleted_chats(self, days_old: int = 30) -> int:
        """Permanently remove chats soft-deleted more than ``days_old`` ago."""
        ...

    async def list_incognito_for_session(self, session_id: UUID) -> builtin_list[Any]:
        """Return incognito chats associated with the given session UUID."""
        ...

    async def add(self, chat: Any) -> Any:
        """Persist a pre-built Chat ORM object (used by incognito path)."""
        ...

    async def update(self, chat: Any) -> Any:
        """Persist changes to an already-loaded Chat object."""
        ...

    async def commit(self) -> None:
        """Durably persist any pending writes for this service.

        Legacy impl flushes + commits the underlying SQLAlchemy session.
        Cached impl is a no-op — chat-state writes go through the
        WriteQueue with documented eventual consistency, and direct
        backend writes (incognito session etc.) are already synchronous.

        Raises whatever the underlying transaction raises; callers must
        not silently swallow these errors.
        """
        ...


# --- Chat-scoped Protocols ------------------------------------------------


@runtime_checkable
class IChatMemoryService(Protocol):
    """Chat-scoped memory (findings, entities, coverage, files, plugin_ext).

    The plugin-critical service — the sapresalesbot `ContextEnricher`
    depends on the exact `enrich_scope` + `render` semantics via
    `feedback_render_must_include_plugin_extensions`. Protocol extraction
    must preserve that contract byte-for-byte.

    TODO: extract full signatures from `services/chat_memory_service.py`:
    `upsert_plugin_ext`, `enrich_scope`, `render`, `account_candidates`,
    `search_findings`.
    """

    async def hydrate(
        self,
        chat_id: UUID,
        *,
        user_id: str | None = None,
    ) -> Any:
        """Load chat-scoped memory rows and return the current projection."""
        ...

    def snapshot(self) -> Any:
        """Return the current in-memory projection."""
        ...

    async def preprocess_new_files(
        self,
        chat_id: UUID,
        file_ids: Iterable[UUID],
        *,
        file_service: Any = None,
        head_chars: int = 4000,
        research_session_id: UUID | None = None,
    ) -> Any:
        """Extract entities/facts/summaries for newly attached files."""
        ...

    def render_appendix_block(
        self,
        agent_type: str = "coordinator",
        max_chars: int = 3500,
        mode: Any = None,
    ) -> str:
        """Render chat memory for prompt appendix injection."""
        ...

    async def consolidate_from_pool(
        self,
        chat_id: UUID,
        *,
        claims: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        research_session_id: UUID | None,
        source_step: int,
        origin: str = "web",
        coverage_topics: list[dict[str, Any]] | None = None,
    ) -> int:
        """Persist a finished turn's verified knowledge as durable findings
        (and per-topic coverage rows, on the cached path)."""
        ...


@runtime_checkable
class ISessionService(Protocol):
    """Incognito-session lifecycle (browser-tab-scoped ephemeral chats)."""

    async def get_by_token(self, session_token: str) -> Any | None: ...
    async def get_by_token_for_user(
        self, session_token: str, user_id: str
    ) -> Any | None: ...
    async def get(self, session_id: UUID) -> Any | None: ...
    async def get_or_create_session(
        self, user_id: str, session_token: str | None = None
    ) -> tuple[Any, str, bool]: ...
    async def update(self, session: Any) -> Any: ...
    async def touch_session(self, session_id: UUID) -> Any | None: ...
    async def count_incognito_chats(self, session_id: UUID) -> int: ...
    async def can_create_chat(self, session_id: UUID) -> bool: ...
    async def cleanup_expired(self) -> int: ...
    async def get_session_status(
        self, session_token: str | None, user_id: str
    ) -> dict[str, Any]: ...


@runtime_checkable
class IChatSourcePoolService(Protocol):
    """Chat-scoped source URL pool — BM25 + semantic search over per-chat sources.

    The legacy ``ChatSourcePoolService`` satisfies this Protocol structurally.
    The coordinator uses only ``build_search_index`` and ``search``; both are
    included here so the type-checker can validate call sites without importing
    the concrete class (which requires a live DB session).

    Architecture note (F-OTHER.3): the coordinator receives a
    ``ChatSourcePoolService | None`` from its caller. Under the cached path the
    caller may pass ``None`` (no pool available); the coordinator gracefully
    falls back to direct-context mode. A full cached impl is deferred until the
    orchestrator wires the cached source list into the coordinator's call site.
    """

    async def build_search_index(
        self,
        chat_id: UUID,
        compute_embeddings: bool = True,
    ) -> object:
        """Build an in-memory hybrid BM25 + embedding search index."""
        ...

    async def search(
        self,
        query: str,
        limit: int = 10,
        query_embedding: object = None,
    ) -> list[Any]:
        """Search sources with hybrid BM25 + vector similarity.

        Returns Source-like objects sorted by hybrid score.
        """
        ...


@runtime_checkable
class IMessageService(Protocol):
    """Chat message CRUD.

    Cached impl stores messages inside `ChatState.messages[]`, so every
    method is chat-scoped. Methods that historically took only
    `message_id` (`update_content`, `set_research_session`) now require
    `chat_id` as well — callers know the chat (it's in the route path).
    """

    async def create(
        self,
        chat_id: UUID,
        role: Any,
        content: str,
    ) -> Any: ...
    async def get_with_chat(
        self,
        message_id: UUID,
        chat_id: UUID,
    ) -> Any | None: ...
    async def list_messages(
        self,
        chat_id: UUID,
        limit: int = 100,
        offset: int = 0,
        before: datetime | None = None,
    ) -> tuple[list[Any], int]: ...
    async def update_content(
        self,
        message_id: UUID,
        content: str,
        *,
        chat_id: UUID,
    ) -> Any | None: ...
    async def delete_subsequent(
        self,
        chat_id: UUID,
        after: datetime,
    ) -> int: ...
    async def set_research_session(
        self,
        message_id: UUID,
        research_session_id: UUID,
        *,
        chat_id: UUID,
    ) -> Any | None: ...
    async def get_conversation_history(
        self,
        chat_id: UUID,
        limit: int = 20,
    ) -> list[dict[str, str]]: ...


@runtime_checkable
class ISourceService(Protocol):
    """Source (URL) CRUD. Chat-scoped under the cached impl.

    In the new schema, sources live under `ChatState.sources[]` with
    dedup-by-URL. The legacy `research_session_id` scoping is preserved
    as metadata on the Source entry so per-session views still work.
    """

    async def create(
        self,
        research_session_id: UUID,
        url: str,
        title: str | None = None,
        snippet: str | None = None,
        content: str | None = None,
        relevance_score: float | None = None,
        source_type: str = "web",
        source_metadata: dict[str, Any] | None = None,
        *,
        chat_id: UUID,
    ) -> Any: ...
    async def create_many(
        self,
        research_session_id: UUID,
        sources: list[dict[str, Any]],
        *,
        chat_id: UUID,
    ) -> list[Any]: ...
    async def list_by_session(
        self,
        research_session_id: UUID,
        limit: int = 100,
        *,
        chat_id: UUID | None = None,
    ) -> list[Any]: ...
    async def update_content(
        self,
        source_id: UUID,
        content: str,
        *,
        chat_id: UUID,
    ) -> Any | None: ...
    async def get_by_url(
        self,
        research_session_id: UUID,
        url: str,
        *,
        chat_id: UUID,
    ) -> Any | None: ...


@runtime_checkable
class IResearchSessionService(Protocol):
    """Research session lifecycle. Chat-scoped under the cached impl.

    Legacy `create(message_id, query, …)` derives chat_id from the Message
    row. Cached impl must receive chat_id explicitly.
    """

    async def create(
        self,
        message_id: UUID,
        query: str,
        research_depth: str = "auto",
        query_classification: dict[str, Any] | None = None,
        *,
        chat_id: UUID,
    ) -> Any: ...
    async def get(self, session_id: UUID, *, chat_id: UUID) -> Any | None: ...
    async def get_by_message(
        self, message_id: UUID, *, chat_id: UUID
    ) -> Any | None: ...
    async def get_active_session_by_chat(
        self, chat_id: UUID, user_id: str
    ) -> Any | None: ...
    async def update_plan(
        self,
        session_id: UUID,
        plan: dict[str, Any],
        *,
        chat_id: UUID,
    ) -> Any | None: ...
    async def add_observation(
        self,
        session_id: UUID,
        observation: str,
        step_index: int,
        *,
        chat_id: UUID,
    ) -> Any | None: ...
    async def add_reasoning_step(
        self,
        session_id: UUID,
        step_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        chat_id: UUID,
    ) -> Any | None: ...
    async def complete(
        self,
        session_id: UUID,
        error_message: str | None = None,
        *,
        chat_id: UUID,
    ) -> Any | None: ...
    async def cancel(
        self,
        session_id: UUID,
        *,
        chat_id: UUID,
    ) -> Any | None: ...
    async def update_classification(
        self,
        session_id: UUID,
        classification: dict[str, Any],
        *,
        chat_id: UUID,
    ) -> Any | None: ...


@runtime_checkable
class IFileUploadService(Protocol):
    """Uploaded file metadata + chunks.

    Cached impl stores metadata in the ``uploaded_files`` list table and
    chunks in the ``file_chunks`` append-only table (batched at 1000 rows/call).
    Legacy impl uses the ``uploaded_files`` + ``file_chunks`` ORM tables.
    """

    def validate_file(
        self,
        filename: str,
        file_size: int,
        content_type: str | None = None,
    ) -> tuple[bool, str | None, Any]: ...

    async def validate_session_quota(
        self,
        owner_id: str,
        session_id: UUID | None,
        new_file_size: int,
    ) -> tuple[bool, str | None]: ...

    async def upload_file(
        self,
        owner_id: str,
        filename: str,
        file_content: Any,
        file_size: int,
        session_id: UUID | None = None,
        content_type: str | None = None,
    ) -> tuple[Any | None, str | None]: ...

    async def process_file(self, file_id: UUID) -> tuple[bool, str | None]: ...

    async def get_session_files(
        self,
        owner_id: str,
        session_id: UUID | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Any], int]: ...

    async def get_for_user(self, file_id: UUID, owner_id: str) -> Any | None: ...
    async def get(self, file_id: UUID) -> Any | None: ...

    async def get_file_chunks(
        self,
        file_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Any]: ...

    async def get_first_chunk(self, file_id: UUID) -> Any | None: ...
    async def delete_file(self, file_id: UUID, owner_id: str) -> bool: ...
    async def delete_expired_files(self) -> int: ...


# --- Export Protocol -----------------------------------------------------


@runtime_checkable
class IExportService(Protocol):
    """Read-only chat export service.

    Composes reads from ``IChatService``, ``IMessageService``,
    ``IResearchSessionService``, and ``ISourceService`` — no direct SQL.

    All methods raise ``ValueError`` if the chat or message is not found
    or not owned by the requesting user.
    """

    async def export_markdown(
        self,
        chat_id: UUID,
        user_id: str,
        include_metadata: bool = True,
        include_sources: bool = True,
    ) -> str:
        """Export full chat conversation as Markdown."""
        ...

    async def export_json(
        self,
        chat_id: UUID,
        user_id: str,
    ) -> dict[str, Any]:
        """Export full chat conversation as a JSON-serialisable dict."""
        ...

    async def export_report_markdown(
        self,
        message_id: UUID,
        user_id: str,
    ) -> str:
        """Export a single agent message as a standalone Markdown report."""
        ...

    async def export_provenance_markdown(
        self,
        message_id: UUID,
        user_id: str,
    ) -> str:
        """Export the verification / provenance report for a message as Markdown."""
        ...


# --- Plugin-owned Protocols ----------------------------------------------


@runtime_checkable
class IJobManager(Protocol):
    """sapresalesbot PrepJob lifecycle (heartbeat bypasses queue).
    TODO: extract full signatures from `src/sapresalesbot/services/job_manager.py`.
    """


__all__ = [
    # Ready (full signatures)
    "ChatFullView",
    "IChatService",
    "IExportService",
    "IFeedbackService",
    "IAuditLogService",
    "IResearchEventService",
    "ITemplateService",
    "ISessionService",
    # Stubbed (TODO in Wave 5 per-service PRs)
    "ICustomAgentService",
    "IPreferencesService",
    "IDataSourceService",
    "IUserService",
    "IChatMemoryService",
    "IChatSourcePoolService",
    "IMessageService",
    "ISourceService",
    "IResearchSessionService",
    "IFileUploadService",
    "IJobManager",
]
