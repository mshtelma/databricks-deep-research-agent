"""Background research job manager.

Handles job lifecycle, concurrency limits, heartbeats, and state checkpointing.

Key features:
- Background job execution decoupled from HTTP request lifecycle
- Per-user concurrency limits (default: 2 concurrent jobs)
- Heartbeat mechanism for zombie job detection (30s threshold)
- Automatic cleanup of interrupted jobs on startup
- Support for job cancellation from any app instance

Architecture:
- Jobs are tracked in research_sessions table (not separate jobs table)
- Background tasks run via asyncio.create_task()
- Events are persisted via existing EventBuffer mechanism
- Multi-instance support via worker_id and heartbeat tracking
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.core.app_config import get_app_config
from deep_research.core.logging_utils import get_logger
from deep_research.models.research_session import ResearchSession, ResearchStatus

if TYPE_CHECKING:
    from deep_research.agent.tools.web_crawler import WebCrawler
    from deep_research.services.llm.client import LLMClient
    from deep_research.services.search.brave import BraveSearchClient
    from deep_research.storage.documents import ResearchSessionState

logger = get_logger(__name__)

_MCP_SOURCE_PREFIX = "mcp:"


def _mcp_server_name_from_source_id(source_id: str) -> str | None:
    """Return the MCP server name encoded in a discovery source ID."""
    if not isinstance(source_id, str) or not source_id.startswith(_MCP_SOURCE_PREFIX):
        return None
    name = source_id[len(_MCP_SOURCE_PREFIX):].strip()
    return name or None


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def normalize_mcp_source_selection(
    *,
    source_scope: str | None,
    enabled_sources: list[str] | None,
    disabled_sources: list[str] | None,
    enabled_mcp_servers: list[str] | None,
) -> tuple[list[str] | None, list[str] | None, list[str] | None]:
    """Normalize MCP discovery source IDs into runtime MCP attachments.

    The chat source browser now treats MCP servers as sources with IDs like
    ``mcp:tavily_mcp``. The execution framework, however, attaches MCP via
    ``enabled_mcp_servers`` and must never hand ``mcp:*`` IDs to the enterprise
    tool factory. This helper keeps both client contracts compatible.
    """
    disabled_source_ids = list(disabled_sources or [])
    disabled_mcp_names = {
        name
        for source_id in disabled_source_ids
        if (name := _mcp_server_name_from_source_id(source_id)) is not None
    }

    normalized_enabled_sources: list[str] | None
    mcp_names_from_sources: list[str] = []
    if enabled_sources is None:
        normalized_enabled_sources = None
    else:
        normalized_enabled_sources = []
        for source_id in enabled_sources:
            mcp_name = _mcp_server_name_from_source_id(source_id)
            if mcp_name is None:
                normalized_enabled_sources.append(source_id)
            else:
                mcp_names_from_sources.append(mcp_name)

    requested_mcp_names = mcp_names_from_sources + list(enabled_mcp_servers or [])
    normalized_mcp_servers = [
        name
        for name in _dedupe_preserving_order(requested_mcp_names)
        if name not in disabled_mcp_names
    ]

    if source_scope == "web_only":
        normalized_mcp_servers = []

    return (
        normalized_enabled_sources,
        disabled_sources,
        normalized_mcp_servers or None,
    )


def _get_max_concurrent_jobs() -> int:
    """Get max concurrent jobs per user from config."""
    return get_app_config().jobs.max_concurrent_per_user


def get_max_concurrent_jobs() -> int:
    """Get max concurrent jobs per user from config (public API)."""
    return get_app_config().jobs.max_concurrent_per_user


def _get_heartbeat_interval() -> int:
    """Get heartbeat interval in seconds from config."""
    return get_app_config().jobs.heartbeat_interval_seconds


def _get_zombie_threshold() -> int:
    """Get zombie threshold in seconds from config."""
    return get_app_config().jobs.zombie_threshold_seconds


class JobManager:
    """Manages background research jobs.

    Responsibilities:
    - Submit new jobs with concurrency limit enforcement
    - Track active jobs per worker instance
    - Update heartbeats for zombie detection
    - Cancel jobs (in-memory + database status)
    - Clean up interrupted jobs on startup
    """

    def __init__(self) -> None:
        """Initialize the job manager."""
        self._active_tasks: dict[UUID, asyncio.Task[None]] = {}
        self._worker_id = f"{os.getpid()}-{uuid4().hex[:8]}"
        self._cleanup_task: asyncio.Task[None] | None = None
        self._running = False
        # Optional; populated in `main.py` lifespan when cached storage is on.
        # Threaded into `stream_research(..., storage_stack=...)` per job so
        # cached-impl services can talk to the in-memory StorageStack.
        self._storage_stack: Any = None

    def set_storage_stack(self, stack: Any) -> None:
        """Attach the process-singleton `StorageStack` (cached storage mode)."""
        self._storage_stack = stack

    @property
    def worker_id(self) -> str:
        """Get this worker's unique identifier."""
        return self._worker_id

    @property
    def active_job_count(self) -> int:
        """Get the number of active jobs in this worker."""
        return len(self._active_tasks)

    async def start(self, session_maker: Any) -> None:
        """Start the job manager.

        Call this on app startup. It will:
        1. Start the heartbeat loop
        2. Clean up any interrupted jobs from previous runs

        Args:
            session_maker: Async session maker for database access.
        """
        if self._running:
            logger.warning("JOB_MANAGER_ALREADY_RUNNING")
            return

        self._running = True
        self._session_maker = session_maker
        logger.info("JOB_MANAGER_STARTING", worker_id=self._worker_id)

        # Clean up interrupted jobs from prior runs, then start the recurring
        # cleanup loop (replaces the legacy heartbeat loop; _active_tasks is the
        # liveness signal — see _cleanup_interrupted_jobs).
        await self._cleanup_interrupted_jobs()
        self._cleanup_task = asyncio.create_task(self._recurring_cleanup_loop())

        logger.info("JOB_MANAGER_STARTED", worker_id=self._worker_id)

    async def stop(self) -> None:
        """Stop the job manager.

        Call this on app shutdown. It will:
        1. Cancel the heartbeat loop
        2. Cancel all active tasks (they will mark themselves as cancelled)
        3. Wait for tasks to complete
        """
        if not self._running:
            return

        logger.info("JOB_MANAGER_STOPPING", worker_id=self._worker_id, active_jobs=len(self._active_tasks))
        self._running = False

        # Cancel the recurring cleanup loop
        if self._cleanup_task:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        # Cancel all active tasks
        for task in self._active_tasks.values():
            task.cancel()

        # Wait for tasks to finish
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)

        self._active_tasks.clear()
        logger.info("JOB_MANAGER_STOPPED", worker_id=self._worker_id)

    async def submit_job(
        self,
        user_id: str,
        chat_id: UUID,
        query: str,
        query_mode: str,
        research_depth: str,
        verify_sources: bool,
        llm: LLMClient,
        brave_client: BraveSearchClient,
        crawler: WebCrawler,
        conversation_history: list[dict[str, str]],
        system_instructions: str | None,
        output_type: str | None,
        source_scope: str | None,
        enabled_sources: list[str] | None,
        disabled_sources: list[str] | None,
        plugin_manager: Any | None,
        db: AsyncSession,
        user_token: str | None = None,
        file_ids: list[str] | None = None,
        agent_id: str | None = None,
        turn_intent: str = "auto",
        enable_plan_review: bool = False,
        approval_broker: Any | None = None,
        tone: str | None = None,
        output_language: str | None = None,
        enabled_mcp_servers: list[str] | None = None,
        enabled_skills: list[str] | None = None,
        enable_cross_session_memory: bool | None = None,
        allow_live_search: bool | None = None,
        surface_inputs: dict[str, Any] | None = None,
        surface_action: str | None = None,
    ) -> ResearchSession:
        """Submit a new research job.

        This method:
        1. Checks concurrency limit for the user
        2. Creates a ResearchSession record immediately (not deferred)
        3. Starts a background task to run the research
        4. Returns the session so the caller can track it

        Args:
            user_id: User submitting the job.
            chat_id: Chat to associate the research with.
            query: Research query.
            query_mode: Query mode (simple, web_search, deep_research).
            research_depth: Research depth (auto, light, medium, extended).
            verify_sources: Whether to enable citation verification.
            llm: LLM client for agent calls.
            brave_client: Brave search client.
            crawler: Web crawler for fetching pages.
            conversation_history: Previous conversation messages.
            system_instructions: User's custom system instructions.
            output_type: Output type name from registry (e.g., 'meeting_prep').
            source_scope: Source scope (enterprise_only, web_only, all).
            enabled_sources: Whitelist of source IDs to use.
            disabled_sources: Blacklist of source IDs to exclude.
            plugin_manager: Plugin manager for custom phase mode (optional).
            db: Database session.
            user_token: User OAuth token for OBO authentication (007-enterprise Phase 2).
            file_ids: Uploaded file IDs to include in research context.
            agent_id: Custom agent ID to use for this research job.
            enable_plan_review: If True, pause after plan creation for user review.

        Returns:
            The created ResearchSession.

        Raises:
            HTTPException(429): If user has reached concurrency limit.
        """
        from fastapi import HTTPException

        # Fail fast: in cached storage mode every persistence call-site must
        # route through the same in-process StorageStack the framework reads
        # from. If the stack isn't attached yet, the legacy SQL fallback in
        # persist_research_session_start_independent would silently write the
        # Chat row to Lakebase, bypassing the cache — the framework would then
        # fail to hydrate the chat at run start *and* fail to persist at run
        # end, leaving ResearchSession.status stuck at IN_PROGRESS. Better to
        # 500 one request than silently corrupt every run.
        from deep_research.core.config import get_settings

        if (
            get_settings().storage_service_impl == "cached"
            and self._storage_stack is None
        ):
            raise RuntimeError(
                "JobManager.submit_job called in cached storage mode without "
                "a StorageStack attached; wire set_storage_stack() in app "
                "lifespan before accepting jobs."
            )

        enabled_sources, disabled_sources, enabled_mcp_servers = normalize_mcp_source_selection(
            source_scope=source_scope,
            enabled_sources=enabled_sources,
            disabled_sources=disabled_sources,
            enabled_mcp_servers=enabled_mcp_servers,
        )

        # Check concurrency limit
        max_jobs = _get_max_concurrent_jobs()
        count = await self._count_user_active_jobs(user_id, db)
        if count >= max_jobs:
            logger.warning(
                "JOB_LIMIT_EXCEEDED",
                user_id=user_id,
                current_jobs=count,
                limit=max_jobs,
            )
            raise HTTPException(
                status_code=429,
                detail=f"Maximum {max_jobs} concurrent research jobs allowed",
            )

        # Generate IDs
        session_id = uuid4()
        message_id = uuid4()  # Agent message ID
        user_message_id = uuid4()  # User message ID

        # Use the persistence function to create chat, messages, and session
        # in the correct order (satisfying FK constraints).
        #
        # IMPORTANT: pass storage_stack so cached mode takes the
        # _persist_session_start_cached branch, which creates the
        # ChatDocument in the same in-process cache that stream_research
        # later reads from. Without this, the chat is written to Lakebase
        # directly and the framework's cache-backed reads see it as missing
        # (FWK_EXISTING_SOURCES_CACHE_LOAD_FAILED + FWK_PERSISTENCE_FAILED).
        from deep_research.agent.persistence import persist_research_session_start_independent

        await persist_research_session_start_independent(
            chat_id=chat_id,
            user_id=user_id,
            user_query=query,
            user_message_id=user_message_id,
            agent_message_id=message_id,
            research_session_id=session_id,
            research_depth=research_depth,
            query_mode=query_mode,
            # Job management columns
            worker_id=self._worker_id,
            last_heartbeat=datetime.now(UTC),
            verify_sources=verify_sources,
            storage_stack=self._storage_stack,
        )

        # Build the response model from values we just persisted. We can't
        # round-trip through SQLAlchemy here: in cached storage mode
        # persist_research_session_start_independent routes through
        # _persist_session_start_cached, which stores the session inside the
        # StorageStack ChatDocument (a JSONB blob) rather than as a row in
        # research_sessions. db.get(ResearchSession, session_id) would then
        # return None even though the write succeeded, blowing up every
        # submit in cached mode. Every field _session_to_response reads is
        # either a param we just handed to the persistence call or a value
        # we just generated, so construct the instance directly.
        started_at = datetime.now(UTC)
        session = ResearchSession(
            id=session_id,
            message_id=message_id,
            chat_id=chat_id,
            user_id=user_id,
            query=query,
            research_depth=research_depth,
            query_mode=query_mode,
            status=ResearchStatus.IN_PROGRESS,
            started_at=started_at,
            worker_id=self._worker_id,
            last_heartbeat=started_at,
            verify_sources=verify_sources,
        )

        logger.info(
            "JOB_SUBMITTED",
            session_id=str(session_id),
            user_id=user_id,
            chat_id=str(chat_id),
            query_mode=query_mode,
            research_depth=research_depth,
            output_type=output_type,
            source_scope=source_scope,
            enabled_sources=enabled_sources,
            enabled_sources_count=(
                len(enabled_sources) if enabled_sources is not None else None
            ),
            disabled_sources_count=len(disabled_sources) if disabled_sources else 0,
            enabled_mcp_servers=enabled_mcp_servers,
            enabled_skills_count=(
                len(enabled_skills) if enabled_skills is not None else None
            ),
            file_count=len(file_ids) if file_ids else 0,
        )

        # Emit lifecycle hook: job_submitted
        if plugin_manager:
            try:
                from deep_research.plugins.lifecycle import EventEmitter

                emitter = EventEmitter(plugin_manager)
                await emitter.job_submitted(
                    job_id=session_id,
                    chat_id=str(chat_id),
                    query=query,
                    user_id=user_id,
                    query_mode=query_mode,
                    research_depth=research_depth,
                    verify_sources=verify_sources,
                    output_type=output_type,
                )
            except Exception as e:
                logger.warning(
                    "LIFECYCLE_HOOK_EMISSION_FAILED",
                    hook="on_job_submitted",
                    error=str(e)[:200],
                )

        # Start background task
        task = asyncio.create_task(
            self._run_job(
                session_id=session_id,
                message_id=message_id,
                user_id=user_id,
                chat_id=chat_id,
                query=query,
                query_mode=query_mode,
                research_depth=research_depth,
                verify_sources=verify_sources,
                llm=llm,
                brave_client=brave_client,
                crawler=crawler,
                conversation_history=conversation_history,
                system_instructions=system_instructions,
                output_type=output_type,
                source_scope=source_scope,
                enabled_sources=enabled_sources,
                disabled_sources=disabled_sources,
                plugin_manager=plugin_manager,
                user_token=user_token,
                file_ids=file_ids,
                agent_id=agent_id,
                turn_intent=turn_intent,
                enable_plan_review=enable_plan_review,
                approval_broker=approval_broker,
                tone=tone,
                output_language=output_language,
                enabled_mcp_servers=enabled_mcp_servers,
                enabled_skills=enabled_skills,
                enable_cross_session_memory=enable_cross_session_memory,
                allow_live_search=allow_live_search,
                surface_inputs=surface_inputs,
                surface_action=surface_action,
            )
        )
        self._active_tasks[session_id] = task

        return session

    async def cancel_job(
        self,
        chat_id: UUID,
        session_id: UUID,
        user_id: str,
        db: AsyncSession,
    ) -> bool:
        """Cancel a running job.

        This method:
        1. Verifies the user owns the job (via chat_id + session_id from the URL)
        2. Cancels the in-memory task (if this worker owns it)
        3. Updates the database status (works across instances)

        Args:
            chat_id: Chat owning the session (from URL path param).
            session_id: Job/session ID to cancel.
            user_id: User requesting cancellation (for ownership check).
            db: Database session.

        Returns:
            True if job was cancelled, False if not found or not owned.
        """
        from deep_research.agent.persistence import (
            persist_research_session_cancelled_independent,
        )
        from deep_research.agent.session_lookup import load_session_control_view
        from deep_research.core.config import get_settings

        settings = get_settings()

        # Verify ownership via the unified session lookup — chat_id from the
        # URL is matched against the chat's owner in the ChatDocument (cached
        # mode) or the research_sessions row (legacy mode).
        view = await load_session_control_view(
            chat_id,
            session_id,
            user_id,
            settings=settings,
            storage_stack=self._storage_stack,
            db=db,
        )
        if view is None:
            logger.warning(
                "JOB_CANCEL_DENIED",
                session_id=str(session_id),
                chat_id=str(chat_id),
                user_id=user_id,
            )
            return False

        # Cancel in-memory task if this worker owns it
        if session_id in self._active_tasks:
            logger.info(
                "JOB_CANCELLING_LOCAL_TASK",
                session_id=str(session_id),
            )
            self._active_tasks[session_id].cancel()

        # Update status (routes through cached path when enabled, else SQL).
        await persist_research_session_cancelled_independent(
            research_session_id=session_id,
            storage_stack=self._storage_stack,
            chat_id=chat_id,
        )

        logger.info(
            "JOB_CANCELLED",
            session_id=str(session_id),
            chat_id=str(chat_id),
            user_id=user_id,
        )
        return True

    async def get_user_jobs(
        self,
        user_id: str,
        status: str | None,
        db: AsyncSession | None = None,  # noqa: ARG002  # kept for signature back-compat
        limit: int = 50,
    ) -> list[ResearchSession]:
        """Return research sessions for a user, optionally filtered by status.

        Reads from the storage stack (``chat_state.state.research_sessions``)
        — never the legacy ``research_sessions`` table. Returns in-memory
        ``ResearchSession`` DTOs (no SQL identity) for response back-compat.
        """
        if self._storage_stack is None:
            raise RuntimeError(
                "JobManager.get_user_jobs requires a StorageStack; "
                "call set_storage_stack() in app lifespan."
            )
        pairs = await self._storage_stack.backend.list_user_jobs(
            user_id, status=status, limit=limit
        )
        return [_session_state_to_orm(chat_id, rs, user_id) for chat_id, rs in pairs]

    async def get_chat_active_job(
        self,
        chat_id: UUID,
        user_id: str,
        db: AsyncSession | None = None,  # noqa: ARG002  # kept for signature back-compat
    ) -> ResearchSession | None:
        """Return the active research session for a chat, or None.

        Reads from the storage stack; ``user_id`` scopes the lookup for
        ownership.
        """
        if self._storage_stack is None:
            raise RuntimeError(
                "JobManager.get_chat_active_job requires a StorageStack; "
                "call set_storage_stack() in app lifespan."
            )
        rs = await self._storage_stack.backend.get_active_session_for_chat(
            chat_id, user_id
        )
        if rs is None:
            return None
        return _session_state_to_orm(chat_id, rs, user_id)

    async def _run_job(
        self,
        session_id: UUID,
        message_id: UUID,
        user_id: str,
        chat_id: UUID,
        query: str,
        query_mode: str,
        research_depth: str,
        verify_sources: bool,
        llm: LLMClient,
        brave_client: BraveSearchClient,
        crawler: WebCrawler,
        conversation_history: list[dict[str, str]],
        system_instructions: str | None,
        output_type: str | None = None,
        source_scope: str | None = None,
        enabled_sources: list[str] | None = None,
        disabled_sources: list[str] | None = None,
        plugin_manager: Any | None = None,
        user_token: str | None = None,
        file_ids: list[str] | None = None,
        agent_id: str | None = None,
        turn_intent: str = "auto",
        enable_plan_review: bool = False,
        approval_broker: Any | None = None,
        tone: str | None = None,
        output_language: str | None = None,
        enabled_mcp_servers: list[str] | None = None,
        enabled_skills: list[str] | None = None,
        enable_cross_session_memory: bool | None = None,
        allow_live_search: bool | None = None,
        surface_inputs: dict[str, Any] | None = None,
        surface_action: str | None = None,
    ) -> None:
        """Execute research job in background.

        This method runs the full research pipeline in the background,
        persisting events as they occur. The job status is updated
        automatically on completion, cancellation, or error.

        Args:
            session_id: Research session ID.
            message_id: Agent message ID.
            user_id: User ID.
            chat_id: Chat ID.
            query: Research query.
            query_mode: Query mode.
            research_depth: Research depth.
            verify_sources: Whether to verify sources.
            llm: LLM client.
            brave_client: Brave search client.
            crawler: Web crawler.
            conversation_history: Conversation history.
            system_instructions: Custom system instructions.
            output_type: Output type name from registry (e.g., 'meeting_prep').
            source_scope: Source scope (enterprise_only, web_only, all).
            enabled_sources: Whitelist of source IDs to use.
            disabled_sources: Blacklist of source IDs to exclude.
            plugin_manager: Plugin manager for custom phase mode (optional).
            user_token: User OAuth token for OBO authentication (007-enterprise Phase 2).
            file_ids: Uploaded file IDs to include in research context.
            agent_id: Custom agent ID to use for this research job.
            enable_plan_review: If True, pause after plan creation for user review.
        """
        from deep_research.agent.orchestrator import OrchestrationConfig, stream_research
        from deep_research.db.session import get_session_maker
        from deep_research.output.registry import get_output_registry

        logger.info(
            "JOB_STARTING",
            session_id=str(session_id),
            query=query[:100],
            output_type=output_type,
            file_count=len(file_ids) if file_ids else 0,
        )

        try:
            (
                enabled_sources,
                disabled_sources,
                enabled_mcp_servers,
            ) = normalize_mcp_source_selection(
                source_scope=source_scope,
                enabled_sources=enabled_sources,
                disabled_sources=disabled_sources,
                enabled_mcp_servers=enabled_mcp_servers,
            )

            # Get output configuration from registry if output_type is specified
            output_schema = None
            output_format = "markdown"
            structured_system_prompt = None
            structured_user_prompt = None

            # Auto-resolve output_type from plugin when not specified by the
            # frontend.  When exactly one plugin implements OutputTypeProvider,
            # every query should use its output type (plugin-driven app).
            if not output_type and plugin_manager:
                try:
                    from deep_research.output.protocol import OutputTypeProvider

                    providers = [
                        p
                        for p in plugin_manager.get_plugins()
                        if isinstance(p, OutputTypeProvider)
                    ]
                    if len(providers) == 1:
                        output_type = providers[0].output_type_name
                        logger.info(
                            "OUTPUT_TYPE_AUTO_RESOLVED",
                            extra={
                                "output_type": output_type,
                                "plugin": getattr(
                                    providers[0], "name", type(providers[0]).__name__
                                ),
                            },
                        )
                    elif len(providers) > 1:
                        logger.warning(
                            "OUTPUT_TYPE_AUTO_RESOLVE_AMBIGUOUS",
                            extra={
                                "provider_count": len(providers),
                                "plugins": [
                                    getattr(p, "name", type(p).__name__)
                                    for p in providers
                                ],
                                "note": "Set output_type explicitly to resolve ambiguity.",
                            },
                        )
                except Exception:
                    logger.exception("OUTPUT_TYPE_AUTO_RESOLVE_FAILED")

            if output_type:
                registry = get_output_registry()
                provider = registry.get(output_type)
                if provider:
                    synth_config = provider.get_synthesizer_config()
                    # Use get_output_model() to get the Pydantic class (not JSON schema dict)
                    # The synthesizer expects a class type with __name__ attribute
                    output_schema = (
                        provider.get_output_model()
                        if hasattr(provider, "get_output_model")
                        else None
                    )
                    structured_system_prompt = synth_config.custom_prompt
                    structured_user_prompt = provider.get_synthesizer_prompt()
                    output_format = "json" if output_schema else "markdown"
                    logger.info(
                        "JOB_OUTPUT_TYPE_RESOLVED",
                        session_id=str(session_id),
                        output_type=output_type,
                        has_schema=output_schema is not None,
                        schema_name=getattr(output_schema, "__name__", None) if output_schema else None,
                        has_system_prompt=structured_system_prompt is not None,
                        has_user_prompt=structured_user_prompt is not None,
                    )
                else:
                    logger.warning(
                        "JOB_OUTPUT_TYPE_NOT_FOUND",
                        session_id=str(session_id),
                        output_type=output_type,
                    )

            # Resolve per-run tone/output-language: an explicit per-request value
            # wins; otherwise fall back to the research-depth config defaults
            # (default_tone / default_output_language). Both absent => None =>
            # unchanged synthesis (byte-identical default path).
            effective_tone = tone
            effective_output_language = output_language
            if effective_tone is None or effective_output_language is None:
                try:
                    from deep_research.agent.config import get_research_type_config

                    _valid_depths = {"light", "medium", "extended"}
                    _depth = research_depth if research_depth in _valid_depths else "medium"
                    _depth_config = get_research_type_config(_depth)
                    if effective_tone is None:
                        effective_tone = _depth_config.default_tone
                    if effective_output_language is None:
                        effective_output_language = _depth_config.default_output_language
                except Exception:
                    # Defaulting is best-effort; never block a job on config lookup.
                    logger.debug("RESEARCH_TYPE_TONE_DEFAULT_LOOKUP_FAILED", exc_info=True)

            config = OrchestrationConfig(
                query_mode=query_mode,
                research_depth=research_depth,
                system_instructions=system_instructions,
                tone=effective_tone,
                output_language=effective_output_language,
                message_id=message_id,
                research_session_id=session_id,
                is_draft=False,  # Chat already created
                verify_sources=verify_sources,
                session_pre_created=True,  # Session already created by JobManager
                output_format=output_format,
                output_schema=output_schema,
                structured_system_prompt=structured_system_prompt,
                structured_user_prompt=structured_user_prompt,
                source_scope=source_scope,
                enabled_sources=enabled_sources,
                disabled_sources=disabled_sources,
                enabled_mcp_servers=enabled_mcp_servers,
                enabled_skills=enabled_skills,
                enable_cross_session_memory=enable_cross_session_memory,
                allow_live_search=allow_live_search,
                surface_inputs=surface_inputs,
                surface_action=surface_action,
                user_token=user_token,  # OBO auth for enterprise tools
                file_ids=file_ids,
                agent_id=agent_id,
                turn_intent=turn_intent,
                enable_plan_review=enable_plan_review,
                approval_broker=approval_broker,
            )

            # Auto-resolve workflow_ref from output_type when a plugin provides
            # a matching workflow.  Bridges OutputTypeProvider (synthesizer) and
            # WorkflowProviderPlugin (research pipeline).  Without this, workflow_ref
            # is only set via custom agents in the DB, so plugin YAML workflows are
            # never triggered.
            if output_type and not config.workflow_ref and plugin_manager:
                from deep_research.plugins.base import WorkflowProviderPlugin

                for plugin in plugin_manager.get_plugins():
                    if isinstance(plugin, WorkflowProviderPlugin):
                        try:
                            if plugin.get_workflow_yaml(output_type) is not None:
                                config.workflow_ref = output_type
                                logger.info(
                                    "WORKFLOW_REF_AUTO_SET",
                                    extra={
                                        "output_type": output_type,
                                        "plugin": getattr(plugin, "name", type(plugin).__name__),
                                    },
                                )
                                break
                        except Exception:
                            pass

            async def _consume_research_stream(
                research_db: AsyncSession | None,
            ) -> None:
                """Consume the research stream to completion.

                ``research_db`` is deliberately nullable: F-JOB-B moved every
                reader/persister inside the stream onto either the cached
                StorageStack path or self-opening independent sessions via
                ``get_session_maker()``. Passing ``None`` avoids holding a
                request-scoped session across the long stream (see comment
                below the call site).
                """
                async for _event in stream_research(
                    query=query,
                    llm=llm,
                    brave_client=brave_client,
                    crawler=crawler,
                    conversation_history=conversation_history,
                    user_id=user_id,
                    chat_id=str(chat_id),
                    config=config,
                    db=research_db,
                    plugin_manager=plugin_manager,
                    storage_stack=self._storage_stack,
                ):
                    # Events are persisted by the orchestrator
                    # We just iterate to completion
                    pass

            # F-JOB-B: All readers and persisters inside the stream now run on
            # the cached StorageStack path (no session_maker needed) or open
            # their own independent sessions via get_session_maker() internally.
            # Passing db=None is safe:
            #   - _load_existing_sources: guarded by `if db is None: return []`
            #   - FileUploadService / DataSourceService: guarded by
            #     `if db is None and storage_stack is None`
            #   - _load_enterprise_tools: guarded by `if db is not None and user_id`
            #   - All persistence helpers: route through factories on cached path.
            # The F-JOB-A InterfaceError swallow is no longer needed because no
            # session is held for the duration of the stream.
            timed_out = False
            timeout = config.research_timeout_seconds
            try:
                await asyncio.wait_for(
                    _consume_research_stream(None),
                    timeout=timeout,
                )
            except TimeoutError:
                timed_out = True

            # -- Post-stream handling with fresh sessions --

            if timed_out:
                logger.error(
                    "RESEARCH_TIMEOUT",
                    session_id=str(session_id),
                    timeout_seconds=timeout,
                )
                try:
                    from deep_research.agent.persistence import (
                        persist_research_session_failed_independent,
                    )

                    await persist_research_session_failed_independent(
                        research_session_id=session_id,
                        agent_message_id=message_id,
                        error_message=f"Research timed out after {timeout} seconds",
                        storage_stack=self._storage_stack,
                        chat_id=chat_id,
                    )
                except Exception as timeout_persist_err:
                    logger.warning(
                        "TIMEOUT_PERSIST_FAILED",
                        session_id=str(session_id),
                        error=str(timeout_persist_err)[:200],
                    )
                return

            # Orchestrator already committed COMPLETED via independent session
            # (persist_research_session_complete_update_independent).
            # Read with a fresh session for logging + lifecycle hook only.
            try:
                from deep_research.agent.session_lookup import (
                    load_session_control_view,
                )
                from deep_research.core.config import get_settings

                settings = get_settings()
                completion_sm = get_session_maker()
                async with completion_sm() as completion_db:
                    view = await load_session_control_view(
                        chat_id,
                        session_id,
                        user_id,
                        settings=settings,
                        storage_stack=self._storage_stack,
                        db=completion_db,
                    )
                    if view is not None and view.status == ResearchStatus.COMPLETED:
                        duration_seconds = (
                            (datetime.now(UTC) - view.started_at).total_seconds()
                            if view.started_at
                            else 0.0
                        )
                        logger.info(
                            "JOB_COMPLETED",
                            session_id=str(session_id),
                        )

                        # Emit lifecycle hook: job_completed
                        if plugin_manager:
                            try:
                                from deep_research.plugins.lifecycle import EventEmitter

                                emitter = EventEmitter(plugin_manager)
                                await emitter.job_completed(
                                    job_id=session_id,
                                    duration_seconds=duration_seconds,
                                    output=None,
                                    output_type=output_type or "generic",
                                    event_count=0,
                                )
                            except Exception as e:
                                logger.warning(
                                    "LIFECYCLE_HOOK_EMISSION_FAILED",
                                    hook="on_job_completed",
                                    error=str(e)[:200],
                                )
                    elif view is not None:
                        # The research stream drained but the orchestrator
                        # never transitioned the session to COMPLETED/FAILED
                        # (e.g. end-of-run persistence failed). Force-FAIL
                        # through the cached-aware helper so the ChatDocument
                        # and Lakebase agree, the UI stops polling, and the
                        # heartbeat loop stops updating this row. This repairs
                        # the historical "stuck at in_progress forever" bug
                        # observed in prod when storage_stack was not threaded
                        # into submit_job.
                        logger.warning(
                            "JOB_COMPLETION_STATUS_UNEXPECTED",
                            session_id=str(session_id),
                            actual_status=str(view.status),
                        )
                        try:
                            from deep_research.agent.persistence import (
                                persist_research_session_failed_independent,
                            )

                            await persist_research_session_failed_independent(
                                research_session_id=session_id,
                                agent_message_id=message_id,
                                error_message="persistence_transition_missing",
                                storage_stack=self._storage_stack,
                                chat_id=chat_id,
                            )
                            logger.error(
                                "JOB_COMPLETION_FORCED_FAIL",
                                session_id=str(session_id),
                                previous_status=str(view.status),
                            )
                        except Exception as force_fail_err:
                            # Double fault: the repair path itself failed.
                            # Log + continue — _run_job's finally block still
                            # pops _active_tasks so the heartbeat loop stops
                            # touching this session on the next tick.
                            logger.error(
                                "JOB_COMPLETION_FORCED_FAIL_FAILED",
                                session_id=str(session_id),
                                error=str(force_fail_err)[:200],
                            )
            except Exception as completion_err:
                logger.warning(
                    "JOB_COMPLETION_CHECK_FAILED",
                    session_id=str(session_id),
                    error=str(completion_err)[:200],
                )

        except asyncio.CancelledError:
            logger.info(
                "JOB_CANCELLED_BY_TASK",
                session_id=str(session_id),
            )
            from deep_research.agent.persistence import (
                persist_research_session_cancelled_independent,
            )

            # Routes through cached path when enabled; the underlying SQL
            # UPDATE is gated on status == IN_PROGRESS so it no-ops when the
            # session has already transitioned.
            try:
                await persist_research_session_cancelled_independent(
                    research_session_id=session_id,
                    storage_stack=self._storage_stack,
                    chat_id=chat_id,
                )
            except Exception as cancel_persist_err:
                logger.warning(
                    "CANCEL_PERSIST_FAILED",
                    session_id=str(session_id),
                    error=str(cancel_persist_err)[:200],
                )
            raise

        except Exception as e:
            logger.exception(
                "JOB_FAILED",
                session_id=str(session_id),
                error=str(e),
            )
            from deep_research.agent.persistence import (
                persist_research_session_failed_independent,
            )
            from deep_research.agent.session_lookup import (
                load_session_control_view,
            )
            from deep_research.core.config import get_settings

            settings = get_settings()
            error_session_maker = get_session_maker()
            async with error_session_maker() as db:
                view = await load_session_control_view(
                    chat_id,
                    session_id,
                    user_id,
                    settings=settings,
                    storage_stack=self._storage_stack,
                    db=db,
                )
                if view is not None and view.status == ResearchStatus.IN_PROGRESS:
                    try:
                        await persist_research_session_failed_independent(
                            research_session_id=session_id,
                            agent_message_id=message_id,
                            error_message=str(e)[:500],
                            storage_stack=self._storage_stack,
                            chat_id=chat_id,
                        )
                    except Exception as fail_persist_err:
                        logger.warning(
                            "ERROR_PERSIST_FAILED",
                            session_id=str(session_id),
                            error=str(fail_persist_err)[:200],
                        )

                    # Emit lifecycle hook: job_failed
                    if plugin_manager:
                        try:
                            from deep_research.plugins.lifecycle import EventEmitter

                            # Classify error
                            error_class = type(e).__name__
                            error_category = "unknown"
                            is_recoverable = False

                            if "Database" in error_class or "Operational" in error_class:
                                error_category = "database"
                                is_recoverable = True
                            elif "Timeout" in error_class or "Connection" in error_class:
                                error_category = "network"
                                is_recoverable = True
                            elif "Validation" in error_class or "JSONDecode" in error_class:
                                error_category = "validation"
                                is_recoverable = False
                            elif "rate" in str(e).lower() or "quota" in str(e).lower():
                                error_category = "rate_limit"
                                is_recoverable = True

                            emitter = EventEmitter(plugin_manager)
                            await emitter.job_failed(
                                job_id=session_id,
                                error=e,
                                error_category=error_category,
                                is_recoverable=is_recoverable,
                            )
                        except Exception as hook_error:
                            logger.warning(
                                "LIFECYCLE_HOOK_EMISSION_FAILED",
                                hook="on_job_failed",
                                error=str(hook_error)[:200],
                            )
                elif view is not None:
                    logger.info(
                        "JOB_ERROR_SKIPPED_TERMINAL_STATUS",
                        session_id=str(session_id),
                        current_status=str(view.status),
                        error=str(e)[:200],
                    )

        finally:
            self._active_tasks.pop(session_id, None)

    async def _recurring_cleanup_loop(self) -> None:
        """Run ``_cleanup_interrupted_jobs`` every 5 minutes while running.

        Replaces the legacy ``_heartbeat_loop`` (which wrote ``last_heartbeat``
        to the now-dropped ``research_sessions`` table). The ``_active_tasks``
        dict on this worker is the canonical liveness signal; the periodic
        cleanup is a safety net for sessions that fail to self-clean.
        """
        interval = 300  # 5 minutes
        while self._running:
            try:
                await asyncio.sleep(interval)
                if not self._running:
                    break
                await self._cleanup_interrupted_jobs()
            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "RECURRING_CLEANUP_ERROR",
                    worker_id=self._worker_id,
                    error=str(exc)[:200],
                )

    async def _cleanup_interrupted_jobs(self) -> None:
        """Mark stale ``in_progress`` research sessions ``failed`` via the backend.

        A session is stale when its ``last_heartbeat`` is older than
        ``_get_zombie_threshold()`` (or NULL) AND its id is NOT in
        ``self._active_tasks`` (not running on this worker). Runs once at
        startup and every 5 minutes. Operates on the storage stack's
        ``chat_state`` JSONB — never the legacy ``research_sessions`` table.
        """
        if self._storage_stack is None:
            logger.warning(
                "CLEANUP_SKIPPED_NO_STORAGE_STACK", worker_id=self._worker_id
            )
            return
        cutoff = datetime.now(UTC) - timedelta(seconds=_get_zombie_threshold())
        exclude_ids = list(self._active_tasks.keys())
        try:
            marked = await self._storage_stack.backend.mark_stale_research_sessions_failed(
                cutoff=cutoff,
                exclude_session_ids=exclude_ids,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "CLEANUP_ERROR", worker_id=self._worker_id, error=str(exc)[:200]
            )
            return
        if marked > 0:
            logger.info(
                "CLEANUP_MARKED_STALE_SESSIONS",
                worker_id=self._worker_id,
                count=marked,
                cutoff=cutoff.isoformat(),
            )
        else:
            logger.info("CLEANUP_NO_STALE_SESSIONS", worker_id=self._worker_id)

    async def _count_user_active_jobs(
        self,
        user_id: str,
        db: AsyncSession | None = None,  # noqa: ARG002  # kept for signature back-compat
    ) -> int:
        """Count active research sessions for a user via the storage backend."""
        if self._storage_stack is None:
            raise RuntimeError(
                "JobManager._count_user_active_jobs requires a StorageStack; "
                "call set_storage_stack() in app lifespan."
            )
        return int(await self._storage_stack.backend.count_active_research_sessions(user_id))


def _parse_research_status(value: str) -> ResearchStatus:
    """Parse a status string to ``ResearchStatus``; unknown → FAILED (defensive)."""
    try:
        return ResearchStatus(value)
    except ValueError:
        logger.warning("UNKNOWN_RESEARCH_STATUS", status=value)
        return ResearchStatus.FAILED


def _session_state_to_orm(
    chat_id: UUID,
    rs: ResearchSessionState,
    user_id: str,
) -> ResearchSession:
    """Build an in-memory ``ResearchSession`` from a JSONB ``ResearchSessionState``.

    ``ResearchSession`` is used as a data carrier only (no SQL identity); callers
    must not pass it to a DB session. Fields absent from the JSONB state
    (``query``/``query_mode``/``research_depth``) default safely — job
    list/count responses do not depend on them.
    """
    return ResearchSession(
        id=rs.id,
        message_id=rs.message_id,
        chat_id=chat_id,
        user_id=user_id,
        query="",
        query_mode="deep_research",
        research_depth="auto",
        status=_parse_research_status(rs.status),
        last_heartbeat=rs.last_heartbeat,
        started_at=rs.started_at,
        completed_at=rs.completed_at,
    )


# Global instance (initialized in main.py)
_job_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance.

    Raises:
        RuntimeError: If JobManager hasn't been initialized.
    """
    if _job_manager is None:
        raise RuntimeError(
            "JobManager not initialized. Ensure main.py calls initialize_job_manager() on startup."
        )
    return _job_manager


def initialize_job_manager() -> JobManager:
    """Initialize the global job manager instance.

    Call this in main.py during app startup.

    Returns:
        The initialized JobManager.
    """
    global _job_manager
    if _job_manager is not None:
        logger.warning("JOB_MANAGER_ALREADY_INITIALIZED")
        return _job_manager

    _job_manager = JobManager()
    return _job_manager
