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

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.core.app_config import get_app_config
from deep_research.core.logging_utils import get_logger
from deep_research.models.research_session import ResearchSession, ResearchStatus

if TYPE_CHECKING:
    from deep_research.agent.tools.web_crawler import WebCrawler
    from deep_research.services.llm.client import LLMClient
    from deep_research.services.search.brave import BraveSearchClient

logger = get_logger(__name__)


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
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._running = False

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

        # Start heartbeat loop
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Clean up interrupted jobs
        await self._cleanup_interrupted_jobs()

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

        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

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
        db: AsyncSession,
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
            db: Database session.

        Returns:
            The created ResearchSession.

        Raises:
            HTTPException(429): If user has reached concurrency limit.
        """
        from fastapi import HTTPException

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
        # in the correct order (satisfying FK constraints)
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
        )

        # Refresh the passed-in db session and fetch the created session
        # (persist_research_session_start_independent uses independent session)
        await db.commit()  # Ensure any prior changes are committed
        session = await db.get(ResearchSession, session_id)
        if not session:
            raise RuntimeError(f"Failed to fetch created session {session_id}")

        logger.info(
            "JOB_SUBMITTED",
            session_id=str(session_id),
            user_id=user_id,
            chat_id=str(chat_id),
            query_mode=query_mode,
            research_depth=research_depth,
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
            )
        )
        self._active_tasks[session_id] = task

        return session

    async def cancel_job(
        self,
        session_id: UUID,
        user_id: str,
        db: AsyncSession,
    ) -> bool:
        """Cancel a running job.

        This method:
        1. Verifies the user owns the job
        2. Cancels the in-memory task (if this worker owns it)
        3. Updates the database status (works across instances)

        Args:
            session_id: Job/session ID to cancel.
            user_id: User requesting cancellation (for ownership check).
            db: Database session.

        Returns:
            True if job was cancelled, False if not found or not owned.
        """
        # Verify ownership
        session = await db.get(ResearchSession, session_id)
        if not session or session.user_id != user_id:
            logger.warning(
                "JOB_CANCEL_DENIED",
                session_id=str(session_id),
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

        # Update status in DB (works even if different worker owns it)
        session.status = ResearchStatus.CANCELLED
        session.completed_at = datetime.now(UTC)
        await db.commit()

        logger.info(
            "JOB_CANCELLED",
            session_id=str(session_id),
            user_id=user_id,
        )
        return True

    async def get_user_jobs(
        self,
        user_id: str,
        status: str | None,
        db: AsyncSession,
        limit: int = 50,
    ) -> list[ResearchSession]:
        """Get jobs for a user, optionally filtered by status.

        Args:
            user_id: User to get jobs for.
            status: Optional status filter (in_progress, completed, failed, cancelled).
            db: Database session.
            limit: Maximum number of jobs to return.

        Returns:
            List of ResearchSession objects.
        """
        stmt = select(ResearchSession).where(ResearchSession.user_id == user_id)

        if status:
            stmt = stmt.where(ResearchSession.status == status)

        stmt = stmt.order_by(ResearchSession.created_at.desc()).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_chat_active_job(
        self,
        chat_id: UUID,
        user_id: str,
        db: AsyncSession,
    ) -> ResearchSession | None:
        """Get the active job for a specific chat.

        Args:
            chat_id: Chat to check.
            user_id: User (for security).
            db: Database session.

        Returns:
            Active ResearchSession if one exists, None otherwise.
        """
        stmt = (
            select(ResearchSession)
            .where(ResearchSession.chat_id == chat_id)
            .where(ResearchSession.user_id == user_id)
            .where(ResearchSession.status == ResearchStatus.IN_PROGRESS)
            .order_by(ResearchSession.created_at.desc())
            .limit(1)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

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
    ) -> None:
        """Execute research job in background.

        This method runs the full research pipeline in the background,
        persisting events as they occur. The job status is updated
        automatically on completion, cancellation, or error.
        """
        from deep_research.agent.orchestrator import OrchestrationConfig, stream_research
        from deep_research.db.session import get_session_maker

        logger.info(
            "JOB_STARTING",
            session_id=str(session_id),
            query=query[:100],
        )

        try:
            config = OrchestrationConfig(
                query_mode=query_mode,
                research_depth=research_depth,
                system_instructions=system_instructions,
                message_id=message_id,
                research_session_id=session_id,
                is_draft=False,  # Chat already created
                verify_sources=verify_sources,
                session_pre_created=True,  # Session already created by JobManager
            )

            # Use fresh session maker to trigger token refresh check
            session_maker = get_session_maker()
            async with session_maker() as db:
                async for _event in stream_research(
                    query=query,
                    llm=llm,
                    brave_client=brave_client,
                    crawler=crawler,
                    conversation_history=conversation_history,
                    user_id=user_id,
                    chat_id=str(chat_id),
                    config=config,
                    db=db,
                ):
                    # Events are persisted by the orchestrator
                    # We just iterate to completion
                    pass

                # Mark completed
                session = await db.get(ResearchSession, session_id)
                if session and session.status == ResearchStatus.IN_PROGRESS:
                    session.status = ResearchStatus.COMPLETED
                    session.completed_at = datetime.now(UTC)
                    await db.commit()
                    logger.info(
                        "JOB_COMPLETED",
                        session_id=str(session_id),
                    )

        except asyncio.CancelledError:
            logger.info(
                "JOB_CANCELLED_BY_TASK",
                session_id=str(session_id),
            )
            # Use fresh session maker to trigger token refresh check
            cancel_session_maker = get_session_maker()
            async with cancel_session_maker() as db:
                session = await db.get(ResearchSession, session_id)
                if session and session.status == ResearchStatus.IN_PROGRESS:
                    session.status = ResearchStatus.CANCELLED
                    session.completed_at = datetime.now(UTC)
                    await db.commit()
            raise

        except Exception as e:
            logger.exception(
                "JOB_FAILED",
                session_id=str(session_id),
                error=str(e),
            )
            # Use fresh session maker to trigger token refresh check
            error_session_maker = get_session_maker()
            async with error_session_maker() as db:
                session = await db.get(ResearchSession, session_id)
                if session:
                    session.status = ResearchStatus.FAILED
                    session.error_message = str(e)[:500]  # Truncate error message
                    session.completed_at = datetime.now(UTC)
                    await db.commit()

        finally:
            self._active_tasks.pop(session_id, None)

    async def _heartbeat_loop(self) -> None:
        """Update heartbeat for all active jobs.

        Runs every heartbeat_interval_seconds while the manager is running.
        Updates the last_heartbeat timestamp for all jobs owned by this worker.

        Note: We call get_session_maker() directly instead of using the cached
        self._session_maker to ensure OAuth token refresh is triggered. The
        token refresh logic in get_session_maker() -> get_engine() checks for
        expired tokens and recreates the engine if needed.
        """
        from deep_research.db.session import get_session_maker

        while self._running:
            try:
                await asyncio.sleep(_get_heartbeat_interval())

                if not self._active_tasks:
                    continue

                # Use fresh session maker to trigger token refresh check
                session_maker = get_session_maker()
                async with session_maker() as db:
                    await db.execute(
                        update(ResearchSession)
                        .where(ResearchSession.id.in_(list(self._active_tasks.keys())))
                        .values(last_heartbeat=datetime.now(UTC))
                    )
                    await db.commit()

                logger.debug(
                    "HEARTBEAT_UPDATED",
                    worker_id=self._worker_id,
                    job_count=len(self._active_tasks),
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "HEARTBEAT_ERROR",
                    error=str(e),
                )

    async def _cleanup_interrupted_jobs(self) -> None:
        """Clean up jobs interrupted by previous app restart.

        Finds jobs with:
        - status = in_progress
        - last_heartbeat older than zombie_threshold_seconds (or NULL)

        These jobs were running when the app restarted and need cleanup.
        """
        from deep_research.db.session import get_session_maker

        cutoff = datetime.now(UTC) - timedelta(seconds=_get_zombie_threshold())

        # Use fresh session maker to trigger token refresh check
        session_maker = get_session_maker()
        async with session_maker() as db:
            stmt = (
                select(ResearchSession)
                .where(ResearchSession.status == ResearchStatus.IN_PROGRESS)
                .where(
                    (ResearchSession.last_heartbeat < cutoff)
                    | (ResearchSession.last_heartbeat.is_(None))
                )
            )
            result = await db.execute(stmt)
            interrupted = list(result.scalars().all())

            if not interrupted:
                logger.info("CLEANUP_NO_INTERRUPTED_JOBS")
                return

            logger.info(
                "CLEANUP_FOUND_INTERRUPTED_JOBS",
                count=len(interrupted),
            )

            for session in interrupted:
                # For now, mark as failed
                # Future: could resume from execution_state if available
                session.status = ResearchStatus.FAILED
                session.error_message = "Job interrupted by app restart"
                session.completed_at = datetime.now(UTC)

                logger.info(
                    "CLEANUP_JOB_MARKED_FAILED",
                    session_id=str(session.id),
                    user_id=session.user_id,
                )

            await db.commit()

    async def _count_user_active_jobs(
        self,
        user_id: str,
        db: AsyncSession,
    ) -> int:
        """Count active jobs for a user.

        Args:
            user_id: User to count jobs for.
            db: Database session.

        Returns:
            Number of in_progress jobs for the user.
        """
        result = await db.scalar(
            select(func.count(ResearchSession.id))
            .where(ResearchSession.user_id == user_id)
            .where(ResearchSession.status == ResearchStatus.IN_PROGRESS)
        )
        return result or 0


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
