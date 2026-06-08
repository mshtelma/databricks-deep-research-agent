"""Async deploy lifecycle for ``DeploymentMode.*``.

W12 (Phase 3 of the codex-flagged deployment-pipeline fix plan). The
previous ``POST /api/v1/deployments`` flow ran the entire translator
chain synchronously inside the request, with the row pinned to PENDING
and ACTIVE/FAILED with no intermediate state. That was wrong on two
counts:

- the DEPLOYING status was defined but never written, so UI polls had
  no way to distinguish "still working" from "finished";
- a translator that took multi-minute MLflow log/register/deploy calls
  blocked the request thread and timed out HTTP clients.

This runner replaces that inline flow with a proper async job model:

1. ``submit(deployment_id, mode)`` queues the work and returns
   immediately. The API handler can 202-Accept the request.
2. ``_run`` claims the row (PENDING → DEPLOYING + heartbeat columns
   populated), spawns a heartbeat loop alongside the translator
   pipeline, and lands the row in ACTIVE or FAILED.
3. ``_janitor`` periodically sweeps for rows where the heartbeat
   went stale (worker crashed, process killed, GC pause beyond the
   timeout). Stale rows are marked FAILED with
   ``error_message="worker_zombie"`` and the translator's
   ``deactivate()`` is dispatched best-effort against any partially-
   created external resources (so we don't leak UC models, App rows,
   etc.).
4. Cancellation: DELETE on a PENDING/DEPLOYING row sets
   ``cancel_requested = true``. The heartbeat loop reads it each tick
   and aborts the inner task; the row lands in FAILED with
   ``error_message="cancelled"``.

The runner is wired as a FastAPI lifespan singleton. ``start()`` kicks
the janitor and recovery sweep (re-claims any PENDING rows that were
mid-flight when the process restarted — they get re-run, not orphaned).
``shutdown()`` signals all in-flight tasks, awaits up to a grace
window, and marks survivors FAILED with
``error_message="server_shutdown"`` so they're not stuck in DEPLOYING.

Sized around ~300 LOC per the architect's pre-commit estimate.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deep_research.models.agent_deployment import (
    ACTIVE_STATUSES,
    MAX_CLEANUP_ATTEMPTS,
    AgentDeployment,
    DeploymentMode,
    DeploymentStatus,
)
from deep_research.models.agent_v2 import AgentRevision, AgentV2
from deep_research.services.deployment import (
    DeploymentCleanupError,
    translator_for,
)
from deep_research.services.deployment_service import DeploymentService

logger = logging.getLogger(__name__)


# --- Tunables ---------------------------------------------------------------

#: How often a running worker writes a heartbeat row update.
HEARTBEAT_INTERVAL_SECONDS: int = 30
#: How long before a row's heartbeat is considered stale (zombie).
#: 4x heartbeat gives slack for transient DB hiccups + GC pauses.
HEARTBEAT_TIMEOUT_SECONDS: int = HEARTBEAT_INTERVAL_SECONDS * 4
#: How often the janitor scans for stale rows.
JANITOR_INTERVAL_SECONDS: int = 60
#: Per-user limit on concurrent in-flight deploys. Distinct from research
#: jobs (those have their own quota in ``app_config``). Default sized
#: small because each deploy chews real Databricks resources.
DEFAULT_MAX_CONCURRENT_PER_USER: int = 2
#: Grace window during shutdown — workers get this long to wrap up
#: before being marked failed.
SHUTDOWN_GRACE_SECONDS: int = 30


class DeploymentBudgetExceededError(Exception):
    """Raised by ``submit()`` when the per-user concurrency cap is hit.

    The API layer translates this into HTTP 429 with a ``Retry-After``
    hint so the client can back off without re-clicking through the
    wizard.
    """

    def __init__(self, user_id: str, current: int, limit: int) -> None:
        super().__init__(
            f"user {user_id!r} has {current} in-flight deploys (limit {limit})"
        )
        self.user_id = user_id
        self.current = current
        self.limit = limit


class DeploymentJobRunner:
    """Owns the async lifecycle of deployment rows post-create.

    Construction is cheap; the actual janitor + recovery sweep start
    lazily via ``await runner.start()`` so the FastAPI lifespan can
    decide when to kick them.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        max_concurrent_per_user: int = DEFAULT_MAX_CONCURRENT_PER_USER,
        heartbeat_interval_seconds: int = HEARTBEAT_INTERVAL_SECONDS,
        heartbeat_timeout_seconds: int = HEARTBEAT_TIMEOUT_SECONDS,
        janitor_interval_seconds: int = JANITOR_INTERVAL_SECONDS,
    ) -> None:
        self._session_factory = session_factory
        self._max_concurrent_per_user = max_concurrent_per_user
        self._heartbeat_interval = heartbeat_interval_seconds
        self._heartbeat_timeout = heartbeat_timeout_seconds
        self._janitor_interval = janitor_interval_seconds

        # In-flight task registry: deployment_id -> asyncio.Task. Keyed so
        # cancel() can locate the task without scanning.
        self._tasks: dict[UUID, asyncio.Task[None]] = {}
        # Per-user counters for the concurrency budget. Simpler than a
        # semaphore because we need ``try_acquire`` semantics.
        self._in_flight_per_user: dict[str, int] = {}
        # Unique id for this runner instance — written to ``worker_id``
        # so the recovery sweep on restart can tell its own old rows
        # apart from another process's.
        self._worker_id_prefix = uuid.uuid4().hex[:16]

        self._janitor_task: asyncio.Task[None] | None = None
        self._shutting_down = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Kick the janitor + recover orphans from previous runs."""
        if self._janitor_task is not None:
            return
        await self._recover_orphans()
        self._janitor_task = asyncio.create_task(
            self._janitor_loop(), name="deployment-janitor"
        )

    async def shutdown(self) -> None:
        """Stop accepting new work, signal in-flight tasks, then mark
        survivors as failed so they don't sit in DEPLOYING forever.
        """
        self._shutting_down = True
        if self._janitor_task is not None:
            self._janitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._janitor_task
            self._janitor_task = None

        if not self._tasks:
            return

        # Give workers a window to finish cleanly.
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks.values(), return_exceptions=True),
                timeout=SHUTDOWN_GRACE_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                "DEPLOYMENT_RUNNER_SHUTDOWN_TIMEOUT in_flight=%s",
                len(self._tasks),
            )

        # Cancel anything still running and mark FAILED so the UI stops
        # polling on next restart.
        for deployment_id, task in list(self._tasks.items()):
            if not task.done():
                task.cancel()
                await self._mark_failed(
                    deployment_id, error_message="server_shutdown"
                )
        self._tasks.clear()

    # ------------------------------------------------------------------
    # Submit / cancel
    # ------------------------------------------------------------------

    def submit(
        self,
        deployment_id: UUID,
        user_id: str,
    ) -> None:
        """Spawn the deploy task. Raises if the user is over budget."""
        if self._shutting_down:
            raise RuntimeError("DeploymentJobRunner is shutting down")
        current = self._in_flight_per_user.get(user_id, 0)
        if current >= self._max_concurrent_per_user:
            raise DeploymentBudgetExceededError(
                user_id, current, self._max_concurrent_per_user
            )
        self._in_flight_per_user[user_id] = current + 1
        task = asyncio.create_task(
            self._run(deployment_id, user_id),
            name=f"deployment-{deployment_id}",
        )
        self._tasks[deployment_id] = task

    async def cancel(self, deployment_id: UUID) -> bool:
        """Request cancellation. Returns True if the row was in-flight.

        Sets ``cancel_requested`` on the row and lets the heartbeat loop
        observe it on the next tick. Idempotent: cancelling a terminal
        or unknown row is a no-op returning False.
        """
        async with self._session_factory() as session:
            deployment = await session.get(AgentDeployment, deployment_id)
            if deployment is None or deployment.status not in ACTIVE_STATUSES:
                return False
            deployment.cancel_requested = True
            await session.commit()
        return True

    # ------------------------------------------------------------------
    # Core run loop
    # ------------------------------------------------------------------

    async def _run(self, deployment_id: UUID, user_id: str) -> None:
        worker_id = f"{self._worker_id_prefix}:{uuid.uuid4().hex[:8]}"
        heartbeat_stop = asyncio.Event()
        heartbeat_task: asyncio.Task[None] | None = None
        try:
            agent, revision, deployment = await self._claim(
                deployment_id, worker_id
            )
            if deployment is None:
                # Row vanished between submit() and _run (raced with a
                # force-delete). Nothing to do.
                return

            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(deployment_id, heartbeat_stop),
                name=f"heartbeat-{deployment_id}",
            )

            mode = DeploymentMode(deployment.mode)
            translator = translator_for(mode)

            # Validate / translate / deploy. Each call may raise — the
            # outer try captures it and lands the row in FAILED.
            validation = await translator.validate(
                agent, revision, deployment.config
            )
            if not validation.valid:
                await self._mark_failed(
                    deployment_id,
                    error_message=(
                        "validation: "
                        + "; ".join(e.message for e in validation.errors)
                    ),
                )
                return

            artifact = await translator.translate(
                agent, revision, deployment.config
            )
            await self._raise_if_cancel_requested(deployment_id)
            result = await translator.deploy(artifact, deployment.config, deployment)

            if result.success:
                async with self._session_factory() as session:
                    svc = DeploymentService(session)
                    updated = await svc.update_status(
                        deployment_id,
                        DeploymentStatus.ACTIVE,
                        endpoint_name=result.endpoint_name,
                        model_name=result.model_name,
                        external_resource_ids=result.external_resource_ids or None,
                    )
                    # TEMPORARY SHIM — D2 visibility flip for IN_APP mode.
                    # The chat composer's MessageInput.tsx filters agents by
                    # ``a.visibility === 'workspace'`` to populate the picker.
                    # Until the proper D2 rewire (reading from
                    # ``agent_deployments`` directly) lands, we mirror the
                    # ACTIVE transition by flipping agent.visibility to
                    # 'workspace' so newly-deployed agents appear in the
                    # picker without any frontend changes.
                    # See .omc/plans/we-don-t-need-legacy-composed-wren.md §D2
                    # for the real fix.
                    if updated.mode == DeploymentMode.IN_APP.value:
                        agent_row = await session.get(AgentV2, updated.agent_id)
                        if agent_row is not None:
                            agent_row.visibility = "workspace"
                    await session.commit()
            else:
                await self._mark_failed(
                    deployment_id,
                    error_message=result.error_message or "deploy_failed",
                )

        except _Cancelled:
            await self._mark_failed(deployment_id, error_message="cancelled")
        except Exception as exc:  # noqa: BLE001 -- surface as FAILED row
            logger.exception(
                "DEPLOYMENT_RUN_FAILED deployment=%s", deployment_id
            )
            await self._mark_failed(deployment_id, error_message=str(exc))
        finally:
            heartbeat_stop.set()
            if heartbeat_task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await heartbeat_task
            self._tasks.pop(deployment_id, None)
            self._in_flight_per_user[user_id] = max(
                0, self._in_flight_per_user.get(user_id, 1) - 1
            )

    async def _claim(
        self, deployment_id: UUID, worker_id: str
    ) -> tuple[AgentV2 | None, AgentRevision | None, AgentDeployment | None]:
        """Atomically transition pending → deploying + set worker fields.

        Returns the joined (agent, revision, deployment) tuple so the
        caller doesn't have to issue extra round-trips per status update.
        """
        async with self._session_factory() as session:
            deployment = await session.get(AgentDeployment, deployment_id)
            if deployment is None:
                return None, None, None
            if deployment.status not in (
                DeploymentStatus.PENDING.value,
                DeploymentStatus.DEPLOYING.value,
            ):
                # Already terminal (race with cancel-then-mark-failed).
                return None, None, None

            now = datetime.now(UTC)
            deployment.status = DeploymentStatus.DEPLOYING.value
            deployment.worker_id = worker_id
            deployment.last_heartbeat = now
            deployment.heartbeat_timeout_at = now + timedelta(
                seconds=self._heartbeat_timeout
            )
            agent = await session.get(AgentV2, deployment.agent_id)
            revision = await session.get(AgentRevision, deployment.revision_id)
            await session.commit()
            return agent, revision, deployment

    async def _heartbeat_loop(
        self,
        deployment_id: UUID,
        stop: asyncio.Event,
    ) -> None:
        """Refresh heartbeat columns until ``stop`` is set; raise on cancel."""
        while not stop.is_set():
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=self._heartbeat_interval
                )
                return  # stop fired
            except TimeoutError:
                pass

            now = datetime.now(UTC)
            async with self._session_factory() as session:
                deployment = await session.get(AgentDeployment, deployment_id)
                if deployment is None:
                    return
                if deployment.cancel_requested:
                    # Push the cancellation up to the run loop's outer
                    # try/except via a private sentinel that's narrow
                    # enough not to be confused with real errors.
                    deployment.last_heartbeat = now
                    await session.commit()
                    raise _Cancelled()
                deployment.last_heartbeat = now
                deployment.heartbeat_timeout_at = now + timedelta(
                    seconds=self._heartbeat_timeout
                )
                await session.commit()

    async def _raise_if_cancel_requested(self, deployment_id: UUID) -> None:
        async with self._session_factory() as session:
            deployment = await session.get(AgentDeployment, deployment_id)
            if deployment is not None and deployment.cancel_requested:
                raise _Cancelled()

    async def _mark_failed(
        self, deployment_id: UUID, *, error_message: str
    ) -> None:
        async with self._session_factory() as session:
            svc = DeploymentService(session)
            try:
                await svc.update_status(
                    deployment_id,
                    DeploymentStatus.FAILED,
                    error_message=error_message,
                )
                # Clear runtime columns — keeps the in-flight index small.
                deployment = await session.get(AgentDeployment, deployment_id)
                if deployment is not None:
                    deployment.worker_id = None
                    deployment.heartbeat_timeout_at = None
                await session.commit()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "DEPLOYMENT_MARK_FAILED_FAILED deployment=%s",
                    deployment_id,
                )

    # ------------------------------------------------------------------
    # Janitor + recovery
    # ------------------------------------------------------------------

    async def _janitor_loop(self) -> None:
        while not self._shutting_down:
            try:
                await asyncio.sleep(self._janitor_interval)
                await self._sweep_zombies()
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001
                logger.exception("DEPLOYMENT_JANITOR_TICK_FAILED")

    async def _sweep_zombies(self) -> None:
        now = datetime.now(UTC)
        async with self._session_factory() as session:
            stmt = (
                select(AgentDeployment)
                .where(
                    AgentDeployment.status.in_(
                        (
                            DeploymentStatus.PENDING.value,
                            DeploymentStatus.DEPLOYING.value,
                        )
                    )
                )
                .where(AgentDeployment.heartbeat_timeout_at < now)
            )
            result = await session.execute(stmt)
            zombies = list(result.scalars().all())

        for zombie in zombies:
            if zombie.id in self._tasks:
                # Live in this process — heartbeat must just be slow.
                # Skip rather than fight ourselves.
                continue
            logger.warning(
                "DEPLOYMENT_ZOMBIE_SWEEP deployment=%s last_heartbeat=%s",
                zombie.id,
                zombie.last_heartbeat,
            )
            # Re-read inside _cleanup_zombie so attempts/status reflect any
            # writes that landed between the sweep query and now (the
            # detached `zombie` object holds stale field values otherwise —
            # architect flagged this race in the post-impl review).
            await self._cleanup_zombie(zombie.id)

    async def _cleanup_zombie(self, deployment_id: UUID) -> None:
        # Re-read the row in a fresh session so the cleanup decision uses
        # current ``cleanup_attempts`` / ``status`` (the sweep cursor and
        # this call can be milliseconds apart but enough for a concurrent
        # cancel-then-mark-failed to have landed). If the row is already
        # terminal or vanished, bail without dispatching translator
        # deactivate.
        async with self._session_factory() as session:
            deployment = await session.get(AgentDeployment, deployment_id)
            if deployment is None or deployment.status not in (
                DeploymentStatus.PENDING.value,
                DeploymentStatus.DEPLOYING.value,
            ):
                return
            # Snapshot the fields we need before the session closes.
            mode_value = deployment.mode
            attempts_so_far = deployment.cleanup_attempts
        # Best-effort translator cleanup so external resources don't leak.
        # If the cleanup itself fails, transition through cleanup_failed
        # after exhausting attempts — matches the W4 contract.
        try:
            translator = translator_for(DeploymentMode(mode_value))
            # Refetch the row inside a fresh session for the translator
            # call — the translator reads ``external_resource_ids`` to
            # locate upstream artifacts, and we want a current snapshot.
            async with self._session_factory() as session:
                deployment = await session.get(AgentDeployment, deployment_id)
                if deployment is None:
                    return
            await translator.deactivate(deployment)
        except DeploymentCleanupError as exc:
            attempts = attempts_so_far + 1
            async with self._session_factory() as session:
                svc = DeploymentService(session)
                if attempts >= MAX_CLEANUP_ATTEMPTS:
                    await svc.mark_cleanup_failed(
                        deployment_id,
                        error_message=f"worker_zombie + {exc}",
                    )
                else:
                    await svc.increment_cleanup_attempts(deployment_id)
                await session.commit()
            return
        except Exception:  # noqa: BLE001
            logger.exception(
                "ZOMBIE_TRANSLATOR_DEACTIVATE_RAISED deployment=%s",
                deployment_id,
            )

        await self._mark_failed(deployment_id, error_message="worker_zombie")

    async def _recover_orphans(self) -> None:
        """On startup, mark any PENDING rows from prior runs FAILED.

        DEPLOYING rows are left for the janitor to sweep — those had
        external state in progress, so the translator-cleanup path is
        the right channel for them.
        """
        async with self._session_factory() as session:
            stmt = select(AgentDeployment).where(
                AgentDeployment.status == DeploymentStatus.PENDING.value
            )
            result = await session.execute(stmt)
            orphans = list(result.scalars().all())
        for orphan in orphans:
            await self._mark_failed(
                orphan.id, error_message="server_restart_before_start"
            )


class _Cancelled(Exception):
    """Internal sentinel — caught by the outer ``_run`` try/except to
    distinguish user-initiated cancel from other failures.
    """


__all__ = [
    "DEFAULT_MAX_CONCURRENT_PER_USER",
    "DeploymentBudgetExceededError",
    "DeploymentJobRunner",
    "HEARTBEAT_INTERVAL_SECONDS",
    "HEARTBEAT_TIMEOUT_SECONDS",
    "JANITOR_INTERVAL_SECONDS",
]
