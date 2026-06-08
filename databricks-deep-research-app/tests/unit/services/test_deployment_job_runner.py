"""W13: focused unit tests for ``DeploymentJobRunner`` (Phase 3 / W12).

The runner orchestrates the async deploy lifecycle: claim → heartbeat
→ translate → deploy → land-active|failed, plus a janitor that sweeps
heartbeat-stale rows and an orphan-recovery sweep on startup. The
full happy-path requires a real Postgres because it mutates rows
across multiple sessions; that lives in the integration suite. This
file pins the parts that can be exercised with the existing
mock-session fixtures:

1. ``submit`` raises ``DeploymentBudgetExceededError`` past the per-user
   cap.
2. ``cancel`` sets ``cancel_requested`` on the row when it's in an
   active status, and is a no-op for terminal rows.
3. ``_mark_failed`` lands the row in ``failed`` and clears the runtime
   worker columns so the in-flight index stays small.
4. ``_recover_orphans`` marks every PENDING row FAILED with
   ``error_message="server_restart_before_start"`` (DEPLOYING rows are
   left for the janitor on purpose).

The W2 state-machine contract (``failed`` is terminal, ``cleanup_failed``
is terminal, etc.) is pinned in
``tests/unit/models/test_agent_deployment_status_sets.py``.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.models.agent_deployment import (
    AgentDeployment,
    DeploymentMode,
    DeploymentStatus,
)
from deep_research.services.deployment.job_runner import (
    DEFAULT_MAX_CONCURRENT_PER_USER,
    DeploymentBudgetExceededError,
    DeploymentJobRunner,
)


def _make_row(
    status: DeploymentStatus = DeploymentStatus.PENDING,
    *,
    cleanup_attempts: int = 0,
    deployed_by: str = "user-1",
) -> AgentDeployment:
    row = AgentDeployment(
        agent_id=uuid4(),
        revision_id=uuid4(),
        mode=DeploymentMode.IN_APP.value,
        status=status.value,
        config={"mode": "in_app"},
        deployed_by=deployed_by,
    )
    row.id = uuid4()
    row.cleanup_attempts = cleanup_attempts
    row.cancel_requested = False
    row.worker_id = None
    row.last_heartbeat = None
    row.heartbeat_timeout_at = None
    return row


def _make_runner(
    *, max_concurrent_per_user: int = DEFAULT_MAX_CONCURRENT_PER_USER
) -> tuple[DeploymentJobRunner, MagicMock]:
    """Build a runner whose session factory yields fresh mocks per call.

    Each call to ``session_factory()`` returns an ``AsyncMock`` configured
    as a context-manager (so ``async with self._session_factory() as s:``
    in the runner returns a session-shaped mock).
    """

    def _new_session() -> AsyncMock:
        s = AsyncMock()
        s.commit = AsyncMock()
        s.get = AsyncMock(return_value=None)  # default; tests override
        s.execute = AsyncMock()
        # async with self._session_factory() as session: ...
        s.__aenter__ = AsyncMock(return_value=s)
        s.__aexit__ = AsyncMock(return_value=False)
        return s

    factory = MagicMock(side_effect=_new_session)
    runner = DeploymentJobRunner(
        session_factory=factory,
        max_concurrent_per_user=max_concurrent_per_user,
    )
    return runner, factory


class TestBudget:
    @pytest.mark.asyncio
    async def test_submit_under_budget_creates_task(self) -> None:
        runner, _factory = _make_runner(max_concurrent_per_user=2)
        # We don't await the spawned task — the patched session factory
        # means `_run` finds nothing and exits quickly.
        runner.submit(uuid4(), user_id="alice")
        assert runner._in_flight_per_user["alice"] == 1  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_submit_at_budget_raises(self) -> None:
        runner, _factory = _make_runner(max_concurrent_per_user=1)
        runner.submit(uuid4(), user_id="bob")
        with pytest.raises(DeploymentBudgetExceededError) as exc_info:
            runner.submit(uuid4(), user_id="bob")
        assert exc_info.value.user_id == "bob"
        assert exc_info.value.limit == 1
        assert exc_info.value.current == 1

    @pytest.mark.asyncio
    async def test_budget_is_per_user(self) -> None:
        runner, _factory = _make_runner(max_concurrent_per_user=1)
        runner.submit(uuid4(), user_id="alice")
        # bob has his own quota — must not be blocked by alice.
        runner.submit(uuid4(), user_id="bob")
        assert runner._in_flight_per_user["alice"] == 1  # noqa: SLF001
        assert runner._in_flight_per_user["bob"] == 1  # noqa: SLF001

    def test_submit_during_shutdown_raises_runtime_error(self) -> None:
        runner, _factory = _make_runner()
        runner._shutting_down = True  # noqa: SLF001 -- direct flip for test
        with pytest.raises(RuntimeError):
            runner.submit(uuid4(), user_id="alice")


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_sets_flag_when_row_is_active(self) -> None:
        runner, factory = _make_runner()
        row = _make_row(status=DeploymentStatus.DEPLOYING)
        # Stage the session.get(...) call result.
        first_session = factory.side_effect()
        first_session.get = AsyncMock(return_value=row)
        factory.side_effect = [first_session]

        result = await runner.cancel(row.id)

        assert result is True
        assert row.cancel_requested is True
        first_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancel_is_noop_when_row_is_terminal(self) -> None:
        runner, factory = _make_runner()
        row = _make_row(status=DeploymentStatus.DEACTIVATED)
        first_session = factory.side_effect()
        first_session.get = AsyncMock(return_value=row)
        factory.side_effect = [first_session]

        result = await runner.cancel(row.id)

        assert result is False
        assert row.cancel_requested is False
        # Terminal rows aren't mutated so no commit fires.
        first_session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancel_missing_row_returns_false(self) -> None:
        runner, factory = _make_runner()
        first_session = factory.side_effect()
        first_session.get = AsyncMock(return_value=None)
        factory.side_effect = [first_session]

        assert await runner.cancel(uuid4()) is False


class TestMarkFailed:
    @pytest.mark.asyncio
    async def test_mark_failed_lands_terminal_and_clears_worker(self) -> None:
        runner, factory = _make_runner()
        row = _make_row(status=DeploymentStatus.DEPLOYING)
        row.worker_id = "w-123"
        row.last_heartbeat = datetime.now(UTC)

        session = factory.side_effect()
        session.get = AsyncMock(return_value=row)
        factory.side_effect = [session]

        with patch(
            "deep_research.services.deployment.job_runner.DeploymentService"
        ) as MockSvc:
            instance = MockSvc.return_value
            instance.update_status = AsyncMock()
            await runner._mark_failed(  # noqa: SLF001
                row.id, error_message="boom"
            )

        instance.update_status.assert_awaited_once()
        # Worker columns get cleared so the in-flight index doesn't bloat.
        assert row.worker_id is None
        assert row.heartbeat_timeout_at is None


class TestOrphanRecovery:
    @pytest.mark.asyncio
    async def test_recover_orphans_marks_pending_rows_failed(self) -> None:
        runner, factory = _make_runner()
        pending_rows = [
            _make_row(status=DeploymentStatus.PENDING),
            _make_row(status=DeploymentStatus.PENDING),
        ]

        session = factory.side_effect()
        scalars = MagicMock()
        scalars.all = MagicMock(return_value=pending_rows)
        exec_result = MagicMock()
        exec_result.scalars = MagicMock(return_value=scalars)
        session.execute = AsyncMock(return_value=exec_result)

        # Subsequent `_mark_failed` calls each open a new session — supply
        # placeholder sessions for them.
        mark_failed_session_1 = factory.side_effect()
        mark_failed_session_1.get = AsyncMock(return_value=pending_rows[0])
        mark_failed_session_2 = factory.side_effect()
        mark_failed_session_2.get = AsyncMock(return_value=pending_rows[1])

        factory.side_effect = [
            session,
            mark_failed_session_1,
            mark_failed_session_2,
        ]

        with patch(
            "deep_research.services.deployment.job_runner.DeploymentService"
        ) as MockSvc:
            instance = MockSvc.return_value
            instance.update_status = AsyncMock()
            await runner._recover_orphans()  # noqa: SLF001

        # One update_status per orphan, all to FAILED with the recovery
        # error message.
        assert instance.update_status.await_count == 2
        for call in instance.update_status.await_args_list:
            assert call.args[1] == DeploymentStatus.FAILED
            assert call.kwargs["error_message"] == "server_restart_before_start"
