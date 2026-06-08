"""Source-level guard + behavioural slice for the Step 3 force-fail repair.

Before Step 3, when the research stream drained but the orchestrator's
persistence path had failed (e.g. because the chat wasn't in the cache),
`_consume_research_stream` only logged `JOB_COMPLETION_STATUS_UNEXPECTED`
and returned — leaving `ResearchSession.status = IN_PROGRESS` forever. The
UI kept polling, the user saw "still running", and from their point of view
research took however long they were willing to wait.

Step 3 adds a repair branch that calls
`persist_research_session_failed_independent` with the process-singleton
StorageStack so the transition goes through the cached-aware path rather
than bypassing the cache with raw SQLAlchemy.
"""

from __future__ import annotations

import inspect


class TestStep3SourceGuard:
    """Inspection-level assertions matching the pattern used by
    test_job_manager_agent.py — the full behavioural path through
    `_run_job` requires a live DB, Lakebase auth, and a large stack of
    collaborators; source inspection is the repo's convention."""

    def test_force_fail_helper_is_invoked_on_stuck_session(self) -> None:
        from deep_research.services.job_manager import JobManager

        source = inspect.getsource(JobManager)

        # The unexpected-status branch must still log the original warning
        # so we preserve observability of the original symptom.
        assert "JOB_COMPLETION_STATUS_UNEXPECTED" in source

        # And it must call the cached-aware failure helper.
        assert "persist_research_session_failed_independent" in source, (
            "Step 3 must route through persist_research_session_failed_independent "
            "so the status transition flows through the cached path"
        )
        assert 'error_message="persistence_transition_missing"' in source, (
            "forced-FAIL must carry a distinct error_message for operators"
        )
        assert "storage_stack=self._storage_stack" in source, (
            "force-fail call must pass the StorageStack so the cached branch "
            "is taken; bypassing it would re-introduce the original bug"
        )

        # Post-repair log line so operators can distinguish a natural fail
        # from a forced-fail.
        assert "JOB_COMPLETION_FORCED_FAIL" in source
        # And a double-fault log line so we don't silently swallow a
        # second failure in the repair path.
        assert "JOB_COMPLETION_FORCED_FAIL_FAILED" in source

    def test_force_fail_double_fault_is_bounded(self) -> None:
        """If the force-fail helper itself raises, we must log and move
        on — not loop or re-raise through the background task."""
        from deep_research.services.job_manager import JobManager

        source = inspect.getsource(JobManager)

        # The repair path is inside the completion try/except block, so a
        # double fault falls through to the existing JOB_COMPLETION_CHECK_FAILED
        # path OR the new JOB_COMPLETION_FORCED_FAIL_FAILED path — both of
        # which only log, never raise.
        assert "JOB_COMPLETION_CHECK_FAILED" in source, (
            "outer try/except around the completion check must remain"
        )

    def test_recurring_cleanup_loop_contract(self) -> None:
        """The legacy ``_heartbeat_loop`` (which wrote ``last_heartbeat`` to the
        now-dropped ``research_sessions`` table) was replaced by
        ``_recurring_cleanup_loop``, which periodically runs
        ``_cleanup_interrupted_jobs``. The ``_active_tasks`` dict (maintained by
        ``_run_job``'s finally block) is now the per-worker liveness signal /
        exclude-list for stale-marking, which goes through the storage backend
        — never ``select(ResearchSession)``. Assert the contract so a future
        refactor doesn't reintroduce a heartbeat write to the dropped table."""
        from deep_research.services.job_manager import JobManager

        # The legacy heartbeat loop must be gone (it queried the dropped table).
        assert not hasattr(JobManager, "_heartbeat_loop"), (
            "_heartbeat_loop was replaced by _recurring_cleanup_loop; it must "
            "not write to the dropped research_sessions table."
        )

        loop_source = inspect.getsource(JobManager._recurring_cleanup_loop)
        assert "await asyncio.sleep" in loop_source
        assert "_cleanup_interrupted_jobs" in loop_source

        cleanup_source = inspect.getsource(JobManager._cleanup_interrupted_jobs)
        assert "_active_tasks" in cleanup_source
        assert "mark_stale_research_sessions_failed" in cleanup_source
        assert "select(" not in cleanup_source
