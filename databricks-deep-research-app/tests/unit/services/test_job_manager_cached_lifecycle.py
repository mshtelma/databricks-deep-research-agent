"""Source-level guards for the cached-aware lifecycle handlers in `_run_job`.

Before this change, every post-stream transition in ``JobManager._run_job``
(timeout / completion-check / cancel / error) and ``cancel_job`` called
``get_session_maker()`` directly and tried to mutate a SQLAlchemy
``ResearchSession`` row. In cached storage mode that row doesn't exist —
session state lives in the ChatDocument — so the transitions silently
no-op'd. Sessions got stuck at ``IN_PROGRESS`` forever, plugin lifecycle
hooks never fired, and the SSE polling loop never closed cleanly.

The fix routes every such transition through the cached-aware persistence
helpers (``persist_research_session_failed_independent`` /
``persist_research_session_cancelled_independent``) and reads status
through ``load_session_control_view``. The full behavioural path through
``_run_job`` requires a live DB, LLM client, etc., so we guard at the
source level — the same convention as
``test_job_manager_force_fail.py`` and ``test_job_manager_agent.py``.
"""

from __future__ import annotations

import inspect


def _job_manager_source() -> str:
    from deep_research.services.job_manager import JobManager

    return inspect.getsource(JobManager)


class TestTimeoutHandlerRoutesThroughCachedHelper:
    def test_timeout_calls_failed_helper_with_stack(self) -> None:
        source = _job_manager_source()
        # The timeout block must route through the cached-aware failure
        # helper, passing both storage_stack and chat_id so the cached
        # branch fires.
        assert "RESEARCH_TIMEOUT" in source
        assert "persist_research_session_failed_independent" in source
        assert "Research timed out after" in source, (
            "timeout error_message must be preserved verbatim so the UI can "
            "surface it to the user"
        )

    def test_timeout_does_not_bypass_cached_path_with_raw_session_maker(
        self,
    ) -> None:
        """Before the fix this block opened `get_session_maker()` directly
        and mutated `ResearchSession.status`. That's exactly what the
        cached-aware helper replaces. A regression would put this pattern
        back into the `_run_job` post-stream handlers (not the startup
        cleanup path, which still legitimately operates on SQL rows)."""
        from deep_research.services.job_manager import JobManager

        # Scope the check to `_run_job` only. `_cleanup_interrupted_jobs`
        # still uses raw SQL mutation — that's a separate code path that
        # only runs on startup against whatever rows exist in Lakebase.
        source = inspect.getsource(JobManager._run_job)
        assert "session.status = ResearchStatus.FAILED" not in source, (
            "_run_job post-stream handlers must not mutate SQL rows "
            "directly; route through persist_research_session_failed_independent"
        )
        assert "session.status = ResearchStatus.CANCELLED" not in source, (
            "_run_job cancel handler must not mutate SQL rows directly; "
            "route through persist_research_session_cancelled_independent"
        )


class TestCancellationHandlerRoutesThroughCachedHelper:
    def test_cancellederror_calls_cancelled_helper(self) -> None:
        source = _job_manager_source()
        # CancelledError block must invoke the cancelled helper, not
        # `db.get + session.status = CANCELLED + commit`.
        assert "asyncio.CancelledError" in source
        assert "persist_research_session_cancelled_independent" in source
        assert "JOB_CANCELLED_BY_TASK" in source, (
            "preserve the original log line for operator continuity"
        )

    def test_cancel_job_uses_lookup_and_helper(self) -> None:
        """`JobManager.cancel_job` previously fetched a ResearchSession via
        `db.get` and set `.status = CANCELLED` directly. Cached mode broke
        both sides. The new code routes through `load_session_control_view`
        for ownership and `persist_research_session_cancelled_independent`
        for the transition."""
        from deep_research.services.job_manager import JobManager

        source = inspect.getsource(JobManager.cancel_job)
        assert "load_session_control_view" in source
        assert "persist_research_session_cancelled_independent" in source
        assert "JOB_CANCEL_DENIED" in source, (
            "ownership-denied log preserved"
        )


class TestErrorHandlerRoutesThroughCachedHelper:
    def test_exception_handler_calls_failed_helper(self) -> None:
        source = _job_manager_source()
        # Error path must route through the cached-aware failure helper
        # AND emit the lifecycle hook on real transitions.
        assert "JOB_FAILED" in source
        assert "emitter.job_failed" in source, (
            "plugin lifecycle hook must still fire after error persist"
        )
        assert "error_category" in source

    def test_completion_check_uses_storage_aware_lookup(self) -> None:
        source = _job_manager_source()
        # Completion-check must read through `load_session_control_view`
        # (so cached mode resolves the ChatDocument status) rather than
        # `db.get(ResearchSession, ...)`.
        assert "load_session_control_view" in source
        assert "JOB_COMPLETED" in source, "completion log preserved"


class TestForceFailRepairStillWorksAfterRefactor:
    """Step 3's force-fail repair branch must remain wired up after the
    Step 4 post-stream refactor — otherwise the "stuck at IN_PROGRESS"
    safety net disappears when end-of-run persistence silently fails."""

    def test_force_fail_branch_preserved(self) -> None:
        source = _job_manager_source()
        assert "JOB_COMPLETION_STATUS_UNEXPECTED" in source
        assert "JOB_COMPLETION_FORCED_FAIL" in source
        assert "JOB_COMPLETION_FORCED_FAIL_FAILED" in source
        # Force-fail still threads storage_stack + chat_id.
        assert "persistence_transition_missing" in source
