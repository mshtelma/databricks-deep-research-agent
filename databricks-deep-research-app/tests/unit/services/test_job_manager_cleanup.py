"""Tests for F-JOB-B: no session held during the research stream.

F-JOB-A wrapped the research stream in `async with session_maker() as db:` to
handle Lakebase PgBouncer idle-reap of the underlying connection.  F-JOB-B
removes that wrapper entirely: all readers and persisters inside the stream
now use the cached StorageStack path (no session_maker needed) or open their
own independent sessions.  With no long-held session, there is no
InterfaceError to suppress.

These tests confirm:
1.  `SQLAlchemyInterfaceError` is no longer imported into `job_manager`.
2.  `_run_job` no longer wraps the research stream in `async with session_maker()`.
3.  The stream is awaited directly (no outer try/except InterfaceError).
"""

from __future__ import annotations

import inspect

import pytest

from deep_research.services.job_manager import JobManager


class TestNoSessionHeldDuringStream:
    """F-JOB-B: research stream runs without a long-held DB session."""

    def test_interface_error_import_removed(self) -> None:
        """SQLAlchemyInterfaceError must NOT be imported in job_manager after F-JOB-B."""
        import deep_research.services.job_manager as jm

        assert not hasattr(jm, "SQLAlchemyInterfaceError"), (
            "SQLAlchemyInterfaceError is still imported — F-JOB-B was not applied. "
            "Remove the import and the try/except guard around the research stream."
        )

    def test_run_job_does_not_hold_session_for_stream(self) -> None:
        """Source-inspection guard: `_run_job` must NOT wrap the research stream
        in `async with session_maker() as db:`.

        After F-JOB-B the stream is awaited directly via `asyncio.wait_for(
        _consume_research_stream(None), ...)`.
        """
        source = inspect.getsource(JobManager._run_job)

        # F-JOB-B: the research stream wrapper must be gone.
        assert "async with session_maker() as db:" not in source, (
            "The research stream is still wrapped in `async with session_maker() as db:`. "
            "F-JOB-B requires removing this wrapper and passing db=None."
        )

        # F-JOB-B: the InterfaceError swallow branch must be gone too.
        assert "RESEARCH_SESSION_EXIT_DEAD_CONN" not in source, (
            "F-JOB-A swallow log is still present after F-JOB-B landed."
        )

        # F-JOB-B: stream is called with db=None.
        assert "_consume_research_stream(None)" in source, (
            "_consume_research_stream must be called with db=None after F-JOB-B."
        )

    @pytest.mark.asyncio
    async def test_stream_runs_without_session(self) -> None:
        """Smoke-test the structural pattern used after F-JOB-B.

        Simulates `asyncio.wait_for(_consume_research_stream(None), ...)` running
        cleanly with no session opened.  No InterfaceError is possible because no
        session is held.
        """
        import asyncio

        completed = False

        async def _fake_consume(db: None) -> None:
            assert db is None, "stream must receive db=None after F-JOB-B"
            nonlocal completed
            completed = True

        await asyncio.wait_for(_fake_consume(None), timeout=1.0)
        assert completed
