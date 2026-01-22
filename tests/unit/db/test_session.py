"""Tests for database session management."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.db import session


class TestDisposeEngineAsync:
    """Tests for async engine disposal helper."""

    @pytest.mark.asyncio
    async def test_dispose_engine_success(self) -> None:
        """Should dispose engine successfully."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        await session._dispose_engine_async(mock_engine)

        mock_engine.dispose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dispose_engine_handles_error(self) -> None:
        """Should log warning and not raise on disposal error."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock(side_effect=Exception("Disposal failed"))

        # Should not raise
        await session._dispose_engine_async(mock_engine)

        mock_engine.dispose.assert_awaited_once()


class TestEngineDisposalTracking:
    """Tests for engine disposal task tracking."""

    def setup_method(self) -> None:
        """Reset module state before each test."""
        session._engine = None
        session._async_session_maker = None
        session._credential_provider = None
        session._pending_disposal_tasks.clear()

    def teardown_method(self) -> None:
        """Clean up module state after each test."""
        session._engine = None
        session._async_session_maker = None
        session._credential_provider = None
        session._pending_disposal_tasks.clear()

    @pytest.mark.asyncio
    async def test_disposal_task_tracked_and_removed(self) -> None:
        """Verify disposal tasks are tracked and removed on completion."""
        # Create a mock engine
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        # Get the running event loop
        loop = asyncio.get_running_loop()

        # Create and track a disposal task
        task = loop.create_task(session._dispose_engine_async(mock_engine))
        session._pending_disposal_tasks.add(task)
        task.add_done_callback(session._pending_disposal_tasks.discard)

        # Task should be tracked
        assert len(session._pending_disposal_tasks) == 1

        # Wait for task completion
        await task

        # Task should be removed after completion
        assert len(session._pending_disposal_tasks) == 0

    @pytest.mark.asyncio
    async def test_multiple_disposal_tasks_tracked(self) -> None:
        """Verify multiple concurrent disposal tasks are properly tracked."""
        mock_engines = [AsyncMock() for _ in range(3)]
        for engine in mock_engines:
            engine.dispose = AsyncMock()

        loop = asyncio.get_running_loop()

        # Create multiple disposal tasks
        tasks = []
        for engine in mock_engines:
            task = loop.create_task(session._dispose_engine_async(engine))
            session._pending_disposal_tasks.add(task)
            task.add_done_callback(session._pending_disposal_tasks.discard)
            tasks.append(task)

        # All tasks should be tracked
        assert len(session._pending_disposal_tasks) == 3

        # Wait for all tasks
        await asyncio.gather(*tasks)

        # All tasks should be removed
        assert len(session._pending_disposal_tasks) == 0


class TestGetEngineTokenRefresh:
    """Tests for token refresh logic in get_engine."""

    def setup_method(self) -> None:
        """Reset module state before each test."""
        session._engine = None
        session._async_session_maker = None
        session._credential_provider = None
        session._pending_disposal_tasks.clear()

    def teardown_method(self) -> None:
        """Clean up module state after each test."""
        session._engine = None
        session._async_session_maker = None
        session._credential_provider = None
        session._pending_disposal_tasks.clear()

    @pytest.mark.asyncio
    async def test_engine_disposal_on_token_expiry(self) -> None:
        """Verify engine is properly disposed when token expires."""
        # Create mock credential provider with expired token
        mock_provider = MagicMock()
        mock_cred = MagicMock()
        mock_cred.is_expired = True
        mock_provider._credential = mock_cred
        mock_provider.get_credential = MagicMock()
        mock_provider.build_connection_url = MagicMock(
            return_value="postgresql+asyncpg://user:pass@localhost/db"
        )

        # Create mock engine
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        # Set up module state
        session._credential_provider = mock_provider
        session._engine = mock_engine
        session._async_session_maker = MagicMock()

        # Mock settings for Lakebase
        mock_settings = MagicMock()
        mock_settings.use_lakebase = True
        mock_settings.debug = False
        mock_settings.database_url = None

        # Patch create_async_engine to avoid actual DB connection
        with patch.object(session, "create_async_engine") as mock_create_engine:
            new_mock_engine = MagicMock()
            mock_create_engine.return_value = new_mock_engine

            # Call get_engine which should detect expired token and dispose
            engine = session.get_engine(mock_settings)

            # Verify credential refresh was triggered
            mock_provider.get_credential.assert_called_once_with(force_refresh=True)

            # Verify new engine was created
            mock_create_engine.assert_called_once()
            assert engine == new_mock_engine

            # Old engine should have been scheduled for disposal
            # Wait briefly to allow background task to complete
            await asyncio.sleep(0.1)


class TestResetEngine:
    """Tests for module-level state reset."""

    def setup_method(self) -> None:
        """Reset module state before each test."""
        session._engine = None
        session._async_session_maker = None

    def teardown_method(self) -> None:
        """Clean up module state after each test."""
        session._engine = None
        session._async_session_maker = None

    @pytest.mark.asyncio
    async def test_close_db_disposes_engine(self) -> None:
        """Verify close_db properly disposes the engine."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        session._engine = mock_engine
        session._async_session_maker = MagicMock()

        await session.close_db()

        mock_engine.dispose.assert_awaited_once()
        assert session._engine is None
        assert session._async_session_maker is None
