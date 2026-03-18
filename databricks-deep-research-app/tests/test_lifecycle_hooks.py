"""
Unit tests for plugin lifecycle callback system.

Tests the core hook emission mechanism including:
- Hook emission and delivery
- Timeout protection
- Error isolation
- Throttling
- Validation error unpacking
- Hook cache management
- Feature flags
"""

import asyncio
import dataclasses
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field, ValidationError

from deep_research.plugins.lifecycle import (
    JobSubmittedEvent,
    JobStartedEvent,
    JobCompletedEvent,
    JobFailedEvent,
    SynthesisConfigEvent,
    SynthesisChunkEvent,
    ValidationErrorEvent,
    ValidationSuccessEvent,
    JobLifecycleListener,
    EventEmitter,
    create_validation_error_event,
    unpack_validation_error,
)
from deep_research.plugins.manager import PluginManager, HookMetrics


# ============================================================================
# Mock Plugin for Testing
# ============================================================================


class MockPlugin:
    """Mock plugin that records all events received."""

    def __init__(self, name: str = "mock_plugin"):
        self.name = name
        self.version = "1.0.0"
        self.events_received: list[tuple[str, Any]] = []
        self.call_count = 0
        self._should_timeout = False
        self._should_error = False

    def initialize(self, app_config: Any) -> None:
        """Initialize plugin."""
        pass

    def shutdown(self) -> None:
        """Shutdown plugin."""
        pass

    # Lifecycle hooks
    def on_job_submitted(self, event: JobSubmittedEvent) -> None:
        self.call_count += 1
        self.events_received.append(("on_job_submitted", event))
        if self._should_timeout:
            import time
            time.sleep(100)  # Simulate hang
        if self._should_error:
            raise ValueError("Simulated error")

    def on_job_started(self, event: JobStartedEvent) -> None:
        self.call_count += 1
        self.events_received.append(("on_job_started", event))

    def on_job_completed(self, event: JobCompletedEvent) -> None:
        self.call_count += 1
        self.events_received.append(("on_job_completed", event))

    def on_job_failed(self, event: JobFailedEvent) -> None:
        self.call_count += 1
        self.events_received.append(("on_job_failed", event))

    def on_synthesis_config(self, event: SynthesisConfigEvent) -> None:
        self.call_count += 1
        self.events_received.append(("on_synthesis_config", event))

    def on_synthesis_chunk(self, event: SynthesisChunkEvent) -> None:
        self.call_count += 1
        self.events_received.append(("on_synthesis_chunk", event))

    def on_validation_error(self, event: ValidationErrorEvent) -> None:
        self.call_count += 1
        self.events_received.append(("on_validation_error", event))

    def on_validation_success(self, event: ValidationSuccessEvent) -> None:
        self.call_count += 1
        self.events_received.append(("on_validation_success", event))


class AsyncMockPlugin(MockPlugin):
    """Mock plugin with async hooks."""

    async def on_job_submitted(self, event: JobSubmittedEvent) -> None:
        self.call_count += 1
        self.events_received.append(("on_job_submitted", event))
        if self._should_timeout:
            await asyncio.sleep(100)  # Simulate async hang
        if self._should_error:
            raise ValueError("Async simulated error")


# ============================================================================
# Test Hook Emission Basics
# ============================================================================


@pytest.mark.asyncio
async def test_hook_emission_basic():
    """Test that hooks are called correctly."""
    plugin = MockPlugin()
    manager = PluginManager()
    manager._plugins = [plugin]
    manager._build_hook_cache()

    # Emit hook
    job_id = uuid4()
    event = JobSubmittedEvent(
        job_id=job_id,
        chat_id="test-chat",
        query="test query",
        user_id="test-user",
        research_config={"mode": "test"},
        timestamp=datetime.now(timezone.utc),
    )

    stats = await manager.emit_hook("on_job_submitted", event)

    # Verify hook was called
    assert plugin.call_count == 1
    assert len(plugin.events_received) == 1
    assert plugin.events_received[0][0] == "on_job_submitted"
    assert plugin.events_received[0][1].job_id == job_id

    # Verify stats
    assert stats["succeeded"] == 1
    assert stats["failed"] == 0
    assert stats["timed_out"] == 0


@pytest.mark.asyncio
async def test_hook_emission_multiple_plugins():
    """Test that multiple plugins receive hooks."""
    plugin1 = MockPlugin(name="plugin1")
    plugin2 = MockPlugin(name="plugin2")
    manager = PluginManager()
    manager._plugins = [plugin1, plugin2]
    manager._build_hook_cache()

    event = JobStartedEvent(
        job_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
    )

    stats = await manager.emit_hook("on_job_started", event)

    # Both plugins received the event
    assert plugin1.call_count == 1
    assert plugin2.call_count == 1
    assert stats["succeeded"] == 2


@pytest.mark.asyncio
async def test_hook_emission_async_plugin():
    """Test that async hooks work."""
    plugin = AsyncMockPlugin()
    manager = PluginManager()
    manager._plugins = [plugin]
    manager._build_hook_cache()

    event = JobSubmittedEvent(
        job_id=uuid4(),
        chat_id="test",
        query="test",
        user_id="user",
        research_config={},
        timestamp=datetime.now(timezone.utc),
    )

    stats = await manager.emit_hook("on_job_submitted", event)

    assert plugin.call_count == 1
    assert stats["succeeded"] == 1


# ============================================================================
# Test Timeout Protection
# ============================================================================


@pytest.mark.asyncio
async def test_hook_timeout_protection():
    """Test that hooks timeout after configured duration."""
    from concurrent.futures import ThreadPoolExecutor

    plugin = MockPlugin()
    plugin._should_timeout = True  # Will hang

    manager = PluginManager()
    manager._plugins = [plugin]
    manager._hook_timeout = 1  # 1 second timeout
    manager._executor = ThreadPoolExecutor(max_workers=4)
    manager._build_hook_cache()

    event = JobSubmittedEvent(
        job_id=uuid4(),
        chat_id="test",
        query="test",
        user_id="user",
        research_config={},
        timestamp=datetime.now(timezone.utc),
    )

    import time
    start = time.time()
    stats = await manager.emit_hook("on_job_submitted", event, timeout_override=1)
    elapsed = time.time() - start

    # Should timeout quickly (around 1 second, not 100)
    assert elapsed < 3
    assert stats["timed_out"] == 1
    assert stats["succeeded"] == 0

    # Metrics should track timeout
    metrics = manager.get_hook_metrics()
    assert "on_job_submitted" in metrics
    assert metrics["on_job_submitted"]["total_timeouts"] == 1


@pytest.mark.asyncio
async def test_hook_timeout_async():
    """Test timeout for async hooks."""
    plugin = AsyncMockPlugin()
    plugin._should_timeout = True

    manager = PluginManager()
    manager._plugins = [plugin]
    manager._hook_timeout = 1
    manager._build_hook_cache()

    event = JobSubmittedEvent(
        job_id=uuid4(),
        chat_id="test",
        query="test",
        user_id="user",
        research_config={},
        timestamp=datetime.now(timezone.utc),
    )

    stats = await manager.emit_hook("on_job_submitted", event, timeout_override=1)

    assert stats["timed_out"] == 1


# ============================================================================
# Test Error Isolation
# ============================================================================


@pytest.mark.asyncio
async def test_hook_error_isolation():
    """Test that plugin errors don't crash framework."""
    plugin = MockPlugin()
    plugin._should_error = True  # Will raise exception

    manager = PluginManager()
    manager._plugins = [plugin]
    manager._build_hook_cache()

    event = JobSubmittedEvent(
        job_id=uuid4(),
        chat_id="test",
        query="test",
        user_id="user",
        research_config={},
        timestamp=datetime.now(timezone.utc),
    )

    # Should not raise - error is caught
    stats = await manager.emit_hook("on_job_submitted", event)

    assert stats["failed"] == 1
    assert stats["succeeded"] == 0
    assert len(stats["errors"]) == 1
    assert stats["errors"][0]["error_type"] == "ValueError"

    # Metrics should track failure
    metrics = manager.get_hook_metrics()
    assert metrics["on_job_submitted"]["total_failures"] == 1


@pytest.mark.asyncio
async def test_hook_error_isolation_multiple_plugins():
    """Test that one plugin error doesn't affect others."""
    plugin1 = MockPlugin(name="plugin1")
    plugin1._should_error = True
    plugin2 = MockPlugin(name="plugin2")  # Works fine

    manager = PluginManager()
    manager._plugins = [plugin1, plugin2]
    manager._build_hook_cache()

    # Use on_job_submitted which checks _should_error
    event = JobSubmittedEvent(
        job_id=uuid4(),
        chat_id="test",
        query="test",
        user_id="user",
        research_config={},
        timestamp=datetime.now(timezone.utc),
    )

    stats = await manager.emit_hook("on_job_submitted", event)

    # Plugin2 should still succeed
    assert stats["failed"] == 1
    assert stats["succeeded"] == 1
    assert plugin2.call_count == 1


# ============================================================================
# Test Validation Error Unpacking
# ============================================================================


def test_validation_error_unpacking():
    """Test that ValidationError is unpacked correctly."""

    class TestSchema(BaseModel):
        name: str = Field(min_length=5)
        age: int = Field(gt=0)
        email: str = Field(pattern=r".+@.+\..+")

    # Trigger multiple validation errors
    try:
        TestSchema(name="ab", age=-5, email="invalid")
    except ValidationError as e:
        # Test helper function
        error_info = unpack_validation_error(e)

        assert error_info["error_type"] == "ValidationError"
        assert len(error_info["errors"]) == 3
        assert len(error_info["fields"]) == 3

        # Check field paths
        field_paths = [f["path"] for f in error_info["fields"]]
        assert "name" in field_paths
        assert "age" in field_paths
        assert "email" in field_paths


def test_create_validation_error_event():
    """Test create_validation_error_event helper."""

    class TestSchema(BaseModel):
        required_field: str

    try:
        TestSchema(required_field=None)  # type: ignore
    except ValidationError as e:
        event = create_validation_error_event(
            job_id=uuid4(),
            error=e,
            raw_output='{"required_field": null}',
            attempt_number=2,
            max_attempts=3,
        )

        assert event.error_type == "ValidationError"
        assert event.error_count == 1
        assert event.failed_fields is not None
        assert len(event.failed_fields) == 1
        assert event.failed_fields[0]["path"] == "required_field"
        assert event.raw_output is not None
        assert event.attempt_number == 2


# ============================================================================
# Test Hook Cache Management
# ============================================================================


def test_hook_cache_building():
    """Test that hook cache is built correctly."""
    plugin = MockPlugin()
    manager = PluginManager()
    manager._plugins = [plugin]

    cache = manager._build_hook_cache()

    # Should have all hooks that plugin implements
    expected_hooks = [
        "on_job_submitted",
        "on_job_started",
        "on_job_completed",
        "on_job_failed",
        "on_synthesis_config",
        "on_synthesis_chunk",
        "on_validation_error",
        "on_validation_success",
    ]

    for hook in expected_hooks:
        assert hook in cache
        assert len(cache[hook]) == 1
        assert cache[hook][0][0] == plugin


def test_hook_cache_lazy_rebuild():
    """Test that cache is rebuilt when invalidated."""
    manager = PluginManager()
    manager._plugins = [MockPlugin()]

    # Build initial cache
    cache1 = manager._get_hook_cache()
    version1 = manager._cache_version

    # Register new plugin
    manager.register_plugin(MockPlugin(name="plugin2"))

    # Cache should be invalidated
    assert manager._hook_cache is None
    assert manager._cache_version == version1 + 1

    # Should rebuild on next access
    cache2 = manager._get_hook_cache()
    assert cache2 is not None
    assert len(cache2["on_job_submitted"]) == 2


def test_hook_cache_protocol_check():
    """Test that only JobLifecycleListener plugins are cached."""

    class NonListenerPlugin:
        name = "non_listener"
        version = "1.0"

        def initialize(self, config): pass
        def shutdown(self): pass

    manager = PluginManager()
    manager._plugins = [
        MockPlugin(),
        NonListenerPlugin(),
    ]

    cache = manager._build_hook_cache()

    # Only MockPlugin should be in cache (implements JobLifecycleListener)
    assert "on_job_submitted" in cache
    assert len(cache["on_job_submitted"]) == 1
    assert isinstance(cache["on_job_submitted"][0][0], MockPlugin)


# ============================================================================
# Test Feature Flags
# ============================================================================


@pytest.mark.asyncio
async def test_feature_flag_disabled():
    """Test that hooks are skipped when disabled."""

    class MockConfig:
        class Plugins:
            class LifecycleHooks:
                enabled = False

            lifecycle_hooks = LifecycleHooks()

        plugins = Plugins()

    plugin = MockPlugin()
    manager = PluginManager()
    manager._plugins = [plugin]
    manager._config = MockConfig()
    manager._build_hook_cache()

    event = JobSubmittedEvent(
        job_id=uuid4(),
        chat_id="test",
        query="test",
        user_id="user",
        research_config={},
        timestamp=datetime.now(timezone.utc),
    )

    stats = await manager.emit_hook("on_job_submitted", event)

    # Hook should be skipped
    assert stats.get("skipped") is True
    assert stats.get("reason") == "feature_disabled"
    assert plugin.call_count == 0


@pytest.mark.asyncio
async def test_feature_flag_allowlist():
    """Test that only allowlisted hooks are called."""

    class MockConfig:
        class Plugins:
            class LifecycleHooks:
                enabled = True
                allowlist = {"on_job_submitted"}
                denylist = set()

            lifecycle_hooks = LifecycleHooks()

        plugins = Plugins()

    plugin = MockPlugin()
    manager = PluginManager()
    manager._plugins = [plugin]
    manager._config = MockConfig()
    manager._build_hook_cache()

    # Allowed hook
    event1 = JobSubmittedEvent(
        job_id=uuid4(),
        chat_id="test",
        query="test",
        user_id="user",
        research_config={},
        timestamp=datetime.now(timezone.utc),
    )
    stats1 = await manager.emit_hook("on_job_submitted", event1)
    assert stats1.get("succeeded") == 1

    # Not in allowlist
    event2 = JobStartedEvent(
        job_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
    )
    stats2 = await manager.emit_hook("on_job_started", event2)
    assert stats2.get("skipped") is True
    assert stats2.get("reason") == "not_in_allowlist"

    assert plugin.call_count == 1  # Only first hook called


@pytest.mark.asyncio
async def test_feature_flag_denylist():
    """Test that denylisted hooks are blocked."""

    class MockConfig:
        class Plugins:
            class LifecycleHooks:
                enabled = True
                allowlist = set()
                denylist = {"on_job_submitted"}

            lifecycle_hooks = LifecycleHooks()

        plugins = Plugins()

    plugin = MockPlugin()
    manager = PluginManager()
    manager._plugins = [plugin]
    manager._config = MockConfig()
    manager._build_hook_cache()

    # Denylisted hook
    event = JobSubmittedEvent(
        job_id=uuid4(),
        chat_id="test",
        query="test",
        user_id="user",
        research_config={},
        timestamp=datetime.now(timezone.utc),
    )
    stats = await manager.emit_hook("on_job_submitted", event)

    assert stats.get("skipped") is True
    assert stats.get("reason") == "in_denylist"
    assert plugin.call_count == 0


# ============================================================================
# Test EventEmitter Helper
# ============================================================================


@pytest.mark.asyncio
async def test_event_emitter_job_submitted():
    """Test EventEmitter reduces boilerplate."""
    plugin = MockPlugin()
    manager = PluginManager()
    manager._plugins = [plugin]
    manager._build_hook_cache()

    emitter = EventEmitter(manager)

    # One-line emission
    await emitter.job_submitted(
        job_id=uuid4(),
        chat_id="test-chat",
        query="test query",
        user_id="test-user",
        query_mode="deep_research",
    )

    assert plugin.call_count == 1
    assert plugin.events_received[0][0] == "on_job_submitted"


@pytest.mark.asyncio
async def test_event_emitter_validation_error():
    """Test EventEmitter unpacks validation errors."""

    class TestSchema(BaseModel):
        name: str = Field(min_length=5)

    plugin = MockPlugin()
    manager = PluginManager()
    manager._plugins = [plugin]
    manager._build_hook_cache()

    emitter = EventEmitter(manager)

    try:
        TestSchema(name="ab")
    except ValidationError as e:
        await emitter.validation_error(
            job_id=uuid4(),
            error=e,
            raw_output='{"name": "ab"}',
        )

    assert plugin.call_count == 1
    event = plugin.events_received[0][1]
    assert isinstance(event, ValidationErrorEvent)
    assert event.error_count == 1
    assert len(event.failed_fields) == 1


@pytest.mark.asyncio
async def test_event_emitter_error_recovery():
    """Test EventEmitter recovers from event creation errors."""
    plugin = MockPlugin()
    manager = PluginManager()
    manager._plugins = [plugin]
    manager._build_hook_cache()

    emitter = EventEmitter(manager)

    # Deliberately pass invalid data that would crash event creation
    # EventEmitter should catch this and use fallback
    try:
        # This would normally crash, but _safe_emit catches it
        await emitter._safe_emit(
            "on_job_submitted",
            lambda: JobSubmittedEvent(
                job_id="invalid",  # Not a UUID - will crash
                chat_id="test",
                query="test",
                user_id="user",
                research_config={},
                timestamp=datetime.now(timezone.utc),
            ),
            fallback_data={"job_id": "test"},
        )
    except Exception:
        # If we get here, error recovery didn't work
        pytest.fail("EventEmitter should catch and recover from errors")


# ============================================================================
# Test Hook Metrics
# ============================================================================


@pytest.mark.asyncio
async def test_hook_metrics_tracking():
    """Test that hook metrics are tracked correctly."""
    plugin = MockPlugin()
    manager = PluginManager()
    manager._plugins = [plugin]
    manager._build_hook_cache()

    # Call hooks multiple times
    for _ in range(3):
        await manager.emit_hook(
            "on_job_submitted",
            JobSubmittedEvent(
                job_id=uuid4(),
                chat_id="test",
                query="test",
                user_id="user",
                research_config={},
                timestamp=datetime.now(timezone.utc),
            ),
        )

    metrics = manager.get_hook_metrics()
    hook_metrics = metrics["on_job_submitted"]

    assert hook_metrics["total_calls"] == 3
    assert hook_metrics["success_rate"] == 1.0
    assert hook_metrics["total_failures"] == 0
    assert hook_metrics["total_timeouts"] == 0


@pytest.mark.asyncio
async def test_hook_metrics_with_failures():
    """Test metrics track failures correctly."""
    plugin1 = MockPlugin(name="plugin1")
    plugin1._should_error = True
    plugin2 = MockPlugin(name="plugin2")  # Works fine

    manager = PluginManager()
    manager._plugins = [plugin1, plugin2]
    manager._build_hook_cache()

    # Use on_job_submitted which checks _should_error
    await manager.emit_hook(
        "on_job_submitted",
        JobSubmittedEvent(
            job_id=uuid4(),
            chat_id="test",
            query="test",
            user_id="user",
            research_config={},
            timestamp=datetime.now(timezone.utc),
        ),
    )

    metrics = manager.get_hook_metrics()
    hook_metrics = metrics["on_job_submitted"]

    # 2 total calls (2 plugins), 1 success, 1 failure
    assert hook_metrics["total_calls"] == 2
    assert hook_metrics["success_rate"] == 0.5
    assert hook_metrics["total_failures"] == 1


# ============================================================================
# Test Event Immutability
# ============================================================================


def test_event_immutability():
    """Test that events are frozen and cannot be modified."""
    event = JobSubmittedEvent(
        job_id=uuid4(),
        chat_id="test",
        query="test",
        user_id="user",
        research_config={},
        timestamp=datetime.now(timezone.utc),
    )

    # Should not be able to modify frozen dataclass
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        event.query = "modified"  # type: ignore


# ============================================================================
# Test Protocol Compliance
# ============================================================================


def test_protocol_compliance():
    """Test that MockPlugin has all required lifecycle methods."""
    plugin = MockPlugin()
    # Check duck typing - plugin has the methods
    assert hasattr(plugin, "on_job_submitted")
    assert hasattr(plugin, "on_job_started")
    assert hasattr(plugin, "on_validation_error")
    assert callable(plugin.on_job_submitted)


def test_partial_protocol_implementation():
    """Test that plugins can implement subset of hooks."""

    class PartialPlugin:
        name = "partial"
        version = "1.0"

        def initialize(self, config): pass
        def shutdown(self): pass

        # Only implement one hook
        def on_job_submitted(self, event): pass

    plugin = PartialPlugin()
    manager = PluginManager()
    manager._plugins = [plugin]

    cache = manager._build_hook_cache()

    # Should have only the implemented hook
    assert "on_job_submitted" in cache
    assert "on_job_started" not in cache
