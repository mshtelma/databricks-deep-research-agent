"""
Plugin Manager
==============

Manages plugin discovery, initialization, and lifecycle.

Now includes lifecycle callback emission with:
- Timeout protection (prevents infinite hangs)
- Thread-safe lazy hook cache
- Throttling for high-frequency hooks
- Hook metrics and monitoring
- Feature flags for gradual rollout
"""

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import TYPE_CHECKING, Any

from deep_research.agent.tools.base import ResearchContext, ResearchTool

if TYPE_CHECKING:
    from deep_research.core.app_config import AppConfig
from deep_research.agent.pipeline.protocols import (
    CustomPhase,
    PhaseProvider,
    PipelineCustomization,
    PipelineCustomizer,
)
from deep_research.agent.tools.registry import ToolRegistry
from deep_research.plugins.base import (
    PromptProvider,
    ResearchPlugin,
    ToolProvider,
)
from deep_research.plugins.discovery import discover_plugins

logger = logging.getLogger(__name__)


class PluginManagerError(Exception):
    """Exception raised for plugin manager errors."""

    pass


@dataclass
class HookThrottleConfig:
    """Throttling configuration for high-frequency hooks."""

    hook_name: str
    min_interval_seconds: float  # Minimum time between calls
    batch_size: int = 1  # Batch multiple events together


@dataclass
class HookMetrics:
    """Metrics for monitoring hook health."""

    hook_name: str
    total_calls: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_timeouts: int = 0
    total_duration_ms: float = 0
    last_call_time: datetime | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.total_successes / self.total_calls

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration."""
        if self.total_successes == 0:
            return 0.0
        return self.total_duration_ms / self.total_successes


@dataclass
class PluginManager:
    """
    Manages plugin discovery, initialization, and lifecycle.

    Responsibilities:
    - Discover plugins via entry points
    - Initialize plugins with configuration
    - Collect tools from ToolProvider plugins
    - Collect prompt overrides from PromptProvider plugins
    - Emit lifecycle callbacks to JobLifecycleListener plugins
    - Graceful shutdown of all plugins

    Lifecycle Hooks:
    - Timeout protection (30s default)
    - Thread-safe lazy cache
    - Throttling for high-frequency hooks
    - Hook metrics and monitoring
    - Feature flags for gradual rollout
    """

    _plugins: list[ResearchPlugin] = field(default_factory=list)
    _tool_registry: ToolRegistry = field(default_factory=ToolRegistry)
    _initialized: bool = False
    # Phase and customization registries
    _phase_registry: dict[str, CustomPhase] = field(default_factory=dict)
    _pipeline_customizations: dict[str, PipelineCustomization] = field(default_factory=dict)

    # Lifecycle hook state
    _hook_cache: dict[str, list[tuple[Any, Callable]]] | None = None
    _cache_lock: RLock = field(default_factory=RLock)
    _cache_version: int = 0
    _hook_timeout: int = 30  # seconds
    _executor: ThreadPoolExecutor | None = None
    _max_sync_workers: int = 4

    # Throttling state
    _throttle_configs: dict[str, HookThrottleConfig] = field(default_factory=dict)
    _last_hook_times: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    _batched_events: dict[tuple[str, str], list] = field(default_factory=lambda: defaultdict(list))

    # Metrics
    _hook_metrics: dict[str, HookMetrics] = field(default_factory=lambda: defaultdict(lambda: HookMetrics(hook_name="unknown")))

    # Config for feature flags
    _config: Any = None

    def discover_and_load(self, app_config: "AppConfig") -> None:
        """
        Discover and initialize all plugins.

        Args:
            app_config: Application configuration (AppConfig instance)

        Note:
            Plugin initialization failures are logged but don't prevent
            other plugins from loading or the app from starting.

            Plugins can be disabled via app_config.plugins configuration:
                plugins:
                  my_plugin:
                    enabled: false
        """
        if self._initialized:
            logger.warning("PluginManager already initialized, skipping")
            return

        # Store config for feature flags
        self._config = app_config

        # Get lifecycle hooks config
        hooks_config = getattr(app_config, "plugins", None)
        if hooks_config:
            hooks_config = getattr(hooks_config, "lifecycle_hooks", None)

        # Initialize thread pool for sync hooks
        max_workers = self._max_sync_workers
        timeout = self._hook_timeout
        if hooks_config:
            max_workers = getattr(hooks_config, "max_sync_workers", max_workers)
            timeout = getattr(hooks_config, "timeout_seconds", timeout)

        self._max_sync_workers = max_workers
        self._hook_timeout = timeout
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="plugin_hook_",
        )

        # Configure throttling for high-frequency hooks
        self._throttle_configs["on_synthesis_chunk"] = HookThrottleConfig(
            hook_name="on_synthesis_chunk",
            min_interval_seconds=2.0,  # Max once per 2 seconds
            batch_size=5,  # Batch up to 5 chunks
        )

        logger.info(
            "Lifecycle hooks configured: timeout=%ds, workers=%d",
            timeout,
            max_workers,
        )

        # Discover plugin classes
        plugin_classes = discover_plugins()
        logger.info("Discovered %d plugin(s)", len(plugin_classes))

        # Get plugin configuration if available
        plugins_config = getattr(app_config, "plugins", None)

        # Instantiate and initialize each plugin
        for plugin_cls in plugin_classes:
            try:
                # Create instance first to get its name
                plugin = plugin_cls()

                # Check if plugin is disabled via configuration
                if (
                    plugins_config is not None
                    and not plugins_config.is_enabled(plugin.name)
                ):
                    logger.info(
                        "Skipping disabled plugin: %s",
                        plugin.name,
                    )
                    continue

                plugin.initialize(app_config)
                self._plugins.append(plugin)
                logger.info(
                    "Loaded plugin: %s v%s",
                    plugin.name,
                    plugin.version,
                )
            except AttributeError as e:
                # Plugin doesn't implement required protocol
                logger.warning(
                    "Plugin class %s doesn't implement ResearchPlugin: %s",
                    plugin_cls.__name__,
                    e,
                )
            except Exception as e:
                # Initialization failed - log and continue
                logger.warning(
                    "Failed to initialize plugin %s: %s",
                    getattr(plugin_cls, "__name__", str(plugin_cls)),
                    e,
                )

        # Register tools from all ToolProvider plugins
        self._register_tools()

        # Register phases and customizations from PhaseProvider/PipelineCustomizer plugins
        self._register_phases()
        self._register_customizations()

        # Build hook cache for lifecycle callbacks
        self._build_hook_cache()

        self._initialized = True
        logger.info(
            "PluginManager initialized: %d plugins, %d tools, %d hooks",
            len(self._plugins),
            len(self._tool_registry),
            len(self._hook_cache) if self._hook_cache else 0,
        )

    def _register_tools(self) -> None:
        """Collect and register tools from all ToolProvider plugins."""
        # Create a minimal context for tool collection
        from uuid import uuid4

        context = ResearchContext(
            chat_id=uuid4(),
            user_id="system",
            research_type="medium",
        )

        for plugin in self._plugins:
            if isinstance(plugin, ToolProvider):
                try:
                    tools = plugin.get_tools(context)
                    for tool in tools:
                        try:
                            self._tool_registry.register(tool)
                        except Exception:
                            # Conflict - register with prefix
                            self._tool_registry.register_with_prefix(
                                tool,
                                prefix=plugin.name,
                            )
                    logger.debug(
                        "Registered %d tools from plugin '%s'",
                        len(tools),
                        plugin.name,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to get tools from plugin '%s': %s",
                        plugin.name,
                        e,
                    )

    def _register_phases(self) -> None:
        """Register phases from all PhaseProvider plugins."""
        for plugin in self._plugins:
            if isinstance(plugin, PhaseProvider):
                try:
                    phases = plugin.get_custom_phases()
                    for phase in phases:
                        if phase.name in self._phase_registry:
                            logger.warning(
                                "Phase '%s' already registered, skipping duplicate from '%s'",
                                phase.name,
                                plugin.name,
                            )
                            continue
                        self._phase_registry[phase.name] = phase
                        logger.info(
                            "Registered phase '%s' from plugin '%s'",
                            phase.name,
                            plugin.name,
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to register phases from plugin '%s': %s",
                        plugin.name,
                        e,
                    )

    def _register_customizations(self) -> None:
        """Register pipeline customizations from all PipelineCustomizer plugins."""
        for plugin in self._plugins:
            if isinstance(plugin, PipelineCustomizer):
                try:
                    customization = plugin.get_pipeline_customization()
                    if customization:
                        self._pipeline_customizations[plugin.name] = customization
                        logger.info(
                            "Registered pipeline customization from plugin '%s': "
                            "disabled_agents=%s, phase_insertions=%d",
                            plugin.name,
                            customization.disabled_agents,
                            len(customization.phase_insertions),
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to register customization from plugin '%s': %s",
                        plugin.name,
                        e,
                    )

    def get_tools(self, context: ResearchContext) -> list[ResearchTool]:  # noqa: ARG002
        """
        Get all registered tools.

        Args:
            context: Research context (for future context-aware filtering)

        Returns:
            List of all registered tools
        """
        return list(self._tool_registry)

    def get_tool_registry(self) -> ToolRegistry:
        """Get the underlying tool registry."""
        return self._tool_registry

    def get_prompt_overrides(self, context: ResearchContext) -> dict[str, str]:
        """
        Collect prompt overrides from all PromptProvider plugins.

        Args:
            context: Research context for prompt customization

        Returns:
            Dict mapping agent names to prompt additions.
            Later plugins override earlier ones for the same key.
        """
        overrides: dict[str, str] = {}

        for plugin in self._plugins:
            if isinstance(plugin, PromptProvider):
                try:
                    plugin_overrides = plugin.get_prompt_overrides(context)
                    overrides.update(plugin_overrides)
                except Exception as e:
                    logger.warning(
                        "Failed to get prompt overrides from plugin '%s': %s",
                        plugin.name,
                        e,
                    )

        return overrides

    def get_plugins(self) -> list[ResearchPlugin]:
        """Get all loaded plugins."""
        return list(self._plugins)

    def get_plugin(self, name: str) -> ResearchPlugin | None:
        """
        Get a plugin by name.

        Args:
            name: Plugin name to look up

        Returns:
            The plugin, or None if not found
        """
        for plugin in self._plugins:
            if plugin.name == name:
                return plugin
        return None

    def get_phase(self, name: str) -> CustomPhase | None:
        """Get a registered phase by name."""
        return self._phase_registry.get(name)

    def get_all_phases(self) -> dict[str, CustomPhase]:
        """Get all registered phases."""
        return dict(self._phase_registry)

    def get_pipeline_customization(self) -> PipelineCustomization | None:
        """Get merged pipeline customization from all plugins.

        Returns None if no plugins provide customization.
        Later plugins override earlier ones for conflicting settings.
        """
        if not self._pipeline_customizations:
            return None

        # Merge all customizations
        merged_disabled: set[str] = set()
        merged_insertions: list = []
        merged_overrides: dict = {}

        for customization in self._pipeline_customizations.values():
            merged_disabled.update(customization.disabled_agents)
            merged_insertions.extend(customization.phase_insertions)
            merged_overrides.update(customization.agent_overrides)

        return PipelineCustomization(
            disabled_agents=merged_disabled,
            phase_insertions=merged_insertions,
            agent_overrides=merged_overrides,
        )

    def has_custom_phase_mode(self) -> bool:
        """Check if any plugin has disabled planner and defined phases."""
        customization = self.get_pipeline_customization()
        # DIAGNOSTIC: Log has_custom_phase_mode check details
        logger.info(
            f"HAS_CUSTOM_PHASE_MODE_CHECK has_customization={customization is not None} disabled_agents={list(customization.disabled_agents) if customization else []} num_insertions={len(customization.phase_insertions) if customization else 0} num_phases={len(self._phase_registry)}"
        )
        if not customization:
            return False
        return (
            "planner" in customization.disabled_agents
            and bool(customization.phase_insertions)
            and bool(self._phase_registry)
        )

    def shutdown(self) -> None:
        """
        Shutdown all plugins.

        Calls shutdown() on each plugin, logging any errors.
        Clears internal state after shutdown.
        """
        logger.info("Shutting down %d plugin(s)", len(self._plugins))

        for plugin in self._plugins:
            try:
                plugin.shutdown()
                logger.debug("Shutdown plugin: %s", plugin.name)
            except Exception as e:
                logger.warning(
                    "Error shutting down plugin '%s': %s",
                    plugin.name,
                    e,
                )

        # Shutdown thread pool
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

        self._plugins.clear()
        self._tool_registry.clear()
        self._phase_registry.clear()
        self._pipeline_customizations.clear()
        self._hook_cache = None
        self._initialized = False
        logger.info("PluginManager shutdown complete")

    @property
    def initialized(self) -> bool:
        """Check if the manager has been initialized."""
        return self._initialized

    def __len__(self) -> int:
        """Return number of loaded plugins."""
        return len(self._plugins)

    # ========================================================================
    # Lifecycle Hook Emission (NEW)
    # ========================================================================

    def register_plugin(self, plugin: ResearchPlugin) -> None:
        """
        Register a plugin dynamically and invalidate hook cache.

        Thread-safe method for adding plugins after initialization.

        Args:
            plugin: Plugin instance to register
        """
        with self._cache_lock:
            self._plugins.append(plugin)
            self._hook_cache = None  # Invalidate cache
            self._cache_version += 1
            logger.info(
                "PLUGIN_REGISTERED plugin=%s cache_invalidated=True version=%d",
                getattr(plugin, "name", plugin.__class__.__name__),
                self._cache_version,
            )

    def _get_hook_cache(self) -> dict[str, list[tuple[Any, Callable]]]:
        """
        Get hook cache, rebuilding if needed (lazy + thread-safe).

        Returns:
            Dict mapping hook names to list of (plugin, hook_fn) tuples
        """
        with self._cache_lock:
            if self._hook_cache is None:
                self._hook_cache = self._build_hook_cache()
                logger.debug(
                    "HOOK_CACHE_REBUILT hooks=%s version=%d",
                    list(self._hook_cache.keys()),
                    self._cache_version,
                )
            return self._hook_cache

    def _build_hook_cache(self) -> dict[str, list[tuple[Any, Callable]]]:
        """
        Build hook cache from current plugins.

        Scans all plugins for JobLifecycleListener protocol methods.

        Returns:
            Dict mapping hook names to list of (plugin, callable) tuples
        """
        cache: dict[str, list[tuple[Any, Callable]]] = {}

        # All possible hooks from JobLifecycleListener protocol
        all_hooks = [
            # Job lifecycle hooks
            "on_job_submitted",
            "on_job_started",
            "on_job_completed",
            "on_job_failed",
            # Synthesis lifecycle hooks
            "on_synthesis_config",
            "on_synthesis_started",
            "on_synthesis_chunk",
            "on_synthesis_completed",
            # Validation hooks
            "on_validation_error",
            "on_validation_success",
            # Generic hooks
            "on_stream_event",
            # Enterprise data source hooks (007-enterprise-data-sources, T064)
            "on_data_source_query",
            "on_template_applied",
            "on_custom_agent_selected",
            "on_data_landscape_built",
        ]

        for hook_name in all_hooks:
            hooks = []
            for plugin in self._plugins:
                # Check if plugin has this hook method (duck typing)
                # Don't rely on isinstance() check as Protocol matching can be unreliable
                if hasattr(plugin, hook_name):
                    hook_fn = getattr(plugin, hook_name)
                    if callable(hook_fn):
                        hooks.append((plugin, hook_fn))

            if hooks:
                cache[hook_name] = hooks
                plugin_names = [
                    getattr(p, "name", p.__class__.__name__)
                    for p, _ in hooks
                ]
                logger.info(
                    "Plugin hook registered: %s -> %s",
                    hook_name,
                    plugin_names,
                )

        return cache

    async def emit_hook(
        self,
        hook_name: str,
        event: Any,
        timeout_override: int | None = None,
    ) -> dict[str, Any]:
        """
        Emit lifecycle hook with timeout, throttling, and error isolation.

        CRITICAL: Hooks can be sync or async, but all have:
        - Timeout protection (prevents infinite hangs)
        - Error isolation (plugin failures don't crash framework)
        - Metrics tracking (monitoring)
        - Feature flags (gradual rollout)

        Args:
            hook_name: Name of hook method (e.g., "on_job_submitted")
            event: Event object to pass to hook
            timeout_override: Override default timeout

        Returns:
            Dict with execution statistics for monitoring

        Example:
            stats = await plugin_manager.emit_hook(
                "on_job_submitted",
                JobSubmittedEvent(job_id="123", ...)
            )
            # stats = {"succeeded": 2, "failed": 0, "timed_out": 0}
        """
        # Check feature flags
        if self._config:
            hooks_config = getattr(self._config, "plugins", None)
            if hooks_config:
                hooks_config = getattr(hooks_config, "lifecycle_hooks", None)
                if hooks_config:
                    if not getattr(hooks_config, "enabled", True):
                        return {"skipped": True, "reason": "feature_disabled"}

                    allowlist = getattr(hooks_config, "allowlist", set())
                    if allowlist and hook_name not in allowlist:
                        return {"skipped": True, "reason": "not_in_allowlist"}

                    denylist = getattr(hooks_config, "denylist", set())
                    if hook_name in denylist:
                        return {"skipped": True, "reason": "in_denylist"}

        # Check if this hook should be throttled
        throttle = self._throttle_configs.get(hook_name)
        if throttle:
            return await self._emit_throttled(hook_name, event, throttle)
        else:
            return await self._emit_immediate(hook_name, event, timeout_override)

    async def _emit_immediate(
        self,
        hook_name: str,
        event: Any,
        timeout_override: int | None = None,
    ) -> dict[str, Any]:
        """Emit hook immediately without throttling."""
        timeout = timeout_override or self._hook_timeout
        cache = self._get_hook_cache()  # Thread-safe lazy access
        hooks = cache.get(hook_name, [])

        stats = {
            "hook_name": hook_name,
            "total_plugins": len(hooks),
            "succeeded": 0,
            "failed": 0,
            "timed_out": 0,
            "errors": [],
        }

        for plugin, hook_fn in hooks:
            plugin_name = getattr(plugin, "name", plugin.__class__.__name__)

            start = time.time()
            try:
                if asyncio.iscoroutinefunction(hook_fn):
                    # Async hook with timeout
                    await asyncio.wait_for(hook_fn(event), timeout=timeout)
                else:
                    # Sync hook in bounded executor with timeout
                    loop = asyncio.get_running_loop()
                    await asyncio.wait_for(
                        loop.run_in_executor(self._executor, hook_fn, event),
                        timeout=timeout,
                    )

                duration_ms = (time.time() - start) * 1000
                stats["succeeded"] += 1

                # Update metrics
                metrics = self._hook_metrics[hook_name]
                metrics.total_calls += 1
                metrics.total_successes += 1
                metrics.total_duration_ms += duration_ms
                metrics.last_call_time = datetime.now()

            except TimeoutError:
                stats["timed_out"] += 1
                stats["errors"].append({
                    "plugin": plugin_name,
                    "error": "timeout",
                    "timeout_seconds": timeout,
                })

                # Update metrics
                metrics = self._hook_metrics[hook_name]
                metrics.total_calls += 1
                metrics.total_timeouts += 1

                logger.warning(
                    "PLUGIN_HOOK_TIMEOUT plugin=%s hook=%s timeout=%d",
                    plugin_name,
                    hook_name,
                    timeout,
                )

            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append({
                    "plugin": plugin_name,
                    "error": str(e)[:200],
                    "error_type": type(e).__name__,
                })

                # Update metrics
                metrics = self._hook_metrics[hook_name]
                metrics.total_calls += 1
                metrics.total_failures += 1

                logger.exception(
                    "PLUGIN_HOOK_ERROR plugin=%s hook=%s",
                    plugin_name,
                    hook_name,
                )

        # Log performance warnings
        if stats["failed"] > 0 or stats["timed_out"] > 0:
            logger.warning(
                "PLUGIN_HOOK_STATS hook=%s succeeded=%d failed=%d timed_out=%d",
                hook_name,
                stats["succeeded"],
                stats["failed"],
                stats["timed_out"],
            )

        return stats

    async def _emit_throttled(
        self,
        hook_name: str,
        event: Any,
        config: HookThrottleConfig,
    ) -> dict[str, Any]:
        """Emit with rate limiting and batching."""
        job_id = getattr(event, "job_id", None)
        key = (hook_name, str(job_id))

        now = time.time()
        last_time = self._last_hook_times[key]
        elapsed = now - last_time

        # Add to batch
        self._batched_events[key].append(event)

        # Check if we should emit
        should_emit = (
            elapsed >= config.min_interval_seconds
            or len(self._batched_events[key]) >= config.batch_size
        )

        if should_emit:
            batched = self._batched_events[key]
            self._batched_events[key] = []
            self._last_hook_times[key] = now

            # Create batch event
            batch_event = self._create_batch_event(hook_name, batched)
            return await self._emit_immediate(hook_name, batch_event)

        return {"throttled": True, "batch_size": len(self._batched_events[key])}

    def _create_batch_event(self, hook_name: str, events: list) -> Any:
        """Combine multiple events into batch."""
        if hook_name == "on_synthesis_chunk":
            # Combine chunks
            combined_content = "".join(
                getattr(e, "content_chunk", "") for e in events
            )
            last_event = events[-1]
            # Create new event with combined content
            return type(last_event)(
                job_id=last_event.job_id,
                content_chunk=combined_content,
                chunk_length=len(combined_content),
                cumulative_length=last_event.cumulative_length,
                metadata={"is_batch": True, "batch_size": len(events)},
                timestamp=last_event.timestamp,
            )
        return events[-1]  # Fallback: just use last event

    def get_hook_metrics(self) -> dict[str, dict]:
        """
        Get current hook performance metrics.

        Returns:
            Dict mapping hook names to metric dicts with:
            - total_calls, success_rate, avg_duration_ms, etc.
        """
        return {
            name: {
                "total_calls": m.total_calls,
                "success_rate": m.success_rate,
                "avg_duration_ms": m.avg_duration_ms,
                "total_timeouts": m.total_timeouts,
                "total_failures": m.total_failures,
            }
            for name, m in self._hook_metrics.items()
        }
