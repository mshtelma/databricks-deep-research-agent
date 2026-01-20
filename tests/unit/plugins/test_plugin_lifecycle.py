"""Integration tests for plugin lifecycle.

Tests the full plugin lifecycle from discovery to shutdown.
"""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.agent.tools.base import ResearchContext, ResearchTool, ToolDefinition, ToolResult
from deep_research.core.app_config import AppConfig, PluginConfig, PluginsConfig
from deep_research.plugins.base import PromptProvider, ResearchPlugin, ToolProvider
from deep_research.plugins.manager import PluginManager


# Full lifecycle test plugin
class LifecycleTestPlugin:
    """A plugin that tracks its lifecycle for testing."""

    def __init__(self) -> None:
        self._name = "lifecycle-test"
        self._version = "1.0.0"
        self._initialized = False
        self._shutdown_called = False
        self._config: dict[str, object] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def initialize(self, app_config: Any) -> None:
        self._initialized = True
        # Extract plugin-specific config
        if hasattr(app_config, "plugins"):
            plugin_config = app_config.plugins.get(self.name)
            if plugin_config:
                self._config = dict(plugin_config.settings)

    def shutdown(self) -> None:
        self._shutdown_called = True

    def get_tools(self, context: ResearchContext) -> list[ResearchTool]:
        return [LifecycleTestTool()]

    def get_prompt_overrides(self, context: ResearchContext) -> dict[str, str]:
        return {"researcher": "Lifecycle test instructions"}


@dataclass
class LifecycleTestTool:
    """Test tool for lifecycle tests."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="lifecycle-test-tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, arguments: dict[str, Any], context: ResearchContext) -> ToolResult:
        return ToolResult(content="Test result", success=True)

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        return []


def create_test_context() -> ResearchContext:
    """Create a test ResearchContext."""
    return ResearchContext(
        chat_id=uuid4(),
        user_id="test-user",
        research_type="medium",
    )


class TestPluginLifecycle:
    """Integration tests for the full plugin lifecycle."""

    def test_full_lifecycle(self) -> None:
        """Test complete plugin lifecycle: discovery -> init -> use -> shutdown."""
        # Create app config with plugin settings
        plugins_config = PluginsConfig(
            configs={
                "lifecycle-test": PluginConfig(
                    enabled=True,
                    settings={"custom_setting": "test_value"},
                ),
            }
        )

        # Mock the app_config to have plugins attribute
        app_config = MagicMock()
        app_config.plugins = plugins_config

        # Create manager and discover plugins
        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [LifecycleTestPlugin]
            manager.discover_and_load(app_config=app_config)

        # Verify plugin was loaded
        assert len(manager) == 1
        assert manager.initialized

        # Get the plugin instance
        plugin = manager.get_plugin("lifecycle-test")
        assert plugin is not None
        assert isinstance(plugin, LifecycleTestPlugin)
        assert plugin._initialized

        # Verify config was passed
        assert plugin._config == {"custom_setting": "test_value"}

        # Verify tools are registered
        context = create_test_context()
        tools = manager.get_tools(context)
        assert len(tools) == 1
        assert tools[0].definition.name == "lifecycle-test-tool"

        # Verify prompt overrides
        overrides = manager.get_prompt_overrides(context)
        assert "researcher" in overrides

        # Shutdown
        manager.shutdown()

        assert plugin._shutdown_called
        assert not manager.initialized
        assert len(manager) == 0

    def test_multiple_plugins_lifecycle(self) -> None:
        """Test lifecycle with multiple plugins - one enabled, one disabled."""

        class EnabledPlugin:
            @property
            def name(self) -> str:
                return "enabled-plugin"

            @property
            def version(self) -> str:
                return "1.0.0"

            def initialize(self, app_config: Any) -> None:
                pass

            def shutdown(self) -> None:
                pass

        class DisabledPlugin:
            @property
            def name(self) -> str:
                return "disabled-plugin"

            @property
            def version(self) -> str:
                return "1.0.0"

            def initialize(self, app_config: Any) -> None:
                pass

            def shutdown(self) -> None:
                pass

        plugins_config = PluginsConfig(
            configs={
                "enabled-plugin": PluginConfig(enabled=True),
                "disabled-plugin": PluginConfig(enabled=False),
            }
        )
        app_config = MagicMock()
        app_config.plugins = plugins_config

        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [EnabledPlugin, DisabledPlugin]
            manager.discover_and_load(app_config=app_config)

        # Only enabled plugin should be loaded
        assert len(manager) == 1
        assert manager.get_plugin("enabled-plugin") is not None
        assert manager.get_plugin("disabled-plugin") is None

    def test_plugin_initialization_failure_isolation(self) -> None:
        """Test that one plugin's failure doesn't affect others."""

        class FailingPlugin:
            @property
            def name(self) -> str:
                return "failing"

            @property
            def version(self) -> str:
                return "1.0.0"

            def initialize(self, app_config: Any) -> None:
                raise RuntimeError("Initialization failed!")

            def shutdown(self) -> None:
                pass

        class WorkingPlugin:
            def __init__(self) -> None:
                self._initialized = False

            @property
            def name(self) -> str:
                return "working"

            @property
            def version(self) -> str:
                return "1.0.0"

            def initialize(self, app_config: Any) -> None:
                self._initialized = True

            def shutdown(self) -> None:
                pass

        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [FailingPlugin, WorkingPlugin]
            manager.discover_and_load(app_config=MagicMock())

        # Only working plugin should be loaded
        assert len(manager) == 1
        plugin = manager.get_plugin("working")
        assert plugin is not None
        assert plugin._initialized

    def test_tool_registry_persists_through_lifecycle(self) -> None:
        """Test that tool registry maintains tools correctly."""
        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [LifecycleTestPlugin]
            manager.discover_and_load(app_config=MagicMock())

        # Get registry
        registry = manager.get_tool_registry()
        assert len(registry) == 1

        # Get same registry again - should be same instance
        registry2 = manager.get_tool_registry()
        assert registry is registry2

        # Verify tool is accessible
        tool = registry.get("lifecycle-test-tool")
        assert tool is not None

        # After shutdown, registry should be cleared
        manager.shutdown()
        assert len(registry) == 0

    def test_prompt_overrides_from_multiple_plugins(self) -> None:
        """Test that prompt overrides merge correctly from multiple plugins."""

        class Plugin1:
            @property
            def name(self) -> str:
                return "plugin1"

            @property
            def version(self) -> str:
                return "1.0.0"

            def initialize(self, app_config: Any) -> None:
                pass

            def shutdown(self) -> None:
                pass

            def get_prompt_overrides(self, context: ResearchContext) -> dict[str, str]:
                return {
                    "researcher": "Plugin1 researcher",
                    "planner": "Plugin1 planner",
                }

        class Plugin2:
            @property
            def name(self) -> str:
                return "plugin2"

            @property
            def version(self) -> str:
                return "1.0.0"

            def initialize(self, app_config: Any) -> None:
                pass

            def shutdown(self) -> None:
                pass

            def get_prompt_overrides(self, context: ResearchContext) -> dict[str, str]:
                return {
                    "researcher": "Plugin2 researcher",  # Overrides Plugin1
                    "synthesizer": "Plugin2 synthesizer",
                }

        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [Plugin1, Plugin2]
            manager.discover_and_load(app_config=MagicMock())

        context = create_test_context()
        overrides = manager.get_prompt_overrides(context)

        # Plugin2 overrides Plugin1 for researcher
        assert overrides["researcher"] == "Plugin2 researcher"
        # Plugin1's planner preserved
        assert overrides["planner"] == "Plugin1 planner"
        # Plugin2's synthesizer added
        assert overrides["synthesizer"] == "Plugin2 synthesizer"
