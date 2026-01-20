"""Unit tests for PluginManager."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from deep_research.agent.tools.base import ResearchContext, ResearchTool, ToolDefinition, ToolResult
from deep_research.plugins.base import PromptProvider, ResearchPlugin, ToolProvider
from deep_research.plugins.manager import PluginManager, PluginManagerError


# Test fixtures - concrete implementations of protocols


@dataclass
class MockPlugin:
    """Simple plugin that only implements ResearchPlugin."""

    _name: str = "mock-plugin"
    _version: str = "1.0.0"
    _initialized: bool = False
    _shutdown_called: bool = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def initialize(self, app_config: Any) -> None:
        self._initialized = True

    def shutdown(self) -> None:
        self._shutdown_called = True


@dataclass
class MockTool:
    """Simple tool implementation."""

    _name: str = "mock-tool"
    _description: str = "A mock tool for testing"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters={"type": "object", "properties": {}},
        )

    async def execute(self, arguments: dict[str, Any], context: ResearchContext) -> ToolResult:
        return ToolResult(content="Mock result", success=True)

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        return []


@dataclass
class MockToolProviderPlugin:
    """Plugin that provides tools."""

    _name: str = "tool-provider"
    _version: str = "1.0.0"
    _tools: list[ResearchTool] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def initialize(self, app_config: Any) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_tools(self, context: ResearchContext) -> list[ResearchTool]:
        if self._tools is not None:
            return self._tools
        return [MockTool()]


@dataclass
class MockPromptProviderPlugin:
    """Plugin that provides prompt overrides."""

    _name: str = "prompt-provider"
    _version: str = "1.0.0"
    _overrides: dict[str, str] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def initialize(self, app_config: Any) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def get_prompt_overrides(self, context: ResearchContext) -> dict[str, str]:
        if self._overrides is not None:
            return self._overrides
        return {"researcher": "Custom instructions for researcher"}


class MockFullPlugin:
    """Plugin implementing all protocols."""

    def __init__(self) -> None:
        self._initialized = False
        self._shutdown_called = False

    @property
    def name(self) -> str:
        return "full-plugin"

    @property
    def version(self) -> str:
        return "2.0.0"

    def initialize(self, app_config: Any) -> None:
        self._initialized = True

    def shutdown(self) -> None:
        self._shutdown_called = True

    def get_tools(self, context: ResearchContext) -> list[ResearchTool]:
        return [MockTool(_name="full-plugin-tool")]

    def get_prompt_overrides(self, context: ResearchContext) -> dict[str, str]:
        return {"synthesizer": "Full plugin synthesizer instructions"}


def create_test_context() -> ResearchContext:
    """Create a test ResearchContext."""
    return ResearchContext(
        chat_id=uuid4(),
        user_id="test-user",
        research_type="medium",
    )


class TestPluginManagerDiscovery:
    """Tests for plugin discovery and loading."""

    def test_discover_and_load_empty(self) -> None:
        """Should handle no plugins gracefully."""
        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = []
            manager.discover_and_load(app_config=MagicMock())

        assert manager.initialized
        assert len(manager) == 0

    def test_discover_and_load_single_plugin(self) -> None:
        """Should load a single plugin."""
        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [MockPlugin]
            manager.discover_and_load(app_config=MagicMock())

        assert manager.initialized
        assert len(manager) == 1

    def test_discover_and_load_multiple_plugins(self) -> None:
        """Should load multiple plugins."""
        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [MockPlugin, MockToolProviderPlugin, MockPromptProviderPlugin]
            manager.discover_and_load(app_config=MagicMock())

        assert len(manager) == 3

    def test_skips_reinitialization(self) -> None:
        """Should not reinitialize if already initialized."""
        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [MockPlugin]
            manager.discover_and_load(app_config=MagicMock())
            first_count = len(manager)

            # Try to initialize again
            mock_discover.return_value = [MockPlugin, MockToolProviderPlugin]
            manager.discover_and_load(app_config=MagicMock())

        assert len(manager) == first_count  # Should not change

    def test_handles_plugin_without_protocol(self) -> None:
        """Should skip plugins that don't implement ResearchPlugin."""

        class BadPlugin:
            pass

        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [BadPlugin]
            manager.discover_and_load(app_config=MagicMock())

        assert len(manager) == 0

    def test_handles_plugin_initialization_error(self) -> None:
        """Should skip plugins that fail to initialize."""

        class FailingPlugin:
            @property
            def name(self) -> str:
                return "failing"

            @property
            def version(self) -> str:
                return "1.0.0"

            def initialize(self, app_config: Any) -> None:
                raise RuntimeError("Failed to initialize")

            def shutdown(self) -> None:
                pass

        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [FailingPlugin]
            manager.discover_and_load(app_config=MagicMock())

        assert len(manager) == 0


class TestPluginManagerTools:
    """Tests for tool registration and retrieval."""

    def test_registers_tools_from_tool_provider(self) -> None:
        """Should register tools from ToolProvider plugins."""
        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [MockToolProviderPlugin]
            manager.discover_and_load(app_config=MagicMock())

        context = create_test_context()
        tools = manager.get_tools(context)
        assert len(tools) == 1
        assert tools[0].definition.name == "mock-tool"

    def test_get_tool_registry(self) -> None:
        """Should return the tool registry."""
        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [MockToolProviderPlugin]
            manager.discover_and_load(app_config=MagicMock())

        registry = manager.get_tool_registry()
        assert registry is not None
        assert len(registry) == 1

    def test_handles_tool_name_conflict_with_prefix(self) -> None:
        """Should prefix tool names on conflict."""
        # Create two plugins that provide tools with the same name
        plugin1 = MockToolProviderPlugin(_name="plugin1", _tools=[MockTool(_name="shared-tool")])
        plugin2 = MockToolProviderPlugin(_name="plugin2", _tools=[MockTool(_name="shared-tool")])

        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            # Return class types that will be instantiated
            mock_discover.return_value = []
            manager.discover_and_load(app_config=MagicMock())

        # Manually add the plugins to test conflict handling
        manager._plugins.append(plugin1)
        manager._plugins.append(plugin2)
        manager._register_tools()

        registry = manager.get_tool_registry()
        # Should have both tools, one with prefix
        assert len(registry) == 2

    def test_handles_get_tools_error(self) -> None:
        """Should handle errors from get_tools gracefully."""

        class BrokenToolProvider:
            @property
            def name(self) -> str:
                return "broken"

            @property
            def version(self) -> str:
                return "1.0.0"

            def initialize(self, app_config: Any) -> None:
                pass

            def shutdown(self) -> None:
                pass

            def get_tools(self, context: ResearchContext) -> list[ResearchTool]:
                raise RuntimeError("Failed to get tools")

        manager = PluginManager()
        manager._initialized = True
        manager._plugins.append(BrokenToolProvider())
        manager._register_tools()

        # Should not raise, just skip the broken plugin
        assert len(manager.get_tool_registry()) == 0


class TestPluginManagerPrompts:
    """Tests for prompt override collection."""

    def test_collects_prompt_overrides(self) -> None:
        """Should collect prompt overrides from PromptProvider plugins."""
        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [MockPromptProviderPlugin]
            manager.discover_and_load(app_config=MagicMock())

        context = create_test_context()
        overrides = manager.get_prompt_overrides(context)

        assert "researcher" in overrides
        assert overrides["researcher"] == "Custom instructions for researcher"

    def test_merges_multiple_prompt_providers(self) -> None:
        """Should merge overrides from multiple providers (later wins)."""
        plugin1 = MockPromptProviderPlugin(
            _name="p1",
            _overrides={"researcher": "First", "planner": "First planner"},
        )
        plugin2 = MockPromptProviderPlugin(
            _name="p2",
            _overrides={"researcher": "Second", "synthesizer": "Second synth"},
        )

        manager = PluginManager()
        manager._initialized = True
        manager._plugins.extend([plugin1, plugin2])

        context = create_test_context()
        overrides = manager.get_prompt_overrides(context)

        assert overrides["researcher"] == "Second"  # Later wins
        assert overrides["planner"] == "First planner"
        assert overrides["synthesizer"] == "Second synth"

    def test_handles_prompt_provider_error(self) -> None:
        """Should handle errors from get_prompt_overrides gracefully."""

        class BrokenPromptProvider:
            @property
            def name(self) -> str:
                return "broken"

            @property
            def version(self) -> str:
                return "1.0.0"

            def initialize(self, app_config: Any) -> None:
                pass

            def shutdown(self) -> None:
                pass

            def get_prompt_overrides(self, context: ResearchContext) -> dict[str, str]:
                raise RuntimeError("Failed to get prompts")

        manager = PluginManager()
        manager._initialized = True
        manager._plugins.append(BrokenPromptProvider())

        context = create_test_context()
        overrides = manager.get_prompt_overrides(context)

        assert overrides == {}  # Should return empty, not raise


class TestPluginManagerLifecycle:
    """Tests for plugin lifecycle management."""

    def test_shutdown_all_plugins(self) -> None:
        """Should shutdown all plugins."""
        plugin1 = MockPlugin(_name="p1")
        plugin2 = MockPlugin(_name="p2")

        manager = PluginManager()
        manager._initialized = True
        manager._plugins.extend([plugin1, plugin2])

        manager.shutdown()

        assert plugin1._shutdown_called
        assert plugin2._shutdown_called
        assert not manager.initialized
        assert len(manager) == 0

    def test_shutdown_handles_errors(self) -> None:
        """Should continue shutdown even if one plugin fails."""

        class FailingShutdownPlugin:
            @property
            def name(self) -> str:
                return "failing"

            @property
            def version(self) -> str:
                return "1.0.0"

            def initialize(self, app_config: Any) -> None:
                pass

            def shutdown(self) -> None:
                raise RuntimeError("Shutdown failed")

        plugin1 = FailingShutdownPlugin()
        plugin2 = MockPlugin()

        manager = PluginManager()
        manager._initialized = True
        manager._plugins.extend([plugin1, plugin2])

        manager.shutdown()  # Should not raise

        assert plugin2._shutdown_called
        assert not manager.initialized

    def test_get_plugins(self) -> None:
        """Should return copy of plugins list."""
        plugin = MockPlugin()

        manager = PluginManager()
        manager._initialized = True
        manager._plugins.append(plugin)

        plugins = manager.get_plugins()
        assert len(plugins) == 1
        assert plugins[0] is plugin

        # Modifying returned list should not affect manager
        plugins.clear()
        assert len(manager.get_plugins()) == 1

    def test_get_plugin_by_name(self) -> None:
        """Should find plugin by name."""
        plugin1 = MockPlugin(_name="alpha")
        plugin2 = MockPlugin(_name="beta")

        manager = PluginManager()
        manager._initialized = True
        manager._plugins.extend([plugin1, plugin2])

        found = manager.get_plugin("beta")
        assert found is plugin2

        not_found = manager.get_plugin("gamma")
        assert not_found is None


class TestPluginManagerFullPlugin:
    """Tests for plugins implementing all protocols."""

    def test_full_plugin_provides_tools_and_prompts(self) -> None:
        """Should collect both tools and prompts from full plugin."""
        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [MockFullPlugin]
            manager.discover_and_load(app_config=MagicMock())

        context = create_test_context()

        tools = manager.get_tools(context)
        assert len(tools) == 1
        assert tools[0].definition.name == "full-plugin-tool"

        overrides = manager.get_prompt_overrides(context)
        assert "synthesizer" in overrides


class TestPluginManagerConfiguration:
    """Tests for plugin configuration via app_config.plugins."""

    def test_disabled_plugin_is_skipped(self) -> None:
        """Should skip plugins that are disabled via configuration."""
        from deep_research.core.app_config import PluginConfig, PluginsConfig

        # Create a config that disables the mock-plugin
        plugins_config = PluginsConfig(
            configs={
                "mock-plugin": PluginConfig(enabled=False),
            }
        )
        app_config = MagicMock()
        app_config.plugins = plugins_config

        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [MockPlugin]
            manager.discover_and_load(app_config=app_config)

        # Plugin should not be loaded
        assert len(manager) == 0

    def test_enabled_plugin_is_loaded(self) -> None:
        """Should load plugins that are explicitly enabled."""
        from deep_research.core.app_config import PluginConfig, PluginsConfig

        plugins_config = PluginsConfig(
            configs={
                "mock-plugin": PluginConfig(enabled=True),
            }
        )
        app_config = MagicMock()
        app_config.plugins = plugins_config

        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [MockPlugin]
            manager.discover_and_load(app_config=app_config)

        assert len(manager) == 1

    def test_unconfigured_plugin_is_enabled_by_default(self) -> None:
        """Should load plugins that have no configuration (default enabled)."""
        from deep_research.core.app_config import PluginsConfig

        # Empty plugins config - nothing configured
        plugins_config = PluginsConfig(configs={})
        app_config = MagicMock()
        app_config.plugins = plugins_config

        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [MockPlugin]
            manager.discover_and_load(app_config=app_config)

        # Plugin should still be loaded (default enabled)
        assert len(manager) == 1

    def test_works_without_plugins_config(self) -> None:
        """Should work when app_config has no plugins attribute."""
        # MagicMock without plugins attribute
        app_config = MagicMock(spec=[])

        manager = PluginManager()

        with patch("deep_research.plugins.manager.discover_plugins") as mock_discover:
            mock_discover.return_value = [MockPlugin]
            manager.discover_and_load(app_config=app_config)

        # Plugin should be loaded
        assert len(manager) == 1
