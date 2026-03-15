"""Unit tests for plugin discovery module."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from deep_research.plugins.discovery import (
    PLUGIN_ENTRY_POINT_GROUP,
    discover_plugins,
    discover_tools,
    get_plugin_count,
)


class TestDiscoverPlugins:
    """Tests for discover_plugins function."""

    def test_returns_empty_list_when_no_plugins(self) -> None:
        """Should return empty list when no entry points exist."""
        with patch("deep_research.plugins.discovery.importlib.metadata.entry_points") as mock_ep:
            mock_ep.return_value = []
            result = discover_plugins()
            assert result == []
            mock_ep.assert_called_once_with(group=PLUGIN_ENTRY_POINT_GROUP)

    def test_loads_single_plugin(self) -> None:
        """Should load and return a single plugin class."""

        class MockPlugin:
            pass

        mock_entry_point = MagicMock()
        mock_entry_point.name = "test_plugin"
        mock_entry_point.value = "test.module:TestPlugin"
        mock_entry_point.load.return_value = MockPlugin

        with patch("deep_research.plugins.discovery.importlib.metadata.entry_points") as mock_ep:
            mock_ep.return_value = [mock_entry_point]
            result = discover_plugins()

            assert len(result) == 1
            assert result[0] is MockPlugin
            mock_entry_point.load.assert_called_once()

    def test_loads_multiple_plugins(self) -> None:
        """Should load and return multiple plugin classes."""

        class MockPlugin1:
            pass

        class MockPlugin2:
            pass

        ep1 = MagicMock()
        ep1.name = "plugin1"
        ep1.value = "test.module:Plugin1"
        ep1.load.return_value = MockPlugin1

        ep2 = MagicMock()
        ep2.name = "plugin2"
        ep2.value = "test.module:Plugin2"
        ep2.load.return_value = MockPlugin2

        with patch("deep_research.plugins.discovery.importlib.metadata.entry_points") as mock_ep:
            mock_ep.return_value = [ep1, ep2]
            result = discover_plugins()

            assert len(result) == 2
            assert MockPlugin1 in result
            assert MockPlugin2 in result

    def test_handles_import_error_gracefully(self) -> None:
        """Should skip plugins that fail to import."""
        mock_entry_point = MagicMock()
        mock_entry_point.name = "bad_plugin"
        mock_entry_point.value = "nonexistent.module:BadPlugin"
        mock_entry_point.load.side_effect = ImportError("Module not found")

        with patch("deep_research.plugins.discovery.importlib.metadata.entry_points") as mock_ep:
            mock_ep.return_value = [mock_entry_point]
            result = discover_plugins()

            assert result == []

    def test_handles_generic_error_gracefully(self) -> None:
        """Should skip plugins that raise other exceptions."""
        mock_entry_point = MagicMock()
        mock_entry_point.name = "broken_plugin"
        mock_entry_point.value = "test.module:BrokenPlugin"
        mock_entry_point.load.side_effect = Exception("Something went wrong")

        with patch("deep_research.plugins.discovery.importlib.metadata.entry_points") as mock_ep:
            mock_ep.return_value = [mock_entry_point]
            result = discover_plugins()

            assert result == []

    def test_handles_entry_points_error(self) -> None:
        """Should return empty list if entry_points call fails."""
        with patch("deep_research.plugins.discovery.importlib.metadata.entry_points") as mock_ep:
            mock_ep.side_effect = Exception("Failed to get entry points")
            result = discover_plugins()
            assert result == []

    def test_continues_after_failed_plugin(self) -> None:
        """Should continue loading other plugins after one fails."""

        class GoodPlugin:
            pass

        bad_ep = MagicMock()
        bad_ep.name = "bad_plugin"
        bad_ep.value = "test:Bad"
        bad_ep.load.side_effect = ImportError("Not found")

        good_ep = MagicMock()
        good_ep.name = "good_plugin"
        good_ep.value = "test:Good"
        good_ep.load.return_value = GoodPlugin

        with patch("deep_research.plugins.discovery.importlib.metadata.entry_points") as mock_ep:
            mock_ep.return_value = [bad_ep, good_ep]
            result = discover_plugins()

            assert len(result) == 1
            assert result[0] is GoodPlugin


class TestDiscoverTools:
    """Tests for discover_tools function."""

    def test_returns_empty_list_when_no_tools(self) -> None:
        """Should return empty list when no tool entry points exist."""
        with patch("deep_research.plugins.discovery.importlib.metadata.entry_points") as mock_ep:
            mock_ep.return_value = []
            result = discover_tools()
            assert result == []
            mock_ep.assert_called_once_with(group="deep_research.tools")

    def test_loads_tool_classes(self) -> None:
        """Should load and return tool classes."""

        class MockTool:
            pass

        mock_entry_point = MagicMock()
        mock_entry_point.name = "test_tool"
        mock_entry_point.value = "test.tools:TestTool"
        mock_entry_point.load.return_value = MockTool

        with patch("deep_research.plugins.discovery.importlib.metadata.entry_points") as mock_ep:
            mock_ep.return_value = [mock_entry_point]
            result = discover_tools()

            assert len(result) == 1
            assert result[0] is MockTool

    def test_handles_import_error(self) -> None:
        """Should skip tools that fail to import."""
        mock_entry_point = MagicMock()
        mock_entry_point.name = "bad_tool"
        mock_entry_point.value = "nonexistent:Tool"
        mock_entry_point.load.side_effect = ImportError("Not found")

        with patch("deep_research.plugins.discovery.importlib.metadata.entry_points") as mock_ep:
            mock_ep.return_value = [mock_entry_point]
            result = discover_tools()
            assert result == []

    def test_handles_entry_points_error(self) -> None:
        """Should return empty list if entry_points call fails."""
        with patch("deep_research.plugins.discovery.importlib.metadata.entry_points") as mock_ep:
            mock_ep.side_effect = Exception("Failed")
            result = discover_tools()
            assert result == []


class TestGetPluginCount:
    """Tests for get_plugin_count function."""

    def test_returns_zero_when_no_plugins(self) -> None:
        """Should return 0 when no plugins registered."""
        with patch("deep_research.plugins.discovery.importlib.metadata.entry_points") as mock_ep:
            mock_ep.return_value = []
            result = get_plugin_count()
            assert result == 0

    def test_returns_correct_count(self) -> None:
        """Should return correct plugin count."""
        mock_eps = [MagicMock() for _ in range(3)]
        with patch("deep_research.plugins.discovery.importlib.metadata.entry_points") as mock_ep:
            mock_ep.return_value = mock_eps
            result = get_plugin_count()
            assert result == 3

    def test_returns_zero_on_error(self) -> None:
        """Should return 0 if entry_points call fails."""
        with patch("deep_research.plugins.discovery.importlib.metadata.entry_points") as mock_ep:
            mock_ep.side_effect = Exception("Failed")
            result = get_plugin_count()
            assert result == 0
