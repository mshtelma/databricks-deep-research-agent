"""Tests for workflow resolution (012-workflow-provider).

Covers _resolve_workflow and _filter_workflow_tools in framework_orchestrator.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from deep_research.agent.framework_orchestrator import (
    _filter_workflow_tools,
    _resolve_workflow,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_YAML = """\
id: test-plugin-wf
name: Plugin Workflow
version: 1
root:
  id: root
  type: sequence
  label: Root
  children:
    - id: researcher
      type: agent
      label: Researcher
      config:
        subtype: researcher
        tools: [web_search, web_crawl, custom_tool]
"""


def _make_config(workflow_ref: str | None = None) -> Any:
    """Create a minimal OrchestrationConfig-like object."""
    cfg = MagicMock()
    cfg.workflow_ref = workflow_ref
    return cfg


def _make_plugin(
    name: str = "test_plugin",
    yaml_content: str | None = MINIMAL_YAML,
    raises: Exception | None = None,
) -> Any:
    """Create a mock plugin implementing WorkflowProviderPlugin."""
    plugin = MagicMock()
    plugin.name = name

    if raises:
        plugin.get_workflow_yaml.side_effect = raises
    else:
        plugin.get_workflow_yaml.return_value = yaml_content

    return plugin


def _make_plugin_manager(plugins: list[Any] | None = None) -> Any:
    """Create a mock PluginManager."""
    pm = MagicMock()
    pm.get_plugins.return_value = plugins or []
    return pm


# ---------------------------------------------------------------------------
# _resolve_workflow tests
# ---------------------------------------------------------------------------


class TestResolveWorkflow:
    """Tests for the _resolve_workflow function."""

    @patch("deep_research.agent.framework_orchestrator.translate")
    def test_none_ref_calls_translate(self, mock_translate: MagicMock) -> None:
        """workflow_ref=None should call translate() and return its result."""
        config = _make_config(workflow_ref=None)
        tool_names = ["web_search", "web_crawl"]
        mock_translate.return_value = MagicMock()

        result = _resolve_workflow(config, tool_names, None)

        mock_translate.assert_called_once_with(config, available_tools=tool_names)
        assert result is mock_translate.return_value

    @patch("deep_research.agent.framework_orchestrator.translate")
    def test_empty_ref_calls_translate(self, mock_translate: MagicMock) -> None:
        """workflow_ref="" should be treated as None and call translate()."""
        config = _make_config(workflow_ref="")
        tool_names = ["web_search"]
        mock_translate.return_value = MagicMock()

        result = _resolve_workflow(config, tool_names, None)

        mock_translate.assert_called_once_with(config, available_tools=tool_names)
        assert result is mock_translate.return_value

    def test_plugin_yaml_resolved(self) -> None:
        """Plugin that returns YAML should be used to build workflow."""
        config = _make_config(workflow_ref="my_workflow")
        plugin = _make_plugin(yaml_content=MINIMAL_YAML)
        pm = _make_plugin_manager([plugin])
        tool_names = ["web_search", "web_crawl", "custom_tool"]

        result = _resolve_workflow(config, tool_names, pm)

        plugin.get_workflow_yaml.assert_called_once_with("my_workflow")
        assert result.id == "test-plugin-wf"

    def test_plugin_not_found_raises(self) -> None:
        """No plugin matching ref should raise ValueError."""
        config = _make_config(workflow_ref="nonexistent")
        plugin = _make_plugin(yaml_content=None)
        pm = _make_plugin_manager([plugin])
        tool_names = ["web_search"]

        with pytest.raises(ValueError, match="workflow_ref='nonexistent'"):
            _resolve_workflow(config, tool_names, pm)

    def test_no_plugin_manager_raises(self) -> None:
        """No plugin manager with a ref set should raise ValueError."""
        config = _make_config(workflow_ref="some_ref")

        with pytest.raises(ValueError, match="workflow_ref='some_ref'"):
            _resolve_workflow(config, [], None)

    def test_plugin_error_continues(self) -> None:
        """Plugin that raises should be skipped; next plugin tried."""
        config = _make_config(workflow_ref="my_workflow")
        bad_plugin = _make_plugin(name="bad", raises=RuntimeError("boom"))
        good_plugin = _make_plugin(name="good", yaml_content=MINIMAL_YAML)
        pm = _make_plugin_manager([bad_plugin, good_plugin])
        tool_names = ["web_search", "web_crawl", "custom_tool"]

        result = _resolve_workflow(config, tool_names, pm)

        bad_plugin.get_workflow_yaml.assert_called_once_with("my_workflow")
        good_plugin.get_workflow_yaml.assert_called_once_with("my_workflow")
        assert result.id == "test-plugin-wf"

    def test_first_plugin_wins(self) -> None:
        """First plugin returning YAML wins; subsequent plugins not called."""
        config = _make_config(workflow_ref="my_workflow")
        first = _make_plugin(name="first", yaml_content=MINIMAL_YAML)
        second = _make_plugin(name="second", yaml_content=MINIMAL_YAML)
        pm = _make_plugin_manager([first, second])
        tool_names = ["web_search", "web_crawl", "custom_tool"]

        _resolve_workflow(config, tool_names, pm)

        first.get_workflow_yaml.assert_called_once()
        second.get_workflow_yaml.assert_not_called()


# ---------------------------------------------------------------------------
# _filter_workflow_tools tests
# ---------------------------------------------------------------------------


class TestFilterWorkflowTools:
    """Tests for the _filter_workflow_tools function."""

    def test_tool_filtering(self) -> None:
        """Tools not in available_tools should be removed from agent nodes."""
        from databricks_deep_research import load_workflow_from_string

        defn = load_workflow_from_string(MINIMAL_YAML)
        # Only web_search is available; web_crawl and custom_tool should be removed
        _filter_workflow_tools(defn, ["web_search"])

        researcher = defn.root.children[0]
        assert researcher.config["tools"] == ["web_search"]

    def test_all_tools_available(self) -> None:
        """When all tools are available, nothing is removed."""
        from databricks_deep_research import load_workflow_from_string

        defn = load_workflow_from_string(MINIMAL_YAML)
        _filter_workflow_tools(defn, ["web_search", "web_crawl", "custom_tool"])

        researcher = defn.root.children[0]
        assert set(researcher.config["tools"]) == {
            "web_search",
            "web_crawl",
            "custom_tool",
        }

    def test_no_tools_config(self) -> None:
        """Agent nodes without tools config should not raise."""
        from databricks_deep_research import load_workflow_from_string

        yaml_no_tools = """\
id: no-tools-wf
name: No Tools
version: 1
root:
  id: root
  type: agent
  label: Root Agent
  config:
    subtype: coordinator
"""
        defn = load_workflow_from_string(yaml_no_tools)
        # Should not raise
        _filter_workflow_tools(defn, ["web_search"])
