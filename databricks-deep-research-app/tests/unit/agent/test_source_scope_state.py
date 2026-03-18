"""Tests for source scope helper methods on ResearchState.

Part of 008-data-source-selection feature implementation.
"""

import pytest

from deep_research.agent.state import ResearchState
from deep_research.schemas.source_scope import SourceScope, SourceScopeConfig


class TestSourceScopeHelpers:
    """Test source scope helper methods on ResearchState."""

    def test_is_web_search_allowed_no_scope(self) -> None:
        """When source_scope_config is None, web search is allowed."""
        state = ResearchState(query="test query")
        assert state.source_scope_config is None
        assert state.is_web_search_allowed() is True

    def test_is_web_search_allowed_all_scope(self) -> None:
        """When scope is ALL, web search is allowed."""
        state = ResearchState(query="test query")
        state.source_scope_config = SourceScopeConfig(scope=SourceScope.ALL)
        assert state.is_web_search_allowed() is True

    def test_is_web_search_allowed_web_only_scope(self) -> None:
        """When scope is WEB_ONLY, web search is allowed."""
        state = ResearchState(query="test query")
        state.source_scope_config = SourceScopeConfig(scope=SourceScope.WEB_ONLY)
        assert state.is_web_search_allowed() is True

    def test_is_web_search_allowed_enterprise_only_scope(self) -> None:
        """When scope is ENTERPRISE_ONLY, web search is NOT allowed."""
        state = ResearchState(query="test query")
        state.source_scope_config = SourceScopeConfig(scope=SourceScope.ENTERPRISE_ONLY)
        assert state.is_web_search_allowed() is False

    def test_is_enterprise_search_allowed_no_scope(self) -> None:
        """When source_scope_config is None, enterprise sources are allowed."""
        state = ResearchState(query="test query")
        assert state.is_enterprise_search_allowed() is True

    def test_is_enterprise_search_allowed_all_scope(self) -> None:
        """When scope is ALL, enterprise sources are allowed."""
        state = ResearchState(query="test query")
        state.source_scope_config = SourceScopeConfig(scope=SourceScope.ALL)
        assert state.is_enterprise_search_allowed() is True

    def test_is_enterprise_search_allowed_enterprise_only_scope(self) -> None:
        """When scope is ENTERPRISE_ONLY, enterprise sources are allowed."""
        state = ResearchState(query="test query")
        state.source_scope_config = SourceScopeConfig(scope=SourceScope.ENTERPRISE_ONLY)
        assert state.is_enterprise_search_allowed() is True

    def test_is_enterprise_search_allowed_web_only_scope(self) -> None:
        """When scope is WEB_ONLY, enterprise sources are NOT allowed."""
        state = ResearchState(query="test query")
        state.source_scope_config = SourceScopeConfig(scope=SourceScope.WEB_ONLY)
        assert state.is_enterprise_search_allowed() is False

    def test_get_active_scope_no_config(self) -> None:
        """When source_scope_config is None, get_active_scope returns 'all'."""
        state = ResearchState(query="test query")
        assert state.get_active_scope() == "all"

    def test_get_active_scope_all(self) -> None:
        """get_active_scope returns correct string for ALL scope."""
        state = ResearchState(query="test query")
        state.source_scope_config = SourceScopeConfig(scope=SourceScope.ALL)
        assert state.get_active_scope() == "all"

    def test_get_active_scope_enterprise_only(self) -> None:
        """get_active_scope returns correct string for ENTERPRISE_ONLY scope."""
        state = ResearchState(query="test query")
        state.source_scope_config = SourceScopeConfig(scope=SourceScope.ENTERPRISE_ONLY)
        assert state.get_active_scope() == "enterprise_only"

    def test_get_active_scope_web_only(self) -> None:
        """get_active_scope returns correct string for WEB_ONLY scope."""
        state = ResearchState(query="test query")
        state.source_scope_config = SourceScopeConfig(scope=SourceScope.WEB_ONLY)
        assert state.get_active_scope() == "web_only"

    def test_source_scope_config_type_hint(self) -> None:
        """Verify source_scope_config can be assigned SourceScopeConfig."""
        state = ResearchState(query="test query")
        config = SourceScopeConfig(
            scope=SourceScope.ENTERPRISE_ONLY,
            enabled_sources=["source1", "source2"],
            disabled_sources=["source3"],
        )
        state.source_scope_config = config
        assert state.source_scope_config.scope == SourceScope.ENTERPRISE_ONLY
        assert state.source_scope_config.enabled_sources == ["source1", "source2"]
        assert state.source_scope_config.disabled_sources == ["source3"]


class TestSourceScopeStateIntegration:
    """Integration tests for source scope in ResearchState to_dict."""

    def test_to_dict_with_source_scope(self) -> None:
        """to_dict correctly serializes source_scope_config."""
        state = ResearchState(query="test query")
        state.source_scope_config = SourceScopeConfig(
            scope=SourceScope.ENTERPRISE_ONLY,
            enabled_sources=["vs1"],
        )

        result = state.to_dict()

        assert result["source_scope_config"] is not None
        assert result["source_scope_config"]["scope"] == SourceScope.ENTERPRISE_ONLY
        assert result["source_scope_config"]["enabled_sources"] == ["vs1"]

    def test_to_dict_without_source_scope(self) -> None:
        """to_dict correctly handles None source_scope_config."""
        state = ResearchState(query="test query")

        result = state.to_dict()

        assert result["source_scope_config"] is None
