"""Unit tests for SourceScopeConfig.

Pure unit tests testing the scope filtering logic — no mocks needed.
"""

from dataclasses import dataclass

from deep_research.schemas.source_scope import SourceScope, SourceScopeConfig

# =========================================================================
# TestIsTypeEnabled
# =========================================================================


class TestIsTypeEnabled:
    """Tests for SourceScopeConfig.is_type_enabled."""

    def test_enterprise_only_blocks_web_search(self) -> None:
        """ENTERPRISE_ONLY scope blocks web_search."""
        config = SourceScopeConfig(scope=SourceScope.ENTERPRISE_ONLY)

        assert config.is_type_enabled("web_search") is False

    def test_enterprise_only_allows_enterprise_types(self) -> None:
        """ENTERPRISE_ONLY scope allows enterprise types."""
        config = SourceScopeConfig(scope=SourceScope.ENTERPRISE_ONLY)

        assert config.is_type_enabled("vector_search") is True
        assert config.is_type_enabled("genie") is True
        assert config.is_type_enabled("knowledge_assistant") is True

    def test_web_only_blocks_enterprise_types(self) -> None:
        """WEB_ONLY scope blocks all enterprise source types."""
        config = SourceScopeConfig(scope=SourceScope.WEB_ONLY)

        assert config.is_type_enabled("vector_search") is False
        assert config.is_type_enabled("genie") is False
        assert config.is_type_enabled("knowledge_assistant") is False

    def test_web_only_allows_web_search(self) -> None:
        """WEB_ONLY scope allows web_search."""
        config = SourceScopeConfig(scope=SourceScope.WEB_ONLY)

        assert config.is_type_enabled("web_search") is True

    def test_toggle_override_disables_type(self) -> None:
        """Type-level toggle can disable a specific type."""
        config = SourceScopeConfig(enable_genie=False)

        assert config.is_type_enabled("genie") is False
        # Other types still enabled
        assert config.is_type_enabled("vector_search") is True

    def test_all_scope_allows_everything(self) -> None:
        """ALL scope allows all source types."""
        config = SourceScopeConfig(scope=SourceScope.ALL)

        assert config.is_type_enabled("vector_search") is True
        assert config.is_type_enabled("genie") is True
        assert config.is_type_enabled("knowledge_assistant") is True
        assert config.is_type_enabled("web_search") is True
        assert config.is_type_enabled("uploaded_file") is True

    def test_unknown_type_defaults_to_true(self) -> None:
        """Unknown source type defaults to enabled."""
        config = SourceScopeConfig()

        assert config.is_type_enabled("custom_unknown") is True

    def test_uploaded_file_toggle(self) -> None:
        """Uploaded file toggle can be disabled."""
        config = SourceScopeConfig(enable_uploaded_files=False)

        assert config.is_type_enabled("uploaded_file") is False


# =========================================================================
# TestIsSourceEnabled
# =========================================================================


class TestIsSourceEnabled:
    """Tests for SourceScopeConfig.is_source_enabled."""

    def test_disabled_source_blocked(self) -> None:
        """Source in disabled_sources list is blocked."""
        config = SourceScopeConfig(disabled_sources=["my-source"])

        assert config.is_source_enabled("my-source", "vector_search") is False

    def test_enabled_whitelist_restricts_to_listed_only(self) -> None:
        """When enabled_sources is set, only those sources are allowed."""
        config = SourceScopeConfig(enabled_sources=["a", "b"])

        assert config.is_source_enabled("a", "vector_search") is True
        assert config.is_source_enabled("b", "vector_search") is True
        assert config.is_source_enabled("c", "vector_search") is False

    def test_type_disabled_blocks_even_if_source_enabled(self) -> None:
        """Type-level block takes precedence over source whitelist."""
        config = SourceScopeConfig(
            scope=SourceScope.WEB_ONLY,
            enabled_sources=["vs1"],
        )

        assert config.is_source_enabled("vs1", "vector_search") is False

    def test_no_lists_allows_all_sources(self) -> None:
        """No enabled/disabled lists allows all sources."""
        config = SourceScopeConfig()

        assert config.is_source_enabled("anything", "vector_search") is True
        assert config.is_source_enabled("other", "genie") is True

    def test_disabled_takes_precedence_over_enabled(self) -> None:
        """disabled_sources is checked before enabled_sources."""
        config = SourceScopeConfig(
            enabled_sources=["blocked-source"],
            disabled_sources=["blocked-source"],
        )

        assert config.is_source_enabled("blocked-source", "vector_search") is False


# =========================================================================
# TestFilterSources
# =========================================================================


@dataclass
class _MockSource:
    """Simple mock source for filter testing."""

    name: str
    type: str


class TestFilterSources:
    """Tests for SourceScopeConfig.filter_sources."""

    def test_filters_mixed_list(self) -> None:
        """Filters out disabled sources from a mixed list."""
        config = SourceScopeConfig(disabled_sources=["blocked"])
        sources = [
            _MockSource("ok", "vector_search"),
            _MockSource("blocked", "vector_search"),
            _MockSource("also-ok", "genie"),
        ]

        filtered = config.filter_sources(sources)

        assert len(filtered) == 2
        names = [s.name for s in filtered]
        assert "ok" in names
        assert "also-ok" in names
        assert "blocked" not in names

    def test_custom_getters(self) -> None:
        """filter_sources works with custom name/type getter lambdas."""
        config = SourceScopeConfig(disabled_sources=["src-x"])
        sources = [
            {"source_name": "src-x", "source_type": "vector_search"},
            {"source_name": "src-y", "source_type": "genie"},
        ]

        filtered = config.filter_sources(
            sources,
            name_getter=lambda x: x["source_name"],
            type_getter=lambda x: x["source_type"],
        )

        assert len(filtered) == 1
        assert filtered[0]["source_name"] == "src-y"

    def test_empty_sources_returns_empty(self) -> None:
        """Filtering empty list returns empty list."""
        config = SourceScopeConfig()

        filtered = config.filter_sources([])

        assert filtered == []

    def test_scope_filters_by_type(self) -> None:
        """Scope-level type filter removes all sources of blocked types."""
        config = SourceScopeConfig(scope=SourceScope.WEB_ONLY)
        sources = [
            _MockSource("vs1", "vector_search"),
            _MockSource("web1", "web_search"),
            _MockSource("genie1", "genie"),
        ]

        filtered = config.filter_sources(sources)

        assert len(filtered) == 1
        assert filtered[0].name == "web1"


# =========================================================================
# TestToDict
# =========================================================================


class TestToDict:
    """Tests for SourceScopeConfig.to_dict."""

    def test_default_config_to_dict(self) -> None:
        """Default config serializes with expected keys."""
        config = SourceScopeConfig()
        d = config.to_dict()

        assert d["scope"] == SourceScope.ALL
        assert d["enabled_sources"] is None
        assert d["disabled_sources"] == []
        assert d["enable_vector_search"] is True
        assert d["enable_web_search"] is True

    def test_custom_config_to_dict(self) -> None:
        """Custom config values are preserved in dict."""
        config = SourceScopeConfig(
            scope=SourceScope.ENTERPRISE_ONLY,
            enabled_sources=["src-1"],
            disabled_sources=["src-2"],
            enable_web_search=False,
        )
        d = config.to_dict()

        assert d["scope"] == SourceScope.ENTERPRISE_ONLY
        assert d["enabled_sources"] == ["src-1"]
        assert d["disabled_sources"] == ["src-2"]
        assert d["enable_web_search"] is False
