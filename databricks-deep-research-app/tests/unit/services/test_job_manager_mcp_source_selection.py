"""Tests for MCP source selection normalization in JobManager."""

from deep_research.services.job_manager import normalize_mcp_source_selection


def test_mcp_source_ids_become_mcp_attachments_not_enterprise_sources() -> None:
    """mcp:* source IDs should not be passed to enterprise tool loading."""
    enabled_sources, disabled_sources, enabled_mcp_servers = normalize_mcp_source_selection(
        source_scope="enterprise_only",
        enabled_sources=["mcp:tavily_mcp", "vs:catalog.schema.index"],
        disabled_sources=[],
        enabled_mcp_servers=None,
    )

    assert enabled_sources == ["vs:catalog.schema.index"]
    assert disabled_sources == []
    assert enabled_mcp_servers == ["tavily_mcp"]


def test_mcp_only_selection_preserves_explicit_empty_enterprise_selection() -> None:
    """MCP-only runs must not expand enterprise sources by converting [] to None."""
    enabled_sources, _, enabled_mcp_servers = normalize_mcp_source_selection(
        source_scope="enterprise_only",
        enabled_sources=["mcp:tavily_mcp"],
        disabled_sources=[],
        enabled_mcp_servers=None,
    )

    assert enabled_sources == []
    assert enabled_mcp_servers == ["tavily_mcp"]


def test_disabled_mcp_source_removes_legacy_and_derived_mcp_attachment() -> None:
    """Disabled mcp:* IDs should block both source-derived and legacy selections."""
    enabled_sources, disabled_sources, enabled_mcp_servers = normalize_mcp_source_selection(
        source_scope="enterprise_only",
        enabled_sources=["mcp:tavily_mcp"],
        disabled_sources=["mcp:tavily_mcp"],
        enabled_mcp_servers=["tavily_mcp", "sales_mcp"],
    )

    assert enabled_sources == []
    assert disabled_sources == ["mcp:tavily_mcp"]
    assert enabled_mcp_servers == ["sales_mcp"]


def test_web_only_scope_drops_mcp_attachments() -> None:
    """A web-only request must not attach MCP even if a legacy client sends it."""
    enabled_sources, _, enabled_mcp_servers = normalize_mcp_source_selection(
        source_scope="web_only",
        enabled_sources=["mcp:tavily_mcp"],
        disabled_sources=[],
        enabled_mcp_servers=["sales_mcp"],
    )

    assert enabled_sources == []
    assert enabled_mcp_servers is None
