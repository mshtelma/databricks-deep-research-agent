"""Tests for the academic_search Designer tool kind (P3 — surface feature 4.5)."""

from __future__ import annotations

from databricks_deep_research.tools.protocol import ToolKind

from deep_research.agent_designer.registry import tool_kinds_payload


def _academic() -> dict:
    by_kind = {t["kind"]: t for t in tool_kinds_payload()}
    assert "academic_search" in by_kind, "academic_search must be an addable Designer tool kind"
    return by_kind["academic_search"]


def test_academic_search_is_a_valid_framework_tool_kind() -> None:
    # The framework factory + ToolKind must accept it, else a saved workflow fails.
    assert "academic_search" in {str(getattr(k, "value", k)) for k in ToolKind}


def test_academic_search_exposes_provider_enum() -> None:
    schema = _academic()["config_schema"]
    provider = schema["properties"]["provider"]
    assert provider["enum"] == ["arxiv", "openalex", "pubmed_central", "semantic_scholar"]
    assert provider["default"] == "arxiv"  # key-less default


def test_academic_search_api_key_is_password_widget() -> None:
    schema = _academic()["config_schema"]
    assert schema["properties"]["api_key"].get("x-widget") == "password"
