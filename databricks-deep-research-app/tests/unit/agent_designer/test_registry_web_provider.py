"""Per-tool web-search provider fields exposed by the designer registry.

These fields drive the inspector dropdown (``SchemaField`` renders ``enum`` as a
Select). They must be merged onto ``web_search``/``web_research`` without a JSON
``default`` (so a freshly-added tool stays provider-absent = inherit), without
touching non-web kinds, and without mutating the module-level ``_TOOL_KIND_META``.
"""
from __future__ import annotations

import copy
from typing import Any

from deep_research.agent_designer.registry import _TOOL_KIND_META, tool_kinds_payload
from deep_research.core.app_config import SEARCH_PROVIDERS


def _schemas() -> dict[str, Any]:
    return {t["kind"]: t["config_schema"] for t in tool_kinds_payload()}


def test_web_search_exposes_provider_enum() -> None:
    props = _schemas()["web_search"]["properties"]
    assert props["provider"]["enum"] == list(SEARCH_PROVIDERS)
    for key in ("model", "model_family", "timeout_seconds", "resolve_redirects"):
        assert key in props
    assert "max_results" in props  # existing field preserved


def test_web_research_exposes_provider_enum() -> None:
    props = _schemas()["web_research"]["properties"]
    assert props["provider"]["enum"] == list(SEARCH_PROVIDERS)
    assert "total_results" in props  # existing field preserved


def test_provider_fields_have_no_json_default() -> None:
    props = _schemas()["web_search"]["properties"]
    for key in ("provider", "model", "model_family", "timeout_seconds", "resolve_redirects"):
        assert "default" not in props[key], f"{key} must not carry a JSON default"


def test_non_web_kinds_have_no_provider() -> None:
    schemas = _schemas()
    assert "provider" not in schemas["web_crawl"].get("properties", {})
    assert "provider" not in schemas["vector_search"]["properties"]


def test_tool_kind_meta_not_mutated_across_calls() -> None:
    before = copy.deepcopy(_TOOL_KIND_META["web_search"]["config_schema"])
    tool_kinds_payload()
    tool_kinds_payload()
    assert _TOOL_KIND_META["web_search"]["config_schema"] == before
    assert "provider" not in _TOOL_KIND_META["web_search"]["config_schema"]["properties"]


def test_model_field_is_grouped_endpoint_dropdown() -> None:
    """The endpoint field exposes per-family dropdown options via x-enumOptions,
    NOT a strict JSON ``enum`` — so the generic enum validator never rejects a
    custom endpoint outside the workspace list."""
    model = _schemas()["web_research"]["properties"]["model"]
    assert "enum" not in model  # not a strict constraint
    options = model["x-enumOptions"]
    assert isinstance(options, list) and options
    values = {o["value"] for o in options}
    assert "databricks-gemini-3-1-flash-lite" in values
    assert "databricks-gpt-5-mini" in values
    # Options carry family group headers for the labeled dropdown.
    assert {"Gemini", "OpenAI"} <= {o.get("group") for o in options}
