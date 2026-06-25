"""Generic enum validation for builtin tool config.

``semantic_validation_errors`` now rejects a config value that is not a member of
its schema property's ``enum`` (e.g. a typo'd web-search ``provider`` or
``model_family``). This runs on every save path — including UI-only saves, which
bypass ``normalize_ast`` — so a bad provider is caught at design time instead of
surfacing as a runtime ValueError. Absent / blank values are skipped (they inherit
the workspace default).
"""
from __future__ import annotations

from typing import Any

from deep_research.agent_designer.semantic_validation import semantic_validation_errors


def _ast_with_web_search_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "1.0",
        "name": "t",
        "description": "d",
        "tools": [{"name": "web_search", "kind": "web_search", "config": config}],
        "root": {
            "id": "r",
            "type": "agent",
            "config": {"subtype": "researcher", "tools": ["web_search"]},
        },
    }


def test_unknown_provider_is_rejected() -> None:
    errors = semantic_validation_errors(_ast_with_web_search_config({"provider": "bing"}))
    assert any("provider" in e.message and "bing" in e.message for e in errors)


def test_valid_provider_passes() -> None:
    errors = semantic_validation_errors(
        _ast_with_web_search_config(
            {"provider": "databricks", "model": "databricks-gpt-5"}
        )
    )
    assert not any("provider" in e.message for e in errors)


def test_absent_provider_passes() -> None:
    errors = semantic_validation_errors(_ast_with_web_search_config({}))
    assert not any("provider" in e.message for e in errors)


def test_blank_provider_passes() -> None:
    errors = semantic_validation_errors(_ast_with_web_search_config({"provider": "  "}))
    assert not any("provider" in e.message for e in errors)


def test_unknown_model_family_is_rejected() -> None:
    errors = semantic_validation_errors(
        _ast_with_web_search_config(
            {"provider": "databricks", "model_family": "claude"}
        )
    )
    assert any("model_family" in e.message for e in errors)


def test_valid_model_family_passes() -> None:
    errors = semantic_validation_errors(
        _ast_with_web_search_config(
            {"provider": "databricks", "model_family": "gemini"}
        )
    )
    assert not any("model_family" in e.message for e in errors)


def test_model_family_contradicting_endpoint_is_blocked() -> None:
    """openai family on a Gemini endpoint is a guaranteed runtime 400 — block it
    at save time with a blocking error (the AIS AccountResearch incident)."""
    errors = semantic_validation_errors(
        _ast_with_web_search_config(
            {
                "provider": "databricks",
                "model": "databricks-gemini-3-1-flash-lite",
                "model_family": "openai",
            }
        )
    )
    matching = [
        e for e in errors if "model_family" in e.message and "contradicts" in e.message
    ]
    assert matching, "expected a blocking model_family/endpoint mismatch error"
    assert all(e.severity == "blocking" for e in matching)


def test_matching_endpoint_and_family_passes() -> None:
    errors = semantic_validation_errors(
        _ast_with_web_search_config(
            {
                "provider": "databricks",
                "model": "databricks-gpt-5-mini",
                "model_family": "openai",
            }
        )
    )
    assert not any("contradicts" in e.message for e in errors)


def test_family_only_without_endpoint_passes() -> None:
    """Family-only (no endpoint) is not a contradiction — the endpoint is
    resolved to that family's default downstream."""
    errors = semantic_validation_errors(
        _ast_with_web_search_config(
            {"provider": "databricks", "model_family": "openai"}
        )
    )
    assert not any("contradicts" in e.message for e in errors)


def test_custom_undetectable_endpoint_with_family_passes() -> None:
    """A custom endpoint whose family can't be inferred trusts the explicit
    family (no false-positive block)."""
    errors = semantic_validation_errors(
        _ast_with_web_search_config(
            {
                "provider": "databricks",
                "model": "acme-search-v2",
                "model_family": "openai",
            }
        )
    )
    assert not any("contradicts" in e.message for e in errors)
