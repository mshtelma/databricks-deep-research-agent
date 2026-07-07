from __future__ import annotations

import asyncio
import json
from typing import Any

from deep_research.agent_designer.registry import tool_kinds_payload


def _by_kind() -> dict[str, dict[str, Any]]:
    return {item["kind"]: item for item in tool_kinds_payload()}


def test_callable_tool_kinds_are_visible_in_designer_registry() -> None:
    kinds = _by_kind()

    assert {"decorated", "uc_function", "python_function", "registered", "enterprise"}.issubset(
        kinds
    )
    assert kinds["decorated"]["label"] == "Python Function"
    assert kinds["uc_function"]["label"] == "Unity Catalog Function"
    # The three Python-ish kinds must not share a label in the shared picker.
    assert kinds["python_function"]["label"] == "Inline Python Function"
    assert kinds["registered"]["label"] == "Registered Python Tool"


def test_uc_tool_kind_is_retired_from_authoring() -> None:
    """uc_tool has no discovery and no runtime registration path; it must not
    be offered as a declarable kind (direct {type: uc_tool} refs stay parseable
    for imported YAML, which is a ToolStepForm concern, not a registry one)."""
    assert "uc_tool" not in _by_kind()


def test_python_function_schema_matches_decorated_factory_contract() -> None:
    schema = _by_kind()["decorated"]["config_schema"]

    assert schema["required"] == ["import"]
    assert "import" in schema["properties"]
    assert "requires_confirmation" in schema["properties"]


def test_uc_function_schema_is_the_first_class_obo_sql_contract() -> None:
    """uc_function carries the deterministic-functions contract (config.function
    + auto-introspected params), not the retired external-alias shape
    (config.function_name)."""
    schema = _by_kind()["uc_function"]["config_schema"]

    assert schema["required"] == ["function"]
    assert "function" in schema["properties"]
    assert "params" in schema["properties"]
    assert schema["properties"]["function"].get("x-widget") == "uc-function-picker"
    assert "function_name" not in schema["properties"]


def test_enterprise_schema_captures_external_runtime_name() -> None:
    schema = _by_kind()["enterprise"]["config_schema"]

    assert schema["required"] == ["tool_name"]
    assert "tool_name" in schema["properties"]


def test_list_tool_kinds_uses_same_callable_registry() -> None:
    from deep_research.agent_designer.framework_tools import ListToolKindsTool

    result = asyncio.run(ListToolKindsTool().execute({}, _context=None))  # type: ignore[arg-type]
    payload = json.loads(result.content)

    kinds = set(payload["kinds"])
    assert {"decorated", "uc_function", "python_function", "registered", "enterprise"}.issubset(
        kinds
    )
    assert "uc_tool" not in kinds
