from __future__ import annotations

import asyncio
import json
from typing import Any

from deep_research.agent_designer.registry import tool_kinds_payload


def _by_kind() -> dict[str, dict[str, Any]]:
    return {item["kind"]: item for item in tool_kinds_payload()}


def test_callable_tool_kinds_are_visible_in_designer_registry() -> None:
    kinds = _by_kind()

    assert {"decorated", "uc_function", "uc_tool", "enterprise"}.issubset(kinds)
    assert kinds["decorated"]["label"] == "Python Function"
    assert kinds["uc_function"]["label"] == "Unity Catalog Function"


def test_python_function_schema_matches_decorated_factory_contract() -> None:
    schema = _by_kind()["decorated"]["config_schema"]

    assert schema["required"] == ["import"]
    assert "import" in schema["properties"]
    assert "requires_confirmation" in schema["properties"]


def test_uc_callable_schemas_capture_external_runtime_names() -> None:
    kinds = _by_kind()

    uc_function_schema = kinds["uc_function"]["config_schema"]
    uc_tool_schema = kinds["uc_tool"]["config_schema"]

    assert uc_function_schema["required"] == ["function_name"]
    assert "function_name" in uc_function_schema["properties"]
    assert uc_tool_schema["required"] == ["tool_name"]
    assert "tool_name" in uc_tool_schema["properties"]


def test_list_tool_kinds_uses_same_callable_registry() -> None:
    from deep_research.agent_designer.framework_tools import ListToolKindsTool

    result = asyncio.run(ListToolKindsTool().execute({}, _context=None))  # type: ignore[arg-type]
    payload = json.loads(result.content)

    assert {"decorated", "uc_function", "uc_tool", "enterprise"}.issubset(set(payload["kinds"]))
