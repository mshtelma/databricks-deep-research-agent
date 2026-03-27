"""Read-only namespace inspection tool for the shared compute namespace.

Returns a structured, filtered view of user-defined variables stored by
``PythonComputeTool``.  Does not execute code or expose raw Python
introspection.  Budget-free: does not count against agent tool-call limits.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from databricks_deep_research.tools.protocol import (
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

_DEFAULT_DESCRIPTION = (
    "List variables stored in the shared compute namespace. Returns names, "
    "types, and values of variables persisted by compute() calls. Use this "
    "to check what operands the researcher stored before computing. "
    "Supports filtering: prefix=\"op\" for operand variables. "
    "This tool is free — it does not count against your tool call budget."
)


class ComputeNamespaceListTool:
    """Read-only inspection tool for the shared compute namespace.

    Returns a structured, filtered view of user-defined variables stored
    by PythonComputeTool.  Does not execute code or expose raw Python
    introspection.  Budget-free: does not count against agent tool-call limits.
    """

    def __init__(
        self,
        *,
        compute_resolver: Callable[[], Any | None],
        name: str = "compute_namespace_list",
        description: str = "",
    ) -> None:
        self._resolve_compute = compute_resolver
        self._name = name
        self._description = description or _DEFAULT_DESCRIPTION

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "prefix": {
                        "type": "string",
                        "description": "Filter variables by name prefix (e.g. 'op')",
                    },
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Request specific variable names",
                    },
                    "max_items": {
                        "type": "integer",
                        "description": "Maximum variables to return (default 50)",
                    },
                },
                "additionalProperties": False,
            },
            source_type="builtin",
            source_kind=SourceKind.builtin,
            metadata={"budget_free": True},
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return arguments

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext,
    ) -> ToolResult:
        compute_tool = self._resolve_compute()
        if compute_tool is None:
            return ToolResult(
                content="No shared compute namespace available.",
                success=True,
                data={"source_kind": SourceKind.builtin},
            )

        entries = compute_tool.list_user_namespace(
            prefix=arguments.get("prefix"),
            names=arguments.get("names"),
            max_items=arguments.get("max_items", 50),
        )
        if not entries:
            return ToolResult(
                content="Namespace is empty — no variables stored yet.",
                success=True,
                data={"source_kind": SourceKind.builtin},
            )

        lines = [f"Namespace variables ({len(entries)}):"]
        for e in entries:
            lines.append(f"  {e['name']}: {e['type']} = {e.get('value', '?')}")
        return ToolResult(
            content="\n".join(lines),
            success=True,
            data={"source_kind": SourceKind.builtin},
        )
