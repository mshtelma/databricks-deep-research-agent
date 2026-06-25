"""Compatibility rules for MCP tool schemas and arguments.

MCP servers own their tool schemas, but some servers expose combinations that
are individually schema-valid and still rejected by the downstream provider.
This module keeps those compatibility fixes data-driven and centralized instead
of scattering per-server branches through the generic MCP adapter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArgumentNormalization:
    """One argument rewrite performed by a compatibility transform."""

    field: str
    original: Any
    normalized: Any


class MCPCompatibilityTransform(Protocol):
    """Schema + argument transform for one known MCP compatibility concern."""

    def normalize_schema(self, schema: dict[str, Any]) -> None:
        """Mutate a discovered tool schema before exposing it to the model."""

    def normalize_arguments(self, arguments: dict[str, Any]) -> tuple[ArgumentNormalization, ...]:
        """Mutate model-emitted arguments before sending them to the MCP server."""


@dataclass(frozen=True)
class EnumFieldCompatibility:
    """Enum-like compatibility for one tool parameter."""

    parameter: str
    allowed_values: tuple[str, ...] = ()
    aliases: dict[str, str] = field(default_factory=dict)
    default: str | None = None
    description: str | None = None
    case_sensitive: bool = False

    def _key(self, value: str) -> str:
        return value if self.case_sensitive else value.strip().lower()

    def normalize_argument_value(self, value: Any) -> tuple[Any, bool]:
        if not isinstance(value, str):
            return value, False
        aliases = {
            self._key(alias): replacement
            for alias, replacement in self.aliases.items()
        }
        replacement = aliases.get(self._key(value))
        if replacement is None:
            return value, False
        return replacement, replacement != value

    def normalize_schema(self, schema: dict[str, Any]) -> None:
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return
        parameter_schema = properties.get(self.parameter)
        if not isinstance(parameter_schema, dict):
            return

        if self.allowed_values:
            allowed_keys = {self._key(value) for value in self.allowed_values}
            enum_values = parameter_schema.get("enum")
            if isinstance(enum_values, list):
                parameter_schema["enum"] = [
                    value
                    for value in enum_values
                    if isinstance(value, str) and self._key(value) in allowed_keys
                ] or list(self.allowed_values)
            else:
                parameter_schema["enum"] = list(self.allowed_values)

        if self.default is not None:
            default = parameter_schema.get("default")
            normalized_default, changed = self.normalize_argument_value(default)
            if changed:
                parameter_schema["default"] = normalized_default
            elif (
                "default" not in parameter_schema
                or isinstance(default, str)
                and self.allowed_values
                and self._key(default) not in {self._key(v) for v in self.allowed_values}
            ):
                parameter_schema["default"] = self.default

        if self.description is not None:
            parameter_schema["description"] = self.description

    def normalize_arguments(self, arguments: dict[str, Any]) -> tuple[ArgumentNormalization, ...]:
        if self.parameter not in arguments:
            return ()
        original = arguments[self.parameter]
        normalized, changed = self.normalize_argument_value(original)
        if not changed:
            return ()
        arguments[self.parameter] = normalized
        return (ArgumentNormalization(self.parameter, original, normalized),)


@dataclass(frozen=True)
class MCPToolCompatibilityRule:
    """Compatibility profile selected by MCP tool and optional source label."""

    name: str
    tool_names: tuple[str, ...] = ()
    source_labels: tuple[str, ...] = ()
    transforms: tuple[MCPCompatibilityTransform, ...] = ()

    def matches(self, tool_name: str, source_label: str) -> bool:
        if self.tool_names and tool_name not in self.tool_names:
            return False
        return not (self.source_labels and source_label not in self.source_labels)


_TAVILY_SEARCH_DEPTH = EnumFieldCompatibility(
    parameter="search_depth",
    allowed_values=("basic", "advanced"),
    aliases={
        "fast": "basic",
        "ultra-fast": "basic",
    },
    default="basic",
    description=(
        "The depth of the search. Use 'basic' for generic/current lookups or "
        "'advanced' for more thorough search. Low-latency Tavily depths are "
        "disabled by this app because they are not compatible with all exposed "
        "Tavily arguments."
    ),
)


# Extension point for future MCP quirks:
#
# * Add provider-specific facts as scoped MCPToolCompatibilityRule data here.
# * Keep the generic MCP adapter free of provider branches.
# * Scope rules as narrowly as possible with tool_names and, when needed,
#   source_labels so unrelated servers with similar parameter names are not
#   changed accidentally.
_MCP_COMPATIBILITY_RULES: tuple[MCPToolCompatibilityRule, ...] = (
    MCPToolCompatibilityRule(
        name="tavily_search_depth",
        tool_names=("tavily_search",),
        transforms=(_TAVILY_SEARCH_DEPTH,),
    ),
)


def normalize_mcp_schema(
    tool_name: str,
    source_label: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Patch known MCP tool schemas before exposing them to the model."""
    for rule in _MCP_COMPATIBILITY_RULES:
        if not rule.matches(tool_name, source_label):
            continue
        for transform in rule.transforms:
            transform.normalize_schema(schema)
    return schema


def normalize_mcp_arguments(
    tool_name: str,
    source_label: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Apply known MCP argument compatibility guards before transport."""
    for rule in _MCP_COMPATIBILITY_RULES:
        if not rule.matches(tool_name, source_label):
            continue
        for transform in rule.transforms:
            for change in transform.normalize_arguments(arguments):
                logger.info(
                    "MCP_TOOL_ARGS_NORMALIZED rule=%s tool=%s source=%s field=%s "
                    "from=%s to=%s",
                    rule.name,
                    tool_name,
                    source_label,
                    change.field,
                    change.original,
                    change.normalized,
                )
    return arguments
