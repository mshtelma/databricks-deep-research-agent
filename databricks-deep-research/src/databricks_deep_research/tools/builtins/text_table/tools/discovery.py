"""TableDiscoveryTool — list tables and register them as discovered bindings."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

from databricks_deep_research.tools.protocol import (
    ToolContext,
    ToolDefinition,
    ToolResult,
)

from ..binding import BindingInfo, BindingSource
from ..discovery_provider import TableDiscoveryProvider
from ..error_codes import ErrorCode, ToolError, ToolErrorException
from ..registry import TableBindingRegistry
from ..schema_cache import Schema, SchemaCache
from ..sql_compiler import compile_select
from ._common import SqlExecutor, get_user_token

__all__ = ["TableDiscoveryTool"]

logger = logging.getLogger(__name__)

_DETAIL_VALUES = ("basic", "schema", "full")
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)"
)
_MAX_SAMPLE_VALUE_CHARS = 256


def _error_result(error: ToolError) -> ToolResult:
    return ToolResult(
        content=json.dumps({"error": error.to_dict()}),
        success=False,
        error=str(error.error_code),
        data={"error": error.to_dict()},
    )


def _redact_sample_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = _EMAIL_RE.sub("[redacted-email]", value)
    text = _SSN_RE.sub("[redacted-ssn]", text)
    text = _PHONE_RE.sub("[redacted-phone]", text)
    if len(text) > _MAX_SAMPLE_VALUE_CHARS:
        return text[: _MAX_SAMPLE_VALUE_CHARS - 3] + "..."
    return text


class TableDiscoveryTool:
    """List tables matching an optional pattern and register them as discovered.

    The provider is optional — when ``None``, every call returns a
    ``DISCOVERY_UNAVAILABLE`` error result (no exceptions). This lets the
    tool be unconditionally constructed by factories even when discovery
    is not configured.
    """

    def __init__(
        self,
        *,
        provider: TableDiscoveryProvider | None,
        registry: TableBindingRegistry,
        schema_cache: SchemaCache | None = None,
        sql_executor: SqlExecutor | None = None,
        name: str = "table_discovery",
        description: str | None = None,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._schema_cache = schema_cache
        self._sql_executor = sql_executor
        self._name = name
        self._description = description

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description or (
                "List tables exposed to this agent. Optionally filter by "
                "substring and request additional schema/full detail. "
                "Discovered tables are registered into the binding registry "
                "and can be used with table_search/read/etc."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name_pattern": {
                        "type": "string",
                        "description": (
                            "Optional substring to filter table names "
                            "(case-insensitive)."
                        ),
                    },
                    "detail": {
                        "type": "string",
                        "enum": list(_DETAIL_VALUES),
                        "default": "basic",
                        "description": (
                        "basic = name + description; schema = also "
                        "returns column types; full = schema + sample "
                            "row probe."
                        ),
                    },
                },
                "required": [],
            },
            source_type="builtin",
            source_kind="text_table",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        name_pattern = arguments.get("name_pattern")
        if name_pattern is not None and not isinstance(name_pattern, str):
            raise ValueError("'name_pattern' must be a string when provided")

        detail = arguments.get("detail", "basic")
        if not isinstance(detail, str) or detail not in _DETAIL_VALUES:
            raise ValueError(
                f"'detail' must be one of {_DETAIL_VALUES}; got {detail!r}"
            )
        return {"name_pattern": name_pattern, "detail": detail}

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        name_pattern: str | None = arguments.get("name_pattern")
        detail: str = arguments.get("detail", "basic")

        if self._provider is None:
            return _error_result(
                ToolError(
                    error_code=ErrorCode.DISCOVERY_UNAVAILABLE,
                    message=(
                        "no TableDiscoveryProvider configured for this tool; "
                        "table discovery is not available in this workspace"
                    ),
                    hint=(
                        "Wire a TableDiscoveryProvider implementation when "
                        "constructing TableDiscoveryTool, or use the BOUND "
                        "bindings already registered in YAML."
                    ),
                )
            )

        if detail in ("schema", "full") and self._schema_cache is None:
            return _error_result(
                ToolError(
                    error_code=ErrorCode.DISCOVERY_UNAVAILABLE,
                    message=(
                        f"detail={detail!r} requires a SchemaCache, but none "
                        "was wired into TableDiscoveryTool"
                    ),
                    hint="Pass schema_cache=... at tool construction.",
                )
            )

        user_token = get_user_token(context.extras)

        try:
            discovered = await self._provider.list_tables(
                user_token=user_token, name_pattern=name_pattern
            )
        except ToolErrorException as exc:
            return _error_result(exc.error)
        except Exception as exc:  # noqa: BLE001 — surface as ToolError
            return _error_result(
                ToolError(
                    error_code=ErrorCode.DISCOVERY_UNAVAILABLE,
                    message=f"discovery provider raised: {exc!r}",
                    details={"exception": type(exc).__name__},
                )
            )

        items: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        for info in discovered:
            if info.source is not BindingSource.DISCOVERED:
                # Coerce to DISCOVERED — providers may return BOUND-shaped
                # records; we never trust them as BOUND at runtime.
                info = BindingInfo(
                    name=info.name,
                    fqn=info.fqn,
                    source=BindingSource.DISCOVERED,
                    description=info.description,
                    roles=info.roles,
                    numeric_columns=info.numeric_columns,
                    structured_passages=info.structured_passages,
                    verbose=info.verbose,
                )
            canonical, warning = self._registry.register_discovered(info)
            if warning is not None:
                warnings.append(warning.to_dict())
            entry: dict[str, Any] = {
                "name": canonical,
                "fqn": info.fqn,
                "description": info.description,
            }
            schema: Schema | None = None
            if detail in ("schema", "full"):
                schema_result = self._schema(info, user_token)
                if isinstance(schema_result, ToolError):
                    warnings.append(schema_result.to_dict())
                else:
                    schema = schema_result
                    entry["schema"] = self._schema_payload(schema)
            if detail == "full" and schema is not None:
                sample = self._sample_payload(info, schema, user_token)
                if isinstance(sample, ToolError):
                    warnings.append(sample.to_dict())
                    entry["sample"] = []
                else:
                    entry["sample"] = sample
            items.append(entry)

        payload: dict[str, Any] = {"tables": items}
        if warnings:
            payload["warnings"] = warnings
        return ToolResult(
            content=json.dumps(payload),
            data=payload,
        )

    def _schema(self, info: BindingInfo, user_token: str) -> Schema | ToolError:
        cache = self._schema_cache
        assert cache is not None  # gate-checked above
        try:
            return cache.get(info.fqn, user_token)
        except ToolErrorException as exc:
            return exc.error
        except Exception as exc:  # noqa: BLE001 — degrade per-table
            return ToolError(
                error_code=ErrorCode.SCHEMA_FETCH_FAILED,
                message=f"failed to fetch schema for {info.fqn}: {exc!r}",
                binding=info.name,
                details={"fqn": info.fqn},
            )

    @staticmethod
    def _schema_payload(schema: Schema) -> list[dict[str, Any]]:
        return [
            {
                "name": col.name,
                "data_type": col.data_type,
                "nullable": col.nullable,
            }
            for col in schema.columns
        ]

    def _sample_payload(
        self, info: BindingInfo, schema: Schema, user_token: str
    ) -> list[dict[str, Any]] | ToolError:
        if self._sql_executor is None:
            return ToolError(
                error_code=ErrorCode.DISCOVERY_UNAVAILABLE,
                message=(
                    f"detail='full' could not sample {info.fqn}: no SQL executor "
                    "was wired into TableDiscoveryTool"
                ),
                binding=info.name,
                details={"fqn": info.fqn},
            )
        try:
            sql, params = compile_select(info.fqn, schema, limit=1)
            rows = self._sql_executor(sql, params, user_token)
        except ToolErrorException as exc:
            return exc.error
        except Exception as exc:  # noqa: BLE001 — degrade per-table
            return ToolError(
                error_code=ErrorCode.DISCOVERY_UNAVAILABLE,
                message=f"failed to sample {info.fqn}: {exc!r}",
                binding=info.name,
                details={"fqn": info.fqn, "exception": type(exc).__name__},
            )
        return [
            {key: _redact_sample_value(value) for key, value in row.items()}
            for row in rows[:1]
        ]

    # -- ComputeCallableProvider --------------------------------------------

    @property
    def compute_name(self) -> str:
        return "table_discovery"

    def to_compute_callable(
        self, *, compute: Any
    ) -> Callable[..., list[dict[str, Any]]]:
        """Return a synchronous callable usable inside the compute sandbox.

        The callable accepts ``user_token`` (required), an optional
        ``name_pattern``, and an optional ``detail`` arg. It runs the
        provider, registers discovered bindings, and returns a plain list
        of dicts. Errors raise ``ToolErrorException`` directly — no JSON
        envelope.
        """
        del compute
        provider = self._provider
        registry = self._registry
        schema_cache = self._schema_cache
        sql_executor = self._sql_executor

        def _call(
            *,
            user_token: str = "",
            name_pattern: str | None = None,
            detail: str = "basic",
        ) -> list[dict[str, Any]]:
            if detail not in _DETAIL_VALUES:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.INVALID_FILTER,
                        message=(
                            f"detail must be one of {_DETAIL_VALUES}; "
                            f"got {detail!r}"
                        ),
                    )
                )
            if provider is None:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.DISCOVERY_UNAVAILABLE,
                        message=(
                            "no TableDiscoveryProvider configured for this tool"
                        ),
                    )
                )
            if detail in ("schema", "full") and schema_cache is None:
                raise ToolErrorException(
                    ToolError(
                        error_code=ErrorCode.DISCOVERY_UNAVAILABLE,
                        message=(
                            f"detail={detail!r} requires a SchemaCache, "
                            "but none was wired into TableDiscoveryTool"
                        ),
                    )
                )

            import asyncio

            discovered = asyncio.run(
                provider.list_tables(
                    user_token=user_token, name_pattern=name_pattern
                )
            )

            items: list[dict[str, Any]] = []
            for info in discovered:
                if info.source is not BindingSource.DISCOVERED:
                    info = BindingInfo(
                        name=info.name,
                        fqn=info.fqn,
                        source=BindingSource.DISCOVERED,
                        description=info.description,
                        roles=info.roles,
                        numeric_columns=info.numeric_columns,
                        structured_passages=info.structured_passages,
                        verbose=info.verbose,
                    )
                canonical, _warning = registry.register_discovered(info)
                entry: dict[str, Any] = {
                    "name": canonical,
                    "fqn": info.fqn,
                    "description": info.description,
                }
                if detail in ("schema", "full") and schema_cache is not None:
                    payload = self._schema(info, user_token)
                    if not isinstance(payload, ToolError):
                        entry["schema"] = self._schema_payload(payload)
                        if detail == "full" and sql_executor is not None:
                            sample = self._sample_payload(info, payload, user_token)
                            if not isinstance(sample, ToolError):
                                entry["sample"] = sample
                items.append(entry)
            return items

        return _call
