"""MCP (Model Context Protocol) toolset adapter.

Discovers tools from an MCP server at construction time, validates each
schema against OpenAI tool-calling constraints, and yields a list of
:class:`ResearchTool` instances usable from ``Agent(tools=[*toolset])``.

Key design points:

- Credentials live in :class:`MCPAuth` (threaded into the transport
  headers); they NEVER appear in :class:`ToolContext.extras`.
- ``$ref`` / ``$defs`` are inlined at construction time so the LLM sees
  fully-resolved schemas (some providers reject ``$ref`` at the root).
- ``ImageContent`` / ``EmbeddedResource`` parts are converted to
  ``ToolResult.data["mcp_attachments"]``; the LLM-facing ``content``
  carries a marker like ``[image attached: image/png, 12345 bytes]``.
- The MCP client itself is provided by the application — this module
  speaks to a duck-typed client with ``list_tools()`` and ``call_tool()``
  methods so callers can swap in real or test transports.

Phase 3 ships a synchronous discovery path. Streaming and notification
support are deferred.

Security model
--------------
``MCPToolset`` accepts a remote ``url=`` argument that is passed to the
underlying transport. Without validation this is an SSRF vector — a
caller can point the transport at internal services (loopback, link-local
IMDS, RFC1918 ranges) or non-HTTP schemes. PR3c hardens this entry point
with a deny-by-default URL validator (see :mod:`mcp_security` and
``docs/security/mcp-ssrf.md``).

Policy summary:

- Allowed schemes: ``http``, ``https`` only.
- Blocked IP ranges: loopback, private (RFC1918), link-local, CGNAT,
  multicast, reserved, IPv4-unspecified, IPv6 unspecified/ULA/link-local/
  site-local/multicast, IPv4-mapped IPv6 (``::ffff:0:0/96``), 6to4
  (``2002::/16``).
- DNS pinning: hostnames are resolved at validation time and substituted
  with the resolved IP literal in the URL handed to the transport. This
  eliminates DNS rebinding because the transport never re-resolves.
- Dev escape hatch: pass ``allowed_hosts=[...]`` (kwarg) or set
  ``DDR_MCP_ALLOWED_HOSTS=host1,host2`` (env). Hosts in the allowlist
  skip IP-range checks. Use sparingly; intended for local development
  against ``localhost``/``127.0.0.1`` MCP stubs.

Caller inventory (verified during PR3c planning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``api/__init__.py:68`` — re-export only (no construction).
- ``tests/unit/tools/test_mcp_auth.py:59`` — ``client=fake``.
- ``tests/unit/tools/test_mcp_discovery.py:38,50,60,67,76,92,103,110``
  — ``client=fake`` or expected to raise.
- ``tests/unit/tools/test_mcp_image_content.py:44,53,65,83`` —
  ``client=fake``.
- ``databricks-deep-research-app/`` — ZERO callers.

All test callers pass ``client=...`` (no URL), so SSRF validation only
triggers when ``url=`` is explicit. Existing tests are unaffected by
the validator wiring in PR3c.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Any, Literal, cast

from databricks_deep_research.tools.mcp_auth import MCPAuth
from databricks_deep_research.tools.mcp_security import (
    MCPSecurityError,
    validate_mcp_url,
)
from databricks_deep_research.tools.protocol import (
    ResearchTool,
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)


class MCPSchemaError(ValueError):
    """Raised when an MCP-discovered tool schema cannot be exposed to OpenAI."""


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve ``$ref``/``$defs`` references.

    Pure: does NOT mutate ``schema``. The caller may pass a cached/shared
    schema dict; this function reads ``$defs``/``definitions`` and returns
    a new dict with refs inlined and the defs section removed.
    """
    defs = schema.get("$defs") or schema.get("definitions") or {}
    # Build a copy of the top-level schema without the defs section so the
    # original dict is preserved for repeated calls.
    schema_without_defs = {
        k: v for k, v in schema.items() if k not in ("$defs", "definitions")
    }

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and len(node) == 1:
                key = node["$ref"].rsplit("/", 1)[-1]
                target = defs.get(key)
                if target is None:
                    return node
                return _resolve(target)
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return cast(dict[str, Any], _resolve(schema_without_defs))


def _validate_openai_compatible(schema: dict[str, Any]) -> None:
    """Reject schemas with features OpenAI tool calling does not support."""
    if "oneOf" in schema:
        raise MCPSchemaError("Top-level 'oneOf' is not supported by OpenAI tool calling.")
    if schema.get("type") and isinstance(schema["type"], list):
        raise MCPSchemaError(
            "Top-level multi-type unions are not supported by OpenAI tool calling."
        )
    # ``properties`` is required for object schemas; allow empty.
    if schema.get("type") == "object" and "properties" not in schema:
        schema["properties"] = {}


class _MCPTool:
    """Adapts a single discovered MCP tool to the :class:`ResearchTool` protocol."""

    def __init__(
        self,
        client: Any,
        name: str,
        original_name: str,
        input_schema: dict[str, Any],
        description: str,
    ) -> None:
        self._client = client
        self._original_name = original_name
        self._definition = ToolDefinition(
            name=name,
            description=description,
            parameters=input_schema,
            source_type="mcp",
            source_kind=SourceKind.builtin,
            metadata={"mcp_original_name": original_name},
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return dict(arguments or {})

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,  # noqa: ARG002 — protocol surface
    ) -> ToolResult:
        result = await self._call(self._original_name, arguments)
        return self._normalize_result(result)

    async def _call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        call = getattr(self._client, "call_tool", None)
        if call is None:
            raise RuntimeError("MCP client missing call_tool()")
        # Support sync + async call_tool implementations.
        try:
            response = call(tool_name, arguments)
        except TypeError:
            response = call(name=tool_name, arguments=arguments)
        if hasattr(response, "__await__"):
            response = await response
        return response

    def _normalize_result(self, mcp_result: Any) -> ToolResult:
        text_parts: list[str] = []
        attachments: list[dict[str, Any]] = []
        content_parts = getattr(mcp_result, "content", None) or []

        for part in content_parts:
            ptype = getattr(part, "type", None) or (
                part.get("type") if isinstance(part, dict) else None
            )
            if ptype == "text":
                text = (
                    getattr(part, "text", None)
                    if not isinstance(part, dict)
                    else part.get("text", "")
                )
                if text:
                    text_parts.append(str(text))
            elif ptype == "image":
                mime = getattr(part, "mimeType", None) or (
                    part.get("mimeType", "") if isinstance(part, dict) else ""
                )
                image_bytes = getattr(part, "data", None) or (
                    part.get("data", b"") if isinstance(part, dict) else b""
                )
                size = len(image_bytes) if isinstance(image_bytes, (bytes, bytearray)) else 0
                text_parts.append(f"[image attached: {mime}, {size} bytes]")
                attachments.append({
                    "kind": "image",
                    "mime_type": mime,
                    "bytes": bytes(image_bytes) if isinstance(image_bytes, (bytes, bytearray)) else None,
                })
            elif ptype == "resource":
                uri = getattr(part, "uri", None) or (
                    part.get("uri", "") if isinstance(part, dict) else ""
                )
                resource = getattr(part, "resource", None) or (
                    part.get("resource") if isinstance(part, dict) else None
                )
                text_parts.append(f"[resource attached: {uri}]")
                attachments.append({
                    "kind": "resource",
                    "uri": uri,
                    "data": resource,
                })

        data: dict[str, Any] = {}
        if attachments:
            data["mcp_attachments"] = attachments

        return ToolResult(
            content="\n".join(text_parts),
            success=not getattr(mcp_result, "isError", False),
            data=data,
        )


class MCPToolset:
    """Adapter that exposes an MCP server's tools as :class:`ResearchTool` instances.

    Args:
        url: MCP endpoint URL. When set, the URL is run through
            :func:`databricks_deep_research.tools.mcp_security.validate_mcp_url`
            before the transport is constructed: non-``http(s)`` schemes,
            blocked IP ranges (loopback, private, link-local, CGNAT,
            multicast, IPv4-mapped/6to4 v6), and DNS-rebinding-prone
            hostnames are rejected with :class:`MCPSecurityError`.
            Hostnames that resolve to public IPs are rewritten to the
            resolved IP literal (pinned-IP) so the transport never
            re-resolves.
        transport: ``"http"`` (default), ``"stdio"``, or ``"sse"``.
        allow: Optional whitelist of tool names. When set, only these
            discovered tools are exposed.
        deny: Optional blacklist of tool names. Always applied after
            ``allow``.
        name_prefix: Prepended to each exposed tool name to namespace.
        auth: Optional :class:`MCPAuth` strategy for transport headers.
        client: Pre-built MCP client (overrides ``url``/``transport``).
            Useful for tests and dependency injection. When ``client`` is
            provided the SSRF validator does NOT run.
        allowed_hosts: Dev escape hatch — hostnames in this list (case
            insensitive) bypass the IP-range check, e.g. ``["localhost"]``
            for a local MCP stub. The env var ``DDR_MCP_ALLOWED_HOSTS``
            (comma-separated) supplies the same list when the kwarg is
            ``None``. See ``docs/security/mcp-ssrf.md`` for the full
            policy.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        transport: Literal["http", "stdio", "sse"] = "http",
        allow: list[str] | None = None,
        deny: list[str] | None = None,
        name_prefix: str = "",
        auth: MCPAuth | None = None,
        client: Any | None = None,
        allowed_hosts: list[str] | None = None,
    ) -> None:
        # Resolve the effective allowlist: explicit kwarg wins, else fall
        # back to the env var. Normalise to ``None`` when the result is
        # empty so downstream allowlist logic can ``if allowed_hosts:``.
        effective_hosts: list[str] | None = allowed_hosts
        if effective_hosts is None:
            env_raw = os.environ.get("DDR_MCP_ALLOWED_HOSTS", "")
            env_hosts = [h.strip() for h in env_raw.split(",") if h.strip()]
            effective_hosts = env_hosts or None

        validated_url = url
        if url is not None:
            validated_url = validate_mcp_url(url, allowed_hosts=effective_hosts)

        self._url = validated_url
        self._transport = transport
        self._auth = auth
        self._client = client or self._build_client(validated_url, transport, auth)
        self._tools: list[_MCPTool] = list(self._discover(allow, deny, name_prefix))

    @staticmethod
    def _build_client(url: str | None, transport: str, auth: MCPAuth | None) -> Any:
        """Instantiate an MCP client. Defers the heavy lift to the optional ``mcp`` SDK.

        Tests typically pass a pre-built ``client`` so this is rarely hit.
        """
        if url is None:
            raise ValueError("MCPToolset requires either url= or client=")
        try:
            from mcp.client.session import ClientSession  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "MCPToolset requires the `mcp` package; install via `pip install mcp`."
            ) from exc
        # Real client construction is transport-specific; the user usually
        # provides the client directly. The default path is intentionally
        # minimal so tests don't need to spin up a real transport.
        headers = auth.headers() if auth is not None else {}
        return ClientSession(url=url, transport=transport, headers=headers)

    def _discover(
        self,
        allow: list[str] | None,
        deny: list[str] | None,
        name_prefix: str,
    ) -> Iterator[_MCPTool]:
        list_tools = getattr(self._client, "list_tools", None)
        if list_tools is None:
            raise RuntimeError("MCP client missing list_tools()")
        discovered = list_tools()
        # Some clients return iterables; others return a response object with `.tools`.
        items = getattr(discovered, "tools", None) or discovered
        allow_set = set(allow) if allow is not None else None
        deny_set = set(deny) if deny is not None else set()

        for item in items:
            tool_name = getattr(item, "name", None) or (
                item.get("name") if isinstance(item, dict) else None
            )
            if not tool_name:
                continue
            if allow_set is not None and tool_name not in allow_set:
                continue
            if tool_name in deny_set:
                continue

            input_schema = getattr(item, "inputSchema", None) or (
                item.get("inputSchema") if isinstance(item, dict) else None
            ) or {"type": "object", "properties": {}}
            description = getattr(item, "description", None) or (
                item.get("description", "") if isinstance(item, dict) else ""
            )
            try:
                inlined = _inline_refs(dict(input_schema))
                _validate_openai_compatible(inlined)
            except MCPSchemaError:
                logger.warning(
                    "MCP_TOOL_SKIPPED name=%s reason=schema_invalid", tool_name
                )
                continue

            exposed_name = f"{name_prefix}{tool_name}" if name_prefix else tool_name
            yield _MCPTool(
                client=self._client,
                name=exposed_name,
                original_name=tool_name,
                input_schema=inlined,
                description=description,
            )

    @property
    def tools(self) -> list[ResearchTool]:
        return list(self._tools)

    def __iter__(self) -> Iterator[ResearchTool]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)


__all__ = ["MCPSchemaError", "MCPSecurityError", "MCPToolset"]
