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

import asyncio
import inspect
import logging
import os
from collections.abc import Callable, Iterator
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from databricks_deep_research.tools.mcp_auth import ApiKey, BearerToken, MCPAuth
from databricks_deep_research.tools.mcp_security import (
    MCPSecurityError,
    validate_mcp_url,
)
from databricks_deep_research.tools.protocol import (
    ResearchTool,
    SourceInfo,
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)

# Default source kind for MCP research tools. ``qa_assistant`` (NL question →
# NL answer) is the closest match to an MCP tool's prose output AND — crucially
# for citeability (spec §4.3 #11) — it is a NON-``builtin`` kind, so the ReAct
# loop routes MCP results through ``admit_tool_result`` (admission + pool
# write) instead of the builtin bypass that drops every source. See
# ``_MCPTool`` for the per-result ``SourceInfo`` that becomes the citeable
# pool entry.
_DEFAULT_MCP_SOURCE_KIND: str = SourceKind.qa_assistant
# Synthetic URL scheme for MCP-sourced evidence. The framework's admission
# layer (``source_aware._normalize_source``) requires a non-empty ``url`` to
# admit a source; the LLM never sees raw URLs (UrlRegistry maps indices), and
# this scheme namespaces MCP evidence away from real web URLs.
_MCP_SOURCE_SCHEME: str = "mcp"

# Callable that maps a secret reference (``scope/key``) to its resolved
# credential value. Hosts supply a Databricks-secret-scope-backed
# implementation; the framework never reads secrets directly.
SecretResolver = Callable[[str], "str | None"]


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
    """Adapts a single discovered MCP tool to the :class:`ResearchTool` protocol.

    Citeability (spec §4.3 #11): the tool's ``source_kind`` defaults to
    ``qa_assistant`` (a NON-``builtin`` kind), so the ReAct loop sends its
    results through ``admit_tool_result`` rather than the builtin bypass that
    silently drops every source. ``_normalize_result`` emits a
    :class:`SourceInfo` carrying the tool's text answer plus a synthetic
    ``mcp://`` URL, which becomes the citeable pool entry. Pass
    ``source_kind=SourceKind.builtin`` to opt a side-effecting MCP tool OUT of
    evidence admission (its results then behave like ``compute`` — visible to
    the model, never cited).
    """

    def __init__(
        self,
        client: Any,
        name: str,
        original_name: str,
        input_schema: dict[str, Any],
        description: str,
        *,
        source_kind: str = _DEFAULT_MCP_SOURCE_KIND,
        source_label: str = "",
    ) -> None:
        self._client = client
        self._original_name = original_name
        self._source_kind = source_kind
        # Stable, human-readable label for the synthetic citeable source (e.g.
        # the server's name_prefix). Falls back to the exposed tool name.
        self._source_label = source_label or name
        # Synthetic source URL — opaque, namespaced, deterministic per tool.
        # Never a real network address (the transport URL is SSRF-validated
        # separately); this only exists so admission can register a citeable
        # source for the tool's prose answer.
        self._source_url = f"{_MCP_SOURCE_SCHEME}://{self._source_label}/{original_name}"
        # ``tool_source_kind`` (source_aware) prefers an explicit ``source_kind``
        # ONLY when it is non-``builtin``; for the builtin opt-out it falls back
        # to heuristics over ``source_type``. So a ``builtin`` MCP tool must also
        # carry ``source_type="builtin"`` for the ReAct loop to actually bypass
        # admission (behave like ``compute``). Citeable tools keep ``mcp``.
        source_type = "builtin" if source_kind == SourceKind.builtin else "mcp"
        self._definition = ToolDefinition(
            name=name,
            description=description,
            parameters=input_schema,
            source_type=source_type,
            source_kind=source_kind,
            metadata={
                "mcp_original_name": original_name,
                "source_name": self._source_label,
                "source_url": self._source_url,
            },
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

        content = "\n".join(text_parts)
        success = not getattr(mcp_result, "isError", False)

        # Citeable-source emission (spec §4.3 #11). For a citeable MCP tool
        # (non-``builtin`` source_kind) that returned real text, attach a
        # SourceInfo so ``admit_tool_result`` registers the answer as evidence
        # that reaches the pool. ``builtin`` MCP tools (opt-out, side-effecting)
        # emit no source — they bypass admission like ``compute``.
        sources: list[SourceInfo] = []
        if (
            success
            and content.strip()
            and self._source_kind != SourceKind.builtin
        ):
            sources.append(
                SourceInfo(
                    url=self._source_url,
                    title=self._source_label,
                    snippet=content[:800],
                    content=content,
                    source_type="mcp",
                    source_kind=self._source_kind,
                )
            )

        return ToolResult(
            content=content,
            success=success,
            sources=sources,
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
        source_kind: :class:`SourceKind` value applied to every discovered
            tool. Defaults to ``qa_assistant`` so MCP results are CITEABLE
            (admitted as evidence). Pass ``SourceKind.builtin`` to opt the
            whole server's tools out of evidence admission.
        source_label: Human-readable label for synthetic citeable sources;
            defaults to ``name_prefix`` (trimmed) or the tool name.
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
        source_kind: str = _DEFAULT_MCP_SOURCE_KIND,
        source_label: str = "",
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
        self._source_kind = source_kind
        self._source_label = source_label or name_prefix.rstrip("_") or ""
        self._client = client or self._build_client(validated_url, transport, auth)
        self._tools: list[_MCPTool] = list(self._discover(allow, deny, name_prefix))

    @staticmethod
    def _build_client(url: str | None, transport: str, auth: MCPAuth | None) -> Any:
        """Instantiate an MCP client. Defers the heavy lift to the optional ``mcp`` SDK.

        Real callers (and tests) pass a pre-built ``client=`` — e.g. the app's
        ``DatabricksMCPClient``. A real HTTP/SSE MCP client needs an async
        transport connection (``streamablehttp_client`` / ``sse_client``) that
        this synchronous factory cannot establish, so the bare-URL path fails
        closed with a clear error instead of half-building a session. (The
        ``mcp`` SDK's ``ClientSession`` takes read/write streams, not a URL.)
        """
        if url is None:
            raise ValueError("MCPToolset requires either url= or client=")
        import importlib.util

        if importlib.util.find_spec("mcp") is None:
            raise RuntimeError(
                "MCPToolset requires the optional `mcp` package, which is not "
                "installed. Install the MCP extra: `pip install "
                "databricks-deep-research-app[mcp]` (or `pip install mcp`). "
                "The default research path never imports `mcp`."
            )
        raise NotImplementedError(
            f"MCPToolset cannot build an HTTP MCP client for {url!r} "
            f"(transport={transport!r}, auth="
            f"{'set' if auth is not None else 'unset'}) from a URL alone — "
            "pass a transport-specific client= instead."
        )

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
        # The Databricks MCP client exposes a *synchronous* ``list_tools`` that
        # internally drives an async ``_get_tools_async`` on its own event loop;
        # other clients may expose ``list_tools`` as a coroutine function. When
        # the call returns a coroutine, run it to completion here. Hosts that
        # invoke discovery from within a running event loop MUST do so off-loop
        # (e.g. ``asyncio.to_thread(build_mcp_toolsets, ...)``) so this
        # ``asyncio.run`` — and the SDK's own internal loop — never nests inside a
        # live loop (which would raise and leave the coroutine un-awaited).
        if inspect.iscoroutine(discovered):
            discovered = asyncio.run(discovered)
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
                source_kind=self._source_kind,
                source_label=self._source_label,
            )

    @property
    def tools(self) -> list[ResearchTool]:
        return list(self._tools)

    @property
    def source_label(self) -> str:
        """The server label (set from ``MCPServerConfig.name`` by
        :func:`build_mcp_toolset`). Lets a host map a built toolset back to its
        configured server name without reaching into private state — used to
        attach a server's tools to the agents that bind it via
        ``AgentNodeConfig.mcp_servers``."""
        return self._source_label

    def __iter__(self) -> Iterator[ResearchTool]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)


# ---------------------------------------------------------------------------
# Declarative server config + runtime builder
# ---------------------------------------------------------------------------


class MCPServerConfig(BaseModel):
    """Declarative config for ONE remote MCP server, persisted on a workflow.

    Mirrors :class:`CustomToolDef` — a user-authored tool config stored on the
    agent/workflow definition. Optional-with-default: a workflow with no
    ``mcp_servers`` declares zero servers and behaves exactly as before.

    Security:

    * Only stateless HTTP/SSE transports are accepted (no stdio session pool).
    * Credentials are supplied ONLY via ``secret_ref`` — a Databricks
      secret-scope reference (``scope/key`` or ``{{secrets/scope/key}}``).
      An inline token is rejected at validation time and is NEVER logged.
    * The ``url`` is SSRF-validated by :class:`MCPToolset` at build time.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    """Unique server name; used as the source label and default name prefix."""

    url: str = Field(default="")
    """Remote MCP endpoint (http/https). SSRF-validated when the toolset is built.

    Required for ``client_kind='http'``. For ``client_kind='databricks'`` it is
    DERIVED at runtime (host + allowlisted path) and any persisted value is
    IGNORED — never trusted — so a tampered host cannot redirect the client."""

    client_kind: Literal["http", "databricks"] = "http"
    """Transport client. ``http`` (default): a stateless HTTP/SSE MCP server at
    ``url``. ``databricks``: a Databricks managed / UC-connection MCP server
    reached via ``DatabricksMCPClient`` under the caller's OBO identity; its URL
    is derived from the workspace host + ``connection_name`` / ``managed_target``."""

    connection_name: str = ""
    """For ``client_kind='databricks'`` EXTERNAL MCP: the Unity Catalog
    connection name backing the proxy at ``/api/2.0/mcp/external/{name}``."""

    managed_target: str = ""
    """For ``client_kind='databricks'`` MANAGED MCP: the path suffix after
    ``/api/2.0/mcp/`` — e.g. ``functions/{catalog}/{schema}``,
    ``vector-search/{catalog}/{schema}``, or ``genie/{space_id}``. Validated
    against an allowlist of patterns by the app adapter before use."""

    transport: Literal["http", "sse"] = "http"
    """Stateless transport only. ``stdio`` is intentionally unsupported."""

    auth_type: Literal["none", "bearer", "api_key"] = "none"
    """Auth strategy. ``bearer``/``api_key`` require ``secret_ref``."""

    secret_ref: str | None = None
    """Databricks secret-scope reference resolving to the credential value.

    NEVER an inline credential. Format: ``scope/key`` or
    ``{{secrets/scope/key}}``. Resolved at runtime by the host's
    secret resolver; the resolved value is threaded into transport headers
    and never logged.
    """

    api_key_header: str = "X-API-Key"
    """Header name used when ``auth_type='api_key'``."""

    allow: list[str] | None = None
    """Optional allowlist of tool names to expose from this server."""

    deny: list[str] | None = None
    """Optional denylist of tool names; applied after ``allow``."""

    name_prefix: str = ""
    """Optional prefix namespacing this server's tool names."""

    strategy: Literal["fast", "deep"] = "fast"
    """Discovery strategy. ``fast``: discover once and cache (default).
    ``deep``: re-discover per research step (the host controls cadence)."""

    citeable: bool = True
    """When True (default) the server's tools are admitted as evidence
    (``qa_assistant`` source kind). When False they bypass admission like
    ``compute`` (visible to the model, never cited)."""

    @model_validator(mode="after")
    def _validate_auth(self) -> MCPServerConfig:
        # client_kind / target coherence (B1). ``http`` needs a literal url;
        # ``databricks`` derives its url at runtime and needs EXACTLY ONE of
        # connection_name (external) or managed_target (managed).
        if self.client_kind == "http":
            if not self.url:
                raise ValueError(
                    f"mcp server {self.name!r}: client_kind='http' requires a 'url'"
                )
        else:  # databricks
            has_conn = bool(self.connection_name.strip())
            has_managed = bool(self.managed_target.strip())
            if has_conn == has_managed:
                raise ValueError(
                    f"mcp server {self.name!r}: client_kind='databricks' requires "
                    "exactly one of 'connection_name' (external) or "
                    "'managed_target' (managed)"
                )
        # Secret-ref-only credentials. An auth strategy that needs a credential
        # MUST point at a secret scope; a bare/inline token is rejected so a
        # credential can never be persisted in the AST or surface in a log.
        if self.auth_type in ("bearer", "api_key") and not self.secret_ref:
            raise ValueError(
                f"mcp server {self.name!r}: auth_type={self.auth_type!r} requires "
                "a secret_ref (Databricks secret-scope reference); inline "
                "credentials are not permitted."
            )
        return self

    def source_kind(self) -> str:
        """Resolve the SourceKind for this server's tools (citeable vs builtin)."""
        return _DEFAULT_MCP_SOURCE_KIND if self.citeable else SourceKind.builtin


def build_mcp_auth(
    config: MCPServerConfig,
    secret_resolver: SecretResolver | None,
) -> MCPAuth | None:
    """Build the :class:`MCPAuth` strategy for *config*, resolving its secret.

    ``secret_resolver`` maps a ``secret_ref`` to its credential value (the host
    injects a Databricks-secret-scope-backed resolver). Returns ``None`` for
    ``auth_type='none'``. The resolved credential is NEVER logged.

    Raises:
        ValueError: If a credential is required but no resolver / value is available.
    """
    if config.auth_type == "none":
        return None
    if config.secret_ref is None:  # pragma: no cover — guarded by model validator
        raise ValueError(f"mcp server {config.name!r}: missing secret_ref")
    if secret_resolver is None:
        raise ValueError(
            f"mcp server {config.name!r}: auth_type={config.auth_type!r} needs a "
            "secret resolver but none was provided"
        )
    secret_value = secret_resolver(config.secret_ref)
    if not secret_value:
        raise ValueError(
            f"mcp server {config.name!r}: secret_ref resolved to an empty value"
        )
    if config.auth_type == "bearer":
        return BearerToken(token=secret_value)
    return ApiKey(header=config.api_key_header, value=secret_value)


def build_mcp_toolset(
    config: MCPServerConfig,
    *,
    secret_resolver: SecretResolver | None = None,
    client: Any | None = None,
    allowed_hosts: list[str] | None = None,
) -> MCPToolset:
    """Construct one :class:`MCPToolset` from a declarative :class:`MCPServerConfig`.

    SSRF validation and tool discovery happen inside the toolset constructor.
    ``client`` is injectable for tests (no network, skips SSRF as documented on
    :class:`MCPToolset`). Auth headers are derived from ``secret_ref`` via
    ``secret_resolver`` and never logged.
    """
    auth = build_mcp_auth(config, secret_resolver)
    logger.info(
        "MCP_TOOLSET_BUILD name=%s transport=%s auth_type=%s strategy=%s "
        "citeable=%s has_secret_ref=%s name_prefix=%s",
        config.name,
        config.transport,
        config.auth_type,
        config.strategy,
        config.citeable,
        config.secret_ref is not None,
        config.name_prefix,
    )
    return MCPToolset(
        url=None if client is not None else config.url,
        transport=config.transport,
        allow=config.allow,
        deny=config.deny,
        name_prefix=config.name_prefix,
        auth=auth,
        client=client,
        allowed_hosts=allowed_hosts,
        source_kind=config.source_kind(),
        source_label=config.name,
    )


__all__ = [
    "MCPSchemaError",
    "MCPSecurityError",
    "MCPServerConfig",
    "MCPToolset",
    "SecretResolver",
    "build_mcp_auth",
    "build_mcp_toolset",
]
