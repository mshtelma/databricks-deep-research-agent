"""App-side adapter: build per-request MCP toolsets with OBO identity.

Bridges the framework's declarative :class:`MCPServerConfig` (persisted on a
workflow's ``mcp_servers``) to live :class:`MCPToolset` instances, resolving
credentials from Databricks secret scopes ON BEHALF OF the calling user.

Design (mirrors the per-agent ``domain_filter`` runtime-override precedent in
``framework_orchestrator``):

* One toolset is built PER REQUEST so each run gets fresh discovery and the
  caller's identity (never the app service principal — see
  ``project-shell-app-table-tools-run-as-sp``).
* Secrets are read via ``WorkspaceClient.secrets.get_secret`` using the SAME
  OBO client the rest of the request uses; resolved values are threaded into
  the MCP transport auth headers and NEVER logged.
* The discovered tools are injected through ``ToolResolver.override`` by the
  caller — they are not resolved through the static factory chain.

Optional-with-default: an empty ``mcp_servers`` list yields zero toolsets and
the request behaves exactly as before.
"""

from __future__ import annotations

import base64
import logging
import re
from typing import Any

from databricks_deep_research import (
    MCPServerConfig,
    MCPToolset,
    SecretResolver,
    build_mcp_toolset,
    resolve_workspace_client,
)
from databricks_deep_research.tools.mcp_security import MCPSecurityError

logger = logging.getLogger(__name__)

# A Unity Catalog identifier segment (catalog / schema / connection / space id).
_UC_IDENT = re.compile(r"^[A-Za-z0-9_]+$")

# A Databricks secret-scope / key segment: alphanumerics + ``-``/``_``/``.`` only
# (HIGH-1 — rejects newlines, URL-encoding, and path traversal in a secret_ref).
_SECRET_REF_SEGMENT = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")

# Allowlisted managed-MCP path suffixes (after ``/api/2.0/mcp/``). The persisted
# value is matched against these patterns before the URL is built, so a tampered
# ``managed_target`` cannot redirect the request off the managed-MCP namespace.
_MANAGED_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^functions/[A-Za-z0-9_]+/[A-Za-z0-9_]+$"),
    re.compile(r"^vector-search/[A-Za-z0-9_]+/[A-Za-z0-9_]+$"),
    re.compile(r"^genie/[A-Za-z0-9_]+$"),
)


class MCPConfigError(ValueError):
    """Raised when a Databricks MCP server config cannot be safely resolved."""


def _derive_databricks_mcp_url(host: str, server: MCPServerConfig) -> str:
    """Derive a Databricks MCP URL from the live *host* + an allowlisted path.

    The persisted ``server.url`` is intentionally ignored — the URL is built only
    from the trusted workspace host and a validated ``connection_name`` /
    ``managed_target``, so a tampered persisted host cannot redirect the client
    (Codex review CRIT6). Raises :class:`MCPConfigError` on an invalid target.
    """
    base = (host or "").rstrip("/")
    if not base:
        raise MCPConfigError(
            f"mcp server {server.name!r}: cannot derive a Databricks MCP URL "
            "(workspace host unavailable)"
        )
    if server.connection_name.strip():
        conn = server.connection_name.strip()
        if not _UC_IDENT.match(conn):
            raise MCPConfigError(
                f"mcp server {server.name!r}: invalid connection_name {conn!r}"
            )
        return f"{base}/api/2.0/mcp/external/{conn}"
    target = server.managed_target.strip().strip("/")
    if not any(pattern.match(target) for pattern in _MANAGED_TARGET_PATTERNS):
        raise MCPConfigError(
            f"mcp server {server.name!r}: unsupported managed_target "
            f"{server.managed_target!r} (expected functions/<cat>/<schema>, "
            "vector-search/<cat>/<schema>, or genie/<space_id>)"
        )
    return f"{base}/api/2.0/mcp/{target}"


def _build_databricks_mcp_client(url: str, workspace_client: Any) -> Any:
    """Construct a ``DatabricksMCPClient`` (guarded import) for *url* under OBO.

    ``databricks-mcp`` is an optional dependency; when it is not installed the
    Databricks-MCP path fails closed with a clear error rather than silently
    degrading. The client is duck-compatible with the framework's
    ``MCPToolset(client=...)`` (``list_tools`` / ``call_tool``).
    """
    try:
        from databricks_mcp import DatabricksMCPClient
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise MCPConfigError(
            "Databricks MCP requires the 'databricks-mcp' package, which is not "
            "installed in this environment"
        ) from exc
    return DatabricksMCPClient(server_url=url, workspace_client=workspace_client)


def _parse_secret_ref(secret_ref: str) -> tuple[str, str]:
    """Parse a ``scope/key`` (or ``{{secrets/scope/key}}``) reference.

    Returns ``(scope, key)``. Raises ``ValueError`` on a malformed ref.
    """
    ref = secret_ref.strip()
    # Accept the Databricks templated form ``{{secrets/scope/key}}``.
    if ref.startswith("{{") and ref.endswith("}}"):
        inner = ref[2:-2].strip()
        if inner.startswith("secrets/"):
            inner = inner[len("secrets/"):]
        ref = inner
    parts = [p for p in ref.split("/") if p]
    if len(parts) != 2:
        raise ValueError(
            "mcp secret_ref must be 'scope/key' (or '{{secrets/scope/key}}'); "
            f"got {secret_ref!r}"
        )
    scope, key = parts
    # HIGH-1 (security review): enforce a strict character-class allowlist on both
    # segments. Without it a crafted ref could smuggle a newline (HTTP header
    # injection into the secrets API call) or URL-encoded path traversal
    # (``..%2F``) into the scope/key. Databricks secret scope/key names are
    # alphanumerics plus ``-``/``_``/``.`` — reject anything else (and anything
    # that survived as ``%``-encoding or whitespace).
    if not (_SECRET_REF_SEGMENT.match(scope) and _SECRET_REF_SEGMENT.match(key)):
        raise ValueError(
            "mcp secret_ref scope/key may contain only letters, digits, '-', "
            "'_', and '.' (1-128 chars each)"
        )
    return scope, key


def _make_secret_resolver(workspace_client: Any | None) -> SecretResolver:
    """Return a ``SecretResolver`` reading Databricks secret scopes via *workspace_client*.

    The returned callable maps a ``scope/key`` reference to its decoded secret
    value. It NEVER logs the value. Raises ``ValueError`` when no workspace
    client is available but a secret is requested.
    """

    def _resolve(secret_ref: str) -> str | None:
        if workspace_client is None:
            raise ValueError(
                "an MCP server needs a secret but no Databricks workspace client "
                "is available to read it (no OBO identity?)"
            )
        scope, key = _parse_secret_ref(secret_ref)
        resp = workspace_client.secrets.get_secret(scope, key)
        raw = getattr(resp, "value", None)
        if not raw:
            return None
        # Databricks returns the secret value base64-encoded.
        try:
            return base64.b64decode(raw).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            # Fall back to the raw string if it was not base64 (defensive).
            return str(raw)

    return _resolve


def _summarize_exception(exc: BaseException, *, limit: int = 6) -> str:
    """Flatten an exception (incl. ``ExceptionGroup``/``BaseExceptionGroup``)
    into ``Type: message`` parts for diagnostics.

    The anyio/``mcp`` streamable-http client wraps the real failure (e.g. an
    httpx ``403 Forbidden`` from the MCP proxy) inside an ``ExceptionGroup``, so
    the bare ``type(exc).__name__`` is useless ("ExceptionGroup"). This unwraps
    the leaves. SAFE TO LOG: httpx/anyio exception strings carry the request URL
    + response status, NOT request headers (the OBO token never appears); each
    leaf message is truncated.
    """
    parts: list[str] = []
    stack: list[BaseException] = [exc]
    seen = 0
    while stack and len(parts) < limit:
        current = stack.pop()
        seen += 1
        if seen > 50:  # cycle/explosion guard
            break
        nested = getattr(current, "exceptions", None)
        if nested:
            stack.extend(nested)
        else:
            parts.append(f"{type(current).__name__}: {str(current)[:200]}")
    return " | ".join(parts) or type(exc).__name__


def build_mcp_toolsets(
    mcp_servers: list[MCPServerConfig],
    *,
    sp_client: Any | None,
    user_token: str | None,
) -> list[MCPToolset]:
    """Build one :class:`MCPToolset` per configured server, with OBO identity.

    A server whose toolset cannot be built (SSRF rejection, discovery failure,
    missing secret) is skipped with a warning rather than failing the whole
    request — the remaining servers and the rest of the workflow proceed.

    Args:
        mcp_servers: Declarative server configs from the workflow definition.
        sp_client: The request's service-principal/default workspace client.
        user_token: OBO token; when present, the toolset (and its secret reads)
            run as the calling user.

    Returns:
        The successfully-built toolsets (possibly empty).
    """
    if not mcp_servers:
        return []

    # Fail closed (Codex review CRIT2): a Databricks managed / UC-connection MCP
    # server is reached strictly as the calling user. Without an OBO token it
    # would run as the service principal, so refuse the whole batch rather than
    # silently degrade identity. (The host preflight ``workflow_requires_databricks``
    # is the primary gate; this is the backstop.)
    databricks_servers = [s for s in mcp_servers if s.client_kind == "databricks"]
    if databricks_servers and not user_token:
        names = ", ".join(s.name for s in databricks_servers)
        raise MCPConfigError(
            "Databricks MCP servers require the caller's identity (OBO token); "
            f"none was available for: {names}"
        )

    obo_client = resolve_workspace_client(sp_client=sp_client, user_token=user_token)
    secret_resolver = _make_secret_resolver(obo_client)
    _host = getattr(getattr(obo_client, "config", None), "host", "") or ""

    toolsets: list[MCPToolset] = []
    for server in mcp_servers:
        try:
            if server.client_kind == "databricks":
                url = _derive_databricks_mcp_url(_host, server)
                client = _build_databricks_mcp_client(url, obo_client)
                toolset = build_mcp_toolset(server, client=client)
            else:
                toolset = build_mcp_toolset(server, secret_resolver=secret_resolver)
        except MCPSecurityError as exc:
            logger.warning(
                "MCP_SERVER_SSRF_REJECTED name=%s reason=%s — server skipped",
                server.name,
                str(exc)[:200],
            )
            continue
        except Exception as exc:
            # Unwrap ExceptionGroup leaves so the real cause (e.g. an httpx
            # ``403`` from the MCP proxy — an OBO scope gap) is visible. SAFE:
            # exception strings carry URL+status, not auth headers (see
            # ``_summarize_exception``).
            logger.warning(
                "MCP_SERVER_BUILD_FAILED name=%s reason=%s detail=%s — server skipped",
                server.name,
                type(exc).__name__,
                _summarize_exception(exc),
            )
            continue
        logger.info(
            "MCP_SERVER_READY name=%s tools=%d strategy=%s citeable=%s obo=%s",
            server.name,
            len(toolset),
            server.strategy,
            server.citeable,
            bool(user_token),
        )
        toolsets.append(toolset)
    return toolsets
