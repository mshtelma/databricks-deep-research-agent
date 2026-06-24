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
import json
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


def _safe_token_scopes(token: str | None) -> str:
    """Decode an OAuth JWT's ``scope`` claim for diagnostics.

    Returns the space-delimited scope string (e.g. ``"all-apis offline_access"``
    or ``"apps sql catalog.connections"``) so we can SEE, at runtime, exactly
    which scopes the OBO token actually carries — the decisive evidence for a
    403 on a scoped endpoint (a missing scope vs a present one).

    NEVER logs the token itself; only the (non-secret) ``scope`` claim. Best
    effort: returns a short marker for a PAT (non-JWT) or an unparseable token.
    """
    if not token:
        return "(none)"
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return "(non-jwt/pat)"
        pad = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(pad))
        scope = payload.get("scope") or payload.get("scopes") or ""
        if isinstance(scope, list):
            scope = " ".join(str(s) for s in scope)
        return str(scope) or "(no-scope-claim)"
    except Exception:  # noqa: BLE001 — diagnostics must never raise
        return "(decode-failed)"


def _summarize_exception(exc: BaseException, *, limit: int = 6) -> str:
    """Flatten an exception (incl. ``ExceptionGroup``/``BaseExceptionGroup``)
    into ``Type: message`` parts for diagnostics.

    The anyio/``mcp`` streamable-http client wraps the real failure (e.g. an
    httpx ``403 Forbidden`` from the MCP proxy) inside an ``ExceptionGroup``, so
    the bare ``type(exc).__name__`` is useless ("ExceptionGroup"). This unwraps
    the leaves AND, for an httpx error, captures the response **body** — the
    authoritative reason (e.g. ``PERMISSION_DENIED`` naming the missing
    scope/grant). SAFE TO LOG: the body is a server-side error message and the
    exception string carries URL + status, NOT request headers (the OBO token
    never appears); every field is truncated.
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
            continue
        detail = f"{type(current).__name__}: {str(current)[:200]}"
        # Capture an httpx HTTPStatusError response body when present — this is
        # where Databricks puts the real cause (error_code + message) for a 403.
        response = getattr(current, "response", None)
        if response is not None:
            status = getattr(response, "status_code", "?")
            text = ""
            try:
                text = response.text or ""
            except Exception:  # noqa: BLE001 — streamed body not yet read
                try:
                    text = (response.read() or b"").decode("utf-8", "replace")
                except Exception:  # noqa: BLE001
                    text = "(body-unavailable)"
            detail += f" [http status={status} body={text[:300]!r}]"
        parts.append(detail)
    return " | ".join(parts) or type(exc).__name__


def build_mcp_toolsets(
    mcp_servers: list[MCPServerConfig],
    *,
    sp_client: Any | None,
    user_token: str | None,
) -> list[MCPToolset]:
    """Build one :class:`MCPToolset` per configured server.

    Identity model (see ``docs/mcp-external-obo-scope.md``):

    * EXTERNAL Databricks MCP (UC connection,
      ``/api/2.0/mcp/external/{connection}``) is invoked as the app SERVICE
      PRINCIPAL. Databricks authorizes the external-MCP proxy via the
      connection's ``USE_CONNECTION`` grant on the app SP (the documented
      pattern), and there is no ``user_api_scope`` that authorizes the OBO user
      to invoke it.
    * MANAGED Databricks MCP (functions / vector-search / genie) and ``http``
      servers stay on the OBO user identity (secret reads included) — those have
      real Apps scopes and carry per-user data.

    A server whose toolset cannot be built (SSRF rejection, discovery failure,
    missing secret) is skipped with a warning rather than failing the whole
    request — the remaining servers and the rest of the workflow proceed.

    Args:
        mcp_servers: Declarative server configs from the workflow definition.
        sp_client: The request's service-principal/default workspace client. Used
            directly for EXTERNAL Databricks MCP, and as the host/base for the
            OBO client.
        user_token: OBO token; when present, MANAGED Databricks MCP and secret
            reads run as the calling user.

    Returns:
        The successfully-built toolsets (possibly empty).
    """
    if not mcp_servers:
        return []

    # Identity model for Databricks-proxied MCP (see docs/mcp-external-obo-scope.md):
    #
    # * MANAGED MCP (functions / vector-search / genie) is reached strictly as the
    #   calling USER (OBO). It serves UC-governed, per-user data and HAS valid Apps
    #   OAuth scopes, so we fail closed without an OBO token (Codex CRIT2) rather
    #   than silently degrade identity to the SP.
    # * EXTERNAL MCP (UC connection) is reached as the app SERVICE PRINCIPAL:
    #   Databricks authorizes the external proxy via the connection's
    #   USE_CONNECTION grant on the SP, and NO user_api_scope authorizes the OBO
    #   user to invoke it (verified: `all-apis` is rejected by the Apps API,
    #   `catalog.connections` is list-only). So external is exempt from the OBO
    #   requirement and is built with the SP client in the loop below.
    databricks_servers = [s for s in mcp_servers if s.client_kind == "databricks"]
    managed_databricks = [
        s for s in databricks_servers if not s.connection_name.strip()
    ]
    if managed_databricks and not user_token:
        names = ", ".join(s.name for s in managed_databricks)
        raise MCPConfigError(
            "Managed Databricks MCP servers (functions/vector-search/genie) "
            f"require the caller's identity (OBO token); none was available "
            f"for: {names}"
        )

    obo_client = resolve_workspace_client(sp_client=sp_client, user_token=user_token)
    # App service-principal client (NO OBO) for EXTERNAL MCP. With user_token=None
    # this returns ``sp_client`` unchanged (the app's default/SP identity).
    sp_only_client = resolve_workspace_client(sp_client=sp_client, user_token=None)
    secret_resolver = _make_secret_resolver(obo_client)
    _host = getattr(getattr(obo_client, "config", None), "host", "") or ""

    # Diagnostic (read-only): log the OBO token's actual scope claim when a
    # MANAGED Databricks MCP server is in play (those run as the OBO user, so a
    # 403 there is an OBO scope gap). The token itself is never logged — only the
    # (non-secret) ``scope`` claim.
    if managed_databricks:
        logger.info(
            "MCP_OBO_TOKEN_SCOPES managed_servers=%d scope=%r",
            len(managed_databricks),
            _safe_token_scopes(user_token),
        )

    toolsets: list[MCPToolset] = []
    for server in mcp_servers:
        identity = "-"
        try:
            if server.client_kind == "databricks":
                # External (UC-connection) → app SP; managed → OBO user. See the
                # identity-model note above + docs/mcp-external-obo-scope.md.
                is_external = bool(server.connection_name.strip())
                identity_client = sp_only_client if is_external else obo_client
                identity = "sp" if is_external else "obo"
                if identity_client is None:
                    raise MCPConfigError(
                        f"mcp server {server.name!r}: no "
                        f"{'service-principal' if is_external else 'OBO'} "
                        "workspace client available to build the Databricks MCP "
                        "client"
                    )
                url = _derive_databricks_mcp_url(_host, server)
                client = _build_databricks_mcp_client(url, identity_client)
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
            # ``403`` from the MCP proxy) is visible, including the response
            # body. SAFE: exception/body strings carry URL+status+server error
            # message, not auth headers (see ``_summarize_exception``).
            logger.warning(
                "MCP_SERVER_BUILD_FAILED name=%s reason=%s identity=%s detail=%s "
                "— server skipped",
                server.name,
                type(exc).__name__,
                identity,
                _summarize_exception(exc),
            )
            continue
        logger.info(
            "MCP_SERVER_READY name=%s tools=%d strategy=%s citeable=%s identity=%s",
            server.name,
            len(toolset),
            server.strategy,
            server.citeable,
            identity,
        )
        toolsets.append(toolset)
    return toolsets
