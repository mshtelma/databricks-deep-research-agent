"""Adapter that exposes Databricks DiscoveryService results for the
Agent Designer chat tool-call surface (discover_sources).

Returns a normalized list of DiscoveredResource objects across Designer source
kinds. Delta tables are accepted as a manual/asset kind even when the backing
DiscoveryService does not enumerate Unity Catalog tables.
OBO scoping is enforced by the underlying DiscoveryService.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from deep_research.core.auth import get_user_workspace_client, get_workspace_client

if TYPE_CHECKING:
    from databricks_deep_research.tools.builtins.text_table.tools._common import (
        SqlExecutor,
    )

logger = logging.getLogger(__name__)

SourceKind = Literal[
    "vector_index",
    "genie_space",
    "knowledge_assistant",
    "serving_endpoint",
    "delta_table",
    "sql_warehouse",
    "mcp_server",
    "skill",
    "uc_catalog",
    "uc_schema",
    "uc_function",
]

# Map from DiscoveryService DataSourceType.value strings to our SourceKind literals.
# KNOWLEDGE_ASSISTANT covers both knowledge_assistant and serving_endpoint in the
# underlying service; we expose it as "knowledge_assistant" by default.
_SOURCE_TYPE_TO_KIND: dict[str, SourceKind] = {
    "vector_search": "vector_index",
    "genie": "genie_space",
    "knowledge_assistant": "knowledge_assistant",
}

# discover_all() only yields these kinds; every other kind is opt-in (queried
# directly), so a uc_* / sql_warehouse / mcp / skill request never pays for a
# full discovery sweep.
_DISCOVERY_SERVICE_KINDS: frozenset[str] = frozenset(
    {"vector_index", "genie_space", "knowledge_assistant", "serving_endpoint"}
)
# Unity Catalog browse kinds (catalog -> schema -> function cascade), backed by
# the OBO SQL executor over information_schema (see uc_metadata).
_UC_BROWSE_KINDS: frozenset[str] = frozenset(
    {"uc_catalog", "uc_schema", "uc_function"}
)


class DiscoveredResource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: SourceKind
    source_id: str | None = None
    name: str
    full_name: str | None = None
    description: str | None = None
    status: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class _DiscoveryResponse(Protocol):
    """Structural protocol for the object returned by the discovery service."""

    @property
    def sources(self) -> list[Any]:
        ...


class _DiscoveryServiceProto(Protocol):
    """Structural protocol matching the methods we use on DiscoveryService.

    Matches DiscoveryService.discover_all() at
    src/deep_research/services/discovery_service.py:758.
    """

    async def discover_all(
        self,
        user_id: str,
        user_token: str | None = None,
        **kwargs: Any,
    ) -> _DiscoveryResponse:
        ...


class _WarehousesAPIProto(Protocol):
    """Structural subset of the Databricks SQL Warehouses SDK API."""

    def list(self) -> Iterable[Any]:
        ...

    def get(self, id: str) -> Any:
        ...

    def start(self, id: str) -> Any:
        ...


class _WorkspaceClientProto(Protocol):
    @property
    def warehouses(self) -> _WarehousesAPIProto:
        ...


def _workspace_client_for_user_token(user_token: str | None) -> _WorkspaceClientProto:
    if user_token:
        return get_user_workspace_client(user_token)
    return get_workspace_client()


class DesignerDiscoveryAdapter:
    """Normalizes DiscoveryService output into DiscoveredResource list for the
    chat tool-call surface."""

    def __init__(
        self,
        discovery_service: _DiscoveryServiceProto,
        *,
        workspace_client_factory: (
            Callable[[str | None], _WorkspaceClientProto] | None
        ) = None,
    ) -> None:
        self._svc = discovery_service
        self._workspace_client_factory = (
            workspace_client_factory or _workspace_client_for_user_token
        )

    async def list_for_user(
        self,
        user_token: str,
        kinds: list[SourceKind] | None = None,
        user_id: str = "",
        parent: str | None = None,
        name_prefix: str = "",
    ) -> list[DiscoveredResource]:
        """Discover all sources for the given user and return normalized resources.

        Args:
            user_token: OBO token for the user. Passed to DiscoveryService for
                        per-user access scoping when present.
            kinds: Optional filter — only return resources whose kind is in this
                   list. When None, all discovered kinds are returned.
            user_id: Stable authenticated user id for cache keying. Required
                     when no OBO token is available, such as local profile auth.

        Returns:
            List of DiscoveredResource objects, one per discovered source.
        """
        cache_user_id = user_id or user_token
        if not cache_user_id:
            raise ValueError("Either user_id or user_token must be provided")

        resources: list[DiscoveredResource] = []
        include_discovery_resources = kinds is None or any(
            kind in _DISCOVERY_SERVICE_KINDS for kind in kinds
        )
        if include_discovery_resources:
            response = await self._svc.discover_all(
                user_id=cache_user_id,
                user_token=user_token or None,
                include_all_endpoints=True,
            )

            for source in response.sources:
                metadata: dict[str, Any] = dict(source.metadata) if source.metadata else {}
                # source.source_type is a DataSourceType enum or its string value
                raw_type = source.source_type
                type_str: str = (
                    raw_type.value if hasattr(raw_type, "value") else str(raw_type)
                )

                kind = _SOURCE_TYPE_TO_KIND.get(type_str)
                if (
                    kind == "knowledge_assistant"
                    and metadata.get("is_knowledge_assistant") is False
                ):
                    kind = "serving_endpoint"
                if kind is None:
                    # Unknown source type — skip rather than error
                    continue

                if kinds is not None and kind not in kinds:
                    continue

                # Extract full_name: for vector indexes the index_name metadata field
                # holds the full three-part name; fall back to endpoint_name or name.
                full_name: str | None = (
                    metadata.get("index_name")
                    or metadata.get("space_id")
                    or metadata.get("endpoint_name")
                    or getattr(source, "endpoint_name", None)
                    or None
                )

                resources.append(
                    DiscoveredResource(
                        kind=kind,
                        source_id=getattr(source, "source_id", None),
                        name=source.name,
                        full_name=full_name,
                        description=source.description,
                        status=getattr(source, "status", None),
                        capabilities=list(getattr(source, "capabilities", []) or []),
                        metadata=metadata,
                    )
                )

        if kinds is None or "sql_warehouse" in kinds:
            try:
                resources.extend(self._list_sql_warehouses(user_token or None))
            except Exception:
                if kinds is not None and "sql_warehouse" in kinds:
                    raise
                logger.warning("DESIGNER_SQL_WAREHOUSE_DISCOVERY_SKIPPED", exc_info=True)

        # MCP servers and skills are OPT-IN discovery kinds: surfaced only when
        # explicitly requested (the UI requests them by kind), never as part of an
        # unfiltered "discover everything" sweep — so the default resource set is
        # byte-identical to before this feature.
        if kinds is not None and "mcp_server" in kinds:
            try:
                resources.extend(self._list_mcp_servers(user_token or None))
            except Exception:
                logger.warning("DESIGNER_MCP_DISCOVERY_FAILED", exc_info=True)
                raise

        if kinds is not None and "skill" in kinds:
            try:
                resources.extend(await self._list_skills(user_token or None))
            except Exception:
                logger.warning("DESIGNER_SKILL_DISCOVERY_FAILED", exc_info=True)
                raise

        # UC browse (catalog/schema/function) — opt-in, parent-parameterized.
        # Fail-soft: a missing warehouse or query error yields an empty list so
        # the picker degrades to manual FQN entry rather than surfacing a 500.
        if kinds is not None and any(k in _UC_BROWSE_KINDS for k in kinds):
            try:
                # Offload the blocking information_schema query so a cold-warehouse
                # poll (up to ~5s) never stalls the event loop mid-cascade.
                resources.extend(
                    await asyncio.to_thread(
                        self._list_uc_resources,
                        kinds,
                        parent,
                        name_prefix,
                        user_token or None,
                    )
                )
            except Exception:
                logger.warning("DESIGNER_UC_BROWSE_FAILED", exc_info=True)

        return resources

    def _list_mcp_servers(self, user_token: str | None) -> list[DiscoveredResource]:
        """Discover external UC-connection MCP servers (kind='mcp_server')."""
        from deep_research.services.mcp_discovery import discover_mcp_connections

        client = self._workspace_client_factory(user_token)
        servers = discover_mcp_connections(client)
        return [
            DiscoveredResource(
                kind="mcp_server",
                source_id=server.connection_name or server.name,
                name=server.name,
                full_name=server.connection_name or None,
                description=server.description or None,
                metadata={
                    "client_kind": server.client_kind,
                    "connection_name": server.connection_name,
                    "managed_target": server.managed_target,
                    **server.metadata,
                },
            )
            for server in servers
        ]

    async def _list_skills(self, user_token: str | None) -> list[DiscoveredResource]:
        """List the caller's available skills (kind='skill')."""
        from deep_research.services.skill_runtime import list_runtime_skills

        client = self._workspace_client_factory(user_token)
        metas = await list_runtime_skills(
            workspace_client=client, user_token=user_token
        )
        return [
            DiscoveredResource(
                kind="skill",
                source_id=meta.name,
                name=meta.name,
                description=meta.description,
            )
            for meta in metas
        ]

    def _build_sql_executor(self, user_token: str | None) -> SqlExecutor | None:
        """Build an OBO SQL executor for information_schema browse, or None when
        no SQL warehouse is configured (browse then degrades to manual entry)."""
        from databricks_deep_research.tools.builtins.text_table.runtime_wiring import (
            StatementExecutionTableSQL,
        )

        from deep_research.agent.workflow_runner_factory import (
            _resolve_table_warehouse_id,
        )

        warehouse_id = _resolve_table_warehouse_id()
        if not warehouse_id:
            logger.info("DESIGNER_UC_BROWSE_SKIP reason=no_warehouse")
            return None
        return StatementExecutionTableSQL(
            workspace_client=self._workspace_client_factory(user_token),
            warehouse_id=warehouse_id,
            timeout_sec=5.0,
        )

    def _list_uc_resources(
        self,
        kinds: list[SourceKind],
        parent: str | None,
        name_prefix: str,
        user_token: str | None,
    ) -> list[DiscoveredResource]:
        """Browse Unity Catalog via OBO SQL: catalogs, schemas (``parent`` =
        catalog), or functions (``parent`` = ``catalog.schema``). Only the
        requested kinds are queried; each needs its parent to have been chosen."""
        from deep_research.agent_designer import uc_metadata

        sql = self._build_sql_executor(user_token)
        if sql is None:
            return []
        out: list[DiscoveredResource] = []
        if "uc_catalog" in kinds:
            out.extend(
                _uc_rows_to_resources(
                    "uc_catalog",
                    uc_metadata.list_catalogs(sql, name_prefix=name_prefix),
                )
            )
        if "uc_schema" in kinds and parent:
            out.extend(
                _uc_rows_to_resources(
                    "uc_schema",
                    uc_metadata.list_schemas(sql, parent, name_prefix=name_prefix),
                )
            )
        if "uc_function" in kinds and parent and "." in parent:
            catalog, schema = parent.split(".", 1)
            out.extend(
                _uc_rows_to_resources(
                    "uc_function",
                    uc_metadata.list_functions(
                        sql, catalog, schema, name_prefix=name_prefix
                    ),
                )
            )
        return out

    def get_function_signature(
        self, user_token: str | None, fqn: str
    ) -> dict[str, Any]:
        """Introspect a single UC function's signature for the live picker.

        Fail-soft: returns empty params (with a ``warning``) when no warehouse is
        configured; a malformed FQN raises ``ValueError`` for the caller to 400."""
        from deep_research.agent_designer import uc_metadata

        sql = self._build_sql_executor(user_token)
        if sql is None:
            return {
                "function": fqn,
                "params": [],
                "scalar": True,
                "warning": (
                    "No SQL warehouse configured; parameters could not be "
                    "introspected. Enter parameters manually if the function "
                    "takes arguments."
                ),
            }
        return uc_metadata.get_signature(sql, fqn)

    def _list_sql_warehouses(self, user_token: str | None) -> list[DiscoveredResource]:
        client = self._workspace_client_factory(user_token)
        resources = [
            resource
            for warehouse in client.warehouses.list()
            if (resource := _sql_warehouse_resource(warehouse)) is not None
        ]
        resources.sort(key=lambda item: (item.name.lower(), item.source_id or ""))
        return resources

    async def start_sql_warehouse(
        self,
        *,
        user_token: str,
        warehouse_id: str,
    ) -> DiscoveredResource:
        """Start a stopped SQL warehouse and return its current resource shape."""
        client = self._workspace_client_factory(user_token or None)
        warehouse = client.warehouses.get(id=warehouse_id)
        if _enum_value(getattr(warehouse, "state", None)).upper() == "STOPPED":
            client.warehouses.start(id=warehouse_id)
            warehouse = client.warehouses.get(id=warehouse_id)
        resource = _sql_warehouse_resource(warehouse)
        if resource is None:
            raise ValueError(f"SQL warehouse {warehouse_id!r} was not returned by Databricks")
        return resource


def _uc_rows_to_resources(
    kind: SourceKind, rows: list[dict[str, str]]
) -> list[DiscoveredResource]:
    """Map uc_metadata row dicts ({name, full_name?, description?}) to resources,
    reusing the DiscoveredResource contract the resource-select widget consumes."""
    return [
        DiscoveredResource(
            kind=kind,
            source_id=row.get("full_name") or row["name"],
            name=row["name"],
            full_name=row.get("full_name"),
            description=row.get("description"),
        )
        for row in rows
    ]


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", None) or getattr(value, "name", None) or value
    return str(raw) if raw is not None else ""


def _warehouse_string(warehouse: Any, attr: str) -> str:
    value = getattr(warehouse, attr, None)
    return str(value).strip() if value is not None else ""


def _sql_warehouse_resource(warehouse: Any) -> DiscoveredResource | None:
    warehouse_id = _warehouse_string(warehouse, "id")
    name = _warehouse_string(warehouse, "name") or warehouse_id
    if not warehouse_id and not name:
        return None

    state = _enum_value(getattr(warehouse, "state", None))
    warehouse_type = _enum_value(getattr(warehouse, "warehouse_type", None))
    cluster_size = _warehouse_string(warehouse, "cluster_size")
    description_parts = [part for part in (warehouse_type, cluster_size) if part]

    metadata: dict[str, Any] = {"warehouse_id": warehouse_id}
    if state:
        metadata["state"] = state
    if warehouse_type:
        metadata["warehouse_type"] = warehouse_type
    if cluster_size:
        metadata["cluster_size"] = cluster_size
    for attr in ("auto_stop_mins", "creator_name", "enable_serverless_compute"):
        value = getattr(warehouse, attr, None)
        if value is not None:
            metadata[attr] = value

    return DiscoveredResource(
        kind="sql_warehouse",
        source_id=warehouse_id or None,
        name=name,
        full_name=name,
        description=", ".join(description_parts) or None,
        status=state or None,
        capabilities=["sql", "table_queries"],
        metadata=metadata,
    )


# Three-part Unity Catalog identifier: catalog.schema.name. Each segment is a
# Databricks-legal identifier (letters / digits / underscore, leading letter or
# underscore). Used for deterministic FQN extraction from free-text intent
# before falling back to LLM-driven fuzzy matching.
#
# The lookahead allows a trailing dot (sentence punctuation) but rejects a
# dot followed by another identifier segment — that would mean the captured
# 3-part name is actually a prefix of a longer (malformed) identifier and we
# should not extract it as a UC FQN.
_FQN_THREE_PART_RE = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)"
    r"(?![A-Za-z0-9_])"
    r"(?!\.[A-Za-z_])"
)


class IntentMatch(BaseModel):
    """One free-text → workspace-resource match produced by ``match_text_to_resources``."""

    model_config = ConfigDict(extra="forbid")

    resource: DiscoveredResource
    score: int
    matched_via: Literal["fqn_exact", "fqn_ci", "name_exact", "name_ci"]
    matched_text: str


MatchVia = Literal["fqn_exact", "fqn_ci", "name_exact", "name_ci"]


def _candidate_identities(resource: DiscoveredResource) -> list[str]:
    out: list[str] = []
    for value in (resource.full_name, resource.source_id, resource.name):
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                out.append(cleaned)
    return out


def match_text_to_resources(
    intent: str,
    resources: list[DiscoveredResource],
    *,
    min_score: int = 60,
) -> list[IntentMatch]:
    """Match free-text mentions in ``intent`` to ``resources`` deterministically.

    Returns matches whose score >= ``min_score``, sorted by score descending,
    then by resource name for stable ordering.

    Scoring (additive across match kinds is NOT supported — best per resource wins):

    * 100 — three-part FQN appears verbatim in intent AND equals a resource
            ``full_name`` / ``source_id`` / ``name``.
    *  90 — three-part FQN appears verbatim and equals a resource identifier
            case-insensitively.
    *  80 — resource identifier appears verbatim as a substring of intent.
    *  60 — resource identifier appears as a case-insensitive substring.

    Generic across :data:`SourceKind` — does not interpret the resource kind.
    """

    if not intent or not resources:
        return []

    intent_text = intent
    intent_lower = intent.lower()
    fqn_hits: set[str] = set(_FQN_THREE_PART_RE.findall(intent_text))
    fqn_hits_lower: set[str] = {hit.lower() for hit in fqn_hits}

    seen_resource_ids: set[int] = set()
    matches: list[IntentMatch] = []
    for resource in resources:
        if id(resource) in seen_resource_ids:
            continue
        best: tuple[int, MatchVia, str] | None = None  # (score, matched_via, matched_text)
        for identity in _candidate_identities(resource):
            identity_lower = identity.lower()
            is_three_part = identity.count(".") == 2
            candidate: tuple[int, MatchVia, str]
            if is_three_part and identity in fqn_hits:
                candidate = (100, "fqn_exact", identity)
            elif is_three_part and identity_lower in fqn_hits_lower:
                candidate = (90, "fqn_ci", identity)
            elif identity in intent_text:
                candidate = (80, "name_exact", identity)
            elif identity_lower in intent_lower:
                candidate = (60, "name_ci", identity)
            else:
                continue
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None or best[0] < min_score:
            continue
        seen_resource_ids.add(id(resource))
        matches.append(
            IntentMatch(
                resource=resource,
                score=best[0],
                matched_via=best[1],
                matched_text=best[2],
            )
        )

    matches.sort(key=lambda m: (-m.score, m.resource.name))
    return matches


def extract_fqn_candidates(intent: str) -> list[str]:
    """Return three-part FQN-shaped substrings found in ``intent``.

    Useful when the workspace catalog could not be enumerated and the only
    deterministic signal is the literal identifier pattern.
    """

    if not intent:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for hit in _FQN_THREE_PART_RE.findall(intent):
        if hit in seen:
            continue
        seen.add(hit)
        out.append(hit)
    return out
