"""Adapter that exposes Databricks DiscoveryService results for the
Agent Designer chat tool-call surface (discover_sources).

Returns a normalized list of DiscoveredResource objects across Designer source
kinds. Delta tables are accepted as a manual/asset kind even when the backing
DiscoveryService does not enumerate Unity Catalog tables.
OBO scoping is enforced by the underlying DiscoveryService.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

SourceKind = Literal[
    "vector_index",
    "genie_space",
    "knowledge_assistant",
    "serving_endpoint",
    "delta_table",
]

# Map from DiscoveryService DataSourceType.value strings to our SourceKind literals.
# KNOWLEDGE_ASSISTANT covers both knowledge_assistant and serving_endpoint in the
# underlying service; we expose it as "knowledge_assistant" by default.
_SOURCE_TYPE_TO_KIND: dict[str, SourceKind] = {
    "vector_search": "vector_index",
    "genie": "genie_space",
    "knowledge_assistant": "knowledge_assistant",
}


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


class DesignerDiscoveryAdapter:
    """Normalizes DiscoveryService output into DiscoveredResource list for the
    chat tool-call surface."""

    def __init__(self, discovery_service: _DiscoveryServiceProto) -> None:
        self._svc = discovery_service

    async def list_for_user(
        self,
        user_token: str,
        kinds: list[SourceKind] | None = None,
        user_id: str = "",
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

        response = await self._svc.discover_all(
            user_id=cache_user_id,
            user_token=user_token or None,
            include_all_endpoints=True,
        )

        resources: list[DiscoveredResource] = []
        for source in response.sources:
            metadata: dict[str, Any] = dict(source.metadata) if source.metadata else {}
            # source.source_type is a DataSourceType enum or its string value
            raw_type = source.source_type
            type_str: str = (
                raw_type.value if hasattr(raw_type, "value") else str(raw_type)
            )

            kind = _SOURCE_TYPE_TO_KIND.get(type_str)
            if kind == "knowledge_assistant" and metadata.get("is_knowledge_assistant") is False:
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

        return resources
