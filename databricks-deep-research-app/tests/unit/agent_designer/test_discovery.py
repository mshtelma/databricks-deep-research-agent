"""Tests for DesignerDiscoveryAdapter using a fake DiscoveryService."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from deep_research.agent_designer.discovery import (
    DesignerDiscoveryAdapter,
    DiscoveredResource,
)


def _make_source(
    source_type_value: str,
    name: str,
    endpoint_name: str = "",
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Build a minimal fake DiscoveredSource-like object."""
    # Use a simple namespace so attribute access works without importing
    # the real DiscoveredSource (which has Databricks SDK deps).
    source_type = SimpleNamespace(value=source_type_value)
    return SimpleNamespace(
        source_type=source_type,
        name=name,
        endpoint_name=endpoint_name,
        description=description,
        metadata=metadata or {},
    )


class _FakeDiscoveryResponse:
    def __init__(self, sources: list[Any]) -> None:
        self.sources = sources


class _FakeDiscoveryService:
    def __init__(self, sources: list[Any]) -> None:
        self._sources = sources
        self.calls: list[tuple[str, str | None, dict[str, Any]]] = []

    async def discover_all(
        self,
        user_id: str,
        user_token: str | None = None,
        **kwargs: Any,
    ) -> _FakeDiscoveryResponse:
        self.calls.append((user_id, user_token, kwargs))
        return _FakeDiscoveryResponse(self._sources)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_lists_vector_indexes() -> None:
    """Vector Search sources are mapped to the vector_index kind."""
    fake = _FakeDiscoveryService([
        _make_source(
            "vector_search",
            "idx1",
            metadata={"index_name": "cat.sch.idx1"},
            description="d",
        ),
    ])
    adapter = DesignerDiscoveryAdapter(fake)
    out = await adapter.list_for_user(user_token="t1", user_id="u1")

    assert len(out) == 1
    assert out[0].kind == "vector_index"
    assert out[0].name == "idx1"
    assert out[0].full_name == "cat.sch.idx1"
    assert out[0].description == "d"
    # The underlying service uses the stable user id for cache keying.
    assert fake.calls == [("u1", "t1", {"include_all_endpoints": True})]


async def test_local_profile_auth_uses_user_id_without_token() -> None:
    """Local profile auth has no OBO token, but discovery still needs a user id."""
    fake = _FakeDiscoveryService([
        _make_source("vector_search", "idx1"),
    ])
    adapter = DesignerDiscoveryAdapter(fake)

    out = await adapter.list_for_user(user_token="", user_id="local-user")

    assert len(out) == 1
    assert fake.calls == [("local-user", None, {"include_all_endpoints": True})]


async def test_filters_by_kinds() -> None:
    """When kinds is provided, only matching resources are returned."""
    fake = _FakeDiscoveryService([
        _make_source("vector_search", "idx1"),
        _make_source("genie", "g1"),
    ])
    adapter = DesignerDiscoveryAdapter(fake)
    out = await adapter.list_for_user(user_token="t", kinds=["vector_index"])

    assert all(r.kind == "vector_index" for r in out)
    assert any(r.name == "idx1" for r in out)
    assert not any(r.name == "g1" for r in out)


async def test_empty_payload() -> None:
    """Empty source list returns empty result without error."""
    fake = _FakeDiscoveryService([])
    adapter = DesignerDiscoveryAdapter(fake)
    out = await adapter.list_for_user(user_token="t")
    assert out == []


async def test_includes_genie_spaces_and_kas() -> None:
    """All four source kinds are surfaced when present."""
    fake = _FakeDiscoveryService([
        _make_source("vector_search", "v1"),
        _make_source("genie", "g1", description="genie d"),
        _make_source("knowledge_assistant", "ka1"),
    ])
    adapter = DesignerDiscoveryAdapter(fake)
    out = await adapter.list_for_user(user_token="t")

    kinds = {r.kind for r in out}
    assert kinds == {"vector_index", "genie_space", "knowledge_assistant"}
    assert len(out) == 3


async def test_non_ka_serving_endpoints_are_exposed_separately() -> None:
    """Generic serving endpoints are available to the Designer as serving_endpoint."""
    fake = _FakeDiscoveryService([
        _make_source(
            "knowledge_assistant",
            "model-endpoint",
            endpoint_name="model-endpoint",
            metadata={
                "endpoint_name": "model-endpoint",
                "is_knowledge_assistant": False,
            },
        ),
    ])
    adapter = DesignerDiscoveryAdapter(fake)

    out = await adapter.list_for_user(user_token="t")

    assert len(out) == 1
    assert out[0].kind == "serving_endpoint"
    assert out[0].full_name == "model-endpoint"


async def test_unknown_source_type_is_skipped() -> None:
    """Sources with unrecognized types are silently dropped."""
    fake = _FakeDiscoveryService([
        _make_source("unknown_type", "x1"),
        _make_source("vector_search", "v1"),
    ])
    adapter = DesignerDiscoveryAdapter(fake)
    out = await adapter.list_for_user(user_token="t")

    assert len(out) == 1
    assert out[0].kind == "vector_index"


async def test_no_kind_filter_returns_all() -> None:
    """Passing kinds=None returns every discovered source."""
    fake = _FakeDiscoveryService([
        _make_source("vector_search", "v1"),
        _make_source("genie", "g1"),
        _make_source("knowledge_assistant", "ka1"),
    ])
    adapter = DesignerDiscoveryAdapter(fake)
    out = await adapter.list_for_user(user_token="tok", kinds=None)

    assert len(out) == 3


async def test_full_name_falls_back_to_endpoint_name() -> None:
    """full_name uses endpoint_name when no index_name/space_id metadata key exists."""
    fake = _FakeDiscoveryService([
        _make_source(
            "knowledge_assistant",
            "my-assistant",
            endpoint_name="my-assistant-endpoint",
            metadata={"endpoint_name": "my-assistant-endpoint"},
        ),
    ])
    adapter = DesignerDiscoveryAdapter(fake)
    out = await adapter.list_for_user(user_token="t")

    assert len(out) == 1
    assert out[0].full_name == "my-assistant-endpoint"


async def test_genie_full_name_from_space_id_metadata() -> None:
    """Genie sources use space_id from metadata as full_name."""
    fake = _FakeDiscoveryService([
        _make_source(
            "genie",
            "My Space",
            metadata={"space_id": "abc-123", "title": "My Space"},
        ),
    ])
    adapter = DesignerDiscoveryAdapter(fake)
    out = await adapter.list_for_user(user_token="t")

    assert len(out) == 1
    assert out[0].kind == "genie_space"
    assert out[0].full_name == "abc-123"


async def test_returned_objects_are_discovered_resource_instances() -> None:
    """Each returned item is a DiscoveredResource Pydantic model."""
    fake = _FakeDiscoveryService([
        _make_source("vector_search", "v1"),
    ])
    adapter = DesignerDiscoveryAdapter(fake)
    out = await adapter.list_for_user(user_token="t")

    assert all(isinstance(r, DiscoveredResource) for r in out)
