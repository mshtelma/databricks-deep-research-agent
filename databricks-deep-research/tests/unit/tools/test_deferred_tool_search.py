"""Unit tests for the deferred-tool registry + ``tool_search`` builtin (§5.5).

Covers the RAG-over-tools primitives in isolation from the ReAct loop:

* the registry classifies eager vs deferred and renders name-only stubs;
* :meth:`DeferredToolRegistry.match` returns full schemas for ``select:``-style
  exact names and keyword queries, WITHOUT promoting;
* ``tool_search`` returns matched schemas, promotes them, and invokes the
  injected promotion recorder exactly once with the promoted names;
* fail-closed: :meth:`schema_status_for` raises ``KeyError`` for an unregistered
  name (the loop's reject signal).
"""

from __future__ import annotations

import json

import pytest

from databricks_deep_research.tools.builtins.tool_search import (
    TOOL_SEARCH_NAME,
    ToolSearchTool,
)
from databricks_deep_research.tools.deferred import (
    DeferredToolRegistry,
    SchemaStatus,
    first_line,
)
from databricks_deep_research.tools.protocol import ToolContext, ToolDefinition


def _defn(name: str, *, source_kind: str = "qa_assistant") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} does a thing.\nMore detail on the next line.",
        parameters={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
        source_kind=source_kind,
    )


def _registry(deferred: set[str]) -> DeferredToolRegistry:
    defs = [
        _defn("alpha_search"),
        _defn("beta_lookup"),
        _defn("compute", source_kind="builtin"),
    ]
    return DeferredToolRegistry(defs, deferred_names=deferred)


# ---------------------------------------------------------------------------
# Registry classification + stubs
# ---------------------------------------------------------------------------


class TestRegistryClassification:
    def test_status_eager_deferred_promoted(self) -> None:
        reg = _registry({"alpha_search", "beta_lookup"})
        assert reg.schema_status_for("compute") == SchemaStatus.EAGER
        assert reg.schema_status_for("alpha_search") == SchemaStatus.DEFERRED
        assert reg.is_deferred("alpha_search") is True

        reg.promote(["alpha_search"])
        assert reg.schema_status_for("alpha_search") == SchemaStatus.PROMOTED
        assert reg.is_deferred("alpha_search") is False

    def test_schema_status_unknown_raises_keyerror(self) -> None:
        """Fail-closed signal: an unregistered name has no determinable status."""
        reg = _registry({"alpha_search"})
        with pytest.raises(KeyError):
            reg.schema_status_for("never_registered")

    def test_stub_definition_is_name_and_one_liner_only(self) -> None:
        reg = _registry({"alpha_search"})
        stub = reg.stub_definition("alpha_search")
        assert stub.name == "alpha_search"
        # Full parameter schema is withheld until promotion.
        assert stub.parameters == {"type": "object", "properties": {}}
        # Only the first line of the description leaks into the catalog.
        assert "alpha_search does a thing." in stub.description
        assert "next line" not in stub.description
        assert "tool_search" in stub.description

    def test_deferred_names_excludes_promoted(self) -> None:
        reg = _registry({"alpha_search", "beta_lookup"})
        assert reg.deferred_names() == ["alpha_search", "beta_lookup"]
        reg.promote(["beta_lookup"])
        assert reg.deferred_names() == ["alpha_search"]
        assert reg.promoted_count() == 1

    def test_first_line_caps_length(self) -> None:
        assert first_line("a\nb\nc") == "a"
        assert first_line("   \n  hello world  \n x") == "hello world"
        assert len(first_line("x" * 500)) == 200


# ---------------------------------------------------------------------------
# match() — select: + keyword forms, no promotion side effect
# ---------------------------------------------------------------------------


class TestRegistryMatch:
    def test_match_by_exact_names(self) -> None:
        reg = _registry({"alpha_search", "beta_lookup"})
        matches = reg.match(names=["beta_lookup"])
        assert [m.name for m in matches] == ["beta_lookup"]
        # Full schema returned (NOT the stub).
        assert matches[0].parameters["required"] == ["q"]

    def test_match_skips_non_deferred_and_unknown_names(self) -> None:
        reg = _registry({"alpha_search"})
        # compute is eager (already schema'd); zzz is unknown — both skipped.
        matches = reg.match(names=["compute", "zzz", "alpha_search"])
        assert [m.name for m in matches] == ["alpha_search"]

    def test_match_by_query_keyword(self) -> None:
        reg = _registry({"alpha_search", "beta_lookup"})
        matches = reg.match(query="beta")
        assert [m.name for m in matches] == ["beta_lookup"]

    def test_match_does_not_promote(self) -> None:
        reg = _registry({"alpha_search"})
        reg.match(names=["alpha_search"])
        # match is a pure lookup; status stays DEFERRED until promote().
        assert reg.schema_status_for("alpha_search") == SchemaStatus.DEFERRED


# ---------------------------------------------------------------------------
# ToolSearchTool
# ---------------------------------------------------------------------------


class TestToolSearchTool:
    @pytest.mark.asyncio
    async def test_returns_schema_and_promotes_and_records(self) -> None:
        reg = _registry({"alpha_search", "beta_lookup"})
        recorded: list[list[str]] = []
        tool = ToolSearchTool(reg, recorder=recorded.append)

        # Tool is always eager (never itself deferred).
        assert tool.definition.name == TOOL_SEARCH_NAME
        assert tool.definition.metadata.get("budget_free") is True

        args = tool.validate_arguments({"names": ["alpha_search"]})
        result = await tool.execute(args, ToolContext())

        assert result.success is True
        assert result.data["matched"] == 1
        assert result.data["promoted"] == ["alpha_search"]
        # The full schema is in the content payload.
        payload = json.loads(result.content.split("\n", 1)[1])
        assert payload[0]["name"] == "alpha_search"
        assert payload[0]["parameters"]["required"] == ["q"]
        # Promotion happened and was recorded exactly once.
        assert reg.schema_status_for("alpha_search") == SchemaStatus.PROMOTED
        assert recorded == [["alpha_search"]]

    @pytest.mark.asyncio
    async def test_miss_does_not_promote_or_record(self) -> None:
        reg = _registry({"alpha_search"})
        recorded: list[list[str]] = []
        tool = ToolSearchTool(reg, recorder=recorded.append)

        args = tool.validate_arguments({"query": "nonexistent-term"})
        result = await tool.execute(args, ToolContext())

        assert result.success is True
        assert result.data["matched"] == 0
        # Empty match never widens the catalog.
        assert reg.promoted_count() == 0
        assert recorded == []

    def test_validate_requires_names_or_query(self) -> None:
        reg = _registry({"alpha_search"})
        tool = ToolSearchTool(reg)
        with pytest.raises(ValueError):
            tool.validate_arguments({})

    def test_validate_accepts_comma_separated_string(self) -> None:
        reg = _registry({"alpha_search", "beta_lookup"})
        tool = ToolSearchTool(reg)
        args = tool.validate_arguments({"names": "alpha_search, beta_lookup"})
        assert args["names"] == ["alpha_search", "beta_lookup"]

    @pytest.mark.asyncio
    async def test_works_without_recorder(self) -> None:
        """No runtime store wired => promotion still succeeds (log-only)."""
        reg = _registry({"alpha_search"})
        tool = ToolSearchTool(reg)  # recorder=None
        args = tool.validate_arguments({"names": ["alpha_search"]})
        result = await tool.execute(args, ToolContext())
        assert result.data["promoted"] == ["alpha_search"]
