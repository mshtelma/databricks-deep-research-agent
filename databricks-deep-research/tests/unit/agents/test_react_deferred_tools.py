"""ReactLoop integration tests for RAG-over-tools / deferred tools (§5.5).

Pins the loop-level contract:

* a many-tool catalog (above the threshold) lists deferred tools by NAME +
  one-liner only, and auto-injects ``tool_search``;
* a ``tool_search`` round returns the matched full schema AND promotes the tool,
  so the NEXT LLM call's catalog carries that tool's full parameter schema;
* fail-closed: a tool that survives filtering but is not registered in the
  deferred catalog is rejected (no silent un-schema'd tool reaches the LLM);
* the promotion is recorded in the append-only RuntimeState (a diagnostic);
* default (small catalog) path is BYTE-IDENTICAL — every tool listed in full,
  no ``tool_search`` injected.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from databricks_deep_research.agents.react_loop import ReactLoop
from databricks_deep_research.llm.client import LLMResponse, ToolCall
from databricks_deep_research.tools.builtins.tool_search import TOOL_SEARCH_NAME
from databricks_deep_research.tools.protocol import (
    ToolContext,
    ToolDefinition,
    ToolResult,
)
from databricks_deep_research.workflow.runtime_core.store import (
    TypedRuntimeStateStore,
)


def _make_tool(
    name: str,
    *,
    source_kind: str = "qa_assistant",
    deferrable: bool | None = None,
) -> MagicMock:
    meta: dict[str, Any] = {}
    if deferrable is not None:
        meta["deferrable"] = deferrable
    tool = MagicMock()
    tool.definition = ToolDefinition(
        name=name,
        description=f"{name} description line one.\nsecond line hidden.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        source_kind=source_kind,
        metadata=meta,
    )
    tool.validate_arguments = MagicMock(side_effect=lambda a: a)
    tool.execute = AsyncMock(
        return_value=ToolResult(content=f"{name} ran", success=True, sources=[])
    )
    return tool


def _tc(tc_id: str, name: str, args: str = "{}") -> ToolCall:
    return ToolCall(id=tc_id, function_name=name, arguments=args)


def _resp(content: str = "", tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=tool_calls or [], model="test")


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "test query"}]


def _names_in(defs: list[dict[str, Any]]) -> set[str]:
    return {d["function"]["name"] for d in defs}


def _def_for(defs: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(d for d in defs if d["function"]["name"] == name)


# ---------------------------------------------------------------------------
# Catalog construction
# ---------------------------------------------------------------------------


class TestDeferredCatalog:
    def test_below_threshold_is_byte_identical(self) -> None:
        """Default path: no registry, no tool_search, full schemas listed."""
        tools = [_make_tool("a"), _make_tool("b")]
        loop = ReactLoop(MagicMock(), tools, max_tool_calls=10)

        assert loop._deferred_registry is None
        assert TOOL_SEARCH_NAME not in _names_in(loop._tool_defs)
        # Byte-identical to mapping the pre-existing converter over the tools.
        expected = [ReactLoop._to_openai_tool(t) for t in tools]
        assert loop._tool_defs == expected

    def test_explicit_flag_engages_below_threshold(self) -> None:
        tools = [_make_tool("a"), _make_tool("b")]
        loop = ReactLoop(MagicMock(), tools, max_tool_calls=10, defer_tools=True)
        assert loop._deferred_registry is not None
        assert TOOL_SEARCH_NAME in _names_in(loop._tool_defs)

    def test_many_tools_listed_by_name_only(self) -> None:
        """Above threshold: deferred tools contribute name + one-liner stubs."""
        tools = [_make_tool(f"tool_{i}") for i in range(5)]
        loop = ReactLoop(
            MagicMock(), tools, max_tool_calls=10, defer_threshold=3
        )

        assert loop._deferred_registry is not None
        names = _names_in(loop._tool_defs)
        # tool_search auto-injected; every research tool still listed by NAME.
        assert TOOL_SEARCH_NAME in names
        assert {f"tool_{i}" for i in range(5)} <= names

        # Each deferred tool is a STUB: empty params + one-line description.
        stub = _def_for(loop._tool_defs, "tool_0")["function"]
        assert stub["parameters"] == {"type": "object", "properties": {}}
        assert "description line one." in stub["description"]
        assert "second line hidden." not in stub["description"]

        # tool_search itself is eager: it has real parameters.
        search_def = _def_for(loop._tool_defs, TOOL_SEARCH_NAME)["function"]
        assert "names" in search_def["parameters"]["properties"]

    def test_explicit_deferrable_metadata_selects_subset(self) -> None:
        """Only tools stamping deferrable=True are deferred when any opts in."""
        deferred = _make_tool("dyn", deferrable=True)
        eager = _make_tool("native")  # no opt-in
        loop = ReactLoop(
            MagicMock(), [deferred, eager], max_tool_calls=10, defer_tools=True
        )
        reg = loop._deferred_registry
        assert reg is not None
        assert reg.is_deferred("dyn") is True
        assert reg.is_deferred("native") is False
        # native keeps its full schema in the catalog.
        native_def = _def_for(loop._tool_defs, "native")["function"]
        assert "query" in native_def["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Fail-closed
# ---------------------------------------------------------------------------


class TestFailClosed:
    def test_unregistered_surviving_tool_is_rejected(self) -> None:
        """A tool present in the loop but absent from the registry is rejected
        rather than listed with a silently-missing schema (DeerFlow §184)."""
        tools = [_make_tool("a"), _make_tool("b")]
        loop = ReactLoop(MagicMock(), tools, max_tool_calls=10, defer_tools=True)
        assert loop._deferred_registry is not None

        rogue = _make_tool("rogue_never_registered")
        with pytest.raises(ValueError, match="fail-closed"):
            loop._build_tool_defs([rogue])


# ---------------------------------------------------------------------------
# Promotion via tool_search → next-call catalog
# ---------------------------------------------------------------------------


class TestPromotionFlow:
    @pytest.mark.asyncio
    async def test_tool_search_promotes_and_records(self) -> None:
        tool_a = _make_tool("alpha_lookup")
        tool_b = _make_tool("beta_lookup")
        store = TypedRuntimeStateStore(query="q")
        ctx = ToolContext(extras={"_framework_runtime_store": store})

        # Capture the catalog the LLM sees on each call.
        seen_catalogs: list[set[str]] = []
        seen_alpha_params: list[dict[str, Any]] = []
        call_num = 0

        async def mock_complete(
            messages: Any, tier: Any, **kwargs: Any
        ) -> LLMResponse:
            nonlocal call_num
            call_num += 1
            tools = kwargs.get("tools") or []
            seen_catalogs.append(_names_in(tools))
            for d in tools:
                if d["function"]["name"] == "alpha_lookup":
                    seen_alpha_params.append(d["function"]["parameters"])
            if call_num == 1:
                # Fetch alpha_lookup's full schema via tool_search.
                return _resp(tool_calls=[
                    _tc("t1", TOOL_SEARCH_NAME,
                        json.dumps({"names": ["alpha_lookup"]})),
                ])
            if call_num == 2:
                # Now call the promoted tool.
                return _resp(tool_calls=[
                    _tc("t2", "alpha_lookup", json.dumps({"query": "x"})),
                ])
            return _resp(content="Done.")

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=mock_complete)

        loop = ReactLoop(
            llm,
            [tool_a, tool_b],
            tool_context=ctx,
            max_tool_calls=10,
            defer_tools=True,
            node_id="n1",
        )
        result = await loop.execute(_messages())

        # Call 1 saw alpha_lookup as a STUB (empty params).
        assert seen_alpha_params[0] == {"type": "object", "properties": {}}
        # Call 2 (after promotion) saw the FULL schema.
        assert seen_alpha_params[1]["required"] == ["query"]

        # The deferred tool actually executed after promotion.
        tool_a.execute.assert_awaited_once()
        # tool_search is budget-free => only the real tool counts.
        assert result.tool_calls_made == 1

        # Promotion recorded in the append-only RuntimeState as a diagnostic.
        runtime = store.runtime()
        promo = [
            r for r in runtime.diagnostics.records
            if r.category == "tool_promotion"
        ]
        assert len(promo) == 1
        assert "alpha_lookup" in promo[0].message
        assert promo[0].node_id == "n1"

    @pytest.mark.asyncio
    async def test_unpromoted_deferred_tool_call_is_rejected(self) -> None:
        """Calling a deferred tool BEFORE fetching its schema is gated.

        The loop only exposes a stub; the model 'guessing' the call before
        tool_search must not silently succeed — the tool stays deferred and the
        execution gate refuses it (tool not in the active set)."""
        tool_a = _make_tool("alpha_lookup")
        call_num = 0

        async def mock_complete(
            messages: Any, tier: Any, **kwargs: Any
        ) -> LLMResponse:
            nonlocal call_num
            call_num += 1
            if call_num == 1:
                # Call the deferred tool directly with a guessed schema.
                return _resp(tool_calls=[
                    _tc("t1", "alpha_lookup", json.dumps({"query": "x"})),
                ])
            return _resp(content="Gave up.")

        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=mock_complete)

        loop = ReactLoop(
            llm, [tool_a], max_tool_calls=10, defer_tools=True, node_id="n1"
        )
        await loop.execute(_messages())

        # The stub carries no real schema and the tool never ran.
        tool_a.execute.assert_not_awaited()
