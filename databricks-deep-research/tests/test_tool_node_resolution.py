"""Tool-node execution: resolver-first resolution, literals, structured outputs.

Covers the Phase-1 tool-node upgrades: resolution through the ToolResolver
(declarations, overrides, legacy fallback), ``input_literals``,
``output_data_key`` envelopes, ``bind_namespace``, ``fail_on_error``,
``enforce_output_schema``, tool events, user_token threading, and the
isolate-subworkflow override carry.
"""

from __future__ import annotations

from typing import Any

import pytest

from databricks_deep_research.events.types import (
    NodeCompletedEvent,
    NodeStartedEvent,
    StreamEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import (
    ToolContext,
    ToolDefinition,
    ToolResult,
)
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.tools.resolver import ToolResolver
from databricks_deep_research.workflow.definition import (
    NodeType,
    ToolDeclaration,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.executor import WorkflowExecutor
from databricks_deep_research.workflow.state import WorkflowState
from tests.conftest import build_mock_llm_client, collect_events, events_of_type


def sample_double(x: int) -> dict[str, Any]:
    """Double a number (imported by the decorated-tool declaration below)."""
    return {"result": x * 2}


class RecordingTool:
    """Minimal ResearchTool that records calls and returns a canned result."""

    def __init__(
        self,
        name: str = "fake_tool",
        result: ToolResult | None = None,
        source_kind: str = "builtin",
    ) -> None:
        self._name = name
        self._result = result or ToolResult(content="ok", data={"result": 42})
        self._source_kind = source_kind
        self.calls: list[dict[str, Any]] = []
        self.contexts: list[ToolContext] = []

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description="recording test tool",
            parameters={"type": "object", "properties": {}},
            source_kind=self._source_kind,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return dict(arguments)

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        self.calls.append(dict(arguments))
        self.contexts.append(context)
        return self._result


def _tool_node(config: dict[str, Any], node_id: str = "t1") -> WorkflowNode:
    return WorkflowNode(id=node_id, type=NodeType.tool, label=node_id, config=config)


def _definition(
    root: WorkflowNode, tools: list[ToolDeclaration] | None = None
) -> WorkflowDefinition:
    return WorkflowDefinition(
        id="tool-node-test",
        name="Tool Node Test",
        root=root,
        tools=tools or [],
    )


async def _run(
    definition: WorkflowDefinition,
    *,
    state: WorkflowState | None = None,
    **executor_kwargs: Any,
) -> tuple[WorkflowExecutor, WorkflowState, list[StreamEvent]]:
    executor = WorkflowExecutor(definition, build_mock_llm_client(), **executor_kwargs)
    run_state = state or WorkflowState(query="test query")
    events = await collect_events(executor, run_state)
    return executor, run_state, events


async def test_resolves_override_with_mcp_typed_ref() -> None:
    """Per-request overrides (the MCP injection path) are reachable from tool nodes."""
    tool = RecordingTool(name="fake_mcp")
    node = _tool_node({
        "ref": {"type": "mcp", "name": "fake_mcp"},
        "input_literals": {"window": 20},
        "output_key": "out",
    })
    definition = _definition(node)
    executor = WorkflowExecutor(definition, build_mock_llm_client())
    executor._resolver.override("fake_mcp", tool)

    state = WorkflowState(query="q")
    await collect_events(executor, state)

    assert tool.calls == [{"window": 20}]
    assert state.get("out") == "ok"


async def test_resolves_declared_decorated_tool() -> None:
    """Declared tools (factory chain) are reachable from tool nodes.

    The default chain is import-gated (deny-all), so the host passes an
    explicitly allow-listed DecoratedToolFactory.
    """
    from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
    from databricks_deep_research.tools.factories.decorated import DecoratedToolFactory

    decl = ToolDeclaration(
        name="doubler",
        kind="decorated",
        config={"import": "tests.test_tool_node_resolution:sample_double"},
    )
    node = _tool_node({
        "ref": {"name": "doubler"},
        "input_literals": {"x": 21},
        "output_key": "doubled",
    })
    _executor, state, events = await _run(
        _definition(node, tools=[decl]),
        tool_factories=[
            BuiltinToolFactory(),
            DecoratedToolFactory(allowed_import_prefixes=("tests",)),
        ],
    )

    assert state.get("doubled") is not None
    completed = events_of_type(events, NodeCompletedEvent)
    assert any(e.node_id == "t1" for e in completed)


async def test_default_chain_rejects_decorated_imports() -> None:
    """X1: the default factory chain fails closed on decorated imports."""
    decl = ToolDeclaration(
        name="doubler",
        kind="decorated",
        config={"import": "tests.test_tool_node_resolution:sample_double"},
    )
    node = _tool_node({"ref": {"name": "doubler"}, "output_key": "doubled"})
    with pytest.raises(Exception, match="not allowed on this host"):
        await _run(_definition(node, tools=[decl]))


async def test_legacy_registry_fallback_when_resolver_has_no_legacy() -> None:
    """Injected resolvers without a legacy registry fall back to the executor's."""
    tool = RecordingTool(name="legacy_tool")
    registry = ToolRegistry()
    registry.register_builtin("legacy_tool", tool)
    resolver = ToolResolver(
        declarations=None,
        factories=[],
        factory_context=ToolFactoryContext(),
    )
    node = _tool_node({"ref": {"name": "legacy_tool"}, "output_key": "out"})
    _executor, state, _events = await _run(
        _definition(node), tool_resolver=resolver, tool_registry=registry
    )

    assert tool.calls == [{}]
    assert state.get("out") == "ok"


async def test_unresolved_mcp_ref_raises_pointed_error() -> None:
    node = _tool_node({"ref": {"type": "mcp", "name": "ghost"}, "output_key": "out"})
    with pytest.raises(Exception, match="not discovered"):
        await _run(_definition(node))


async def test_literals_merge_with_state_mapping() -> None:
    tool = RecordingTool()
    node = _tool_node({
        "ref": {"name": "fake_tool"},
        "input_mapping": {"prices": "loaded_prices"},
        "input_literals": {"window": 20},
        "output_key": "out",
    })
    definition = _definition(node)
    executor = WorkflowExecutor(definition, build_mock_llm_client())
    executor._resolver.override("fake_tool", tool)
    state = WorkflowState(query="q")
    state.append("init", "loaded_prices", [1.0, 2.0, 3.0])
    await collect_events(executor, state)

    assert tool.calls == [{"window": 20, "prices": [1.0, 2.0, 3.0]}]


def test_literal_mapping_collision_rejected() -> None:
    from databricks_deep_research.agents.config import ToolNodeConfig

    with pytest.raises(ValueError, match="same argument"):
        ToolNodeConfig(
            ref={"name": "x"},
            input_mapping={"a": "k"},
            input_literals={"a": 1},
        )


async def test_output_data_key_envelope_includes_success_and_error() -> None:
    failing = RecordingTool(
        name="fails",
        result=ToolResult(content="bad", success=False, data={"partial": 1}, error="boom"),
    )
    node = _tool_node({
        "ref": {"name": "fails"},
        "output_key": "out",
        "output_data_key": "out_data",
    })
    definition = _definition(node)
    executor = WorkflowExecutor(definition, build_mock_llm_client())
    executor._resolver.override("fails", failing)
    state = WorkflowState(query="q")
    await collect_events(executor, state)

    # fail_on_error defaults False: content stored (legacy semantics) and the
    # data envelope carries success/error for downstream conditionals.
    assert state.get("out") == "bad"
    assert state.get("out_data") == {"partial": 1, "success": False, "error": "boom"}


async def test_fail_on_error_raises_after_result_event() -> None:
    failing = RecordingTool(
        name="fails", result=ToolResult(content="bad", success=False, error="boom")
    )
    node = _tool_node({
        "ref": {"name": "fails"},
        "output_key": "out",
        "fail_on_error": True,
    })
    definition = _definition(node)
    executor = WorkflowExecutor(definition, build_mock_llm_client())
    executor._resolver.override("fails", failing)
    state = WorkflowState(query="q")

    events: list[StreamEvent] = []
    with pytest.raises(Exception, match="failed"):
        async for event in executor.execute(state):
            events.append(event)
    # The result event fires before the raise so consumers see the failure.
    result_events = events_of_type(events, ToolResultEvent)
    assert any(not e.tool_success for e in result_events)


async def test_enforce_output_schema_missing_required_key_raises() -> None:
    tool = RecordingTool(result=ToolResult(content="ok", data={"foo": 1}))
    node = _tool_node({
        "ref": {"name": "fake_tool"},
        "output_key": "out",
        "enforce_output_schema": True,
        "output_schema": {"type": "object", "required": ["status"]},
    })
    definition = _definition(node)
    executor = WorkflowExecutor(definition, build_mock_llm_client())
    executor._resolver.override("fake_tool", tool)
    with pytest.raises(Exception, match="required key"):
        await collect_events(executor, WorkflowState(query="q"))


async def test_tool_events_ordered_and_arguments_truncated() -> None:
    tool = RecordingTool()
    node = _tool_node({
        "ref": {"name": "fake_tool"},
        "input_mapping": {"doc": "big_doc"},
        "output_key": "out",
    })
    definition = _definition(node)
    executor = WorkflowExecutor(definition, build_mock_llm_client())
    executor._resolver.override("fake_tool", tool)
    state = WorkflowState(query="q")
    state.append("init", "big_doc", "x" * 2000)
    events = await collect_events(executor, state)

    node_events = [
        e for e in events
        if e.node_id == "t1"
        and isinstance(e, (NodeStartedEvent, ToolCallEvent, ToolResultEvent, NodeCompletedEvent))
    ]
    assert [type(e) for e in node_events] == [
        NodeStartedEvent, ToolCallEvent, ToolResultEvent, NodeCompletedEvent,
    ]
    call = events_of_type(events, ToolCallEvent)[0]
    assert len(call.arguments["doc"]) == 500  # truncated for the event payload
    assert len(tool.calls[0]["doc"]) == 2000  # tool received the full value


async def test_user_token_threaded_into_tool_context_extras() -> None:
    tool = RecordingTool()
    node = _tool_node({"ref": {"name": "fake_tool"}, "output_key": "out"})
    definition = _definition(node)
    executor = WorkflowExecutor(definition, build_mock_llm_client())
    executor._resolver.override("fake_tool", tool)
    await collect_events(executor, WorkflowState(query="q", user_token="tok-123"))

    assert tool.contexts[0].extras.get("user_token") == "tok-123"


async def test_bind_namespace_injects_into_compute_singleton() -> None:
    compute_decl = ToolDeclaration(name="compute", kind="compute", config={})
    tool = RecordingTool(result=ToolResult(content="ok", data={"result": [1, 2, 3]}))
    node = _tool_node({
        "ref": {"name": "fake_tool"},
        "output_key": "out",
        "bind_namespace": "sma_series",
    })
    definition = _definition(node, tools=[compute_decl])
    executor = WorkflowExecutor(definition, build_mock_llm_client())
    executor._resolver.override("fake_tool", tool)
    # Materialize the run's compute singleton (normally an agent binding does).
    compute = await executor._resolver.resolve("compute")
    await collect_events(executor, WorkflowState(query="q"))

    assert compute.get_variable("sma_series") == [1, 2, 3]


async def test_bind_namespace_fail_soft_without_compute() -> None:
    tool = RecordingTool()
    node = _tool_node({
        "ref": {"name": "fake_tool"},
        "output_key": "out",
        "bind_namespace": "unused",
    })
    definition = _definition(node)
    executor = WorkflowExecutor(definition, build_mock_llm_client())
    executor._resolver.override("fake_tool", tool)
    _state = WorkflowState(query="q")
    events = await collect_events(executor, _state)

    assert any(
        isinstance(e, NodeCompletedEvent) and e.node_id == "t1" for e in events
    )


async def test_tool_node_sources_admitted_to_pool() -> None:
    """Citeable DAG-step results flow through admission into the sources pool."""
    from databricks_deep_research.tools.protocol import SourceInfo

    long_content = (
        "Test query analysis: the 20-day simple moving average shows a steady "
        "upward drift across the test query window with strong evidence. "
    ) * 30
    tool = RecordingTool(
        name="citeable_fn",
        source_kind="qa_assistant",
        result=ToolResult(
            content=long_content,
            success=True,
            sources=[SourceInfo(
                url="function://citeable_fn/abc123",
                title="citeable_fn",
                snippet=long_content[:200],
                content=long_content,
                source_type="function",
                source_kind="qa_assistant",
            )],
        ),
    )
    node = _tool_node({"ref": {"name": "citeable_fn"}, "output_key": "out"})
    definition = WorkflowDefinition(
        id="pool-test", name="pool-test", root=node,
        pools=[{"name": "sources"}],
    )
    executor = WorkflowExecutor(definition, build_mock_llm_client())
    executor._resolver.override("citeable_fn", tool)
    state = WorkflowState(query="test query")
    events = await collect_events(executor, state)

    pool_items = executor._pools["sources"].snapshot()
    assert any(
        item.get("url") == "function://citeable_fn/abc123" for item in pool_items
    )
    result_event = events_of_type(events, ToolResultEvent)[0]
    assert result_event.raw_source_count == 1
    assert result_event.accepted_source_count == 1
    assert result_event.source_count == 1


async def test_builtin_tool_node_sources_bypass_pool() -> None:
    """builtin-kind results stay out of the evidence pool (compute parity)."""
    from databricks_deep_research.tools.protocol import SourceInfo

    tool = RecordingTool(
        name="plain_fn",
        source_kind="builtin",
        result=ToolResult(
            content="ok",
            sources=[SourceInfo(url="function://plain_fn/x", title="t")],
        ),
    )
    node = _tool_node({"ref": {"name": "plain_fn"}, "output_key": "out"})
    definition = WorkflowDefinition(
        id="pool-test-2", name="pool-test-2", root=node,
        pools=[{"name": "sources"}],
    )
    executor = WorkflowExecutor(definition, build_mock_llm_client())
    executor._resolver.override("plain_fn", tool)
    events = await collect_events(executor, WorkflowState(query="q"))

    assert executor._pools["sources"].snapshot() == []
    result_event = events_of_type(events, ToolResultEvent)[0]
    assert result_event.accepted_source_count == 0
    assert result_event.raw_source_count == 1


async def test_isolated_child_resolver_carries_overrides() -> None:
    tool = RecordingTool(name="fake_mcp")
    node = _tool_node({"ref": {"name": "noop"}, "output_key": "out"})
    definition = _definition(node)
    executor = WorkflowExecutor(definition, build_mock_llm_client())
    executor._resolver.override("fake_mcp", tool)

    child = executor._build_isolated_child_resolver(definition)
    resolved = await child.resolve({"type": "mcp", "name": "fake_mcp"})
    assert resolved is tool
