"""Tests for the subworkflow node executor (``_exec_subworkflow``).

The subworkflow node runs a nested :class:`WorkflowDefinition` (supplied
``inline``) against a SEPARATE child :class:`WorkflowState` and a fresh child
:class:`WorkflowExecutor` that shares the parent's tool resolver / registries /
context. These tests drive the parent through the real ``execute()`` path
(via ``collect_events``) and assert on the parent state and pools.

Coverage:
* inline child runs and its mapped output appears in the parent (output_mapping)
* the child's terminal output is stored under ``output_key``
* depth guard trips when nesting exceeds ``max_subworkflow_depth``
* ``isolate`` keeps parent pools untouched; ``inherit`` makes child writes visible
* the SAME inline definition via TWO sibling subworkflow nodes does not raise a
  duplicate-id error (separate child states)
* a child node with ``error_handling.on_error="skip"`` that fails does not abort
  the parent subworkflow
* a workflow with NO subworkflow node never invokes ``_exec_subworkflow``
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from databricks_deep_research.agents.isolation import AgentOutput
from databricks_deep_research.api.vfs.in_memory import InMemoryBackend
from databricks_deep_research.errors import WorkflowError
from databricks_deep_research.tools.builtins.compute import PythonComputeTool
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.resolver import ToolResolver
from databricks_deep_research.workflow.definition import (
    NodeType,
    ToolDeclaration,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.executor import WorkflowExecutor
from databricks_deep_research.workflow.state import WorkflowState
from tests.conftest import (
    build_mock_llm_client as _mock_llm_client,
)
from tests.conftest import (
    collect_events as _collect_events,
)

# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


def _agent_node(
    node_id: str,
    *,
    output_key: str = "findings",
    pool_writes: list[dict[str, Any]] | None = None,
    error_handling: dict[str, Any] | None = None,
) -> WorkflowNode:
    """A minimal agent node. The patched ``execute_agent`` supplies its output."""
    config: dict[str, Any] = {"subtype": "researcher", "output_key": output_key}
    if pool_writes is not None:
        config["pool_writes"] = pool_writes
    return WorkflowNode(
        id=node_id,
        type=NodeType.agent,
        label=node_id,
        config=config,
        error_handling=error_handling,  # type: ignore[arg-type]
    )


def _inline_child(
    *,
    child_id: str = "child-wf",
    body: WorkflowNode | None = None,
    output_keys: list[str] | None = None,
    pools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Serialise a single-node child workflow definition to an ``inline`` dict."""
    root = body or _agent_node("child-agent", output_key="findings")
    defn = WorkflowDefinition(
        id=child_id,
        name=child_id,
        root=root,
        output_keys=output_keys or ["findings"],
        pools=pools or [],
    )
    return defn.model_dump(mode="python")


def _subworkflow_node(
    node_id: str,
    inline: dict[str, Any],
    *,
    output_mapping: dict[str, str] | None = None,
    output_key: str = "subworkflow_result",
    pool_mode: str = "inherit",
    params: dict[str, Any] | None = None,
    input_mapping: dict[str, str] | None = None,
    max_subworkflow_depth: int = 5,
) -> WorkflowNode:
    config: dict[str, Any] = {
        "ref": node_id,
        "inline": inline,
        "output_key": output_key,
        "pool_mode": pool_mode,
        "max_subworkflow_depth": max_subworkflow_depth,
    }
    if output_mapping is not None:
        config["output_mapping"] = output_mapping
    if params is not None:
        config["params"] = params
    if input_mapping is not None:
        config["input_mapping"] = input_mapping
    return WorkflowNode(
        id=node_id,
        type=NodeType.subworkflow,
        label=node_id,
        config=config,
    )


def _parent_defn(root: WorkflowNode, pools: list[dict[str, Any]] | None = None) -> WorkflowDefinition:
    return WorkflowDefinition(
        id="parent-wf",
        name="Parent Workflow",
        root=root,
        pools=pools or [],
        output_keys=["output"],
    )


def _fake_agent(
    *,
    value: str = "child output",
    pool_name: str | None = None,
    pool_item: dict[str, Any] | None = None,
    fail_node_ids: frozenset[str] = frozenset(),
) -> Any:
    """A stand-in for ``execute_agent``.

    Replicates the two harness behaviours these tests depend on: it appends the
    agent output to ``state`` under ``config.output_key`` and (optionally) adds
    one item to a named pool. Raising for ``fail_node_ids`` exercises the
    child's ``error_handling``.
    """

    async def fake_execute_agent(node_id: str, **kwargs: Any) -> AgentOutput:
        if node_id in fail_node_ids:
            raise RuntimeError(f"boom in {node_id}")
        config = kwargs["config"]
        state: WorkflowState = kwargs["state"]
        pools: dict[str, Any] = kwargs.get("pools") or {}
        state.append(node_id, config.output_key, value)
        if pool_name is not None and pool_name in pools and pool_item is not None:
            pools[pool_name].add(pool_item)
        return AgentOutput(content=value, output_key=config.output_key, events=[])

    return fake_execute_agent


# ---------------------------------------------------------------------------
# Output mapping + output_key
# ---------------------------------------------------------------------------


class TestSubworkflowOutputs:
    @pytest.mark.asyncio
    async def test_mapped_output_appears_in_parent_state(self) -> None:
        """``output_mapping`` copies a child key into the parent state."""
        inline = _inline_child(output_keys=["findings"])
        sub = _subworkflow_node(
            "sub",
            inline,
            output_mapping={"parent_findings": "findings"},
        )
        defn = _parent_defn(sub)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="q")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_fake_agent(value="mapped value"),
        ):
            await _collect_events(executor, state)

        assert state.get("parent_findings") == "mapped value"

    @pytest.mark.asyncio
    async def test_child_terminal_output_stored_under_output_key(self) -> None:
        """The child's last declared output key lands under ``output_key``."""
        inline = _inline_child(output_keys=["findings"])
        sub = _subworkflow_node("sub", inline, output_key="sub_result")
        defn = _parent_defn(sub)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="q")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_fake_agent(value="terminal value"),
        ):
            await _collect_events(executor, state)

        assert state.get("sub_result") == "terminal value"

    @pytest.mark.asyncio
    async def test_input_mapping_and_params_seed_child(self) -> None:
        """``input_mapping`` (from parent) and ``params`` (literal) reach the child."""
        seen: dict[str, Any] = {}

        async def capturing_agent(node_id: str, **kwargs: Any) -> AgentOutput:
            child_state: WorkflowState = kwargs["state"]
            seen["mapped"] = child_state.get("mapped_in")
            seen["param"] = child_state.get("literal_in")
            config = kwargs["config"]
            child_state.append(node_id, config.output_key, "ok")
            return AgentOutput(content="ok", output_key=config.output_key, events=[])

        inline = _inline_child(output_keys=["findings"])
        sub = _subworkflow_node(
            "sub",
            inline,
            input_mapping={"mapped_in": "parent_src"},
            params={"literal_in": "literal-value"},
        )
        defn = _parent_defn(sub)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="q")
        state.append("seed", "parent_src", "from-parent")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=capturing_agent,
        ):
            await _collect_events(executor, state)

        assert seen["mapped"] == "from-parent"
        assert seen["param"] == "literal-value"


# ---------------------------------------------------------------------------
# Depth guard
# ---------------------------------------------------------------------------


class TestDepthGuard:
    @pytest.mark.asyncio
    async def test_depth_exceeded_raises_workflow_error(self) -> None:
        """A subworkflow whose inline body itself nests a subworkflow beyond
        ``max_subworkflow_depth`` raises ``WorkflowError``."""
        # Innermost child (a plain agent).
        inner_inline = _inline_child(child_id="inner", output_keys=["findings"])
        # Middle child's root IS a subworkflow node embedding the innermost, with
        # a depth cap of 1. The middle executor runs at depth 1, so descending
        # into this inner sub computes next_depth=2 > 1 and trips the guard. The
        # cap is read from each node's OWN config, so it must live on the INNER
        # node (the one whose descent should be blocked).
        middle_sub = _subworkflow_node(
            "inner-sub", inner_inline, max_subworkflow_depth=1
        )
        middle_inline = WorkflowDefinition(
            id="middle",
            name="middle",
            root=middle_sub,
            output_keys=["subworkflow_result"],
        ).model_dump(mode="python")
        # Outer subworkflow runs at depth 0 -> descends to middle at depth 1 (ok).
        outer = _subworkflow_node("outer-sub", middle_inline)
        defn = _parent_defn(outer)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="q")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_fake_agent(),
        ), pytest.raises(WorkflowError, match="max_subworkflow_depth"):
            await _collect_events(executor, state)

    @pytest.mark.asyncio
    async def test_unresolvable_ref_without_inline_raises(self) -> None:
        """A bare ``ref`` with no ``inline`` and no registry raises ``WorkflowError``."""
        node = WorkflowNode(
            id="sub",
            type=NodeType.subworkflow,
            label="sub",
            config={"ref": "some-named-workflow"},
        )
        defn = _parent_defn(node)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="q")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_fake_agent(),
        ), pytest.raises(WorkflowError, match="not resolvable"):
            await _collect_events(executor, state)


# ---------------------------------------------------------------------------
# Pool modes
# ---------------------------------------------------------------------------


class TestPoolModes:
    @pytest.mark.asyncio
    async def test_inherit_makes_child_writes_visible_in_parent_pools(self) -> None:
        """``inherit`` binds the child to the parent pools, so child pool writes
        are visible to the parent."""
        child_agent = _agent_node(
            "child-agent",
            output_key="findings",
            pool_writes=[{"pool": "observations", "extract": "findings"}],
        )
        inline = _inline_child(body=child_agent, output_keys=["findings"])
        sub = _subworkflow_node("sub", inline, pool_mode="inherit")
        defn = _parent_defn(sub, pools=[{"name": "observations"}])
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="q")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_fake_agent(
                value="finding text",
                pool_name="observations",
                pool_item={"text": "child finding", "source": "child"},
            ),
        ):
            await _collect_events(executor, state)

        parent_pool = executor._pools.get("observations")
        assert parent_pool is not None
        assert parent_pool.count() == 1
        assert parent_pool.items[0]["text"] == "child finding"

    @pytest.mark.asyncio
    async def test_isolate_leaves_parent_pools_untouched(self) -> None:
        """``isolate`` runs the child against its own fresh pools; the parent
        pool stays empty."""
        child_agent = _agent_node(
            "child-agent",
            output_key="findings",
            pool_writes=[{"pool": "observations", "extract": "findings"}],
        )
        inline = _inline_child(
            body=child_agent,
            output_keys=["findings"],
            pools=[{"name": "observations"}],
        )
        sub = _subworkflow_node("sub", inline, pool_mode="isolate")
        defn = _parent_defn(sub, pools=[{"name": "observations"}])
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="q")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_fake_agent(
                value="finding text",
                pool_name="observations",
                pool_item={"text": "child finding", "source": "child"},
            ),
        ):
            await _collect_events(executor, state)

        parent_pool = executor._pools.get("observations")
        assert parent_pool is not None
        assert parent_pool.count() == 0

    @pytest.mark.asyncio
    async def test_merge_folds_child_pool_back_into_parent(self) -> None:
        """``merge`` runs isolated, then folds child pool items into the parent."""
        child_agent = _agent_node(
            "child-agent",
            output_key="findings",
            pool_writes=[{"pool": "observations", "extract": "findings"}],
        )
        inline = _inline_child(
            body=child_agent,
            output_keys=["findings"],
            pools=[{"name": "observations"}],
        )
        sub = _subworkflow_node("sub", inline, pool_mode="merge")
        defn = _parent_defn(sub, pools=[{"name": "observations"}])
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="q")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_fake_agent(
                value="finding text",
                pool_name="observations",
                pool_item={"text": "merged finding", "source": "child"},
            ),
        ):
            await _collect_events(executor, state)

        parent_pool = executor._pools.get("observations")
        assert parent_pool is not None
        assert parent_pool.count() == 1
        assert parent_pool.items[0]["text"] == "merged finding"


# ---------------------------------------------------------------------------
# Sibling reuse of the same inline definition
# ---------------------------------------------------------------------------


class TestSiblingReuse:
    @pytest.mark.asyncio
    async def test_same_inline_via_two_siblings_no_duplicate_id_error(self) -> None:
        """Two sibling subworkflow nodes running the SAME inline definition do
        not collide on duplicate node ids (each uses a separate child state)."""
        inline = _inline_child(child_id="shared", output_keys=["findings"])
        sub_a = _subworkflow_node(
            "sub-a", inline, output_mapping={"a_out": "findings"}
        )
        sub_b = _subworkflow_node(
            "sub-b", inline, output_mapping={"b_out": "findings"}
        )
        root = WorkflowNode(
            id="seq",
            type=NodeType.sequence,
            label="seq",
            children=[sub_a, sub_b],
        )
        defn = _parent_defn(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="q")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_fake_agent(value="sibling value"),
        ):
            await _collect_events(executor, state)

        assert state.get("a_out") == "sibling value"
        assert state.get("b_out") == "sibling value"


# ---------------------------------------------------------------------------
# Child error handling does not abort the parent
# ---------------------------------------------------------------------------


class TestChildErrorHandling:
    @pytest.mark.asyncio
    async def test_child_skip_on_error_does_not_abort_parent(self) -> None:
        """A child node with ``on_error='skip'`` that fails is skipped; the
        parent subworkflow completes and the second child still runs."""
        failing = _agent_node(
            "failing-child",
            output_key="failing_out",
            error_handling={"on_error": "skip"},
        )
        ok = _agent_node("ok-child", output_key="ok_out")
        child_seq = WorkflowNode(
            id="child-seq",
            type=NodeType.sequence,
            label="child-seq",
            children=[failing, ok],
        )
        inline = _inline_child(
            child_id="child-with-skip",
            body=child_seq,
            output_keys=["ok_out"],
        )
        sub = _subworkflow_node(
            "sub", inline, output_mapping={"parent_ok": "ok_out"}
        )
        defn = _parent_defn(sub)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="q")

        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_fake_agent(
                value="ok value", fail_node_ids=frozenset({"failing-child"})
            ),
        ):
            events = await _collect_events(executor, state)

        # Parent reached completion (no exception) and the surviving child output
        # propagated back.
        assert state.get("parent_ok") == "ok value"
        from databricks_deep_research.events.types import (
            NodeSkippedEvent,
            WorkflowCompletedEvent,
        )

        assert any(isinstance(e, WorkflowCompletedEvent) for e in events)
        skipped = [e for e in events if isinstance(e, NodeSkippedEvent)]
        assert any(e.node_id == "failing-child" for e in skipped)


# ---------------------------------------------------------------------------
# Default path: no subworkflow node
# ---------------------------------------------------------------------------


class TestNoSubworkflowNode:
    @pytest.mark.asyncio
    async def test_workflow_without_subworkflow_never_invokes_handler(self) -> None:
        """A plain agent workflow never dispatches ``_exec_subworkflow``."""
        root = _agent_node("solo", output_key="output")
        defn = _parent_defn(root)
        executor = WorkflowExecutor(defn, _mock_llm_client())
        state = WorkflowState(query="q")

        with patch.object(
            WorkflowExecutor, "_exec_subworkflow", autospec=True
        ) as spy, patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_fake_agent(value="solo value"),
        ):
            await _collect_events(executor, state)

        spy.assert_not_called()
        assert state.get("output") == "solo value"


# ---------------------------------------------------------------------------
# Scratchpad scope (compute namespace + VFS) per ``pool_mode`` — US-302
# ---------------------------------------------------------------------------


_COMPUTE_DECL = ToolDeclaration(name="compute", kind="compute")


def _compute_inline_child(*, output_keys: list[str] | None = None) -> dict[str, Any]:
    """An inline child whose single agent node binds the ``compute`` tool.

    Declaring ``compute`` both on the child definition (so the child resolver
    can construct it) and on the agent node (so ``_exec_agent`` resolves it and
    threads the instance into ``execute_agent(..., tools=...)``) lets a capturing
    fake agent observe the child's *actual* resolved compute instance.
    """
    agent = WorkflowNode(
        id="child-agent",
        type=NodeType.agent,
        label="child-agent",
        config={"subtype": "researcher", "output_key": "findings", "tools": ["compute"]},
    )
    defn = WorkflowDefinition(
        id="compute-child",
        name="compute-child",
        root=agent,
        output_keys=output_keys or ["findings"],
        tools=[_COMPUTE_DECL],
    )
    return defn.model_dump(mode="python")


def _capture_compute_agent(sink: dict[str, Any]) -> Any:
    """Fake ``execute_agent`` that records the resolved compute tool it received."""

    async def fake(node_id: str, **kwargs: Any) -> AgentOutput:
        tools: list[Any] = kwargs.get("tools") or []
        for tool in tools:
            if isinstance(tool, PythonComputeTool):
                sink["child_compute"] = tool
        config = kwargs["config"]
        state: WorkflowState = kwargs["state"]
        state.append(node_id, config.output_key, "ok")
        return AgentOutput(content="ok", output_key=config.output_key, events=[])

    return fake


def _parent_with_compute(
    sub: WorkflowNode,
    *,
    workspace_client: Any | None = None,
    user_token: str | None = None,
    vfs: Any | None = None,
) -> WorkflowExecutor:
    """Build a parent executor whose resolver can construct ``compute`` + VFS.

    Threads explicit identity (``workspace_client``/``user_token``) and a VFS
    through the factory context so the isolate path's identity-preservation and
    private-VFS behaviour can be asserted.
    """
    from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory

    extras: dict[str, Any] = {}
    if vfs is not None:
        extras["_framework_vfs"] = vfs
    factory_context = ToolFactoryContext(
        workspace_client=workspace_client,
        user_token=user_token,
        extras=extras,
    )
    resolver = ToolResolver(
        declarations=[_COMPUTE_DECL],
        factories=[BuiltinToolFactory()],
        factory_context=factory_context,
    )
    defn = _parent_defn(sub)
    return WorkflowExecutor(defn, _mock_llm_client(), tool_resolver=resolver)


class TestScratchpadScope:
    @pytest.mark.asyncio
    async def test_isolate_gives_child_fresh_compute_instance(self) -> None:
        """``isolate``: the child resolves a DIFFERENT compute instance than the
        parent, with an empty namespace — so a parent-injected variable is hidden
        and child writes don't reach the parent's namespace."""
        sub = _subworkflow_node(
            "sub", _compute_inline_child(), pool_mode="isolate"
        )
        executor = _parent_with_compute(sub)
        # Pre-seed the PARENT compute instance with a variable.
        parent_compute = await executor._resolver.resolve("compute")
        assert isinstance(parent_compute, PythonComputeTool)
        parent_compute.inject_variable("parent_var", 42)

        sink: dict[str, Any] = {}
        state = WorkflowState(query="q")
        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_capture_compute_agent(sink),
        ):
            await _collect_events(executor, state)

        child_compute = sink["child_compute"]
        assert isinstance(child_compute, PythonComputeTool)
        # Different instance and the parent's variable is NOT visible.
        assert child_compute is not parent_compute
        assert child_compute.list_user_namespace() == []
        # A write into the child namespace stays private to the child.
        child_compute.inject_variable("child_var", 99)
        parent_names = {e["name"] for e in parent_compute.list_user_namespace()}
        assert "child_var" not in parent_names
        assert "parent_var" in parent_names

    @pytest.mark.asyncio
    async def test_isolate_gives_child_private_vfs(self) -> None:
        """``isolate``: the child resolver carries a DIFFERENT ``_framework_vfs``
        than the parent's, so scratchpad files do not cross the boundary."""
        parent_vfs = InMemoryBackend()
        await parent_vfs.write("/parent.txt", "from parent")
        sub = _subworkflow_node(
            "sub", _compute_inline_child(), pool_mode="isolate"
        )
        executor = _parent_with_compute(sub, vfs=parent_vfs)

        child_resolver = executor._build_isolated_child_resolver(
            WorkflowDefinition.model_validate(sub.config["inline"])
        )
        child_vfs = child_resolver.factory_context.extras["_framework_vfs"]
        assert isinstance(child_vfs, InMemoryBackend)
        assert child_vfs is not parent_vfs
        # Child VFS starts empty; the parent's file is invisible to it.
        assert await child_vfs.exists("/parent.txt") is False
        # A child write does not appear in the parent VFS.
        await child_vfs.write("/child.txt", "from child")
        assert await parent_vfs.exists("/child.txt") is False
        assert await parent_vfs.exists("/parent.txt") is True

    @pytest.mark.asyncio
    async def test_inherit_shares_parent_compute_instance(self) -> None:
        """``inherit``: the child resolves the SAME compute singleton, so a
        parent-injected variable is visible (producer→consumer by handle)."""
        sub = _subworkflow_node(
            "sub", _compute_inline_child(), pool_mode="inherit"
        )
        executor = _parent_with_compute(sub)
        parent_compute = await executor._resolver.resolve("compute")
        assert isinstance(parent_compute, PythonComputeTool)
        parent_compute.inject_variable("shared_var", "hello")

        sink: dict[str, Any] = {}
        state = WorkflowState(query="q")
        with patch(
            "databricks_deep_research.workflow.executor.execute_agent",
            side_effect=_capture_compute_agent(sink),
        ):
            await _collect_events(executor, state)

        child_compute = sink["child_compute"]
        assert child_compute is parent_compute
        child_names = {e["name"] for e in child_compute.list_user_namespace()}
        assert "shared_var" in child_names

    @pytest.mark.asyncio
    async def test_isolate_preserves_obo_identity(self) -> None:
        """``isolate`` swaps the VFS + cache but PRESERVES identity fields, so the
        child runs as the same principal (OBO) as the parent."""
        sentinel_ws = object()
        sub = _subworkflow_node(
            "sub", _compute_inline_child(), pool_mode="isolate"
        )
        executor = _parent_with_compute(
            sub, workspace_client=sentinel_ws, user_token="obo-token-xyz"
        )

        child_resolver = executor._build_isolated_child_resolver(
            WorkflowDefinition.model_validate(sub.config["inline"])
        )
        child_ctx = child_resolver.factory_context
        parent_ctx = executor._resolver.factory_context
        # Identity carries over verbatim.
        assert child_ctx.workspace_client is sentinel_ws
        assert child_ctx.workspace_client is parent_ctx.workspace_client
        assert child_ctx.user_token == "obo-token-xyz"
        assert child_ctx.user_token == parent_ctx.user_token
        # But the scratchpad keys diverge: a private VFS and a fresh resolver
        # cache (so compute is a fresh instance, not the parent's singleton).
        assert (
            child_ctx.extras["_framework_vfs"]
            is not parent_ctx.extras.get("_framework_vfs")
        )
        assert child_ctx.extras["_resolver_cache"] is not parent_ctx.extras.get(
            "_resolver_cache"
        )
