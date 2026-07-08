"""python_function tool: schema, args, execution, factory gating, lifecycle."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from databricks_deep_research.tools.builtins.python_function import (
    PythonFunctionTool,
    compile_params_schema,
)
from databricks_deep_research.tools.code_executor import (
    SandboxSession,
    SandboxSessionHolder,
)
from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import ToolContext
from databricks_deep_research.workflow.definition import (
    NodeType,
    ToolDeclaration,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.executor import WorkflowExecutor, run_workflow
from databricks_deep_research.workflow.state import WorkflowState


class TestParamsSchema:
    def test_compiles_types_defaults_required(self) -> None:
        schema = compile_params_schema([
            {"name": "prices", "type": "array", "required": True},
            {"name": "window", "type": "int", "default": 20, "description": "w"},
        ])
        assert schema["properties"]["prices"]["type"] == "array"
        assert schema["properties"]["window"] == {
            "type": "integer", "description": "w", "default": 20,
        }
        assert schema["required"] == ["prices"]

    def test_invalid_param_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid name"):
            compile_params_schema([{"name": "not valid"}])


def _tool(**overrides: Any) -> PythonFunctionTool:
    session = overrides.pop("session", None)
    kwargs: dict[str, Any] = {
        "name": "fn",
        "code": "result = x + 1",
        "params": [{"name": "x", "type": "integer", "required": True}],
        "session_provider": (lambda: session) if session else None,
        "backend": "subprocess" if session else "restricted",
    }
    kwargs.update(overrides)
    return PythonFunctionTool(**kwargs)


class TestValidateArguments:
    def test_fills_defaults_and_drops_undeclared(self) -> None:
        session = MagicMock(spec=SandboxSession)
        tool = PythonFunctionTool(
            name="fn",
            code="result = 1",
            params=[
                {"name": "a", "type": "integer", "required": True},
                {"name": "b", "type": "integer", "default": 5},
            ],
            session_provider=lambda: session,
        )
        assert tool.validate_arguments({"a": 1, "zz": 9}) == {"a": 1, "b": 5}

    def test_missing_required_raises(self) -> None:
        session = MagicMock(spec=SandboxSession)
        tool = PythonFunctionTool(
            name="fn",
            code="result = 1",
            params=[{"name": "a", "required": True}],
            session_provider=lambda: session,
        )
        with pytest.raises(ValueError, match="missing required"):
            tool.validate_arguments({})


class TestExecuteThroughSession:
    async def test_execute_envelope_and_citeable_source(self) -> None:
        session = SandboxSession()
        try:
            tool = PythonFunctionTool(
                name="doubler",
                code="result = x * 2",
                params=[{"name": "x", "type": "integer", "required": True}],
                description="doubles x",
                session_provider=lambda: session,
                citeable=True,
            )
            result = await tool.execute({"x": 21}, ToolContext())
            assert result.success
            assert result.data["result"] == 42
            assert "doubler completed" in result.content
            assert len(result.sources) == 1
            assert result.sources[0].url.startswith("function://doubler/")
            assert result.sources[0].source_kind == "qa_assistant"
        finally:
            await session.close()

    async def test_bind_result_feeds_reads_namespace_chain(self) -> None:
        session = SandboxSession()
        try:
            producer = PythonFunctionTool(
                name="producer",
                code="result = [1.0, 2.0, 3.0]",
                session_provider=lambda: session,
                bind_result="series",
            )
            consumer = PythonFunctionTool(
                name="consumer",
                code="result = sum(series)",
                session_provider=lambda: session,
                reads_namespace=["series"],
            )
            assert (await producer.execute({}, ToolContext())).success
            result = await consumer.execute({}, ToolContext())
            assert result.success and result.data["result"] == 6.0
        finally:
            await session.close()

    async def test_reads_namespace_missing_is_clear_error(self) -> None:
        session = SandboxSession()
        try:
            tool = PythonFunctionTool(
                name="fn",
                code="result = ghost",
                session_provider=lambda: session,
                reads_namespace=["ghost"],
            )
            result = await tool.execute({}, ToolContext())
            assert result.success is False
            assert "reads_namespace" in (result.error or "")
        finally:
            await session.close()


def _decl(config: dict[str, Any], name: str = "fn") -> ToolDeclaration:
    return ToolDeclaration(name=name, kind="python_function", config=config)


class TestFactoryGating:
    async def test_construction_time_code_validation(self) -> None:
        factory = BuiltinToolFactory()
        with pytest.raises(ValueError, match="not allowed"):
            await factory.create(
                _decl({"code": "import os\nresult = 1"}), ToolFactoryContext()
            )

    async def test_unvetted_module_rejected(self) -> None:
        factory = BuiltinToolFactory()
        with pytest.raises(ValueError, match="not vetted"):
            await factory.create(
                _decl({"code": "result = 1", "extra_allowed_modules": ["requests"]}),
                ToolFactoryContext(),
            )

    async def test_restricted_backend_gated_by_default(self) -> None:
        factory = BuiltinToolFactory()
        with pytest.raises(ValueError, match="disabled on this host"):
            await factory.create(
                _decl({"code": "result = 1", "backend": "restricted"}),
                ToolFactoryContext(),
            )

    async def test_restricted_backend_allowed_via_host_switch(self) -> None:
        factory = BuiltinToolFactory()
        ctx = ToolFactoryContext(
            extras={"_allow_inprocess_python_function": True}
        )
        tool = await factory.create(
            _decl({"code": "result = 1", "backend": "restricted"}), ctx
        )
        result = await tool.execute({}, ToolContext())
        assert result.success and result.data["result"] == 1

    async def test_live_data_lib_mode_gated(self) -> None:
        factory = BuiltinToolFactory()
        with pytest.raises(ValueError, match="trust switch"):
            await factory.create(
                _decl({
                    "code": "result = 1",
                    "extra_allowed_modules": ["numpy"],
                    "data_lib_mode": "live",
                }),
                ToolFactoryContext(),
            )

    async def test_subprocess_tools_share_one_holder_session(self) -> None:
        factory = BuiltinToolFactory()
        ctx = ToolFactoryContext()
        tool_a = await factory.create(_decl({"code": "result = 1"}, name="a"), ctx)
        tool_b = await factory.create(_decl({"code": "result = 2"}, name="b"), ctx)
        holder = ctx.extras.get("_sandbox_session")
        assert isinstance(holder, SandboxSessionHolder)
        try:
            assert (await tool_a.execute({}, ToolContext())).success
            assert (await tool_b.execute({}, ToolContext())).success
            assert holder.peek() is not None
        finally:
            await holder.aclose()


class TestRunLifecycle:
    async def test_session_closed_at_run_end(self) -> None:
        defn = WorkflowDefinition(
            id="wf", name="wf",
            tools=[_decl({"code": "result = 7"})],
            root=WorkflowNode(
                id="t1", type=NodeType.tool, label="t1",
                config={"ref": {"name": "fn"}, "output_key": "out",
                        "output_data_key": "out_data"},
            ),
        )
        state, _events = await run_workflow(defn, MagicMock(), initial_state={"query": "q"})
        assert state.get("out_data")["result"] == 7

        # Re-derive the holder used by that run and assert it was drained.
        # run_workflow constructs its executor internally, so instead run once
        # more with an explicit executor to observe the lifecycle directly.
        executor = WorkflowExecutor(defn, MagicMock())
        run_state = WorkflowState(query="q")
        async for _ in executor.execute(run_state):
            pass
        holder = executor._resolver.factory_context.extras.get("_sandbox_session")
        assert isinstance(holder, SandboxSessionHolder)
        assert holder.peek() is None  # closed (and cleared) at run end

    async def test_isolated_child_resolver_gets_fresh_session_slot(self) -> None:
        defn = WorkflowDefinition(
            id="wf", name="wf",
            tools=[_decl({"code": "result = 7"})],
            root=WorkflowNode(
                id="t1", type=NodeType.tool, label="t1",
                config={"ref": {"name": "fn"}, "output_key": "out"},
            ),
        )
        executor = WorkflowExecutor(defn, MagicMock())
        executor._resolver.factory_context.extras["_sandbox_session"] = (
            SandboxSessionHolder()
        )
        child = executor._build_isolated_child_resolver(defn)
        assert "_sandbox_session" not in child.factory_context.extras
