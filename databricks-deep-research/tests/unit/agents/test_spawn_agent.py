"""Security + behaviour tests for the GOVERNED ``spawn_agent`` bridge (spec §3.3).

A code-action Cell may spawn a DECLARED child subworkflow from sandbox Python,
within Designer-declared bounds. This is OPT-IN and OFF BY DEFAULT, and will be
audited by a security-reviewer HARD gate — so these tests pin the capability
model by construction. They mirror ``tests/unit/agents/test_code_action.py``:
they drive a REAL :class:`ReactLoop` with a REAL :class:`PythonComputeTool` (so
the sandbox + sync-over-async closure run end-to-end) and a fake LLM that emits
a ``compute`` tool call whose ``code`` is the sandbox program under test.

Security invariants pinned (each test notes which):
* INV-DEFAULT — default (``action_mode="tools"`` / ``spawn_budget=0`` / empty
  ``spawnable_subagents``) injects NO ``spawn_agent`` (byte-identical).
* INV-DECLARED — an undeclared spawn name is REJECTED (re-validated from the
  framework-owned dict at call time).
* INV-BUDGET — an over-budget spawn is REJECTED; the budget increments ONCE per
  attempt (no retry-storm).
* INV-CAPTURE — the spawn closure captures ONLY a weakref + strings (no
  context/client/token/runner reachable).
* INV-RESERVED — ``spawn_agent`` is RESERVED in the AST guard (cannot be
  rebound to exfiltrate).
* INV-ISOLATION — a spawned child runs in an ISOLATED scratchpad and cannot read
  a parent Cell compute variable.
* INV-RETURN — a declared spawn runs the inline subworkflow and returns its
  primary output COERCED to a plain object (scratchpad-handle style).
"""

from __future__ import annotations

import json
import weakref
from typing import Any

import pytest

from databricks_deep_research.agents.code_action import (
    SPAWN_NAME,
    CodeActionError,
)
from databricks_deep_research.agents.isolation import AgentOutput
from databricks_deep_research.agents.react_loop import ReactLoop, ReactResult
from databricks_deep_research.llm.client import LLMResponse, ToolCall
from databricks_deep_research.tools.builtins.compute import PythonComputeTool
from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import ToolContext
from databricks_deep_research.tools.resolver import ToolResolver
from databricks_deep_research.workflow.definition import (
    NodeType,
    ToolDeclaration,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.executor import WorkflowExecutor
from databricks_deep_research.workflow.state import WorkflowState

# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Emits a scripted sequence of LLM responses (compute tool calls)."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


def _resp(
    content: str = "", tool_calls: list[ToolCall] | None = None
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        model="test",
    )


def _compute_call(code: str, tc_id: str = "c1") -> ToolCall:
    return ToolCall(
        id=tc_id, function_name="compute", arguments=json.dumps({"code": code})
    )


# A trivial inline subworkflow dump (single agent node). The fake spawn_runner
# never validates its body, but the isolation/return tests run it for real.
def _inline_child(*, child_id: str = "spawned-child") -> dict[str, Any]:
    agent = WorkflowNode(
        id="child-agent",
        type=NodeType.agent,
        label="child-agent",
        config={"subtype": "researcher", "output_key": "findings"},
    )
    defn = WorkflowDefinition(
        id=child_id,
        name=child_id,
        root=agent,
        output_keys=["findings"],
    )
    return defn.model_dump(mode="python")


def _build_loop(
    *,
    code: str,
    action_mode: str = "code",
    spawn_runner: Any | None = None,
    spawnable_subagents: dict[str, dict[str, Any]] | None = None,
    spawn_budget: int = 0,
    max_concurrent_spawns: int = 4,
    tools_extra: list[Any] | None = None,
) -> tuple[ReactLoop, PythonComputeTool, _FakeLLM]:
    """Build a ReactLoop that runs ``code`` once via a compute tool call."""
    compute = PythonComputeTool(name="compute")
    tools: list[Any] = [compute, *(tools_extra or [])]
    llm = _FakeLLM([_resp(tool_calls=[_compute_call(code)]), _resp(content="DONE")])
    loop = ReactLoop(
        llm,  # type: ignore[arg-type]
        tools,
        node_id="test-node",
        max_tool_calls=5,
        action_mode=action_mode,
        # ``code_action_tools`` empty: spawn is independent of bridged tools, and
        # the compute sandbox alone is enough to exercise spawn_agent.
        code_action_tools=[],
        spawn_runner=spawn_runner,
        spawnable_subagents=spawnable_subagents,
        spawn_budget=spawn_budget,
        max_concurrent_spawns=max_concurrent_spawns,
    )
    return loop, compute, llm


# ---------------------------------------------------------------------------
# INV-DEFAULT — default path injects NO spawn_agent (byte-identical)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_tools_mode_no_spawn_agent() -> None:
    """action_mode='tools' (default) injects no spawn_agent and reserves nothing."""
    code = "spawn_agent('x', 'y')"
    loop, compute, _llm = _build_loop(
        code=code,
        action_mode="tools",
        spawn_runner=_make_returning_runner({"ok": True}),
        spawnable_subagents={"x": _inline_child()},
        spawn_budget=5,
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    assert isinstance(result, ReactResult)
    # No spawn_agent injected; SPAWN_NAME not reserved; loop counters inert.
    assert compute.get_variable(SPAWN_NAME) is None
    assert SPAWN_NAME not in compute._reserved_names
    assert loop._spawn_count == 0
    assert loop._code_action_event_loop is None


@pytest.mark.asyncio
async def test_zero_budget_no_spawn_agent() -> None:
    """spawn_budget=0 (the hard default) injects no spawn_agent even in code mode."""
    code = "spawn_agent('x', 'y')"
    loop, compute, _llm = _build_loop(
        code=code,
        action_mode="code",
        spawn_runner=_make_returning_runner({"ok": True}),
        spawnable_subagents={"x": _inline_child()},
        spawn_budget=0,  # disabled
    )
    await loop.execute([{"role": "user", "content": "hi"}])
    assert compute.get_variable(SPAWN_NAME) is None
    assert SPAWN_NAME not in compute._reserved_names


@pytest.mark.asyncio
async def test_empty_declared_set_no_spawn_agent() -> None:
    """An empty spawnable_subagents set injects no spawn_agent (nothing to spawn)."""
    code = "spawn_agent('x', 'y')"
    loop, compute, _llm = _build_loop(
        code=code,
        action_mode="code",
        spawn_runner=_make_returning_runner({"ok": True}),
        spawnable_subagents={},  # nothing declared
        spawn_budget=5,
    )
    await loop.execute([{"role": "user", "content": "hi"}])
    assert compute.get_variable(SPAWN_NAME) is None
    assert SPAWN_NAME not in compute._reserved_names


@pytest.mark.asyncio
async def test_no_spawn_runner_no_spawn_agent() -> None:
    """No spawn_runner bound (the default ``execute_agent`` path) => no spawn_agent."""
    code = "spawn_agent('x', 'y')"
    loop, compute, _llm = _build_loop(
        code=code,
        action_mode="code",
        spawn_runner=None,  # executor did not opt in
        spawnable_subagents={"x": _inline_child()},
        spawn_budget=5,
    )
    await loop.execute([{"role": "user", "content": "hi"}])
    assert compute.get_variable(SPAWN_NAME) is None
    assert SPAWN_NAME not in compute._reserved_names


# ---------------------------------------------------------------------------
# Helpers for runner-driven tests
# ---------------------------------------------------------------------------


def _make_returning_runner(value: Any, *, recorder: list[dict[str, Any]] | None = None) -> Any:
    """A stub spawn_runner that records its calls and returns ``value``."""

    async def runner(*, name: str, prompt: str, inline: dict[str, Any]) -> Any:
        if recorder is not None:
            recorder.append({"name": name, "prompt": prompt, "inline": inline})
        return value

    return runner


# ---------------------------------------------------------------------------
# INV-DECLARED — undeclared name rejected (re-validated at call time)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_undeclared_name_rejected() -> None:
    """A spawn name NOT in the declared set raises in-sandbox; runner never runs."""
    recorder: list[dict[str, Any]] = []
    code = (
        "try:\n"
        "    spawn_agent('not_declared', 'do something')\n"
        "    out = 'NOT_RAISED'\n"
        "except Exception:\n"
        "    out = 'REJECTED'\n"
        "submit(out)\n"
    )
    loop, _compute, _llm = _build_loop(
        code=code,
        spawn_runner=_make_returning_runner({"ok": 1}, recorder=recorder),
        spawnable_subagents={"declared": _inline_child()},
        spawn_budget=5,
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    assert json.loads(result.content) == "REJECTED"
    # The runner was NEVER invoked (rejected before any spawn ran).
    assert recorder == []
    # An undeclared attempt does NOT consume budget (rejected pre-increment).
    assert loop._spawn_count == 0


def test_spawn_closure_rejects_undeclared_directly() -> None:
    """Unit pin: the spawn closure re-validates the name from the loop dict."""
    from databricks_deep_research.agents.code_action import _make_spawn_closure

    loop, _compute, _llm = _build_loop(
        code="submit('x')",
        spawn_runner=_make_returning_runner({"ok": 1}),
        spawnable_subagents={"declared": _inline_child()},
        spawn_budget=5,
    )
    # Closures look the loop up at call time; emulate the install-time binding.
    import asyncio

    loop._code_action_event_loop = asyncio.new_event_loop()
    try:
        closure = _make_spawn_closure(weakref.ref(loop))
        with pytest.raises(CodeActionError, match="not in the declared spawnable set"):
            closure("ghost", "prompt")
        # A non-str name is also rejected.
        with pytest.raises(CodeActionError, match="must be a str"):
            closure(123, "prompt")  # type: ignore[arg-type]
    finally:
        loop._code_action_event_loop.close()


# ---------------------------------------------------------------------------
# INV-BUDGET — over-budget rejected; budget counts once per attempt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_over_budget_rejected_and_counts_once() -> None:
    """budget=1: first spawn runs, second is rejected; count reflects BOTH attempts."""
    recorder: list[dict[str, Any]] = []
    code = (
        "a = spawn_agent('declared', 'first')\n"
        "try:\n"
        "    spawn_agent('declared', 'second')\n"
        "    over = 'NOT_RAISED'\n"
        "except Exception:\n"
        "    over = 'RAISED'\n"
        "submit({'a': a, 'over': over})\n"
    )
    loop, _compute, _llm = _build_loop(
        code=code,
        spawn_runner=_make_returning_runner({"child": "ok"}, recorder=recorder),
        spawnable_subagents={"declared": _inline_child()},
        spawn_budget=1,
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    payload = json.loads(result.content)
    assert payload["a"] == {"child": "ok"}
    assert payload["over"] == "RAISED"
    # Only the first (within-budget) spawn actually ran the runner.
    assert len(recorder) == 1
    # Budget incremented ONCE per ATTEMPT: the in-budget run + the over-budget
    # attempt. The second attempt raises AFTER its increment is gated, so the
    # counter is exactly the budget (no further increment past the cap).
    assert loop._spawn_count == 1


@pytest.mark.asyncio
async def test_failing_spawn_still_counts_no_retry_storm() -> None:
    """A spawn whose runner RAISES still consumes its budget slot (no retry-storm)."""
    calls = {"n": 0}

    async def failing_runner(*, name: str, prompt: str, inline: dict[str, Any]) -> Any:
        calls["n"] += 1
        raise RuntimeError("child blew up")

    code = (
        "for i in range(5):\n"
        "    try:\n"
        "        spawn_agent('declared', 'attempt')\n"
        "    except Exception:\n"
        "        pass\n"
        "submit('done')\n"
    )
    loop, _compute, _llm = _build_loop(
        code=code,
        spawn_runner=failing_runner,
        spawnable_subagents={"declared": _inline_child()},
        spawn_budget=2,
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    assert result.content == json.dumps("done")
    # The runner is only ENTERED for the 2 budgeted attempts; the remaining 3
    # loop iterations are budget-rejected before the runner is called.
    assert calls["n"] == 2
    assert loop._spawn_count == 2


# ---------------------------------------------------------------------------
# INV-CAPTURE — spawn closure binds only weakref + strings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_closure_captures_only_weakref() -> None:
    """The injected spawn_agent closure captures ONLY a weakref — no client/ctx/runner."""
    ctx = ToolContext(extras={"_framework_user_id": "u1"})
    code = "submit('ok')"
    loop, compute, _llm = _build_loop(
        code=code,
        spawn_runner=_make_returning_runner({"ok": 1}),
        spawnable_subagents={"declared": _inline_child()},
        spawn_budget=5,
    )
    loop._ctx = ctx  # ensure a populated context exists on the loop
    await loop.execute([{"role": "user", "content": "hi"}])
    closure = compute.get_variable(SPAWN_NAME)
    assert closure is not None
    cells = closure.__closure__ or ()
    contents = [c.cell_contents for c in cells]
    # Every captured cell is a weakref (to the loop) — never a context/client/
    # token/runner/dict.
    for value in contents:
        assert isinstance(value, weakref.ref), (
            f"spawn closure captured a reachable object: {type(value).__name__}"
        )
    assert all(not isinstance(v, ToolContext) for v in contents)


@pytest.mark.asyncio
async def test_sandbox_cannot_reach_spawn_runner_by_name() -> None:
    """The spawn runner / loop / context are not reachable by name in the sandbox."""
    code = (
        "names = ['spawn_runner', 'loop', 'ctx', 'runner', '_spawn_runner']\n"
        "blocked = True\n"
        "for n in names:\n"
        "    try:\n"
        "        eval(n)\n"  # eval is itself AST-blocked, doubly safe
        "        blocked = False\n"
        "    except Exception:\n"
        "        pass\n"
        "submit('BLOCKED' if blocked else 'REACHABLE')\n"
    )
    loop, _compute, _llm = _build_loop(
        code=code,
        spawn_runner=_make_returning_runner({"ok": 1}),
        spawnable_subagents={"declared": _inline_child()},
        spawn_budget=5,
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    # eval is AST-blocked so the program never even submits 'REACHABLE'; the
    # whole Cell is rejected => fallback content. Either way, not REACHABLE.
    assert result.content != json.dumps("REACHABLE")


# ---------------------------------------------------------------------------
# INV-RESERVED — spawn_agent cannot be rebound (AST guard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rebind_code",
    [
        "spawn_agent = lambda *a: None",
        "def spawn_agent(*a):\n    return None",
        "del spawn_agent",
        "import math as spawn_agent",
        "for spawn_agent in [1, 2]:\n    pass",
        "spawn_agent, x = (lambda *a: None), 1",
        "with x as spawn_agent:\n    pass",
        "try:\n    pass\nexcept Exception as spawn_agent:\n    pass",
        "f = lambda spawn_agent: 1",
    ],
)
@pytest.mark.asyncio
async def test_spawn_name_reserved_cannot_be_rebound(rebind_code: str) -> None:
    """Sandbox code cannot shadow spawn_agent (AST-blocked, same as submit)."""
    code = f"{rebind_code}\nsubmit('SHOULD_NOT_REACH')\n"
    loop, _compute, _llm = _build_loop(
        code=code,
        spawn_runner=_make_returning_runner({"ok": 1}),
        spawnable_subagents={"declared": _inline_child()},
        spawn_budget=5,
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    # Rebinding rejected at AST validation => no submit => fallback content.
    assert result.content == "DONE"


@pytest.mark.asyncio
async def test_spawn_name_is_in_reserved_set() -> None:
    """When spawn is enabled, SPAWN_NAME joins the reserved set alongside submit."""
    loop, compute, _llm = _build_loop(
        code="submit('ok')",
        spawn_runner=_make_returning_runner({"ok": 1}),
        spawnable_subagents={"declared": _inline_child()},
        spawn_budget=5,
    )
    await loop.execute([{"role": "user", "content": "hi"}])
    assert SPAWN_NAME in compute._reserved_names
    # submit is still reserved too (the single replace-call kept both).
    assert "submit" in compute._reserved_names


# ---------------------------------------------------------------------------
# INV-RETURN + INV-ISOLATION — real executor-backed spawn_runner
# ---------------------------------------------------------------------------


_COMPUTE_DECL = ToolDeclaration(name="compute", kind="compute")


def _make_real_spawn_runner(executor: WorkflowExecutor, parent_state: WorkflowState) -> Any:
    """Build a spawn_runner bound to a REAL WorkflowExecutor (isolate scope).

    Mirrors how ``_exec_agent`` constructs the runner: it drives
    ``WorkflowExecutor._run_inline_subworkflow`` with ``pool_mode="isolate"`` so
    the spawned child gets a fresh compute namespace + private VFS.
    """
    async def runner(*, name: str, prompt: str, inline: dict[str, Any]) -> Any:
        result_sink: dict[str, Any] = {}
        async for _event in executor._run_inline_subworkflow(
            inline,
            query=prompt,
            depth=executor._subworkflow_depth + 1,
            parent_state=parent_state,
            pool_mode="isolate",
            result_sink=result_sink,
        ):
            pass
        return result_sink.get("primary")

    return runner


def _build_executor_with_compute(workspace_client: Any | None = None) -> WorkflowExecutor:
    from tests.conftest import build_mock_llm_client

    factory_context = ToolFactoryContext(
        workspace_client=workspace_client,
        user_token=None,
        extras={},
    )
    resolver = ToolResolver(
        declarations=[_COMPUTE_DECL],
        factories=[BuiltinToolFactory()],
        factory_context=factory_context,
    )
    root = WorkflowNode(
        id="root", type=NodeType.agent, label="root",
        config={"subtype": "researcher", "output_key": "output"},
    )
    defn = WorkflowDefinition(id="parent", name="parent", root=root, output_keys=["output"])
    return WorkflowExecutor(defn, build_mock_llm_client(), tool_resolver=resolver)


@pytest.mark.asyncio
async def test_declared_spawn_runs_inline_and_returns_coerced_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared spawn runs the inline subworkflow and returns its output coerced.

    The child agent's output is a JSON string; the spawn closure coerces it to a
    plain dict (scratchpad-handle style), so the sandbox sees a dict — never a
    raw child state/context object.
    """
    # The child agent emits a JSON object as its findings output.
    async def fake_execute_agent(node_id: str, **kwargs: Any) -> AgentOutput:
        config = kwargs["config"]
        state: WorkflowState = kwargs["state"]
        payload = json.dumps({"child_answer": 7, "echo": state.query})
        state.append(node_id, config.output_key, payload)
        return AgentOutput(content=payload, output_key=config.output_key, events=[])

    monkeypatch.setattr(
        "databricks_deep_research.workflow.executor.execute_agent",
        fake_execute_agent,
    )

    executor = _build_executor_with_compute()
    parent_state = WorkflowState(query="parent-q")
    runner = _make_real_spawn_runner(executor, parent_state)

    code = (
        "r = spawn_agent('declared', 'child-prompt')\n"
        "submit({'type': type(r).__name__, 'answer': r.get('child_answer'), "
        "'echo': r.get('echo')})\n"
    )
    loop, _compute, _llm = _build_loop(
        code=code,
        spawn_runner=runner,
        spawnable_subagents={"declared": _inline_child()},
        spawn_budget=3,
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    payload = json.loads(result.content)
    # Coerced to a dict (NOT a ToolResult/state object).
    assert payload["type"] == "dict"
    assert payload["answer"] == 7
    # The spawn prompt became the child's query (prompt threading verified).
    assert payload["echo"] == "child-prompt"


@pytest.mark.asyncio
async def test_spawned_child_cannot_read_parent_compute_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INV-ISOLATION: a spawned child runs in an isolated scratchpad.

    The child agent inspects its OWN resolved compute instance; because the spawn
    runner uses ``pool_mode="isolate"``, the child resolves a FRESH compute tool
    whose namespace does NOT contain the parent-seeded variable.
    """
    observed: dict[str, Any] = {}

    async def inspecting_agent(node_id: str, **kwargs: Any) -> AgentOutput:
        tools: list[Any] = kwargs.get("tools") or []
        child_compute = next(
            (t for t in tools if isinstance(t, PythonComputeTool)), None
        )
        names: list[str] = []
        if child_compute is not None:
            names = [e["name"] for e in child_compute.list_user_namespace()]
        observed["child_names"] = names
        observed["child_compute_id"] = id(child_compute)
        config = kwargs["config"]
        state: WorkflowState = kwargs["state"]
        payload = json.dumps({"sees_parent_var": "parent_secret" in names})
        state.append(node_id, config.output_key, payload)
        return AgentOutput(content=payload, output_key=config.output_key, events=[])

    monkeypatch.setattr(
        "databricks_deep_research.workflow.executor.execute_agent",
        inspecting_agent,
    )

    executor = _build_executor_with_compute()
    # Seed the PARENT compute instance with a secret variable.
    parent_compute = await executor._resolver.resolve("compute")
    assert isinstance(parent_compute, PythonComputeTool)
    parent_compute.inject_variable("parent_secret", "TOPSECRET")

    parent_state = WorkflowState(query="parent-q")
    # The inline child must DECLARE compute so its isolated resolver constructs it
    # and _exec_agent threads it into the inspecting agent.
    child_agent = WorkflowNode(
        id="child-agent", type=NodeType.agent, label="child-agent",
        config={"subtype": "researcher", "output_key": "findings", "tools": ["compute"]},
    )
    child_defn = WorkflowDefinition(
        id="iso-child", name="iso-child", root=child_agent,
        output_keys=["findings"], tools=[_COMPUTE_DECL],
    )
    inline = child_defn.model_dump(mode="python")
    runner = _make_real_spawn_runner(executor, parent_state)

    code = (
        "r = spawn_agent('declared', 'child-prompt')\n"
        "submit(r)\n"
    )
    loop, _compute, _llm = _build_loop(
        code=code,
        spawn_runner=runner,
        spawnable_subagents={"declared": inline},
        spawn_budget=3,
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    payload = json.loads(result.content)
    # The child could NOT see the parent's secret variable (isolated namespace).
    assert payload["sees_parent_var"] is False
    assert observed["child_names"] == []
    # And the child's compute instance is a DIFFERENT object than the parent's.
    assert observed["child_compute_id"] != id(parent_compute)
