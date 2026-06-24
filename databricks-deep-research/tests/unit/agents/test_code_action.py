"""Security + behaviour tests for the MemEx code-action bridge (spec §1.4).

This is the highest-security-scrutiny feature: a sandbox bridge that lets
agent-emitted Python call real research tools as functions. These tests are the
security-reviewer gate. They drive a REAL :class:`ReactLoop` with a REAL
:class:`PythonComputeTool` (so the gated spine + sandbox + sync-over-async
closure run end-to-end) and a fake LLM that emits ``compute`` tool calls whose
``code`` is the sandbox program under test.

Mandatory matrix (from the design contract):
1. A closure call decrements the per-tool budget; over-budget raises in-sandbox.
2. A ``requires_confirmation`` tool from code triggers HITL; denial raises
   in-sandbox, no execute.
3. Accepted sources from a code-action call reach the pool (parity w/ JSON).
4. Sandbox code CANNOT reach WorkspaceClient / ExecutionContext / auth token /
   exec globals / unintended tool methods (each blocked).
5. ``submit(non_jsonable)`` raises; ``submit(valid)`` becomes node output.
6. AST guard still blocks ``__globals__`` / ``__class__`` / ``open`` / ``eval``
   / import escapes WITH closures present.
7. ``action_mode="tools"`` (default) path is byte-identical (no closures, no
   ``submit``).
"""

from __future__ import annotations

import ast
import asyncio
import json
import sys
from typing import Any

import pytest

from databricks_deep_research.agents.code_action import (
    SUBMIT_NAME,
    CodeActionError,
    SubmitSink,
)
from databricks_deep_research.agents.react_loop import ReactLoop, ReactResult
from databricks_deep_research.llm.client import LLMResponse, ToolCall
from databricks_deep_research.tools.builtins.compute import (
    PythonComputeTool,
    _validate_ast,
)
from databricks_deep_research.tools.protocol import (
    SourceInfo,
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------


class _RecordingSearchTool:
    """A minimal real ``ResearchTool`` (web source-kind) that records calls.

    NOT a MagicMock — the gated spine + admission run against a concrete tool.
    ``execute`` returns one web source so admission (empty profile => accept-all
    for web) admits it, exercising the source/pool-write path.
    """

    def __init__(self, name: str = "web_search") -> None:
        self._name = name
        self.calls: list[dict[str, Any]] = []

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=f"recording {self._name}",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            source_type="web",
            source_kind=SourceKind.web,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not arguments.get("query"):
            raise ValueError("query required")
        return {"query": str(arguments["query"])}

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        return ToolResult(
            content=json.dumps({"answer": f"result for {arguments['query']}"}),
            success=True,
            sources=[
                SourceInfo(
                    url=f"https://example.com/{len(self.calls)}",
                    title=f"src {arguments['query']}",
                    snippet="snippet text",
                    source_type="web",
                    source_kind="web",
                    relevance_score=0.9,
                )
            ],
            # No tool_query => empty admission profile (current_step=None,
            # root_query="") => web accept-all, so the parity test exercises the
            # accepted-source -> pool path without fighting the relevance scorer.
            data={},
        )


class _ConfirmTool(_RecordingSearchTool):
    """A tool that requires HITL confirmation before executing."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=f"confirm {self._name}",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            source_type="web",
            source_kind=SourceKind.web,
            metadata={"requires_confirmation": True, "approval_reason": "sensitive"},
        )


class _DenyingBroker:
    """Approval broker that denies every request (no execute should follow)."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    async def request(
        self,
        request_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        reason: str = "",
        owner_user_id: str | None = None,
    ) -> Any:
        self.requests.append(tool_name)

        class _Decision:
            approved = False
            reason = "denied by policy"
            approver = "tester"

        return _Decision()


class _FakeLLM:
    """Emits a scripted sequence of LLM responses (compute tool calls)."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


def _compute_call(code: str, tc_id: str = "c1") -> ToolCall:
    return ToolCall(
        id=tc_id, function_name="compute", arguments=json.dumps({"code": code})
    )


def _resp(
    content: str = "", tool_calls: list[ToolCall] | None = None
) -> LLMResponse:
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        model="test",
    )


def _build_loop(
    *,
    code: str,
    tools_extra: list[Any] | None = None,
    code_action_tools: list[str] | None,
    action_mode: str = "code",
    per_tool_limits: dict[str, int] | None = None,
    tool_context: ToolContext | None = None,
    search_tool: _RecordingSearchTool | None = None,
    max_tool_calls: int = 5,
) -> tuple[ReactLoop, _RecordingSearchTool, PythonComputeTool, _FakeLLM]:
    """Build a ReactLoop that will run ``code`` once via a compute tool call."""
    search = search_tool or _RecordingSearchTool("web_search")
    compute = PythonComputeTool(name="compute")
    tools: list[Any] = [compute, search, *(tools_extra or [])]
    llm = _FakeLLM(
        [
            _resp(tool_calls=[_compute_call(code)]),
            _resp(content="DONE"),
        ]
    )
    loop = ReactLoop(
        llm,  # type: ignore[arg-type]
        tools,
        tool_context=tool_context,
        node_id="test-node",
        max_tool_calls=max_tool_calls,
        action_mode=action_mode,
        code_action_tools=code_action_tools,
        per_tool_limits=per_tool_limits,
    )
    return loop, search, compute, llm


# ---------------------------------------------------------------------------
# 7. action_mode="tools" default is byte-identical (no closures, no submit)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_tools_mode_injects_no_closures_or_submit() -> None:
    """Default action_mode='tools' must not inject submit() or tool closures."""
    # Sandbox code that references submit / web_search as names — under default
    # mode neither exists, so it raises NameError in-sandbox (surfaced as error).
    code = "submit({'x': 1})"
    loop, _search, compute, _llm = _build_loop(
        code=code, code_action_tools=["web_search"], action_mode="tools"
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    assert isinstance(result, ReactResult)
    # No reserved names, no submit injected.
    assert compute.get_variable(SUBMIT_NAME) is None
    assert compute.get_variable("web_search") is None
    assert compute._reserved_names == frozenset()
    assert loop._code_action_tool_names == ()
    assert loop._code_action_event_loop is None


@pytest.mark.asyncio
async def test_default_mode_setup_returns_none() -> None:
    """_setup_code_action is a no-op (None) under the default mode."""
    loop, _s, _c, _llm = _build_loop(
        code="1+1", code_action_tools=["web_search"], action_mode="tools"
    )
    loop._apply_step_tool_selection()
    assert loop._setup_code_action() is None


@pytest.mark.asyncio
async def test_default_mode_registers_no_code_action_hook() -> None:
    """Default mode never registers the code_action before-execute hook."""
    loop, _s, compute, _llm = _build_loop(
        code="1+1", code_action_tools=["web_search"], action_mode="tools"
    )
    await loop.execute([{"role": "user", "content": "hi"}])
    # The code-action hook key must be absent; only text-table-style hooks (if
    # any) may exist. No reserved names either.
    assert "code_action_callables" not in compute._before_execute_hooks
    assert compute._reserved_names == frozenset()


@pytest.mark.asyncio
async def test_default_mode_finalize_is_passthrough() -> None:
    """_finalize_result passes content/sources/events through unchanged.

    Proves the default action_mode='tools' result is byte-identical to a direct
    ReactResult construction (no code-action sinks, no submit override).
    """
    loop, _s, _c, _llm = _build_loop(
        code="1+1", code_action_tools=["web_search"], action_mode="tools"
    )
    sentinel_events: list[Any] = [object()]  # type: ignore[list-item]
    sentinel_sources: list[Any] = [{"url": "u"}]
    out = loop._finalize_result(
        content="hello",
        call_count=3,
        events=sentinel_events,  # type: ignore[arg-type]
        total_usage={"total_tokens": 9},
        sources=sentinel_sources,
        submit_sink=None,
    )
    assert out.content == "hello"
    assert out.tool_calls_made == 3
    assert out.events is sentinel_events  # same list object — no copy/merge
    assert out.sources is sentinel_sources
    assert out.token_usage == {"total_tokens": 9}


# ---------------------------------------------------------------------------
# 3. Accepted sources from a code-action call reach the pool (parity)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_action_sources_reach_pool() -> None:
    """A tool called from code admits sources into ReactResult.sources."""
    code = (
        "rows = web_search(query='hello')\n"
        "submit({'n': len(rows) if isinstance(rows, list) else 1})\n"
    )
    loop, search, _compute, _llm = _build_loop(
        code=code, code_action_tools=["web_search"]
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    # The real tool executed exactly once via the gated spine.
    assert search.calls == [{"query": "hello"}]
    # Accepted source flowed to ReactResult.sources (pool parity).
    assert len(result.sources) == 1
    assert result.sources[0]["url"] == "https://example.com/1"
    # submit() became the node output.
    assert json.loads(result.content) == {"n": 1}


@pytest.mark.asyncio
async def test_code_action_emits_tool_events() -> None:
    """A code-action call emits ToolCall + ToolResult events (observability)."""
    code = "web_search(query='q')\nsubmit('ok')\n"
    loop, _search, _compute, _llm = _build_loop(
        code=code, code_action_tools=["web_search"]
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    kinds = [e.event_type for e in result.events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds


# ---------------------------------------------------------------------------
# 5. submit() typed capture
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_valid_becomes_node_output() -> None:
    code = "submit({'answer': 42, 'items': [1, 2, 3]})"
    loop, _search, _compute, _llm = _build_loop(
        code=code, code_action_tools=["web_search"]
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    assert json.loads(result.content) == {"answer": 42, "items": [1, 2, 3]}


@pytest.mark.asyncio
async def test_submit_non_jsonable_raises_in_sandbox() -> None:
    """submit() of a non-JSON-able value raises (surfaced as a tool error).

    The error is caught by the compute sandbox's generic handler, so no value is
    recorded and the node content falls back to the final LLM content.
    """
    # A set is not JSON-serialisable.
    code = "submit({1, 2, 3})"
    loop, _search, _compute, _llm = _build_loop(
        code=code, code_action_tools=["web_search"]
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    # No submit recorded => content falls back to the final LLM content.
    assert result.content == "DONE"


def test_submit_sink_rejects_non_jsonable() -> None:
    sink = SubmitSink()
    with pytest.raises(CodeActionError, match="JSON-serialisable"):
        sink.record({1, 2, 3})
    assert sink.submitted is False


def test_submit_sink_records_valid() -> None:
    sink = SubmitSink()
    sink.record({"ok": True})
    assert sink.submitted is True
    assert sink.value == {"ok": True}


# ---------------------------------------------------------------------------
# 1. Per-tool budget decrement + over-budget raises in-sandbox
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_action_decrements_per_tool_budget() -> None:
    """Each closure call ticks the SAME per-tool counter as the JSON path."""
    code = (
        "web_search(query='a')\n"
        "web_search(query='b')\n"
        "submit('done')\n"
    )
    loop, search, _compute, _llm = _build_loop(
        code=code,
        code_action_tools=["web_search"],
        per_tool_limits={"web_search": 5},
    )
    await loop.execute([{"role": "user", "content": "hi"}])
    assert loop._per_tool_counts["web_search"] == 2
    assert len(search.calls) == 2


@pytest.mark.asyncio
async def test_code_action_over_budget_raises_no_execute() -> None:
    """Once the per-tool budget is hit, further code calls raise (no execute)."""
    # limit=1: first call executes, second is budget-exhausted (raises in-sandbox).
    code = (
        "web_search(query='a')\n"
        "try:\n"
        "    web_search(query='b')\n"
        "    over = 'NOT_RAISED'\n"
        "except Exception as e:\n"
        "    over = 'RAISED'\n"
        "submit(over)\n"
    )
    loop, search, _compute, _llm = _build_loop(
        code=code,
        code_action_tools=["web_search"],
        per_tool_limits={"web_search": 1},
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    assert json.loads(result.content) == "RAISED"
    # Only the first (within-budget) call actually executed the tool.
    assert len(search.calls) == 1


# ---------------------------------------------------------------------------
# 2. HITL: requires_confirmation tool from code triggers gate; denial raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_action_hitl_denial_raises_no_execute() -> None:
    """A denied HITL tool called from code raises in-sandbox and never executes."""
    confirm = _ConfirmTool("sensitive_search")
    broker = _DenyingBroker()
    ctx = ToolContext(
        extras={
            "_framework_approval_broker": broker,
            "_framework_user_id": "u1",
        }
    )
    code = (
        "try:\n"
        "    sensitive_search(query='secret')\n"
        "    out = 'NOT_RAISED'\n"
        "except Exception:\n"
        "    out = 'DENIED'\n"
        "submit(out)\n"
    )
    loop, _search, _compute, _llm = _build_loop(
        code=code,
        tools_extra=[confirm],
        code_action_tools=["sensitive_search"],
        tool_context=ctx,
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    assert json.loads(result.content) == "DENIED"
    # Broker was consulted; the tool NEVER executed.
    assert broker.requests == ["sensitive_search"]
    assert confirm.calls == []


# ---------------------------------------------------------------------------
# 4. Capability model: sandbox cannot reach client/context/auth/globals/methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_closure_returns_object_not_toolresult() -> None:
    """The closure return value is the coerced object, not a ToolResult."""
    code = (
        "r = web_search(query='x')\n"
        "submit({'type': type(r).__name__})\n"
    )
    loop, _search, _compute, _llm = _build_loop(
        code=code, code_action_tools=["web_search"]
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    # coerce_to_object parses the JSON dict the tool returned.
    assert json.loads(result.content) == {"type": "dict"}


@pytest.mark.asyncio
async def test_closure_has_no_reachable_context_or_client() -> None:
    """Probe the closure object: no __self__/__closure__ exposes ctx/client.

    Sandbox dunder access (__closure__, __globals__, __self__) is AST-blocked,
    so the model cannot even attempt this. We additionally assert from OUTSIDE
    the sandbox that the closure's cell contents contain no ExecutionContext /
    WorkspaceClient / token — only a weakref + the tool name string.
    """
    code = "submit('ok')"
    loop, _search, compute, _llm = _build_loop(
        code=code, code_action_tools=["web_search"]
    )
    await loop.execute([{"role": "user", "content": "hi"}])
    closure = compute.get_variable("web_search")
    assert closure is not None
    cells = closure.__closure__ or ()
    contents = [c.cell_contents for c in cells]
    import weakref as _wr

    # Every captured cell is either a weakref (to the loop) or a plain str
    # (the tool name) — never a context/client/token/dict.
    for value in contents:
        assert isinstance(value, (_wr.ref, str)), (
            f"closure captured a reachable object: {type(value).__name__}"
        )
    # The tool context (which holds extras/broker) must NOT be among captures.
    assert all(not isinstance(v, ToolContext) for v in contents)


@pytest.mark.parametrize(
    "forbidden",
    ["loop", "ctx", "context", "client", "ws", "token", "self", "exec_globals"],
)
@pytest.mark.asyncio
async def test_sandbox_cannot_reference_framework_objects(forbidden: str) -> None:
    """No framework object (loop/ctx/client/token) is reachable by name.

    The sandbox has no ``dir()``/``globals()``/``vars()`` builtins (enumeration
    is impossible by design — a strong property), so we probe each forbidden
    name directly: referencing it raises ``NameError`` in-sandbox.
    """
    code = (
        f"try:\n"
        f"    _x = {forbidden}\n"
        f"    out = 'REACHABLE'\n"
        f"except NameError:\n"
        f"    out = 'BLOCKED'\n"
        f"submit(out)\n"
    )
    loop, _search, _compute, _llm = _build_loop(
        code=code, code_action_tools=["web_search"]
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    assert json.loads(result.content) == "BLOCKED"


@pytest.mark.asyncio
async def test_sandbox_bridged_names_resolve() -> None:
    """The bridged tool + submit DO resolve in the sandbox (positive control)."""
    code = (
        "ok = callable(web_search) and callable(submit)\n"
        "submit(ok)\n"
    )
    loop, _search, _compute, _llm = _build_loop(
        code=code, code_action_tools=["web_search"]
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    assert json.loads(result.content) is True


# ---------------------------------------------------------------------------
# 6. AST guard still blocks escapes WITH closures present
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "escape_code",
    [
        "web_search.__globals__",
        "().__class__.__bases__",
        "open('/etc/passwd')",
        "eval('1+1')",
        "exec('x=1')",
        "__import__('os')",
        "web_search.__closure__",
        "submit.__globals__['__builtins__']",
    ],
)
@pytest.mark.asyncio
async def test_ast_guard_blocks_escapes_with_closures_present(
    escape_code: str,
) -> None:
    """Classic sandbox escapes remain blocked even with closures injected."""
    code = f"x = {escape_code}\nsubmit('SHOULD_NOT_REACH')\n"
    loop, _search, _compute, _llm = _build_loop(
        code=code, code_action_tools=["web_search"]
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    # The escape was blocked => submit never ran => content falls back to "DONE".
    assert result.content == "DONE"


@pytest.mark.parametrize(
    "rebind_code",
    [
        "submit = lambda v: None",
        "web_search = lambda **k: []",
        "def submit(v):\n    return None",
        "del submit",
        "import math as submit",
        "for submit in [1, 2]:\n    pass",
        "submit, x = (lambda v: None), 1",
        # `with ... as <reserved>` / `except ... as <reserved>` are bindings too
        # (the gap the security-reviewer found: optional_vars / handler name were
        # not walked by the reserved-name guard).
        "with x as submit:\n    pass",
        "with x as (submit, y):\n    pass",
        "with x as web_search:\n    pass",
        "try:\n    pass\nexcept Exception as submit:\n    pass",
        # match patterns + lambda param (security-reviewer round 2) — end-to-end.
        "match 1:\n    case submit:\n        pass",
        "match 1:\n    case [*submit]:\n        pass",
        "f = lambda submit: 1",
    ],
)
@pytest.mark.asyncio
async def test_reserved_names_cannot_be_rebound(rebind_code: str) -> None:
    """Sandbox code cannot shadow submit / tool closures (AST-blocked)."""
    code = f"{rebind_code}\nsubmit('SHOULD_NOT_REACH')\n"
    loop, _search, _compute, _llm = _build_loop(
        code=code, code_action_tools=["web_search"]
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    # Rebinding rejected at AST validation => no submit => fallback content.
    assert result.content == "DONE"


# ---------------------------------------------------------------------------
# Allowlist enforcement (Top-Fix 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_allowlisted_tool_not_bridged() -> None:
    """A bound tool NOT on code_action_tools gets no closure."""
    # web_search is bound but the allowlist is empty => not bridged.
    code = "submit('ok')"
    loop, _search, compute, _llm = _build_loop(
        code=code, code_action_tools=[]
    )
    await loop.execute([{"role": "user", "content": "hi"}])
    # Empty allowlist => code-action inactive => no closures, no submit.
    assert compute.get_variable("web_search") is None
    assert compute.get_variable(SUBMIT_NAME) is None


@pytest.mark.asyncio
async def test_compute_itself_never_bridged() -> None:
    """Even if 'compute' is on the allowlist it is not exposed as a closure."""
    code = "submit('ok')"
    loop, _search, compute, _llm = _build_loop(
        code=code, code_action_tools=["compute", "web_search"]
    )
    await loop.execute([{"role": "user", "content": "hi"}])
    # compute is builtin source-kind => filtered out; web_search remains.
    assert "compute" not in loop._code_action_tool_names
    assert "web_search" in loop._code_action_tool_names
    # submit is reserved but is NOT a bridged tool name.
    assert SUBMIT_NAME not in loop._code_action_tool_names


# ---------------------------------------------------------------------------
# HIGH-fix regression: reserved-name binding via with/except (AST guard gap)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src",
    [
        # with / except as (security-reviewer round 1)
        "with cm() as submit:\n    pass",
        "with cm() as (submit, other):\n    pass",
        "with a() as x, b() as submit:\n    pass",
        "try:\n    pass\nexcept Exception as submit:\n    pass",
        "async def f():\n    async with cm() as submit:\n        pass",
        # match capture / star / mapping-rest (security-reviewer round 2)
        "match v:\n    case submit:\n        pass",
        "match v:\n    case [a, submit]:\n        pass",
        "match v:\n    case [*submit]:\n        pass",
        "match v:\n    case {'k': submit}:\n        pass",
        "match v:\n    case {**submit}:\n        pass",
        "match v:\n    case object() as submit:\n        pass",
        # lambda parameter + starred unpacking (completeness sweep)
        "f = lambda submit: submit",
        "*submit, x = [1, 2]",
    ],
)
def test_validate_ast_rejects_reserved_binding_any_form(src: str) -> None:
    """``_validate_ast`` rejects rebinding a reserved name via ANY binding form.

    Direct unit pin for both security-reviewer rounds: ``with``/``except ... as``
    (round 1) and ``match`` patterns + ``type``/lambda/star (round 2). The guard
    is complete-by-construction — a Store/Del ``Name`` check plus the ``str``-named
    binders — so adding a construct cannot silently reopen the shadowing escape.
    """
    tree = ast.parse(src)
    with pytest.raises(ValueError, match="reserved name 'submit'"):
        _validate_ast(tree, frozenset({"submit"}))


@pytest.mark.parametrize(
    "src",
    [
        "with cm() as other:\n    pass",
        "try:\n    pass\nexcept Exception as other:\n    pass",
        "with cm() as submit:\n    pass",  # reserved set EMPTY => allowed
    ],
)
def test_validate_ast_allows_non_reserved_or_empty(src: str) -> None:
    """A non-reserved name (or empty reserved set) leaves with/except bindings.

    Proves the guard is inert in the default path (``reserved_names`` empty —
    the byte-identical ``action_mode='tools'`` contract) and fires only for an
    actual reserved name.
    """
    reserved = frozenset() if "as submit" in src else frozenset({"submit"})
    _validate_ast(ast.parse(src), reserved)  # must not raise


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="PEP 695 type-parameter syntax requires Python 3.12+ (on 3.11 it is a "
    "SyntaxError, rejected before the reserved-name guard runs)",
)
@pytest.mark.parametrize(
    "src",
    [
        "def f[submit]():\n    pass",
        "class C[submit]:\n    pass",
        "type X[submit] = int",
    ],
)
def test_validate_ast_rejects_reserved_pep695_type_params(src: str) -> None:
    """PEP 695 type parameters (3.12+) cannot shadow a reserved name either.

    Closes the last binding form: ``type_params`` is walked version-safely so a
    ``def f[submit]()`` cannot rebind the gated ``submit`` on 3.12+.
    """
    with pytest.raises(ValueError, match="reserved name 'submit'"):
        _validate_ast(ast.parse(src), frozenset({"submit"}))


# ---------------------------------------------------------------------------
# H2-fix: a code-action call is time-bounded — the worker thread always releases
# ---------------------------------------------------------------------------


class _HangingTool(_RecordingSearchTool):
    """A web tool whose ``execute`` blocks far longer than the call timeout."""

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        self.calls.append(dict(arguments))
        await asyncio.sleep(5)  # >> the tiny per-call timeout set by the test
        return ToolResult(content="{}", success=True, sources=[], data={})


@pytest.mark.asyncio
async def test_code_action_call_timeout_releases_and_raises() -> None:
    """A stuck gated call hits the per-call timeout and raises in-sandbox.

    Guards the no-timeout thread-pool-exhaustion DoS: the closure's blocking
    wait is bounded, so the 2-slot compute worker pool is released and the
    sandbox sees a normal tool error (caught here -> 'TIMEOUT').
    """
    hanging = _HangingTool("slow_search")
    code = (
        "try:\n"
        "    slow_search(query='x')\n"
        "    out = 'NO_TIMEOUT'\n"
        "except Exception:\n"
        "    out = 'TIMEOUT'\n"
        "submit(out)\n"
    )
    loop, _s, _c, _llm = _build_loop(
        code=code,
        tools_extra=[hanging],
        code_action_tools=["slow_search"],
    )
    loop._code_action_call_timeout = 0.3  # bound the wait for the test
    result = await loop.execute([{"role": "user", "content": "hi"}])
    assert json.loads(result.content) == "TIMEOUT"
    # The tool's execute() WAS entered (the call reached the gated spine) before
    # the timeout cancelled it.
    assert len(hanging.calls) == 1


# ---------------------------------------------------------------------------
# MEDIUM-fix: pandas/numpy are NOT auto-enabled in the code-action sandbox
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pandas_numpy_not_auto_enabled() -> None:
    """Code-action must not silently widen the sandbox with pandas/numpy.

    ``numpy.load(allow_pickle=True)`` / ``pandas.read_pickle`` are file/network/
    RCE reach via plain attribute calls; auto-enabling them would bypass the AST
    sandbox. The bridge works without them (a safe subset is a deferred,
    separately-reviewed follow-up).
    """
    code = (
        "try:\n"
        "    import numpy\n"
        "    out = 'IMPORTED'\n"
        "except Exception:\n"
        "    out = 'BLOCKED'\n"
        "submit(out)\n"
    )
    loop, _s, compute, _llm = _build_loop(
        code=code, code_action_tools=["web_search"]
    )
    result = await loop.execute([{"role": "user", "content": "hi"}])
    # Import is rejected in-sandbox (security property — primary assertion).
    assert json.loads(result.content) == "BLOCKED"
    # And neither module leaked into the sandbox's import allowlist / namespace.
    assert "numpy" not in compute._allowed_modules
    assert "pandas" not in compute._allowed_modules
    assert compute.get_variable("numpy") is None
    assert compute.get_variable("pandas") is None
