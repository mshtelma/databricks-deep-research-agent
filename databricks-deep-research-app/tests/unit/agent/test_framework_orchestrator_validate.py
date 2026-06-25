"""Anti-regression: ``ToolResolver.validate_all()`` MUST be awaited in
``framework_orchestrator.py`` before ``runner.stream(...)`` is called.

Layer 3 of the layered tool-context validation (see
``.claude/plans/let-s-design-the-validation-zany-tide.md``). Without this
guard, a workflow that declares a tool whose factory cannot construct
(e.g. ``schema_cache`` is ``None`` because ``STORAGE_WAREHOUSE_ID`` was
not propagated to the deployed app) burns coordinator + planner LLM
tokens before failing mid-stream with the misleading
``WorkflowError: Node 'X' is missing declared tools: [...]`` message.

This is encoded as a static AST test (rather than an integration test
through ``stream_research_via_framework``) because the orchestrator's
real path requires a workflow definition, executor, runner, etc. — the
existing integration tests in ``test_framework_orchestrator.py`` already
mock ``build_app_workflow_runner`` and never exercise the lines around
the ``validate_all`` call. A static check is the right granularity for
"this call must be present and ordered correctly".
"""

from __future__ import annotations

import ast
import pathlib


def _orchestrator_path() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    return (
        here.parents[3]
        / "src"
        / "deep_research"
        / "agent"
        / "framework_orchestrator.py"
    )


def _line_of_call(tree: ast.AST, predicate) -> int | None:
    """Return the lineno of the FIRST ``ast.Call`` node for which
    ``predicate(call)`` returns True, or None if no such call exists.

    Walks the entire tree because the orchestrator is a multi-thousand-
    line module and nesting depth varies — using ``ast.walk`` keeps the
    test resilient to local refactors that move the calls between
    ``async with`` blocks.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and predicate(node):
            return node.lineno
    return None


def _is_validate_all_call(node: ast.Call) -> bool:
    """``tool_resolver.validate_all()`` — Attribute call on
    ``tool_resolver`` whose attr is ``validate_all``."""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    return func.attr == "validate_all" and (
        isinstance(func.value, ast.Name) and func.value.id == "tool_resolver"
    )


def _is_runner_stream_call(node: ast.Call) -> bool:
    """``runner.stream(...)`` — Attribute call on ``runner`` whose attr
    is ``stream``."""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    return func.attr == "stream" and (
        isinstance(func.value, ast.Name) and func.value.id == "runner"
    )


class TestValidateAllIsAwaitedBeforeStream:
    def test_validate_all_call_is_present(self) -> None:
        """The orchestrator must call ``tool_resolver.validate_all()`` —
        if this fails, Layer 3 was deleted or refactored away."""
        tree = ast.parse(_orchestrator_path().read_text())
        assert _line_of_call(tree, _is_validate_all_call) is not None, (
            "framework_orchestrator.py must call tool_resolver.validate_all() "
            "before runner.stream(...). See Layer 3 of the layered "
            "tool-context validation plan."
        )

    def test_runner_stream_call_is_present(self) -> None:
        """Sanity: ``runner.stream`` must still exist — if this fails the
        orchestrator's structure changed enough that the ordering check
        below would be vacuous."""
        tree = ast.parse(_orchestrator_path().read_text())
        assert _line_of_call(tree, _is_runner_stream_call) is not None, (
            "framework_orchestrator.py no longer calls runner.stream(...). "
            "Update this anti-regression test to track the new entry point."
        )

    def test_validate_all_appears_before_runner_stream(self) -> None:
        """The whole point of Layer 3 is failing BEFORE LLM tokens are
        spent — validate_all must precede the runner.stream call in
        source order."""
        tree = ast.parse(_orchestrator_path().read_text())
        validate_line = _line_of_call(tree, _is_validate_all_call)
        stream_line = _line_of_call(tree, _is_runner_stream_call)
        assert validate_line is not None and stream_line is not None
        assert validate_line < stream_line, (
            f"tool_resolver.validate_all() at line {validate_line} must "
            f"appear BEFORE runner.stream() at line {stream_line}. Layer 3 "
            "guard relies on this ordering to fail before LLM tokens are spent."
        )

    def test_validate_all_is_awaited(self) -> None:
        """validate_all is async — calling it without await would silently
        no-op (the coroutine is never scheduled). Verify the call is
        wrapped in an ``Await`` node."""
        tree = ast.parse(_orchestrator_path().read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Await):
                inner = node.value
                if isinstance(inner, ast.Call) and _is_validate_all_call(inner):
                    return
        raise AssertionError(
            "tool_resolver.validate_all() must be awaited; an unawaited "
            "coroutine call here is a silent no-op (the Layer 3 guard would "
            "never run)."
        )
