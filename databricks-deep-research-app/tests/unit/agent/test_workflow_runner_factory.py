"""Tests for ``build_app_workflow_runner`` and its anti-regression guardrail.

These tests encode the project convention: ALL app workflow execution must
construct the ``WorkflowRunner`` via ``build_app_workflow_runner`` so the
context auto-detects ``BRAVE_API_KEY`` / ``JINA_API_KEY`` from env.

The grep test at the bottom is the lint that prevents a recurrence of the
2026-05-24 incident where ``framework_orchestrator`` silently dropped the
Brave key by bypassing ``ToolFactoryContext.from_defaults``.
"""

from __future__ import annotations

import ast
import pathlib
from unittest.mock import MagicMock

import pytest
from databricks_deep_research import WorkflowRunner
from databricks_deep_research.llm.client import FrameworkLLMClient

from deep_research.agent.workflow_runner_factory import (
    build_app_workflow_runner,
)


def _fake_llm() -> MagicMock:
    """Return a MagicMock that passes isinstance checks for the LLM client."""
    mock = MagicMock(spec=FrameworkLLMClient)
    return mock


class TestBuildAppWorkflowRunner:
    def test_reads_brave_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """BRAVE_API_KEY in env → search_client + api_keys['brave'] populated."""
        monkeypatch.setenv("BRAVE_API_KEY", "test-key-12345")
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(), workspace_client=None, user_token=None,
        )
        assert isinstance(runner, WorkflowRunner)
        ctx = runner.factory_context
        assert ctx.search_client is not None, (
            "BraveSearchAdapter must be built when BRAVE_API_KEY is set"
        )
        assert ctx.api_keys.get("brave") == "test-key-12345"

    def test_missing_brave_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without BRAVE_API_KEY: search_client is None, api_keys lacks 'brave'.

        With ``strict_tool_resolution=True`` on the runner, declared
        web_research / brave_search tools then fail loudly at executor
        startup — the desired behavior (vs the historical silent
        Insufficient-Evidence pattern)."""
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(), workspace_client=None, user_token=None,
        )
        ctx = runner.factory_context
        assert ctx.search_client is None
        assert "brave" not in ctx.api_keys

    def test_empty_string_brave_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BRAVE_API_KEY='' (empty string) is treated like missing — defensive
        check because secret scopes sometimes round-trip to empty strings."""
        monkeypatch.setenv("BRAVE_API_KEY", "")
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(), workspace_client=None, user_token=None,
        )
        ctx = runner.factory_context
        assert ctx.search_client is None
        assert "brave" not in ctx.api_keys

    def test_workspace_client_preserved(self) -> None:
        """Caller's explicit workspace_client is preserved verbatim — must NOT
        be replaced by from_defaults's auto-detection branch."""
        sentinel = object()
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(),
            workspace_client=sentinel,  # type: ignore[arg-type]
            user_token="obo-token-xyz",
        )
        ctx = runner.factory_context
        assert ctx.workspace_client is sentinel
        assert ctx.user_token == "obo-token-xyz"

    def test_returns_fresh_runner_per_call(self) -> None:
        """Each call returns an independent WorkflowRunner — the runner is
        documented as not thread-safe, so per-request construction is the
        project convention."""
        a = build_app_workflow_runner(
            llm_client=_fake_llm(), workspace_client=None, user_token=None,
        )
        b = build_app_workflow_runner(
            llm_client=_fake_llm(), workspace_client=None, user_token=None,
        )
        assert a is not b
        assert a.factory_context is not b.factory_context


class TestAntiRegressionLint:
    """Lint: no direct WorkflowExecutor / ToolFactoryContext construction in
    app code. Encoded as a test so CI fails on a recurrence of the
    2026-05-24 silent-failure incident.

    Uses AST inspection rather than regex so docstrings and comments that
    mention the class names verbatim aren't false-positives.
    """

    # Names that, when called as constructors in app code, indicate a bypass
    # of ``build_app_workflow_runner``.
    _FORBIDDEN_NAMES = {"WorkflowExecutor", "ToolFactoryContext"}

    # Only the helper module is allowed to call these constructors (it wraps
    # ``ToolFactoryContext.from_defaults`` for the entire app).
    _ALLOWLIST_FILES = {"workflow_runner_factory.py"}

    @classmethod
    def _app_src_root(cls) -> pathlib.Path:
        # tests/unit/agent/test_workflow_runner_factory.py → src/deep_research
        here = pathlib.Path(__file__).resolve()
        return here.parents[3] / "src" / "deep_research"

    @classmethod
    def _scan_file(cls, py: pathlib.Path) -> list[str]:
        """Return a list of "path:line: rendered" entries for forbidden calls.

        AST walk — finds ``Call`` nodes whose ``func`` is either
        ``Name("WorkflowExecutor"|"ToolFactoryContext")`` (e.g.,
        ``WorkflowExecutor(...)``) or ``Attribute`` whose value is one of
        the forbidden names (e.g., ``module.WorkflowExecutor(...)``).
        Does NOT flag ``.from_defaults(...)`` because that is a classmethod
        producing a properly-initialized context — though the only app-code
        site that should call it lives inside the allowlisted helper.
        """
        try:
            tree = ast.parse(py.read_text(), filename=str(py))
        except SyntaxError:
            return []
        hits: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            target_name: str | None = None
            if isinstance(func, ast.Name):
                target_name = func.id
            elif isinstance(func, ast.Attribute):
                target_name = func.attr
            if target_name not in cls._FORBIDDEN_NAMES:
                continue
            # For Attribute calls, ignore ``.from_defaults`` chains — the
            # attribute being called is the class name, but if the call is
            # actually ``ToolFactoryContext.from_defaults(...)`` the outer
            # node's func.attr is ``from_defaults`` (not in forbidden set),
            # so it never reaches here. This branch only catches direct
            # ``X(...)`` constructors.
            hits.append(f"{py}:{node.lineno}")
        return hits

    def test_app_code_has_no_direct_workflow_construction(self) -> None:
        src = self._app_src_root()
        assert src.is_dir(), f"Expected app src at {src}, not found"
        offenders: list[str] = []
        for py in src.rglob("*.py"):
            if py.name in self._ALLOWLIST_FILES:
                continue
            for hit in self._scan_file(py):
                rel = pathlib.Path(hit).relative_to(src.parent)
                offenders.append(str(rel))
        assert not offenders, (
            "Direct WorkflowExecutor / ToolFactoryContext construction in "
            "app code bypasses BRAVE_API_KEY auto-detection. Use "
            "build_app_workflow_runner(...) instead. Offenders:\n"
            + "\n".join(offenders)
        )
