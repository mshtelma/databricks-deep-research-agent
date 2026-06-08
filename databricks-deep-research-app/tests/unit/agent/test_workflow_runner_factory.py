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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from databricks_deep_research import WorkflowRunner
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.tools.builtins.text_table import (
    BindingInfo,
    BindingSource,
    TableBindingRegistry,
)

from deep_research.agent.adapters.table_discovery_adapter import (
    DesignerTableDiscoveryProvider,
)
from deep_research.agent.workflow_runner_factory import (
    assert_runtime_can_satisfy_workflows,
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
        # Pin the global provider to brave so the default-search-client wiring is
        # a no-op; this test exercises the brave-no-key path. (The databricks
        # default's ctx.search_client behavior is covered separately in
        # test_workflow_runner_factory_search_client.py.)
        monkeypatch.setattr(
            "deep_research.core.app_config.get_app_config",
            lambda: SimpleNamespace(search=SimpleNamespace(provider="brave")),
        )
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
        # Brave-path test: pin provider so the default (databricks) wiring no-ops.
        monkeypatch.setattr(
            "deep_research.core.app_config.get_app_config",
            lambda: SimpleNamespace(search=SimpleNamespace(provider="brave")),
        )
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(), workspace_client=None, user_token=None,
        )
        ctx = runner.factory_context
        assert ctx.search_client is None
        assert "brave" not in ctx.api_keys

    def test_workspace_client_preserved(self) -> None:
        """Caller's explicit workspace_client is preserved verbatim — must NOT
        be replaced by from_defaults's auto-detection branch. (When a user
        token IS present, an OBO client is derived from it instead — that path
        is covered by TestOBOResolution.)"""
        sentinel = object()
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(),
            workspace_client=sentinel,  # type: ignore[arg-type]
            user_token=None,
        )
        ctx = runner.factory_context
        assert ctx.workspace_client is sentinel
        assert ctx.user_token is None

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

    def test_table_registry_and_discovery_provider_are_wired(self) -> None:
        """Every runner gets a fresh TableBindingRegistry + a
        DesignerTableDiscoveryProvider so the framework's table_* tools can
        construct without a separate setup call."""
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(), workspace_client=None, user_token=None,
        )
        ctx = runner.factory_context
        assert isinstance(ctx.table_registry, TableBindingRegistry)
        assert isinstance(
            ctx.table_discovery_provider, DesignerTableDiscoveryProvider
        )

    def test_table_sql_runtime_dependencies_wired_when_warehouse_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A workspace client plus warehouse id wires SchemaCache + SQL executor."""
        monkeypatch.setenv("TABLE_TOOLS_WAREHOUSE_ID", "wh-123")
        statement_execution = MagicMock()
        statement_execution.execute_statement.return_value = SimpleNamespace(
            statement_id="stmt-1",
            status=SimpleNamespace(state="SUCCEEDED"),
            manifest=SimpleNamespace(
                schema=SimpleNamespace(
                    columns=[
                        SimpleNamespace(name="col_name"),
                        SimpleNamespace(name="data_type"),
                    ]
                )
            ),
            result=SimpleNamespace(
                data_array=[
                    ["id", "string"],
                    ["text", "string"],
                ]
            ),
        )
        workspace_client = SimpleNamespace(statement_execution=statement_execution)

        runner = build_app_workflow_runner(
            llm_client=_fake_llm(),
            workspace_client=workspace_client,  # type: ignore[arg-type]
            user_token=None,
        )

        ctx = runner.factory_context
        assert ctx.schema_cache is not None
        assert ctx.sql_executor is not None
        schema = ctx.schema_cache.get("cat.sch.docs", "obo-token")
        assert [col.name for col in schema.columns] == ["id", "text"]
        statement_execution.execute_statement.assert_called_once()
        call = statement_execution.execute_statement.call_args.kwargs
        assert call["statement"] == "DESCRIBE TABLE `cat`.`sch`.`docs`"
        assert call["warehouse_id"] == "wh-123"

    def test_table_registries_are_independent_per_runner(self) -> None:
        """Discovered bindings in one runner must not leak into another —
        registries are per-request state."""
        a = build_app_workflow_runner(
            llm_client=_fake_llm(), workspace_client=None, user_token=None,
        )
        b = build_app_workflow_runner(
            llm_client=_fake_llm(), workspace_client=None, user_token=None,
        )
        assert a.factory_context.table_registry is not (
            b.factory_context.table_registry
        )

    def test_static_bindings_flow_into_discovery_provider(self) -> None:
        """Designer-supplied static bindings are visible from
        ``list_tables`` even with no UC scopes / workspace client."""
        import asyncio

        static = BindingInfo(
            name="alpha",
            fqn="cat.sch.alpha",
            source=BindingSource.DISCOVERED,
        )
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(),
            workspace_client=None,
            user_token=None,
            table_static_bindings=[static],
        )
        provider = runner.factory_context.table_discovery_provider
        assert isinstance(provider, DesignerTableDiscoveryProvider)
        out = asyncio.run(provider.list_tables(user_token=""))
        assert {info.name for info in out} == {"alpha"}


class TestAssertRuntimeCanSatisfyWorkflows:
    """Layer 2 boot-time guard: the helper must group every unmet ctx field
    with the kinds blocked by it and emit a remediation hint per blocker.

    Encoded as anti-regression so a future ``ToolFactoryContext`` field
    rename or hint deletion fails CI loudly instead of silently regressing
    the failure-mode introduced on 2026-05-27 (text-table tools failing
    mid-stream because ``STORAGE_WAREHOUSE_ID`` propagated nowhere)."""

    def test_no_op_when_all_required_fields_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BRAVE_API_KEY", "k")
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(),
            workspace_client=None,
            user_token=None,
        )
        ctx = runner.factory_context
        # No declared kinds → no required fields → no error.
        assert_runtime_can_satisfy_workflows(ctx, declared_kinds=[])

    def test_no_op_when_kind_has_no_required_ctx(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``web_crawl`` has no required ctx fields → never blocks."""
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(),
            workspace_client=None,
            user_token=None,
        )
        assert_runtime_can_satisfy_workflows(
            runner.factory_context, declared_kinds=["web_crawl"]
        )

    def test_raises_when_search_client_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without BRAVE_API_KEY, ``web_search`` declares search_client →
        helper must raise listing the missing field, the blocked kind, and
        the BRAVE-key remediation hint."""
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        # Pin provider to brave so no default search_client is wired — this test
        # exercises the boot guard's "search_client missing" path. (Under the
        # databricks default the backend IS present, so the guard would not fire
        # — that positive case is covered by the search-client factory tests.)
        monkeypatch.setattr(
            "deep_research.core.app_config.get_app_config",
            lambda: SimpleNamespace(search=SimpleNamespace(provider="brave")),
        )
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(),
            workspace_client=None,
            user_token=None,
        )
        with pytest.raises(RuntimeError) as exc_info:
            assert_runtime_can_satisfy_workflows(
                runner.factory_context,
                declared_kinds=["web_search", "web_research"],
            )
        message = str(exc_info.value)
        assert "APP_BOOT_TOOL_DEPS_MISSING:" in message
        assert "search_client=None" in message
        assert "web_search" in message and "web_research" in message
        assert "BRAVE_API_KEY" in message  # remediation hint

    def test_groups_multiple_kinds_under_one_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Five table_* kinds share schema_cache + sql_executor; the helper
        must group them under the missing field rather than emit five
        separate sentences."""
        monkeypatch.delenv("STORAGE_WAREHOUSE_ID", raising=False)
        monkeypatch.delenv("TABLE_TOOLS_WAREHOUSE_ID", raising=False)
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(),
            workspace_client=None,
            user_token=None,
        )
        with pytest.raises(RuntimeError) as exc_info:
            assert_runtime_can_satisfy_workflows(
                runner.factory_context,
                declared_kinds=[
                    "table_search",
                    "table_read",
                    "table_neighbors",
                ],
            )
        message = str(exc_info.value)
        # All three kinds must appear under each shared field once.
        assert "schema_cache=None blocks tool kinds:" in message
        assert "sql_executor=None blocks tool kinds:" in message
        for kind in ("table_search", "table_read", "table_neighbors"):
            assert kind in message
        assert "STORAGE_WAREHOUSE_ID" in message  # warehouse remediation hint

    def test_emits_each_remediation_hint_only_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``schema_cache`` and ``sql_executor`` both fix from STORAGE_WAREHOUSE_ID;
        the helper must dedupe the hint so the message stays scannable."""
        monkeypatch.delenv("STORAGE_WAREHOUSE_ID", raising=False)
        monkeypatch.delenv("TABLE_TOOLS_WAREHOUSE_ID", raising=False)
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(),
            workspace_client=None,
            user_token=None,
        )
        with pytest.raises(RuntimeError) as exc_info:
            assert_runtime_can_satisfy_workflows(
                runner.factory_context,
                declared_kinds=["table_search", "table_read"],
            )
        # Same hint string ("Set STORAGE_WAREHOUSE_ID ... preflight.resolve_warehouse_id_or_fail.")
        # must appear exactly once even though two fields would produce it.
        message = str(exc_info.value)
        assert message.count("preflight.resolve_warehouse_id_or_fail") == 1


class TestTextTableWiringIncompleteWarning:
    """When the workspace client is wired but the warehouse id is unset,
    ``build_app_workflow_runner`` must emit the
    ``TEXT_TABLE_WIRING_INCOMPLETE`` WARNING — promoted from the previous
    invisible INFO line on 2026-05-27 so log scanners catch the deploy
    misconfiguration that triggered the original incident."""

    def test_warning_fires_when_workspace_present_but_warehouse_missing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        monkeypatch.delenv("STORAGE_WAREHOUSE_ID", raising=False)
        monkeypatch.delenv("TABLE_TOOLS_WAREHOUSE_ID", raising=False)
        # Ensure settings.storage_warehouse_id is empty too — the resolver
        # falls back to settings before declaring None.
        from deep_research.core import config as _config_module

        monkeypatch.setattr(
            _config_module,
            "get_settings",
            lambda: SimpleNamespace(storage_warehouse_id=""),
        )
        statement_execution = MagicMock()
        workspace_client = SimpleNamespace(
            statement_execution=statement_execution
        )
        with caplog.at_level(
            logging.WARNING, logger="deep_research.agent.workflow_runner_factory"
        ):
            runner = build_app_workflow_runner(
                llm_client=_fake_llm(),
                workspace_client=workspace_client,  # type: ignore[arg-type]
                user_token=None,
            )
        assert any(
            "TEXT_TABLE_WIRING_INCOMPLETE" in rec.message
            and rec.levelno == logging.WARNING
            for rec in caplog.records
        ), (
            "Expected a TEXT_TABLE_WIRING_INCOMPLETE WARNING when "
            "workspace_client is present but warehouse_id is None"
        )
        ctx = runner.factory_context
        assert ctx.schema_cache is None
        assert ctx.sql_executor is None


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


class TestOBOResolution:
    """``build_app_workflow_runner`` must bake an OBO client when a user token
    is present so Databricks tools run AS THE USER (the 2026-05-30 fix for the
    main-UI failure where table reads ran as the app SP and were denied)."""

    def _sp(self, host: str = "https://wsp.example.databricks.com") -> MagicMock:
        sp = MagicMock(name="sp_client")
        sp.config.host = host
        return sp

    def test_obo_client_baked_into_table_executor_when_token_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from databricks_deep_research.core import databricks_auth

        monkeypatch.setenv("TABLE_TOOLS_WAREHOUSE_ID", "wh-test")
        sp = self._sp()
        obo = MagicMock(name="obo_client")
        monkeypatch.setattr(
            databricks_auth, "WorkspaceClient", MagicMock(return_value=obo)
        )
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(), workspace_client=sp, user_token="user-tok"
        )
        ctx = runner.factory_context
        assert ctx.workspace_client is obo
        assert ctx.user_token == "user-tok"
        # The text-table SQL executor — the tool that was denied in prod —
        # now runs statements through the OBO (user) client.
        assert ctx.sql_executor is not None
        assert ctx.sql_executor._workspace_client is obo

    def test_sp_client_used_when_no_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from databricks_deep_research.core import databricks_auth

        monkeypatch.setenv("TABLE_TOOLS_WAREHOUSE_ID", "wh-test")
        sp = self._sp()
        wc = MagicMock(name="WorkspaceClient")
        monkeypatch.setattr(databricks_auth, "WorkspaceClient", wc)
        runner = build_app_workflow_runner(
            llm_client=_fake_llm(), workspace_client=sp, user_token=None
        )
        wc.assert_not_called()
        ctx = runner.factory_context
        assert ctx.workspace_client is sp
        assert ctx.sql_executor is not None
        assert ctx.sql_executor._workspace_client is sp
