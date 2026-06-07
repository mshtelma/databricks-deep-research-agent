"""Scaffold-and-Run live integration test.

For each case (see ``fixtures/scaffold_and_run_cases.yaml``):
  1. Drives the agent designer with a real LLM, captures the proposed AST.
  2. Validates the AST passes ``load_workflow_from_dict``.
  3. Runs the workflow via ``WorkflowRunner`` with a real query.
  4. Dumps every designer SSE event, every framework StreamEvent, the AST
     (JSON + YAML), validation results, and the final output to
     ``tests/_runs/<ISO-ts>-<case_id>/``.

Skipped unless ``DATABRICKS_TOKEN``/``DATABRICKS_CONFIG_PROFILE`` AND
``BRAVE_API_KEY`` are set. Slow: ~3-10 min per case.

Run with:
    make test-scaffold-and-run                         # both cases
    make test-scaffold-and-run CASE=investment_research # one case
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from databricks_deep_research.workflow.loader import load_workflow_from_dict
from tests.complex._scaffold_run_capture import (
    _bound_tool_names,
    append_jsonl,
    emit_console_report,
    make_run_dir,
    summarize_events,
    write_json,
    write_summary_md,
    write_text,
    write_yaml,
)
from tests.shared import requires_databricks

from deep_research.agent.adapters.llm_adapter import create_framework_llm_client
from deep_research.agent.workflow_runner_factory import build_app_workflow_runner
from deep_research.agent_designer.assets import resolve_default_table_warehouse_id
from deep_research.agent_designer.discovery import DesignerDiscoveryAdapter
from deep_research.agent_designer.llm_adapter import AppLLMAdapter
from deep_research.agent_designer.orchestrator import (
    DesignerChatOrchestrator,
    MutationProposedEvent,
    _quality_advice,
    _validate_ast,
)
from deep_research.agent_designer.prompt_grounding import (
    ground_prompt,
    prompt_grounding_sse_result,
)
from deep_research.agent_designer.tool_contract import (
    project_resolved_tool_contract,
    sanitized_resolved_tool_contract_summary,
)
from deep_research.core.app_config import DEFAULT_CONFIG_PATH, clear_config_cache
from deep_research.services.discovery_service import DiscoveryService
from deep_research.services.llm.client import LLMClient

logger = logging.getLogger(__name__)

_FIXTURE = Path(__file__).parent / "fixtures" / "scaffold_and_run_cases.yaml"
_RUNS_ROOT = Path(__file__).parent.parent / "_runs"  # databricks-deep-research-app/tests/_runs
_ENV_REF_RE = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


def _load_cases() -> list[dict[str, Any]]:
    return yaml.safe_load(_FIXTURE.read_text())["cases"]


@contextlib.contextmanager
def _maybe_override_app_search(case: dict[str, Any]) -> Iterator[None]:
    """Point app config at a copy of app.yaml with the case's web-search
    provider override, so the designer stamps it onto the web tools it generates
    (``ast_normalizer._normalize_web_search_provider``). No-op unless the case
    sets ``app_search_provider``. Restores the prior env + config cache on exit.
    """
    provider = case.get("app_search_provider")
    if not provider:
        yield
        return

    base = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
    search = base.setdefault("search", {})
    search["provider"] = provider
    if provider == "databricks":
        db = search.setdefault("databricks", {})
        if case.get("app_search_endpoint"):
            db["endpoint"] = case["app_search_endpoint"]
        if case.get("app_search_timeout") is not None:
            db["timeout_seconds"] = case["app_search_timeout"]

    fd, tmp_path = tempfile.mkstemp(suffix=".app.yaml")
    os.close(fd)
    Path(tmp_path).write_text(yaml.safe_dump(base))
    prev = os.environ.get("APP_CONFIG_PATH")
    os.environ["APP_CONFIG_PATH"] = tmp_path
    clear_config_cache()
    logger.warning(
        "SCAFFOLD_APP_SEARCH_OVERRIDE provider=%s endpoint=%s timeout=%s",
        provider,
        search.get("databricks", {}).get("endpoint"),
        search.get("databricks", {}).get("timeout_seconds"),
    )
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("APP_CONFIG_PATH", None)
        else:
            os.environ["APP_CONFIG_PATH"] = prev
        clear_config_cache()
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def _assert_web_tools_use_provider(ast: dict[str, Any], provider: str) -> None:
    """Every declared web_search/web_research tool must carry the expected
    provider — proof the designer stamped the configured backend."""
    web = [
        t
        for t in ast.get("tools", [])
        if isinstance(t, dict) and t.get("kind") in {"web_search", "web_research"}
    ]
    assert web, "designer declared no web_search/web_research tool to stamp"
    for tool in web:
        cfg = tool.get("config") or {}
        assert cfg.get("provider") == provider, (
            f"web tool {tool.get('name')!r} provider={cfg.get('provider')!r} "
            f"!= expected {provider!r} (designer stamp did not fire)"
        )


@pytest.mark.asyncio
async def test_officeqa_scaffold_uses_prompt_only_resource_grounding() -> None:
    cases = {case["id"]: case for case in _load_cases()}
    case = cases["officeqa_treasury_army_1945"]

    assert not case.get("assets"), (
        "OfficeQA scaffold must name resources in the prompt instead of "
        "supplying exact structured asset config"
    )
    assert not any(
        key in case
        for key in ("config", "tool_config", "designer_assets", "grounded_assets")
    )
    for resource_name in case["expected_resource_names"]:
        assert resource_name in case["intent"]
    assert case["requires_table_tools_warehouse"] is True
    assert resolve_default_table_warehouse_id() == "d837825f69a03500"
    assert {
        "vector_search",
        "table_search",
        "table_read",
        "table_load",
        "compute",
    }.issubset(set(case["expected_tool_kinds"]))

    grounding = await ground_prompt(
        intent=case["intent"],
        existing_assets=[],
        discovery=None,
        default_warehouse_id="warehouse-from-trusted-test-default",
    )
    _assert_prompt_grounding_payload(prompt_grounding_sse_result(grounding), case)

    assets_by_name = {
        str(asset.full_name): asset for asset in grounding.resolved_assets
    }
    assert set(case["expected_resource_names"]).issubset(assets_by_name)
    assert (
        assets_by_name["main.officeqa_benchmark.treasury_chunks_vs_index"].kind
        == "vector_index"
    )
    assert assets_by_name["main.officeqa_benchmark.treasury_chunks"].kind == "delta_table"
    assert assets_by_name["main.officeqa_benchmark.treasury_tables"].kind == "delta_table"
    assert all(
        assets_by_name[name].usage == "required"
        for name in case["expected_resource_names"]
    )
    assert all(
        asset.metadata.get("warehouse_id") == "warehouse-from-trusted-test-default"
        for asset in assets_by_name.values()
        if asset.kind == "delta_table"
    )

    contract = project_resolved_tool_contract(grounding, intent=case["intent"])
    assert contract is not None
    summary = sanitized_resolved_tool_contract_summary(contract)
    _assert_resolved_tool_contract_payload(summary, case)


def _env_refs(value: Any) -> set[str]:
    if isinstance(value, str):
        match = _ENV_REF_RE.match(value)
        return {match.group(1)} if match else set()
    if isinstance(value, list):
        refs: set[str] = set()
        for item in value:
            refs.update(_env_refs(item))
        return refs
    if isinstance(value, dict):
        refs: set[str] = set()
        for item in value.values():
            refs.update(_env_refs(item))
        return refs
    return set()


def _expand_env_refs(value: Any) -> Any:
    if isinstance(value, str):
        match = _ENV_REF_RE.match(value)
        if match:
            return os.environ[match.group(1)]
        return value
    if isinstance(value, list):
        return [_expand_env_refs(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_refs(item) for key, item in value.items()}
    return value


def _declared_tools(ast: dict[str, Any]) -> list[dict[str, Any]]:
    return [tool for tool in ast.get("tools", []) if isinstance(tool, dict)]


def _walk_nodes(node: Any) -> list[dict[str, Any]]:
    if not isinstance(node, dict):
        return []
    nodes = [node]
    config = node.get("config") if isinstance(node.get("config"), dict) else {}
    body = config.get("body")
    if isinstance(body, dict):
        nodes.extend(_walk_nodes(body))
    for child in node.get("children") or []:
        nodes.extend(_walk_nodes(child))
    return nodes


def _config_string_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        values: set[str] = set()
        for item in value:
            values.update(_config_string_values(item))
        return values
    if isinstance(value, dict):
        values: set[str] = set()
        for item in value.values():
            values.update(_config_string_values(item))
        return values
    return set()


def _assert_structural_tool_contract(ast: dict[str, Any], case: dict[str, Any]) -> None:
    tools = _declared_tools(ast)
    tool_kinds_by_name = {
        str(tool.get("name")): str(tool.get("kind"))
        for tool in tools
        if isinstance(tool.get("name"), str)
    }
    declared_kinds = set(tool_kinds_by_name.values())
    bound_names = _bound_tool_names(ast)

    expected_kinds = set(case.get("expected_tool_kinds", []))
    missing_kinds = expected_kinds - declared_kinds
    assert not missing_kinds, f"missing expected tool kinds: {sorted(missing_kinds)}"

    unbound_expected = [
        name
        for name, kind in tool_kinds_by_name.items()
        if kind in expected_kinds and name not in bound_names
    ]
    assert not unbound_expected, (
        f"expected tools are declared but not node-bound: {sorted(unbound_expected)}"
    )

    forbidden_kinds = set(case.get("forbidden_tool_kinds", []))
    present_forbidden = forbidden_kinds & declared_kinds
    assert not present_forbidden, f"forbidden tool kinds declared: {sorted(present_forbidden)}"

    required_assets = [
        asset
        for asset in case.get("assets", [])
        if asset.get("usage") == "required" and asset.get("full_name")
    ]
    for asset in required_assets:
        full_name = str(asset["full_name"])
        if asset.get("kind") == "vector_index":
            assert any(
                (tool.get("config") or {}).get("index_name") == full_name for tool in tools
            ), f"required vector index not referenced by any tool: {full_name}"
        if asset.get("kind") == "delta_table":
            assert any(
                (tool.get("config") or {}).get("table_name") == full_name for tool in tools
            ), f"required Delta table not referenced by any tool: {full_name}"

    expected_resource_names = {str(name) for name in case.get("expected_resource_names", [])}
    if expected_resource_names:
        config_values: set[str] = set()
        for tool in tools:
            config_values.update(_config_string_values(tool.get("config") or {}))
        missing_resources = expected_resource_names - config_values
        assert not missing_resources, (
            "expected prompt-named resources not referenced by tool config: "
            f"{sorted(missing_resources)}"
        )


def _assert_prompt_grounding_events(
    designer_events: list[Any],
    case: dict[str, Any],
) -> None:
    expected_resource_names = {str(name) for name in case.get("expected_resource_names", [])}
    if not expected_resource_names:
        return
    assert not case.get("assets"), (
        "prompt-grounding scaffold cases must not supply structured assets; "
        "resource names should come from the initial prompt"
    )
    grounding_events = [
        event
        for event in designer_events
        if getattr(event, "type", None) == "tool_result"
        and getattr(event, "tool_name", None) == "prompt_grounding"
    ]
    assert grounding_events, "designer did not emit prompt_grounding diagnostics"
    _assert_prompt_grounding_payload(grounding_events[-1].result, case)


def _assert_runtime_forbidden_policy_is_prompted(case: dict[str, Any]) -> None:
    if not case.get("forbidden_runtime_tool_kinds"):
        return
    intent = str(case.get("intent", "")).lower()
    assert (
        "do not use public web tools" in intent
        or "do not use web tools" in intent
        or "no public web" in intent
    ), (
        "forbidden_runtime_tool_kinds must be grounded in the initial "
        "user prompt, not assumed by the scaffold test"
    )


def _assert_resolved_tool_contract_events(
    designer_events: list[Any],
    case: dict[str, Any],
) -> None:
    expected_resource_names = {str(name) for name in case.get("expected_resource_names", [])}
    if not expected_resource_names:
        return
    contract_events = [
        event
        for event in designer_events
        if getattr(event, "type", None) == "tool_result"
        and getattr(event, "tool_name", None) == "resolved_tool_contract"
    ]
    assert contract_events, "designer did not emit resolved_tool_contract diagnostics"
    _assert_resolved_tool_contract_payload(contract_events[-1].result, case)


def _assert_prompt_grounding_payload(
    result: dict[str, Any],
    case: dict[str, Any],
) -> None:
    expected_resource_names = {str(name) for name in case.get("expected_resource_names", [])}
    if not expected_resource_names:
        return
    assert result["schema"] == "prompt_grounding.v1"
    assert result["safe_to_build_blueprint"] is True
    assert result["resolved_assets_count"] >= len(expected_resource_names)
    assert result["resource_kinds"].get("vector_index", 0) >= 1
    assert result["resource_kinds"].get("delta_table", 0) >= 2
    resolved_resources = {
        str(resource.get("identity")): resource
        for resource in result.get("resolved_resources", [])
        if isinstance(resource, dict)
    }
    missing_resources = expected_resource_names - set(resolved_resources)
    assert not missing_resources, (
        "prompt grounding did not resolve expected prompt-named resources: "
        f"{sorted(missing_resources)}"
    )
    assert resolved_resources[
        "main.officeqa_benchmark.treasury_chunks_vs_index"
    ].get("kind") == "vector_index"
    assert resolved_resources[
        "main.officeqa_benchmark.treasury_chunks"
    ].get("kind") == "delta_table"
    assert resolved_resources[
        "main.officeqa_benchmark.treasury_tables"
    ].get("kind") == "delta_table"
    assert all(
        resolved_resources[name].get("usage") == "required"
        for name in expected_resource_names
    )
    assert all(
        resolved_resources[name].get("access_status") != "inaccessible"
        for name in expected_resource_names
    )
    assert set(case.get("expected_tool_kinds", [])).issubset(
        set(result.get("ready_tool_kinds", []))
    )
    blocking = [
        diagnostic
        for diagnostic in result.get("diagnostics", [])
        if diagnostic.get("blocking")
    ]
    assert blocking == []


def _assert_resolved_tool_contract_payload(
    result: dict[str, Any],
    case: dict[str, Any],
) -> None:
    expected_resource_names = {str(name) for name in case.get("expected_resource_names", [])}
    if not expected_resource_names:
        return
    assert result["schema"] == "resolved_tool_contract.v1"
    assert result["available"] is True
    assert result["evidence_policy"] == "corpus_only"
    assert set(case.get("expected_tool_kinds", [])).issubset(
        set(result.get("ready_tool_kinds", []))
    )
    assert set(case.get("forbidden_tool_kinds", [])).issubset(
        set(result.get("forbidden_tool_kinds", []))
    )
    resources = {
        str(resource.get("identity")): resource
        for resource in result.get("resources", [])
        if isinstance(resource, dict)
    }
    missing = expected_resource_names - set(resources)
    assert not missing, (
        "resolved tool contract did not include expected prompt-named "
        f"resources: {sorted(missing)}"
    )
    required_terms = set(result.get("required_terms", []))
    assert {"officeqa", "treasury", "chunks", "vector", "compute"} & required_terms
    assert len(required_terms) >= 2


def _assert_ast_resolved_contract_metadata(
    ast: dict[str, Any],
    case: dict[str, Any],
) -> None:
    if not case.get("expected_resource_names"):
        return
    summary = ast.get("resolved_tool_contract_summary")
    assert isinstance(summary, dict), "AST missing resolved_tool_contract_summary"
    _assert_resolved_tool_contract_payload(summary, case)
    assert ast.get("evidence_policy") == "corpus_only"
    assert len(ast.get("required_prompt_terms") or []) >= 2
    assert not ast.get("placeholder_pending_nodes"), (
        "contract-specialized blueprint should not carry stale "
        "placeholder_pending_nodes"
    )
    plan_configs = [
        node.get("config") or {}
        for node in _walk_nodes(ast.get("root"))
        if node.get("type") == "plan_and_execute"
    ]
    required_groups = [
        group
        for config in plan_configs
        for group in config.get("required_tool_kind_groups", [])
        if isinstance(group, list)
    ]
    assert ["vector_search"] in required_groups
    assert ["compute"] in required_groups
    assert any(
        {"table_search", "table_read", "table_load"} & set(group)
        for group in required_groups
    ), "plan_and_execute must gate completion on a table tool family"


def _runtime_tool_kinds(
    ast: dict[str, Any],
    runner_events: list[Any],
) -> list[str]:
    tool_kinds_by_name = {
        str(tool.get("name")): str(tool.get("kind"))
        for tool in _declared_tools(ast)
        if isinstance(tool.get("name"), str)
    }
    kinds: list[str] = []
    for event in runner_events:
        if getattr(event, "event_type", None) != "tool_call":
            continue
        tool_name = str(getattr(event, "tool_name", "") or "")
        kind = tool_kinds_by_name.get(tool_name)
        if kind:
            kinds.append(kind)
    return kinds


@requires_databricks
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["id"])
async def test_scaffold_and_run(case: dict[str, Any]) -> None:
    if case.get("requires_brave", True) and not os.environ.get("BRAVE_API_KEY"):
        pytest.skip("BRAVE_API_KEY not set")
    if (
        case.get("requires_table_tools_warehouse")
        and not resolve_default_table_warehouse_id()
    ):
        pytest.skip("TABLE_TOOLS_WAREHOUSE_ID or STORAGE_WAREHOUSE_ID not set")
    missing_env = sorted(
        ref for ref in _env_refs(case.get("assets", [])) if not os.environ.get(ref)
    )
    if missing_env:
        pytest.skip(f"missing required env vars for case assets: {missing_env}")

    assets = _expand_env_refs(case.get("assets", []))
    case = {**case, "assets": assets}

    run_dir = make_run_dir(_RUNS_ROOT, case["id"])
    logger.warning("SCAFFOLD_AND_RUN start case=%s artifact_dir=%s", case["id"], run_dir)

    # Echo case spec ----------------------------------------------------------
    write_text(run_dir / "intent.txt", case["intent"])
    write_text(run_dir / "query.txt", case["query"])
    write_json(run_dir / "case.json", case)
    write_json(run_dir / "assets.json", assets)
    _assert_runtime_forbidden_policy_is_prompted(case)

    # ── Designer phase ───────────────────────────────────────────────────────
    designer_events_path = run_dir / "designer" / "events.jsonl"
    messages = [{"role": "user", "content": case["intent"]}]
    write_json(run_dir / "designer" / "messages.json", messages)

    # Share a single app LLMClient across designer + runner phases — this is
    # exactly what prod does: the designer uses AppLLMAdapter(app_llm), and the
    # workflow runner uses create_framework_llm_client(app_llm) so both
    # resolve through the SAME app.yaml tier mapping (analytical → sonnet, etc.)
    # and reuse the same auth / OpenAI client.
    app_llm = LLMClient()
    try:
        orchestrator = DesignerChatOrchestrator(
            AppLLMAdapter(app_llm),
            DesignerDiscoveryAdapter(DiscoveryService()),
        )

        ast: dict[str, Any] | None = None
        designer_events: list[Any] = []
        t0 = time.monotonic()
        # Apply the per-case web-search provider override for the designer turn so
        # the normalizer stamps the chosen backend onto the generated web tools.
        with _maybe_override_app_search(case):
            async for ev in orchestrator.run_turn(
                messages=messages,
                current_ast=None,
                session_id=f"scaffold-and-run-{case['id']}",
                user_token="local-test-token",
                current_user_id="local-test-user",
                assets=assets,
            ):
                designer_events.append(ev)
                append_jsonl(designer_events_path, ev)
                if isinstance(ev, MutationProposedEvent):
                    ast = ev.new_ast
        designer_wall = time.monotonic() - t0

        assert ast is not None, (
            f"Designer never emitted a MutationProposedEvent. "
            f"See {designer_events_path}. Event types seen: "
            f"{summarize_events(designer_events)}"
        )
        _assert_prompt_grounding_events(designer_events, case)
        _assert_resolved_tool_contract_events(designer_events, case)

        # Persist the AST in two forms
        write_json(run_dir / "designer" / "workflow.json", ast)
        write_yaml(run_dir / "designer" / "workflow.yaml", ast)

        # When the case forces a web-search provider, confirm the designer
        # stamped it onto the generated web tools (the runner then uses it).
        if case.get("app_search_provider"):
            _assert_web_tools_use_provider(ast, case["app_search_provider"])

        # Validation + advice (private helpers from orchestrator)
        validation_errors, workflow_summary = _validate_ast(ast)
        advice = _quality_advice(ast)
        write_json(
            run_dir / "designer" / "validation.json",
            {"errors": validation_errors, "summary": workflow_summary},
        )
        write_json(run_dir / "designer" / "advice.json", advice)

        assert not validation_errors, (
            f"AST failed validation: {validation_errors}. See {run_dir}/designer/."
        )
        # Plan v2.1 M10 — severity-graded advice contract. Each advice
        # record carries an explicit ``severity`` field
        # (``blocking|warning|info``); only blocking entries fail the
        # scaffold gate. Existing detectors default to ``blocking`` for
        # back-compat (US-01 landed the severity field with that
        # default), so the strict ``assert not advice`` semantics are
        # preserved for any detector that hasn't been re-tagged. Detectors
        # that explicitly downgrade their advice to ``warning`` / ``info``
        # (e.g., heuristic post-processors kept as telemetry post-PR-3)
        # surface in logs but no longer fail the run.
        blocking_advice = [
            a for a in (advice or []) if a.get("severity", "blocking") == "blocking"
        ]
        non_blocking_advice = [
            a for a in (advice or []) if a.get("severity", "blocking") != "blocking"
        ]
        if non_blocking_advice:
            # Log without failing — surface for observability.
            print(
                f"[scaffold-and-run] non-blocking advice "
                f"({len(non_blocking_advice)}): {non_blocking_advice}"
            )
        assert not blocking_advice, (
            f"AST failed designer quality gate (blocking advice): "
            f"{blocking_advice}. See {run_dir}/designer/."
        )
        assert workflow_summary is not None  # tautological with len(errors)==0

        # Sanity floors so we catch a "trivially empty" designer regression early
        assert workflow_summary["node_count"] >= case["expected_min_node_count"], (
            f"node_count={workflow_summary['node_count']} below "
            f"expected_min_node_count={case['expected_min_node_count']}"
        )
        assert workflow_summary["tool_count"] >= case["expected_min_tool_count"], (
            f"tool_count={workflow_summary['tool_count']} below "
            f"expected_min_tool_count={case['expected_min_tool_count']}"
        )
        _assert_structural_tool_contract(ast, case)
        _assert_ast_resolved_contract_metadata(ast, case)

        # Round-trip — the runner accepts dicts, but confirm the framework-level
        # loader is happy with this AST before we burn LLM tokens.
        definition = load_workflow_from_dict(ast)

        # ── Runner phase ────────────────────────────────────────────────────
        runner_events_path = run_dir / "runner" / "events.jsonl"

        # Build framework LLM the prod way (same path as
        # `agent/framework_orchestrator.py`): take the app's LLMClient and
        # extract its tier mapping so the workflow runs with the SAME models
        # the deployed app uses. The workflow's own `models:` section (if the
        # designer included one) layers on top via `_resolve_client.derive()`.
        framework_llm = create_framework_llm_client(app_llm)
        try:
            from deep_research.core.auth import get_workspace_client

            ws_client = get_workspace_client()
        except Exception as exc:  # noqa: BLE001
            logger.warning("WORKSPACE_CLIENT_UNAVAILABLE: %s", exc)
            ws_client = None

        # Build through the app factory so web and table-tool dependencies are
        # wired the same way production requests are.
        runner = build_app_workflow_runner(
            llm_client=framework_llm,
            workspace_client=ws_client,
            user_token=None,
        )

        runner_events: list[Any] = []
        t0 = time.monotonic()
        async for ev in runner.stream(workflow=definition, query=case["query"]):
            runner_events.append(ev)
            append_jsonl(runner_events_path, ev)
        runner_wall = time.monotonic() - t0
    finally:
        await app_llm.close()

    result = runner.last_result
    assert result is not None, "runner.last_result is None after stream completion"

    # Dump runner result
    write_text(run_dir / "runner" / "output.md", result.output or "")
    write_json(run_dir / "runner" / "sources.json", result.sources)

    # Surface report + sources + reflector verdict to stdout for interactive
    # runs (Makefile invokes pytest with ``-s`` so this lands in the terminal).
    # Failure is non-fatal — the dump is best-effort observability only.
    try:
        emit_console_report(
            case_id=case["id"],
            report=result.output,
            sources=result.sources,
            runtime_state=result.runtime_state,
        )
    except Exception:  # noqa: BLE001 — dump is purely informational
        logger.exception("emit_console_report failed (test continues)")

    runtime_state = result.runtime_state
    runtime_tool_kinds = _runtime_tool_kinds(ast, runner_events)
    write_json(
        run_dir / "runner" / "result.json",
        {
            "output_chars": len(result.output or ""),
            "source_count": len(result.sources or []),
            "event_count": len(runner_events),
            "event_histogram": summarize_events(runner_events),
            "runtime_tool_kinds": runtime_tool_kinds,
            "runtime_state": runtime_state.model_dump(mode="json") if runtime_state else None,
        },
    )

    runtime_kinds = set(runtime_tool_kinds)
    expected_runtime_kinds = set(case.get("expected_runtime_tool_kinds", []))
    missing_runtime_kinds = expected_runtime_kinds - runtime_kinds
    assert not missing_runtime_kinds, (
        f"expected runtime tool kinds were not called: {sorted(missing_runtime_kinds)}"
    )
    expected_any_runtime_kinds = set(case.get("expected_runtime_any_tool_kinds", []))
    if expected_any_runtime_kinds:
        assert expected_any_runtime_kinds & runtime_kinds, (
            "none of the expected runtime tool kind family was called: "
            f"{sorted(expected_any_runtime_kinds)}"
        )

    forbidden_runtime_kinds = set(case.get("forbidden_runtime_tool_kinds", []))
    runtime_forbidden = forbidden_runtime_kinds & runtime_kinds
    assert not runtime_forbidden, (
        f"forbidden runtime tool kinds were called: {sorted(runtime_forbidden)}"
    )

    output = result.output or ""
    for term in case.get("expected_answer_terms", []):
        assert term in output, f"expected answer term {term!r} not found in output"
    # Each group is satisfied if ANY of its terms appears; ALL groups must be
    # satisfied. Robust to phrasing variance (mirrors expected_runtime_any_*).
    for group in case.get("expected_answer_term_groups", []):
        assert any(term in output for term in group), (
            f"none of answer-term group {group!r} found in output"
        )

    # Phrasing-robust proof the search backend returned cited sources (the core
    # signal for web-search provider cases — avoids brittle answer-term gates).
    min_sources = case.get("expected_min_source_count")
    if min_sources is not None:
        assert len(result.sources or []) >= min_sources, (
            f"expected >= {min_sources} cited sources, got {len(result.sources or [])}"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    write_summary_md(
        run_dir,
        case=case,
        designer_wall_s=designer_wall,
        designer_event_counts=summarize_events(designer_events),
        workflow_summary=workflow_summary,
        validation_errors=validation_errors,
        advice=advice,
        runner_wall_s=runner_wall,
        runner_event_counts=summarize_events(runner_events),
        output_chars=len(result.output or ""),
        source_count=len(result.sources or []),
    )

    logger.warning(
        "SCAFFOLD_AND_RUN done case=%s artifact_dir=%s "
        "designer_wall=%.1fs runner_wall=%.1fs nodes=%d output_chars=%d sources=%d",
        case["id"], run_dir, designer_wall, runner_wall,
        workflow_summary["node_count"], len(result.output or ""), len(result.sources or []),
    )
