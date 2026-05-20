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

import logging
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from databricks_deep_research import WorkflowRunner
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.workflow.loader import load_workflow_from_dict
from tests.complex._scaffold_run_capture import (
    append_jsonl,
    emit_console_report,
    make_run_dir,
    summarize_events,
    write_json,
    write_summary_md,
    write_text,
    write_yaml,
)
from tests.shared import requires_brave, requires_databricks

from deep_research.agent.adapters.llm_adapter import create_framework_llm_client
from deep_research.agent_designer.discovery import DesignerDiscoveryAdapter
from deep_research.agent_designer.llm_adapter import AppLLMAdapter
from deep_research.agent_designer.orchestrator import (
    DesignerChatOrchestrator,
    MutationProposedEvent,
    _quality_advice,
    _validate_ast,
)
from deep_research.services.discovery_service import DiscoveryService
from deep_research.services.llm.client import LLMClient

logger = logging.getLogger(__name__)

_FIXTURE = Path(__file__).parent / "fixtures" / "scaffold_and_run_cases.yaml"
_RUNS_ROOT = Path(__file__).parent.parent / "_runs"  # databricks-deep-research-app/tests/_runs


def _load_cases() -> list[dict[str, Any]]:
    return yaml.safe_load(_FIXTURE.read_text())["cases"]


@requires_databricks
@requires_brave
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["id"])
async def test_scaffold_and_run(case: dict[str, Any]) -> None:
    run_dir = make_run_dir(_RUNS_ROOT, case["id"])
    logger.warning("SCAFFOLD_AND_RUN start case=%s artifact_dir=%s", case["id"], run_dir)

    # Echo case spec ----------------------------------------------------------
    write_text(run_dir / "intent.txt", case["intent"])
    write_text(run_dir / "query.txt", case["query"])
    write_json(run_dir / "case.json", case)

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
        async for ev in orchestrator.run_turn(
            messages=messages,
            current_ast=None,
            session_id=f"scaffold-and-run-{case['id']}",
            user_token="local-test-token",
            current_user_id="local-test-user",
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

        # Persist the AST in two forms
        write_json(run_dir / "designer" / "workflow.json", ast)
        write_yaml(run_dir / "designer" / "workflow.yaml", ast)

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
        assert not advice, (
            f"AST failed designer quality gate: {advice}. See {run_dir}/designer/."
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

        # IMPORTANT: use from_defaults(), not the bare constructor — the bare
        # constructor leaves search_client=None which causes the BuiltinToolFactory
        # to fail every web_search resolution silently. The previous run produced
        # source_count=0 ("Potemkin research") because researchers had no Brave
        # search client to retrieve URLs. from_defaults wires up search_client
        # from BRAVE_API_KEY (env), pre-builds the BraveSearchAdapter, and also
        # populates api_keys for other factory paths.
        import os as _os

        runner = WorkflowRunner(
            llm_client=framework_llm,
            factory_context=ToolFactoryContext.from_defaults(
                workspace_client=ws_client,
                user_token=None,
                brave_api_key=_os.environ.get("BRAVE_API_KEY"),
            ),
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
    write_json(
        run_dir / "runner" / "result.json",
        {
            "output_chars": len(result.output or ""),
            "source_count": len(result.sources or []),
            "event_count": len(runner_events),
            "event_histogram": summarize_events(runner_events),
            "runtime_state": runtime_state.model_dump(mode="json") if runtime_state else None,
        },
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
