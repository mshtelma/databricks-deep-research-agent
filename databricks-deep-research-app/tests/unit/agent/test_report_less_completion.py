"""Report-less (pure tool-node / no-LLM) workflow completion.

Covers the fix that lets a SUCCESSFUL workflow with no synthesized report still
record a terminal COMPLETED transition + assistant message — while NEVER marking
a timed-out or failed run COMPLETED (the regression the review flagged).
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.workflow.state import WorkflowState

from deep_research.agent.framework_orchestrator import (
    _render_output_value,
    _synthesize_terminal_report,
    stream_research_via_framework,
)
from deep_research.schemas.streaming import StreamErrorEvent

# ---------------------------------------------------------------------------
# _synthesize_terminal_report — pure
# ---------------------------------------------------------------------------


class TestSynthesizeTerminalReport:
    def test_joins_declared_output_keys(self) -> None:
        state = WorkflowState()
        state.append("t1", "sma_summary", "SMA(20) = 42.0")
        wf_def = SimpleNamespace(output_keys=["sma_summary"])
        report = _synthesize_terminal_report(state, wf_def)  # type: ignore[arg-type]
        assert "sma_summary" in report
        assert "SMA(20) = 42.0" in report

    def test_renders_dict_result_field(self) -> None:
        state = WorkflowState()
        state.append("t1", "out", {"result": "40.0", "rows": [{"result": "40.0"}]})
        wf_def = SimpleNamespace(output_keys=["out"])
        report = _synthesize_terminal_report(state, wf_def)  # type: ignore[arg-type]
        assert "40.0" in report

    def test_empty_state_returns_nonempty_floor(self) -> None:
        state = WorkflowState()
        wf_def = SimpleNamespace(output_keys=["missing"])
        report = _synthesize_terminal_report(state, wf_def)  # type: ignore[arg-type]
        assert report == "Workflow completed."
        assert report  # never empty -> never a NULL assistant message

    def test_render_value_variants(self) -> None:
        assert _render_output_value("  hi  ") == "hi"
        assert _render_output_value({"result": 42}) == "42"
        assert _render_output_value(3.14) == "3.14"


# ---------------------------------------------------------------------------
# Full-path gating via the streaming orchestrator (fake runner)
# ---------------------------------------------------------------------------


def _mock_config(**overrides: Any) -> MagicMock:
    defaults: dict[str, Any] = {
        "query_mode": "deep_research",
        "research_depth": "auto",
        "system_instructions": None,
        "message_id": uuid4(),
        "research_session_id": uuid4(),
        "is_draft": False,
        "session_pre_created": False,
        "verify_sources": True,
        "output_format": "markdown",
        "output_schema": None,
        "synthesis_mode": "simple",
        "enable_post_verification": False,
        "file_ids": None,
        "source_scope": None,
        "enabled_sources": None,
        "disabled_sources": None,
        "user_token": None,
        "model_overrides": None,
        "domain_filter": None,
        "agent_id": None,
        "workflow_ref": None,
        "research_timeout_seconds": 1800,
    }
    defaults.update(overrides)
    config = MagicMock()
    for key, value in defaults.items():
        setattr(config, key, value)
    return config


def _mock_tracker() -> MagicMock:
    tracker = MagicMock()
    tracker.process_event.return_value = []
    tracker.should_persist.return_value = False
    delta = MagicMock()
    delta.final_report = ""
    delta.verification_summary = {}
    tracker.get_persistence_delta.return_value = delta
    return tracker


def _enter_common_patches(
    stack: ExitStack, mock_runner: MagicMock, mock_tracker: MagicMock
) -> AsyncMock:
    """Apply the shared orchestrator patch stack; return the _persist_completion mock."""
    fo = "deep_research.agent.framework_orchestrator"
    stack.enter_context(
        patch(f"{fo}.build_app_workflow_runner", return_value=mock_runner)
    )
    stack.enter_context(
        patch(f"{fo}.create_framework_llm_client", return_value=MagicMock())
    )
    stack.enter_context(
        patch(f"{fo}.create_framework_tools", new_callable=AsyncMock, return_value=[])
    )
    stack.enter_context(
        patch(f"{fo}._load_file_search_tool", new_callable=AsyncMock, return_value=None)
    )
    stack.enter_context(
        patch(f"{fo}._load_enterprise_tools", new_callable=AsyncMock, return_value=[])
    )
    stack.enter_context(
        patch(f"{fo}._load_existing_sources", new_callable=AsyncMock, return_value=[])
    )
    stack.enter_context(patch(f"{fo}.translate", return_value=MagicMock()))
    stack.enter_context(patch(f"{fo}.ExecutionContext", return_value=MagicMock()))
    stack.enter_context(patch(f"{fo}.DomainContextTracker", return_value=mock_tracker))
    stack.enter_context(patch(f"{fo}.safe_mlflow_run"))
    stack.enter_context(patch(f"{fo}.safe_tool_span"))
    stack.enter_context(patch(f"{fo}.safe_update_trace"))
    stack.enter_context(patch(f"{fo}._persist_simple_response", new_callable=AsyncMock))
    stack.enter_context(patch(f"{fo}._get_pool_sources", return_value=[]))
    stack.enter_context(
        patch(
            f"{fo}._extract_verification_from_framework_state",
            return_value=([], None),
        )
    )
    stack.enter_context(patch(f"{fo}._buffer_event", new_callable=AsyncMock))
    stack.enter_context(patch(f"{fo}._flush_event_buffer", new_callable=AsyncMock))
    persist = stack.enter_context(
        patch(
            f"{fo}._persist_completion",
            new_callable=AsyncMock,
            return_value={"messages": 1},
        )
    )
    return persist


class TestReportLessCompletionGating:
    @pytest.mark.asyncio
    async def test_successful_report_less_run_persists_synthesized_message(
        self,
    ) -> None:
        """Success + empty report -> a non-empty synthesized report is persisted
        through the normal completion path (not a NULL-content status flip)."""
        from databricks_deep_research.events.types import (
            WorkflowCompletedEvent as FwkWorkflowCompletedEvent,
        )

        completed = FwkWorkflowCompletedEvent(
            node_id="main",
            timestamp="2025-01-01T00:00:01Z",
            workflow_id="pure_tool",
            duration_ms=10.0,
            total_tokens=0,
            final_report="",  # report-less: no synthesized prose
            total_sources=0,
        )

        async def _stream(*args: Any, **kwargs: Any) -> Any:
            yield completed

        mock_runner = MagicMock()
        mock_runner.stream = _stream
        mock_runner.factory_context = ToolFactoryContext()
        config = _mock_config()

        with ExitStack() as stack:
            persist = _enter_common_patches(stack, mock_runner, _mock_tracker())
            async for _evt in stream_research_via_framework(
                query="compute pct change",
                llm=MagicMock(),
                brave_client=MagicMock(),
                crawler=MagicMock(),
                config=config,
                chat_id=str(uuid4()),
                user_id="user-1",
            ):
                pass

        persist.assert_awaited_once()
        # final_report is the 5th positional arg — must be non-empty (synthesized).
        assert persist.await_args is not None
        final_report_arg = persist.await_args.args[4]
        assert isinstance(final_report_arg, str) and final_report_arg.strip()

    @pytest.mark.asyncio
    async def test_timeout_does_not_flip_to_completed(self) -> None:
        """A RESEARCH_TIMEOUT (no WorkflowCompletedEvent) must NOT be persisted as
        COMPLETED — the completion path is gated on positive completion."""

        async def _slow_stream(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0.5)
            yield None  # never reached before the 0.2s timeout fires

        mock_runner = MagicMock()
        mock_runner.stream = _slow_stream
        mock_runner.factory_context = ToolFactoryContext()
        config = _mock_config(research_timeout_seconds=0.2)

        events: list[Any] = []
        with ExitStack() as stack:
            persist = _enter_common_patches(stack, mock_runner, _mock_tracker())
            async for evt in stream_research_via_framework(
                query="slow research",
                llm=MagicMock(),
                brave_client=MagicMock(),
                crawler=MagicMock(),
                config=config,
                chat_id=str(uuid4()),
                user_id="user-1",
            ):
                events.append(evt)

        persist.assert_not_awaited()
        assert any(
            isinstance(e, StreamErrorEvent) and e.error_code == "RESEARCH_TIMEOUT"
            for e in events
        )
