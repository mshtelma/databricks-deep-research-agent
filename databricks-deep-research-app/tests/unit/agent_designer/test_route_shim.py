"""Tests for the W5c designer route shim.

The shim replaces ~600 LOC of hand-coded LLM-tool-call loop with a thin
translator that drives the framework workflow defined in
``designer_workflow.yaml`` and emits ``DesignerSSEEvent``s.

These tests assert the SHIM WIRES CORRECTLY — they do not exercise the
full LLM flow (no real Databricks credentials in unit tests). The shim
must always yield a ``DoneEvent`` at the end of the stream, even when
the underlying framework call fails — that's the contract the frontend
relies on for closing the SSE channel cleanly.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest
from databricks_deep_research.workflow.loader import load_workflow_from_dict

from deep_research.agent_designer.discovery import DesignerDiscoveryAdapter
from deep_research.agent_designer.orchestrator import (
    DesignerChatOrchestrator,
    DoneEvent,
    ErrorEvent,
    LLMClientProto,
    MessageEvent,
    MutationProposedEvent,
    _derive_workflow_llm_client,
    _mutation_event_for_ast_change,
)


class _FakeAppLLM:
    """Stub app LLMClient that records calls without performing any HTTP.

    The shim extracts the underlying app LLM via ``getattr(self._llm, "_llm")``
    and then asks ``create_framework_llm_client`` to wrap it. That call will
    fail at ``app_llm._ensure_fresh_client()`` because this fake exposes none
    of the real ``LLMClient`` internals — the SHIM CONTRACT is then to
    translate the failure into an ``ErrorEvent`` followed by a ``DoneEvent``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    async def close(self) -> None:
        return None


class _FakeAdapter:
    """LLMClientProto-shaped adapter that wraps the fake app LLM."""

    def __init__(self) -> None:
        self._llm = _FakeAppLLM()

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        # Not exercised — the shim never calls back into the legacy LLM
        # protocol. Provided so the adapter still satisfies the Protocol.
        if False:
            yield None


class _FakeDiscovery:
    async def list_for_user(
        self,
        *,
        user_token: str,
        kinds: list[str] | None = None,
        user_id: str = "",
    ) -> list[Any]:
        return []


@pytest.mark.asyncio
async def test_run_turn_wires_to_framework_and_emits_done() -> None:
    """End-to-end smoke: build the orchestrator, drive run_turn, assert a
    terminal ``DoneEvent`` arrives.

    With a fake app LLM the framework client build fails and the shim
    converts the exception into an ``ErrorEvent`` followed by ``DoneEvent``.
    Either path is a valid SHIM CORRECTNESS signal — what we're proving here
    is the imports resolve, the run_turn body executes, and the SSE event
    stream terminates correctly.
    """
    adapter = _FakeAdapter()
    orchestrator = DesignerChatOrchestrator(
        cast(LLMClientProto, adapter),
        cast(DesignerDiscoveryAdapter, _FakeDiscovery()),
    )

    events: list[Any] = []

    async def consume() -> None:
        async for ev in orchestrator.run_turn(
            messages=[
                {"role": "user", "content": "Build an investment research assistant"},
            ],
            current_ast=None,
            session_id="test",
            user_token="",
            current_user_id="test-user",
        ):
            events.append(ev)

    # 1-second timeout — the shim must NOT block waiting on a real LLM.
    # The fake app LLM forces an early ErrorEvent + DoneEvent path.
    try:
        await asyncio.wait_for(consume(), timeout=5.0)
    except TimeoutError:  # pragma: no cover - safety net
        pytest.fail("run_turn did not terminate within 5s with a fake LLM")

    # Shim contract: terminal event is always DoneEvent.
    assert events, "run_turn yielded zero events"
    assert isinstance(events[-1], DoneEvent), (
        f"Expected DoneEvent terminal, got {type(events[-1]).__name__}. "
        f"Full sequence: {[type(e).__name__ for e in events]}"
    )

    # The shim must surface SOME visible signal — either an ErrorEvent
    # (when the framework client fails to build, as with a fake LLM) or
    # a MessageEvent / MutationProposedEvent if the full path runs.
    assert any(
        isinstance(ev, ErrorEvent | MessageEvent | MutationProposedEvent)
        for ev in events
    ), (
        "Expected at least one of ErrorEvent / MessageEvent / "
        f"MutationProposedEvent. Got: {[type(e).__name__ for e in events]}"
    )


@pytest.mark.asyncio
async def test_run_turn_preserves_size_limit_enforcement() -> None:
    """Pre-flight size-limit checks must still raise BEFORE the workflow
    runs, so the route can return HTTP 413 instead of an SSE error frame.
    """
    from deep_research.agent_designer.orchestrator import (
        MAX_AST_BYTES,
        RequestTooLargeError,
    )

    adapter = _FakeAdapter()
    orchestrator = DesignerChatOrchestrator(
        cast(LLMClientProto, adapter),
        cast(DesignerDiscoveryAdapter, _FakeDiscovery()),
    )

    # Build an oversized AST to trip the limit.
    oversized = {"root": {"id": "x", "type": "agent", "label": "x", "config": {
        "padding": "A" * (MAX_AST_BYTES + 1024)
    }}}

    with pytest.raises(RequestTooLargeError):
        # run_turn enforces the limit synchronously before iteration starts;
        # the first __anext__ call surfaces the exception.
        gen = orchestrator.run_turn(
            messages=[{"role": "user", "content": "hi"}],
            current_ast=oversized,
            session_id="test",
            user_token="",
            current_user_id="u",
        )
        await gen.__anext__()


def test_mutating_tool_result_can_surface_late_ast_cache_updates() -> None:
    """Late update_block calls must replace an earlier stale mutation snapshot."""
    stale_ast = {
        "root": {
            "id": "root",
            "type": "agent",
            "label": "Researcher",
            "config": {
                "subtype": "researcher",
                "user_prompt_template": "Execute the following research step.",
            },
        }
    }
    latest_ast = {
        "root": {
            "id": "root",
            "type": "agent",
            "label": "Researcher",
            "config": {
                "subtype": "researcher",
                "user_prompt_template": (
                    "## Sub-Questions to Answer\n"
                    "1. What source-backed facts answer {query}?\n"
                    "2. Which sources conflict?\n"
                    "3. What is unknown?\n"
                    "4. What evidence is strongest?\n"
                    "5. What should the final report use?\n"
                ),
            },
        }
    }

    event = _mutation_event_for_ast_change(
        tool_name="update_block",
        tool_call_id="tool_update_block_architect",
        raw_ast=latest_ast,
        last_ast_seen=stale_ast,
        normalization_fixes=[],
    )

    assert event is not None
    assert event.tool_name == "update_block"
    assert event.old_ast == stale_ast
    assert event.new_ast == latest_ast


def test_workflow_local_model_tiers_are_applied_to_designer_client() -> None:
    class _FakeFrameworkClient:
        def __init__(self) -> None:
            self.mapping: dict[str, Any] | None = None

        def derive(self, mapping: dict[str, Any]) -> str:
            self.mapping = mapping
            return "derived-client"

    workflow = load_workflow_from_dict({
        "id": "designer",
        "name": "Designer",
        "version": 1,
        "models": {
            "critic": {
                "endpoints": ["databricks-gpt-5-5"],
                "fallback_on_429": False,
                "rotation_strategy": "PRIORITY",
            }
        },
        "root": {
            "id": "root",
            "type": "agent",
            "label": "Root",
            "config": {
                "subtype": "synthesizer",
                "model_tier": "critic",
                "output_key": "output",
            },
        },
    })
    client = _FakeFrameworkClient()

    derived = _derive_workflow_llm_client(client, workflow)

    assert derived == "derived-client"
    assert client.mapping is not None
    assert "critic" in client.mapping
