"""Unit tests for the save-time critic gate in agents_v2.

Tests focus on the gate helper functions (``_extract_intent_from_definition``,
``_run_save_critic_gate``, ``_raise_if_critic_blocks``) so the gate logic is
covered without needing the full FastAPI test client + DB stack. The
integration of the gate into the route handlers is exercised by the broader
api unit suite (106 tests, all still passing).
"""
from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from deep_research.agent_designer.workflow_critic import CritiqueResult
from deep_research.api.v1.agents_v2 import (
    _extract_intent_from_definition,
    _raise_if_critic_blocks,
    _run_save_critic_gate,
)

# ---------------------------------------------------------------------------
# _extract_intent_from_definition
# ---------------------------------------------------------------------------


class TestExtractIntent:
    def test_returns_designer_goal_from_plan_and_execute_metadata(self) -> None:
        wf = {
            "root": {
                "type": "sequence",
                "children": [
                    {
                        "type": "plan_and_execute",
                        "config": {
                            "synthesis_metadata": {
                                "designer_goal": "investment research on NVDA",
                                "designer_domain": "Investment Research",
                            },
                            "body": {"type": "sequence", "children": []},
                        },
                    }
                ],
            }
        }
        assert (
            _extract_intent_from_definition(wf) == "investment research on NVDA"
        )

    def test_recurses_into_body(self) -> None:
        wf = {
            "root": {
                "type": "plan_and_execute",
                "config": {
                    "body": {
                        "type": "plan_and_execute",
                        "config": {
                            "synthesis_metadata": {
                                "designer_goal": "nested goal text"
                            },
                        },
                    },
                },
            }
        }
        assert _extract_intent_from_definition(wf) == "nested goal text"

    def test_returns_empty_when_missing(self) -> None:
        wf = {
            "root": {
                "type": "agent",
                "config": {"subtype": "researcher"},
                "children": [],
            }
        }
        assert _extract_intent_from_definition(wf) == ""

    def test_handles_malformed_input(self) -> None:
        assert _extract_intent_from_definition({}) == ""
        assert _extract_intent_from_definition({"root": None}) == ""
        assert _extract_intent_from_definition(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _run_save_critic_gate
# ---------------------------------------------------------------------------


def _wf_with_intent(intent: str) -> dict[str, Any]:
    return {
        "root": {
            "type": "plan_and_execute",
            "config": {
                "synthesis_metadata": {"designer_goal": intent},
                "body": {
                    "type": "sequence",
                    "children": [
                        {
                            "type": "agent",
                            "label": "a1",
                            "config": {
                                "subtype": "researcher",
                                "system_prompt": "x" * 200,
                            },
                        }
                    ],
                },
            },
        }
    }


class TestRunSaveCriticGate:
    @pytest.mark.asyncio
    async def test_skips_when_no_intent(self, monkeypatch) -> None:
        """An AST with no designer_goal — i.e., a legacy agent — bypasses
        the gate entirely (returns (None, [])). No LLM call is made."""
        wf = {"root": {"type": "agent", "config": {}, "children": []}}
        called = False

        async def _no_call(**_: Any) -> CritiqueResult:
            nonlocal called
            called = True
            return CritiqueResult(verdict="pass", summary="ok")

        monkeypatch.setattr(
            "deep_research.api.v1.agents_v2.critique_workflow_against_intent",
            _no_call,
        )
        critique, warnings = await _run_save_critic_gate(
            wf, llm_client=object(), force=False
        )
        assert critique is None
        assert warnings == []
        assert called is False

    @pytest.mark.asyncio
    async def test_skips_when_no_llm_client(self) -> None:
        wf = _wf_with_intent("research X")
        critique, warnings = await _run_save_critic_gate(
            wf, llm_client=None, force=False
        )
        assert critique is None
        assert warnings == []

    @pytest.mark.asyncio
    async def test_skips_when_adapter_init_fails(self, monkeypatch) -> None:
        """If the AppLLMAdapter cannot be constructed, the gate must fail
        open — never block save on infrastructure problems."""
        wf = _wf_with_intent("research X")

        class _BoomAdapter:
            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("adapter unavailable")

        # Patch the module so the import inside _run_save_critic_gate fails.
        # Inserting a synthetic module is the cleanest path.
        import sys
        import types

        fake_module = types.ModuleType(
            "deep_research.agent.adapters.llm_adapter"
        )
        fake_module.AppLLMAdapter = _BoomAdapter  # type: ignore[attr-defined]
        monkeypatch.setitem(
            sys.modules,
            "deep_research.agent.adapters.llm_adapter",
            fake_module,
        )
        critique, warnings = await _run_save_critic_gate(
            wf, llm_client=object(), force=False
        )
        assert critique is None
        assert warnings == []

    @pytest.mark.asyncio
    async def test_pass_verdict_returns_no_warnings(self, monkeypatch) -> None:
        wf = _wf_with_intent("research X")

        async def _pass(**_: Any) -> CritiqueResult:
            return CritiqueResult(verdict="pass", summary="all good")

        monkeypatch.setattr(
            "deep_research.api.v1.agents_v2.critique_workflow_against_intent",
            _pass,
        )
        # Stub the AppLLMAdapter so the gate can build it.
        import sys
        import types

        fake_module = types.ModuleType(
            "deep_research.agent.adapters.llm_adapter"
        )

        class _OK:
            def __init__(self, *_: Any, **__: Any) -> None: ...

        fake_module.AppLLMAdapter = _OK  # type: ignore[attr-defined]
        monkeypatch.setitem(
            sys.modules,
            "deep_research.agent.adapters.llm_adapter",
            fake_module,
        )
        critique, warnings = await _run_save_critic_gate(
            wf, llm_client=object(), force=False
        )
        assert critique is not None
        assert critique.verdict == "pass"
        assert warnings == []

    @pytest.mark.asyncio
    async def test_needs_revision_warns_without_blocking(self, monkeypatch) -> None:
        wf = _wf_with_intent("research X")

        async def _nr(**_: Any) -> CritiqueResult:
            return CritiqueResult(verdict="needs_revision", summary="be more specific")

        monkeypatch.setattr(
            "deep_research.api.v1.agents_v2.critique_workflow_against_intent",
            _nr,
        )
        import sys
        import types

        fake_module = types.ModuleType(
            "deep_research.agent.adapters.llm_adapter"
        )

        class _OK:
            def __init__(self, *_: Any, **__: Any) -> None: ...

        fake_module.AppLLMAdapter = _OK  # type: ignore[attr-defined]
        monkeypatch.setitem(
            sys.modules,
            "deep_research.agent.adapters.llm_adapter",
            fake_module,
        )
        critique, warnings = await _run_save_critic_gate(
            wf, llm_client=object(), force=False
        )
        assert critique is not None
        assert critique.verdict == "needs_revision"
        assert len(warnings) == 1
        assert "needs_revision" in warnings[0]


# ---------------------------------------------------------------------------
# _raise_if_critic_blocks
# ---------------------------------------------------------------------------


class TestRaiseIfBlocks:
    def test_pass_does_not_raise(self) -> None:
        _raise_if_critic_blocks(
            CritiqueResult(verdict="pass", summary="ok"), force=False
        )

    def test_needs_revision_does_not_raise(self) -> None:
        _raise_if_critic_blocks(
            CritiqueResult(verdict="needs_revision", summary="x"), force=False
        )

    def test_fail_raises_422(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _raise_if_critic_blocks(
                CritiqueResult(verdict="fail", summary="off-topic"),
                force=False,
            )
        assert exc_info.value.status_code == 422
        assert "critique" in exc_info.value.detail
        assert exc_info.value.detail["critique"]["verdict"] == "fail"
        assert "?force=true" in exc_info.value.detail["message"]

    def test_fail_with_force_does_not_raise(self) -> None:
        # ?force=true → save proceeds even on fail.
        _raise_if_critic_blocks(
            CritiqueResult(verdict="fail", summary="off-topic"), force=True
        )

    def test_none_critique_does_not_raise(self) -> None:
        # When the gate was skipped (e.g., no intent), critique is None and
        # the gate must not block.
        _raise_if_critic_blocks(None, force=False)
