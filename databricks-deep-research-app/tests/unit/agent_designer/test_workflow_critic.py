"""Unit tests for ``workflow_critic`` — the LLM-as-judge critic that verifies
the generated workflow actually answers the user's request.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from deep_research.agent_designer.workflow_critic import (
    AgentFinding,
    CoverageGap,
    CritiqueResult,
    OutputGap,
    _build_critic_messages,
    _critique_tool_schema,
    _extract_agents,
    _fallback_critique,
    critique_workflow_against_intent,
)

# ---------------------------------------------------------------------------
# Fakes mirroring the orchestrator's LLMStreamChunk / LLMToolCall shape
# ---------------------------------------------------------------------------


@dataclass
class _FakeToolCall:
    id: str
    name: str
    arguments: Any


@dataclass
class _FakeChunk:
    content: str | None = None
    tool_call: _FakeToolCall | None = None
    finish: bool = False


class _FakeLLM:
    """Streams a single tool-call chunk then a finish chunk."""

    def __init__(self, args: Any | None, *, name: str = "emit_critique") -> None:
        self._args = args
        self._name = name

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[_FakeChunk]:
        async def _gen() -> AsyncIterator[_FakeChunk]:
            if self._args is not None:
                yield _FakeChunk(
                    tool_call=_FakeToolCall(
                        id="t1", name=self._name, arguments=self._args
                    )
                )
            yield _FakeChunk(finish=True)

        return _gen()


class _RaisingLLM:
    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[_FakeChunk]:
        async def _gen() -> AsyncIterator[_FakeChunk]:
            raise RuntimeError("provider exploded")
            yield  # pragma: no cover — unreachable

        return _gen()


# ---------------------------------------------------------------------------
# Helpers for building toy workflow definitions
# ---------------------------------------------------------------------------


def _agent(
    node_id: str,
    *,
    system_prompt: str,
    subtype: str = "researcher",
    tools: list[str] | None = None,
    model_tier: str = "analytical",
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": "agent",
        "label": node_id,
        "config": {
            "subtype": subtype,
            "system_prompt": system_prompt,
            "tools": tools or [],
            "model_tier": model_tier,
        },
        "children": [],
    }


def _wrap_root(*children: dict[str, Any]) -> dict[str, Any]:
    return {
        "root": {
            "id": "root",
            "type": "sequence",
            "label": "root",
            "config": {},
            "children": list(children),
        },
        "tools": [],
    }


# ---------------------------------------------------------------------------
# _extract_agents
# ---------------------------------------------------------------------------


class TestExtractAgents:
    def test_collects_top_level_agents(self) -> None:
        wf = _wrap_root(
            _agent("a1", system_prompt="prompt A"),
            _agent("a2", system_prompt="prompt B"),
        )
        agents = _extract_agents(wf)
        assert [a["node_path"] for a in agents] == [
            "root.children[0]",
            "root.children[1]",
        ]
        assert agents[0]["label"] == "a1"
        assert agents[0]["subtype"] == "researcher"
        assert agents[0]["system_prompt_excerpt"] == "prompt A"

    def test_recurses_into_plan_and_execute(self) -> None:
        wf = {
            "root": {
                "id": "root",
                "type": "plan_and_execute",
                "label": "p&e",
                "config": {
                    "planner": {
                        "subtype": "planner",
                        "system_prompt": "planner prompt",
                        "label": "planner",
                    },
                    "evaluator": {
                        "subtype": "evaluator",
                        "system_prompt": "evaluator prompt",
                        "label": "evaluator",
                    },
                    "body": {
                        "id": "body",
                        "type": "sequence",
                        "label": "body",
                        "config": {},
                        "children": [_agent("body_agent", system_prompt="body prompt")],
                    },
                },
                "children": [],
            },
            "tools": [],
        }
        agents = _extract_agents(wf)
        labels = [a["label"] for a in agents]
        assert "planner" in labels
        assert "evaluator" in labels
        assert "body_agent" in labels

    def test_excerpt_truncates_long_prompts(self) -> None:
        long_prompt = "x" * 5000
        wf = _wrap_root(_agent("a", system_prompt=long_prompt))
        agents = _extract_agents(wf)
        excerpt = agents[0]["system_prompt_excerpt"]
        assert len(excerpt) < len(long_prompt)
        assert excerpt.endswith("...(truncated)")


# ---------------------------------------------------------------------------
# critique_workflow_against_intent
# ---------------------------------------------------------------------------


class TestCritiqueFlow:
    @pytest.mark.asyncio
    async def test_pass_verdict_round_trips(self) -> None:
        wf = _wrap_root(_agent("a", system_prompt="prompt"))
        llm = _FakeLLM(
            args={
                "verdict": "pass",
                "summary": "All agents address the intent.",
                "agent_findings": [],
                "coverage_gaps": [],
                "output_gaps": [],
            }
        )
        result = await critique_workflow_against_intent(
            definition=wf,
            intent="research AAPL",
            required_outputs=["thesis"],
            llm=llm,
        )
        assert isinstance(result, CritiqueResult)
        assert result.verdict == "pass"
        assert result.summary == "All agents address the intent."
        assert result.agent_findings == []

    @pytest.mark.asyncio
    async def test_needs_revision_verdict_with_findings_round_trips(self) -> None:
        wf = _wrap_root(_agent("a1", system_prompt="thin"))
        llm = _FakeLLM(
            args={
                "verdict": "needs_revision",
                "summary": "Two agents are shallow.",
                "agent_findings": [
                    {
                        "node_path": "root.children[0]",
                        "label": "a1",
                        "severity": "needs_revision",
                        "finding": "Agent system_prompt does not name the topic.",
                        "suggested_action": "Call update_block on this node with 80-300 words.",
                    },
                ],
                "coverage_gaps": [],
                "output_gaps": [],
            }
        )
        result = await critique_workflow_against_intent(
            definition=wf, intent="research X", required_outputs=[], llm=llm
        )
        assert result.verdict == "needs_revision"
        assert len(result.agent_findings) == 1
        assert result.agent_findings[0].severity == "needs_revision"
        assert "update_block" in result.agent_findings[0].suggested_action

    @pytest.mark.asyncio
    async def test_fail_verdict_with_coverage_and_output_gaps(self) -> None:
        wf = _wrap_root(_agent("a1", system_prompt="off-topic"))
        llm = _FakeLLM(
            args={
                "verdict": "fail",
                "summary": "Workflow does not answer the request.",
                "agent_findings": [],
                "coverage_gaps": [
                    {
                        "aspect": "competitor analysis",
                        "rationale": "No agent investigates competitors.",
                    }
                ],
                "output_gaps": [
                    {
                        "required_output": "Competitor benchmarking table",
                        "rationale": "No agent emits competitor data.",
                    }
                ],
            }
        )
        result = await critique_workflow_against_intent(
            definition=wf,
            intent="investment research on NVDA",
            required_outputs=["Competitor benchmarking table"],
            llm=llm,
        )
        assert result.verdict == "fail"
        assert isinstance(result.coverage_gaps[0], CoverageGap)
        assert isinstance(result.output_gaps[0], OutputGap)

    @pytest.mark.asyncio
    async def test_args_as_json_string_parses(self) -> None:
        """Some providers serialize tool-call args as JSON strings, not dicts.
        The parser must handle both."""
        wf = _wrap_root(_agent("a", system_prompt="p"))
        args_json = json.dumps(
            {
                "verdict": "pass",
                "summary": "ok",
                "agent_findings": [],
                "coverage_gaps": [],
                "output_gaps": [],
            }
        )
        llm = _FakeLLM(args=args_json)
        result = await critique_workflow_against_intent(
            definition=wf, intent="research X", required_outputs=[], llm=llm
        )
        assert result.verdict == "pass"


class TestFallbackPaths:
    @pytest.mark.asyncio
    async def test_empty_intent_returns_fallback(self) -> None:
        wf = _wrap_root(_agent("a", system_prompt="p"))
        llm = _FakeLLM(args=None)
        result = await critique_workflow_against_intent(
            definition=wf, intent="", required_outputs=[], llm=llm
        )
        # Empty intent → fallback before LLM is even called.
        assert result.verdict == "needs_revision"
        assert "empty intent" in result.summary

    @pytest.mark.asyncio
    async def test_missing_root_returns_fallback(self) -> None:
        llm = _FakeLLM(args=None)
        result = await critique_workflow_against_intent(
            definition={"tools": []},  # no root
            intent="research X",
            required_outputs=[],
            llm=llm,
        )
        assert result.verdict == "needs_revision"
        assert "root" in result.summary

    @pytest.mark.asyncio
    async def test_no_agents_returns_fallback(self) -> None:
        wf = {
            "root": {
                "id": "root",
                "type": "sequence",
                "label": "root",
                "config": {},
                "children": [],
            },
            "tools": [],
        }
        llm = _FakeLLM(args=None)
        result = await critique_workflow_against_intent(
            definition=wf, intent="research X", required_outputs=[], llm=llm
        )
        assert result.verdict == "needs_revision"

    @pytest.mark.asyncio
    async def test_llm_raises_returns_fallback(self) -> None:
        wf = _wrap_root(_agent("a", system_prompt="p"))
        result = await critique_workflow_against_intent(
            definition=wf,
            intent="research X",
            required_outputs=[],
            llm=_RaisingLLM(),
        )
        assert result.verdict == "needs_revision"
        assert "raised" in result.summary

    @pytest.mark.asyncio
    async def test_llm_returns_no_tool_call_returns_fallback(self) -> None:
        wf = _wrap_root(_agent("a", system_prompt="p"))
        # FakeLLM with args=None streams only the finish chunk.
        result = await critique_workflow_against_intent(
            definition=wf,
            intent="research X",
            required_outputs=[],
            llm=_FakeLLM(args=None),
        )
        assert result.verdict == "needs_revision"
        assert "emit_critique" in result.summary

    @pytest.mark.asyncio
    async def test_malformed_tool_args_returns_fallback(self) -> None:
        wf = _wrap_root(_agent("a", system_prompt="p"))
        result = await critique_workflow_against_intent(
            definition=wf,
            intent="research X",
            required_outputs=[],
            llm=_FakeLLM(args={"not_a_verdict_field": "garbage"}),
        )
        assert result.verdict == "needs_revision"
        assert "validation" in result.summary

    def test_fallback_critique_helper(self) -> None:
        result = _fallback_critique("some reason")
        assert result.verdict == "needs_revision"
        assert "some reason" in result.summary


class TestPromptShape:
    def test_critic_messages_include_intent_and_agents(self) -> None:
        wf = _wrap_root(_agent("a1", system_prompt="prompt"))
        agents = _extract_agents(wf)
        messages = _build_critic_messages(
            intent="investment research on NVDA",
            required_outputs=["thesis"],
            agents=agents,
        )
        assert messages[0]["role"] == "system"
        assert "Workflow Critic" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        payload = json.loads(messages[1]["content"])
        assert payload["intent"] == "investment research on NVDA"
        assert payload["required_outputs"] == ["thesis"]
        assert len(payload["agents"]) == 1

    def test_tool_schema_constrains_critique_shape(self) -> None:
        schema = _critique_tool_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "emit_critique"
        params = schema["function"]["parameters"]
        # Confirm Pydantic-derived schema declares the required fields.
        assert "verdict" in params["properties"]
        assert "summary" in params["properties"]
        assert "agent_findings" in params["properties"]
        assert set(params["properties"]["verdict"]["enum"]) == {
            "pass",
            "needs_revision",
            "fail",
        }


class TestAgentFindingStructure:
    def test_agent_finding_severity_enum(self) -> None:
        f = AgentFinding(
            node_path="root.children[0]",
            label="a",
            severity="fail",
            finding="x",
            suggested_action="y",
        )
        assert f.severity == "fail"

    def test_agent_finding_rejects_unknown_severity(self) -> None:
        from pydantic import ValidationError as _ValidationError

        with pytest.raises(_ValidationError):
            AgentFinding(
                node_path="root.children[0]",
                label="a",
                severity="catastrophic",  # type: ignore[arg-type]
                finding="x",
                suggested_action="y",
            )
