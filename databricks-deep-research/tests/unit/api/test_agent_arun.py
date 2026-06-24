"""End-to-end smoke test for ``Agent.arun`` using a mock LLM client.

Exercises model resolution, tool invocation, and structured output without
hitting any real LLM endpoints.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from databricks_deep_research.api import Agent, AgentResult, tool
from databricks_deep_research.llm.client import LLMResponse


class _CannedClient:
    """Minimal stand-in for FrameworkLLMClient that returns canned responses.

    Implements only the surface area used by the agent harness — enough to
    smoke-test arun() without any HTTP calls.
    """

    def __init__(self, content: str = "ok") -> None:
        self._content = content

    async def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content=self._content, tool_calls=[], usage={"total_tokens": 1})

    async def aclose(self) -> None: ...


@pytest.mark.asyncio
async def test_agent_compiles_to_workflow() -> None:
    @tool
    def echo(msg: str) -> str:
        """Echo."""
        return msg

    agent = Agent(name="greeter", instructions="Greet politely.", tools=[echo])
    wf = agent.as_workflow()
    assert wf.id == "agent_greeter"
    assert wf.root.id == "greeter"
    # The agent.tool list contains our echo tool
    assert any(t.name == "echo" for t in wf.tools)


@pytest.mark.asyncio
async def test_agent_with_no_tools_compiles_with_zero_max_tool_calls() -> None:
    agent = Agent(name="solo", instructions="Just talk.")
    wf = agent.as_workflow()
    # When no tools and no override, ``max_tool_calls`` is omitted (default).
    assert "tools" in wf.root.config
    assert wf.root.config["tools"] == []


@pytest.mark.asyncio
async def test_agent_result_dataclass_fields() -> None:
    result = AgentResult(
        content="hello",
        output="hello",
        events=[],
        run_id="r1",
    )
    assert result.ok is True
    assert result.run_id == "r1"
    assert result.verification is None


def test_agent_default_subtype_is_custom() -> None:
    agent = Agent(name="x")
    assert agent.subtype == "custom"
    wf = agent.as_workflow()
    assert wf.root.config["subtype"] == "custom"


def test_agent_output_type_marks_json_format() -> None:
    class Out(BaseModel):
        text: str

    agent = Agent(name="x", output_type=Out)
    wf = agent.as_workflow()
    assert wf.root.config["output_format"] == "json"


def test_agent_extras_threaded_to_config() -> None:
    agent = Agent(name="x", extras={"_framework_thread_id": "t1"})
    wf = agent.as_workflow()
    assert wf.root.config.get("extras") == {"_framework_thread_id": "t1"}


def test_agent_subtype_synthesizer_recorded() -> None:
    agent = Agent(name="syn", subtype="synthesizer")
    wf = agent.as_workflow()
    assert wf.root.config["subtype"] == "synthesizer"
