"""Verify FrameworkLLMClient emits a ModelCallEvent after model resolution."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_complete_emits_model_call_event() -> None:
    from databricks_deep_research.events.types import ModelCallEvent
    from databricks_deep_research.llm.client import FrameworkLLMClient

    # Mock the AsyncOpenAI client so .chat.completions.create returns a fake
    fake_openai = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="hi", tool_calls=None))]
    fake_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    fake_response.model = "databricks-claude-opus-4-6"
    fake_openai.chat.completions.create = AsyncMock(return_value=fake_response)

    client = FrameworkLLMClient(
        openai_client=fake_openai,
        model_mapping={"complex": "databricks-claude-opus-4-6"},
    )

    captured: list = []
    await client.complete(
        messages=[{"role": "user", "content": "hi"}],
        tier="complex",
        event_sink=lambda ev: captured.append(ev),
        node_id="architect",
    )

    assert len(captured) == 1
    ev = captured[0]
    assert isinstance(ev, ModelCallEvent)
    assert ev.event_type == "model_call"
    assert ev.tier == "complex"
    assert ev.model == "databricks-claude-opus-4-6"
    assert ev.node_id == "architect"


@pytest.mark.asyncio
async def test_complete_no_event_sink_does_not_raise() -> None:
    """Verify complete() works normally when no event_sink is passed."""
    from databricks_deep_research.llm.client import FrameworkLLMClient

    fake_openai = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="hello", tool_calls=None))]
    fake_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    fake_response.model = "databricks-claude-haiku"
    fake_openai.chat.completions.create = AsyncMock(return_value=fake_response)

    client = FrameworkLLMClient(
        openai_client=fake_openai,
        model_mapping={"analytical": "databricks-claude-haiku"},
    )

    result = await client.complete(
        messages=[{"role": "user", "content": "hello"}],
        tier="analytical",
    )
    assert result.content == "hello"
