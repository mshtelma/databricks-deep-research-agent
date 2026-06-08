"""Adapter from the app's LLMClient streaming-with-tools method into the
orchestrator's structural LLMClientProto.

The app's ``LLMClient.stream_with_tools`` yields ``StreamWithToolsChunk``
objects that may carry:

- ``content``    — partial text from the LLM
- ``tool_calls`` — a list of completed ``ToolCall`` objects (emitted once on
                   the final chunk when ``is_done=True``)
- ``is_done``    — signals end-of-stream

The orchestrator's ``LLMClientProto.stream`` yields ``LLMStreamChunk``
objects with a one-tool-call-per-chunk shape:

- ``content``   — text fragment
- ``tool_call`` — exactly one ``LLMToolCall`` (or None)
- ``finish``    — True on the last chunk

This adapter fans out each ``StreamWithToolsChunk.tool_calls`` list into
individual ``LLMStreamChunk`` items, then emits a finish chunk.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from deep_research.agent_designer.orchestrator import LLMStreamChunk, LLMToolCall
from deep_research.services.llm.types import ModelTier

# Default tier used for the Agent Designer chat session.
# Analytical gives a good balance of reasoning quality and latency.
_DEFAULT_TIER = ModelTier.ANALYTICAL


class AppLLMAdapter:
    """Wraps the app's existing LLMClient into the orchestrator's LLMClientProto.

    The adapter translates ``LLMClient.stream_with_tools`` (which yields
    ``StreamWithToolsChunk`` with a list of tool calls on the final chunk) into
    the orchestrator's ``LLMClientProto.stream`` shape (one ``LLMStreamChunk``
    per tool call, then a finish chunk).
    """

    def __init__(self, app_llm_client: Any, tier: ModelTier = _DEFAULT_TIER) -> None:
        self._llm = app_llm_client
        self._tier = tier

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[LLMStreamChunk]:
        """Adapt stream_with_tools output into LLMStreamChunk events.

        Yields:
            LLMStreamChunk(content=...) for each content fragment.
            LLMStreamChunk(tool_call=...) for each completed tool call.
            LLMStreamChunk(finish=True) as the final sentinel.
        """
        async for swt_chunk in self._llm.stream_with_tools(
            messages=messages,
            tools=tools,
            tier=self._tier,
        ):
            # Yield content fragment (may be empty string — skip those)
            if swt_chunk.content:
                yield LLMStreamChunk(content=swt_chunk.content)

            # Fan out tool calls into individual chunks
            if swt_chunk.tool_calls:
                for tc in swt_chunk.tool_calls:
                    yield LLMStreamChunk(
                        tool_call=LLMToolCall(
                            id=tc.id,
                            name=tc.name,
                            arguments=tc.arguments,
                        )
                    )

            # Emit finish sentinel on the last chunk
            if swt_chunk.is_done:
                yield LLMStreamChunk(finish=True)
                return
