"""Unit tests for DeepResearchResponsesAgent (US-301/305).

Mocks WorkflowRunner so tests don't require a live workspace. Verifies:
  - load_context lazy-loads runner + definition
  - predict raises before load_context
  - predict_stream emits Responses-API delta events for agent_stream_chunk
  - predict_stream emits a final 'response.output_item.done' event
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research.services.deployment.responses_agent import (
    DeepResearchResponsesAgent,
)


def _write_definition(tmp_path: Path) -> str:
    # Top-level required fields: id, name, root.
    # Node required fields: type, id, label.
    # Sequence requires >=1 child; using a single agent leaf for the test.
    definition = {
        "id": "test-workflow",
        "name": "test-workflow",
        "version": 1,
        "tools": [],
        "root": {
            "type": "sequence",
            "id": "root",
            "label": "Root",
            "children": [
                {
                    "type": "agent",
                    "id": "researcher",
                    "label": "Researcher",
                    "config": {"subtype": "researcher"},
                }
            ],
        },
    }
    p = tmp_path / "workflow_definition.json"
    p.write_text(json.dumps(definition), encoding="utf-8")
    return str(p)


def _make_context(workflow_definition_path: str) -> MagicMock:
    ctx = MagicMock()
    ctx.artifacts = {"workflow_definition": workflow_definition_path}
    return ctx


class TestPredictBeforeLoadContext:
    def test_predict_raises(self) -> None:
        agent = DeepResearchResponsesAgent()
        with pytest.raises(RuntimeError, match="load_context must be called"):
            agent.predict(MagicMock(), {"input": {"text": "hi"}})


class TestLoadContext:
    def test_load_context_initializes_runner_and_definition(
        self, tmp_path: Path
    ) -> None:
        agent = DeepResearchResponsesAgent()
        path = _write_definition(tmp_path)

        with patch(
            "deep_research.services.deployment.responses_agent.WorkflowRunner.from_databricks"
        ) as mock_from_databricks:
            mock_from_databricks.return_value = MagicMock()
            agent.load_context(_make_context(path))

        assert agent._runner is not None
        assert agent._definition is not None
        # WorkflowRunner.from_databricks() takes no args; the workflow
        # definition is passed at run()/stream() time, not at construction.
        mock_from_databricks.assert_called_once_with()


class TestPredict:
    def test_predict_with_text_input(self, tmp_path: Path) -> None:
        agent = DeepResearchResponsesAgent()
        path = _write_definition(tmp_path)

        # Build a fake result: output text + 2 sources.
        fake_result = MagicMock(
            output="research output text",
            sources=[
                MagicMock(url="https://a.example", title="A"),
                MagicMock(url="https://b.example", title="B"),
            ],
        )
        fake_runner = MagicMock()
        fake_runner.run = AsyncMock(return_value=fake_result)

        with patch(
            "deep_research.services.deployment.responses_agent.WorkflowRunner.from_databricks"
        ) as mock_from:
            mock_from.return_value = fake_runner
            agent.load_context(_make_context(path))

        result = agent.predict(MagicMock(), {"input": {"text": "What is X?"}})

        assert result["output"]["text"] == "research output text"
        assert result["output"]["sources"] == [
            {"url": "https://a.example", "title": "A"},
            {"url": "https://b.example", "title": "B"},
        ]
        # run() was called with the loaded definition + the user query.
        call = fake_runner.run.await_args
        assert call.kwargs["query"] == "What is X?"
        assert call.kwargs["workflow"] is agent._definition

    def test_predict_with_messages_input(self, tmp_path: Path) -> None:
        agent = DeepResearchResponsesAgent()
        path = _write_definition(tmp_path)

        fake_result = MagicMock(output="ok", sources=[])
        fake_runner = MagicMock()
        fake_runner.run = AsyncMock(return_value=fake_result)

        with patch(
            "deep_research.services.deployment.responses_agent.WorkflowRunner.from_databricks"
        ) as mock_from:
            mock_from.return_value = fake_runner
            agent.load_context(_make_context(path))

        agent.predict(
            MagicMock(),
            {
                "input": {
                    "messages": [
                        {"role": "user", "content": "hello"},
                        {"role": "user", "content": "follow-up"},
                    ]
                }
            },
        )
        # Last message wins.
        assert fake_runner.run.await_args.kwargs["query"] == "follow-up"


class TestPredictStream:
    @pytest.mark.asyncio
    async def test_emits_delta_for_agent_stream_chunk(
        self, tmp_path: Path
    ) -> None:
        agent = DeepResearchResponsesAgent()
        path = _write_definition(tmp_path)

        # Build 3 events: agent_stream_chunk x2 + a non-chunk event that
        # must NOT yield a delta.
        chunk1 = MagicMock(event_type="agent_stream_chunk", chunk="hello")
        chunk2 = MagicMock(event_type="agent_stream_chunk", chunk=" world")
        node_started = MagicMock(event_type="node_started")

        async def fake_stream(**kwargs):  # noqa: ARG001
            yield node_started
            yield chunk1
            yield chunk2

        fake_runner = MagicMock()
        fake_runner.stream = fake_stream

        with patch(
            "deep_research.services.deployment.responses_agent.WorkflowRunner.from_databricks"
        ) as mock_from:
            mock_from.return_value = fake_runner
            agent.load_context(_make_context(path))

        events = [e async for e in agent.predict_stream(
            MagicMock(), {"input": {"text": "q"}}
        )]

        # Expect: 2 deltas (one per agent_stream_chunk) + 1 final done event.
        # node_started must NOT produce a delta.
        assert len(events) == 3
        assert events[0] == {"type": "response.output_text.delta", "delta": "hello"}
        assert events[1] == {"type": "response.output_text.delta", "delta": " world"}
        assert events[2] == {"type": "response.output_item.done"}

    @pytest.mark.asyncio
    async def test_predict_stream_raises_before_load_context(self) -> None:
        agent = DeepResearchResponsesAgent()
        # Anext-ing a generator that immediately raises is the right idiom.
        gen = agent.predict_stream(MagicMock(), {"input": {"text": "q"}})
        with pytest.raises(RuntimeError, match="load_context must be called"):
            await gen.__anext__()
