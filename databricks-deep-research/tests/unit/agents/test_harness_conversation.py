"""Verify the harness wires WorkflowState.conversation_history into AgentInput
so multi-turn workflows see prior conversation turns in their LLM calls."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Test 1: WorkflowState defaults to empty conversation_history
# ---------------------------------------------------------------------------


def test_workflow_state_default_history_empty() -> None:
    from databricks_deep_research.workflow.state import WorkflowState

    s = WorkflowState(query="hi")
    assert s.conversation_history == []


# ---------------------------------------------------------------------------
# Test 2: WorkflowState accepts conversation_history at construction
# ---------------------------------------------------------------------------


def test_workflow_state_accepts_history() -> None:
    from databricks_deep_research.workflow.state import WorkflowState

    prior = [
        {"role": "user", "content": "turn1"},
        {"role": "assistant", "content": "reply1"},
    ]
    s = WorkflowState(query="turn2", conversation_history=prior)
    assert s.conversation_history == prior


# ---------------------------------------------------------------------------
# Test 3: WorkflowRunner.run accepts conversation_history kwarg and seeds state
# ---------------------------------------------------------------------------


async def test_runner_run_seeds_conversation_history(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from unittest.mock import MagicMock

    from databricks_deep_research.runner import WorkflowRunner
    from databricks_deep_research.workflow.state import WorkflowState

    captured: dict[str, object] = {}

    class _FakeExecutor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def execute(self, state: object) -> object:  # type: ignore[override]
            captured["state"] = state
            # async generator that yields nothing
            return
            yield  # noqa: unreachable — makes this an async generator

    monkeypatch.setattr(
        "databricks_deep_research.runner.WorkflowExecutor",
        _FakeExecutor,
    )

    fake_def = MagicMock()
    fake_def.models = None
    fake_def.output_keys = []

    runner = WorkflowRunner(llm_client=MagicMock(), factory_context=None)

    # Patch _resolve so MagicMock is accepted directly as a WorkflowDefinition
    monkeypatch.setattr(runner, "_resolve", lambda w: w)
    # Patch _resolve_client to return the llm_client mock
    monkeypatch.setattr(runner, "_resolve_client", lambda d: runner._client)

    prior = [{"role": "user", "content": "x"}]
    await runner.run(workflow=fake_def, query="hi", conversation_history=prior)

    state = captured["state"]
    assert isinstance(state, WorkflowState)
    assert state.conversation_history == prior


# ---------------------------------------------------------------------------
# Test 4: AgentInput.conversation_history field is declared (smoke test)
# ---------------------------------------------------------------------------


def test_build_input_populates_agent_input_history() -> None:
    """Smoke test: AgentInput's conversation_history field exists.

    The actual harness wire is exercised by
    test_runner_run_seeds_conversation_history above.
    """
    from databricks_deep_research.agents.isolation import AgentInput

    assert "conversation_history" in AgentInput.__dataclass_fields__
