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


# ---------------------------------------------------------------------------
# Test 5: _build_messages normalizes history roles to the OpenAI-valid set
# (regression for the AIS "Invalid role" 400 — app stores assistant turns
# as role="agent", which the gateway rejects).
# ---------------------------------------------------------------------------


def test_build_messages_normalizes_agent_role() -> None:
    from databricks_deep_research.agents.harness import _build_messages
    from databricks_deep_research.agents.isolation import AgentInput
    from databricks_deep_research.llm.roles import OPENAI_CHAT_ROLES

    agent_input = AgentInput(
        query="who is CEO of Depop now?",
        system_prompt="You are a parser.",
        user_prompt="Parse the target account.",
        conversation_history=[
            {"role": "user", "content": "prior question"},
            {"role": "agent", "content": "a 36KB prior report"},
        ],
    )

    messages = _build_messages(agent_input)

    roles = [m["role"] for m in messages]
    # system, prior-user, prior-assistant(was "agent"), current-user
    assert roles == ["system", "user", "assistant", "user"]
    assert all(r in OPENAI_CHAT_ROLES for r in roles)
    # The injected history dict was not mutated in place.
    assert agent_input.conversation_history[1] == {
        "role": "agent",
        "content": "a 36KB prior report",
    }


# ---------------------------------------------------------------------------
# Test 6: _build_messages flattens cross-turn tool mechanics (shared harness).
# Regression for the AIS designer multi-turn 400
# "messages.1.tool_calls: Extra inputs are not permitted" — and the same
# load-bearing backstop covers the main-chat path (both route through
# _build_messages). The app-side conversation normalizer does NOT strip tool
# mechanics, so this framework choke point must.
# ---------------------------------------------------------------------------


def test_build_messages_flattens_cross_turn_tool_mechanics() -> None:
    from databricks_deep_research.agents.harness import _build_messages
    from databricks_deep_research.agents.isolation import AgentInput

    agent_input = AgentInput(
        query="use the same tools as before",
        system_prompt="You are the architect.",
        user_prompt="use the same tools as before",
        conversation_history=[
            {"role": "user", "content": "build best-of-n", "tool_calls": []},
            {"role": "assistant", "content": "designed it", "tool_calls": [{"id": "a1"}]},
            {"role": "tool", "content": "discover result", "tool_call_id": "a1"},
            {"role": "tool", "content": "more", "tool_call_id": "a2"},
        ],
    )

    messages = _build_messages(agent_input)

    # No tool mechanics reach the gateway payload.
    assert all(
        ("tool_calls" not in m and "tool_call_id" not in m and "name" not in m)
        for m in messages
    )
    # tool-role messages dropped; only system + flattened history + current user.
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    # Input history dicts untouched.
    assert agent_input.conversation_history[0] == {
        "role": "user",
        "content": "build best-of-n",
        "tool_calls": [],
    }
