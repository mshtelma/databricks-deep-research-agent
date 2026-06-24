"""Unit tests for the follow-up turn gate (intent routing).

These tests are intentionally domain-agnostic — no example/benchmark-specific
strings — and assert the routing *logic*, not any particular topic.
"""

from unittest.mock import AsyncMock

import pytest

from deep_research.agent.followup import (
    FollowupClassification,
    TurnIntent,
    decide_turn_intent,
)
from deep_research.agent.framework_orchestrator import _resolve_turn_intent
from deep_research.agent.orchestration_config import OrchestrationConfig
from deep_research.services.llm.types import LLMResponse


def _classification_response(
    *, answerable: bool, follow_up_type: str, web_searchable: bool = False
) -> LLMResponse:
    classification = FollowupClassification(
        answerable_from_prior_research=answerable,
        follow_up_type=follow_up_type,
        web_searchable=web_searchable,
        reasoning="unit-test reasoning",
    )
    return LLMResponse(
        content=classification.model_dump_json(),
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        endpoint_id="test-endpoint",
        duration_ms=1.0,
        structured=classification,
    )


# ---------------------------------------------------------------------------
# _resolve_turn_intent — request/legacy mapping (synchronous, no LLM)
# ---------------------------------------------------------------------------


class TestResolveTurnIntent:
    def test_legacy_simple_query_mode_on_agent_chat_maps_to_chat(self) -> None:
        config = OrchestrationConfig(query_mode="simple", agent_id="agent-123")
        assert _resolve_turn_intent(config) == TurnIntent.CHAT

    def test_simple_query_mode_without_agent_is_not_forced_to_chat(self) -> None:
        # No agent_id → the legacy mapping does not apply; default turn_intent.
        config = OrchestrationConfig(query_mode="simple", agent_id=None)
        assert _resolve_turn_intent(config) == TurnIntent.AUTO

    def test_explicit_chat_override(self) -> None:
        config = OrchestrationConfig(agent_id="agent-123", turn_intent="chat")
        assert _resolve_turn_intent(config) == TurnIntent.CHAT

    def test_explicit_research_override(self) -> None:
        config = OrchestrationConfig(agent_id="agent-123", turn_intent="research")
        assert _resolve_turn_intent(config) == TurnIntent.RESEARCH

    def test_default_is_auto(self) -> None:
        config = OrchestrationConfig(agent_id="agent-123")
        assert _resolve_turn_intent(config) == TurnIntent.AUTO

    def test_unknown_turn_intent_falls_back_to_auto(self) -> None:
        config = OrchestrationConfig(agent_id="agent-123", turn_intent="nonsense")
        assert _resolve_turn_intent(config) == TurnIntent.AUTO

    def test_explicit_research_overrides_legacy_simple_mapping(self) -> None:
        # NEW precedence: an explicit "research" (the chat UI's per-turn control,
        # now shown for agent chats in every mode) wins over the legacy
        # query_mode == "simple" → chat shorthand. Without this, Simple + an
        # explicit re-run request would be silently forced into a chat turn.
        config = OrchestrationConfig(
            query_mode="simple", agent_id="agent-123", turn_intent="research"
        )
        assert _resolve_turn_intent(config) == TurnIntent.RESEARCH

    def test_explicit_chat_with_simple_mode_stays_chat(self) -> None:
        config = OrchestrationConfig(
            query_mode="simple", agent_id="agent-123", turn_intent="chat"
        )
        assert _resolve_turn_intent(config) == TurnIntent.CHAT

    def test_simple_mode_auto_still_maps_to_chat_on_agent_chat(self) -> None:
        # Back-compat: with the default (auto) intent, Simple on a custom-agent
        # chat still means "answer from gathered data" (legacy clients that only
        # sent query_mode and never an explicit turn_intent).
        config = OrchestrationConfig(
            query_mode="simple", agent_id="agent-123", turn_intent="auto"
        )
        assert _resolve_turn_intent(config) == TurnIntent.CHAT


# ---------------------------------------------------------------------------
# decide_turn_intent — routing (async; LLM mocked for AUTO only)
# ---------------------------------------------------------------------------


class TestDecideTurnIntent:
    @pytest.mark.asyncio
    async def test_explicit_research_skips_classification(self) -> None:
        llm = AsyncMock()
        decision = await decide_turn_intent(
            query="anything",
            conversation_history=[],
            prior_findings_summary="prior stuff",
            has_prior_research=True,
            requested=TurnIntent.RESEARCH,
            llm=llm,
        )
        assert decision.route == "research"
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_chat_skips_classification(self) -> None:
        llm = AsyncMock()
        decision = await decide_turn_intent(
            query="anything",
            conversation_history=[],
            prior_findings_summary="prior stuff",
            has_prior_research=True,
            requested=TurnIntent.CHAT,
            llm=llm,
        )
        assert decision.route == "chat"
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_without_prior_research_runs_research(self) -> None:
        llm = AsyncMock()
        decision = await decide_turn_intent(
            query="anything",
            conversation_history=[],
            prior_findings_summary="",
            has_prior_research=False,
            requested=TurnIntent.AUTO,
            llm=llm,
        )
        assert decision.route == "research"
        # First turn: never spend an LLM call.
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_answerable_from_prior_routes_chat(self) -> None:
        # Regression for the core bug: an obscure fact that is NOT general
        # knowledge but IS in the gathered material must route to chat.
        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=_classification_response(
                answerable=True, follow_up_type="complex_follow_up"
            )
        )
        decision = await decide_turn_intent(
            query="who is the head of the org we researched?",
            conversation_history=[{"role": "assistant", "content": "report text"}],
            prior_findings_summary="findings naming the head of the org",
            has_prior_research=True,
            requested=TurnIntent.AUTO,
            llm=llm,
        )
        assert decision.route == "chat"
        llm.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_auto_clarification_routes_chat_even_if_not_marked_answerable(
        self,
    ) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=_classification_response(
                answerable=False, follow_up_type="clarification"
            )
        )
        decision = await decide_turn_intent(
            query="what did you mean by that?",
            conversation_history=[{"role": "assistant", "content": "report text"}],
            prior_findings_summary="findings",
            has_prior_research=True,
            requested=TurnIntent.AUTO,
            llm=llm,
        )
        assert decision.route == "chat"

    @pytest.mark.asyncio
    async def test_auto_new_topic_runs_research(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=_classification_response(
                answerable=False, follow_up_type="new_topic"
            )
        )
        decision = await decide_turn_intent(
            query="now research a completely different company",
            conversation_history=[{"role": "assistant", "content": "report text"}],
            prior_findings_summary="findings",
            has_prior_research=True,
            requested=TurnIntent.AUTO,
            llm=llm,
        )
        assert decision.route == "research"

    @pytest.mark.asyncio
    async def test_auto_classification_failure_defaults_to_research(self) -> None:
        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("gateway boom"))
        decision = await decide_turn_intent(
            query="anything",
            conversation_history=[],
            prior_findings_summary="findings",
            has_prior_research=True,
            requested=TurnIntent.AUTO,
            llm=llm,
        )
        # Never silently drop a real research request on classifier failure.
        assert decision.route == "research"


# ---------------------------------------------------------------------------
# decide_turn_intent — bounded live-search escape hatch (spec §4.7, Tier 3)
# ---------------------------------------------------------------------------


class TestDecideTurnIntentLiveSearch:
    @pytest.mark.asyncio
    async def test_not_in_pool_but_web_searchable_routes_live_search(self) -> None:
        # The escape hatch: not answerable from prior, but a focused web lookup.
        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=_classification_response(
                answerable=False,
                follow_up_type="complex_follow_up",
                web_searchable=True,
            )
        )
        decision = await decide_turn_intent(
            query="what is today's price for the thing we researched?",
            conversation_history=[{"role": "assistant", "content": "report text"}],
            prior_findings_summary="findings without the live figure",
            has_prior_research=True,
            requested=TurnIntent.AUTO,
            llm=llm,
            allow_live_search=True,
        )
        assert decision.route == "live_search"

    @pytest.mark.asyncio
    async def test_live_search_off_by_default_routes_research(self) -> None:
        # Same classification, but the caller did NOT opt in → legacy research.
        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=_classification_response(
                answerable=False,
                follow_up_type="complex_follow_up",
                web_searchable=True,
            )
        )
        decision = await decide_turn_intent(
            query="what is today's price for the thing we researched?",
            conversation_history=[{"role": "assistant", "content": "report text"}],
            prior_findings_summary="findings",
            has_prior_research=True,
            requested=TurnIntent.AUTO,
            llm=llm,
            # allow_live_search omitted → defaults False (byte-identical legacy)
        )
        assert decision.route == "research"

    @pytest.mark.asyncio
    async def test_not_web_searchable_routes_research_even_when_enabled(self) -> None:
        # Enabled, but the question needs a full investigation, not a quick lookup.
        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=_classification_response(
                answerable=False,
                follow_up_type="new_topic",
                web_searchable=False,
            )
        )
        decision = await decide_turn_intent(
            query="now do a deep comparison of five new entities",
            conversation_history=[{"role": "assistant", "content": "report text"}],
            prior_findings_summary="findings",
            has_prior_research=True,
            requested=TurnIntent.AUTO,
            llm=llm,
            allow_live_search=True,
        )
        assert decision.route == "research"

    @pytest.mark.asyncio
    async def test_answerable_from_pool_unaffected_by_live_search_flag(self) -> None:
        # In-pool answers still route to chat — never live_search — regardless.
        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=_classification_response(
                answerable=True,
                follow_up_type="complex_follow_up",
                web_searchable=True,  # even if model also flags this
            )
        )
        decision = await decide_turn_intent(
            query="who is the head of the org we researched?",
            conversation_history=[{"role": "assistant", "content": "report text"}],
            prior_findings_summary="findings naming the head",
            has_prior_research=True,
            requested=TurnIntent.AUTO,
            llm=llm,
            allow_live_search=True,
        )
        assert decision.route == "chat"

    @pytest.mark.asyncio
    async def test_explicit_research_override_skips_live_search(self) -> None:
        llm = AsyncMock()
        decision = await decide_turn_intent(
            query="anything",
            conversation_history=[],
            prior_findings_summary="prior stuff",
            has_prior_research=True,
            requested=TurnIntent.RESEARCH,
            llm=llm,
            allow_live_search=True,
        )
        assert decision.route == "research"
        llm.complete.assert_not_called()
