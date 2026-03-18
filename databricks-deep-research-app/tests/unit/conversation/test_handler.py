"""Unit tests for conversation handler."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from deep_research.conversation.handler import (
    ConversationHandler,
    ConversationResponse,
    DefaultConversationHandler,
)
from deep_research.conversation.intent import ConversationIntent, IntentType


class TestConversationResponse:
    """Tests for ConversationResponse dataclass."""

    def test_create_minimal(self) -> None:
        """Should create with defaults."""
        response = ConversationResponse()
        assert response.response_type == "direct"
        assert response.content is None
        assert response.should_run_research is False
        assert response.research_query is None
        assert response.context_to_include == []
        assert response.metadata == {}

    def test_factory_direct(self) -> None:
        """Should create direct response."""
        response = ConversationResponse.direct("Hello!")
        assert response.response_type == "direct"
        assert response.content == "Hello!"
        assert response.should_run_research is False

    def test_factory_research(self) -> None:
        """Should create research response."""
        response = ConversationResponse.research(
            query="What is AI?",
            context=[{"type": "previous", "content": "..."}],
        )
        assert response.response_type == "research"
        assert response.should_run_research is True
        assert response.research_query == "What is AI?"
        assert len(response.context_to_include) == 1

    def test_factory_redirect(self) -> None:
        """Should create redirect response."""
        response = ConversationResponse.redirect({"target": "other_handler"})
        assert response.response_type == "redirect"
        assert response.should_run_research is False
        assert response.metadata["target"] == "other_handler"


class TestConversationHandlerProtocol:
    """Tests for ConversationHandler protocol."""

    def test_default_handler_implements_protocol(self) -> None:
        """DefaultConversationHandler should implement protocol."""
        handler = DefaultConversationHandler()
        assert isinstance(handler, ConversationHandler)

    def test_custom_handler_implements_protocol(self) -> None:
        """Custom implementation should be recognized."""

        class CustomHandler:
            async def handle(
                self,
                intent: ConversationIntent,
                message: str,
                history: list[dict[str, str]] | None = None,
                previous_state=None,
            ) -> ConversationResponse:
                return ConversationResponse.direct("Custom response")

        handler = CustomHandler()
        assert isinstance(handler, ConversationHandler)


class TestDefaultConversationHandler:
    """Tests for DefaultConversationHandler."""

    @pytest.mark.asyncio
    async def test_handles_chitchat_hello(self) -> None:
        """Should handle hello chitchat."""
        handler = DefaultConversationHandler()
        intent = ConversationIntent.chitchat()

        response = await handler.handle(intent, "hello")

        assert response.response_type == "direct"
        assert response.content is not None
        assert "help" in response.content.lower() or "research" in response.content.lower()
        assert response.should_run_research is False

    @pytest.mark.asyncio
    async def test_handles_chitchat_thanks(self) -> None:
        """Should handle thanks chitchat."""
        handler = DefaultConversationHandler()
        intent = ConversationIntent.chitchat()

        response = await handler.handle(intent, "thanks")

        assert response.response_type == "direct"
        assert response.content is not None
        assert "welcome" in response.content.lower()

    @pytest.mark.asyncio
    async def test_handles_clarification_no_state(self) -> None:
        """Should handle clarification without previous state."""
        handler = DefaultConversationHandler()
        intent = ConversationIntent(intent_type=IntentType.CLARIFICATION)

        response = await handler.handle(intent, "What do you mean?")

        assert response.response_type == "direct"
        assert "context" in response.content.lower() or "provide" in response.content.lower()

    @pytest.mark.asyncio
    async def test_handles_clarification_with_state(self) -> None:
        """Should handle clarification with previous state."""
        handler = DefaultConversationHandler()
        intent = ConversationIntent(intent_type=IntentType.CLARIFICATION)

        @dataclass
        class MockState:
            findings: str = "Previous findings..."

        response = await handler.handle(
            intent,
            "What do you mean by optimization?",
            history=[{"role": "assistant", "content": "Here's the analysis..."}],
            previous_state=MockState(),
        )

        assert response.response_type == "research"
        assert response.should_run_research is True
        assert "clarify" in response.research_query.lower()

    @pytest.mark.asyncio
    async def test_handles_follow_up(self) -> None:
        """Should handle follow-up questions."""
        handler = DefaultConversationHandler()
        intent = ConversationIntent.follow_up()

        history = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence..."},
        ]

        response = await handler.handle(intent, "Tell me more about deep learning", history)

        assert response.response_type == "research"
        assert response.should_run_research is True
        assert len(response.context_to_include) > 0

    @pytest.mark.asyncio
    async def test_handles_refinement(self) -> None:
        """Should handle refinement requests."""
        handler = DefaultConversationHandler()
        intent = ConversationIntent(intent_type=IntentType.REFINEMENT)

        history = [{"role": "user", "content": "Original query about trends"}]

        response = await handler.handle(intent, "Focus only on 2023", history)

        assert response.response_type == "research"
        assert response.should_run_research is True
        assert "focus" in response.research_query.lower()

    @pytest.mark.asyncio
    async def test_handles_expansion(self) -> None:
        """Should handle expansion requests."""
        handler = DefaultConversationHandler()
        intent = ConversationIntent(intent_type=IntentType.EXPANSION)

        @dataclass
        class MockState:
            findings: str = "Initial findings..."

        response = await handler.handle(
            intent,
            "Go deeper into the security aspects",
            previous_state=MockState(),
        )

        assert response.response_type == "research"
        assert response.should_run_research is True
        assert "detailed" in response.research_query.lower()

    @pytest.mark.asyncio
    async def test_handles_comparison(self) -> None:
        """Should handle comparison requests."""
        handler = DefaultConversationHandler()
        intent = ConversationIntent(intent_type=IntentType.COMPARISON)

        response = await handler.handle(intent, "Compare Python vs JavaScript")

        assert response.response_type == "research"
        assert response.should_run_research is True
        assert response.research_query == "Compare Python vs JavaScript"

    @pytest.mark.asyncio
    async def test_handles_feedback(self) -> None:
        """Should handle feedback."""
        handler = DefaultConversationHandler()
        intent = ConversationIntent(intent_type=IntentType.FEEDBACK)

        response = await handler.handle(intent, "This was very helpful!")

        assert response.response_type == "direct"
        assert "thank" in response.content.lower()
        assert response.should_run_research is False

    @pytest.mark.asyncio
    async def test_handles_new_research(self) -> None:
        """Should handle new research requests."""
        handler = DefaultConversationHandler()
        intent = ConversationIntent.new_research()

        response = await handler.handle(
            intent,
            "What are the latest trends in quantum computing?"
        )

        assert response.response_type == "research"
        assert response.should_run_research is True
        assert response.research_query == "What are the latest trends in quantum computing?"

    @pytest.mark.asyncio
    async def test_handles_unknown_intent(self) -> None:
        """Should default to research for unknown intent."""
        handler = DefaultConversationHandler()
        intent = ConversationIntent(intent_type=IntentType.UNKNOWN)

        response = await handler.handle(intent, "Some ambiguous message")

        assert response.response_type == "research"
        assert response.should_run_research is True
