"""Unit tests for intent classification."""

import pytest

from deep_research.conversation.intent import (
    ConversationIntent,
    DefaultIntentClassifier,
    IntentClassifier,
    IntentType,
)


class TestIntentType:
    """Tests for IntentType enum."""

    def test_intent_type_values(self) -> None:
        """IntentType should have expected values."""
        assert IntentType.NEW_RESEARCH.value == "new_research"
        assert IntentType.FOLLOW_UP.value == "follow_up"
        assert IntentType.CLARIFICATION.value == "clarification"
        assert IntentType.REFINEMENT.value == "refinement"
        assert IntentType.EXPANSION.value == "expansion"
        assert IntentType.COMPARISON.value == "comparison"
        assert IntentType.CHITCHAT.value == "chitchat"
        assert IntentType.FEEDBACK.value == "feedback"
        assert IntentType.UNKNOWN.value == "unknown"


class TestConversationIntent:
    """Tests for ConversationIntent dataclass."""

    def test_create_minimal(self) -> None:
        """Should create with minimal args."""
        intent = ConversationIntent(intent_type=IntentType.NEW_RESEARCH)
        assert intent.intent_type == IntentType.NEW_RESEARCH
        assert intent.confidence == 1.0
        assert intent.requires_new_research is True
        assert intent.can_use_cached_results is False
        assert intent.suggested_context == []
        assert intent.metadata == {}

    def test_create_full(self) -> None:
        """Should create with all args."""
        intent = ConversationIntent(
            intent_type=IntentType.FOLLOW_UP,
            confidence=0.85,
            requires_new_research=True,
            can_use_cached_results=True,
            suggested_context=["context1"],
            metadata={"key": "value"},
        )
        assert intent.intent_type == IntentType.FOLLOW_UP
        assert intent.confidence == 0.85
        assert intent.requires_new_research is True
        assert intent.can_use_cached_results is True
        assert intent.suggested_context == ["context1"]
        assert intent.metadata == {"key": "value"}

    def test_factory_new_research(self) -> None:
        """Should create new research intent."""
        intent = ConversationIntent.new_research(confidence=0.9)
        assert intent.intent_type == IntentType.NEW_RESEARCH
        assert intent.confidence == 0.9
        assert intent.requires_new_research is True
        assert intent.can_use_cached_results is False

    def test_factory_follow_up(self) -> None:
        """Should create follow-up intent."""
        intent = ConversationIntent.follow_up(
            confidence=0.8,
            context=["prev_query", "prev_result"],
        )
        assert intent.intent_type == IntentType.FOLLOW_UP
        assert intent.confidence == 0.8
        assert intent.requires_new_research is True
        assert intent.can_use_cached_results is True
        assert intent.suggested_context == ["prev_query", "prev_result"]

    def test_factory_clarification(self) -> None:
        """Should create clarification intent."""
        intent = ConversationIntent.clarification(confidence=0.7)
        assert intent.intent_type == IntentType.CLARIFICATION
        assert intent.requires_new_research is False
        assert intent.can_use_cached_results is True

    def test_factory_chitchat(self) -> None:
        """Should create chitchat intent."""
        intent = ConversationIntent.chitchat(confidence=0.95)
        assert intent.intent_type == IntentType.CHITCHAT
        assert intent.requires_new_research is False
        assert intent.can_use_cached_results is False


class TestIntentClassifierProtocol:
    """Tests for IntentClassifier protocol."""

    def test_default_classifier_implements_protocol(self) -> None:
        """DefaultIntentClassifier should implement protocol."""
        classifier = DefaultIntentClassifier()
        assert isinstance(classifier, IntentClassifier)

    def test_custom_classifier_implements_protocol(self) -> None:
        """Custom implementation should be recognized."""

        class CustomClassifier:
            async def classify(
                self,
                message: str,
                history: list[dict[str, str]] | None = None,
                previous_state=None,
            ) -> ConversationIntent:
                return ConversationIntent.new_research()

        classifier = CustomClassifier()
        assert isinstance(classifier, IntentClassifier)


class TestDefaultIntentClassifier:
    """Tests for DefaultIntentClassifier."""

    @pytest.mark.asyncio
    async def test_classifies_chitchat_hello(self) -> None:
        """Should classify 'hello' as chitchat."""
        classifier = DefaultIntentClassifier()
        intent = await classifier.classify("Hello!")
        assert intent.intent_type == IntentType.CHITCHAT

    @pytest.mark.asyncio
    async def test_classifies_chitchat_thanks(self) -> None:
        """Should classify 'thanks' as chitchat."""
        classifier = DefaultIntentClassifier()
        intent = await classifier.classify("Thanks!")
        assert intent.intent_type == IntentType.CHITCHAT

    @pytest.mark.asyncio
    async def test_classifies_clarification(self) -> None:
        """Should classify clarification requests."""
        classifier = DefaultIntentClassifier()

        intent = await classifier.classify("What do you mean by that?")
        assert intent.intent_type == IntentType.CLARIFICATION

        intent = await classifier.classify("Can you explain the first point?")
        assert intent.intent_type == IntentType.CLARIFICATION

    @pytest.mark.asyncio
    async def test_classifies_refinement(self) -> None:
        """Should classify refinement requests."""
        classifier = DefaultIntentClassifier()

        intent = await classifier.classify("Narrow down to just the technical aspects")
        assert intent.intent_type == IntentType.REFINEMENT

        intent = await classifier.classify("Focus on the 2023 data only")
        assert intent.intent_type == IntentType.REFINEMENT

    @pytest.mark.asyncio
    async def test_classifies_expansion(self) -> None:
        """Should classify expansion requests."""
        classifier = DefaultIntentClassifier()

        intent = await classifier.classify("Expand on the security concerns")
        assert intent.intent_type == IntentType.EXPANSION

        intent = await classifier.classify("More details about pricing")
        assert intent.intent_type == IntentType.EXPANSION

    @pytest.mark.asyncio
    async def test_classifies_comparison(self) -> None:
        """Should classify comparison requests."""
        classifier = DefaultIntentClassifier()

        intent = await classifier.classify("Compare React vs Vue")
        assert intent.intent_type == IntentType.COMPARISON

        intent = await classifier.classify("What's the difference between A and B?")
        assert intent.intent_type == IntentType.COMPARISON

    @pytest.mark.asyncio
    async def test_classifies_follow_up_with_keywords(self) -> None:
        """Should classify follow-up with keywords."""
        classifier = DefaultIntentClassifier()
        history = [{"role": "assistant", "content": "Previous response..."}]

        intent = await classifier.classify("Tell me more about that", history)
        assert intent.intent_type == IntentType.FOLLOW_UP

        intent = await classifier.classify("What about the pricing?", history)
        assert intent.intent_type == IntentType.FOLLOW_UP

    @pytest.mark.asyncio
    async def test_defaults_to_new_research(self) -> None:
        """Should default to new research for novel queries."""
        classifier = DefaultIntentClassifier()

        intent = await classifier.classify(
            "What are the latest trends in artificial intelligence?"
        )
        assert intent.intent_type == IntentType.NEW_RESEARCH

    @pytest.mark.asyncio
    async def test_short_message_with_history_is_follow_up(self) -> None:
        """Short messages with history should be follow-ups."""
        classifier = DefaultIntentClassifier()
        history = [{"role": "assistant", "content": "Here's the analysis..."}]

        # Short statement without question mark
        intent = await classifier.classify("Yes, that makes sense", history)
        assert intent.intent_type == IntentType.FOLLOW_UP

    @pytest.mark.asyncio
    async def test_long_chitchat_treated_as_research(self) -> None:
        """Long 'hello' messages should not be chitchat."""
        classifier = DefaultIntentClassifier()

        # This is too long to be simple chitchat
        intent = await classifier.classify(
            "Hello, I'm interested in researching the impact of climate change on agriculture. "
            "Specifically, I want to understand how temperature variations affect crop yields."
        )
        assert intent.intent_type != IntentType.CHITCHAT
