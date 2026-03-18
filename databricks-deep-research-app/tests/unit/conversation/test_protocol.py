"""Unit tests for ConversationProvider protocol."""

from typing import Any

import pytest

from deep_research.conversation.handler import ConversationHandler, DefaultConversationHandler
from deep_research.conversation.intent import DefaultIntentClassifier, IntentClassifier
from deep_research.conversation.protocol import ConversationProvider, DefaultConversationProvider


class TestConversationProviderProtocol:
    """Tests for ConversationProvider protocol."""

    def test_default_provider_implements_protocol(self) -> None:
        """DefaultConversationProvider should implement protocol."""
        provider = DefaultConversationProvider()
        assert isinstance(provider, ConversationProvider)

    def test_custom_provider_implements_protocol(self) -> None:
        """Custom implementation should be recognized."""

        class CustomProvider:
            def get_intent_classifier(self) -> IntentClassifier | None:
                return DefaultIntentClassifier()

            def get_conversation_handler(self) -> ConversationHandler | None:
                return DefaultConversationHandler()

            def get_conversation_config(self) -> dict[str, Any]:
                return {"enable_follow_ups": True}

        provider = CustomProvider()
        assert isinstance(provider, ConversationProvider)

    def test_incomplete_provider_not_recognized(self) -> None:
        """Incomplete implementation should not match protocol."""

        class IncompleteProvider:
            def get_intent_classifier(self) -> IntentClassifier | None:
                return None

            # Missing other required methods

        provider = IncompleteProvider()
        assert not isinstance(provider, ConversationProvider)


class TestDefaultConversationProvider:
    """Tests for DefaultConversationProvider."""

    def test_get_intent_classifier_returns_none(self) -> None:
        """Should return None for default classifier."""
        provider = DefaultConversationProvider()
        assert provider.get_intent_classifier() is None

    def test_get_conversation_handler_returns_none(self) -> None:
        """Should return None for default handler."""
        provider = DefaultConversationProvider()
        assert provider.get_conversation_handler() is None

    def test_get_conversation_config(self) -> None:
        """Should return default config."""
        provider = DefaultConversationProvider()
        config = provider.get_conversation_config()

        assert config["enable_follow_ups"] is True
        assert config["max_history_turns"] == 10
        assert config["context_window_messages"] == 4


class TestCustomConversationProvider:
    """Tests for custom conversation provider implementations."""

    def test_sales_domain_provider(self) -> None:
        """Test a complete custom provider implementation."""

        class SalesConversationProvider:
            def get_intent_classifier(self) -> IntentClassifier | None:
                # Return custom classifier for sales domain
                return None  # Would return SalesIntentClassifier

            def get_conversation_handler(self) -> ConversationHandler | None:
                # Return custom handler for sales conversations
                return None  # Would return SalesConversationHandler

            def get_conversation_config(self) -> dict[str, Any]:
                return {
                    "enable_follow_ups": True,
                    "max_history_turns": 20,  # Longer sales conversations
                    "context_window_messages": 6,
                    "chitchat_responses": {
                        "hello": "Hello! How can I help you learn about our products today?",
                        "thanks": "Thank you for considering us! Is there anything else I can help with?",
                    },
                    "domain_keywords": ["price", "discount", "features", "demo"],
                }

        provider = SalesConversationProvider()
        assert isinstance(provider, ConversationProvider)

        config = provider.get_conversation_config()
        assert config["max_history_turns"] == 20
        assert "chitchat_responses" in config
        assert "domain_keywords" in config

    def test_minimal_provider(self) -> None:
        """Test minimal custom provider that uses all defaults."""

        class MinimalProvider:
            def get_intent_classifier(self) -> IntentClassifier | None:
                return None

            def get_conversation_handler(self) -> ConversationHandler | None:
                return None

            def get_conversation_config(self) -> dict[str, Any]:
                return {}

        provider = MinimalProvider()
        assert isinstance(provider, ConversationProvider)
        assert provider.get_intent_classifier() is None
        assert provider.get_conversation_handler() is None
        assert provider.get_conversation_config() == {}
