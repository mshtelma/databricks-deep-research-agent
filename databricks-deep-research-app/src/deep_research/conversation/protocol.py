"""
Conversation Provider Protocol
==============================

Protocol for plugins that provide custom conversation handling.
"""

from typing import Any, Protocol, runtime_checkable

from deep_research.conversation.handler import ConversationHandler
from deep_research.conversation.intent import IntentClassifier


@runtime_checkable
class ConversationProvider(Protocol):
    """Protocol for plugins that provide conversation handling.

    Implement this protocol to provide custom intent classification
    and conversation handling from a plugin.

    Example:
        >>> class MyDomainConversationProvider:
        ...     def get_intent_classifier(self) -> IntentClassifier | None:
        ...         return MyDomainIntentClassifier()
        ...
        ...     def get_conversation_handler(self) -> ConversationHandler | None:
        ...         return MyDomainConversationHandler()
        ...
        ...     def get_conversation_config(self) -> dict[str, Any]:
        ...         return {
        ...             "enable_follow_ups": True,
        ...             "max_history_turns": 5,
        ...             "context_window_messages": 4,
        ...         }
    """

    def get_intent_classifier(self) -> IntentClassifier | None:
        """Get custom intent classifier from this plugin.

        Returns:
            Custom IntentClassifier, or None to use default
        """
        ...

    def get_conversation_handler(self) -> ConversationHandler | None:
        """Get custom conversation handler from this plugin.

        Returns:
            Custom ConversationHandler, or None to use default
        """
        ...

    def get_conversation_config(self) -> dict[str, Any]:
        """Get conversation configuration from this plugin.

        Returns:
            Dictionary with conversation settings:
            - enable_follow_ups: Whether to support follow-up conversations
            - max_history_turns: Maximum conversation turns to remember
            - context_window_messages: Messages to include as context
            - chitchat_responses: Custom chitchat response overrides
        """
        ...


class DefaultConversationProvider:
    """Default conversation provider.

    Uses the default intent classifier and handler.
    """

    def get_intent_classifier(self) -> IntentClassifier | None:
        """Return None to use default classifier."""
        return None

    def get_conversation_handler(self) -> ConversationHandler | None:
        """Return None to use default handler."""
        return None

    def get_conversation_config(self) -> dict[str, Any]:
        """Return default conversation config."""
        return {
            "enable_follow_ups": True,
            "max_history_turns": 10,
            "context_window_messages": 4,
        }
