"""
Conversation Module
===================

Provides infrastructure for handling follow-up conversations
in research pipelines.

This module enables:
- Intent classification for user messages
- Conversation routing decisions
- Plugin-provided conversation handlers
- Default conversation handling patterns

Example usage:
    from deep_research.conversation import (
        ConversationProvider,
        IntentClassifier,
        ConversationHandler,
        ConversationIntent,
    )

    # Classify user intent
    classifier = IntentClassifier()
    intent = await classifier.classify(message, history)

    # Get appropriate handler
    handler = ConversationHandler()
    response = await handler.handle(intent, message, history, state)
"""

from deep_research.conversation.intent import (
    ConversationIntent,
    IntentType,
    IntentClassifier,
    DefaultIntentClassifier,
)
from deep_research.conversation.handler import (
    ConversationHandler,
    DefaultConversationHandler,
)
from deep_research.conversation.protocol import ConversationProvider

__all__ = [
    # Intent
    "ConversationIntent",
    "IntentType",
    "IntentClassifier",
    "DefaultIntentClassifier",
    # Handler
    "ConversationHandler",
    "DefaultConversationHandler",
    # Protocol
    "ConversationProvider",
]
