"""
Intent Classification
=====================

Provides intent classification for follow-up conversations.

This module defines:
- IntentType: Enum of possible user intents
- ConversationIntent: Result of intent classification
- IntentClassifier: Protocol for classifiers
- DefaultIntentClassifier: Rule-based default implementation
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class IntentType(str, Enum):
    """Types of user intents in a conversation.

    Attributes:
        NEW_RESEARCH: Start a completely new research topic
        FOLLOW_UP: Follow-up question about previous research
        CLARIFICATION: Request clarification of previous answer
        REFINEMENT: Refine or narrow down previous results
        EXPANSION: Expand on a specific aspect
        COMPARISON: Compare multiple topics
        CHITCHAT: General conversation not requiring research
        FEEDBACK: User providing feedback
        UNKNOWN: Could not determine intent
    """

    NEW_RESEARCH = "new_research"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"
    REFINEMENT = "refinement"
    EXPANSION = "expansion"
    COMPARISON = "comparison"
    CHITCHAT = "chitchat"
    FEEDBACK = "feedback"
    UNKNOWN = "unknown"


@dataclass
class ConversationIntent:
    """Result of intent classification.

    Attributes:
        intent_type: Classified intent type
        confidence: Confidence score (0.0 to 1.0)
        requires_new_research: Whether this needs a full research run
        can_use_cached_results: Whether cached results can be reused
        suggested_context: Relevant context to include
        metadata: Additional classification metadata
    """

    intent_type: IntentType
    confidence: float = 1.0
    requires_new_research: bool = True
    can_use_cached_results: bool = False
    suggested_context: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new_research(cls, confidence: float = 1.0) -> "ConversationIntent":
        """Create intent for new research."""
        return cls(
            intent_type=IntentType.NEW_RESEARCH,
            confidence=confidence,
            requires_new_research=True,
            can_use_cached_results=False,
        )

    @classmethod
    def follow_up(
        cls,
        confidence: float = 1.0,
        context: list[str] | None = None,
    ) -> "ConversationIntent":
        """Create intent for follow-up question."""
        return cls(
            intent_type=IntentType.FOLLOW_UP,
            confidence=confidence,
            requires_new_research=True,
            can_use_cached_results=True,
            suggested_context=context or [],
        )

    @classmethod
    def clarification(
        cls,
        confidence: float = 1.0,
    ) -> "ConversationIntent":
        """Create intent for clarification request."""
        return cls(
            intent_type=IntentType.CLARIFICATION,
            confidence=confidence,
            requires_new_research=False,
            can_use_cached_results=True,
        )

    @classmethod
    def chitchat(cls, confidence: float = 1.0) -> "ConversationIntent":
        """Create intent for chitchat."""
        return cls(
            intent_type=IntentType.CHITCHAT,
            confidence=confidence,
            requires_new_research=False,
            can_use_cached_results=False,
        )


@runtime_checkable
class IntentClassifier(Protocol):
    """Protocol for intent classifiers.

    Implement this protocol to provide custom intent classification
    logic for your domain.
    """

    async def classify(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        previous_state: Any | None = None,
    ) -> ConversationIntent:
        """Classify the intent of a user message.

        Args:
            message: Current user message
            history: Conversation history (list of {"role": ..., "content": ...})
            previous_state: Previous research state if available

        Returns:
            ConversationIntent with classification result
        """
        ...


class DefaultIntentClassifier:
    """Default rule-based intent classifier.

    Uses keyword matching and heuristics to classify intents.
    For production use, consider implementing an LLM-based classifier.
    """

    # Keywords indicating follow-up
    FOLLOW_UP_KEYWORDS = {
        "more about",
        "tell me more",
        "what about",
        "how about",
        "also",
        "additionally",
        "furthermore",
        "regarding that",
        "on that topic",
        "related to",
    }

    # Keywords indicating clarification
    CLARIFICATION_KEYWORDS = {
        "what do you mean",
        "can you explain",
        "i don't understand",
        "clarify",
        "what does",
        "what is",
        "define",
        "meaning of",
    }

    # Keywords indicating refinement
    REFINEMENT_KEYWORDS = {
        "narrow down",
        "focus on",
        "specifically",
        "only the",
        "just the",
        "filter",
        "limit to",
    }

    # Keywords indicating expansion
    EXPANSION_KEYWORDS = {
        "expand on",
        "more details",
        "elaborate",
        "go deeper",
        "dive into",
        "explore further",
    }

    # Keywords indicating comparison
    COMPARISON_KEYWORDS = {
        "compare",
        "versus",
        "vs",
        "difference between",
        "better",
        "worse",
        "prefer",
    }

    # Chitchat patterns
    CHITCHAT_PATTERNS = {
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "good morning",
        "good afternoon",
        "how are you",
    }

    async def classify(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        previous_state: Any | None = None,
    ) -> ConversationIntent:
        """Classify intent using rule-based heuristics."""
        message_lower = message.lower().strip()

        # Check for chitchat
        for pattern in self.CHITCHAT_PATTERNS:
            if message_lower.startswith(pattern) and len(message_lower) < 50:
                return ConversationIntent.chitchat(confidence=0.9)

        # Check for clarification
        for keyword in self.CLARIFICATION_KEYWORDS:
            if keyword in message_lower:
                return ConversationIntent(
                    intent_type=IntentType.CLARIFICATION,
                    confidence=0.8,
                    requires_new_research=False,
                    can_use_cached_results=True,
                )

        # Check for refinement
        for keyword in self.REFINEMENT_KEYWORDS:
            if keyword in message_lower:
                return ConversationIntent(
                    intent_type=IntentType.REFINEMENT,
                    confidence=0.8,
                    requires_new_research=True,
                    can_use_cached_results=True,
                )

        # Check for expansion
        for keyword in self.EXPANSION_KEYWORDS:
            if keyword in message_lower:
                return ConversationIntent(
                    intent_type=IntentType.EXPANSION,
                    confidence=0.8,
                    requires_new_research=True,
                    can_use_cached_results=True,
                )

        # Check for comparison
        for keyword in self.COMPARISON_KEYWORDS:
            if keyword in message_lower:
                return ConversationIntent(
                    intent_type=IntentType.COMPARISON,
                    confidence=0.8,
                    requires_new_research=True,
                    can_use_cached_results=False,
                )

        # Check for follow-up (requires history)
        if history and len(history) > 0:
            for keyword in self.FOLLOW_UP_KEYWORDS:
                if keyword in message_lower:
                    return ConversationIntent.follow_up(confidence=0.8)

            # Short messages with history are likely follow-ups
            if len(message_lower) < 100 and not message_lower.endswith("?"):
                return ConversationIntent.follow_up(confidence=0.6)

        # Default to new research
        return ConversationIntent.new_research(confidence=0.7)
