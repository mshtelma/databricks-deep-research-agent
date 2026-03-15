"""
Conversation Handler
====================

Provides handlers for different conversation intents.

This module defines:
- ConversationHandler: Protocol for handlers
- DefaultConversationHandler: Default implementation
- ConversationResponse: Result of handling
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from deep_research.conversation.intent import ConversationIntent, IntentType


@dataclass
class ConversationResponse:
    """Result of handling a conversation turn.

    Attributes:
        response_type: Type of response (direct, research, redirect)
        content: Response content (if direct response)
        should_run_research: Whether to trigger research pipeline
        research_query: Modified query for research (if different from original)
        context_to_include: Context to include in research
        metadata: Additional response metadata
    """

    response_type: str = "direct"  # direct, research, redirect
    content: str | None = None
    should_run_research: bool = False
    research_query: str | None = None
    context_to_include: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def direct(cls, content: str) -> "ConversationResponse":
        """Create a direct response (no research needed)."""
        return cls(
            response_type="direct",
            content=content,
            should_run_research=False,
        )

    @classmethod
    def research(
        cls,
        query: str | None = None,
        context: list[dict[str, Any]] | None = None,
    ) -> "ConversationResponse":
        """Create a response that triggers research."""
        return cls(
            response_type="research",
            should_run_research=True,
            research_query=query,
            context_to_include=context or [],
        )

    @classmethod
    def redirect(cls, metadata: dict[str, Any]) -> "ConversationResponse":
        """Create a redirect response (handled elsewhere)."""
        return cls(
            response_type="redirect",
            should_run_research=False,
            metadata=metadata,
        )


@runtime_checkable
class ConversationHandler(Protocol):
    """Protocol for conversation handlers.

    Implement this protocol to provide custom conversation handling
    for your domain.
    """

    async def handle(
        self,
        intent: ConversationIntent,
        message: str,
        history: list[dict[str, str]] | None = None,
        previous_state: Any | None = None,
    ) -> ConversationResponse:
        """Handle a conversation turn.

        Args:
            intent: Classified intent
            message: Current user message
            history: Conversation history
            previous_state: Previous research state if available

        Returns:
            ConversationResponse with handling result
        """
        ...


class DefaultConversationHandler:
    """Default conversation handler.

    Routes intents to appropriate handling strategies.
    """

    # Chitchat responses
    CHITCHAT_RESPONSES = {
        "hello": "Hello! How can I help you with your research today?",
        "hi": "Hi there! What would you like to research?",
        "hey": "Hey! Ready to help with your research questions.",
        "thanks": "You're welcome! Let me know if you have more questions.",
        "thank you": "You're welcome! Happy to help with any other research.",
        "bye": "Goodbye! Feel free to return anytime for more research.",
        "goodbye": "Goodbye! Have a great day.",
        "good morning": "Good morning! What would you like to research today?",
        "good afternoon": "Good afternoon! How can I help with your research?",
        "how are you": "I'm doing well, thank you! I'm ready to help with your research needs.",
    }

    async def handle(
        self,
        intent: ConversationIntent,
        message: str,
        history: list[dict[str, str]] | None = None,
        previous_state: Any | None = None,
    ) -> ConversationResponse:
        """Handle a conversation turn based on intent."""

        if intent.intent_type == IntentType.CHITCHAT:
            return self._handle_chitchat(message)

        elif intent.intent_type == IntentType.CLARIFICATION:
            return self._handle_clarification(message, history, previous_state)

        elif intent.intent_type == IntentType.FOLLOW_UP:
            return self._handle_follow_up(message, history, previous_state)

        elif intent.intent_type == IntentType.REFINEMENT:
            return self._handle_refinement(message, history, previous_state)

        elif intent.intent_type == IntentType.EXPANSION:
            return self._handle_expansion(message, history, previous_state)

        elif intent.intent_type == IntentType.COMPARISON:
            return self._handle_comparison(message)

        elif intent.intent_type == IntentType.FEEDBACK:
            return self._handle_feedback(message)

        # Default: new research
        return ConversationResponse.research(query=message)

    def _handle_chitchat(self, message: str) -> ConversationResponse:
        """Handle chitchat intents."""
        message_lower = message.lower().strip()

        # Find matching response
        for pattern, response in self.CHITCHAT_RESPONSES.items():
            if message_lower.startswith(pattern):
                return ConversationResponse.direct(response)

        # Default chitchat response
        return ConversationResponse.direct(
            "I'm a research assistant. What topic would you like me to research for you?"
        )

    def _handle_clarification(
        self,
        message: str,
        history: list[dict[str, str]] | None,
        previous_state: Any | None,
    ) -> ConversationResponse:
        """Handle clarification requests."""
        if not previous_state:
            return ConversationResponse.direct(
                "I don't have any previous context to clarify. "
                "Could you provide more details about what you'd like to know?"
            )

        # Get the last assistant response to provide context
        context = []
        if history:
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    context.append({
                        "type": "previous_response",
                        "content": msg.get("content", "")[:2000],  # Limit context size
                    })
                    break

        return ConversationResponse.research(
            query=f"Clarify and explain in more detail: {message}",
            context=context,
        )

    def _handle_follow_up(
        self,
        message: str,
        history: list[dict[str, str]] | None,
        previous_state: Any | None,
    ) -> ConversationResponse:
        """Handle follow-up questions."""
        context = []

        # Include relevant history as context
        if history:
            for msg in history[-4:]:  # Last 4 messages
                context.append({
                    "type": "conversation_history",
                    "role": msg.get("role"),
                    "content": msg.get("content", "")[:1000],
                })

        # Include previous research state summary
        if previous_state and hasattr(previous_state, "findings"):
            context.append({
                "type": "previous_findings",
                "content": str(previous_state.findings)[:2000],
            })

        return ConversationResponse.research(
            query=message,
            context=context,
        )

    def _handle_refinement(
        self,
        message: str,
        history: list[dict[str, str]] | None,
        previous_state: Any | None,
    ) -> ConversationResponse:
        """Handle refinement requests."""
        context = []

        # Include original query for context
        if history:
            for msg in history:
                if msg.get("role") == "user":
                    context.append({
                        "type": "original_query",
                        "content": msg.get("content", ""),
                    })
                    break

        return ConversationResponse.research(
            query=f"Focus specifically on: {message}",
            context=context,
        )

    def _handle_expansion(
        self,
        message: str,
        history: list[dict[str, str]] | None,
        previous_state: Any | None,
    ) -> ConversationResponse:
        """Handle expansion requests."""
        context = []

        if previous_state and hasattr(previous_state, "findings"):
            context.append({
                "type": "previous_findings",
                "content": str(previous_state.findings)[:2000],
            })

        return ConversationResponse.research(
            query=f"Provide more detailed information about: {message}",
            context=context,
        )

    def _handle_comparison(self, message: str) -> ConversationResponse:
        """Handle comparison requests."""
        return ConversationResponse.research(
            query=message,
            context=[{"type": "task", "content": "comparison"}],
        )

    def _handle_feedback(self, message: str) -> ConversationResponse:
        """Handle feedback."""
        return ConversationResponse.direct(
            "Thank you for your feedback! I'll use it to improve future responses. "
            "Is there anything else you'd like to research?"
        )
