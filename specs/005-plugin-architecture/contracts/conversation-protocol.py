"""
Conversation Protocol Definitions
=================================

This file defines the conversation handling protocols for extending
the Deep Research Agent's follow-up message processing.

Plugins can:
- Provide custom intent classification
- Implement read-only QA handlers
- Implement research/update handlers for plan modifications

Location: src/deep_research/plugins/conversation.py
"""

from typing import Protocol, Any, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Intent Classification
# ---------------------------------------------------------------------------

class FollowUpIntent(str, Enum):
    """Types of follow-up intents."""

    QA = "qa"
    """Read-only question about existing content."""

    RESEARCH_UPDATE = "research_update"
    """Request to research more or modify the output."""

    CLARIFICATION_NEEDED = "clarification_needed"
    """Intent unclear, need to ask user for clarification."""

    EXPORT = "export"
    """Request to export content in a specific format."""

    FEEDBACK = "feedback"
    """User providing feedback (positive/negative)."""


@dataclass
class IntentClassification:
    """
    Result of intent classification.

    Contains the detected intent, confidence score, and any
    extracted parameters relevant to the intent.
    """

    intent: FollowUpIntent
    """Detected intent type."""

    confidence: float
    """Confidence score (0.0 to 1.0)."""

    parameters: dict[str, Any] = field(default_factory=dict)
    """
    Extracted parameters based on intent. Examples:

    QA intent:
        {"topic": "competitive positioning", "section": "key_insights"}

    RESEARCH_UPDATE intent:
        {"action": "add_case_study", "industry": "healthcare"}

    CLARIFICATION_NEEDED intent:
        {"ambiguous_element": "target", "question_text": "..."}
    """

    def is_confident(self, threshold: float = 0.5) -> bool:
        """Check if classification meets confidence threshold."""
        return self.confidence >= threshold


# ---------------------------------------------------------------------------
# Response Types
# ---------------------------------------------------------------------------

@dataclass
class FollowUpResponse:
    """Base class for follow-up responses."""

    intent: FollowUpIntent
    """Intent that was handled."""


@dataclass
class AnswerResponse(FollowUpResponse):
    """Response for QA intent - read-only answer."""

    intent: FollowUpIntent = FollowUpIntent.QA

    answer: str
    """The answer text."""

    source_refs: list[str] = field(default_factory=list)
    """Source IDs supporting the answer."""


@dataclass
class UpdateResponse(FollowUpResponse):
    """Response for RESEARCH_UPDATE intent - modified output."""

    intent: FollowUpIntent = FollowUpIntent.RESEARCH_UPDATE

    updated_output: Any
    """The updated output (same type as original)."""

    changes_summary: str
    """Summary of what was changed."""

    tool_calls_used: int = 0
    """Number of tool calls used for the update."""


@dataclass
class ClarificationResponse(FollowUpResponse):
    """Response for CLARIFICATION_NEEDED - asking user for more info."""

    intent: FollowUpIntent = FollowUpIntent.CLARIFICATION_NEEDED

    question: str
    """Clarification question to ask the user."""

    options: list[str] = field(default_factory=list)
    """Optional list of choices for the user."""


# ---------------------------------------------------------------------------
# Handler Protocols
# ---------------------------------------------------------------------------

class IntentClassifier(Protocol):
    """
    Protocol for intent classification.

    Analyzes a follow-up message and determines the user's intent.
    """

    async def classify(
        self,
        message: str,
        current_state: Any,
        recent_context: str | None = None,
    ) -> IntentClassification:
        """
        Classify the intent of a follow-up message.

        Args:
            message: User's follow-up message
            current_state: Current output/state to provide context
            recent_context: Summary of recent conversation turns

        Returns:
            IntentClassification with intent, confidence, and parameters

        Example implementation:
            async def classify(self, message, current_state, recent_context):
                # Use LLM to classify intent
                prompt = INTENT_CLASSIFICATION_PROMPT.format(
                    message=message,
                    state_summary=summarize(current_state),
                    recent=recent_context or "None",
                )
                response = await self.llm.chat([...])
                return parse_classification(response)
        """
        ...


class QAHandler(Protocol):
    """
    Protocol for read-only question answering.

    Answers questions about existing content without modifications.
    """

    async def handle(
        self,
        message: str,
        classification: IntentClassification,
        current_state: Any,
        sources: list[dict[str, Any]],
        recent_context: str | None = None,
    ) -> AnswerResponse:
        """
        Handle a QA (read-only) follow-up message.

        Args:
            message: User's question
            classification: Intent classification result
            current_state: Current output to answer from
            sources: Available sources for citations
            recent_context: Recent conversation context

        Returns:
            AnswerResponse with answer and source references
        """
        ...


class UpdateHandler(Protocol):
    """
    Protocol for research/update operations.

    Can perform additional research and modify the output.
    """

    MAX_TOOL_CALLS: int = 5
    """Maximum tool calls allowed per update."""

    async def handle(
        self,
        message: str,
        classification: IntentClassification,
        current_state: Any,
        sources: list[dict[str, Any]],
        tools: list[Any],
        recent_context: str | None = None,
    ) -> AsyncGenerator["StreamEvent", None]:
        """
        Handle a research/update follow-up message.

        Args:
            message: User's request
            classification: Intent classification result
            current_state: Current output to modify
            sources: Available sources
            tools: Available tools for research
            recent_context: Recent conversation context

        Yields:
            StreamEvent instances for progress updates

        Returns:
            UpdateResponse with modified output
        """
        ...


# ---------------------------------------------------------------------------
# Conversation Provider Protocol
# ---------------------------------------------------------------------------

class ConversationProvider(Protocol):
    """
    Protocol for plugins that customize conversation handling.

    Implement this to provide custom intent classification and
    follow-up handlers for your domain.
    """

    def get_intent_classifier(self) -> IntentClassifier | None:
        """
        Return custom intent classifier.

        If None, the default classifier is used.

        Returns:
            Custom IntentClassifier, or None for default
        """
        ...

    def get_qa_handler(self) -> QAHandler | None:
        """
        Return custom QA handler for read-only questions.

        If None, the default QA handler is used.

        Returns:
            Custom QAHandler, or None for default
        """
        ...

    def get_update_handler(self) -> UpdateHandler | None:
        """
        Return custom handler for research/update requests.

        If None, the default update handler is used.

        Returns:
            Custom UpdateHandler, or None for default
        """
        ...


# ---------------------------------------------------------------------------
# Default Handlers (Reference)
# ---------------------------------------------------------------------------

"""
Default handlers are implemented in:
- src/deep_research/conversation/default.py

class DefaultIntentClassifier:
    '''LLM-based intent classifier.'''

    CONFIDENCE_THRESHOLD = 0.5

    async def classify(self, message, current_state, recent_context):
        # Use LLM to classify intent into QA or RESEARCH_UPDATE
        # If confidence < 0.5, return CLARIFICATION_NEEDED
        ...

class DefaultQAHandler:
    '''Read-only QA using LLM to answer from context.'''

    async def handle(self, message, classification, current_state, sources, recent_context):
        # Build context from current_state
        # Use LLM to generate answer with source citations
        ...

class DefaultUpdateHandler:
    '''Research/update handler using existing pipeline.'''

    MAX_TOOL_CALLS = 5

    async def handle(self, message, classification, current_state, sources, tools, recent_context):
        # Parse the update request
        # Execute limited research (max 5 tool calls)
        # Merge results into current_state
        # Yield progress events
        ...
"""


# ---------------------------------------------------------------------------
# Example: sapresalesbot Conversation Provider
# ---------------------------------------------------------------------------

"""
class SapresalesbotConversationProvider:
    '''Custom conversation handling for sales meeting prep.'''

    def __init__(self, llm_service):
        self.llm = llm_service

    def get_intent_classifier(self) -> IntentClassifier:
        return SalesIntentClassifier(self.llm)

    def get_qa_handler(self) -> QAHandler:
        return SalesQAAgent(self.llm)

    def get_update_handler(self) -> UpdateHandler:
        return SalesUpdateAgent(self.llm)


class SalesIntentClassifier:
    '''Sales-specific intent classification.'''

    CONFIDENCE_THRESHOLD = 0.5

    async def classify(self, message, current_state, recent_context):
        # Sales-specific classification logic
        # Recognizes intents like:
        # - "Tell me more about the competitor" → QA
        # - "Add a case study for healthcare" → RESEARCH_UPDATE
        # - "What questions should I ask?" → QA
        # - "Research the VP of Engineering" → RESEARCH_UPDATE
        ...


class SalesQAAgent:
    '''Read-only QA for sales meeting prep.'''

    async def handle(self, message, classification, current_state, sources, recent_context):
        # Answer questions about:
        # - Meeting plan and POV
        # - Discovery questions
        # - Attendee backgrounds
        # - Competitive positioning
        # Without modifying the plan
        ...


class SalesUpdateAgent:
    '''Research/update agent for sales meeting prep.'''

    MAX_TOOL_CALLS = 5

    async def handle(self, message, classification, current_state, sources, tools, recent_context):
        # Handle requests like:
        # - Add/remove attendees
        # - Research additional topics
        # - Add case studies
        # - Update competitive positioning
        # Limited to 5 tool calls per update
        ...
"""
