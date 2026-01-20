"""Conversation history utilities for LLM context building.

This module provides utilities for building LLM messages with full
conversation history, enabling follow-up queries to reference
previous research reports in their entirety.
"""

from __future__ import annotations


def build_messages_with_history(
    system_prompt: str,
    user_query: str,
    history: list[dict[str, str]] | None = None,
    max_history_messages: int = 10,
) -> list[dict[str, str]]:
    """Build LLM messages with FULL conversation history.

    For follow-ups to a 13K+ word report, the full report IS included
    in history. We don't truncate content - the LLM context window
    handles any necessary truncation.

    Args:
        system_prompt: System prompt for the conversation.
        user_query: Current user query.
        history: Previous messages in the conversation.
            Each message should have 'role' and 'content' keys.
        max_history_messages: Maximum number of history messages to include.
            Default is 10 to balance context with token limits.

    Returns:
        List of message dicts suitable for LLM API calls.

    Example:
        >>> messages = build_messages_with_history(
        ...     system_prompt="You are a helpful assistant.",
        ...     user_query="Tell me more about PyTorch",
        ...     history=[
        ...         {"role": "user", "content": "Compare ML frameworks"},
        ...         {"role": "agent", "content": "Here's a 13K word report..."},
        ...     ],
        ... )
        >>> # Messages will include the full 13K report for context
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    if history:
        # Include recent history (up to max_history_messages)
        # Start from the most recent and work backwards
        recent_history = history[-max_history_messages:]

        for msg in recent_history:
            # Normalize role: 'agent' -> 'assistant' for LLM API compatibility
            role = msg.get("role", "user")
            if role in ("agent", "assistant"):
                role = "assistant"
            elif role != "user":
                # Skip system messages or unknown roles in history
                continue

            content = msg.get("content", "")
            if content:
                messages.append({"role": role, "content": content})

    # Add current user query
    messages.append({"role": "user", "content": user_query})

    return messages


def build_context_summary(
    history: list[dict[str, str]] | None,
    max_summary_chars: int = 500,
) -> str:
    """Build a short summary of conversation context for prompts.

    Unlike build_messages_with_history which preserves full content,
    this function creates a brief summary for inclusion in prompts
    where full history would be too long.

    Args:
        history: Previous messages in the conversation.
        max_summary_chars: Maximum characters for the summary.

    Returns:
        Brief summary string, or empty string if no history.
    """
    if not history:
        return ""

    # Count messages by role
    user_count = sum(1 for m in history if m.get("role") == "user")
    agent_count = sum(1 for m in history if m.get("role") in ("agent", "assistant"))

    # Get first user query and latest agent response preview
    first_query = ""
    latest_response = ""

    for msg in history:
        if msg.get("role") == "user" and not first_query:
            first_query = msg.get("content", "")[:100]
        if msg.get("role") in ("agent", "assistant"):
            latest_response = msg.get("content", "")[:200]

    summary_parts = [
        f"Previous conversation ({user_count} user, {agent_count} assistant messages):",
    ]

    if first_query:
        summary_parts.append(f"Started with: \"{first_query}...\"")

    if latest_response:
        summary_parts.append(f"Latest response preview: \"{latest_response}...\"")

    summary = "\n".join(summary_parts)

    # Truncate if needed
    if len(summary) > max_summary_chars:
        summary = summary[:max_summary_chars] + "..."

    return summary
