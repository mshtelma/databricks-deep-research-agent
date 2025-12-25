"""Context truncation for LLM requests.

Handles reducing message context when it exceeds endpoint context windows
while preserving important content (system prompt, recent messages).
"""

from src.core.logging_utils import get_logger

logger = get_logger(__name__)


def estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough: ~4 chars per token).

    Args:
        text: Input text.

    Returns:
        Estimated token count.
    """
    return max(1, len(text) // 4)


def truncate_messages(
    messages: list[dict[str, str]],
    max_tokens: int,
    preserve_system: bool = True,
    min_recent_messages: int = 2,
) -> list[dict[str, str]]:
    """Truncate messages to fit within token limit.

    Strategy:
    1. Always preserve the system prompt (if present)
    2. Always preserve the most recent user message
    3. Remove older messages from the middle until under limit
    4. If still over limit, truncate older message content

    Args:
        messages: List of chat messages.
        max_tokens: Maximum tokens allowed.
        preserve_system: Whether to always preserve system prompt.
        min_recent_messages: Minimum recent messages to preserve.

    Returns:
        Truncated message list.
    """
    if not messages:
        return []

    # Calculate current token usage
    total_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages)

    if total_tokens <= max_tokens:
        return messages

    logger.info(
        "TRUNCATING_CONTEXT",
        current_tokens=total_tokens,
        max_tokens=max_tokens,
        message_count=len(messages),
    )

    # Separate system message and other messages
    system_message = None
    other_messages: list[dict[str, str]] = []

    for msg in messages:
        if preserve_system and msg.get("role") == "system":
            if system_message is None:
                system_message = msg
            else:
                # Multiple system messages - keep first, treat rest as other
                other_messages.append(msg)
        else:
            other_messages.append(msg)

    # Calculate system message tokens
    system_tokens = estimate_tokens(system_message.get("content", "")) if system_message else 0
    available_tokens = max_tokens - system_tokens

    if available_tokens <= 0:
        # System message alone exceeds limit - truncate it
        if system_message:
            truncated_content = truncate_text(
                system_message.get("content", ""),
                max_tokens - 100,  # Leave room for other messages
            )
            return [{"role": "system", "content": truncated_content}]
        return []

    # Preserve recent messages, remove older ones
    result_messages: list[dict[str, str]] = []
    recent_messages: list[dict[str, str]] = []

    # Separate recent messages from older ones
    if len(other_messages) > min_recent_messages:
        recent_messages = other_messages[-min_recent_messages:]
        older_messages = other_messages[:-min_recent_messages]
    else:
        recent_messages = other_messages
        older_messages = []

    # Calculate recent message tokens
    recent_tokens = sum(estimate_tokens(m.get("content", "")) for m in recent_messages)

    if recent_tokens <= available_tokens:
        # Room for some older messages
        remaining_tokens = available_tokens - recent_tokens

        # Add older messages from newest to oldest until budget exhausted
        kept_older: list[dict[str, str]] = []
        for msg in reversed(older_messages):
            msg_tokens = estimate_tokens(msg.get("content", ""))
            if msg_tokens <= remaining_tokens:
                kept_older.insert(0, msg)
                remaining_tokens -= msg_tokens
            else:
                # Truncate this message if it's the last one we can fit
                if remaining_tokens > 50:  # Only if there's meaningful room
                    truncated = truncate_text(msg.get("content", ""), remaining_tokens)
                    kept_older.insert(0, {"role": msg.get("role", "user"), "content": truncated})
                break

        result_messages = kept_older + recent_messages
    else:
        # Even recent messages exceed budget - truncate them
        result_messages = _truncate_recent_messages(recent_messages, available_tokens)

    # Prepend system message if present
    if system_message:
        result_messages.insert(0, system_message)

    final_tokens = sum(estimate_tokens(m.get("content", "")) for m in result_messages)
    logger.info(
        "TRUNCATION_COMPLETE",
        original_messages=len(messages),
        result_messages=len(result_messages),
        original_tokens=total_tokens,
        result_tokens=final_tokens,
    )

    return result_messages


def _truncate_recent_messages(
    messages: list[dict[str, str]],
    max_tokens: int,
) -> list[dict[str, str]]:
    """Truncate recent messages to fit within token budget.

    Prioritizes the most recent message, truncates older recent messages.

    Args:
        messages: Recent messages to truncate.
        max_tokens: Maximum tokens available.

    Returns:
        Truncated message list.
    """
    if not messages:
        return []

    result: list[dict[str, str]] = []
    remaining_tokens = max_tokens

    # Process from newest to oldest
    for msg in reversed(messages):
        msg_tokens = estimate_tokens(msg.get("content", ""))

        if msg_tokens <= remaining_tokens:
            result.insert(0, msg)
            remaining_tokens -= msg_tokens
        elif remaining_tokens > 50:
            # Truncate this message
            truncated = truncate_text(msg.get("content", ""), remaining_tokens)
            result.insert(0, {"role": msg.get("role", "user"), "content": truncated})
            remaining_tokens = 0
        # Else: skip this message

    return result


def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to approximately fit within token limit.

    Preserves the beginning of the text and adds truncation marker.

    Args:
        text: Text to truncate.
        max_tokens: Maximum tokens allowed.

    Returns:
        Truncated text with marker if truncated.
    """
    if estimate_tokens(text) <= max_tokens:
        return text

    # Estimate characters per token (roughly 4)
    max_chars = max_tokens * 4
    truncation_marker = "\n\n[Content truncated due to context length limits...]"

    if len(text) <= max_chars:
        return text

    # Leave room for truncation marker
    marker_chars = len(truncation_marker)
    available_chars = max_chars - marker_chars

    if available_chars <= 0:
        return truncation_marker.strip()

    # Truncate at a reasonable boundary (sentence or paragraph)
    truncated = text[:available_chars]

    # Try to find a good break point
    for sep in ["\n\n", ". ", ".\n", "\n", " "]:
        idx = truncated.rfind(sep)
        if idx > available_chars // 2:  # Only use if at least half the content
            truncated = truncated[:idx + len(sep)]
            break

    return truncated.rstrip() + truncation_marker


def get_context_window_for_request(
    messages: list[dict[str, str]],
    max_context_window: int,
    max_output_tokens: int,
) -> list[dict[str, str]]:
    """Prepare messages to fit within endpoint context window.

    Accounts for both input and expected output tokens.

    Args:
        messages: Input messages.
        max_context_window: Endpoint's maximum context window.
        max_output_tokens: Expected max output tokens.

    Returns:
        Truncated messages that fit within the context window.
    """
    # Leave room for output tokens plus safety margin
    safety_margin = 500  # Buffer for tokenization variance
    available_for_input = max_context_window - max_output_tokens - safety_margin

    if available_for_input <= 0:
        raise ValueError(
            f"Context window ({max_context_window}) too small for "
            f"output tokens ({max_output_tokens})"
        )

    return truncate_messages(messages, available_for_input)
