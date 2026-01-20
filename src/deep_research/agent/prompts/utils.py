"""Prompt utility functions."""


def build_system_prompt(
    base_prompt: str,
    system_instructions: str | None = None,
) -> str:
    """Build a system prompt with optional user instructions.

    Args:
        base_prompt: The base system prompt for the agent.
        system_instructions: Optional user-defined instructions to include.

    Returns:
        Complete system prompt with user instructions appended if provided.
    """
    if not system_instructions:
        return base_prompt

    return f"""{base_prompt}

## User-Defined Instructions (IMPORTANT)

The user has provided the following custom instructions that MUST be followed:

{system_instructions}

These user instructions take precedence over default behaviors when applicable."""
