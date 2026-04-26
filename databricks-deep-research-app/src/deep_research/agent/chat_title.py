"""Canonical chat-title derivation from user query/message text.

Single source of truth for every site that writes chat.title from user input.
Leaf module — no imports from deep_research.* to keep services/ -> agent/
edges circular-import-safe.
"""

from __future__ import annotations

_MAX_RAW_LENGTH = 50
_TRUNCATE_LENGTH = 47


def derive_chat_title_from_query(text: str | None) -> str:
    """Trim; empty -> ""; <=50 chars raw; >50 chars -> first 47 + "...""."""
    if not text:
        return ""
    stripped = text.strip()
    if not stripped:
        return ""
    if len(stripped) > _MAX_RAW_LENGTH:
        return stripped[:_TRUNCATE_LENGTH] + "..."
    return stripped
