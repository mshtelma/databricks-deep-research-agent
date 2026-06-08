"""Shared visibility enum for agents.

Extracted from models/custom_agent.py (US-507) so that AgentV2 and any
future agent models can import it without pulling in the deleted CustomAgent
SQLAlchemy model.
"""

from enum import StrEnum


class AgentVisibility(StrEnum):
    """Visibility levels for agents."""

    PRIVATE = "private"  # Only creator can see/use
    WORKSPACE = "workspace"  # All workspace users can see/use
    SYSTEM = "system"  # System-provided agents (read-only for users)
