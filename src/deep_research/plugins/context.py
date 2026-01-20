"""
Plugin Context
==============

Re-exports ResearchContext for convenient import by plugins.

Usage in plugins:
    from deep_research.plugins.context import ResearchContext
"""

from deep_research.agent.tools.base import ResearchContext

__all__ = ["ResearchContext"]
