"""Databricks Agent Server package.

This package provides the Databricks-compatible agent server that wraps
the research orchestrator with @invoke and @stream handlers.
"""

from deep_research.agent_server.agent import invoke, stream

__all__ = ["invoke", "stream"]
