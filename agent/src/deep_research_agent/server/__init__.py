"""
Deep Research Agent server package for MLflow-compatible endpoints with embedded UI serving.
"""

from .server import AgentServer, create_server, invoke, stream, parse_server_args
from .app import main

__all__ = ["AgentServer", "create_server", "invoke", "stream", "parse_server_args", "main"]