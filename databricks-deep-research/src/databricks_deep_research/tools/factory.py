"""Tool factory protocol and context — creates ResearchTool from declarations.

Factories are registered with :class:`ToolResolver` and consulted in order
when a tool name references a :class:`ToolDeclaration`.  Each factory
declares which ``kind`` values it supports and creates the corresponding
tool instance given a :class:`ToolFactoryContext` with runtime dependencies.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.workflow.definition import ToolDeclaration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Factory context — runtime dependencies available at creation time
# ---------------------------------------------------------------------------


@dataclass
class ToolFactoryContext:
    """Dependencies available to tool factories at creation time.

    Fields are optional — factories validate the ones they need and raise
    ``ValueError`` with a clear message if a required dependency is missing.
    """

    workspace_client: Any | None = None  # databricks.sdk.WorkspaceClient
    user_token: str | None = None  # OBO token for authenticated calls
    search_client: Any | None = None  # SearchClient protocol (web_search)
    crawler: Any | None = None  # ContentCrawler protocol (web_crawl)
    file_index: Any | None = None  # FileIndex for file_search
    extras: dict[str, Any] = field(default_factory=dict)  # app-specific deps

    @classmethod
    def from_defaults(
        cls,
        *,
        workspace_client: Any | None = None,
        user_token: str | None = None,
        brave_api_key: str | None = None,
        extras: dict[str, Any] | None = None,
    ) -> ToolFactoryContext:
        """Create a context with auto-detected defaults.

        Auto-detects:
        - ``workspace_client``: from ``databricks.sdk.WorkspaceClient()``
          defaults if not provided (falls back to ``None``).
        - ``search_client``: ``BraveSearchAdapter`` from the *brave_api_key*
          parameter or the ``BRAVE_API_KEY`` environment variable.
        - ``crawler``: always ``None`` — ``WebCrawlTool`` uses its built-in
          httpx + trafilatura pipeline when no crawler is injected.

        All auto-detection is wrapped in try/except so missing dependencies
        result in ``None`` fields.  Errors surface later only if a factory
        actually needs the missing dependency.
        """
        # -- workspace_client ---------------------------------------------------
        ws = workspace_client
        if ws is None:
            try:
                from databricks.sdk import WorkspaceClient as _WC

                ws = _WC()
            except Exception:
                logger.debug(
                    "TOOL_FACTORY_CONTEXT workspace_client auto-detect failed; "
                    "set explicitly if needed"
                )
                ws = None

        # -- search_client (Brave) ----------------------------------------------
        api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        search_client: Any = None
        if api_key:
            try:
                from databricks_deep_research.tools.builtins.brave_search import (
                    BraveSearchAdapter,
                )

                search_client = BraveSearchAdapter(api_key=api_key)
            except Exception:
                logger.debug(
                    "TOOL_FACTORY_CONTEXT BraveSearchAdapter creation failed"
                )

        return cls(
            workspace_client=ws,
            user_token=user_token,
            search_client=search_client,
            crawler=None,
            extras=extras or {},
        )


# ---------------------------------------------------------------------------
# Factory protocol
# ---------------------------------------------------------------------------


class ToolFactory(Protocol):
    """Protocol for creating ResearchTool instances from declarations."""

    def supports(self, kind: str) -> bool:
        """Return ``True`` if this factory can create tools of the given kind."""
        ...

    async def create(
        self,
        declaration: ToolDeclaration,
        context: ToolFactoryContext,
    ) -> ResearchTool:
        """Create a tool instance.

        Raises:
            ValueError: If required config or context fields are missing.
        """
        ...
