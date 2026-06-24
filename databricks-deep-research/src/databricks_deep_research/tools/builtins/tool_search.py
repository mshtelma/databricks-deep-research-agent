"""``tool_search`` builtin — on-demand fetch of deferred tool schemas (spec §5.5).

Mirrors this harness's own ``ToolSearch``: when an agent is wired to many tools,
the base prompt lists deferred tools by NAME + a one-line description only. The
model calls ``tool_search`` to load the FULL JSON Schema of the deferred tools it
needs; the matched tools are then **promoted** (their schema becomes visible to
the LLM on the next turn).

The tool is constructor-injected with a :class:`DeferredToolRegistry` (the loop's
single source of truth for deferral). It is always EAGER (never itself deferred)
so the model can always reach it. Promotion is recorded in the append-only
:class:`RuntimeState` via the runtime-store recorder threaded through
``ToolContext.extras`` (reserved key ``_framework_runtime_store``); absent a
store the tool still works (recording is best-effort).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from databricks_deep_research.tools.deferred import DeferredToolRegistry
from databricks_deep_research.tools.protocol import (
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)

__all__ = ["ToolSearchTool", "TOOL_SEARCH_NAME"]

#: Stable name; referenced by the ReAct loop when auto-injecting the tool and by
#: the fail-closed catalog builder (it is always eager).
TOOL_SEARCH_NAME = "tool_search"

#: Callable recording a promotion in the append-only RuntimeState. Injected by
#: the ReAct loop (which holds the typed runtime store + node id); ``None`` when
#: no store is wired, in which case promotion is observable only via logs.
PromotionRecorder = Callable[[list[str]], None]

_DESCRIPTION = (
    "Load the full JSON Schemas of deferred tools so you can call them. Many "
    "tools are listed in your prompt by NAME and a one-line description only; "
    "their full parameter schema is withheld until you fetch it here. Pass "
    "`names` for an exact selection (preferred — use the exact tool names from "
    "the catalog) or `query` to search deferred tools by keyword. The matched "
    "tools become callable on your next turn."
)


class ToolSearchTool:
    """Returns full schemas for matched deferred tools and promotes them."""

    def __init__(
        self,
        registry: DeferredToolRegistry,
        *,
        name: str = TOOL_SEARCH_NAME,
        description: str = "",
        max_results: int = 25,
        recorder: PromotionRecorder | None = None,
    ) -> None:
        self._registry = registry
        self._name = name
        self._description = description or _DESCRIPTION
        self._max_results = max_results
        self._recorder = recorder

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Exact names of deferred tools to load (preferred)."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Keyword query to search deferred tools when exact "
                            "names are unknown."
                        ),
                    },
                },
                "additionalProperties": False,
            },
            source_type="builtin",
            source_kind=SourceKind.builtin,
            # budget_free: schema lookups must not consume the research tool-call
            # budget (mirrors read_skill's progressive-disclosure tool).
            metadata={"budget_free": True},
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        names_raw = arguments.get("names")
        names: list[str] = []
        if isinstance(names_raw, list):
            names = [str(n).strip() for n in names_raw if str(n).strip()]
        elif isinstance(names_raw, str) and names_raw.strip():
            # Tolerate a single string or comma-separated list.
            names = [n.strip() for n in names_raw.split(",") if n.strip()]
        query_raw = arguments.get("query")
        query = query_raw.strip() if isinstance(query_raw, str) else ""
        if not names and not query:
            raise ValueError(
                "tool_search requires a non-empty 'names' list or 'query' string"
            )
        return {"names": names, "query": query}

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        del context  # promotion recording uses the constructor-injected recorder
        names: list[str] = arguments["names"]
        query: str = arguments["query"]

        matches = self._registry.match(
            names=names or None,
            query=query,
            limit=self._max_results,
        )

        if not matches:
            available = ", ".join(self._registry.deferred_names())
            hint = f" Deferred tools available: {available}." if available else ""
            logger.info(
                "TOOL_SEARCH_MISS names=%s query=%r", names, query[:120]
            )
            return ToolResult(
                content=(
                    "No deferred tools matched your request."
                    f"{hint} Use the exact tool names from the catalog."
                ),
                success=True,
                data={"source_kind": SourceKind.builtin, "matched": 0},
            )

        promoted = self._registry.promote([m.name for m in matches])
        self._record_promotion(promoted)

        payload = [
            {
                "name": m.name,
                "description": m.description,
                "parameters": m.parameters,
            }
            for m in matches
        ]
        content = (
            "Loaded full schemas for the following tools — they are now callable:"
            f"\n{json.dumps(payload, indent=2, default=str)}"
        )
        logger.info(
            "TOOL_SEARCH_HIT matched=%d promoted=%s",
            len(matches),
            promoted,
        )
        return ToolResult(
            content=content,
            success=True,
            data={
                "source_kind": SourceKind.builtin,
                "matched": len(matches),
                "promoted": promoted,
            },
        )

    def _record_promotion(self, promoted: list[str]) -> None:
        """Best-effort append of the promotion to the append-only RuntimeState."""
        if not promoted or self._recorder is None:
            return
        try:
            self._recorder(promoted)
        except Exception:  # pragma: no cover — recording must never break a call
            logger.warning(
                "TOOL_SEARCH_RECORD_FAILED promoted=%s", promoted, exc_info=True
            )
