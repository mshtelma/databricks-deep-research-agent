"""Builtin subtype registry.

Each builtin module registers a post-processor and optional config
enrichment function.  The harness calls these automatically when the
agent node's subtype matches a registered builtin.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.events.types import StreamEvent
from databricks_deep_research.workflow.state import WorkflowState

if TYPE_CHECKING:
    from databricks_deep_research.agents.isolation import AgentInput, AgentOutput
    from databricks_deep_research.llm.client import FrameworkLLMClient
    from databricks_deep_research.pools.pool_state import PoolState
    from databricks_deep_research.tools.protocol import ResearchTool, ToolContext

logger = logging.getLogger(__name__)

# Type aliases for clarity
PostProcessFn = Callable[
    [str, Any, AgentNodeConfig, WorkflowState],  # node_id, output, config, state
    list[StreamEvent],
]
ConfigEnrichFn = Callable[
    [AgentNodeConfig, WorkflowState, dict[str, Any] | None],  # config, state, runtime_context
    AgentNodeConfig,
]
ExecuteFn = Callable[
    [
        str,
        AgentNodeConfig,
        WorkflowState,
        "FrameworkLLMClient",
        list["ResearchTool"],
        dict[str, "PoolState"],
        "AgentInput",
        list[dict[str, Any]],
        "ToolContext",
    ],
    Awaitable["AgentOutput | None"],
]


@dataclass
class BuiltinSubtype:
    """Registration entry for a builtin agent subtype."""

    subtype: str
    post_process: PostProcessFn | None = None
    enrich_config: ConfigEnrichFn | None = None
    execute: ExecuteFn | None = None
    default_system_prompt: str = ""
    default_user_prompt: str = ""
    output_model: Any = None  # Pydantic model class


# Global registry
_REGISTRY: dict[str, BuiltinSubtype] = {}


def register_builtin(
    subtype: str,
    *,
    post_process: PostProcessFn | None = None,
    enrich_config: ConfigEnrichFn | None = None,
    execute: ExecuteFn | None = None,
    default_system_prompt: str = "",
    default_user_prompt: str = "",
    output_model: Any = None,
) -> None:
    """Register a builtin subtype with optional hooks."""
    _REGISTRY[subtype] = BuiltinSubtype(
        subtype=subtype,
        post_process=post_process,
        enrich_config=enrich_config,
        execute=execute,
        default_system_prompt=default_system_prompt,
        default_user_prompt=default_user_prompt,
        output_model=output_model,
    )
    logger.debug("BUILTIN_REGISTERED subtype=%s", subtype)


def get_builtin(subtype: str) -> BuiltinSubtype | None:
    """Look up a registered builtin by subtype name."""
    return _REGISTRY.get(subtype)


def list_builtins() -> list[str]:
    """Return all registered builtin subtype names."""
    return list(_REGISTRY.keys())


__all__ = [
    "BuiltinSubtype",
    "PostProcessFn",
    "ConfigEnrichFn",
    "ExecuteFn",
    "register_builtin",
    "get_builtin",
    "list_builtins",
]
