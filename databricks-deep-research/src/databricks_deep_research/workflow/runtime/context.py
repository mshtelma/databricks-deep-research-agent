from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from databricks_deep_research.agents.isolation import AgentOutput
from databricks_deep_research.events.types import StreamEvent
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.pools.pool_state import PoolState
from databricks_deep_research.tools.resolver import ToolResolver
from databricks_deep_research.workflow.context import ExecutionContext
from databricks_deep_research.workflow.definition import WorkflowDefinition, WorkflowNode
from databricks_deep_research.workflow.state import WorkflowState


@dataclass
class PlanExecuteRuntimeContext:
    node: WorkflowNode
    config: Any
    state: WorkflowState
    pools: dict[str, PoolState]
    llm: FrameworkLLMClient
    definition: WorkflowDefinition
    resolver: ToolResolver
    execution_context: ExecutionContext | None
    total_items_processed: int = 0
    replan_cycles: int = 0


@dataclass(frozen=True)
class PlanExecuteRunnerDeps:
    emit: Callable[[StreamEvent], StreamEvent]
    exec_node: Callable[[WorkflowNode, WorkflowState], AsyncGenerator[StreamEvent, None]]
    execute_agent: Callable[..., Awaitable[AgentOutput]]
    now: Callable[[], str]
    logger: Any
    record_step_completed: Callable[[], None]
