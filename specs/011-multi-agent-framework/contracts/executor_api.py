"""
Contract: WorkflowExecutor public API.

This defines the primary entry point for executing workflows.
The executor walks a WorkflowDefinition tree and yields StreamEvents.

Changes from original:
- LLMClient protocol replaced with FrameworkLLMClient (AsyncOpenAI wrapper)
- run_workflow and run_workflow_from_yaml use FrameworkLLMClient or raw AsyncOpenAI
- ExecutionContext references FrameworkLLMClient
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from typing import Any

from openai import AsyncOpenAI

from databricks_deep_research.events.types import StreamEvent
from databricks_deep_research.llm.client import FrameworkLLMClient
from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.workflow.context import ExecutionContext
from databricks_deep_research.workflow.definition import WorkflowDefinition
from databricks_deep_research.workflow.state import WorkflowState


class WorkflowExecutor:
    """Walks a workflow tree, executing each node and yielding streaming events.

    Usage:
        executor = WorkflowExecutor(context)
        async for event in executor.execute(definition, state):
            handle(event)
    """

    def __init__(self, context: ExecutionContext) -> None: ...

    async def execute(
        self,
        definition: WorkflowDefinition,
        state: WorkflowState | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute a workflow definition, yielding events as execution progresses.

        Args:
            definition: The workflow tree to execute. Must pass validation.
            state: Optional pre-initialized state. If None, a new state is created
                   with required_inputs checked against definition.

        Yields:
            StreamEvent instances for every significant execution point.

        Raises:
            ValidationError: If definition fails load-time validation.
            TokenBudgetExceededError: If token budget is exhausted.
            WorkflowCancelledError: If workflow is cancelled mid-execution.
        """
        ...  # pragma: no cover


# --- Convenience functions ---


async def run_workflow(
    definition: WorkflowDefinition,
    llm_client: FrameworkLLMClient,
    query: str,
    *,
    tools: Sequence[ResearchTool] = (),
    user_token: str | None = None,
    enterprise_tools: list[ResearchTool] | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """High-level convenience function: build context + state, execute workflow.

    This is the simplest way to run a workflow from outside the framework.

    Args:
        definition: Workflow to execute (from YAML or programmatic construction).
        llm_client: FrameworkLLMClient instance (wraps AsyncOpenAI with model tier mapping).
        query: The user query to research.
        tools: Additional tools to make available (beyond builtins).
        user_token: OBO token for enterprise tools.
        enterprise_tools: Pre-resolved enterprise tool instances.

    Yields:
        StreamEvent instances.
    """
    ...  # pragma: no cover


async def run_workflow_from_yaml(
    yaml_path: str,
    llm_client: FrameworkLLMClient,
    query: str,
    **kwargs: Any,
) -> AsyncGenerator[StreamEvent, None]:
    """Load a workflow from YAML and execute it.

    Args:
        yaml_path: Path to YAML workflow definition file.
        llm_client: FrameworkLLMClient instance.
        query: The user query.
        **kwargs: Passed to run_workflow().

    Yields:
        StreamEvent instances.
    """
    ...  # pragma: no cover


async def run_workflow_from_yaml_with_openai(
    yaml_path: str,
    openai_client: AsyncOpenAI,
    model_mapping: dict[str, str],
    query: str,
    **kwargs: Any,
) -> AsyncGenerator[StreamEvent, None]:
    """Load a workflow from YAML and execute it, constructing the LLM client from raw AsyncOpenAI.

    Convenience for standalone users who don't want to construct FrameworkLLMClient manually.

    Args:
        yaml_path: Path to YAML workflow definition file.
        openai_client: Raw AsyncOpenAI client.
        model_mapping: Model tier → concrete model name mapping.
        query: The user query.
        **kwargs: Passed to run_workflow().

    Yields:
        StreamEvent instances.
    """
    ...  # pragma: no cover
