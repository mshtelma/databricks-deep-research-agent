"""Databricks Deep Research — standalone multi-agent orchestration framework.

Public API re-exports for convenient top-level imports::

    from databricks_deep_research import (
        WorkflowDefinition,
        WorkflowNode,
        ExecutionContext,
        FrameworkLLMClient,
        StreamEvent,
    )

Execution primitives::

    from databricks_deep_research import (
        WorkflowExecutor,
        run_workflow,
        run_workflow_from_yaml,
    )

Builtin tools::

    from databricks_deep_research.tools.builtins import (
        WebSearchTool,
        WebCrawlTool,
        FileSearchTool,
    )
"""

# Errors
# Register builtin subtypes (side-effect import)
import databricks_deep_research.agents.builtins  # noqa: F401
from databricks_deep_research.errors import (
    NodeBudgetExceededError,
    TokenBudgetExceededError,
    WorkflowCancelledError,
    WorkflowError,
    WorkflowValidationError,
)

# Events
from databricks_deep_research.events.types import (
    FrameworkEvent,
    StreamEvent,
)

# LLM client
from databricks_deep_research.llm.client import (
    FrameworkLLMClient,
    LLMResponse,
    ModelTier,
    ModelTierConfig,
    ToolCall,
    parse_model_config,
)
from databricks_deep_research.runner import WorkflowResult, WorkflowRunner

# Tool protocol
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import (
    RegisteredTable,
    ResearchTool,
    TableRegistry,
    ToolContext,
    ToolDefinition,
    ToolResult,
)
from databricks_deep_research.tracing import (
    setup_mlflow_tracing,
    shutdown_mlflow_tracing,
    trace_span,
)

# Workflow types
from databricks_deep_research.workflow.context import ExecutionContext
from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.executor import (
    WorkflowExecutor,
    run_workflow,
    run_workflow_from_yaml,
)
from databricks_deep_research.workflow.loader import (  # noqa: F401 — side-effect: wires from_yaml/to_yaml/from_dict
    load_workflow,
    load_workflow_from_dict,
    load_workflow_from_string,
    save_workflow,
)
from databricks_deep_research.workflow.runtime_core import RuntimeState
from databricks_deep_research.workflow.runtime_core.api import WorkflowRunRequest, WorkflowRunResult

__all__ = [
    # Workflow definition
    "WorkflowDefinition",
    "WorkflowNode",
    "NodeType",
    # Execution context
    "ExecutionContext",
    # LLM client
    "FrameworkLLMClient",
    "LLMResponse",
    "ModelTier",
    "ModelTierConfig",
    "ToolCall",
    "parse_model_config",
    # Events
    "StreamEvent",
    "FrameworkEvent",
    # Tool protocol & factory
    "ResearchTool",
    "TableRegistry",
    "RegisteredTable",
    "ToolContext",
    "ToolDefinition",
    "ToolResult",
    "ToolFactoryContext",
    # Loader
    "load_workflow",
    "load_workflow_from_dict",
    "load_workflow_from_string",
    "save_workflow",
    # Executor
    "WorkflowExecutor",
    "run_workflow",
    "run_workflow_from_yaml",
    # Runner (high-level convenience API)
    "WorkflowRunner",
    "WorkflowResult",
    "RuntimeState",
    "WorkflowRunRequest",
    "WorkflowRunResult",
    # Tracing
    "setup_mlflow_tracing",
    "shutdown_mlflow_tracing",
    "trace_span",
    # Errors
    "WorkflowError",
    "WorkflowValidationError",
    "WorkflowCancelledError",
    "TokenBudgetExceededError",
    "NodeBudgetExceededError",
]
