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

# Python authoring API (Phases 1 + 2)
from databricks_deep_research.api import (
    Agent,
    AgentResult,
    ApiKey,
    ApprovalBroker,
    ApprovalDecision,
    BearerToken,
    Cite,
    Claim,
    CustomHeaders,
    DeltaCheckpointer,
    Description,
    Evidence,
    GateDeniedEvent,
    GateResumedEvent,
    GateTimeoutEvent,
    GateWaitingEvent,
    InMemoryBackend,
    InMemoryTodoStore,
    InProcessApprovalBroker,
    MCPAuth,
    MCPSchemaError,
    MCPSecurityError,
    MCPToolset,
    Parallel,
    PoolInjectSpec,
    PoolWriteSpec,
    RunContext,
    Sequence,
    SubAgent,
    SummaryInfo,
    Team,
    TeamStrategy,
    Todo,
    TodoStore,
    UCVolumeBackend,
    VerificationSummary,
    VirtualFilesystem,
    create_deep_agent,
    extract_verification,
    extract_verification_from_report,
    requires_approval,
    tool,
    write_todos_tool,
)

# Reusable Databricks OBO/SP tool wiring (shared by every host app)
from databricks_deep_research.core.databricks_auth import (
    build_obo_workspace_client,
    resolve_workspace_client,
)
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
from databricks_deep_research.tools.builtins.databricks_runner import (
    build_databricks_workflow_runner,
    workflow_requires_databricks,
)

# Tool protocol
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import (
    DATABRICKS_BOUND_TOOL_KINDS,
    RegisteredTable,
    ResearchTool,
    TableRegistry,
    ToolContext,
    ToolDefinition,
    ToolResult,
    required_ctx_fields_for_kind,
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
    "required_ctx_fields_for_kind",
    # Databricks OBO/SP tool wiring (reusable by host apps)
    "build_databricks_workflow_runner",
    "build_obo_workspace_client",
    "resolve_workspace_client",
    "workflow_requires_databricks",
    "DATABRICKS_BOUND_TOOL_KINDS",
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
    # Python authoring API (Phase 1)
    "Agent",
    "AgentResult",
    "SubAgent",
    "Sequence",
    "Parallel",
    "tool",
    "RunContext",
    "Cite",
    "Description",
    "PoolWriteSpec",
    "PoolInjectSpec",
    "VerificationSummary",
    "Claim",
    "Evidence",
    "SummaryInfo",
    "extract_verification",
    "extract_verification_from_report",
    "create_deep_agent",
    # Phase 2: HITL
    "ApprovalBroker",
    "ApprovalDecision",
    "InProcessApprovalBroker",
    "requires_approval",
    "GateWaitingEvent",
    "GateResumedEvent",
    "GateDeniedEvent",
    "GateTimeoutEvent",
    # Phase 2: Todos
    "Todo",
    "TodoStore",
    "InMemoryTodoStore",
    "write_todos_tool",
    # Phase 2: VFS
    "VirtualFilesystem",
    "InMemoryBackend",
    "UCVolumeBackend",
    # Phase 2: Checkpoint
    "DeltaCheckpointer",
    # Phase 3: Team
    "Team",
    "TeamStrategy",
    # Phase 3: MCP
    "MCPToolset",
    "MCPSchemaError",
    "MCPSecurityError",
    "MCPAuth",
    "BearerToken",
    "ApiKey",
    "CustomHeaders",
]
