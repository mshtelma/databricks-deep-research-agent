"""Public Python authoring API for ``databricks_deep_research``.

Top-level imports::

    from databricks_deep_research.api import (
        Agent, AgentResult, SubAgent,
        Sequence, Parallel,
        tool, RunContext, Cite,
        extract_verification, VerificationSummary,
        create_deep_agent,
    )

Phase 2 (HITL + VFS + Todos + Checkpointing) and Phase 3 (Team + MCP)
re-exports will be appended in subsequent phases.

Reserved-prefix convention: keys passed via ``Agent.extras`` or
``ToolContext.extras`` that begin with ``_framework_`` are reserved for
the framework's runtime capabilities (approval broker, virtual filesystem,
todos store, checkpointer, thread_id). User-chosen keys MUST NOT use this
prefix.
"""

from databricks_deep_research.api.agent import Agent, create_deep_agent
from databricks_deep_research.api.approval import (
    ApprovalBroker,
    ApprovalDecision,
    InProcessApprovalBroker,
    requires_approval,
)
from databricks_deep_research.api.checkpoint import DeltaCheckpointer
from databricks_deep_research.api.composition import Parallel, Sequence
from databricks_deep_research.api.pools import PoolInjectSpec, PoolWriteSpec
from databricks_deep_research.api.result import AgentResult
from databricks_deep_research.api.subagent import PoolMode, SubAgent
from databricks_deep_research.api.team import Team, TeamStrategy
from databricks_deep_research.api.todos import (
    InMemoryTodoStore,
    Todo,
    TodoStatus,
    TodoStore,
    write_todos_tool,
)
from databricks_deep_research.api.vfs import (
    InMemoryBackend,
    UCVolumeBackend,
    VirtualFilesystem,
)
from databricks_deep_research.citation.extraction import (
    Claim,
    Evidence,
    SummaryInfo,
    VerificationSummary,
    extract_verification,
    extract_verification_from_report,
)
from databricks_deep_research.events.hitl import (
    GateDeniedEvent,
    GateResumedEvent,
    GateTimeoutEvent,
    GateWaitingEvent,
)
from databricks_deep_research.tools.api import (
    Cite,
    Description,
    RunContext,
    tool,
)
from databricks_deep_research.tools.mcp import (
    MCPSchemaError,
    MCPServerConfig,
    MCPToolset,
    SecretResolver,
    build_mcp_toolset,
)
from databricks_deep_research.tools.mcp_auth import (
    ApiKey,
    BearerToken,
    CustomHeaders,
    MCPAuth,
)
from databricks_deep_research.tools.mcp_security import MCPSecurityError

__all__ = [
    # Agent surface
    "Agent",
    "AgentResult",
    "SubAgent",
    "PoolMode",
    "create_deep_agent",
    # Composition
    "Sequence",
    "Parallel",
    # Tools
    "tool",
    "RunContext",
    "Cite",
    "Description",
    # Pool DI
    "PoolWriteSpec",
    "PoolInjectSpec",
    # Verification (Phase 1 framework-side relocation)
    "VerificationSummary",
    "Claim",
    "Evidence",
    "SummaryInfo",
    "extract_verification",
    "extract_verification_from_report",
    # Phase 2: HITL approval
    "ApprovalBroker",
    "ApprovalDecision",
    "InProcessApprovalBroker",
    "requires_approval",
    # Phase 2: Gate events
    "GateWaitingEvent",
    "GateResumedEvent",
    "GateDeniedEvent",
    "GateTimeoutEvent",
    # Phase 2: TodoStore
    "Todo",
    "TodoStatus",
    "TodoStore",
    "InMemoryTodoStore",
    "write_todos_tool",
    # Phase 2: VFS
    "VirtualFilesystem",
    "InMemoryBackend",
    "UCVolumeBackend",
    # Phase 2: Checkpointing
    "DeltaCheckpointer",
    # Phase 3: Team
    "Team",
    "TeamStrategy",
    # Phase 3: MCP
    "MCPToolset",
    "MCPServerConfig",
    "MCPSchemaError",
    "MCPSecurityError",
    "MCPAuth",
    "BearerToken",
    "ApiKey",
    "CustomHeaders",
    "SecretResolver",
    "build_mcp_toolset",
]
