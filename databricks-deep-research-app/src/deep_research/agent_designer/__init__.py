"""Agent designer package."""

from .critic_types import CriticDirective, CriticVerdict, WorkflowAST
from .framework_tools import (
    BindToolToBlockTool,
    DeclareToolTool,
    DiscoverSourcesTool,
    ExtractCriticApprovedTool,
    ParseArchitectAstTool,
    ProposeWorkflowTool,
    SetModelTierTool,
    UpdateBlockTool,
    ValidateTool,
    builtin_designer_tools,
    get_global_registry,
    register_designer_tools,
)
from .sse_events import (
    DesignerSSEEvent,
    DoneEvent,
    ErrorEvent,
    MessageEvent,
    MutationProposedEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from .structural_gate import StructuralGateTool

# Eagerly initialise the process-wide registry on import so that any caller
# building a WorkflowExecutor with this module's tools available can resolve
# ``ToolRef(type="builtin", name=<designer-tool>)`` and
# ``ToolRef(type="enterprise", name=<designer-tool>)`` without an explicit
# wiring step. Tests rely on this side-effect to assert that the new tools
# are discoverable after importing this package.
_GLOBAL_DESIGNER_REGISTRY = get_global_registry()

__all__ = [
    "CriticDirective",
    "CriticVerdict",
    "WorkflowAST",
    "MessageEvent",
    "ToolCallEvent",
    "MutationProposedEvent",
    "ToolResultEvent",
    "ErrorEvent",
    "DoneEvent",
    "DesignerSSEEvent",
    "ProposeWorkflowTool",
    "UpdateBlockTool",
    "BindToolToBlockTool",
    "SetModelTierTool",
    "DeclareToolTool",
    "DiscoverSourcesTool",
    "ValidateTool",
    "ParseArchitectAstTool",
    "ExtractCriticApprovedTool",
    "StructuralGateTool",
    "builtin_designer_tools",
    "register_designer_tools",
    "get_global_registry",
]
