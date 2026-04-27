"""Execution context threaded through workflow runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


@dataclass
class ExecutionContext:
    """Immutable-ish bag of cross-cutting concerns available to every node.

    Reserved-prefix propagation: ``user_id`` and ``approval_broker`` are
    forwarded by the harness into ``ToolContext.extras`` under the
    framework-private keys ``_framework_user_id`` and
    ``_framework_approval_broker`` respectively, so HITL gating
    (``react_loop_hitl.run_hitl_gate``) can resolve per-request ownership.

    Attributes:
        llm_client: The framework LLM client used for all model calls.
            Typed as ``Any`` at runtime to avoid import-time coupling;
            statically narrowed to ``FrameworkLLMClient`` under TYPE_CHECKING.
        checkpoint_handler: Optional handler implementing the checkpoint
            protocol (save / restore state snapshots).
        model_overrides: Per-node model-tier overrides, keyed by node id.
        user_token: OBO token forwarded from the request for enterprise tools.
        enterprise_tools: Loaded enterprise tool instances (e.g. Genie, VS).
        trace_enabled: Whether MLflow / tracing instrumentation is active.
        user_id: Authenticated user id; propagated to
            ``ToolContext.extras["_framework_user_id"]`` for HITL ownership.
        approval_broker: Optional ApprovalBroker; propagated to
            ``ToolContext.extras["_framework_approval_broker"]`` for HITL gating.
    """

    llm_client: Any  # FrameworkLLMClient (Any to avoid import issues at runtime)
    checkpoint_handler: Any | None = None  # CheckpointHandler protocol
    model_overrides: dict[str, str] = field(default_factory=dict)
    user_token: str | None = None
    enterprise_tools: list[Any] = field(default_factory=list)  # list[ResearchTool]
    trace_enabled: bool = True
    tool_call_cache: Any | None = None  # ToolCallCache shared across steps
    user_id: str | None = None
    approval_broker: Any | None = None  # ApprovalBroker protocol
