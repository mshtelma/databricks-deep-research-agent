"""SSE event types emitted by the designer chat surface.

Moved verbatim from orchestrator.py during the W5a refactor. Kept as a
standalone module so they survive the orchestrator's deletion — the
designer route shim (W5c) imports these to translate framework
StreamEvents into the SSE shape the frontend already understands."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# ---- SSE event models (discriminated union) ----


class _SSEBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MessageEvent(_SSEBase):
    type: Literal["message"] = "message"
    content: str


class ToolCallEvent(_SSEBase):
    type: Literal["tool_call"] = "tool_call"
    tool_name: str
    tool_call_id: str
    args: dict[str, Any]


class MutationProposedEvent(_SSEBase):
    type: Literal["mutation_proposed"] = "mutation_proposed"
    tool_name: str
    tool_call_id: str
    old_ast: dict[str, Any] | None
    new_ast: dict[str, Any]
    validation_errors: list[dict[str, Any]]
    summary: dict[str, int] | None  # node_count, tool_count, source_count when valid
    # Layer 2 auto-repair (designer-hardening plan). Each fix records the
    # deterministic rewrite the normalizer applied to the architect's AST
    # — kind ∈ {subtype_rewrite, tier_rewrite, tool_kind_rewrite,
    # pool_spec_rewrite, auto_declare_pool, set_minimum_max_tool_calls,
    # static_parallel_*_rewrite, brace_escape}.
    # Empty list when the architect emitted a clean AST (the common case
    # once Layer 4 architect prompt guardrails settle in).
    normalization_fixes: list[dict[str, Any]] = []


class ToolResultEvent(_SSEBase):
    """Used for tools that don't mutate the AST (discover_sources, list_*, validate)."""

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    result: dict[str, Any]


class ProgressEvent(_SSEBase):
    """Transient progress heartbeat for the UI during long designer turns.

    Emitted on agent-node starts (the slow Opus/GPT-5 steps) and loop
    iterations so the chat shows live activity instead of a frozen spinner
    while a multi-minute turn streams. Purely a streaming status line — the
    frontend renders it transiently and never persists it to the transcript
    (so it can't bloat the resent payload). Distinct from the wire-level SSE
    keepalive comment frame, which the route emits to defeat the gateway's
    idle-connection timeout."""

    type: Literal["progress"] = "progress"
    label: str
    iteration: int | None = None
    total: int | None = None


class ArchitectSynopsisEvent(_SSEBase):
    """Human-readable synopsis of what the designer built or changed this turn.

    Emitted ONCE at finalize (build / surgical-edit / topology lanes). Derived
    purely and deterministically from the final AST (topology + per-node roles +
    bound tools + required outputs + placeholder-pending warnings) — NO extra
    LLM call. Display-only: the frontend renders it as a "What I built" card and
    strips it from the resent transcript (it is not part of the wire
    ``ChatMessage`` schema), so it can never bloat the request payload."""

    type: Literal["architect_synopsis"] = "architect_synopsis"
    headline: str
    topology: str
    change_kind: Literal["created", "edited"] = "created"
    # One row per researcher/agent lane: {label: str, tools: list[str]}.
    lanes: list[dict[str, Any]] = []
    # Ordered downstream stage labels (e.g. synthesis -> coverage review -> final).
    pipeline: list[str] = []
    tools: list[str] = []
    outputs: list[str] = []
    # e.g. "1 lane still uses a default prompt" (from placeholder_pending_nodes).
    warnings: list[str] = []


class CriticReviewEvent(_SSEBase):
    """The LLM-as-judge critic's structured verdict, surfaced as a first-class
    event instead of buried as raw JSON inside the ``validate`` tool result.

    Mirrors :class:`workflow_critic.CritiqueResult` (the same dict the validate
    handler already computes — no extra LLM call). Display-only."""

    type: Literal["critic_review"] = "critic_review"
    verdict: Literal["pass", "needs_revision", "fail"]
    summary: str
    agent_findings: list[dict[str, Any]] = []
    coverage_gaps: list[dict[str, Any]] = []
    output_gaps: list[dict[str, Any]] = []


class ErrorEvent(_SSEBase):
    type: Literal["error"] = "error"
    message: str
    tool_call_id: str | None = None


class DoneEvent(_SSEBase):
    type: Literal["done"] = "done"


DesignerSSEEvent = (
    MessageEvent
    | ToolCallEvent
    | MutationProposedEvent
    | ToolResultEvent
    | ProgressEvent
    | ArchitectSynopsisEvent
    | CriticReviewEvent
    | ErrorEvent
    | DoneEvent
)
