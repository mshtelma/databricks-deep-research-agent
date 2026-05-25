"""Stateless chat session orchestrator for the Agent Designer.

Receives {messages, current_ast} per turn, runs an LLM with the CHAT_TOOLS,
dispatches each tool-call to the matching mutation primitive, validates the
resulting AST via load_workflow_from_dict, and yields SSE events.

Stateless invariant: no per-session storage on the orchestrator. The same
{messages, current_ast} input always produces the same output sequence
(modulo nondeterminism from the LLM).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from deep_research.agent_designer import mutations
from deep_research.agent_designer.assets import (
    DesignerAsset,
    asset_context_payload,
    inspect_assets,
    recommend_tools_for_assets,
)
from deep_research.agent_designer.ast_normalizer import normalize_ast
from deep_research.agent_designer.designer_architect import (
    WorkflowDesignBrief,
    designer_system_prompt,
)
from deep_research.agent_designer.discovery import DesignerDiscoveryAdapter, SourceKind
from deep_research.agent_designer.registry import (
    model_tiers_payload as _model_tiers_payload,
)
from deep_research.agent_designer.registry import (
    node_types_payload as _node_types_payload,
)
from deep_research.agent_designer.registry import (
    query_modes_payload as _query_modes_payload,
)
from deep_research.agent_designer.registry import (
    research_depths_payload as _research_depths_payload,
)
from deep_research.agent_designer.registry import (
    source_kinds_payload as _source_kinds_payload,
)
from deep_research.agent_designer.registry import (
    tool_kinds_payload as _tool_kinds_payload,
)
from deep_research.agent_designer.tools import (
    AddBlockArgs,
    BindToolArgs,
    DeclareToolArgs,
    DeleteBlockArgs,
    DiscoverSourcesArgs,
    InspectAssetsArgs,
    MoveBlockArgs,
    ProposeWorkflowArgs,
    RecommendToolsForAssetsArgs,
    RemoveToolArgs,
    SetModelTierArgs,
    UpdateBlockArgs,
    parse_tool_args,
)
from deep_research.agent_designer.workflow_builder import (
    build_direct_workflow,
    build_web_research_workflow,
)
from deep_research.core.auth import get_service_principal_workspace_client
from deep_research.observability.agent_designer_metrics import (
    log_chat_mutation,
    log_run_principal,
)

from .sse_events import (
    DesignerSSEEvent as DesignerSSEEvent,
)
from .sse_events import (
    DoneEvent as DoneEvent,
)
from .sse_events import (
    ErrorEvent as ErrorEvent,
)
from .sse_events import (
    MessageEvent as MessageEvent,
)
from .sse_events import (
    MutationProposedEvent as MutationProposedEvent,
)
from .sse_events import (
    ToolCallEvent as ToolCallEvent,
)
from .sse_events import (
    ToolResultEvent as ToolResultEvent,
)

# SSE event types — moved to sse_events.py in the W5a refactor. Kept as
# re-exports so existing callers (tests, route handler) continue to
# import them from this module unchanged.
from .sse_events import (
    _SSEBase as _SSEBase,
)

# Public helpers — moved to validation_helpers.py in the W6 refactor. Kept
# as re-exports so existing callers (tests/complex/test_scaffold_and_run.py
# imports these by name from this module) continue to resolve.
from .validation_helpers import (
    _node_count as _node_count,
)
from .validation_helpers import (
    _quality_advice as _quality_advice,
)
from .validation_helpers import (
    _validate_ast as _validate_ast,
)
from .validation_helpers import (
    _validation_error as _validation_error,
)

logger = logging.getLogger(__name__)

# ---- Limits ----

MAX_MESSAGES = 20
MAX_AST_BYTES = 100 * 1024
MAX_PAYLOAD_BYTES = 200 * 1024
MAX_DESIGNER_TOOL_ROUNDS = 4

_READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "discover_sources",
        "inspect_assets",
        "recommend_tools_for_assets",
        "list_node_types",
        "list_tool_kinds",
        "list_modes",
        "validate",
    }
)

_MUTATING_DESIGNER_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "propose_workflow",
        "add_block",
        "update_block",
        "delete_block",
        "move_block",
        "declare_tool",
        "remove_tool",
        "bind_tool_to_block",
        "set_model_tier",
    }
)

# Valid model tier names per app.yaml. Any other value the architect emits
# is normalized to a sensible fallback so the runner doesn't crash on
# Opus-hallucinated tier names like "standard", "medium", "high", etc.
_VALID_MODEL_TIERS: frozenset[str] = frozenset(
    {"simple", "analytical", "complex", "bulk_analysis", "fast"}
)


_ARCHITECT_JSON_BLOCK_RE = __import__("re").compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", __import__("re").DOTALL
)


def _derive_normalization_fixes(
    state: Any, node_id: str
) -> list[dict[str, Any]]:
    """Re-derive Layer 2 fixes from the architect's raw message.

    The framework executor pushes only ``ToolResult.content`` to state, so the
    ``ParseArchitectAstTool``'s ``normalization_fixes`` payload doesn't reach
    the orchestrator. Running :func:`normalize_ast` on the un-normalized AST
    extracted from ``state['architect_message']`` reproduces the same
    deterministic fix list (both code paths use the exact same normalizer).

    Returns an empty list when this isn't the parse_architect_ast node, when
    the architect emitted no JSON block, or when the JSON failed to parse.
    """
    if node_id != "parse_architect_ast":
        return []
    try:
        raw_msg = state.get("architect_message")
    except Exception:
        return []
    if not isinstance(raw_msg, str) or not raw_msg:
        return []
    match = _ARCHITECT_JSON_BLOCK_RE.search(raw_msg)
    if match is None:
        return []
    try:
        raw_ast = json.loads(match.group(1))
    except (ValueError, TypeError):
        return []
    if not isinstance(raw_ast, dict):
        return []
    _, fixes = normalize_ast(raw_ast)
    return [f.to_dict() for f in fixes]


def _derive_workflow_llm_client(framework_llm: Any, workflow_def: Any) -> Any:
    """Apply workflow-local ``models:`` tiers to a framework LLM client.

    ``WorkflowRunner`` already layers YAML ``models:`` definitions onto the
    client before execution. The Designer shim builds ``WorkflowExecutor``
    directly so it can provide a designer-specific tool registry, so it must do
    the same layering here or workflow-local tiers such as ``critic`` fail at
    runtime.
    """
    models = getattr(workflow_def, "models", None)
    if not models:
        return framework_llm

    from databricks_deep_research.llm.client import parse_model_config

    yaml_mapping = parse_model_config(models)
    logger.info(
        "DESIGNER_APPLY_YAML_MODELS tiers=%s",
        list(yaml_mapping.keys()),
    )
    return framework_llm.derive(yaml_mapping)


def _coerce_ast_snapshot(raw: Any) -> dict[str, Any] | None:
    """Coerce a cached/state AST value into a non-empty dict snapshot."""
    if isinstance(raw, dict):
        return raw if raw else None
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if isinstance(decoded, dict) and decoded:
            return decoded
    return None


def _normalize_model_tiers(ast: dict[str, Any]) -> dict[str, Any]:
    """Walk the AST and replace any unknown ``model_tier`` value with a valid
    fallback. Lane researchers / coordinator default to analytical;
    synthesizers default to complex. Mutates a deep copy of the input."""
    import copy as _copy

    if not isinstance(ast, dict):
        return ast
    normalized = _copy.deepcopy(ast)

    def walk(n: Any) -> None:
        if not isinstance(n, dict):
            return
        cfg = n.get("config")
        if isinstance(cfg, dict):
            tier = cfg.get("model_tier")
            if isinstance(tier, str) and tier and tier not in _VALID_MODEL_TIERS:
                subtype = cfg.get("subtype") or ""
                cfg["model_tier"] = (
                    "complex" if subtype == "synthesizer" else "analytical"
                )
            body = cfg.get("body")
            if isinstance(body, dict):
                walk(body)
            evaluator = cfg.get("evaluator")
            if isinstance(evaluator, dict):
                walk(evaluator)
        for child in n.get("children") or []:
            walk(child)

    root = normalized.get("root") or normalized
    walk(root)
    return normalized


def _mutation_event_for_ast_change(
    *,
    tool_name: str,
    tool_call_id: str,
    raw_ast: Any,
    last_ast_seen: dict[str, Any],
    normalization_fixes: list[dict[str, Any]],
) -> MutationProposedEvent | None:
    """Build a mutation event when a cached/state AST differs from the last snapshot."""
    new_ast = _coerce_ast_snapshot(raw_ast)
    if new_ast is None:
        return None
    new_ast = _normalize_model_tiers(new_ast)
    if new_ast == last_ast_seen or not new_ast:
        return None
    errors, summary = _validate_ast(new_ast)
    return MutationProposedEvent(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        old_ast=last_ast_seen,
        new_ast=new_ast,
        validation_errors=errors,
        summary=summary,
        normalization_fixes=normalization_fixes,
    )


_BUILD_REQUEST_TERMS: frozenset[str] = frozenset(
    {
        "build",
        "create",
        "design",
        "generate",
        "make",
        "scaffold",
        "set up",
        "setup",
        "construct",
    }
)

_WORKFLOW_REQUEST_TERMS: frozenset[str] = frozenset(
    {
        "agent",
        "assistant",
        "workflow",
        "researcher",
        "pipeline",
    }
)


class RequestTooLargeError(ValueError):
    """Request exceeded one of the size limits."""


# ---- LLM client protocol ----


class LLMToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    arguments: dict[str, Any]


class LLMStreamChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str | None = None
    tool_call: LLMToolCall | None = None
    finish: bool = False


class LLMClientProto(Protocol):
    """Structural protocol for the chat-completion LLM client."""

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[LLMStreamChunk]: ...


# ---- AST helpers ----

_RESEARCH_INTENT_TERMS: frozenset[str] = frozenset(
    {
        "deep research",
        "research",
        "researcher",
        "web search",
        "search",
        "crawl",
        "summarize",
        "summarise",
        "summary",
        "report",
        "simple workflow",
        "research workflow",
        "investigate",
        "analyze",
        "analyse",
        "find sources",
        "sources",
    }
)


def _extract_required_outputs(ast: dict[str, Any]) -> list[str]:
    """Pull the brief's ``required_outputs`` back out of the generated AST.

    The Designer builder stores them under each plan_and_execute node's
    ``config.synthesis_metadata.designer_required_outputs`` (newline-joined).
    We walk the AST and concatenate any we find; deduplicated and order
    preserved. Returns an empty list when no metadata is present (legacy
    saved agents).
    """
    out: list[str] = []
    seen: set[str] = set()

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        config = node.get("config") or {}
        if isinstance(config, dict):
            meta = config.get("synthesis_metadata") or {}
            if isinstance(meta, dict):
                raw = meta.get("designer_required_outputs") or ""
                if isinstance(raw, str) and raw.strip():
                    for line in raw.splitlines():
                        cleaned = line.strip()
                        if cleaned and cleaned not in seen:
                            out.append(cleaned)
                            seen.add(cleaned)
            body = config.get("body")
            if isinstance(body, dict):
                walk(body)
        for child in node.get("children", []) or []:
            walk(child)

    walk(ast.get("root"))
    return out


async def _critique_ast(
    ast: dict[str, Any],
    intent: str,
    llm: LLMClientProto,
) -> dict[str, Any] | None:
    """Run the LLM-as-judge critic against the AST + the user's intent.

    Returns a plain-dict serialization of :class:`CritiqueResult` so the
    orchestrator can include it directly in the tool-result event. ``None``
    is returned when the inputs are insufficient (no intent or no AST) —
    callers should treat that as "skip critique" rather than as a failure.
    """
    if not ast or not intent or not intent.strip():
        return None
    # Local import to avoid a circular dependency at module load: this module
    # is the consumer of LLMClientProto; workflow_critic mirrors the same
    # protocol so the import direction stays orchestrator → workflow_critic.
    from deep_research.agent_designer.workflow_critic import (
        critique_workflow_against_intent,
    )

    required_outputs = _extract_required_outputs(ast)
    result = await critique_workflow_against_intent(
        definition=ast,
        intent=intent,
        required_outputs=required_outputs,
        llm=llm,
    )
    return result.model_dump()


def _is_research_intent(intent: str) -> bool:
    """Return whether the intent should start from a research workflow scaffold."""
    normalized = intent.casefold()
    return any(term in normalized for term in _RESEARCH_INTENT_TERMS)


def _workflow_name(intent: str) -> str:
    return intent[:80] if intent else "Untitled Agent"


def _propose_research_ast(
    intent: str,
    design_brief: WorkflowDesignBrief | None = None,
) -> dict[str, Any]:
    """Build a useful research workflow instead of a one-node placeholder."""
    return build_web_research_workflow(intent, _workflow_name(intent), design_brief)


def _propose_initial_ast(
    intent: str,
    design_brief: WorkflowDesignBrief | None = None,
) -> dict[str, Any]:
    """Skeleton workflow scaffold from intent. Caller refines via further tool calls."""
    if _is_research_intent(intent):
        return _propose_research_ast(intent, design_brief)

    return build_direct_workflow(intent, _workflow_name(intent))


def _with_architect_system_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepend the YAML-backed Designer architect/critic prompt for the LLM."""
    return [{"role": "system", "content": designer_system_prompt()}, *messages]


def _message_content_to_text(content: Any) -> str:
    """Return plain text from OpenAI-style message content shapes."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


def _last_user_message_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_content_to_text(message.get("content"))
    return ""


def _looks_like_workflow_build_request(messages: list[dict[str, Any]]) -> bool:
    """Heuristic guard for requests that should result in a workflow mutation."""
    normalized = _last_user_message_text(messages).casefold()
    if not normalized:
        return False
    has_build_term = any(term in normalized for term in _BUILD_REQUEST_TERMS)
    has_workflow_term = any(term in normalized for term in _WORKFLOW_REQUEST_TERMS)
    if has_build_term and has_workflow_term:
        return True
    return any(
        phrase in normalized
        for phrase in (
            "agent that",
            "assistant that",
            "workflow that",
            "research assistant",
            "research agent",
        )
    )


def _assistant_tool_calls_message(
    tool_calls: list[LLMToolCall],
    *,
    content: str = "",
) -> dict[str, Any]:
    """Build the assistant transcript message that requested read-only tools."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, default=str),
                },
            }
            for tc in tool_calls
        ],
    }


def _tool_result_message(result: ToolResultEvent) -> dict[str, Any]:
    """Build an OpenAI-compatible tool result transcript message."""
    return {
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "name": result.tool_name,
        "content": json.dumps(result.result, default=str),
    }


def _patch_node_output_models(
    node: Any,
    mapping: dict[str, type],
) -> None:
    """Recursively patch agent nodes' ``output_model`` config entry.

    The framework's ``AgentNodeConfig`` accepts an ``output_model`` field
    (Pydantic class) for structured-output validation; YAML cannot easily
    encode a Python class reference, so the designer route shim assigns it
    programmatically after the workflow is loaded. The patch lives in the
    raw ``node.config`` dict so it survives the executor's later
    ``AgentNodeConfig(**node.config)`` reconstruction.
    """
    if node is None:
        return
    node_id = getattr(node, "id", None)
    config = getattr(node, "config", None)
    if node_id in mapping and isinstance(config, dict):
        config["output_model"] = mapping[node_id]
    for child in getattr(node, "children", None) or []:
        _patch_node_output_models(child, mapping)
    # plan_and_execute nests planner/evaluator inside config rather than children
    if isinstance(config, dict):
        body = config.get("body")
        if body is not None and hasattr(body, "id"):
            _patch_node_output_models(body, mapping)


# ---- Orchestrator ----


class DesignerChatOrchestrator:
    """Dispatches LLM tool-calls to mutation primitives and yields SSE events.

    Stateless: every run_turn invocation accepts {messages, current_ast} from the caller.
    """

    def __init__(
        self,
        llm: LLMClientProto,
        discovery: DesignerDiscoveryAdapter,
    ) -> None:
        self._llm = llm
        self._discovery = discovery

    async def run_turn(
        self,
        messages: list[dict[str, Any]],
        current_ast: dict[str, Any] | None,
        session_id: str | None,
        user_token: str,
        current_user_id: str = "",
        assets: list[DesignerAsset] | list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[DesignerSSEEvent, None]:
        """Drive the designer_workflow.yaml framework workflow and translate
        framework StreamEvents → DesignerSSEEvents for the frontend.

        This is the W5c thin shim that replaces the ~600 LOC hand-coded loop.
        It loads the YAML workflow, programmatically patches the architect /
        critic agent nodes with their structured-output models, builds a live
        WorkflowState seeded from the request, runs the workflow via
        WorkflowRunner.stream(), and emits a DesignerSSEEvent for every
        framework StreamEvent the frontend cares about.
        """
        import os
        from pathlib import Path

        from databricks_deep_research.workflow.loader import load_workflow
        from databricks_deep_research.workflow.state import WorkflowState

        from deep_research.agent.adapters.llm_adapter import (
            create_framework_llm_client,
        )
        from deep_research.agent_designer.critic_types import (
            CriticVerdict,
        )

        # Enforce limits BEFORE the workflow starts (cheap and fast)
        self._check_limits(messages, current_ast, assets)
        # The client owns session persistence; the orchestrator stays stateless.
        _ = session_id

        # Resolve run_as from the AST and log the execution principal for audit.
        run_as_raw = (current_ast or {}).get("run_as", "caller")
        if isinstance(run_as_raw, dict) and "service_principal_id" in run_as_raw:
            sp_id: str = run_as_raw["service_principal_id"]
            from fastapi import HTTPException as _HTTPException

            try:
                await get_service_principal_workspace_client(
                    sp_id=sp_id,
                    requesting_user_id=current_user_id,
                )
            except _HTTPException:
                yield ErrorEvent(
                    message=f"Permission denied: missing CAN_USE_AS on service principal {sp_id!r}",
                )
                return
            log_run_principal(
                {
                    "requested_by_user_id": current_user_id,
                    "executed_as_sp_id": sp_id,
                    "run_kind": "sp",
                }
            )
        else:
            log_run_principal(
                {
                    "requested_by_user_id": current_user_id,
                    "executed_as_sp_id": None,
                    "run_kind": "caller",
                }
            )

        # Unused-but-required by the legacy signature; kept for back-compat.
        _ = user_token

        # 1. Resolve workflow path (allow override via env var for A/B testing).
        wf_path_env = os.environ.get("DESIGNER_WORKFLOW_YAML")
        if wf_path_env:
            wf_path = Path(wf_path_env)
        else:
            wf_path = Path(__file__).parent / "designer_workflow.yaml"

        # 2. Load the workflow and programmatically patch output_model classes
        #    on the architect and critic agent nodes (AgentNodeConfig.output_model
        #    is a free field on the node config dict, validated by the executor
        #    when it constructs AgentNodeConfig from node.config at runtime).
        try:
            workflow_def = load_workflow(str(wf_path))
        except Exception as exc:
            logger.exception("DESIGNER_WORKFLOW_LOAD_FAILED")
            yield ErrorEvent(message=f"Designer workflow load failed: {exc}")
            yield DoneEvent()
            return

        # NOTE: architect intentionally has NO output_model patch — it's a
        # tool-using agent (ReAct loop), and the framework's structured-output
        # path is bypassed when tools are present (Codex iter-2 fix #2). The
        # architect's system prompt instructs it to end its turn with a fenced
        # ```json ... ``` block; parse_architect_ast (next tool node) extracts
        # the AST from that final message via regex. The critic IS patched
        # because it's a no-tools agent and structured output enforces the
        # CriticVerdict shape we need for the loop's stop condition.
        _patch_node_output_models(
            workflow_def.root,
            {
                "critic": CriticVerdict,
            },
        )

        # 3. Build the seeded WorkflowState. The most recent user message
        #    drives the architect; prior turns ride along as
        #    conversation_history (W1 framework primitive).
        user_intent = ""
        if messages:
            last = messages[-1]
            user_intent = _message_content_to_text(last.get("content"))

        prior_messages = messages[:-1] if len(messages) > 1 else []

        state = WorkflowState(
            query=user_intent,
            conversation_history=prior_messages,
        )
        # Seed the keys the architect's user_prompt_template renders. Framework
        # template rendering stringifies values, so encode the current_ast as
        # JSON to round-trip cleanly.
        state.append("init", "user_intent", user_intent)
        state.append("init", "current_ast", json.dumps(current_ast or {}))
        state.append("init", "critic_verdict", "")
        state.append("init", "gate_result", "")
        state.append("init", "designer_assets", asset_context_payload(assets))

        # 4. Build the framework LLM client from the app's underlying LLM.
        #    The orchestrator's `self._llm` is the LLMClientProto adapter
        #    (AppLLMAdapter), which wraps the raw app LLMClient as `._llm`.
        #    Defensively fall back to the adapter itself when no wrapped
        #    client is exposed (test fakes may not declare ._llm).
        from typing import cast as _cast

        from deep_research.services.llm.client import LLMClient as _LLMClient

        underlying_app_llm = getattr(self._llm, "_llm", None) or self._llm
        try:
            framework_llm = create_framework_llm_client(
                _cast(_LLMClient, underlying_app_llm)
            )
            framework_llm = _derive_workflow_llm_client(framework_llm, workflow_def)
        except Exception as exc:
            logger.exception("DESIGNER_FRAMEWORK_LLM_BUILD_FAILED")
            yield ErrorEvent(message=f"Designer LLM init failed: {exc}")
            yield DoneEvent()
            return

        # Build a WorkflowRunner via the app's shared factory so the Designer
        # follows the same construction pattern as every other entry point
        # (see workflow_runner_factory.py for the convention). The Designer
        # threads its own ToolRegistry via runner.stream(tool_registry=...)
        # so `type: tool` nodes (parse_architect_ast, structural_gate,
        # extract_critic_approved) resolve through the pre-populated registry.
        from databricks_deep_research.tools.registry import ToolRegistry

        from deep_research.agent.workflow_runner_factory import (
            build_app_workflow_runner,
        )
        from deep_research.agent_designer.framework_tools import (
            register_designer_tools,
        )

        designer_registry = ToolRegistry()
        # Conversation-local AST cache. During the architect's ReAct loop the
        # workflow state's current_ast does NOT update between tool calls —
        # only the post-agent parse_architect_ast node writes back. So the
        # mutation tools maintain their own cache: propose_workflow seeds it,
        # subsequent update_block/etc. read+write it. Net effect: the LLM
        # doesn't have to echo the AST as an argument anymore, saving ~5K
        # output tokens/call and avoiding mid-call truncation.
        _ast_cache: list[Any] = [None]  # mutable single-cell holder

        def _state_ast_getter() -> Any:
            if _ast_cache[0] is not None:
                return _ast_cache[0]
            try:
                return state.get("current_ast")
            except Exception:  # pragma: no cover — defensive
                return None

        def _state_ast_setter(new_ast: Any) -> None:
            _ast_cache[0] = new_ast

        register_designer_tools(
            designer_registry,
            discovery=self._discovery,
            state_getter=_state_ast_getter,
            state_setter=_state_ast_setter,
            asset_getter=lambda: state.get("designer_assets"),
            # Fix D — wire ParseArchitectAstTool patch mode. BuildBlueprintTool
            # writes ``initial_blueprint`` + ``blueprint_fingerprint`` to state;
            # without these getters, ParseArchitectAstTool falls back to
            # legacy AST parsing and the architect's node_patches JSON is
            # misinterpreted as a standalone AST (the bug observed in
            # event 13 of the failing investment_research scaffold-and-run).
            blueprint_getter=lambda: state.get("initial_blueprint"),
            fingerprint_getter=lambda: state.get("blueprint_fingerprint"),
            # Fix (live run) — EmitTaskSignatureTool now writes its
            # validated payload directly to state.task_signature via this
            # setter. Without it, the classifier agent's free-form prose
            # would land in state.task_signature via the YAML output_key,
            # and downstream build_blueprint sees ``task_signature is
            # required (got None)`` because the prose isn't a dict.
            signature_setter=lambda value: state.append(
                "emit_task_signature", "task_signature", value
            ),
            # Plan v2.1 generic-robustness — expose blueprint lane_keys to
            # the architect prompt via the ``{lane_keys}`` template variable.
            # Without this, the architect cannot READ the content-derived
            # lane keys (computed in ``blueprint.compute_lane_key``) and
            # falls back to ordinal addressing for ``node_patches``, which
            # drifts across signature revisions (plan M7).
            lane_keys_setter=lambda value: state.append(
                "build_blueprint", "lane_keys", value
            ),
            # Plan v2.1 generic-robustness — surface the placeholder-pending
            # node id list as its own state key so the architect's
            # user_prompt_template can render ``{placeholder_pending_nodes}``
            # directly. Without this, the list is still in
            # ``initial_blueprint.placeholder_pending_nodes`` but buried
            # inside a multi-KB JSON blob the LLM tends to skim.
            placeholder_pending_setter=lambda value: state.append(
                "build_blueprint", "placeholder_pending_nodes", value
            ),
            # Plan v2.1 generic-robustness — wire RequestSignatureRevisionTool
            # state setters so the architect's escape valve persists. The
            # outer ``signature_loop`` (designer_workflow.yaml) reads these
            # to decide whether to re-run the classifier with the revision
            # hint. Without these closures, the tool was dead code (its
            # docstring at framework_tools.py:2255 explicitly flagged the
            # YAML loop-back as deferred PR-3 work).
            revision_count_getter=lambda: state.get("revision_count"),
            revision_count_setter=lambda value: state.append(
                "request_signature_revision", "revision_count", value
            ),
            revision_request_setter=lambda value: state.append(
                "request_signature_revision", "revision_request", value
            ),
            error_setter=lambda value: state.append(
                "request_signature_revision", "error", value
            ),
            # Intent-grounding writes back the merged designer_assets payload
            # (UI-selected ∪ LLM-grounded). All downstream consumers — the
            # classifier's user_prompt_template, build_blueprint, the architect's
            # inspect_assets / recommend_tools_for_assets tools — read
            # state.designer_assets via asset_getter, so a single in-place
            # write is sufficient.
            designer_assets_setter=lambda value: state.append(
                "emit_grounded_assets", "designer_assets", value
            ),
        )

        runner = build_app_workflow_runner(
            llm_client=framework_llm,
            # Designer tools don't go through ToolFactoryContext (they're
            # pre-registered in designer_registry); workspace/user_token
            # are unused by the Designer phase.
            workspace_client=None,
            user_token=None,
        )

        last_ast_seen: dict[str, Any] = current_ast or {}
        yielded_done = False

        try:
            async for event in runner.stream(
                workflow_def,
                state=state,
                tool_registry=designer_registry,
                strict_tool_resolution=True,
            ):
                evt_type = getattr(event, "event_type", None)

                if evt_type == "agent_stream_chunk":
                    content = getattr(event, "content", None) or ""
                    if content:
                        yield MessageEvent(content=content)

                elif evt_type == "tool_call":
                    yield ToolCallEvent(
                        tool_name=getattr(event, "tool_name", "unknown"),
                        tool_call_id=getattr(event, "tool_call_id", "") or "",
                        args=getattr(event, "arguments", {}) or {},
                    )

                elif evt_type == "tool_result":
                    # The framework ReAct loop returns an agent's tool events
                    # after the agent finishes, while designer mutation tools
                    # update the conversation-local AST cache during execution.
                    # Emitting on mutating tool results prevents the UI/test
                    # harness from persisting an earlier node snapshot when the
                    # architect made later update_block calls in the same turn.
                    tool_name = str(getattr(event, "tool_name", "") or "")
                    if tool_name in _MUTATING_DESIGNER_TOOL_NAMES:
                        node_id = getattr(event, "node_id", "") or "unknown"
                        mutation_event = _mutation_event_for_ast_change(
                            tool_name=tool_name,
                            tool_call_id=f"tool_{tool_name}_{node_id}",
                            raw_ast=_ast_cache[0],
                            last_ast_seen=last_ast_seen,
                            normalization_fixes=[],
                        )
                        if mutation_event is not None:
                            yield mutation_event
                            last_ast_seen = mutation_event.new_ast

                elif evt_type == "node_completed":
                    node_id = getattr(event, "node_id", "")
                    # After each node, surface a MutationProposedEvent if the
                    # AST has changed.
                    #
                    # AST sources (in priority order):
                    # 1. State.current_ast — populated by parse_architect_ast,
                    #    which runs Layer 2 normalize_ast (consolidations,
                    #    brace-escape, etc.). This is the CANONICAL form that
                    #    downstream nodes + the saved workflow should see.
                    # 2. _ast_cache — populated by architect's internal tool
                    #    calls (propose_workflow/update_block). Contains the
                    #    UN-normalized AST until parse_architect_ast runs.
                    #
                    # We prefer state when parse_architect_ast has produced
                    # something there, otherwise fall back to the in-flight
                    # cache (so partial-iteration emits during the architect's
                    # ReAct loop still surface).
                    state_raw: Any = None
                    try:
                        state_raw = state.get("current_ast")
                    except Exception:
                        state_raw = None
                    has_state = node_id == "parse_architect_ast" and (
                        isinstance(state_raw, dict)
                        or (
                            isinstance(state_raw, str)
                            and state_raw.strip()
                            and state_raw.strip() != "{}"
                        )
                    )
                    if has_state:
                        # parse_architect_ast wrote the normalized AST —
                        # this is canonical. Sync the cache so subsequent
                        # mutation tool calls (e.g. in the next loop iter)
                        # see the normalized version too.
                        raw: Any = state_raw
                        _ast_cache[0] = state_raw
                    else:
                        raw = _ast_cache[0] if _ast_cache[0] is not None else state_raw
                    if raw is not None:
                        # Layer 2 fix surfacing: re-derive the normalization
                        # fixes from the architect's raw message so the UI
                        # can render exactly what was auto-repaired. The
                        # parse_architect_ast tool already applied them
                        # in-place; running normalize_ast on the un-normalized
                        # source AST captures the same deterministic fix list.
                        fixes_payload = _derive_normalization_fixes(
                            state, node_id
                        )
                        mutation_event = _mutation_event_for_ast_change(
                            tool_name="propose_workflow",
                            tool_call_id=f"node_{node_id or 'unknown'}",
                            raw_ast=raw,
                            last_ast_seen=last_ast_seen,
                            normalization_fixes=fixes_payload,
                        )
                        if mutation_event is not None:
                            yield mutation_event
                            last_ast_seen = mutation_event.new_ast

                elif evt_type == "workflow_completed":
                    yield DoneEvent()
                    yielded_done = True
        except Exception as exc:
            logger.exception("DESIGNER_WORKFLOW_STREAM_FAILED")
            yield ErrorEvent(message=str(exc))

        if not yielded_done:
            yield DoneEvent()

    def check_limits(
        self,
        messages: list[dict[str, Any]],
        current_ast: dict[str, Any] | None,
        assets: list[DesignerAsset] | list[dict[str, Any]] | None = None,
    ) -> None:
        """Public alias for pre-flight size validation.

        Call this before opening the SSE stream so a RequestTooLargeError can
        be surfaced as an HTTP 413 rather than as an SSE error event.
        """
        self._check_limits(messages, current_ast, assets)

    def _check_limits(
        self,
        messages: list[dict[str, Any]],
        current_ast: dict[str, Any] | None,
        assets: list[DesignerAsset] | list[dict[str, Any]] | None = None,
    ) -> None:
        if len(messages) > MAX_MESSAGES:
            raise RequestTooLargeError(
                f"messages exceeds {MAX_MESSAGES} turns (got {len(messages)})"
            )
        ast_size = len(json.dumps(current_ast or {}, default=str).encode("utf-8"))
        if ast_size > MAX_AST_BYTES:
            raise RequestTooLargeError(f"current_ast exceeds {MAX_AST_BYTES} bytes ({ast_size})")
        msg_size = len(json.dumps(messages, default=str).encode("utf-8"))
        asset_size = len(json.dumps(assets or [], default=str).encode("utf-8"))
        total = ast_size + msg_size + asset_size
        if total > MAX_PAYLOAD_BYTES:
            raise RequestTooLargeError(f"total payload exceeds {MAX_PAYLOAD_BYTES} bytes ({total})")

    async def _dispatch(
        self,
        tc: LLMToolCall,
        ast: dict[str, Any] | None,
        user_token: str,
        current_user_id: str,
        user_intent: str = "",
        assets: list[DesignerAsset] | list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[DesignerSSEEvent, None]:
        # Tag the active designer-chat trace with the tool name. MLflow's
        # update_current_trace appends, so successive tool calls accumulate
        # — searchable as ``tags.dr.designer_tool LIKE '%propose_workflow%'``
        # without needing per-tool spans. Failures are swallowed; tracing
        # is best-effort and must never break the chat.
        try:
            import mlflow

            mlflow.update_current_trace(tags={"dr.designer_tool.latest": tc.name})
        except Exception:  # pragma: no cover - defensive
            pass

        # Validate args
        try:
            parsed = parse_tool_args(tc.name, tc.arguments)
        except KeyError:
            log_chat_mutation(tc.name, {}, 0, "error")
            yield ErrorEvent(message=f"Unknown tool: {tc.name}", tool_call_id=tc.id)
            return
        except ValidationError as exc:
            log_chat_mutation(tc.name, {}, 0, "error")
            yield ErrorEvent(message=f"Invalid args for {tc.name}: {exc}", tool_call_id=tc.id)
            return

        # Build a small args summary (keys only, no full AST values)
        _args_summary: dict[str, object] = {k: str(v)[:80] for k, v in tc.arguments.items()}

        # Tools that don't mutate AST
        if tc.name == "discover_sources":
            assert isinstance(parsed, DiscoverSourcesArgs)
            valid_source_kinds = {item["kind"] for item in _source_kinds_payload()}
            kinds: list[SourceKind] | None = (
                [k for k in parsed.kinds if k in valid_source_kinds]  # type: ignore[misc]
                if parsed.kinds is not None
                else None
            )
            try:
                resources = await self._discovery.list_for_user(
                    user_token=user_token,
                    kinds=kinds,
                    user_id=current_user_id,
                )
                log_chat_mutation(tc.name, _args_summary, 0, "success")
                yield ToolResultEvent(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result={"resources": [r.model_dump() for r in resources]},
                )
            except Exception as exc:
                logger.exception("AGENT_DESIGNER_DISCOVER_SOURCES_FAILED")
                log_chat_mutation(tc.name, _args_summary, 0, "error")
                yield ErrorEvent(message=f"discover_sources failed: {exc}", tool_call_id=tc.id)
            return

        if tc.name == "inspect_assets":
            assert isinstance(parsed, InspectAssetsArgs)
            raw_assets = parsed.assets if parsed.assets is not None else asset_context_payload(assets)
            result = inspect_assets(raw_assets)
            log_chat_mutation(tc.name, _args_summary, 0, "success")
            yield ToolResultEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result=result,
            )
            return

        if tc.name == "recommend_tools_for_assets":
            assert isinstance(parsed, RecommendToolsForAssetsArgs)
            raw_assets = parsed.assets if parsed.assets is not None else asset_context_payload(assets)
            result = recommend_tools_for_assets(raw_assets, intent=parsed.intent or user_intent)
            log_chat_mutation(tc.name, _args_summary, 0, "success")
            yield ToolResultEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result=result,
            )
            return

        if tc.name == "list_node_types":
            log_chat_mutation(tc.name, _args_summary, 0, "success")
            yield ToolResultEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result={"node_types": _node_types_payload()},
            )
            return

        if tc.name == "list_tool_kinds":
            log_chat_mutation(tc.name, _args_summary, 0, "success")
            yield ToolResultEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result={"tool_kinds": _tool_kinds_payload()},
            )
            return

        if tc.name == "list_modes":
            log_chat_mutation(tc.name, _args_summary, 0, "success")
            yield ToolResultEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result={
                    "model_tiers": _model_tiers_payload(),
                    "query_modes": _query_modes_payload(),
                    "research_depths": _research_depths_payload(),
                    "source_kinds": _source_kinds_payload(),
                },
            )
            return

        if tc.name == "validate":
            errors, summary = _validate_ast(ast or {})
            advice = _quality_advice(ast or {}) if not errors else []
            # The LLM-as-judge critic runs only when structurally valid:
            # there is no point asking the critic about a broken AST.
            # The user's intent is taken from the most recent user message
            # (the same one the Designer LLM is currently responding to).
            critique: dict[str, Any] | None = None
            if not errors:
                try:
                    critique = await _critique_ast(ast or {}, user_intent, self._llm)
                except Exception as exc:  # noqa: BLE001 — critic is advisory
                    # Critic failure must not block validate. Log and continue.
                    logger.warning(
                        "validate-tool critique failed: %s",
                        exc,
                        exc_info=True,
                    )
                    critique = None
            log_chat_mutation(
                tc.name,
                _args_summary,
                len(errors),
                "success" if not errors else "validation_failed",
            )
            yield ToolResultEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result={
                    "valid": not errors,
                    "errors": errors,
                    "summary": summary,
                    # Quality advice: per-agent specialization gaps the LLM
                    # should address via update_block / bind_tool_to_block /
                    # set_model_tier. These do NOT affect ``valid``.
                    "advice": advice,
                    # LLM-as-judge critique: structured verdict on whether
                    # the workflow as built actually answers the user's
                    # intent. ``None`` when skipped (no intent or structural
                    # errors). Verdict ``fail`` blocks save (with override);
                    # ``needs_revision`` warns; ``pass`` is clean. Does NOT
                    # affect ``valid`` — that flag is structural only.
                    "critique": critique,
                },
            )
            return

        # Mutation tools
        if tc.name == "propose_workflow":
            assert isinstance(parsed, ProposeWorkflowArgs)
            new_ast = _propose_initial_ast(parsed.intent, parsed.design_brief)
        else:
            if ast is None:
                log_chat_mutation(tc.name, _args_summary, 0, "error")
                yield ErrorEvent(
                    message=f"Cannot run {tc.name}: no current_ast (call propose_workflow first)",
                    tool_call_id=tc.id,
                )
                return
            try:
                new_ast = self._apply_mutation(tc.name, parsed, ast)
            except (mutations.BlockPathError, mutations.BlockMutationError, ValueError) as exc:
                log_chat_mutation(tc.name, _args_summary, 0, "error")
                yield ErrorEvent(message=str(exc), tool_call_id=tc.id)
                return

        errors, summary = _validate_ast(new_ast)
        log_chat_mutation(
            tc.name,
            _args_summary,
            len(errors),
            "success" if not errors else "validation_failed",
        )
        yield MutationProposedEvent(
            tool_name=tc.name,
            tool_call_id=tc.id,
            old_ast=ast,
            new_ast=new_ast,
            validation_errors=errors,
            summary=summary,
        )

    def _apply_mutation(
        self,
        tool_name: str,
        parsed: BaseModel,
        ast: dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(parsed, AddBlockArgs):
            new_ast, _ = mutations.add_block(
                ast, parsed.parent_path, parsed.node_type, parsed.config, parsed.label
            )
            return new_ast
        if isinstance(parsed, UpdateBlockArgs):
            return mutations.update_block(ast, parsed.path, parsed.patches)
        if isinstance(parsed, DeleteBlockArgs):
            return mutations.delete_block(ast, parsed.path)
        if isinstance(parsed, MoveBlockArgs):
            return mutations.move_block(ast, parsed.from_path, parsed.to_path, parsed.position)
        if isinstance(parsed, DeclareToolArgs):
            return mutations.declare_tool(ast, parsed.kind, parsed.name, parsed.config)
        if isinstance(parsed, RemoveToolArgs):
            return mutations.remove_tool(ast, parsed.name)
        if isinstance(parsed, BindToolArgs):
            return mutations.bind_tool_to_block(ast, parsed.node_path, parsed.tool_name)
        if isinstance(parsed, SetModelTierArgs):
            return mutations.set_model_tier(ast, parsed.node_path, parsed.tier)
        raise ValueError(f"Unhandled mutation: {tool_name}")
