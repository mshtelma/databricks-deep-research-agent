"""Framework ``ResearchTool`` wrappers for the Designer's chat-time mutations.

This module implements US-08 of the harmonization plan: it exposes the
legacy designer chat tools (``propose_workflow``, ``add_block``,
``update_block``, ``bind_tool_to_block``, ``set_model_tier``, ``declare_tool``,
``discover_sources``, ``validate``) plus three new helpers
(``structural_gate`` — re-exported from :mod:`structural_gate`,
``parse_architect_ast``, ``extract_critic_approved``) as
:class:`ResearchTool` implementations so a framework workflow YAML can
invoke them via ``type: tool`` nodes.

Each tool reads its inputs from ``arguments`` (wired in by the node's
``input_mapping`` from state) and returns a ``ToolResult`` whose
``data`` dict carries any state keys the downstream nodes need (most
commonly ``current_ast``).

All mutation logic is delegated to :mod:`deep_research.agent_designer.mutations`
and :mod:`deep_research.agent_designer.workflow_builder` — this module
is a thin adapter layer, never the source of truth for AST shapes.

The new ``ParseArchitectAstTool`` is iter-2 fix #2 from the codex review:
since ``output_model`` is not enforced for agents that emit tool calls,
the architect agent's final assistant message is parsed deterministically
into a workflow AST by this tool instead of being treated as already
structured.

Registration: the framework's :class:`ToolRegistry` is instance-level
(no global registry). :func:`builtin_designer_tools` returns the canonical
list and :func:`register_designer_tools` mutates a registry in-place — the
route shim and unit tests both call the latter to make these tools
resolvable as ``type: tool`` nodes.
"""

from __future__ import annotations

import ast as py_ast
import json
import re
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from databricks_deep_research.tools.protocol import (
    ResearchTool,
    ToolContext,
    ToolDefinition,
    ToolResult,
)
from databricks_deep_research.tools.registry import ToolRegistry

from deep_research.agent_designer import mutations
from deep_research.agent_designer.ast_normalizer import (
    _escape_literal_braces,
    normalize_ast,
)
from deep_research.agent_designer.designer_types import WorkflowDesignBrief
from deep_research.agent_designer.discovery import (
    DesignerDiscoveryAdapter,
    SourceKind,
)
from deep_research.agent_designer.structural_gate import StructuralGateTool
from deep_research.agent_designer.validation_helpers import (
    _quality_advice,
    _validate_ast,
)
from deep_research.agent_designer.workflow_builder import (
    build_direct_workflow,
    build_web_research_workflow,
)

# Reads the current workflow AST from a conversation-local cache. Mutation
# tools call this instead of demanding the LLM echo back the AST as an
# argument — saves ~5K output tokens/call and eliminates mid-tool-call
# truncation when Opus hits its 8192 max_tokens cap.
#
# The cache is NOT the same as WorkflowState — during an architect ReAct
# loop, state.current_ast doesn't update between tool calls (only the
# post-agent parse_architect_ast node writes back). The cache is owned by
# the orchestrator and lives for ONE architect-agent execution.
StateGetter = Callable[[], Any]

# Writes the new AST to the conversation-local cache after a mutation
# tool succeeds. The next call's StateGetter then returns this AST.
StateSetter = Callable[[Any], None]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESEARCH_INTENT_TERMS = (
    "research",
    "web search",
    "web-search",
    "deep research",
    "deep-research",
    "summarize",
    "summary",
    "report",
    "analysis",
    "analyze",
    "investigate",
    "investigation",
    "explain",
    "synthesize",
)


def _coerce_dict(raw: Any) -> dict[str, Any]:
    """Accept either a dict or a JSON string and return a dict (empty on error)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            try:
                parsed = py_ast.literal_eval(stripped)
                return parsed if isinstance(parsed, dict) else {}
            except (SyntaxError, ValueError):
                return {}
    return {}


def _coerce_brief(raw: Any) -> WorkflowDesignBrief | None:
    """Coerce a dict or JSON string into a WorkflowDesignBrief, or None on failure."""
    if raw is None:
        return None
    if isinstance(raw, WorkflowDesignBrief):
        return raw
    payload = _coerce_dict(raw)
    if not payload:
        return None
    try:
        return WorkflowDesignBrief.model_validate(payload)
    except Exception:
        return None


def _is_research_intent(intent: str) -> bool:
    normalized = intent.casefold()
    return any(term in normalized for term in _RESEARCH_INTENT_TERMS)


def _workflow_name(intent: str) -> str:
    return intent[:80] if intent else "Untitled Agent"


def _propose_initial_ast(
    intent: str,
    design_brief: WorkflowDesignBrief | None,
) -> dict[str, Any]:
    """Local copy of ``orchestrator._propose_initial_ast``.

    Replicates the legacy chooser without importing from ``orchestrator.py``
    (which has many other unrelated dependencies the framework-tools layer
    does not need to drag in).
    """
    if _is_research_intent(intent):
        return build_web_research_workflow(
            intent, _workflow_name(intent), design_brief
        )
    return build_direct_workflow(intent, _workflow_name(intent))


def _ast_result(new_ast: dict[str, Any]) -> ToolResult:
    """Return a ToolResult whose content + data both carry the new AST."""
    return ToolResult(
        content=json.dumps(new_ast),
        data={"current_ast": new_ast},
    )


def _commit_to_cache(
    new_ast: dict[str, Any],
    state_setter: StateSetter | None,
) -> dict[str, Any]:
    """Normalize and write the new AST to the conversation-local cache.

    The NEXT mutation tool call reads this value via state_getter without the
    LLM re-emitting it. Normalizing on every write keeps deterministic safety
    defaults intact even when an LLM ``update_block`` replaces a whole config.
    """
    normalized, _ = normalize_ast(new_ast)
    if state_setter is not None:
        with suppress(Exception):  # pragma: no cover - defensive cache write
            state_setter(normalized)
    return normalized


_PROMPT_FIELDS_TO_ESCAPE = ("system_prompt", "user_prompt_template")


def _brace_escape_patches(patches: dict[str, Any]) -> dict[str, Any]:
    """Apply Layer 2 brace-escape to any prompt fields inside ``patches``.

    The architect routinely emits patches like ``{'config': {'system_prompt':
    '... {literal JSON example} ...'}}``. SafeTemplateRenderer rejects those
    at runtime. Escaping at write time guarantees that EVERY update_block
    call leaves the AST in a renderable state — even though the post-agent
    parse_architect_ast normalizer also runs on the final AST.
    """
    if not isinstance(patches, dict):
        return patches
    config = patches.get("config")
    if not isinstance(config, dict):
        return patches
    for field in _PROMPT_FIELDS_TO_ESCAPE:
        val = config.get(field)
        if not isinstance(val, str) or not val:
            continue
        escaped, _ = _escape_literal_braces(val)
        config[field] = escaped
    return patches


def _resolve_current_ast(
    arguments: dict[str, Any],
    state_getter: StateGetter | None,
) -> dict[str, Any]:
    """Resolve the AST from arguments OR from executor state.

    Mutation tools previously required the LLM to echo back the full AST as
    an argument on every call. For a 20K-char AST that wasted ~5K output
    tokens per call and caused mid-tool-call truncation when Opus hit its
    8192 max_tokens cap (the args JSON literally got cut off).

    Now ``current_ast`` is optional: if the LLM passes it, we honor it
    (back-compat); otherwise we read the latest AST from the executor's
    state via ``state_getter``. The result is the same downstream — the
    mutation is applied to the canonical AST — but the per-call cost drops
    by ~5K output tokens.
    """
    raw = arguments.get("current_ast")
    if raw is None and state_getter is not None:
        try:
            raw = state_getter()
        except Exception:  # pragma: no cover — defensive
            raw = None
    return _coerce_dict(raw)


def _error_result(message: str) -> ToolResult:
    """Return a ToolResult signalling an error without exploding the node."""
    return ToolResult(
        content=json.dumps({"error": message}),
        success=False,
        data={"error": message},
        error=message,
    )


def _coerce_critic_approved(raw: Any) -> bool:
    """Return the critic approve flag across framework output shapes.

    The critic normally returns a Pydantic ``CriticVerdict``. Depending on
    which framework boundary serialized it, the extractor may receive the
    original object, a dict, JSON, or Pydantic's repr-like string
    ``"approve=True directives=[]"``. Treat parse failures as not approved so
    malformed output cannot accidentally end the Designer loop.
    """
    if raw is None:
        return False

    approve_attr = getattr(raw, "approve", None)
    if isinstance(approve_attr, bool):
        return approve_attr

    if hasattr(raw, "model_dump"):
        with suppress(Exception):
            dumped = raw.model_dump()
            if isinstance(dumped, dict):
                return bool(dumped.get("approve"))

    verdict = _coerce_dict(raw)
    if verdict:
        return bool(verdict.get("approve"))

    if isinstance(raw, str):
        match = re.search(r"\bapprove\s*=\s*(true|false)\b", raw, re.IGNORECASE)
        if match:
            return match.group(1).casefold() == "true"

    return False


# ---------------------------------------------------------------------------
# Mutation tools
# ---------------------------------------------------------------------------


class ProposeWorkflowTool:
    """Generate the initial AST from a natural-language intent + optional brief.

    Updates the conversation-local AST cache so subsequent mutation tools
    can read it without the LLM having to echo it as an argument.
    """

    def __init__(self, state_setter: StateSetter | None = None) -> None:
        self._state_setter = state_setter

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="propose_workflow",
            description=(
                "Generate the initial workflow AST from an intent. Optional "
                "design_brief steers domain/topology/lanes selection."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": "Natural-language user intent.",
                    },
                    "design_brief": {
                        "description": "Optional WorkflowDesignBrief dict.",
                    },
                },
                "required": ["intent"],
            },
            source_type="builtin",
            source_kind="builtin",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        intent = arguments.get("intent")
        if not isinstance(intent, str) or not intent.strip():
            raise ValueError("propose_workflow requires non-empty 'intent'")
        return arguments

    async def execute(
        self, arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        intent = str(arguments.get("intent") or "")
        brief = _coerce_brief(arguments.get("design_brief"))
        try:
            new_ast = _propose_initial_ast(intent, brief)
        except Exception as exc:  # noqa: BLE001 — surface as tool error
            return _error_result(f"propose_workflow failed: {exc}")
        new_ast = _commit_to_cache(new_ast, self._state_setter)
        return _ast_result(new_ast)


class UpdateBlockTool:
    """Patch fields on an existing node (label, config, error_handling, budget).

    ``current_ast`` is OPTIONAL — the framework reads the latest AST from
    the conversation-local cache via the injected ``state_getter``. After
    the mutation succeeds, the new AST is committed back via ``state_setter``
    so the NEXT mutation tool call sees it.
    """

    def __init__(
        self,
        state_getter: StateGetter | None = None,
        state_setter: StateSetter | None = None,
    ) -> None:
        self._state_getter = state_getter
        self._state_setter = state_setter

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="update_block",
            description=(
                "Patch fields on an existing node. Allowed patch keys: label, "
                "config, error_handling, budget_seconds. "
                "DO NOT pass 'current_ast' — the framework reads it from state "
                "automatically. Passing it wastes ~5K output tokens per call "
                "and can truncate your tool call mid-stream."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Either a node 'id' (e.g. 'lane-fundamentals', "
                            "'synthesizer', 'coordinator') OR a dot-notation "
                            "indexed path (e.g. 'root.children.1.children.0'). "
                            "Prefer the id — it's stable across restructuring "
                            "and matches what propose_workflow returned."
                        ),
                    },
                    "patches": {
                        "type": "object",
                        "description": "Shallow-merged patches.",
                    },
                },
                "required": ["path", "patches"],
            },
            source_type="builtin",
            source_kind="builtin",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        for key in ("path", "patches"):
            if key not in arguments:
                raise ValueError(f"update_block requires '{key}'")
        return arguments

    async def execute(
        self, arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        ast = _resolve_current_ast(arguments, self._state_getter)
        path = str(arguments.get("path") or "")
        patches = arguments.get("patches") or {}
        if not isinstance(patches, dict):
            return _error_result("update_block 'patches' must be a dict")
        # Layer 2 brace-escape: any system_prompt / user_prompt_template the
        # architect emits gets sanitized BEFORE landing in the AST, so the
        # runner's SafeTemplateRenderer accepts it without crashes.
        patches = _brace_escape_patches(patches)
        try:
            new_ast = mutations.update_block(ast, path, patches)
        except (mutations.BlockPathError, mutations.BlockMutationError) as exc:
            return _error_result(f"update_block failed: {exc}")
        new_ast = _commit_to_cache(new_ast, self._state_setter)
        return _ast_result(new_ast)


class AddBlockTool:
    """Append a node to an existing composite node.

    ``current_ast`` is OPTIONAL — the framework reads the latest AST from
    the conversation-local cache via the injected ``state_getter``. After
    the mutation succeeds, the new AST is committed back via ``state_setter``
    so later mutation tool calls in the same architect turn see the added
    node.
    """

    def __init__(
        self,
        state_getter: StateGetter | None = None,
        state_setter: StateSetter | None = None,
    ) -> None:
        self._state_getter = state_getter
        self._state_setter = state_setter

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="add_block",
            description=(
                "Append a new node to a composite node's children list. "
                "Use this for structural additions such as adding a final "
                "coverage reflector after synthesis. DO NOT pass "
                "'current_ast' — the framework reads it from state "
                "automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "parent_path": {
                        "type": "string",
                        "description": (
                            "Parent node id (for example 'main') OR an indexed "
                            "dot path such as 'root'. For plan_and_execute "
                            "bodies, use a path ending in 'config.body'."
                        ),
                    },
                    "node_type": {
                        "type": "string",
                        "enum": [
                            "agent",
                            "tool",
                            "sequence",
                            "parallel",
                            "loop",
                            "conditional",
                            "subworkflow",
                            "plan_and_execute",
                        ],
                    },
                    "config": {
                        "type": "object",
                        "description": "Config dict for the new node.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Human-readable node label.",
                    },
                },
                "required": ["parent_path", "node_type", "config", "label"],
            },
            source_type="builtin",
            source_kind="builtin",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        for key in ("parent_path", "node_type", "config", "label"):
            if key not in arguments:
                raise ValueError(f"add_block requires '{key}'")
        return arguments

    async def execute(
        self, arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        ast = _resolve_current_ast(arguments, self._state_getter)
        parent_path = str(arguments.get("parent_path") or "")
        node_type = str(arguments.get("node_type") or "")
        config = arguments.get("config") or {}
        label = str(arguments.get("label") or "")
        if not isinstance(config, dict):
            return _error_result("add_block 'config' must be a dict")
        escaped_patch = _brace_escape_patches({"config": dict(config)})
        escaped_config = escaped_patch.get("config")
        if isinstance(escaped_config, dict):
            config = escaped_config
        try:
            new_ast, new_node_path = mutations.add_block(
                ast,
                parent_path=parent_path,
                node_type=node_type,
                config=config,
                label=label,
            )
        except (mutations.BlockPathError, mutations.BlockMutationError) as exc:
            return _error_result(f"add_block failed: {exc}")
        new_ast = _commit_to_cache(new_ast, self._state_setter)
        result = _ast_result(new_ast)
        result.data["new_node_path"] = new_node_path
        return result


class BindToolToBlockTool:
    """Bind a declared tool to an agent node's config.tools list.

    ``current_ast`` is OPTIONAL — read from cache via ``state_getter``,
    new AST committed back via ``state_setter``.
    """

    def __init__(
        self,
        state_getter: StateGetter | None = None,
        state_setter: StateSetter | None = None,
    ) -> None:
        self._state_getter = state_getter
        self._state_setter = state_setter

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bind_tool_to_block",
            description=(
                "Bind a declared tool to an agent node's config.tools list. "
                "The tool must already exist in ast['tools']. "
                "DO NOT pass 'current_ast' — read from state automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "node_path": {
                        "type": "string",
                        "description": (
                            "Node id (e.g. 'lane-fundamentals') OR dot-notation "
                            "indexed path (e.g. 'root.children.1.children.0'). "
                            "Prefer the id."
                        ),
                    },
                    "tool_name": {"type": "string"},
                },
                "required": ["node_path", "tool_name"],
            },
            source_type="builtin",
            source_kind="builtin",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        for key in ("node_path", "tool_name"):
            if key not in arguments:
                raise ValueError(f"bind_tool_to_block requires '{key}'")
        return arguments

    async def execute(
        self, arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        ast = _resolve_current_ast(arguments, self._state_getter)
        node_path = str(arguments.get("node_path") or "")
        tool_name = str(arguments.get("tool_name") or "")
        try:
            new_ast = mutations.bind_tool_to_block(ast, node_path, tool_name)
        except (mutations.BlockPathError, mutations.BlockMutationError) as exc:
            return _error_result(f"bind_tool_to_block failed: {exc}")
        new_ast = _commit_to_cache(new_ast, self._state_setter)
        return _ast_result(new_ast)


class SetModelTierTool:
    """Set ``config.model_tier`` on an agent node.

    ``current_ast`` is OPTIONAL — read from cache via ``state_getter``,
    new AST committed back via ``state_setter``.
    """

    def __init__(
        self,
        state_getter: StateGetter | None = None,
        state_setter: StateSetter | None = None,
    ) -> None:
        self._state_getter = state_getter
        self._state_setter = state_setter

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="set_model_tier",
            description=(
                "Set config.model_tier on an agent node. Tier must be one of "
                "the configured model_tiers. "
                "DO NOT pass 'current_ast' — read from state automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "node_path": {
                        "type": "string",
                        "description": (
                            "Node id (e.g. 'lane-fundamentals') OR dot-notation "
                            "indexed path (e.g. 'root.children.1.children.0'). "
                            "Prefer the id."
                        ),
                    },
                    "tier": {"type": "string"},
                },
                "required": ["node_path", "tier"],
            },
            source_type="builtin",
            source_kind="builtin",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        for key in ("node_path", "tier"):
            if key not in arguments:
                raise ValueError(f"set_model_tier requires '{key}'")
        return arguments

    async def execute(
        self, arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        ast = _resolve_current_ast(arguments, self._state_getter)
        node_path = str(arguments.get("node_path") or "")
        tier = str(arguments.get("tier") or "")
        try:
            new_ast = mutations.set_model_tier(ast, node_path, tier)
        except (mutations.BlockPathError, mutations.BlockMutationError) as exc:
            return _error_result(f"set_model_tier failed: {exc}")
        new_ast = _commit_to_cache(new_ast, self._state_setter)
        return _ast_result(new_ast)


class DeclareToolTool:
    """Append a new tool declaration to ``ast['tools']``.

    ``current_ast`` is OPTIONAL — read from cache via ``state_getter``,
    new AST committed back via ``state_setter``.
    """

    def __init__(
        self,
        state_getter: StateGetter | None = None,
        state_setter: StateSetter | None = None,
    ) -> None:
        self._state_getter = state_getter
        self._state_setter = state_setter

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="declare_tool",
            description=(
                "Add a tool declaration to the workflow's top-level tools "
                "section. The name must be unique. "
                "DO NOT pass 'current_ast' — read from state automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "name": {"type": "string"},
                    "config": {"type": "object"},
                },
                "required": ["kind", "name"],
            },
            source_type="builtin",
            source_kind="builtin",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        for key in ("kind", "name"):
            if key not in arguments:
                raise ValueError(f"declare_tool requires '{key}'")
        return arguments

    async def execute(
        self, arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        ast = _resolve_current_ast(arguments, self._state_getter)
        kind = str(arguments.get("kind") or "")
        name = str(arguments.get("name") or "")
        config = arguments.get("config") or {}
        if not isinstance(config, dict):
            config = {}
        try:
            new_ast = mutations.declare_tool(ast, kind, name, config)
        except (mutations.BlockMutationError, mutations.BlockPathError) as exc:
            return _error_result(f"declare_tool failed: {exc}")
        new_ast = _commit_to_cache(new_ast, self._state_setter)
        return _ast_result(new_ast)


# ---------------------------------------------------------------------------
# Read-only tools
# ---------------------------------------------------------------------------


class DiscoverSourcesTool:
    """Discover Databricks resources accessible to the current user.

    The framework path supplies the :class:`DesignerDiscoveryAdapter` via
    constructor DI — exactly the same way the legacy orchestrator does it.
    When no adapter is supplied, the tool degrades to an empty-resource
    payload rather than failing the workflow.
    """

    def __init__(
        self,
        discovery: DesignerDiscoveryAdapter | None = None,
    ) -> None:
        self._discovery = discovery

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="discover_sources",
            description=(
                "Discover Databricks resources the current user can access "
                "(vector indexes, Genie spaces, knowledge assistants, "
                "serving endpoints)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "kinds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional kinds filter.",
                    },
                    "user_token": {
                        "type": "string",
                        "description": "OBO token for per-user scoping.",
                    },
                    "user_id": {
                        "type": "string",
                        "description": (
                            "Authenticated user id. Required when no OBO "
                            "token is available."
                        ),
                    },
                },
            },
            source_type="builtin",
            source_kind="builtin",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return arguments

    async def execute(
        self, arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        if self._discovery is None:
            payload: dict[str, Any] = {"resources": []}
            return ToolResult(content=json.dumps(payload), data=payload)

        raw_kinds = arguments.get("kinds")
        kinds: list[SourceKind] | None
        if isinstance(raw_kinds, list):
            allowed: set[str] = {
                "vector_index",
                "genie_space",
                "knowledge_assistant",
                "serving_endpoint",
            }
            kinds = [k for k in raw_kinds if isinstance(k, str) and k in allowed]  # type: ignore[misc]
            if not kinds:
                kinds = None
        else:
            kinds = None
        user_token = str(arguments.get("user_token") or "")
        user_id = str(arguments.get("user_id") or "")
        try:
            resources = await self._discovery.list_for_user(
                user_token=user_token,
                kinds=kinds,
                user_id=user_id,
            )
        except Exception as exc:  # noqa: BLE001
            return _error_result(f"discover_sources failed: {exc}")
        payload = {"resources": [r.model_dump() for r in resources]}
        return ToolResult(content=json.dumps(payload), data=payload)


class ValidateTool:
    """Validate the AST + emit per-agent specialization advice.

    ``current_ast`` is OPTIONAL — read from state via ``state_getter``.
    """

    def __init__(self, state_getter: StateGetter | None = None) -> None:
        self._state_getter = state_getter

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="validate",
            description=(
                "Validate the current workflow AST against the framework's "
                "schema rules. Returns errors, a summary, and quality advice. "
                "DO NOT pass 'current_ast' — read from state automatically."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            source_type="builtin",
            source_kind="builtin",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return arguments

    async def execute(
        self, arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        ast = _resolve_current_ast(arguments, self._state_getter)
        errors, summary = _validate_ast(ast)
        advice = _quality_advice(ast) if not errors else []
        payload: dict[str, Any] = {
            "valid": not errors,
            "errors": errors,
            "summary": summary,
            "advice": advice,
        }
        return ToolResult(content=json.dumps(payload), data=payload)


# ---------------------------------------------------------------------------
# Iter-2 fix tools
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class ParseArchitectAstTool:
    """Codex iter-2 fix #2 — deterministically extract the architect AST.

    The architect agent's ``output_model`` is not enforced when it also emits
    tool calls, so the final assistant message arrives as free-form text
    around a JSON code block. This tool extracts that block and returns the
    parsed AST as a state-bound value (``current_ast``).
    """

    def __init__(self, state_getter: StateGetter | None = None) -> None:
        self._state_getter = state_getter

    def _cached_ast_result(
        self,
        error: str,
        *,
        parse_fallback: str = "state_cache",
    ) -> ToolResult | None:
        if self._state_getter is None:
            return None
        try:
            cached = self._state_getter()
        except Exception:  # pragma: no cover - defensive cache fallback
            return None
        if isinstance(cached, str):
            try:
                cached = json.loads(cached)
            except (TypeError, ValueError):
                return None
        if not isinstance(cached, dict) or not cached:
            return None
        normalized, fixes = normalize_ast(cached)
        payload = {
            "current_ast": normalized,
            "parse_ok": True,
            "parse_fallback": parse_fallback,
            "error": error,
            "normalization_fixes": [f.to_dict() for f in fixes],
        }
        return ToolResult(content=json.dumps(normalized), data=payload)

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="parse_architect_ast",
            description=(
                "Extract the first ```json ... ``` block from the architect "
                "agent's final assistant message and return it as the new "
                "current_ast."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "raw_message": {
                        "type": "string",
                        "description": "The architect's final assistant text.",
                    },
                },
                "required": ["raw_message"],
            },
            source_type="builtin",
            source_kind="builtin",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if "raw_message" not in arguments:
            raise ValueError("parse_architect_ast requires 'raw_message'")
        return arguments

    async def execute(
        self, arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        raw = arguments.get("raw_message")
        if not isinstance(raw, str):
            raw = "" if raw is None else str(raw)
        match = _JSON_BLOCK_RE.search(raw)
        if match is None:
            cached = self._cached_ast_result("no ```json``` block found in raw_message")
            if cached is not None:
                return cached
            payload: dict[str, Any] = {
                "current_ast": {},
                "parse_ok": False,
                "error": "no ```json``` block found in raw_message",
            }
            return ToolResult(content="{}", data=payload)
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            cached = self._cached_ast_result(f"json.loads failed: {exc}")
            if cached is not None:
                return cached
            payload = {
                "current_ast": {},
                "parse_ok": False,
                "error": f"json.loads failed: {exc}",
            }
            return ToolResult(content="{}", data=payload)
        if not isinstance(parsed, dict):
            cached = self._cached_ast_result("extracted JSON is not an object")
            if cached is not None:
                return cached
            payload = {
                "current_ast": {},
                "parse_ok": False,
                "error": "extracted JSON is not an object",
            }
            return ToolResult(content="{}", data=payload)
        # Layer 2 auto-repair: rewrite invalid identifiers (subtype, tier,
        # tool kind), auto-bind retrieval tools to researchers missing them,
        # and emit NormalizationFix records the SSE stream surfaces to the
        # UI so users see exactly what was repaired. Nothing silent.
        normalized, fixes = normalize_ast(parsed)
        if not normalized:
            cached = self._cached_ast_result("extracted JSON object was empty")
            if cached is not None:
                return cached
        cached = self._cached_ast_result(
            "state cache preferred over extracted JSON block",
            parse_fallback="state_cache_preferred",
        )
        if cached is not None and cached.data["current_ast"] != normalized:
            return cached
        payload = {
            "current_ast": normalized,
            "parse_ok": True,
            "normalization_fixes": [f.to_dict() for f in fixes],
        }
        return ToolResult(content=json.dumps(normalized), data=payload)


class ExtractCriticApprovedTool:
    """Extract the boolean ``approve`` field from the critic's verdict.

    The loop's ``until`` condition reads a flat top-level state key
    (``critic_approved``). The critic emits a structured CriticVerdict whose
    ``.approve`` field is nested; this tool flattens it so the loop's
    state-condition machinery can resolve it directly.
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="extract_critic_approved",
            description=(
                "Read the critic's structured verdict and emit a flat "
                "{critic_approved: bool} payload for the loop's until-clause."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "critic_verdict": {
                        "description": (
                            "The critic agent's output (dict or JSON string)."
                        ),
                    },
                },
                "required": ["critic_verdict"],
            },
            source_type="builtin",
            source_kind="builtin",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if "critic_verdict" not in arguments:
            raise ValueError("extract_critic_approved requires 'critic_verdict'")
        return arguments

    async def execute(
        self, arguments: dict[str, Any], _context: ToolContext
    ) -> ToolResult:
        raw = arguments.get("critic_verdict")
        approved = _coerce_critic_approved(raw)
        payload = {"critic_approved": approved}
        return ToolResult(content=json.dumps(payload), data=payload)


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------


def builtin_designer_tools(
    *,
    discovery: DesignerDiscoveryAdapter | None = None,
    state_getter: StateGetter | None = None,
    state_setter: StateSetter | None = None,
) -> list[ResearchTool]:
    """Return canonical instances of every Designer framework tool.

    The conversation-local AST cache pair (``state_getter`` + ``state_setter``)
    lets mutation tools read the latest AST from the cache and commit
    the new AST back after each successful mutation — so subsequent calls
    in the SAME architect ReAct loop see the previous mutation's result
    without the LLM having to echo the entire AST as an argument.
    """
    return [
        ProposeWorkflowTool(state_setter=state_setter),
        AddBlockTool(state_getter=state_getter, state_setter=state_setter),
        UpdateBlockTool(state_getter=state_getter, state_setter=state_setter),
        BindToolToBlockTool(state_getter=state_getter, state_setter=state_setter),
        SetModelTierTool(state_getter=state_getter, state_setter=state_setter),
        DeclareToolTool(state_getter=state_getter, state_setter=state_setter),
        DiscoverSourcesTool(discovery=discovery),
        ValidateTool(state_getter=state_getter),
        StructuralGateTool(),
        ParseArchitectAstTool(state_getter=state_getter),
        ExtractCriticApprovedTool(),
    ]


def register_designer_tools(
    registry: ToolRegistry,
    *,
    discovery: DesignerDiscoveryAdapter | None = None,
    state_getter: StateGetter | None = None,
    state_setter: StateSetter | None = None,
) -> None:
    """Register every Designer framework tool on *registry*.

    Uses ``register_external`` so the tools resolve through the same path
    enterprise tools take when ``executor.py`` carries an
    ``enterprise_tools`` list — that path is the only registration entry
    the framework currently exposes for runtime-supplied tools without
    framework-side modifications.

    Callers (the future ``api/v1/agent_designer.py`` route shim and the
    unit tests) build a registry, hand it to the executor via
    ``WorkflowExecutor(... tool_registry=registry)``, and the designer
    workflow's ``type: tool`` nodes then resolve by name.

    The ``state_getter`` closure (typically ``lambda: state.get("current_ast")``)
    lets mutation tools read the canonical AST from state instead of demanding
    the LLM echo it back as a tool argument.
    """
    for tool in builtin_designer_tools(
        discovery=discovery,
        state_getter=state_getter,
        state_setter=state_setter,
    ):
        # Register under BOTH paths so YAML refs work whether they specify
        # type=builtin (the YAML default for `ref: {name: foo}`) or
        # type=enterprise. The designer_workflow.yaml's type: tool nodes
        # use the default-builtin path; structural_gate, parse_architect_ast,
        # extract_critic_approved must resolve via builtin lookup.
        registry.register_builtin(tool.definition.name, tool)
        registry.register_external(tool.definition.name, tool)


_GLOBAL_REGISTRY: ToolRegistry | None = None


def get_global_registry() -> ToolRegistry:
    """Return a process-wide registry pre-populated with the designer tools.

    The framework's :class:`ToolRegistry` is not itself a singleton — this
    helper provides the closest equivalent without modifying framework
    code. The tools live in ``_external`` so :func:`ToolRegistry.resolve`
    accepts them via ``ToolRef(type="enterprise", name=...)`` or any of
    the other external types.

    The legacy ``ToolRef(type="builtin", ...)`` path is also supported by
    using ``register_builtin`` for the same instances.
    """
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        registry = ToolRegistry()
        for tool in builtin_designer_tools():
            registry.register_builtin(tool.definition.name, tool)
            registry.register_external(tool.definition.name, tool)
        _GLOBAL_REGISTRY = registry
    return _GLOBAL_REGISTRY


__all__ = [
    "ProposeWorkflowTool",
    "AddBlockTool",
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
