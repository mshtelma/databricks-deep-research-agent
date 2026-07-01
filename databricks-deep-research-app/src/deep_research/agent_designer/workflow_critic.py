"""LLM-as-judge critic for Designer-generated workflows.

The structural validator (``workflow.loader.load_workflow_from_dict``) and the
heuristic check (``semantic_validation.detect_unspecialized_agents``) catch
structural problems and obvious laziness, but neither asks the semantic
question: **given this user intent, does this workflow as built actually
answer the user's request?**

This module adds that check via a single LLM call. The critic returns a
structured verdict the orchestrator surfaces as advice during chat and the
agents_v2 save path enforces as a soft gate (block on ``fail`` only, with
explicit ``?force=true`` override; ``needs_revision`` warns but saves).

The critic reuses the chat orchestrator's injected ``LLMClientProto`` — no
new LLM client wiring required. Cost: one streaming request per invocation,
constrained-decoded against the ``emit_critique`` tool schema.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


# ---- Result types (constrained-decoded by the LLM into JSON) -----------------


class AgentFinding(BaseModel):
    """A per-agent semantic-fit finding."""

    model_config = ConfigDict(extra="ignore")

    node_path: str = Field(
        description="Dot-string path to the agent node, e.g. 'root.children[0]'.",
    )
    label: str = Field(
        description="Human-readable agent label, taken from the AST.",
    )
    severity: Literal["fail", "needs_revision", "minor"] = Field(
        description=(
            "fail — agent fundamentally does not address any aspect of the "
            "user's intent. needs_revision — agent partially addresses the "
            "intent but is shallow or off-target. minor — agent is "
            "essentially correct but could be sharpened."
        ),
    )
    finding: str = Field(
        description="One or two sentences explaining the semantic-fit issue.",
    )
    suggested_action: str = Field(
        description=(
            "Which Designer tool call (update_block / add_block / "
            "bind_tool_to_block / set_model_tier / delete_block / "
            "move_block) the LLM should issue next to fix this, with a "
            "short hint at what to change."
        ),
    )


class CoverageGap(BaseModel):
    """An aspect of the user's intent that no agent in the workflow addresses."""

    model_config = ConfigDict(extra="ignore")

    aspect: str = Field(description="The uncovered aspect of the user's intent.")
    rationale: str = Field(
        description="Why this aspect is missing from the workflow as built.",
    )


class OutputGap(BaseModel):
    """A ``required_output`` from the brief that the workflow cannot produce."""

    model_config = ConfigDict(extra="ignore")

    required_output: str = Field(
        description="The verbatim entry from brief.required_outputs.",
    )
    rationale: str = Field(
        description="Why the workflow as built cannot produce this output.",
    )


class CritiqueResult(BaseModel):
    """Structured verdict from the workflow critic."""

    model_config = ConfigDict(extra="ignore")

    verdict: Literal["pass", "needs_revision", "fail"]
    summary: str = Field(
        description=(
            "One or two sentences stating whether the workflow answers the "
            "user's intent, and what the dominant gap is if any."
        ),
    )
    agent_findings: list[AgentFinding] = Field(default_factory=list)
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)
    output_gaps: list[OutputGap] = Field(default_factory=list)


# ---- Critic LLM client protocol (mirror of orchestrator.LLMClientProto) -----


class _LLMStreamChunkProto(Protocol):
    content: str | None
    tool_call: Any
    finish: bool


class CriticLLMClientProto(Protocol):
    """Structural protocol — mirror of ``orchestrator.LLMClientProto``.

    Declared here to avoid a circular import (orchestrator imports this module
    for ``_critique_ast``). Any object that implements ``stream(messages,
    tools)`` returning an async iterator of stream chunks with optional
    ``tool_call`` attribute satisfies this protocol.
    """

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any: ...


# ---- AST walking ------------------------------------------------------------


def _extract_agents(
    definition: dict[str, Any],
) -> list[dict[str, Any]]:
    """Walk the definition and return one record per agent node.

    Each record: {node_path, label, subtype, system_prompt_excerpt,
    tools_bound, model_tier}. Used as the critic's view of the workflow.
    """
    agents: list[dict[str, Any]] = []

    def _excerpt(prompt: str, limit: int = 1500) -> str:
        prompt = (prompt or "").strip()
        if len(prompt) <= limit:
            return prompt
        return prompt[: limit - 15].rstrip() + "...(truncated)"

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        config = node.get("config") or {}
        if not isinstance(config, dict):
            config = {}
        if node.get("type") == "agent":
            agents.append(
                {
                    "node_path": path,
                    "label": node.get("label") or node.get("id") or "agent",
                    "subtype": str(config.get("subtype") or "agent"),
                    "system_prompt_excerpt": _excerpt(
                        str(config.get("system_prompt") or "")
                    ),
                    "user_prompt_template_excerpt": _excerpt(
                        str(config.get("user_prompt_template") or "")
                    ),
                    "tools_bound": list(config.get("tools") or []),
                    "model_tier": str(config.get("model_tier") or ""),
                }
            )
        if node.get("type") == "plan_and_execute":
            for nested_key in ("planner", "evaluator"):
                nested = config.get(nested_key)
                if isinstance(nested, dict):
                    nested_path = f"{path}.config.{nested_key}"
                    agents.append(
                        {
                            "node_path": nested_path,
                            "label": nested.get("label")
                            or nested.get("id")
                            or nested_key,
                            "subtype": str(nested.get("subtype") or nested_key),
                            "system_prompt_excerpt": _excerpt(
                                str(nested.get("system_prompt") or "")
                            ),
                            "user_prompt_template_excerpt": _excerpt(
                                str(nested.get("user_prompt_template") or "")
                            ),
                            "tools_bound": list(nested.get("tools") or []),
                            "model_tier": str(nested.get("model_tier") or ""),
                        }
                    )
            body = config.get("body")
            if isinstance(body, dict):
                walk(body, f"{path}.config.body")
        for idx, child in enumerate(node.get("children", []) or []):
            walk(child, f"{path}.children[{idx}]")

    walk(definition.get("root"), "root")
    return agents


def _extract_tool_declarations(definition: dict[str, Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in definition.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        tools.append(
            {
                "name": str(tool.get("name") or ""),
                "kind": str(tool.get("kind") or ""),
                "description": str(tool.get("description") or "")[:500],
                "config_keys": sorted(
                    str(key)
                    for key in (tool.get("config") or {})
                    if isinstance(tool.get("config"), dict)
                ),
            }
        )
    return tools


# ---- Tool schema (constrained-decoded by the LLM) ---------------------------


def _critique_tool_schema() -> dict[str, Any]:
    """Return the function-call tool schema for the critic LLM.

    The LLM emits exactly one tool call to ``emit_critique`` with arguments
    matching ``CritiqueResult``'s JSON schema. Anthropic's tool-use enforces
    the schema at the API boundary, so we get constrained-decoded output.
    """
    return {
        "type": "function",
        "function": {
            "name": "emit_critique",
            "description": (
                "Emit your structured critique of the workflow. Call this "
                "tool exactly once."
            ),
            "parameters": CritiqueResult.model_json_schema(),
        },
    }


# ---- Prompt construction ----------------------------------------------------


_CRITIC_SYSTEM_PROMPT = """You are a Workflow Critic. You evaluate a multi-agent research workflow that another LLM (the Designer) just built, and judge whether the workflow as built actually addresses the user's request.

You will be given:
  - The user's original intent (the request that triggered the Designer).
  - The brief's required_outputs (what the final report must contain).
  - The workflow's runtime tool declarations.
  - A list of every agent in the generated workflow with its label, subtype, bound tools, model_tier, and the first ~1500 chars of BOTH its system_prompt and user_prompt_template (judge prompt content using both).

Your job is to emit ONE call to the `emit_critique` tool with a structured verdict:
  - verdict = "pass"           — every agent's system_prompt addresses some aspect of the intent; required_outputs covered; workflow as built will produce a useful answer to the user's request.
  - verdict = "needs_revision" — some agents are off-topic, shallow, or duplicative; some required_outputs are not addressed by any agent. The workflow is recoverable: each problem can be fixed with a Designer tool call (update_block to rewrite a system_prompt, add_block to add a missing aspect, bind_tool_to_block to add a missing data source).
  - verdict = "fail"           — the workflow fundamentally does not answer the user's request (wrong domain, missing core lanes, all agents on generic boilerplate). The Designer must redesign, not patch.

Be strict but fair:
  - An agent whose system_prompt is just generic researcher methodology (e.g., starts with "You are the Researcher agent for a deep research system" and never names the user's actual topic or any concrete data points/metrics/sub-questions) → FAIL severity at the agent level.
  - An agent whose system_prompt names the right domain but lacks the 4 elements (what to investigate / what to cite / what to flag / what NOT to do) → NEEDS_REVISION.
  - An agent whose system_prompt covers all 4 elements with topic-specific content → no finding.

For each required_output in the brief, check whether at least one agent's system_prompt is plausibly tasked with producing that output's content (e.g., a "competitive benchmark table" requires an agent that investigates competitors with comparable metrics).

Tool adequacy is part of semantic fit:
  - A researcher or answer agent expected to gather evidence must have at least one bound runtime tool.
  - The chosen tools must plausibly access the evidence source named by the user's intent and the agent prompts.
  - If the user asks for exact table values, totals, counts, row-level lookups, or numeric computation, vector/web-only tooling is not enough unless another bound tool can read tables or compute.
  - If the user requests private/corpus/vector/table assets, globally declared tools are not enough; compatible tools must be bound to the agent that needs them.
  - Unused, stale, duplicated, or unrelated runtime tool declarations should trigger a revision directive to remove or rebind them.
  - Approve only when the workflow has a concrete evidence path: user query -> tool calls -> observations/sources -> synthesis.

Be concise. Emit at most 8 agent_findings, 4 coverage_gaps, 4 output_gaps. Pick the most consequential ones."""


def _build_critic_messages(
    *,
    intent: str,
    required_outputs: list[str],
    tool_declarations: list[dict[str, Any]],
    agents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    user_content = json.dumps(
        {
            "intent": intent,
            "required_outputs": required_outputs,
            "tool_declarations": tool_declarations,
            "agents": agents,
        },
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---- Parsing the critic's tool call ----------------------------------------


def _parse_tool_call_args(raw: Any) -> dict[str, Any]:
    """The protocol's ``LLMToolCall.arguments`` may be a dict or a JSON
    string depending on the provider; normalize to dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"critic tool-call args not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"critic tool-call args must be an object, got {type(parsed).__name__}"
            )
        return parsed
    raise ValueError(
        f"critic tool-call args has unexpected type {type(raw).__name__}"
    )


def _fallback_critique(reason: str) -> CritiqueResult:
    """Construct a non-blocking placeholder verdict when the critic call
    fails (LLM error, malformed output, etc.). ``needs_revision`` is the
    correct fallback: it surfaces a warning to the LLM/UI without blocking
    save."""
    logger.warning("workflow critic fallback: %s", reason)
    return CritiqueResult(
        verdict="needs_revision",
        summary=(
            f"Critic LLM call did not complete cleanly ({reason}). Treating "
            "as needs_revision so the workflow surfaces for human review "
            "without being hard-blocked."
        ),
    )


# ---- Public API -------------------------------------------------------------


async def critique_workflow_against_intent_ex(
    *,
    definition: dict[str, Any],
    intent: str,
    required_outputs: list[str] | None = None,
    llm: CriticLLMClientProto,
) -> tuple[CritiqueResult, bool]:
    """Like :func:`critique_workflow_against_intent` but also returns an
    ``is_fallback`` flag.

    ``is_fallback`` is True whenever the returned verdict is a non-authoritative
    placeholder (empty intent, malformed AST, no agent nodes, LLM stream error,
    or invalid critic output) rather than a real judgment. Callers that cache
    the verdict MUST NOT persist a fallback as an authoritative result.
    """
    if not intent or not intent.strip():
        return _fallback_critique("empty intent — no semantic question to judge"), True
    if not isinstance(definition, dict) or "root" not in definition:
        return _fallback_critique("missing or malformed definition.root"), True

    agents = _extract_agents(definition)
    if not agents:
        return _fallback_critique("no agent nodes found in workflow"), True

    messages = _build_critic_messages(
        intent=intent,
        required_outputs=required_outputs or [],
        tool_declarations=_extract_tool_declarations(definition),
        agents=agents,
    )
    tool = _critique_tool_schema()

    tool_call_args: Any | None = None
    try:
        async for chunk in llm.stream(messages, [tool]):
            call = getattr(chunk, "tool_call", None)
            if call is not None and getattr(call, "name", None) == "emit_critique":
                tool_call_args = getattr(call, "arguments", None)
                break
            if getattr(chunk, "finish", False):
                break
    except Exception as exc:  # noqa: BLE001 — LLM clients raise heterogeneously
        return _fallback_critique(f"LLM stream raised: {exc}"), True

    if tool_call_args is None:
        return _fallback_critique("critic did not call emit_critique"), True

    try:
        parsed = _parse_tool_call_args(tool_call_args)
        return CritiqueResult.model_validate(parsed), False
    except (ValueError, ValidationError) as exc:
        return _fallback_critique(f"critic output failed validation: {exc}"), True


async def critique_workflow_against_intent(
    *,
    definition: dict[str, Any],
    intent: str,
    required_outputs: list[str] | None = None,
    llm: CriticLLMClientProto,
) -> CritiqueResult:
    """Run the LLM-as-judge critic against a generated workflow.

    Returns a structured :class:`CritiqueResult`. Callers decide what to do
    with the verdict:
      * Chat orchestrator: surface as ``advice`` in the ``validate`` tool result.
      * Save path: advisory by default; strict mode blocks on ``verdict ==
        "fail"`` unless ``?force=true``.
    Thin back-compat wrapper over
    :func:`critique_workflow_against_intent_ex` (drops the ``is_fallback`` flag).
    """
    result, _is_fallback = await critique_workflow_against_intent_ex(
        definition=definition,
        intent=intent,
        required_outputs=required_outputs,
        llm=llm,
    )
    return result


__all__ = [
    "AgentFinding",
    "CoverageGap",
    "OutputGap",
    "CritiqueResult",
    "CriticLLMClientProto",
    "critique_workflow_against_intent",
    "critique_workflow_against_intent_ex",
]
