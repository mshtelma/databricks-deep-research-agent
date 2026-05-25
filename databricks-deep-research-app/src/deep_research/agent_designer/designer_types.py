"""Pure data types and coercion helpers for the Agent Designer brief.

This module contains the Pydantic models and helper functions that represent
a WorkflowDesignBrief and its constituent parts. It has no dependency on the
YAML strategy loader or LLM prompt builder — those live in designer_architect.py.

Splitting these out allows downstream modules (workflow_builder, tools, tests)
to import the types without pulling in the full YAML-backed architect module.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_MAX_BRIEF_ITEMS = 12
_MAX_ITEM_LENGTH = 280
_MAX_LANE_SYSTEM_PROMPT_LENGTH = 3000
_MAX_LANE_USER_PROMPT_TEMPLATE_LENGTH = 4000

# Topology recipes the Designer can generate. Domain-agnostic structural
# shapes; the LLM picks the recipe per task, NOT which lanes/agents go in.
#
# ``parallel_lanes`` (legacy/missing-field default) — coordinator →
#   parallel(N static lane researchers with specialized prompts) →
#   draft synthesizer → coverage reflector → final synthesizer. Each lane
#   runs concurrently; pool-based output aggregation; no planner/router/fallback
#   brittleness. This is the recommended shape when the task has independent
#   lanes that can be investigated once and reconciled at the end.
#
# ``plan_and_execute`` (opt-in) — coordinator → plan_and_execute
#   (planner → router-over-lanes → reflector, looped with evaluator) →
#   synthesizer. Use ONLY when reflection-driven re-planning, dynamic
#   sub-question generation, or conditional lane skipping is genuinely
#   needed. Surfaces the historical router brittleness issue where the
#   planner may not stamp ``current_step.lane`` reliably — track in critic.
#
# ``single_agent`` — coordinator → one specialized agent → output. Right
#   for short factual questions where the multi-lane scaffold is overkill.
TopologyKind = Literal["parallel_lanes", "plan_and_execute", "single_agent"]
GroundingKind = Literal["reclaim", "none", "classical_lite"]


class LaneSpec(BaseModel):
    """A single research-lane record carrying the lane's description and an
    optional specialized researcher prompt the Designer LLM populates when
    calling ``propose_workflow``.

    Backwards-compatible: legacy briefs/profiles that emit ``research_lanes``
    as ``list[str]`` are coerced into ``LaneSpec(description=<str>, system_prompt="")``
    by the field_validator on both :class:`WorkflowDesignBrief` and
    :class:`DomainProfile`.
    """

    model_config = ConfigDict(extra="ignore")

    description: str = ""
    system_prompt: str = ""
    # Designer-authored per-lane researcher user prompt. When non-empty, the
    # builder threads it into the lane researcher's
    # config.user_prompt_template, fully replacing the generic
    # RESEARCHER_USER_PROMPT default. Empty is allowed for backwards
    # compatibility, but designer-generated workflows treat it as a quality
    # defect so the LLM must author use-case-specific prompts.
    user_prompt_template: str = ""

    @field_validator("description", mode="before")
    @classmethod
    def _stringify_description(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @field_validator("system_prompt", mode="before")
    @classmethod
    def _stringify_system_prompt(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @field_validator("user_prompt_template", mode="before")
    @classmethod
    def _stringify_user_prompt_template(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)


class ToolDeclarationSpec(BaseModel):
    """One runtime tool declaration chosen by the Designer LLM."""

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    kind: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    description: str = ""

    @field_validator("name", "kind", "description", mode="before")
    @classmethod
    def _stringify_optional_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @field_validator("config", mode="before")
    @classmethod
    def _coerce_config(cls, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}


class ToolBindingSpec(BaseModel):
    """Bind selected runtime tools to an agent node or node group."""

    model_config = ConfigDict(extra="ignore")

    node_id: str = ""
    tool_names: list[str] = Field(default_factory=list)

    @field_validator("node_id", mode="before")
    @classmethod
    def _stringify_node_id(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @field_validator("tool_names", mode="before")
    @classmethod
    def _coerce_tool_names(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            str(item)
            for item in value
            if isinstance(item, str) and item.strip()
        ]


class ToolPlan(BaseModel):
    """LLM-authored runtime tool plan for the generated workflow."""

    model_config = ConfigDict(extra="ignore")

    tools: list[ToolDeclarationSpec] = Field(default_factory=list)
    bindings: list[ToolBindingSpec] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("tools", mode="before")
    @classmethod
    def _coerce_tools(cls, value: Any) -> list[ToolDeclarationSpec]:
        if not isinstance(value, list):
            return []
        out: list[ToolDeclarationSpec] = []
        for item in value:
            try:
                spec = (
                    item
                    if isinstance(item, ToolDeclarationSpec)
                    else ToolDeclarationSpec.model_validate(item)
                )
            except Exception:
                continue
            if spec.name.strip() and spec.kind.strip():
                out.append(spec)
        return out

    @field_validator("bindings", mode="before")
    @classmethod
    def _coerce_bindings(cls, value: Any) -> list[ToolBindingSpec]:
        if not isinstance(value, list):
            return []
        out: list[ToolBindingSpec] = []
        for item in value:
            try:
                spec = (
                    item
                    if isinstance(item, ToolBindingSpec)
                    else ToolBindingSpec.model_validate(item)
                )
            except Exception:
                continue
            if spec.node_id.strip() and spec.tool_names:
                out.append(spec)
        return out

    @field_validator("rationale", mode="before")
    @classmethod
    def _stringify_rationale(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)


def _coerce_lane_item(item: Any) -> LaneSpec | None:
    """Coerce one raw research_lanes element into a LaneSpec.

    Accepts: str (legacy), dict (new structured shape), LaneSpec (already typed),
    None / empty / non-stringifiable garbage → returns None to filter out.
    """
    if item is None:
        return None
    if isinstance(item, LaneSpec):
        return item if item.description.strip() else None
    if isinstance(item, str):
        cleaned = item.strip()
        return LaneSpec(description=cleaned) if cleaned else None
    if isinstance(item, dict):
        try:
            spec = LaneSpec(**item)
        except Exception:
            return None
        return spec if spec.description.strip() else None
    return None


def _coerce_lane_list(value: Any) -> list[LaneSpec]:
    """Field validator helper: coerce a heterogeneous list into list[LaneSpec]."""
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    out: list[LaneSpec] = []
    for item in value:
        spec = _coerce_lane_item(item)
        if spec is not None:
            out.append(spec)
    return out


class WorkflowDesignBrief(BaseModel):
    """Structured design intent passed from Designer chat into workflow generation."""

    model_config = ConfigDict(extra="ignore")

    workflow_name: str = ""
    workflow_description: str = ""
    domain: str = ""
    user_goal: str = ""
    research_lanes: list[LaneSpec] = Field(default_factory=list)
    required_outputs: list[str] = Field(default_factory=list)
    quality_gates: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    tool_plan: ToolPlan | None = None
    topology: TopologyKind = "parallel_lanes"
    grounding_mode: GroundingKind = "reclaim"

    @field_validator("research_lanes", mode="before")
    @classmethod
    def _coerce_research_lanes(cls, value: Any) -> list[LaneSpec]:
        return _coerce_lane_list(value)

    @field_validator("topology", mode="before")
    @classmethod
    def _coerce_topology(cls, value: Any) -> TopologyKind:
        """Default missing/null/unknown topology values to ``parallel_lanes``.

        This preserves backwards-compat: legacy LLMs that emit briefs without
        a ``topology`` field, and any LLM that emits a topology string we
        don't recognize, both fall to a runnable shape. This is compatibility
        behavior, not an instruction to prefer parallel lanes; the Designer
        prompt requires the LLM to choose topology from the task structure.
        Catching unknown values here (instead of failing Pydantic validation)
        prevents a single bad token from breaking the entire chat turn.
        """
        if value is None or value == "":
            return "parallel_lanes"
        if isinstance(value, str) and value in (
            "parallel_lanes",
            "plan_and_execute",
            "single_agent",
        ):
            return value  # type: ignore[return-value]
        return "parallel_lanes"

    @field_validator("grounding_mode", mode="before")
    @classmethod
    def _coerce_grounding_mode(cls, value: Any) -> GroundingKind:
        """Default missing/null/unknown grounding mode to ``reclaim``.

        Reclaim is the safe default — strict anti-confabulation prompt at
        zero extra LLM cost vs ``none``. The Designer LLM may opt into
        ``classical_lite`` for high-assurance workflows or ``none`` for
        speed-over-accuracy use cases. Unknown values silently fall back to
        ``reclaim`` so a bad token in the brief doesn't break the chat turn.
        """
        if value is None or value == "":
            return "reclaim"
        if isinstance(value, str) and value in ("reclaim", "none", "classical_lite"):
            return value  # type: ignore[return-value]
        return "reclaim"


class DomainProfile(BaseModel):
    """Domain-specific fallback profile loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    label: str
    keywords: list[str] = Field(default_factory=list)
    workflow_name_template: str = ""
    research_lanes: list[LaneSpec] = Field(default_factory=list)
    required_outputs: list[str] = Field(default_factory=list)
    quality_gates: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)

    @field_validator("research_lanes", mode="before")
    @classmethod
    def _coerce_research_lanes(cls, value: Any) -> list[LaneSpec]:
        return _coerce_lane_list(value)


def _clean_text(value: str, *, max_length: int = 500) -> str:
    cleaned = " ".join(str(value).strip().split())
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 15].rstrip() + " ...(truncated)"


def _bounded_multiline(value: str, *, max_length: int) -> str:
    """Return a length-bounded copy of ``value`` that PRESERVES newlines.

    Used for fields where structural Markdown matters (the lane researcher
    ``user_prompt_template`` carries headings and numbered sub-question
    blocks that ``_clean_text``'s whitespace-collapse would destroy).
    Trailing whitespace per line is preserved; leading/trailing whitespace
    on the whole string is stripped.
    """
    cleaned = str(value).strip()
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 15].rstrip() + " ...(truncated)"


def _merge_lists(primary: list[str], fallback: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in [*primary, *fallback]:
        cleaned = _clean_text(item, max_length=_MAX_ITEM_LENGTH)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            merged.append(cleaned)
            seen.add(key)
        if len(merged) >= _MAX_BRIEF_ITEMS:
            break
    return merged


def _merge_lane_lists(
    primary: list[LaneSpec], fallback: list[LaneSpec]
) -> list[LaneSpec]:
    """Merge two LaneSpec lists by description (case-insensitive dedup).

    Preserves the FIRST occurrence's ``system_prompt`` and
    ``user_prompt_template`` so an LLM-supplied specialized prompt wins over
    a fallback profile's blank. Multiline ``user_prompt_template`` content
    is preserved by :func:`_bounded_multiline`.
    """
    merged: list[LaneSpec] = []
    seen: set[str] = set()
    for item in [*primary, *fallback]:
        if not isinstance(item, LaneSpec):
            spec = _coerce_lane_item(item)
            if spec is None:
                continue
        else:
            spec = item
        description = _clean_text(spec.description, max_length=_MAX_ITEM_LENGTH)
        key = description.casefold()
        if not description or key in seen:
            continue
        system_prompt = _clean_text(
            spec.system_prompt, max_length=_MAX_LANE_SYSTEM_PROMPT_LENGTH
        )
        user_prompt_template = _bounded_multiline(
            spec.user_prompt_template,
            max_length=_MAX_LANE_USER_PROMPT_TEMPLATE_LENGTH,
        )
        merged.append(
            LaneSpec(
                description=description,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
            )
        )
        seen.add(key)
        if len(merged) >= _MAX_BRIEF_ITEMS:
            break
    return merged


def _compact_list(values: list[str], *, limit: int = _MAX_BRIEF_ITEMS) -> list[str]:
    compacted: list[str] = []
    for value in values:
        cleaned = _clean_text(value, max_length=_MAX_ITEM_LENGTH)
        if cleaned:
            compacted.append(cleaned)
        if len(compacted) >= limit:
            break
    return compacted


def _compact_lane_descriptions(
    values: list[LaneSpec], *, limit: int = _MAX_BRIEF_ITEMS
) -> list[str]:
    """Render a list of LaneSpec as their cleaned description strings.

    Used for profile summaries and design-brief rendering where only the
    one-line description is shown to the LLM; the per-lane ``system_prompt``
    is consumed downstream by the workflow builder, not here.
    """
    compacted: list[str] = []
    for value in values:
        if isinstance(value, LaneSpec):
            description = value.description
        elif isinstance(value, str):
            description = value
        else:
            continue
        cleaned = _clean_text(description, max_length=_MAX_ITEM_LENGTH)
        if cleaned:
            compacted.append(cleaned)
        if len(compacted) >= limit:
            break
    return compacted


def _coerce_brief(supplied: WorkflowDesignBrief | dict[str, Any] | None) -> WorkflowDesignBrief:
    if supplied is None:
        return WorkflowDesignBrief()
    if isinstance(supplied, WorkflowDesignBrief):
        return supplied
    return WorkflowDesignBrief.model_validate(supplied)
