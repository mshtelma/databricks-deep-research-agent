"""Typed output models for agent subtypes.

Each model corresponds to the structured JSON output produced by a
particular agent subtype.  The workflow engine validates raw LLM output
against these schemas when ``output_mode`` is ``"structured"`` or
``"json"``.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SourceHintOutput(BaseModel):
    """Planner hint describing how a step should use a specific source."""

    source_name: str
    source_type: str
    priority: int = Field(default=2, ge=1, le=3)
    query_hint: str | None = None
    query_strategy: str | None = None
    reasoning: str = ""


class PlanStepOutput(BaseModel):
    """A single planner step with optional source routing hints."""

    id: str
    title: str
    description: str = ""
    step_type: Literal["research", "analysis"] = "research"
    needs_search: bool = True
    source_hints: list[SourceHintOutput] = Field(default_factory=list)
    exclude_sources: list[str] = Field(default_factory=list)
    # Optional per-step researcher user prompt: when populated, the
    # plan_and_execute runtime injects it into the body researcher's
    # config.user_prompt_template for the duration of this step, replacing
    # the generic default. None preserves today's behavior. See
    # workflow/runtime/plan_execute_runner.py for the injection point.
    user_prompt_template: str | None = None

    @field_validator("user_prompt_template", mode="before")
    @classmethod
    def _normalize_user_prompt_template(cls, v: Any) -> str | None:
        if v is None:
            return None
        text = str(v).strip()
        return text or None

    @field_validator("source_hints", mode="before")
    @classmethod
    def _normalize_source_hints(cls, v: Any) -> list[Any]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return []

    @field_validator("exclude_sources", mode="before")
    @classmethod
    def _normalize_exclude_sources(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(item) for item in v]
        return []


class PlanOutput(BaseModel):
    """Output of a planner agent: a research plan with ordered steps."""

    title: str
    thought: str
    steps: list[PlanStepOutput]
    has_enough_context: bool = False
    iteration: int = 1

    @field_validator("steps", mode="before")
    @classmethod
    def _normalize_steps(cls, v: Any) -> list[Any]:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                return []
            if isinstance(parsed, list):
                return parsed
            return []
        if isinstance(v, list):
            return v
        return []


class ReflectionDirective(BaseModel):
    """One structured, actionable directive emitted by a coverage reflector.

    Consumed by the next synthesis pass via the ``RevisionContext``
    abstraction. Designed to be machine-actionable end-to-end: each
    directive names a section, states what's wrong, and prescribes the
    shortest action to address it. A synthesizer can iterate the
    directives list and answer each one in a 1:1 ``DIRECTIVE RESPONSES``
    accountability table.

    Severities:
      * ``critical`` — blocks publication (broken structure, contradicted fact).
      * ``major``    — report should not ship as-is (missing required section,
                       stale value).
      * ``minor``    — polish.
    """

    severity: Literal["critical", "major", "minor"]
    section: str = Field(min_length=1, max_length=200)
    issue: str = Field(min_length=1, max_length=600)
    fix: str = Field(min_length=1, max_length=600)


class ReflectionOutput(BaseModel):
    """Output of a reflector agent: continue / adjust / complete decision.

    The ``directives`` field is the machine-actionable channel that drives a
    downstream synthesizer's revision pass. It defaults to an empty list so
    older reflectors that have not yet been re-prompted continue to validate
    without modification — but a ``model_validator`` warns when a reflector
    emits ``decision="adjust"`` with no directives (a strong signal the
    reflector's schema-compliance has regressed).
    """

    decision: Literal["continue", "adjust", "complete"]
    reasoning: str
    suggested_changes: list[str] | None = Field(default=None)
    evidence_sufficiency: Literal["sufficient", "partial", "insufficient"] | None = Field(default=None)
    failure_mode: str | None = Field(default=None)
    directives: list[ReflectionDirective] = Field(
        default_factory=list,
        description=(
            "Structured, machine-actionable directives the next synthesis "
            "pass MUST address. Required (non-empty) when decision='adjust'; "
            "optional otherwise."
        ),
    )

    @field_validator("suggested_changes", mode="before")
    @classmethod
    def _normalize_suggested_changes(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return []

    @field_validator("directives", mode="before")
    @classmethod
    def _normalize_directives(cls, v: Any) -> list[Any]:
        """Coerce ``None`` and bad shapes to an empty list rather than raising.

        Keeps backward compatibility with reflectors that emit JSON missing
        the ``directives`` key (treated as empty) or send the wrong type
        (treated as empty; warning logged at the caller's stage 8 boundary).
        """
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return []


class EvaluationOutput(BaseModel):
    """Output of an evaluator agent: continue / replan / complete decision."""

    decision: Literal["continue", "replan", "complete"]
    reasoning: str
    suggested_changes: list[str] | None = Field(default=None)

    @field_validator("suggested_changes", mode="before")
    @classmethod
    def _normalize_suggested_changes(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return []


class ExtractedScope(BaseModel):
    """Structured query scope extracted by the coordinator and forwarded to lanes.

    Lets each lane researcher skip the entity-extraction round (which would
    otherwise burn 1-2 search calls per lane re-deriving that "NVDA" means
    NVIDIA Corp). All fields are optional — when the coordinator cannot
    confidently extract scope, lanes fall back to deriving it from the raw
    query the same way they did before this field existed.
    """

    entities: list[str] = Field(default_factory=list)
    time_window: str | None = None
    comparables: list[str] = Field(default_factory=list)
    domain_hints: list[str] = Field(default_factory=list)


class CoordinatorOutput(BaseModel):
    """Output of the coordinator agent: query classification and routing."""

    complexity: str
    is_simple: bool = False
    recommended_depth: str = "standard"
    direct_response: str | None = None
    follow_up_type: str | None = None
    # Optional structured scope forwarded to downstream lanes. None when the
    # query is conversational or scope cannot be reliably inferred.
    extracted_scope: ExtractedScope | None = None


class ResearcherOutput(BaseModel):
    """Output of a researcher agent: findings from a single research step."""

    search_queries: list[str] = Field(default_factory=list)
    observation: str = ""
    key_points: list[str] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)
    research_status: Literal["ok", "blocked", "insufficient_data"] = "ok"
    blocking_reason: str | None = None
    findings: str = ""
    sources_found: int = 0

    @field_validator("findings", mode="before")
    @classmethod
    def _normalize_findings(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)

    @field_validator("observation", mode="before")
    @classmethod
    def _normalize_observation(cls, v: Any, info: Any) -> str:
        if v is None:
            data = getattr(info, "data", {}) or {}
            if isinstance(data, dict):
                return str(data.get("findings", "") or "")
            return ""
        return str(v)


class SynthesizerOutput(BaseModel):
    """Output of the synthesizer agent: the final research report."""

    report: str
    structured_output: Any | None = None


class BackgroundOutput(BaseModel):
    """Output of the background-research agent: initial data landscape."""

    data_landscape: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    query_decomposition: list[str] = Field(default_factory=list)
    discovered_sources: list[dict[str, Any]] = Field(default_factory=list)
