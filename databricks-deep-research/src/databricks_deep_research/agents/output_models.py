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


class ReflectionOutput(BaseModel):
    """Output of a reflector agent: continue / adjust / complete decision."""

    decision: Literal["continue", "adjust", "complete"]
    reasoning: str
    suggested_changes: list[str] | None = Field(default=None)
    evidence_sufficiency: Literal["sufficient", "partial", "insufficient"] | None = Field(default=None)
    failure_mode: str | None = Field(default=None)

    @field_validator("suggested_changes", mode="before")
    @classmethod
    def _normalize_suggested_changes(cls, v: Any) -> list[str]:
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


class CoordinatorOutput(BaseModel):
    """Output of the coordinator agent: query classification and routing."""

    complexity: str
    is_simple: bool = False
    recommended_depth: str = "standard"
    direct_response: str | None = None
    follow_up_type: str | None = None


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
