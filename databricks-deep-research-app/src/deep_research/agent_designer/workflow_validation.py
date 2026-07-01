"""Single source of truth for Designer workflow validation.

This wraps the LLM-as-judge critic (:mod:`workflow_critic`) in a content-addressed
service intended for BOTH the chat build loop and the save path, so the two
cannot disagree (the historical "split-brain": the build loop's ``CriticVerdict``
approved while the save path's ``CritiqueResult`` rejected). The save path uses it
today; wiring the build loop through it is the remaining follow-up.

Design contract (see ``.omc/plans/unify-designer-validation-gate.md``):
  * ONE judge everywhere — :func:`validate_workflow`.
  * Advisory by default; the caller decides whether a verdict blocks (only the
    save path's strict mode does, and only on ``fail``). Structural +
    deterministic-semantic validation already block at request parse.
  * Content-addressed: an unchanged workflow (same ``semantic_hash`` +
    ``intent_hash`` + ``validator_version``) reuses the cached verdict and makes
    ZERO LLM calls.
  * Fallbacks (LLM error / no intent) are NEVER cached as authoritative.

``VALIDATOR_VERSION`` MUST be bumped whenever the critic prompt, the semantic
projection, or the structural rules change — it is part of the cache key, so
bumping it transparently invalidates every stale cache row.
"""
from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from deep_research.agent_designer.critic_types import CriticDirective, CriticVerdict
from deep_research.agent_designer.workflow_critic import (
    AgentFinding,
    CoverageGap,
    CriticLLMClientProto,
    CritiqueResult,
    OutputGap,
    _extract_agents,
    _extract_tool_declarations,
    critique_workflow_against_intent_ex,
)

# Bump on ANY change to: the critic prompt, the semantic projection below, or
# the structural rules. Part of the cache key — bumping invalidates stale rows.
VALIDATOR_VERSION = "v2"


class ValidationSource(StrEnum):
    """Where a :class:`WorkflowValidationResult` came from."""

    FRESH = "fresh"        # a real LLM judgment this call
    CACHE = "cache"        # reused a previously-cached authoritative judgment
    FALLBACK = "fallback"  # LLM error / malformed output — NOT authoritative
    SKIPPED = "skipped"    # no intent / no LLM — nothing to judge


class WorkflowValidationResult(BaseModel):
    """The one validation contract shared by backend, SSE/tool-result, the save
    response body, and (mirrored) the frontend."""

    model_config = ConfigDict(extra="ignore")

    verdict: Literal["pass", "needs_revision", "fail", "skipped"]
    summary: str
    directives: list[CriticDirective] = Field(default_factory=list)
    agent_findings: list[AgentFinding] = Field(default_factory=list)
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)
    output_gaps: list[OutputGap] = Field(default_factory=list)

    # Provenance / caching.
    semantic_hash: str
    intent_hash: str
    validator_version: str
    source: ValidationSource
    cache_hit: bool = False
    cacheable: bool = False


class ValidationCacheProto(Protocol):
    """Pluggable cache keyed by ``(validator_version, intent_hash, semantic_hash)``.

    The DB-backed implementation (migration 034) is wired in by the save path;
    tests inject an in-memory fake. Implementations MUST upsert idempotently so
    concurrent same-hash saves never double-write.
    """

    async def get(
        self,
        *,
        validator_version: str,
        intent_hash: str,
        semantic_hash: str,
    ) -> WorkflowValidationResult | None: ...

    async def put(self, result: WorkflowValidationResult) -> None: ...


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def semantic_projection(
    definition: dict[str, Any],
    intent: str,
    required_outputs: list[str] | None,
) -> str:
    """Canonical JSON of exactly what the critic semantically judges.

    Mirrors the critic's own view (``_extract_agents`` /
    ``_extract_tool_declarations``) so identical critic inputs hash identically.

    Deliberately EXCLUDES tool ``config_keys`` and ``description``: those churn
    under ``CatalogService.materialize_for_save`` (which auto-fills endpoint /
    model defaults at save time) WITHOUT changing the critic's semantic judgment
    — keeping them would make a build-time hash differ from the post-materialize
    save-time hash and defeat skip-if-unchanged. Generated node ``id`` UUIDs are
    excluded implicitly (``_extract_agents`` keys on the index-based
    ``node_path``); ``label`` is retained because the critic prompt consumes it.
    """
    agents = [
        {
            "node_path": a["node_path"],
            "label": a["label"],
            "subtype": a["subtype"],
            "system_prompt": a["system_prompt_excerpt"],
            "user_prompt_template": a.get("user_prompt_template_excerpt", ""),
            "model_tier": a["model_tier"],
            "tools_bound": sorted(str(t) for t in a.get("tools_bound", [])),
        }
        for a in _extract_agents(definition)
    ]
    tools = sorted(
        ({"name": t["name"], "kind": t["kind"]} for t in _extract_tool_declarations(definition)),
        key=lambda t: (t["name"], t["kind"]),
    )
    payload = {
        "validator_version": VALIDATOR_VERSION,
        "intent": (intent or "").strip(),
        "required_outputs": sorted(required_outputs or []),
        "agents": agents,
        "tools": tools,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def compute_semantic_hash(
    definition: dict[str, Any],
    intent: str,
    required_outputs: list[str] | None = None,
) -> str:
    """Content hash of a workflow's semantic projection — the same value
    :func:`validate_workflow` computes for the cache key and stamps on the agent
    row (``last_validation_hash``).

    Exposed so the save path can cheaply answer "is this stamped verdict current
    for this definition?" (GET hydration gating + the background-task stale-race
    guard) without re-running the validator.
    """
    return _sha1(semantic_projection(definition, (intent or "").strip(), required_outputs))


def directives_from_critique(critique: CritiqueResult) -> list[CriticDirective]:
    """Flatten a :class:`CritiqueResult` into actionable revision directives.

    Shared by the save body and the build-loop adapter (:func:`to_critic_verdict`)
    so both surface the same fixes. ``fail``/``needs_revision`` agent findings are
    ``blocking``; coverage/output gaps and ``minor`` findings are ``advisory``.
    """
    directives: list[CriticDirective] = []
    for finding in critique.agent_findings:
        directives.append(
            CriticDirective(
                node_path=finding.node_path,
                issue=finding.finding,
                suggested_action=finding.suggested_action,
                severity="blocking"
                if finding.severity in ("fail", "needs_revision")
                else "advisory",
            )
        )
    for gap in critique.coverage_gaps:
        directives.append(
            CriticDirective(
                node_path="",
                issue=f"Uncovered aspect of the intent: {gap.aspect}",
                suggested_action=gap.rationale,
                severity="advisory",
            )
        )
    for out_gap in critique.output_gaps:
        directives.append(
            CriticDirective(
                node_path="",
                issue=f"Required output not producible: {out_gap.required_output}",
                suggested_action=out_gap.rationale,
                severity="advisory",
            )
        )
    return directives


def to_critic_verdict(result: WorkflowValidationResult) -> CriticVerdict:
    """Adapt the unified result into the build loop's ``CriticVerdict`` shape so
    the architect/critic loop's stop condition and revision directives are driven
    by the SAME judge as the save path. ``pass``/``skipped`` approve; everything
    else returns the directives for the next revision iteration."""
    return CriticVerdict(
        approve=result.verdict in ("pass", "skipped"),
        directives=result.directives,
    )


async def validate_workflow(
    *,
    definition: dict[str, Any],
    intent: str,
    required_outputs: list[str] | None = None,
    llm: CriticLLMClientProto | None,
    cache: ValidationCacheProto | None = None,
) -> WorkflowValidationResult:
    """Validate a workflow against the user's intent — the ONE entry point.

    Never raises and never blocks: it returns a structured verdict; the caller
    decides whether the verdict should block (only the save path's strict mode
    does). Reuses a cached authoritative verdict when the content + intent +
    validator version are unchanged (ZERO LLM calls); never caches a fallback or
    skipped result.
    """
    intent_norm = (intent or "").strip()
    semantic_hash = _sha1(semantic_projection(definition, intent_norm, required_outputs))
    intent_hash = _sha1(intent_norm)

    if not intent_norm:
        return WorkflowValidationResult(
            verdict="skipped",
            summary="No user intent recorded on the workflow — nothing to validate against.",
            semantic_hash=semantic_hash,
            intent_hash=intent_hash,
            validator_version=VALIDATOR_VERSION,
            source=ValidationSource.SKIPPED,
            cacheable=False,
        )

    if cache is not None:
        cached = await cache.get(
            validator_version=VALIDATOR_VERSION,
            intent_hash=intent_hash,
            semantic_hash=semantic_hash,
        )
        if cached is not None:
            return cached.model_copy(
                update={"source": ValidationSource.CACHE, "cache_hit": True}
            )

    if llm is None:
        return WorkflowValidationResult(
            verdict="skipped",
            summary="No LLM client available — semantic validation skipped.",
            semantic_hash=semantic_hash,
            intent_hash=intent_hash,
            validator_version=VALIDATOR_VERSION,
            source=ValidationSource.SKIPPED,
            cacheable=False,
        )

    critique, is_fallback = await critique_workflow_against_intent_ex(
        definition=definition,
        intent=intent_norm,
        required_outputs=required_outputs,
        llm=llm,
    )
    result = WorkflowValidationResult(
        verdict=critique.verdict,
        summary=critique.summary,
        directives=directives_from_critique(critique),
        agent_findings=critique.agent_findings,
        coverage_gaps=critique.coverage_gaps,
        output_gaps=critique.output_gaps,
        semantic_hash=semantic_hash,
        intent_hash=intent_hash,
        validator_version=VALIDATOR_VERSION,
        source=ValidationSource.FALLBACK if is_fallback else ValidationSource.FRESH,
        cache_hit=False,
        cacheable=not is_fallback,
    )
    if cache is not None and result.cacheable:
        await cache.put(result)
    return result


__all__ = [
    "VALIDATOR_VERSION",
    "ValidationSource",
    "WorkflowValidationResult",
    "ValidationCacheProto",
    "semantic_projection",
    "directives_from_critique",
    "to_critic_verdict",
    "validate_workflow",
]
