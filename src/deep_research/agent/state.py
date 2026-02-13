"""ResearchState model for multi-agent workflow."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from deep_research.core.logging_utils import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from deep_research.schemas.source_scope import SourceScopeConfig


class StepType(str, Enum):
    """Type of research plan step."""

    RESEARCH = "research"  # Web search/crawl - executed by Researcher
    ANALYSIS = "analysis"  # Pure reasoning - executed by Synthesizer


class StepStatus(str, Enum):
    """Execution status of a plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class ReflectionDecision(str, Enum):
    """Decision made by Reflector agent."""

    CONTINUE = "continue"  # Proceed to next step
    ADJUST = "adjust"  # Return to Planner for replanning
    COMPLETE = "complete"  # Skip remaining steps, go to Synthesizer


@dataclass
class PlanStep:
    """A single step in a research plan."""

    id: str
    title: str
    description: str
    step_type: StepType
    needs_search: bool
    status: StepStatus = StepStatus.PENDING
    observation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "step_type": self.step_type.value,
            "needs_search": self.needs_search,
            "status": self.status.value,
            "observation": self.observation,
        }


@dataclass
class Plan:
    """A structured research plan created by Planner agent."""

    id: str
    title: str
    thought: str
    steps: list[PlanStep]
    has_enough_context: bool = False
    iteration: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "thought": self.thought,
            "steps": [s.to_dict() for s in self.steps],
            "has_enough_context": self.has_enough_context,
            "iteration": self.iteration,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ReflectionResult:
    """Output from the Reflector agent."""

    decision: ReflectionDecision
    reasoning: str
    suggested_changes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.decision.value,
            "reasoning": self.reasoning,
            "suggested_changes": self.suggested_changes,
        }


@dataclass
class QueryClassification:
    """Result of analyzing a user query."""

    complexity: str  # simple, moderate, complex
    follow_up_type: str  # new_topic, clarification, complex_follow_up
    is_ambiguous: bool
    clarifying_questions: list[str] = field(default_factory=list)
    recommended_depth: str = "auto"
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "complexity": self.complexity,
            "follow_up_type": self.follow_up_type,
            "is_ambiguous": self.is_ambiguous,
            "clarifying_questions": self.clarifying_questions,
            "recommended_depth": self.recommended_depth,
            "reasoning": self.reasoning,
        }


ENTERPRISE_URL_PREFIXES = ("genie://", "vs://", "enterprise://", "ka://")
"""URL prefixes used by enterprise sources (backward compat fallback classifier)."""


@dataclass
class SourceInfo:
    """Information about a research source (web or enterprise)."""

    url: str
    title: str | None = None
    snippet: str | None = None
    content: str | None = None
    relevance_score: float | None = None
    source_type: str = "web"
    """Source type: 'genie', 'vector_search', 'knowledge_assistant', 'web', 'file'."""
    # Extended fields for citation verification
    total_pages: int | None = None
    detected_sections: list[str] | None = None
    content_type: str | None = None

    @property
    def is_enterprise(self) -> bool:
        """Check if this source is from an enterprise data source."""
        return self.source_type in ("genie", "vector_search", "knowledge_assistant")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "source_type": self.source_type,
            "is_enterprise": self.is_enterprise,
            "total_pages": self.total_pages,
            "detected_sections": self.detected_sections,
            "content_type": self.content_type,
        }


@dataclass
class EvidenceInfo:
    """Pre-selected evidence span for citation verification.

    Created during Stage 1 (Evidence Pre-Selection) of the citation pipeline.
    """

    source_url: str
    quote_text: str
    start_offset: int | None = None
    end_offset: int | None = None
    section_heading: str | None = None
    relevance_score: float | None = None
    has_numeric_content: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_url": self.source_url,
            "quote_text": self.quote_text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "section_heading": self.section_heading,
            "relevance_score": self.relevance_score,
            "has_numeric_content": self.has_numeric_content,
        }


@dataclass
class ClaimInfo:
    """Atomic claim extracted from generated content.

    Created during Stage 2 (Interleaved Generation) of the citation pipeline.
    """

    claim_text: str
    claim_type: str  # "general" or "numeric"
    position_start: int
    position_end: int
    evidence: EvidenceInfo | None = None
    confidence_level: str | None = None  # "high", "medium", "low"
    verification_verdict: str | None = None  # "supported", "partial", "unsupported", "contradicted"
    verification_reasoning: str | None = None
    abstained: bool = False
    citation_key: str | None = None  # Primary key like "Arxiv", "Zhipu"
    citation_keys: list[str] | None = None  # All keys for multi-marker sentences
    from_free_block: bool = False  # True if extracted from <free> block (needs verification)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim_text": self.claim_text,
            "claim_type": self.claim_type,
            "position_start": self.position_start,
            "position_end": self.position_end,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "confidence_level": self.confidence_level,
            "verification_verdict": self.verification_verdict,
            "verification_reasoning": self.verification_reasoning,
            "abstained": self.abstained,
            "citation_key": self.citation_key,
            "citation_keys": self.citation_keys,
            "from_free_block": self.from_free_block,
        }


@dataclass
class VerificationSummaryInfo:
    """Summary of verification results for a message.

    Created after Stage 4 (Isolated Verification) completes.
    Updated with Stage 7 metrics after ARE-style verification.
    """

    total_claims: int = 0
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    abstained_count: int = 0
    unsupported_rate: float = 0.0
    contradicted_rate: float = 0.0
    warning: bool = False
    citation_corrections: int = 0

    # Stage 7: ARE-style Verification Retrieval metrics
    claim_revisions: int = 0  # Number of claims revised by Stage 7
    atomic_facts_total: int = 0  # Total atomic facts decomposed
    atomic_facts_verified: int = 0  # Facts verified with evidence
    atomic_facts_softened: int = 0  # Facts softened (no evidence)
    claims_fully_verified: int = 0  # Claims where all facts verified
    claims_partially_softened: int = 0  # Claims with mixed verified/softened
    claims_fully_softened: int = 0  # Claims where all facts softened
    external_searches: int = 0  # Brave searches performed
    new_sources_added: int = 0  # New sources discovered during Stage 7

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_claims": self.total_claims,
            "supported_count": self.supported_count,
            "partial_count": self.partial_count,
            "unsupported_count": self.unsupported_count,
            "contradicted_count": self.contradicted_count,
            "abstained_count": self.abstained_count,
            "unsupported_rate": self.unsupported_rate,
            "contradicted_rate": self.contradicted_rate,
            "warning": self.warning,
            "citation_corrections": self.citation_corrections,
            # Stage 7 metrics
            "claim_revisions": self.claim_revisions,
            "atomic_facts_total": self.atomic_facts_total,
            "atomic_facts_verified": self.atomic_facts_verified,
            "atomic_facts_softened": self.atomic_facts_softened,
            "claims_fully_verified": self.claims_fully_verified,
            "claims_partially_softened": self.claims_partially_softened,
            "claims_fully_softened": self.claims_fully_softened,
            "external_searches": self.external_searches,
            "new_sources_added": self.new_sources_added,
        }


class ResearchDepth(str, Enum):
    """Research depth levels controlling thoroughness."""

    AUTO = "auto"  # Automatically determined based on query complexity
    LIGHT = "light"  # 1-2 search iterations, quick answers
    MEDIUM = "medium"  # 3-5 search iterations, balanced research
    EXTENDED = "extended"  # 6-10 search iterations, thorough analysis


class WorkflowMode(str, Enum):
    """Workflow mode controlling how research steps are determined.

    Part of 007-enterprise-data-sources feature (T051).
    """

    PLANNER = "planner"
    """Let the AI planner determine research steps (default)."""

    MANUAL = "manual"
    """User defines all steps explicitly, bypass planner."""

    HYBRID = "hybrid"
    """User defines initial steps, planner can add more."""


# Mapping from query complexity to default depth
COMPLEXITY_TO_DEPTH: dict[str, str] = {
    "simple": "light",
    "moderate": "medium",
    "complex": "extended",
}


@dataclass
class ResearchState:
    """Runtime state for multi-agent research workflow.

    Passed between agents during execution.
    """

    # Original query context
    query: str
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    session_id: UUID = field(default_factory=uuid4)

    # User preferences
    system_instructions: str | None = None  # Custom instructions from user preferences

    # Query mode configuration (tiered query modes feature)
    query_mode: str = "deep_research"  # simple, web_search, deep_research

    # Research depth configuration (only applies to deep_research mode)
    research_depth: str = "auto"  # auto, light, medium, extended
    effective_depth: str | None = None  # Resolved depth after auto selection

    # Clarification (Coordinator phase)
    enable_clarification: bool = True
    clarification_rounds: int = 0
    max_clarification_rounds: int = 3
    clarification_history: list[str] = field(default_factory=list)
    is_clarification_complete: bool = False

    # Query classification
    query_classification: QueryClassification | None = None
    is_simple_query: bool = False
    direct_response: str | None = None

    # Background investigation (pre-planning)
    background_investigation_results: str = ""

    # =========================================================================
    # Enterprise Data Source Discovery (007-enterprise-data-sources, T035)
    # =========================================================================

    # Source scope configuration (controls which sources are available)
    source_scope_config: SourceScopeConfig | None = None

    # Data landscape from background discovery (US10)
    data_landscape: Any | None = None  # DataLandscape from schemas

    # Per-source query tracking for budgeting
    source_query_counts: dict[str, int] = field(default_factory=dict)

    # Per-source results for attribution
    source_results: dict[str, list[SourceInfo]] = field(default_factory=dict)

    # User's OBO token for enterprise data source access
    user_token: str | None = None

    # Enterprise tools loaded from user data sources (007-enterprise-data-sources Phase 2)
    # Populated by orchestrator from get_enabled_tools_for_user()
    enterprise_tools: list[Any] = field(default_factory=list)
    """Enterprise tools (GenieTool, UserVectorSearchTool) loaded from user data sources."""

    source_quality_history: dict[str, list[str]] = field(default_factory=dict)
    """Per-source quality signal history: source_name -> ['good', 'empty', 'low_content', ...]."""

    # =========================================================================
    # File Upload and Custom Agent Support
    # =========================================================================
    file_ids: list[str] = field(default_factory=list)
    """Uploaded file IDs attached to this research session."""

    file_contents: list[dict[str, Any]] = field(default_factory=list)
    """Pre-loaded file contents for inline prompt injection.

    Each entry: {file_id, filename, file_type, file_size, content, strategy, char_count}
    strategy: 'inline' | 'hybrid' | 'retrieval'
    """

    agent_id: str | None = None
    """Custom agent ID used for this research session."""

    # Per-agent model overrides (009-custom-agent-config)
    model_overrides: dict[str, str] | None = None
    """Tier-to-endpoint overrides from custom agent configuration."""

    # Per-agent domain filter (009-custom-agent-config)
    domain_filter: Any | None = None  # DomainFilterConfig
    """Domain filter override from custom agent configuration."""

    # =========================================================================
    # Manual Workflow Mode (007-enterprise-data-sources, T051)
    # =========================================================================

    # Workflow mode: planner, manual, or hybrid
    workflow_mode: str = "planner"  # WorkflowMode enum value

    # Manual steps defined by user (for MANUAL/HYBRID modes)
    manual_steps: list[Any] = field(default_factory=list)  # ManualStepDefinition

    # Per-step source constraints (from manual steps or planner hints)
    step_source_constraints: dict[str, Any] = field(default_factory=dict)  # step_id -> SourceConstraint

    # Planning
    current_plan: Plan | None = None
    plan_iterations: int = 0
    max_plan_iterations: int = 3

    # Step execution (Researcher phase)
    current_step_index: int = 0
    last_observation: str = ""
    all_observations: list[str] = field(default_factory=list)

    # Reflection
    last_reflection: ReflectionResult | None = None
    reflection_history: list[ReflectionResult] = field(default_factory=list)

    # Sources collected
    sources: list[SourceInfo] = field(default_factory=list)

    # Citation verification (6-stage pipeline)
    evidence_pool: list[EvidenceInfo] = field(default_factory=list)  # Stage 1 output
    claims: list[ClaimInfo] = field(default_factory=list)  # Stage 2-4 output
    verification_summary: VerificationSummaryInfo | None = None  # Post Stage 4
    enable_citation_verification: bool = True  # Feature toggle

    # Token Optimization: Verification result cache (session-scoped)
    # Maps claim fingerprint -> VerificationResult to avoid redundant verification
    # of identical/near-identical claims within the same research session.
    # The cache is NOT persisted - it's cleared when a new research session starts.
    # Key: 16-char MD5 hash of normalized claim text
    # Value: VerificationResult from isolated_verifier
    _verification_cache: dict[str, Any] = field(default_factory=dict, repr=False)

    # Custom phase results storage
    # Maps phase name -> PhaseResult for plugin-provided custom phases
    _phase_results: dict[str, Any] = field(default_factory=dict, repr=False)

    # =========================================================================
    # Async Locks for Parallel Tool Execution (Phase 1: State Safety)
    # =========================================================================
    # CRITICAL: Use asyncio.Lock (NOT threading.RLock) because tool execution
    # is async. threading.RLock would block the event loop and cause deadlocks.
    # Granular locks (one per collection) reduce contention when tools update
    # different collections simultaneously.
    _sources_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _claims_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _evidence_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _cache_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _phase_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _step_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    # Final output (Synthesizer phase)
    final_report: str = ""
    final_report_structured: Any | None = None  # Structured Pydantic output

    # Structured output configuration
    output_format: str = "markdown"  # "markdown" or "json"
    output_schema: type | None = None  # Pydantic model for JSON output

    # Synthesis mode and post-verification configuration
    synthesis_mode: str = "simple"  # "simple" or "reclaim"
    enable_post_verification: bool = False  # Run stages 4-6 after simple generation

    # Custom prompts for structured synthesis (plugin can override)
    structured_system_prompt: str | None = None
    structured_user_prompt: str | None = None

    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    # Cancellation support
    is_cancelled: bool = False

    # =========================================================================
    # Source Scope Helper Methods (008-data-source-selection)
    # =========================================================================

    def is_web_search_allowed(self) -> bool:
        """Check if web search is allowed based on source scope configuration.

        Returns:
            True if web search is allowed (no scope or scope permits web).
            False if scope is 'enterprise_only'.
        """
        if self.source_scope_config is None:
            return True  # No restriction, backward compatible

        # SourceScopeConfig.is_type_enabled() handles all logic:
        # - ENTERPRISE_ONLY: returns False for 'web_search'
        # - WEB_ONLY: returns True for 'web_search'
        # - ALL: returns True for 'web_search'
        return self.source_scope_config.is_type_enabled("web_search")

    def is_enterprise_search_allowed(self) -> bool:
        """Check if enterprise sources are allowed based on source scope.

        Returns:
            True if enterprise sources allowed (no scope or scope permits enterprise).
            False if scope is 'web_only'.
        """
        if self.source_scope_config is None:
            return True

        return (
            self.source_scope_config.is_type_enabled("vector_search")
            or self.source_scope_config.is_type_enabled("genie")
            or self.source_scope_config.is_type_enabled("knowledge_assistant")
        )

    def get_active_scope(self) -> str:
        """Get the active source scope as a string.

        Returns:
            'all', 'enterprise_only', 'web_only', or 'all' if not set.
        """
        if self.source_scope_config is None:
            return "all"
        scope = self.source_scope_config.scope
        return scope.value if hasattr(scope, "value") else str(scope)

    # =========================================================================
    # File Content Helper Methods
    # =========================================================================

    def get_file_context_for_prompt(self, max_chars: int = 0) -> str:
        """Build formatted file content string for LLM prompt injection.

        Returns empty string if no files. When max_chars > 0, enforces total budget
        across all file entries (inline, hybrid, retrieval).
        """
        if not self.file_contents:
            return ""

        parts: list[str] = []
        total = 0

        for fc in self.file_contents:
            strategy = fc.get("strategy", "retrieval")
            filename = fc.get("filename", "unknown")
            content = fc.get("content", "")

            if strategy == "inline":
                entry = f"### File: {filename}\n{content}"
            elif strategy == "hybrid":
                entry = f"### File: {filename} (preview; use file_search for full content)\n{content}"
            else:
                entry = f"### File: {filename} (use file_search tool to search this file)"

            entry_len = len(entry)
            if max_chars > 0 and total + entry_len > max_chars:
                remaining = max_chars - total - 50  # room for truncation marker
                if remaining > 200:
                    parts.append(entry[:remaining] + "\n...[truncated due to size limit]")
                break
            parts.append(entry)
            total += entry_len

        if not parts:
            return ""
        return "## Uploaded File Contents\n\n" + "\n\n".join(parts)

    def has_inline_file_content(self) -> bool:
        """Check if any files have inline or hybrid content."""
        return any(
            fc.get("strategy") in ("inline", "hybrid")
            for fc in self.file_contents
        )

    def get_current_step(self) -> PlanStep | None:
        """Get the current step being executed."""
        if self.current_plan and self.current_step_index < len(self.current_plan.steps):
            return self.current_plan.steps[self.current_step_index]
        return None

    def has_more_steps(self) -> bool:
        """Check if there are more steps to execute."""
        if not self.current_plan:
            return False
        return self.current_step_index < len(self.current_plan.steps)

    def advance_step(self) -> None:
        """Advance to the next step."""
        self.current_step_index += 1

    def mark_step_complete(self, observation: str) -> None:
        """Mark current step as complete with observation (sync version).

        WARNING: This sync version is NOT safe for concurrent access.
        Use mark_step_complete_async() for parallel tool execution.
        """
        step = self.get_current_step()
        if step:
            step.status = StepStatus.COMPLETED
            step.observation = observation
            self.last_observation = observation
            self.all_observations.append(observation)

    async def mark_step_complete_async(self, observation: str) -> None:
        """Mark current step as complete with observation (async-safe)."""
        async with self._step_lock:
            step = self.get_current_step()
            if step:
                step.status = StepStatus.COMPLETED
                step.observation = observation
                self.last_observation = observation
                self.all_observations.append(observation)

    def add_source(self, source: SourceInfo) -> None:
        """Add a source to the collection (sync version for backward compatibility).

        WARNING: This sync version is NOT safe for concurrent access.
        Use add_source_async() for parallel tool execution.
        """
        # Avoid duplicates
        if not any(s.url == source.url for s in self.sources):
            self.sources.append(source)

    async def add_source_async(self, source: SourceInfo) -> None:
        """Add a source to the collection (async-safe, deduplicates by URL).

        Uses asyncio.Lock to prevent race conditions when multiple tools
        try to add sources concurrently.
        """
        async with self._sources_lock:
            if not any(s.url == source.url for s in self.sources):
                self.sources.append(source)

    def get_completed_steps(self) -> list[PlanStep]:
        """Get list of completed steps from current plan.

        Returns:
            List of PlanStep objects with status COMPLETED.
        """
        if not self.current_plan:
            return []
        return [s for s in self.current_plan.steps if s.status == StepStatus.COMPLETED]

    def complete(self, final_report: str, allow_overwrite: bool = False) -> None:
        """Mark research as complete.

        Args:
            final_report: The final synthesized report content.
            allow_overwrite: If False (default), raises error if already completed.
                Set to True to explicitly allow overwriting (use with caution).

        Raises:
            ValueError: If final_report is empty or contains only whitespace.
            RuntimeError: If already completed and allow_overwrite=False.
        """
        if not final_report or not final_report.strip():
            raise ValueError("Cannot complete research with empty report")

        # Idempotency guard: prevent accidental double completion
        if self.completed_at is not None and not allow_overwrite:
            # Idempotent: same content → silently skip (log for visibility)
            if self.final_report == final_report:
                logger.warning(
                    "RESEARCH_COMPLETE_IDEMPOTENT_SKIP",
                    completed_at=str(self.completed_at),
                    report_len=len(final_report),
                )
                return
            # Different content → real bug, raise
            raise RuntimeError(
                f"Research already completed at {self.completed_at} with "
                f"report_len={len(self.final_report) if self.final_report else 0}. "
                f"New report_len={len(final_report)}. "
                f"Use allow_overwrite=True to explicitly overwrite."
            )

        self.final_report = final_report
        self.completed_at = datetime.now(UTC)

    def cancel(self) -> None:
        """Mark research as cancelled."""
        self.is_cancelled = True
        self.completed_at = datetime.now(UTC)

    # =========================================================================
    # Custom Phase Results Storage
    # =========================================================================

    def add_phase_result(self, phase_name: str, result: Any) -> None:
        """Store result from a custom research phase (sync version).

        WARNING: This sync version is NOT safe for concurrent access.
        Use add_phase_result_async() for parallel tool execution.

        Args:
            phase_name: Name of the phase that produced the result
            result: PhaseResult object with output and sources
        """
        if not hasattr(self, "_phase_results") or self._phase_results is None:
            self._phase_results = {}
        self._phase_results[phase_name] = result

    async def add_phase_result_async(self, phase_name: str, result: Any) -> None:
        """Store result from a custom research phase (async-safe).

        Args:
            phase_name: Name of the phase that produced the result
            result: PhaseResult object with output and sources
        """
        async with self._phase_lock:
            if not hasattr(self, "_phase_results") or self._phase_results is None:
                self._phase_results = {}
            self._phase_results[phase_name] = result

    def get_phase_result(self, phase_name: str) -> Any | None:
        """Get result from a completed phase.

        Args:
            phase_name: Name of the phase

        Returns:
            PhaseResult or None if phase hasn't completed
        """
        if not hasattr(self, "_phase_results") or self._phase_results is None:
            return None
        return self._phase_results.get(phase_name)

    def get_all_phase_results(self) -> dict[str, Any]:
        """Get all phase results.

        Returns:
            Dict mapping phase name to PhaseResult
        """
        if not hasattr(self, "_phase_results") or self._phase_results is None:
            return {}
        return dict(self._phase_results)

    def resolve_depth(self) -> str:
        """Resolve effective research depth.

        If research_depth is 'auto', determines depth based on query complexity.
        Otherwise returns the explicitly set depth.

        Returns:
            Effective depth string (light, medium, or extended).
        """
        if self.effective_depth:
            return self.effective_depth

        if self.research_depth != "auto":
            self.effective_depth = self.research_depth
            return self.effective_depth

        # Auto-determine based on query complexity
        if self.query_classification:
            complexity = self.query_classification.complexity
            self.effective_depth = COMPLEXITY_TO_DEPTH.get(complexity, "medium")
        else:
            # Default to medium if no classification available
            self.effective_depth = "medium"

        return self.effective_depth

    def get_max_steps(self) -> int:
        """Get maximum number of research steps for current depth.

        Uses centralized research_types configuration from app.yaml.

        Returns:
            Maximum number of steps to execute.
        """
        from deep_research.agent.config import get_step_limits

        depth = self.resolve_depth()
        step_limits = get_step_limits(depth)
        return step_limits.max

    def get_min_steps(self) -> int:
        """Get minimum number of research steps for current depth.

        Uses centralized research_types configuration from app.yaml.

        Returns:
            Minimum number of steps before early completion is allowed.
        """
        from deep_research.agent.config import get_step_limits

        depth = self.resolve_depth()
        step_limits = get_step_limits(depth)
        return step_limits.min

    def add_evidence(self, evidence: EvidenceInfo) -> None:
        """Add an evidence span to the pool (sync version for backward compatibility).

        WARNING: This sync version is NOT safe for concurrent access.
        Use add_evidence_async() for parallel tool execution.
        """
        self.evidence_pool.append(evidence)

    async def add_evidence_async(self, evidence: EvidenceInfo) -> None:
        """Add an evidence span to the pool (async-safe)."""
        async with self._evidence_lock:
            self.evidence_pool.append(evidence)

    def add_claim(self, claim: ClaimInfo) -> None:
        """Add a claim to the claims list (sync version for backward compatibility).

        WARNING: This sync version is NOT safe for concurrent access.
        Use add_claim_async() for parallel tool execution.
        """
        self.claims.append(claim)

    async def add_claim_async(self, claim: ClaimInfo) -> None:
        """Add a claim to the claims list (async-safe)."""
        async with self._claims_lock:
            self.claims.append(claim)

    def update_verification_summary(self) -> None:
        """Update verification summary from current claims."""
        if not self.claims:
            self.verification_summary = None
            return

        supported = sum(1 for c in self.claims if c.verification_verdict == "supported")
        partial = sum(1 for c in self.claims if c.verification_verdict == "partial")
        unsupported = sum(
            1 for c in self.claims if c.verification_verdict == "unsupported"
        )
        contradicted = sum(
            1 for c in self.claims if c.verification_verdict == "contradicted"
        )
        abstained = sum(1 for c in self.claims if c.abstained)

        total = len(self.claims)
        verified = total - abstained

        self.verification_summary = VerificationSummaryInfo(
            total_claims=total,
            supported_count=supported,
            partial_count=partial,
            unsupported_count=unsupported,
            contradicted_count=contradicted,
            abstained_count=abstained,
            unsupported_rate=unsupported / verified if verified > 0 else 0.0,
            contradicted_rate=contradicted / verified if verified > 0 else 0.0,
            warning=(unsupported / verified > 0.20 if verified > 0 else False)
            or (contradicted / verified > 0.05 if verified > 0 else False),
        )

    # =========================================================================
    # Token Optimization: Verification Cache Methods
    # =========================================================================

    def get_verification_cache(self) -> dict[str, Any]:
        """Get the verification cache dictionary.

        Returns the internal cache dict that can be passed to
        IsolatedVerifier.verify_batch_grouped() for cache reuse.

        Returns:
            Dict mapping claim fingerprints to VerificationResult objects.
        """
        return self._verification_cache

    def get_cached_verification(self, claim_fingerprint: str) -> Any | None:
        """Get a cached verification result by claim fingerprint.

        Args:
            claim_fingerprint: 16-char MD5 hash of normalized claim.

        Returns:
            Cached VerificationResult or None if not cached.
        """
        return self._verification_cache.get(claim_fingerprint)

    def cache_verification(self, claim_fingerprint: str, result: Any) -> None:
        """Cache a verification result for future reuse (sync version).

        WARNING: This sync version is NOT safe for concurrent access.
        Use cache_verification_async() for parallel tool execution.

        Args:
            claim_fingerprint: 16-char MD5 hash of normalized claim.
            result: VerificationResult to cache.
        """
        self._verification_cache[claim_fingerprint] = result

    async def cache_verification_async(self, claim_fingerprint: str, result: Any) -> None:
        """Cache a verification result for future reuse (async-safe).

        Args:
            claim_fingerprint: 16-char MD5 hash of normalized claim.
            result: VerificationResult to cache.
        """
        async with self._cache_lock:
            self._verification_cache[claim_fingerprint] = result

    def clear_verification_cache(self) -> None:
        """Clear the verification cache.

        Called at the start of a new research session or when
        the evidence pool changes significantly.
        """
        self._verification_cache.clear()

    def get_verification_cache_stats(self) -> dict[str, int]:
        """Get statistics about the verification cache.

        Returns:
            Dict with cache statistics (size, etc.).
        """
        return {
            "cache_size": len(self._verification_cache),
        }

    # =========================================================================
    # Enterprise Data Source Methods (007-enterprise-data-sources, T035)
    # =========================================================================

    def record_source_quality(self, source_name: str, signal: str) -> None:
        """Record a quality signal for a source.

        Args:
            source_name: Name of the data source tool.
            signal: Quality signal: 'good' | 'low_content' | 'empty'.
        """
        self.source_quality_history.setdefault(source_name, []).append(signal)

    def get_source_budget(self, source_name: str, default_budget: int = 10) -> int:
        """Get remaining query budget for a source.

        Args:
            source_name: Name of the data source.
            default_budget: Default budget if not configured.

        Returns:
            Number of queries remaining for this source.
        """
        # Default budgets by source type
        default_budgets: dict[str, int] = {
            "vector_search": 15,
            "genie": 5,
            "knowledge_assistant": 5,
            "web_search": 20,
        }

        # Get budget for this source type (extract type from name if needed)
        total_budget = default_budgets.get(source_name, default_budget)

        used = self.source_query_counts.get(source_name, 0)
        return max(0, total_budget - used)

    def record_source_query(self, source_name: str, count: int = 1) -> None:
        """Record that queries were made to a source (sync version).

        WARNING: This sync version is NOT safe for concurrent access.
        Use record_source_query_async() for parallel tool execution.

        Args:
            source_name: Name of the data source.
            count: Number of queries to record.
        """
        current = self.source_query_counts.get(source_name, 0)
        self.source_query_counts[source_name] = current + count

    async def record_source_query_async(self, source_name: str, count: int = 1) -> None:
        """Record that queries were made to a source (async-safe).

        Args:
            source_name: Name of the data source.
            count: Number of queries to record.
        """
        async with self._step_lock:  # Reuse step lock for source tracking
            current = self.source_query_counts.get(source_name, 0)
            self.source_query_counts[source_name] = current + count

    def add_source_result(
        self, source_name: str, result: "SourceInfo"
    ) -> None:
        """Add a result from a specific source (sync version).

        WARNING: This sync version is NOT safe for concurrent access.
        Use add_source_result_async() for parallel tool execution.

        Args:
            source_name: Name of the source that produced the result.
            result: The SourceInfo result to add.
        """
        if source_name not in self.source_results:
            self.source_results[source_name] = []
        self.source_results[source_name].append(result)
        # Also add to main sources list
        self.add_source(result)

    async def add_source_result_async(
        self, source_name: str, result: "SourceInfo"
    ) -> None:
        """Add a result from a specific source (async-safe).

        Args:
            source_name: Name of the source that produced the result.
            result: The SourceInfo result to add.
        """
        async with self._sources_lock:
            if source_name not in self.source_results:
                self.source_results[source_name] = []
            self.source_results[source_name].append(result)
            # Also add to main sources list (already under lock)
            if not any(s.url == result.url for s in self.sources):
                self.sources.append(result)

    def get_source_results(self, source_name: str) -> list["SourceInfo"]:
        """Get all results from a specific source.

        Args:
            source_name: Name of the source.

        Returns:
            List of SourceInfo results from that source.
        """
        return self.source_results.get(source_name, [])

    def get_source_stats(self) -> dict[str, dict[str, int]]:
        """Get statistics about source usage.

        Returns:
            Dict mapping source name to {queries: N, results: M}.
        """
        stats: dict[str, dict[str, int]] = {}

        for source_name in set(self.source_query_counts.keys()) | set(self.source_results.keys()):
            stats[source_name] = {
                "queries": self.source_query_counts.get(source_name, 0),
                "results": len(self.source_results.get(source_name, [])),
            }

        return stats

    # =========================================================================
    # Manual Workflow Mode Methods (007-enterprise-data-sources, T051)
    # =========================================================================

    def is_manual_mode(self) -> bool:
        """Check if workflow is in manual mode."""
        return self.workflow_mode == WorkflowMode.MANUAL.value

    def is_hybrid_mode(self) -> bool:
        """Check if workflow is in hybrid mode."""
        return self.workflow_mode == WorkflowMode.HYBRID.value

    def is_planner_mode(self) -> bool:
        """Check if workflow is in planner mode (default)."""
        return self.workflow_mode == WorkflowMode.PLANNER.value

    def get_source_constraint(self, step_id: str) -> Any | None:
        """Get source constraint for a specific step.

        Args:
            step_id: ID of the step.

        Returns:
            SourceConstraint or None if no constraint defined.
        """
        return self.step_source_constraints.get(step_id)

    def set_source_constraint(self, step_id: str, constraint: Any) -> None:
        """Set source constraint for a step.

        Args:
            step_id: ID of the step.
            constraint: SourceConstraint to apply.
        """
        self.step_source_constraints[step_id] = constraint

    def get_manual_step(self, step_id: str) -> Any | None:
        """Get a manual step definition by ID.

        Args:
            step_id: ID of the step.

        Returns:
            ManualStepDefinition or None if not found.
        """
        for step in self.manual_steps:
            if hasattr(step, "id") and step.id == step_id:
                return step
        return None

    def get_manual_steps_in_order(self) -> list[Any]:
        """Get manual steps sorted by order.

        Returns:
            List of ManualStepDefinition sorted by order field.
        """
        return sorted(
            self.manual_steps,
            key=lambda s: getattr(s, "order", 0)
        )

    def should_use_planner(self) -> bool:
        """Determine if planner should be used based on workflow mode.

        Returns:
            True if planner should create/extend the plan.
        """
        # Planner mode always uses planner
        # Hybrid mode: planner runs AFTER manual steps are converted
        # Manual mode: skip planner entirely
        return self.is_planner_mode() or self.is_hybrid_mode()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "session_id": str(self.session_id),
            "query_mode": self.query_mode,
            "query_classification": self.query_classification.to_dict()
            if self.query_classification
            else None,
            "is_simple_query": self.is_simple_query,
            "current_plan": self.current_plan.to_dict() if self.current_plan else None,
            "plan_iterations": self.plan_iterations,
            "current_step_index": self.current_step_index,
            "all_observations": self.all_observations,
            "sources": [s.to_dict() for s in self.sources],
            "evidence_pool": [e.to_dict() for e in self.evidence_pool],
            "claims": [c.to_dict() for c in self.claims],
            "verification_summary": self.verification_summary.to_dict()
            if self.verification_summary
            else None,
            "enable_citation_verification": self.enable_citation_verification,
            "final_report": self.final_report,
            "is_cancelled": self.is_cancelled,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "phase_results": {
                name: result.to_dict() if hasattr(result, "to_dict") else str(result)
                for name, result in self._phase_results.items()
            } if self._phase_results else {},
            # Enterprise data source fields (007-enterprise-data-sources)
            "source_scope_config": self.source_scope_config.to_dict()
            if self.source_scope_config and hasattr(self.source_scope_config, "to_dict")
            else None,
            "data_landscape": self.data_landscape.to_dict()
            if self.data_landscape and hasattr(self.data_landscape, "to_dict")
            else None,
            "source_quality_history": self.source_quality_history,
            "source_query_counts": self.source_query_counts,
            "source_results": {
                name: [s.to_dict() for s in results]
                for name, results in self.source_results.items()
            },
            # Workflow mode fields (007-enterprise-data-sources)
            "workflow_mode": self.workflow_mode,
            "manual_steps": [
                s.model_dump() if hasattr(s, "model_dump") else s.dict() if hasattr(s, "dict") else str(s)
                for s in self.manual_steps
            ] if self.manual_steps else [],
            "step_source_constraints": {
                step_id: c.model_dump() if hasattr(c, "model_dump") else c.dict() if hasattr(c, "dict") else str(c)
                for step_id, c in self.step_source_constraints.items()
            } if self.step_source_constraints else {},
        }
