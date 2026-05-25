"""Task signature + deterministic topology selector.

Defines the ``TaskSignature`` pydantic contract emitted by the classifier
agent and a pure-Python ``select_topology()`` function that maps the
signature's structural axes to one of the framework's three supported
topologies.

This module owns the only place where signature → topology selection is
authoritative; the scaffolder_specializer agent calls ``select_topology``
through the framework tool layer rather than picking a topology
heuristically. The signature is the contract between everything
downstream: the auditor's checklist, the behavioral probe's conditional
checks, and the scaffolder_specializer's lane-prompt directives all read
from the same TaskSignature.

Plan v2.1 (refined after codex critique) extends the signature with five
explicit structural axes (``step_dependencies_present``,
``independent_workstreams_count``, ``iteration_required``,
``output_aggregation_kind``, ``lane_descriptions``) plus an
``axis_reasoning`` field for low-confidence justifications. The new
``select_topology`` precedence — independence wins first — is the
explicit fix for the Investment failure where six independent domains
were being routed to ``plan_and_execute`` because
``iteration_required=True`` would have fired before the parallel-lanes
check under the original v2 ordering.

The follow-up refinement (Fix A/B in the latest plan) separates the
strict classifier-emission path from the lenient storage-load path:

* :meth:`TaskSignature.from_classifier_emission` validates a freshly
  emitted payload with no default backfill — missing structural axes
  raise ``ValidationError`` so the designer halts cleanly per Plan
  v2.1 M11 rather than producing a wrong AST.
* :meth:`TaskSignature.load_from_storage` pre-fills the legacy defaults
  so previously serialized 7-field payloads still parse.
* :meth:`TaskSignature.tool_schema` returns a JSON schema derived from
  the pydantic model, post-processed for LLM tool APIs (anyOf-collapse,
  title-strip). Used by ``EmitTaskSignatureTool`` so the LLM-facing
  contract stays in lockstep with the model.

Generic-vocabulary discipline applies: no corpus or table names appear
in this module — only signature literals, structural axes, and topology
names.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

AssetSignature = Literal[
    "corpus_only",
    "corpus_plus_web",
    "web_only",
    "structured_only",
    "no_assets",
]

RetrievalPattern = Literal[
    "bounded_lookup",
    "pipelined_retrieve_read_compute",
    "independent_lanes",
    "open_research",
]

QuestionClass = Literal[
    "bounded_lookup",
    "open_research",
    "numeric_aggregation",
    "comparative_analysis",
    "meta_workflow",
]

AmbiguityAxis = Literal[
    "period_basis",
    "entity_scope",
    "geographic_scope",
    "temporal_scope",
    "unit_basis",
]

PrimaryEvidenceKind = Literal[
    "text_chunks",
    "structured_tables",
    "web_articles",
    "qa_assistant",
    "mixed",
]

OutputShape = Literal[
    "single_number",
    "table",
    "paragraph",
    "structured_report",
]

OutputAggregationKind = Literal[
    "single_answer",
    "cross_concern_synthesis",
    "per_concern_report",
]

TopologyName = Literal["single_agent", "parallel_lanes", "plan_and_execute"]

# Framework topology names. Kept as a tuple so ``select_topology`` and the
# probe-side conditional checks can verify topology strings consistently.
TOPOLOGIES: tuple[TopologyName, ...] = (
    "single_agent",
    "parallel_lanes",
    "plan_and_execute",
)


def _collapse_optional_anyof(prop: dict[str, Any]) -> None:
    """Collapse Pydantic's ``anyOf`` shape for ``X | None`` into the
    non-null branch.

    ``axis_reasoning: dict[str, str] | None`` produces
    ``{"anyOf": [{"type": "object", ...}, {"type": "null"}]}`` which
    Databricks-hosted Claude rejects in tool parameter schemas. We
    promote the non-null branch's keys onto the property and remove
    the ``anyOf`` envelope. The ``default`` (typically ``None``) stays
    in place — that is how the LLM signals omission.
    """
    options = prop.get("anyOf")
    if not isinstance(options, list) or not options:
        return
    non_null = [opt for opt in options if isinstance(opt, dict) and opt.get("type") != "null"]
    if len(non_null) != 1:
        # ``X | Y`` (no None) or 3+ branches — leave it alone for the caller
        # to handle deliberately rather than guessing.
        return
    del prop["anyOf"]
    for key, value in non_null[0].items():
        # Don't clobber a more specific key already on the property
        # (e.g., ``description`` from Field()).
        prop.setdefault(key, value)


class SignatureError(ValueError):
    """Raised when a TaskSignature cannot be constructed from a payload.

    Plan v2.1 M11: failure-closed for invalid / missing / low-confidence
    signatures. Callers that previously caught a generic ``Exception`` and
    fell back to brief topology must now propagate ``SignatureError`` so
    the designer flow halts with a clear classification failure rather
    than silently producing a wrong AST.
    """


class TaskSignature(BaseModel):
    """Structured classification of the user's task.

    Emitted by the classifier agent, consumed by the topology selector,
    the scaffolder_specializer's prompt construction, the behavioral
    probe's signature-conditioned checks, and the auditor's checklist.

    The five Plan v2.1 structural axes (``step_dependencies_present``,
    ``independent_workstreams_count``, ``iteration_required``,
    ``output_aggregation_kind``, ``lane_descriptions``) are **required**
    by the model — fresh classifier emissions that omit them raise
    ``ValidationError`` (fail-closed per Plan v2.1 M11). Legacy
    serialized payloads that predate the structural-axis extension load
    via :meth:`TaskSignature.load_from_storage`, which pre-fills the
    defaults declared in :attr:`_LEGACY_STORAGE_DEFAULTS` before
    validating.

    ``retrieval_pattern`` is retained for downstream consumers that read
    it descriptively but no longer drives topology selection when the
    structural axes are set (see :func:`select_topology`).
    """

    model_config = ConfigDict(extra="forbid")

    # Legacy axes (kept for backward compat + downstream descriptive use).
    asset_signature: AssetSignature
    retrieval_pattern: RetrievalPattern
    question_class: QuestionClass
    question_ambiguity: list[AmbiguityAxis] = Field(default_factory=list)
    primary_evidence_kind: PrimaryEvidenceKind
    expected_output_shape: OutputShape
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Classifier's self-reported confidence; "
            "the designer escalates to manual review or "
            "defaults to a permissive topology when below 0.7."
        ),
    )

    # Plan v2.1 structural axes. REQUIRED for fresh classifier emissions
    # (no field-level default) so the LLM-facing tool schema marks them
    # required and ``from_classifier_emission`` rejects partial payloads.
    # ``load_from_storage`` pre-fills the documented defaults from
    # ``_LEGACY_STORAGE_DEFAULTS`` so previously serialized payloads still
    # parse without surprise.
    step_dependencies_present: bool = Field(
        description=(
            "True iff any planned step's input depends on a prior step's "
            "output. Drives plan_and_execute selection in the absence of "
            "independent_workstreams_count >= 2. Derive from work shape, "
            "not from vocabulary in the intent."
        ),
    )
    independent_workstreams_count: int = Field(
        ge=0,
        le=8,
        description=(
            "How many concerns can run concurrently. 0/1 means "
            "single-lane; >=2 means parallel_lanes (independence wins "
            "first in select_topology). Count explicit enumerations in "
            "the brief."
        ),
    )
    iteration_required: bool = Field(
        description=(
            "True iff coverage demands reflection-driven replanning at "
            "the workflow level (the evaluator decides which NEW queries "
            "to issue based on partial findings). Not the same as "
            "per-lane ReAct iteration."
        ),
    )
    output_aggregation_kind: OutputAggregationKind = Field(
        description=(
            "Shape of the synthesizer's output across concerns. Drives "
            "the synthesizer scaffold section layout: ``single_answer`` "
            "(one figure or paragraph), ``cross_concern_synthesis`` "
            "(integrate multiple lanes), ``per_concern_report`` "
            "(separate section per lane)."
        ),
    )
    lane_descriptions: list[str] = Field(
        description=(
            "Extractive (verbatim) phrases from the user intent naming "
            "each independent concern. Length MUST equal "
            "max(independent_workstreams_count, 1) for fresh emissions. "
            "Never invented; hallucinated taxonomies are forbidden."
        ),
    )
    axis_reasoning: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional per-axis short justification for low-confidence "
            "outputs (confidence < 0.7). Keys are structural axis names. "
            "Omit when confidence is high."
        ),
    )

    # Defaults applied ONLY by ``load_from_storage`` for legacy payloads
    # that predate the structural-axis extension. Kept as a class-level
    # constant so the migration surface is auditable from one place.
    _LEGACY_STORAGE_DEFAULTS: ClassVar[dict[str, Any]] = {
        "step_dependencies_present": False,
        "independent_workstreams_count": 1,
        "iteration_required": False,
        "output_aggregation_kind": "single_answer",
        "lane_descriptions": [],
        "axis_reasoning": None,
    }

    @classmethod
    def from_classifier_emission(cls, payload: dict[str, Any]) -> Self:
        """Strict construction path for the classifier's tool emission.

        Does not apply legacy defaults. Any missing structural axis
        raises ``ValidationError`` so the designer fails closed per
        Plan v2.1 M11 instead of producing a wrong AST.

        Also enforces the Plan v2.1 M6 extractive-provenance cross-field
        contract: ``len(lane_descriptions) == max(count, 1)``. The lenient
        :meth:`load_from_storage` path does NOT enforce this — the
        downstream builder's ``_resolve_lane_descriptions`` retains its
        padding/truncation role as a legacy safety net.

        Use this only at :class:`framework_tools.EmitTaskSignatureTool`.
        All other call sites should use :meth:`load_from_storage`.
        """
        sig = cls.model_validate(payload)
        expected = max(sig.independent_workstreams_count, 1)
        if len(sig.lane_descriptions) != expected:
            raise ValueError(
                f"lane_descriptions length {len(sig.lane_descriptions)} "
                f"does not match max(independent_workstreams_count, 1)={expected}"
            )
        return sig

    @classmethod
    def load_from_storage(cls, payload: dict[str, Any]) -> Self:
        """Lenient construction path for previously serialized payloads.

        Pre-fills :attr:`_LEGACY_STORAGE_DEFAULTS` for any structural
        axis missing from the payload, then validates. Used by every
        consumer except :class:`framework_tools.EmitTaskSignatureTool`.

        Accepts dict-like payloads from MLflow trace replay, prior
        scaffold-and-run fixtures, designer state cache, and the
        architect's hand-crafted ``select_topology`` probes.
        """
        if not isinstance(payload, dict):
            raise TypeError(
                f"load_from_storage expects a dict payload, got {type(payload).__name__}"
            )
        filled = dict(payload)
        for key, default in cls._LEGACY_STORAGE_DEFAULTS.items():
            filled.setdefault(key, default)
        return cls.model_validate(filled)

    @classmethod
    def tool_schema(cls) -> dict[str, Any]:
        """Return an LLM-tool-call-friendly JSON schema for this model.

        Wraps :meth:`model_json_schema` with three post-processing steps:

        1. Collapse ``anyOf`` produced by ``X | None`` fields — the
           Databricks-hosted Claude tool API rejects ``anyOf`` in
           parameter schemas. The non-null branch is promoted; the
           field is removed from ``required`` since the None default
           remains the LLM's "omit" signal.
        2. Strip ``"title"`` keys from properties — they cost classifier
           tokens and tool APIs ignore them.
        3. Keep ``Literal`` enums inline (Pydantic v2 already does this
           for ``Literal`` type aliases — no ``$defs`` flattener
           needed for this model).

        Used by :class:`framework_tools.EmitTaskSignatureTool` so the
        LLM-facing tool contract stays in lockstep with the model.
        """
        schema = cls.model_json_schema()
        # Pydantic adds a top-level ``title`` and per-property ``title``;
        # tool APIs do not need them.
        schema.pop("title", None)
        properties = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        for name, prop in list(properties.items()):
            if isinstance(prop, dict):
                prop.pop("title", None)
                _collapse_optional_anyof(prop)
                # If a field carries an explicit default it is structurally
                # optional even when the legacy schema marks it required;
                # ``axis_reasoning`` with ``default=None`` is the prototypical
                # case. Drop it from ``required`` so the LLM can omit it.
                if "default" in prop and name in required:
                    required.remove(name)
        if required:
            schema["required"] = required
        elif "required" in schema:
            del schema["required"]
        return schema


def _has_explicit_structural_axes(sig: TaskSignature) -> bool:
    """Return True iff the signature carries non-default structural axes.

    A signature is "structural" when at least one of the new axes deviates
    from its default. When all axes are at default values, the legacy
    ``retrieval_pattern`` fallback path is used to preserve PR3-B behavior
    for older signatures.
    """
    return (
        sig.step_dependencies_present
        or sig.independent_workstreams_count != 1
        or sig.iteration_required
        or sig.output_aggregation_kind != "single_answer"
        or bool(sig.lane_descriptions)
    )


def _select_topology_from_structural_axes(sig: TaskSignature) -> TopologyName:
    """Plan v2.1 M4 three-rule precedence — independence wins first.

    Rule 1: ``independent_workstreams_count >= 2`` ALWAYS maps to
    ``parallel_lanes``, regardless of per-lane iteration need. Each lane
    can reflect within its own ReAct loop; workflow-level iteration is
    not required for independent concerns.

    Rule 2: ``step_dependencies_present`` OR ``iteration_required`` →
    ``plan_and_execute`` (planner can sequence; evaluator can replan).

    Rule 3: Otherwise → ``single_agent`` (bounded single-step lookup).

    This precedence is the explicit fix for codex CRITICAL-5: under the
    original v2 mapping, a six-domain task with ``iteration_required=True``
    would have fallen into the dependencies-or-iteration branch BEFORE
    the parallel-lanes check, recreating the Investment failure.
    """
    if sig.independent_workstreams_count >= 2:
        return "parallel_lanes"
    if sig.step_dependencies_present or sig.iteration_required:
        return "plan_and_execute"
    return "single_agent"


def _select_topology_from_retrieval_pattern(sig: TaskSignature) -> TopologyName:
    """Legacy PR3-B mapping; fallback for signatures without structural axes."""
    if sig.retrieval_pattern == "pipelined_retrieve_read_compute":
        return "plan_and_execute"
    if sig.retrieval_pattern == "independent_lanes":
        return "parallel_lanes"
    if sig.retrieval_pattern == "bounded_lookup":
        return "single_agent"
    if sig.retrieval_pattern == "open_research":
        # Open research lanes are independent by design — parallel_lanes
        # is the right topology for breadth, not depth-then-compute.
        return "parallel_lanes"
    raise ValueError(f"unknown retrieval_pattern {sig.retrieval_pattern!r}")


def select_topology(sig: TaskSignature) -> TopologyName:
    """Map a TaskSignature to one of the three framework topologies.

    Deterministic; no LLM. Plan v2.1 precedence:

    - If the signature carries explicit structural axes (any of
      ``step_dependencies_present``, ``independent_workstreams_count``,
      ``iteration_required``, ``output_aggregation_kind``,
      ``lane_descriptions`` deviates from default), use the three-rule
      precedence (independence wins first; then deps/iteration; else
      single_agent).
    - Otherwise (legacy signature, all structural axes at default), fall
      back to the PR3-B ``retrieval_pattern`` mapping.

    The fallback preserves behavior for existing serialized signatures
    and any test fixtures authored before the structural-axis extension.
    """
    if _has_explicit_structural_axes(sig):
        return _select_topology_from_structural_axes(sig)
    return _select_topology_from_retrieval_pattern(sig)
