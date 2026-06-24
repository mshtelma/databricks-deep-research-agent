"""Promotion-grade trace capture (spec Wave 5 / feature 6.1).

A :class:`PromotionTrace` is a compact, PII- and injection-safe projection of a
single workflow run's event stream — rich enough to later *synthesize* a
workflow from (feature 6.2), but carrying NO raw argument values or tool output.

Design properties:

* **Value-free.** Tool arguments are recorded as *shapes* (``{name: type_name}``)
  via :func:`_arg_shape`; raw values and tool-result text never enter the trace.
  This is both privacy-preserving and prompt-injection-resistant (the trace is
  later rendered into an LLM brief).
* **Bounded.** Built incrementally (:class:`PromotionTraceBuilder`); each event is
  projected to a small step and the raw event dropped, so memory stays bounded
  regardless of run length. Stream chunks / token spam are ignored; a ``max_steps``
  cap records ``dropped_event_count`` rather than growing without limit.
* **Forward-compatible.** Models use ``extra="ignore"`` and a ``schema_version`` so
  rows persisted by an older build still parse.
* **Generic.** Depends ONLY on ``databricks_deep_research.events`` and the generic
  node/tool lifecycle events — no app, Designer, domain, or topology coupling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from databricks_deep_research.events.types import StreamEvent

# Bound on the persisted query descriptor (chars).
_MAX_QUERY_SHAPE = 200
# Default cap on captured steps (a runaway-loop backstop; over-cap → dropped_event_count).
_DEFAULT_MAX_STEPS = 500


class StepKind(StrEnum):
    """The kind of an observed step in a run."""

    TOOL = "tool"
    AGENT = "agent"
    DECISION = "decision"
    LOOP = "loop"
    SYNTHESIS = "synthesis"


class PromotionStep(BaseModel):
    """One observed step in a run (an agent node, tool call, branch, or loop).

    ``arg_shape`` maps each tool-argument name to its value's TYPE NAME only —
    never the value itself.
    """

    model_config = ConfigDict(extra="ignore")

    order: int
    kind: StepKind
    node_id: str = ""
    subtype: str = ""
    label: str = ""
    tool_name: str = ""
    arg_shape: dict[str, str] = Field(default_factory=dict)
    branch_taken: int | None = None
    loop_iteration: int | None = None
    status: str = "completed"
    produced_key: str = ""
    truncated: bool = False


class PromotionTrace(BaseModel):
    """A value-free, ordered projection of one run's structural behavior."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    run_id: str = ""
    query_shape: str = ""
    is_degenerate: bool = False
    steps: list[PromotionStep] = Field(default_factory=list)
    total_tokens: int = 0
    captured_event_count: int = 0
    dropped_event_count: int = 0


# --- value-free argument-shape derivation ------------------------------------


def _arg_shape(arguments: Any) -> dict[str, str]:
    """Map each argument name to its value's type name only (never the value).

    Returns ``{}`` for any non-dict input. No argument value content ever appears
    in the result — only ``type(value).__name__`` (e.g. ``{"query": "str"}``).
    """
    shape: dict[str, str] = {}
    if not isinstance(arguments, dict):
        return shape
    for key, value in arguments.items():
        shape[str(key)] = type(value).__name__
    return shape


def _bounded(text: Any, limit: int = _MAX_QUERY_SHAPE) -> str:
    """Coerce to a bounded string (defensive against non-str input)."""
    if not isinstance(text, str):
        return ""
    return text if len(text) <= limit else text[:limit]


# --- event -> step projectors (single table; no if/elif sprawl) --------------

_Projector = Callable[[dict[str, Any], int], "PromotionStep | None"]


def _project_tool_call(data: dict[str, Any], order: int) -> PromotionStep | None:
    return PromotionStep(
        order=order,
        kind=StepKind.TOOL,
        node_id=str(data.get("node_id", "")),
        tool_name=str(data.get("tool_name", "")),
        arg_shape=_arg_shape(data.get("arguments")),
    )


def _project_node_started(data: dict[str, Any], order: int) -> PromotionStep | None:
    # Only agent nodes are steps; container nodes (sequence/parallel/loop) are
    # structural noise. Loops/branches are captured via their own events.
    if data.get("node_type") != "agent":
        return None
    return PromotionStep(
        order=order,
        kind=StepKind.AGENT,
        node_id=str(data.get("node_id", "")),
        label=str(data.get("label", "")),
    )


def _project_branch(data: dict[str, Any], order: int) -> PromotionStep | None:
    branch = data.get("branch_index")
    return PromotionStep(
        order=order,
        kind=StepKind.DECISION,
        node_id=str(data.get("node_id", "")),
        branch_taken=branch if isinstance(branch, int) else None,
    )


def _project_loop(data: dict[str, Any], order: int) -> PromotionStep | None:
    iteration = data.get("iteration")
    return PromotionStep(
        order=order,
        kind=StepKind.LOOP,
        node_id=str(data.get("node_id", "")),
        loop_iteration=iteration if isinstance(iteration, int) else None,
    )


def _project_synthesis(data: dict[str, Any], order: int) -> PromotionStep | None:
    return PromotionStep(
        order=order,
        kind=StepKind.SYNTHESIS,
        node_id=str(data.get("node_id", "")),
    )


_PROJECTORS: dict[str, _Projector] = {
    "tool_call": _project_tool_call,
    "node_started": _project_node_started,
    "branch_selected": _project_branch,
    "loop_iteration": _project_loop,
    "synthesis_started": _project_synthesis,
}


class PromotionTraceBuilder:
    """Incrementally project a run's event stream into a :class:`PromotionTrace`.

    Feed every event via :meth:`observe` (cheap, bounded, never raises); call
    :meth:`build` at run completion. Step-producing events become
    :class:`PromotionStep`s; ``node_completed`` / ``agent_output`` /
    ``token_usage`` / ``coordinator_classified`` annotate or accumulate; all
    other events (stream chunks, model-call notices, cache hits) are ignored.
    """

    def __init__(self, *, max_steps: int = _DEFAULT_MAX_STEPS) -> None:
        self._max_steps = max_steps
        self._steps: list[PromotionStep] = []
        self._by_node: dict[str, PromotionStep] = {}
        self._total_tokens = 0
        self._is_degenerate = False
        self._captured = 0
        self._dropped = 0

    def observe(self, event: StreamEvent) -> None:
        """Project one event into the trace. Fail-soft: never raises."""
        try:
            etype = event.event_type
            data = event.model_dump()
        except Exception:
            return

        try:
            if etype == "node_completed":
                self._annotate_status(data)
                return
            if etype == "agent_output":
                self._annotate_output(data)
                return
            if etype == "token_usage":
                total = data.get("total_tokens")
                if isinstance(total, int) and total > self._total_tokens:
                    self._total_tokens = total
                return
            if etype == "coordinator_classified":
                if bool(data.get("is_simple")):
                    self._is_degenerate = True
                return

            projector = _PROJECTORS.get(etype)
            if projector is None:
                return
            step = projector(data, len(self._steps))
            if step is None:
                return
            if len(self._steps) >= self._max_steps:
                self._dropped += 1
                return
            self._steps.append(step)
            self._captured += 1
            if step.node_id:
                self._by_node[step.node_id] = step
        except Exception:
            # Defense in depth: a malformed event must never break a run.
            return

    def _annotate_status(self, data: dict[str, Any]) -> None:
        step = self._by_node.get(str(data.get("node_id", "")))
        status = data.get("status")
        if step is not None and isinstance(status, str):
            step.status = status

    def _annotate_output(self, data: dict[str, Any]) -> None:
        step = self._by_node.get(str(data.get("node_id", "")))
        key = data.get("output_key")
        if step is not None and isinstance(key, str):
            step.produced_key = key

    def build(self, *, run_id: str = "", query_shape: str = "") -> PromotionTrace:
        """Finalize the trace. Safe to call once at run completion."""
        return PromotionTrace(
            run_id=run_id,
            query_shape=_bounded(query_shape),
            is_degenerate=self._is_degenerate,
            steps=list(self._steps),
            total_tokens=self._total_tokens,
            captured_event_count=self._captured,
            dropped_event_count=self._dropped,
        )


def extract_promotion_trace(
    events: Iterable[StreamEvent],
    *,
    run_id: str = "",
    query_shape: str = "",
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> PromotionTrace:
    """Batch convenience: build a :class:`PromotionTrace` from a list of events.

    Equivalent to feeding every event to a :class:`PromotionTraceBuilder`. Used
    by tests and by the ``WorkflowResult.events`` / state-extractor capture path.
    """
    builder = PromotionTraceBuilder(max_steps=max_steps)
    for event in events:
        builder.observe(event)
    return builder.build(run_id=run_id, query_shape=query_shape)
