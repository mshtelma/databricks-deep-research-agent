"""Pydantic schema for declarative agent UI surfaces.

A *surface* is an A2UI-shaped, app-owned description of a small UI an agent
carries inside its workflow definition (``definition["surface"]``): a flat
adjacency-list component tree, a JSON data model bound via JSON Pointers, and
action bindings that compile a form submission into an agent run.

Design invariants:

* The surface is **opaque to the framework engine** — ``load_workflow_from_dict``
  ignores unknown top-level keys, so the raw definition dict (with ``surface``)
  is what persists and travels; the runtime ``WorkflowDefinition`` never sees it.
* Every model is ``extra="forbid"`` so hand-crafted or LLM-authored payloads
  fail loudly instead of smuggling unknown fields past the validator.
* Components reference each other by id (adjacency list) with exactly one
  ``id == "root"`` — LLM-friendly to emit, trivial to validate.
* Dynamic values bind to the data model with ``{"path": "/json/pointer"}``.

Structural/semantic rules beyond field shape (catalog membership, cycles,
binding coverage, reserved keys, size caps) live in
:mod:`deep_research.surface.validation`.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# JSON Pointer subset: one or more `/segment` parts, identifier-ish segments.
# Deliberately narrower than RFC 6901 (no escapes, no array indices in v1) so
# pointers stay unambiguous in prompts, query templates, and TS mirrors.
POINTER_PATTERN = r"^(/[A-Za-z0-9_]+)+$"
_POINTER_RE = re.compile(POINTER_PATTERN)

# Identifier grammar shared by component ids, binding actions, and binding
# input keys. Matches the SafeTemplateRenderer variable grammar so any binding
# input key is usable as a `{placeholder}` in node prompts.
IDENTIFIER_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
_IDENTIFIER_RE = re.compile(IDENTIFIER_PATTERN)

SURFACE_VERSION: Literal[1] = 1


def is_valid_pointer(pointer: str) -> bool:
    """True when *pointer* matches the surface JSON-Pointer subset."""
    return bool(_POINTER_RE.match(pointer))


def is_valid_identifier(name: str) -> bool:
    """True when *name* matches the shared identifier grammar."""
    return bool(_IDENTIFIER_RE.match(name))


class PathRef(BaseModel):
    """A JSON-Pointer reference into the surface data model."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(pattern=POINTER_PATTERN)


# A dynamic value is either a literal scalar or a data-model reference.
# PathRef first so dicts resolve to it; bool before int so smart-union keeps
# booleans booleans.
DynamicValue = PathRef | bool | int | float | str | None


class RunOptions(BaseModel):
    """Per-run options a binding may set (literal or data-model bound)."""

    model_config = ConfigDict(extra="forbid")

    research_depth: DynamicValue = None
    verify_sources: DynamicValue = None
    query_mode: DynamicValue = None
    source_scope: DynamicValue = None
    enable_plan_review: DynamicValue = None
    turn_intent: DynamicValue = None
    tone: DynamicValue = None
    output_language: DynamicValue = None
    enable_cross_session_memory: DynamicValue = None
    allow_live_search: DynamicValue = None


class OutputTarget(BaseModel):
    """Where a binding's run result lands in the data model.

    The runtime patches ``target`` with a run REFERENCE
    (``{"status", "session_id", "message_id"}``) — never a copy of the report —
    so result regions resolve content/claims through the persisted message.
    """

    model_config = ConfigDict(extra="forbid")

    target: str = Field(pattern=POINTER_PATTERN)
    mode: Literal["report"] = "report"


class ActionBinding(BaseModel):
    """Maps a UI action (a Button press) to an agent run.

    v1 security contract: ``kind`` is ``run_agent`` and always means the agent
    that OWNS this surface — there is deliberately no ``agent_id`` field, so a
    surface can never be pointed at another workflow.
    """

    model_config = ConfigDict(extra="forbid")

    action: str = Field(pattern=IDENTIFIER_PATTERN, max_length=64)
    kind: Literal["run_agent"] = "run_agent"
    # Keys become the run's inputs. "query" is special-cased into the job's
    # query field (str values support `{/pointer}` substitution); every other
    # key is seeded into initial workflow state and usable as a `{placeholder}`
    # in node prompts.
    inputs: dict[str, DynamicValue] = Field(default_factory=dict)
    options: RunOptions = Field(default_factory=RunOptions)
    output: OutputTarget
    # Reserved for multi-action surfaces (parallel|queue); v1 is replace-only.
    concurrency: Literal["replace"] = "replace"


class SurfaceComponent(BaseModel):
    """One node of the flat component list; tree structure via id references."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=IDENTIFIER_PATTERN, max_length=64)
    component: str = Field(min_length=1, max_length=64)
    props: dict[str, Any] = Field(default_factory=dict)
    children: list[str] = Field(default_factory=list)


ControlPolicy = Literal["show", "hide", "locked", "advanced"]


class SurfaceRuntimeControls(BaseModel):
    """Host-owned run-control visibility/policy hints for this surface."""

    model_config = ConfigDict(extra="forbid")

    effort: ControlPolicy | None = None
    sources: ControlPolicy | None = None
    verify_sources: ControlPolicy | None = None
    plan_review: ControlPolicy | None = None
    report_style: ControlPolicy | None = None
    cross_session_memory: ControlPolicy | None = None
    live_search: ControlPolicy | None = None


class SurfaceSectionLayout(BaseModel):
    """Host frame section that renders a subset of component ids.

    ``children`` is optional: an empty list means the host infers this section's
    contents from the component tree (see :class:`SurfaceLayout`).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=IDENTIFIER_PATTERN, max_length=64)
    title: str = Field(min_length=1, max_length=80)
    role: Literal["inputs", "results", "custom"]
    children: list[str] = Field(default_factory=list)
    default_open: (
        Literal["before_first_run", "during_run", "after_run", "always", "never"] | None
    ) = None


class SurfaceLayout(BaseModel):
    """Host frame layout hints. Missing layout — or a declared section with empty
    ``children`` — falls back to the host inferring section contents from the
    component tree."""

    model_config = ConfigDict(extra="forbid")

    sections: list[SurfaceSectionLayout] | None = None
    actions: Literal["inline", "host_bar"] | None = None


class Surface(BaseModel):
    """The full declarative UI carried at ``definition["surface"]``."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = SURFACE_VERSION
    components: list[SurfaceComponent] = Field(min_length=1)
    data_model: dict[str, Any] = Field(default_factory=dict)
    bindings: list[ActionBinding] = Field(default_factory=list)
    runtime_controls: SurfaceRuntimeControls | None = None
    layout: SurfaceLayout | None = None


def resolve_pointer(data: Any, pointer: str) -> tuple[bool, Any]:
    """Resolve *pointer* against *data* (dict tree).

    Returns ``(True, value)`` when every segment exists, else ``(False, None)``.
    Never raises — used by validation (missing → warning) and the scaffold
    self-check.
    """
    if not is_valid_pointer(pointer):
        return (False, None)
    current = data
    for segment in pointer.strip("/").split("/"):
        if not isinstance(current, dict) or segment not in current:
            return (False, None)
        current = current[segment]
    return (True, current)


__all__ = [
    "IDENTIFIER_PATTERN",
    "POINTER_PATTERN",
    "SURFACE_VERSION",
    "ActionBinding",
    "DynamicValue",
    "OutputTarget",
    "PathRef",
    "RunOptions",
    "Surface",
    "SurfaceComponent",
    "SurfaceLayout",
    "SurfaceRuntimeControls",
    "SurfaceSectionLayout",
    "is_valid_identifier",
    "is_valid_pointer",
    "resolve_pointer",
]
