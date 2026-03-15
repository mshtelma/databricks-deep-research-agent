"""Condition system for loop exits and conditional branching."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StateCondition(BaseModel):
    """Evaluate a condition against a dot-path state key."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["state"] = "state"
    key: str  # dot-path state key, e.g. "research.step_count"
    operator: str  # eq, neq, gt, lt, gte, lte, contains, in, exists, not_exists
    value: Any = None


class LLMCondition(BaseModel):
    """Evaluate a condition by asking an LLM a yes/no question."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["llm"] = "llm"
    prompt_template: str
    model_tier: str = "simple"
    expected_output: str = "yes"


class CompositeCondition(BaseModel):
    """Combine multiple conditions with boolean logic."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["composite"] = "composite"
    operator: str  # all, any, not
    conditions: list[Condition]


# Discriminated union of all condition types.
Condition = Annotated[
    StateCondition | LLMCondition | CompositeCondition,
    Field(discriminator="type"),
]

# Resolve forward references so Pydantic can handle the recursive type.
CompositeCondition.model_rebuild()


class ConditionBranch(BaseModel):
    """A branch in a conditional node: condition + index of the child to run."""

    model_config = ConfigDict(extra="forbid")
    condition: StateCondition | LLMCondition | CompositeCondition
    child_index: int


# ---------------------------------------------------------------------------
# Helper constructors
# ---------------------------------------------------------------------------


def all_of(*conditions: StateCondition | LLMCondition | CompositeCondition) -> CompositeCondition:
    """Shorthand for a composite AND condition."""
    return CompositeCondition(operator="all", conditions=list(conditions))


def any_of(*conditions: StateCondition | LLMCondition | CompositeCondition) -> CompositeCondition:
    """Shorthand for a composite OR condition."""
    return CompositeCondition(operator="any", conditions=list(conditions))


def negate(condition: StateCondition | LLMCondition | CompositeCondition) -> CompositeCondition:
    """Shorthand for a composite NOT condition (wraps a single child)."""
    return CompositeCondition(operator="not", conditions=[condition])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

_MISSING = object()


def resolve_dot_path(obj: Any, path: str) -> Any:
    """Resolve a dot-separated path like ``'a.b.c'`` on dicts or objects.

    Returns ``_MISSING`` sentinel (module-private) when any segment is absent.
    """
    current = obj
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current.get(segment, _MISSING)
        else:
            current = getattr(current, segment, _MISSING)
        if current is _MISSING:
            return _MISSING
    return current


def evaluate_state_condition(condition: StateCondition, state: Any) -> bool:
    """Evaluate a :class:`StateCondition` against *state*.

    Supports operators: eq, neq, gt, lt, gte, lte, contains, in, exists,
    not_exists.
    """
    resolved = resolve_dot_path(state, condition.key)
    op = condition.operator
    val = condition.value

    if op == "exists":
        return resolved is not _MISSING
    if op == "not_exists":
        return resolved is _MISSING

    # All remaining operators require the key to be present.
    if resolved is _MISSING:
        return False

    if op == "eq":
        return bool(resolved == val)
    if op == "neq":
        return bool(resolved != val)
    if op == "gt":
        return bool(resolved > val)
    if op == "lt":
        return bool(resolved < val)
    if op == "gte":
        return bool(resolved >= val)
    if op == "lte":
        return bool(resolved <= val)
    if op == "contains":
        return val in resolved
    if op == "in":
        return resolved in val

    msg = f"Unknown operator: {op}"
    raise ValueError(msg)
