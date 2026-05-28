"""Condition system for loop exits and conditional branching."""

from __future__ import annotations

import json
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


class ConditionEvaluationError(ValueError):
    """Raised when a condition cannot be evaluated safely."""


def resolve_dot_path(obj: Any, path: str) -> Any:
    """Resolve a dot-separated path like ``'a.b.c'`` on dicts or objects.

    When a segment is a JSON-encoded string (common for agent outputs stored
    in workflow state), the function transparently parses it so that
    ``resolve_dot_path({"reflection": '{"decision":"complete"}'}, "reflection.decision")``
    returns ``"complete"``.

    Returns ``_MISSING`` sentinel (module-private) when any segment is absent.
    """
    current = obj
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current.get(segment, _MISSING)
        elif isinstance(current, str):
            # Agent outputs are stored as raw JSON strings in state.
            # Try to parse so dot-path navigation works.
            try:
                parsed = json.loads(current)
                current = parsed.get(segment, _MISSING) if isinstance(parsed, dict) else _MISSING
            except (json.JSONDecodeError, TypeError):
                current = getattr(current, segment, _MISSING)
        else:
            current = getattr(current, segment, _MISSING)
        if current is _MISSING:
            return _MISSING
    return current


def resolve_dot_path_strict(obj: Any, path: str) -> Any:
    """Resolve a dot-path and raise when any segment is absent."""

    current = obj
    traversed: list[str] = []
    for segment in path.split("."):
        traversed.append(segment)
        if isinstance(current, dict):
            if segment not in current:
                available = ", ".join(sorted(str(key) for key in current)) or "<none>"
                raise ConditionEvaluationError(
                    f"path segment {segment!r} is missing at "
                    f"{'.'.join(traversed[:-1]) or '<root>'}; "
                    f"available keys: {available}"
                )
            current = current[segment]
        elif isinstance(current, str):
            try:
                parsed = json.loads(current)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ConditionEvaluationError(
                    f"cannot descend into non-JSON string at "
                    f"{'.'.join(traversed[:-1]) or '<root>'}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ConditionEvaluationError(
                    f"cannot descend into JSON {type(parsed).__name__} at "
                    f"{'.'.join(traversed[:-1]) or '<root>'}"
                )
            if segment not in parsed:
                available = ", ".join(sorted(str(key) for key in parsed)) or "<none>"
                raise ConditionEvaluationError(
                    f"path segment {segment!r} is missing at "
                    f"{'.'.join(traversed[:-1]) or '<root>'}; "
                    f"available keys: {available}"
                )
            current = parsed[segment]
        else:
            if not hasattr(current, segment):
                raise ConditionEvaluationError(
                    f"path segment {segment!r} is missing at "
                    f"{'.'.join(traversed[:-1]) or '<root>'}"
                )
            current = getattr(current, segment)
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


def evaluate_state_condition_strict(condition: StateCondition, state: Any) -> bool:
    """Evaluate a state condition and raise on unsafe missing operands."""

    op = condition.operator
    val = condition.value

    if op in {"exists", "not_exists"}:
        resolved = resolve_dot_path(state, condition.key)
        return resolved is not _MISSING if op == "exists" else resolved is _MISSING

    resolved = resolve_dot_path_strict(state, condition.key)

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
    raise ConditionEvaluationError(msg)


def evaluate_condition_strict(
    condition: StateCondition | LLMCondition | CompositeCondition,
    state: Any,
) -> bool:
    """Evaluate a condition tree with strict state-path semantics."""

    if isinstance(condition, StateCondition):
        return evaluate_state_condition_strict(condition, state)

    if isinstance(condition, LLMCondition):
        raise ConditionEvaluationError(
            "LLM conditions are not executable by the synchronous workflow "
            "condition evaluator"
        )

    if condition.operator == "all":
        if not condition.conditions:
            raise ConditionEvaluationError("Composite 'all' requires child conditions")
        return all(evaluate_condition_strict(child, state) for child in condition.conditions)
    if condition.operator == "any":
        if not condition.conditions:
            raise ConditionEvaluationError("Composite 'any' requires child conditions")
        return any(evaluate_condition_strict(child, state) for child in condition.conditions)
    if condition.operator == "not":
        if len(condition.conditions) != 1:
            raise ConditionEvaluationError("Composite 'not' requires exactly one child")
        return not evaluate_condition_strict(condition.conditions[0], state)

    raise ConditionEvaluationError(f"Unknown composite operator: {condition.operator}")


def summarize_condition(condition: StateCondition | LLMCondition | CompositeCondition) -> str:
    """Return a compact, diagnostics-safe condition summary."""

    if isinstance(condition, StateCondition):
        if condition.operator in {"exists", "not_exists"}:
            return f"{condition.key} {condition.operator}"
        return f"{condition.key} {condition.operator} {condition.value!r}"
    if isinstance(condition, LLMCondition):
        return "llm condition"
    return f"composite {condition.operator}({len(condition.conditions)} conditions)"
