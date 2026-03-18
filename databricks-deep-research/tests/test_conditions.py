"""Tests for the condition evaluation system."""

from __future__ import annotations

import pytest

from databricks_deep_research.workflow.conditions import (
    _MISSING,
    CompositeCondition,
    StateCondition,
    all_of,
    any_of,
    evaluate_state_condition,
    negate,
    resolve_dot_path,
)
from databricks_deep_research.workflow.state import WorkflowState

# ---------------------------------------------------------------------------
# resolve_dot_path
# ---------------------------------------------------------------------------


class TestResolveDotPath:
    def test_simple_dict_key(self) -> None:
        obj = {"a": 1}
        assert resolve_dot_path(obj, "a") == 1

    def test_nested_dict_keys(self) -> None:
        obj = {"a": {"b": {"c": 42}}}
        assert resolve_dot_path(obj, "a.b.c") == 42

    def test_missing_key_returns_sentinel(self) -> None:
        obj = {"a": 1}
        assert resolve_dot_path(obj, "missing") is _MISSING

    def test_missing_nested_returns_sentinel(self) -> None:
        obj = {"a": {"b": 1}}
        assert resolve_dot_path(obj, "a.x.y") is _MISSING

    def test_attribute_access_on_object(self) -> None:
        class Obj:
            x = 10
        assert resolve_dot_path(Obj(), "x") == 10


# ---------------------------------------------------------------------------
# evaluate_state_condition — operator coverage
# ---------------------------------------------------------------------------


class TestEvaluateStateCondition:
    @pytest.mark.parametrize(
        ("condition", "data", "expected"),
        [
            (StateCondition(key="status", operator="eq", value="done"), {"status": "done"}, True),
            (StateCondition(key="status", operator="eq", value="done"), {"status": "pending"}, False),
            (StateCondition(key="count", operator="neq", value=0), {"count": 5}, True),
            (StateCondition(key="count", operator="gt", value=3), {"count": 5}, True),
            (StateCondition(key="count", operator="gt", value=3), {"count": 2}, False),
            (StateCondition(key="count", operator="lt", value=3), {"count": 1}, True),
            (StateCondition(key="count", operator="gte", value=3), {"count": 3}, True),
            (StateCondition(key="count", operator="lte", value=3), {"count": 3}, True),
            (StateCondition(key="tags", operator="contains", value="ai"), {"tags": ["ai", "ml"]}, True),
            (StateCondition(key="tags", operator="contains", value="ai"), {"tags": ["web"]}, False),
            (
                StateCondition(key="status", operator="in", value=["done", "complete"]),
                {"status": "done"},
                True,
            ),
            (
                StateCondition(key="status", operator="in", value=["done", "complete"]),
                {"status": "pending"},
                False,
            ),
            (StateCondition(key="result", operator="exists"), {"result": "yes"}, True),
            (StateCondition(key="result", operator="exists"), {}, False),
            (StateCondition(key="missing", operator="not_exists"), {}, True),
            (StateCondition(key="missing", operator="not_exists"), {"missing": 1}, False),
        ],
    )
    def test_operator_matrix(
        self,
        condition: StateCondition,
        data: dict[str, object],
        expected: bool,
    ) -> None:
        assert evaluate_state_condition(condition, data) is expected

    def test_missing_key_returns_false_for_comparison_ops(self) -> None:
        cond = StateCondition(key="x", operator="eq", value=1)
        assert evaluate_state_condition(cond, {}) is False

    def test_unknown_operator_raises(self) -> None:
        cond = StateCondition(key="x", operator="bogus", value=1)
        with pytest.raises(ValueError, match="Unknown operator"):
            evaluate_state_condition(cond, {"x": 1})

    def test_dot_path_key(self) -> None:
        cond = StateCondition(key="research.step_count", operator="gte", value=3)
        assert evaluate_state_condition(cond, {"research": {"step_count": 5}}) is True


# ---------------------------------------------------------------------------
# evaluate_state_condition with WorkflowState
# ---------------------------------------------------------------------------


class TestConditionWithWorkflowState:
    def test_state_get_via_dot_path(self) -> None:
        """WorkflowState stores entries by key; dot-path resolves into the value."""
        state = WorkflowState(query="test")
        state.append("node1", "research", {"step_count": 5, "status": "done"})

        # The condition uses resolve_dot_path on the state object.
        # WorkflowState is not a dict, so resolve_dot_path will use getattr.
        # The top-level key 'research' is a WorkflowState attribute (log entry).
        # We test against a plain dict representation instead.
        data = {"research": state.get("research")}
        cond = StateCondition(key="research.step_count", operator="gte", value=3)
        assert evaluate_state_condition(cond, data) is True


# ---------------------------------------------------------------------------
# Composite conditions (all_of / any_of / negate helpers)
# ---------------------------------------------------------------------------


class TestCompositeConditions:
    def test_all_of_true(self) -> None:
        c = all_of(
            StateCondition(key="a", operator="eq", value=1),
            StateCondition(key="b", operator="eq", value=2),
        )
        assert c.operator == "all"
        assert len(c.conditions) == 2

    def test_any_of_construction(self) -> None:
        c = any_of(
            StateCondition(key="a", operator="eq", value=1),
            StateCondition(key="b", operator="eq", value=2),
        )
        assert c.operator == "any"
        assert len(c.conditions) == 2

    def test_negate_construction(self) -> None:
        inner = StateCondition(key="x", operator="eq", value=1)
        c = negate(inner)
        assert c.operator == "not"
        assert len(c.conditions) == 1
        assert c.conditions[0] == inner

    def test_composite_serialisation_roundtrip(self) -> None:
        """Composite conditions should serialize/deserialize via Pydantic."""
        c = all_of(
            StateCondition(key="a", operator="eq", value=1),
            any_of(
                StateCondition(key="b", operator="gt", value=0),
                StateCondition(key="c", operator="exists"),
            ),
        )
        data = c.model_dump()
        rebuilt = CompositeCondition.model_validate(data)
        assert rebuilt.operator == "all"
        assert len(rebuilt.conditions) == 2
