"""Parity tests for the Designer inspector field-group taxonomy.

Guarantees every ``AgentNodeConfig`` field is either placed in a group or
explicitly hidden, so a newly-added framework knob can never silently fall out
of the Designer inspector (it would fail this test instead).
"""

from __future__ import annotations

import pytest
from databricks_deep_research.agents.config import AgentNodeConfig

from deep_research.agent_designer.field_groups import (
    ADVANCED_GROUPS,
    FIELD_GROUPS,
    GROUP_ORDER,
    HIDDEN_FIELDS,
    group_sort_key,
)

_VALID_WIDGETS = {"json"}


def test_every_agent_field_is_grouped_or_hidden() -> None:
    """Every AgentNodeConfig field must be grouped or explicitly hidden."""
    model_fields = set(AgentNodeConfig.model_fields)
    placed = set(FIELD_GROUPS) | HIDDEN_FIELDS
    missing = model_fields - placed
    assert not missing, (
        f"AgentNodeConfig fields with no group + not hidden: {sorted(missing)}. "
        "Add each to FIELD_GROUPS or HIDDEN_FIELDS in field_groups.py."
    )


def test_no_unknown_or_double_placed_fields() -> None:
    """FIELD_GROUPS/HIDDEN_FIELDS only reference real fields, and never both."""
    model_fields = set(AgentNodeConfig.model_fields)
    stale = (set(FIELD_GROUPS) | HIDDEN_FIELDS) - model_fields
    assert not stale, f"Taxonomy references fields not on AgentNodeConfig: {sorted(stale)}"
    both = set(FIELD_GROUPS) & HIDDEN_FIELDS
    assert not both, f"Fields both grouped and hidden: {sorted(both)}"


def test_group_names_are_declared_in_order() -> None:
    """Every group used by a field is listed in GROUP_ORDER, as is each advanced group."""
    used = {meta.group for meta in FIELD_GROUPS.values()}
    assert used <= set(GROUP_ORDER), f"Groups used but not in GROUP_ORDER: {sorted(used - set(GROUP_ORDER))}"
    assert set(GROUP_ORDER) >= ADVANCED_GROUPS, "ADVANCED_GROUPS must all appear in GROUP_ORDER"


def test_widget_overrides_are_valid() -> None:
    bad = {name: meta.widget for name, meta in FIELD_GROUPS.items() if meta.widget not in (None, *_VALID_WIDGETS)}
    assert not bad, f"Unknown widget overrides: {bad}"


def test_orders_are_unique_within_group() -> None:
    """Deterministic ordering: no two fields in a group share an order value."""
    seen: dict[str, set[int]] = {}
    for name, meta in FIELD_GROUPS.items():
        orders = seen.setdefault(meta.group, set())
        assert meta.order not in orders, f"Duplicate order {meta.order} in group {meta.group!r} (field {name})"
        orders.add(meta.order)


@pytest.mark.parametrize(
    ("group", "expected_rank"),
    [(g, i) for i, g in enumerate(GROUP_ORDER)],
)
def test_group_sort_key_follows_declared_order(group: str, expected_rank: int) -> None:
    assert group_sort_key(group)[0] == expected_rank


def test_unknown_group_sorts_last() -> None:
    assert group_sort_key("ZZZ unknown")[0] == len(GROUP_ORDER)
