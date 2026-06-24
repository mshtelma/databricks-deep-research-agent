"""Unit tests for the admission query-profile builder.

These tests lock in the reserved-slots allocator and the node-level
``hint_queries`` plumbing introduced to prevent researcher-subtype
agents from starving their admission profile.

All tests are content-agnostic — no customer or competitor names are
baked into the fixtures. The reservations are validated by invariant,
not by magic numbers, so the tests tolerate future tuning.
"""

from __future__ import annotations

import pytest

from databricks_deep_research.agents import source_aware
from databricks_deep_research.agents.source_aware import (
    _ADMISSION_ENFORCE_NODE_HINTS_ENV,
    _ADMISSION_USE_SLOT_RESERVATIONS_ENV,
    _PROFILE_SLOT_RESERVATIONS,
    _PROFILE_TERM_BUDGET,
    _build_query_profile,
    _reserve_slots,
)

# ---------------------------------------------------------------------------
# _reserve_slots — pure allocator
# ---------------------------------------------------------------------------


def test_slot_reservations_sum_to_budget() -> None:
    """Invariant pinned by a module-level assert; re-check at test time
    so CI catches any future drift."""
    assert sum(_PROFILE_SLOT_RESERVATIONS.values()) == _PROFILE_TERM_BUDGET


def test_long_root_query_does_not_crowd_tool_query() -> None:
    """A verbose root query must not steal every slot from the tool
    query — that was the prod-regression root cause."""
    root = ["aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff", "gggg",
            "hhhh", "iiii", "jjjj", "kkkk", "llll", "mmmm"]
    step: list[str] = []
    hints: list[str] = []
    tool = ["unique1", "unique2", "unique3"]

    out = _reserve_slots(root, step, hints, tool)
    tool_reservation = _PROFILE_SLOT_RESERVATIONS["tool_query"]
    assert all(t in out for t in tool[:tool_reservation])


def test_node_hints_enter_profile() -> None:
    """Hint tokens get their reserved share even when root_query is
    already saturated."""
    root = ["rroot"] * 12        # would dominate without reservations
    hints = ["hinta", "hintb"]
    out = _reserve_slots(root, [], hints, [])
    hint_reservation = _PROFILE_SLOT_RESERVATIONS["hints"]
    assert all(h in out for h in hints[:hint_reservation])


def test_sparse_root_reclaims_unused_slots() -> None:
    """When one source has fewer tokens than its reservation, the
    overflow can be picked up from root_query so the profile stays
    full."""
    root = [f"root{i}" for i in range(20)]
    step = ["stepa"]               # uses 1 of 2 slots
    hints: list[str] = []
    tool = ["tool1"]               # uses 1 of 3 slots

    out = _reserve_slots(root, step, hints, tool)
    assert len(out) == _PROFILE_TERM_BUDGET


def test_budget_is_never_exceeded() -> None:
    """Even with more input than the budget, the output stays capped."""
    root = [f"r{i}" for i in range(50)]
    step = [f"s{i}" for i in range(50)]
    hints = [f"h{i}" for i in range(50)]
    tool = [f"t{i}" for i in range(50)]
    out = _reserve_slots(root, step, hints, tool)
    assert len(out) == _PROFILE_TERM_BUDGET


def test_dedup_preserves_first_occurrence() -> None:
    """A token common to multiple sources appears once, retaining its
    earliest slot ownership."""
    common = "shared"
    root = [common, "onlyroot"]
    step = [common, "onlystep"]
    hints: list[str] = []
    tool = [common, "onlytool"]

    out = _reserve_slots(root, step, hints, tool)
    assert out.count(common) == 1


def test_empty_everything_returns_empty() -> None:
    assert _reserve_slots([], [], [], []) == []


def test_zero_reservation_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting a reservation to 0 must not break allocation — that
    source simply contributes nothing from its reserved share
    (fallback fill may still pick up later)."""
    zeroed = {"root_query": 12, "step_text": 0, "hints": 0, "tool_query": 0}
    monkeypatch.setattr(source_aware, "_PROFILE_SLOT_RESERVATIONS", zeroed)
    out = _reserve_slots(["a", "b", "c"], ["stepx"], ["hintx"], ["toolx"])
    # Only root_query contributed.
    assert "stepx" not in out
    assert "hintx" not in out
    assert "toolx" not in out


# ---------------------------------------------------------------------------
# _build_query_profile — end-to-end via env flags
# ---------------------------------------------------------------------------


def test_legacy_behavior_when_flags_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """With both flags off we preserve the pre-change profile exactly
    (modulo the new optional parameter defaulting to None)."""
    monkeypatch.delenv(_ADMISSION_ENFORCE_NODE_HINTS_ENV, raising=False)
    monkeypatch.delenv(_ADMISSION_USE_SLOT_RESERVATIONS_ENV, raising=False)
    profile = _build_query_profile(
        current_step=None,
        root_query="help prepare meeting with long vocabulary terms here",
        tool_query="competitor analysis",
        node_hint_queries=["battle card"],
    )
    # Legacy behavior: node hints NOT merged when flag is off.
    assert "battle" not in profile["terms"]
    assert "card" not in profile["terms"]


def test_node_hints_honored_when_enforce_flag_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ADMISSION_ENFORCE_NODE_HINTS_ENV, "true")
    monkeypatch.setenv(_ADMISSION_USE_SLOT_RESERVATIONS_ENV, "true")
    profile = _build_query_profile(
        current_step=None,
        root_query="help prepare meeting with long vocabulary terms here",
        tool_query="",
        node_hint_queries=["competitor battle card"],
    )
    # At least one hint token in the profile.
    hint_tokens = {"competitor", "battle", "card"}
    assert profile["terms"], "profile should have terms"
    assert hint_tokens & set(profile["terms"]), (
        f"expected a hint token in {profile['terms']!r}"
    )


def test_tool_query_survives_long_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prod repro: long root query plus useful tool query — slot
    reservations guarantee the tool query makes it into the profile."""
    monkeypatch.setenv(_ADMISSION_USE_SLOT_RESERVATIONS_ENV, "true")
    long_root = (
        "help me prepare for an upcoming meeting with a partner that "
        "builds domain agents using platform capabilities and has many "
        "talk to data requirements and integrations"
    )
    profile = _build_query_profile(
        current_step=None,
        root_query=long_root,
        tool_query="top databricks alternatives market share",
        node_hint_queries=None,
    )
    # At least one tool_query token made it through the budget cap.
    tool_tokens = {"alternatives", "market", "share", "databricks"}
    assert tool_tokens & set(profile["terms"]), (
        f"tool_query tokens missing from profile={profile['terms']!r}"
    )


def test_profile_budget_under_slot_reservations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even under the new allocator the budget cap holds."""
    monkeypatch.setenv(_ADMISSION_USE_SLOT_RESERVATIONS_ENV, "true")
    profile = _build_query_profile(
        current_step=None,
        root_query=" ".join(f"wordx{i}" for i in range(30)),
        tool_query=" ".join(f"wordy{i}" for i in range(30)),
        node_hint_queries=[" ".join(f"wordz{i}" for i in range(30))],
    )
    assert len(profile["terms"]) <= _PROFILE_TERM_BUDGET


def test_kill_switch_disables_node_hints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flipping the env var off ignores node_hint_queries even when
    supplied."""
    monkeypatch.setenv(_ADMISSION_ENFORCE_NODE_HINTS_ENV, "false")
    profile = _build_query_profile(
        current_step=None,
        root_query="short query",
        tool_query=None,
        node_hint_queries=["competitor battle card"],
    )
    hint_tokens = {"competitor", "battle", "card"}
    assert not (hint_tokens & set(profile["terms"])), (
        f"hints leaked despite kill-switch: {profile['terms']!r}"
    )


def test_profile_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same inputs → same profile. Idempotence matters because the
    same admission call can fire repeatedly in a react_loop."""
    monkeypatch.setenv(_ADMISSION_ENFORCE_NODE_HINTS_ENV, "true")
    monkeypatch.setenv(_ADMISSION_USE_SLOT_RESERVATIONS_ENV, "true")
    kwargs: dict = dict(
        current_step=None,
        root_query="a reasonable customer query",
        tool_query="competitor alternative analysis",
        node_hint_queries=["battle card migration"],
    )
    first = _build_query_profile(**kwargs)
    second = _build_query_profile(**kwargs)
    assert first == second
