"""Tests for value-ranked history compaction (spec §1.3).

All pure/mocked — no network, no LLM. The pure ranking/compaction helpers are
tested directly; the ReactLoop integration is exercised by constructing a
minimal loop (MagicMock LLM, no tools) and driving ``_compact_old_tool_results``
with hand-built message lists. The headline guarantee — that with
``evidence_rescue`` off OR empty value metadata the compactor is byte-identical
to the §1.2 ladder — is enforced by golden comparison against a loop run with
the rescue disabled.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock

from databricks_deep_research.agents.history_compaction import (
    RANK_ACCEPTED,
    RANK_BUILTIN,
    RANK_UNACCEPTED,
    compact_tool_message,
    rank_tool_results,
    value_rank,
)
from databricks_deep_research.agents.react_loop import ReactLoop
from databricks_deep_research.agents.tool_offload import hard_clip, line_preserving_truncate
from databricks_deep_research.events.types import ConversationCompactedEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unaccepted(handle: str | None = None) -> dict[str, Any]:
    return {"accepted_source_count": 0, "evidence_quality": "", "offload_handle": handle}


def _builtin() -> dict[str, Any]:
    return {"accepted_source_count": 0, "evidence_quality": "builtin", "offload_handle": None}


def _accepted(n: int = 3) -> dict[str, Any]:
    return {"accepted_source_count": n, "evidence_quality": "web", "offload_handle": None}


def _assistant_tc(tc_id: str) -> dict[str, Any]:
    return {"role": "assistant", "tool_calls": [{"id": tc_id}]}


def _tool_msg(tc_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tc_id, "content": content}


def _make_loop(**kwargs: Any) -> ReactLoop:
    return ReactLoop(llm_client=MagicMock(), tools=[], node_id="test", **kwargs)


# ---------------------------------------------------------------------------
# value_rank
# ---------------------------------------------------------------------------


def test_value_rank_tiers() -> None:
    assert value_rank(_unaccepted()) == RANK_UNACCEPTED
    assert value_rank(_builtin()) == RANK_BUILTIN
    assert value_rank(_accepted()) == RANK_ACCEPTED
    # Accepted dominates builtin even if both signals present.
    assert value_rank({"accepted_source_count": 1, "evidence_quality": "builtin"}) == RANK_ACCEPTED


def test_value_rank_handles_garbage() -> None:
    assert value_rank({"accepted_source_count": "x"}) == RANK_UNACCEPTED
    assert value_rank({}) == RANK_UNACCEPTED


# ---------------------------------------------------------------------------
# rank_tool_results — orders unaccepted < builtin < accepted (lowest first)
# ---------------------------------------------------------------------------


def test_rank_orders_lowest_value_first() -> None:
    messages = [
        _tool_msg("acc", "a"),
        _tool_msg("blt", "b"),
        _tool_msg("una", "c"),
    ]
    value = {"acc": _accepted(), "blt": _builtin(), "una": _unaccepted()}
    ranked = rank_tool_results(messages, value)
    assert [r.tool_call_id for r in ranked] == ["una", "blt", "acc"]
    assert [r.rank for r in ranked] == [RANK_UNACCEPTED, RANK_BUILTIN, RANK_ACCEPTED]


def test_rank_stable_by_index_within_tier() -> None:
    messages = [
        _tool_msg("una1", "a"),
        _tool_msg("una2", "b"),
        _tool_msg("una3", "c"),
    ]
    value = {"una1": _unaccepted(), "una2": _unaccepted(), "una3": _unaccepted()}
    ranked = rank_tool_results(messages, value)
    # Same tier => older (lower index) first.
    assert [r.index for r in ranked] == [0, 1, 2]


def test_rank_skips_messages_without_value_metadata() -> None:
    messages = [_tool_msg("known", "a"), _tool_msg("unknown", "b")]
    ranked = rank_tool_results(messages, {"known": _unaccepted()})
    assert [r.tool_call_id for r in ranked] == ["known"]


def test_rank_respects_upto_window() -> None:
    messages = [_tool_msg("a", "x"), _tool_msg("b", "y"), _tool_msg("c", "z")]
    value = {"a": _unaccepted(), "b": _unaccepted(), "c": _unaccepted()}
    ranked = rank_tool_results(messages, value, upto=2)
    assert [r.tool_call_id for r in ranked] == ["a", "b"]


# ---------------------------------------------------------------------------
# compact_tool_message — handle pointer vs rungs
# ---------------------------------------------------------------------------


def test_compact_offloaded_message_to_handle_pointer() -> None:
    content = "x" * 5000
    out = compact_tool_message(
        content, _unaccepted(handle="web_search_2"), max_chars=800, line_preserving=False
    )
    assert out == "[result in compute var web_search_2]"


def test_compact_non_offloaded_delegates_to_rungs() -> None:
    content = "narrative line\n" + "| col | 123 |\n" * 200
    lp = compact_tool_message(content, _accepted(), max_chars=800, line_preserving=True)
    assert lp == line_preserving_truncate(content, max_chars=800)
    hc = compact_tool_message(content, _accepted(), max_chars=800, line_preserving=False)
    assert hc == hard_clip(content, 800)


def test_compact_empty_handle_falls_through_to_rung() -> None:
    content = "x" * 5000
    out = compact_tool_message(
        content, _unaccepted(handle=""), max_chars=800, line_preserving=False
    )
    assert out == hard_clip(content, 800)


# ---------------------------------------------------------------------------
# ReactLoop integration: rescue protects accepted, sheds unaccepted first
# ---------------------------------------------------------------------------


def _big(tag: str, n: int = 5000) -> str:
    return f"{tag}-" + "x" * n


def test_rescue_compacts_unaccepted_keeps_accepted_under_budget() -> None:
    # Budget chosen so that shedding the one big unaccepted result (5KB -> ~820)
    # brings the window under budget, leaving the (just-over-cap) accepted result
    # protected. una(~820) + acc(900) = ~1720 < 2000.
    loop = _make_loop(max_result_chars=800, compaction_budget_chars=2000)
    una_content = _big("UNA")  # ~5KB, lowest value
    acc_content = "ACC-" + "y" * 900  # 904 chars, just over the cap but high value
    messages = [
        _assistant_tc("una"),
        _tool_msg("una", una_content),
        _assistant_tc("acc"),
        _tool_msg("acc", acc_content),
        _assistant_tc("last"),  # boundary: keeps last iteration intact
    ]
    loop._tool_msg_value = {"una": _unaccepted(), "acc": _accepted()}

    loop._compact_old_tool_results(messages)

    # Unaccepted (lowest value) is compacted; accepted survives intact because
    # shedding the unaccepted result already brought us under budget.
    assert messages[1]["content"] != una_content
    assert messages[3]["content"] == acc_content


def test_rescue_offloaded_message_becomes_handle_pointer() -> None:
    loop = _make_loop(max_result_chars=800)
    messages = [
        _assistant_tc("off"),
        _tool_msg("off", _big("OFF")),
        _assistant_tc("last"),
    ]
    loop._tool_msg_value = {"off": _unaccepted(handle="web_search_0")}

    loop._compact_old_tool_results(messages)

    assert messages[1]["content"] == "[result in compute var web_search_0]"


def test_rescue_emits_event_with_counts_when_compaction_occurs() -> None:
    loop = _make_loop(max_result_chars=800)
    big = _big("UNA")
    messages = [
        _assistant_tc("una"),
        _tool_msg("una", big),
        _assistant_tc("last"),
    ]
    loop._tool_msg_value = {"una": _unaccepted()}

    events = loop._compact_old_tool_results(messages)

    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, ConversationCompactedEvent)
    assert ev.event_type == "conversation_compacted"
    assert ev.node_id == "test"
    assert ev.tokens_saved == len(big) - len(messages[1]["content"])
    assert ev.tokens_saved > 0


def test_no_event_when_nothing_changes() -> None:
    loop = _make_loop(max_result_chars=800)
    # Single tool-call round => last_tc_idx == 0 => nothing to compact.
    messages = [_assistant_tc("una"), _tool_msg("una", _big("UNA"))]
    loop._tool_msg_value = {"una": _unaccepted()}

    events = loop._compact_old_tool_results(messages)

    assert events == []
    assert messages[1]["content"] == _big("UNA")  # untouched


# ---------------------------------------------------------------------------
# GOLDEN: byte-identical to §1.2 ladder when rescue off / no value metadata
# ---------------------------------------------------------------------------


def _golden_messages() -> list[dict[str, Any]]:
    """A multi-round transcript with oversized tool results to force compaction."""
    msgs: list[dict[str, Any]] = []
    for i in range(4):
        tag = f"t{i}"
        msgs.append(_assistant_tc(tag))
        # Mixed narrative + tabular so line_preserving has something to keep.
        body = (
            f"narrative prose for round {i} that should be dropped\n"
            + f"| metric_{i} | {i * 111} | million |\n" * 300
        )
        msgs.append(_tool_msg(tag, body))
    msgs.append(_assistant_tc("final"))  # boundary kept intact
    return msgs


def _run_compaction(strategy: str, *, evidence_rescue: bool, with_value: bool,
                    offload: str = "off") -> list[dict[str, Any]]:
    loop = _make_loop(
        max_result_chars=800,
        compaction_strategy=strategy,
        max_tool_calls=4,  # => _compact_after_rounds = max(2, 4*2//5) = 2
        keep_intact_iterations=1,
        evidence_rescue=evidence_rescue,
        tool_output_offload=offload,
    )
    if with_value:
        loop._tool_msg_value = {
            f"t{i}": (_accepted() if i % 2 else _unaccepted()) for i in range(4)
        }
    messages = _golden_messages()
    loop._compact_old_tool_results(messages)
    return messages


def test_golden_truncate_rescue_off_byte_identical() -> None:
    """evidence_rescue=False reproduces the §1.2 truncate (hard_clip) output."""
    baseline = _run_compaction("truncate", evidence_rescue=False, with_value=True)
    # The §1.2 reference: rescue disabled is the canonical ladder. Recompute the
    # expected contents directly from the rung to pin the bytes.
    ref = _golden_messages()
    last_tc_idx = max(
        i for i, m in enumerate(ref) if m.get("role") == "assistant" and m.get("tool_calls")
    )
    for i in range(last_tc_idx):
        m = ref[i]
        c = m.get("content", "")
        if m.get("role") == "tool" and isinstance(c, str) and len(c) > 800:
            m["content"] = hard_clip(c, 800)
    assert [m.get("content") for m in baseline] == [m.get("content") for m in ref]


def test_golden_truncate_empty_value_metadata_byte_identical() -> None:
    """Empty value metadata => byte-identical to rescue-off truncate ladder."""
    rescue_off = _run_compaction("truncate", evidence_rescue=False, with_value=True)
    no_value = _run_compaction("truncate", evidence_rescue=True, with_value=False)
    assert [m.get("content") for m in no_value] == [m.get("content") for m in rescue_off]


def test_golden_mask_rescue_off_byte_identical() -> None:
    """evidence_rescue=False reproduces the §1.2 mask (line_preserving) output."""
    rescue_off = _run_compaction("mask", evidence_rescue=False, with_value=True)
    ref = _golden_messages()
    tc_indices = [
        i for i, m in enumerate(ref) if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    n = min(1, len(tc_indices))  # keep_intact_iterations=1
    keep_from = tc_indices[-n] if n > 0 else 0
    for i in range(keep_from):
        m = ref[i]
        if m.get("role") == "tool":
            c = m.get("content", "")
            if isinstance(c, str) and len(c) > 800:
                m["content"] = line_preserving_truncate(c, max_chars=800)
    assert [m.get("content") for m in rescue_off] == [m.get("content") for m in ref]


def test_golden_mask_empty_value_metadata_byte_identical() -> None:
    rescue_off = _run_compaction("mask", evidence_rescue=False, with_value=True)
    no_value = _run_compaction("mask", evidence_rescue=True, with_value=False)
    assert [m.get("content") for m in no_value] == [m.get("content") for m in rescue_off]


def test_rescue_budget_zero_matches_ladder_for_non_offloaded() -> None:
    """With budget=0 and no offloaded results, rescue (on, with value) produces
    the same per-message bytes as the rescue-off ladder — only reordered."""
    rescue_off = _run_compaction("truncate", evidence_rescue=False, with_value=True)
    rescue_on = _run_compaction("truncate", evidence_rescue=True, with_value=True)
    # Same final content set (offload off => no handle pointers; budget 0 =>
    # every over-cap result compacted by the same rung).
    assert [m.get("content") for m in rescue_on] == [m.get("content") for m in rescue_off]


def test_rescue_disabled_does_not_touch_side_table_path() -> None:
    """Even with value metadata present, evidence_rescue=False uses the ladder."""
    loop = _make_loop(max_result_chars=800, evidence_rescue=False)
    loop._tool_msg_value = {"off": _unaccepted(handle="web_search_0")}
    messages = [
        _assistant_tc("off"),
        _tool_msg("off", _big("OFF")),
        _assistant_tc("last"),
    ]
    before = copy.deepcopy(messages)
    loop._compact_old_tool_results(messages)
    # Ladder hard-clips; it must NOT emit the handle pointer.
    assert messages[1]["content"] != "[result in compute var web_search_0]"
    assert messages[1]["content"] == hard_clip(before[1]["content"], 800)
