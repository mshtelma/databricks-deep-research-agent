"""Unit tests for agent_designer_metrics observability helpers.

Each test swaps the global MetricsSink to a RecordingSink via use_sink(),
calls one helper, and asserts the emission was recorded exactly once with
the expected name, kind, and labels.
"""
from __future__ import annotations

import pytest

from deep_research.observability.agent_designer_metrics import (
    _ARGS_SUMMARY_MAX_CHARS,
    log_chat_mutation,
    record_etag_conflict,
    record_registry_fetch,
    record_save_latency,
    record_validation_error,
)
from deep_research.storage.observability import RecordingSink, use_sink


# ---------------------------------------------------------------------------
# record_registry_fetch
# ---------------------------------------------------------------------------


def test_record_registry_fetch_emits_histogram() -> None:
    sink = RecordingSink()
    with use_sink(sink):
        record_registry_fetch(42.5)

    assert sink.names() == {"agent_designer.registry_fetch_ms"}
    samples = sink.samples("agent_designer.registry_fetch_ms")
    assert samples == [42.5]


def test_record_registry_fetch_emits_exactly_once() -> None:
    sink = RecordingSink()
    with use_sink(sink):
        record_registry_fetch(1.0)
        record_registry_fetch(2.0)

    assert len(sink.emissions) == 2


# ---------------------------------------------------------------------------
# record_validation_error
# ---------------------------------------------------------------------------


def test_record_validation_error_increments_counter() -> None:
    sink = RecordingSink()
    with use_sink(sink):
        record_validation_error("validation")

    assert "agent_designer.validation_error" in sink.names()
    total = sink.count("agent_designer.validation_error", error_kind="validation")
    assert total == 1.0


def test_record_validation_error_uses_error_kind_label() -> None:
    sink = RecordingSink()
    with use_sink(sink):
        record_validation_error("schema")
        record_validation_error("syntax")

    assert sink.count("agent_designer.validation_error", error_kind="schema") == 1.0
    assert sink.count("agent_designer.validation_error", error_kind="syntax") == 1.0
    # Different label keys must not bleed into each other
    assert sink.count("agent_designer.validation_error", error_kind="validation") == 0.0


# ---------------------------------------------------------------------------
# record_etag_conflict
# ---------------------------------------------------------------------------


def test_record_etag_conflict_increments_counter() -> None:
    sink = RecordingSink()
    with use_sink(sink):
        record_etag_conflict()

    assert sink.count("agent_designer.save_etag_conflict") == 1.0


def test_record_etag_conflict_emits_exactly_once_per_call() -> None:
    sink = RecordingSink()
    with use_sink(sink):
        record_etag_conflict()
        record_etag_conflict()

    assert sink.count("agent_designer.save_etag_conflict") == 2.0


# ---------------------------------------------------------------------------
# record_save_latency
# ---------------------------------------------------------------------------


def test_record_save_latency_emits_histogram_for_create() -> None:
    sink = RecordingSink()
    with use_sink(sink):
        record_save_latency("create", 150.0)

    samples = sink.samples("agent_designer.designer_save_latency", operation="create")
    assert samples == [150.0]


def test_record_save_latency_emits_histogram_for_update() -> None:
    sink = RecordingSink()
    with use_sink(sink):
        record_save_latency("update", 200.0)

    samples = sink.samples("agent_designer.designer_save_latency", operation="update")
    assert samples == [200.0]


def test_record_save_latency_operation_label_separates_buckets() -> None:
    sink = RecordingSink()
    with use_sink(sink):
        record_save_latency("create", 100.0)
        record_save_latency("update", 999.0)

    assert sink.samples("agent_designer.designer_save_latency", operation="create") == [100.0]
    assert sink.samples("agent_designer.designer_save_latency", operation="update") == [999.0]


# ---------------------------------------------------------------------------
# log_chat_mutation — truncation behaviour
# ---------------------------------------------------------------------------


def test_log_chat_mutation_truncates_args_summary(caplog: pytest.LogCaptureFixture) -> None:
    """args_summary must be serialised and truncated to _ARGS_SUMMARY_MAX_CHARS chars."""
    import logging

    long_value = "x" * 1000
    args = {"key": long_value}

    with caplog.at_level(logging.INFO, logger="agent_designer.metrics"):
        log_chat_mutation("add_block", args, 0, "success")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    # The args_summary field in the structured extra must be truncated
    assert len(record.args_summary) <= _ARGS_SUMMARY_MAX_CHARS  # type: ignore[attr-defined]


def test_log_chat_mutation_includes_tool_name_and_outcome(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="agent_designer.metrics"):
        log_chat_mutation("delete_block", {"path": "root.child"}, 0, "success")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.tool_name == "delete_block"  # type: ignore[attr-defined]
    assert record.outcome == "success"  # type: ignore[attr-defined]
    assert record.validation_errors_count == 0  # type: ignore[attr-defined]


def test_log_chat_mutation_records_validation_errors_count(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.INFO, logger="agent_designer.metrics"):
        log_chat_mutation("propose_workflow", {"intent": "build a research agent"}, 3, "validation_failed")

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.validation_errors_count == 3  # type: ignore[attr-defined]
    assert record.outcome == "validation_failed"  # type: ignore[attr-defined]


def test_log_chat_mutation_exact_truncation_boundary(caplog: pytest.LogCaptureFixture) -> None:
    """Verify the boundary condition: exactly MAX chars survives, MAX+1 is cut."""
    import logging

    # Construct args whose JSON repr lands just over the limit
    boundary_value = "a" * (_ARGS_SUMMARY_MAX_CHARS + 50)
    args = {"k": boundary_value}

    with caplog.at_level(logging.INFO, logger="agent_designer.metrics"):
        log_chat_mutation("update_block", args, 0, "success")

    record = caplog.records[0]
    assert len(record.args_summary) == _ARGS_SUMMARY_MAX_CHARS  # type: ignore[attr-defined]
