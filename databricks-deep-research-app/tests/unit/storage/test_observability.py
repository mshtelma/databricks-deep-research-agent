"""Unit tests for `deep_research.storage.observability`."""

from __future__ import annotations

from deep_research.storage.observability import (
    LogStructuredSink,
    MetricsSink,
    RecordingSink,
    get_sink,
    set_sink,
    use_sink,
)


class TestRecordingSink:
    def test_counter_accumulates(self) -> None:
        sink = RecordingSink()
        sink.counter("x", 3, backend="a")
        sink.counter("x", 2, backend="a")
        assert sink.count("x", backend="a") == 5

    def test_counter_label_isolation(self) -> None:
        sink = RecordingSink()
        sink.counter("x", 1, backend="a")
        sink.counter("x", 1, backend="b")
        assert sink.count("x", backend="a") == 1
        assert sink.count("x", backend="b") == 1

    def test_gauge_last_value_wins(self) -> None:
        sink = RecordingSink()
        sink.gauge("load", 1.0, backend="a")
        sink.gauge("load", 5.0, backend="a")
        assert sink.last_gauge("load", backend="a") == 5.0

    def test_histogram_samples_appended(self) -> None:
        sink = RecordingSink()
        sink.histogram("lat", 0.1, backend="a")
        sink.histogram("lat", 0.2, backend="a")
        assert sink.samples("lat", backend="a") == [0.1, 0.2]

    def test_count_defaults_to_zero(self) -> None:
        sink = RecordingSink()
        assert sink.count("absent") == 0.0

    def test_names_of_emissions(self) -> None:
        sink = RecordingSink()
        sink.counter("a")
        sink.gauge("b", 1)
        sink.histogram("c", 0.5)
        assert sink.names() == {"a", "b", "c"}


class TestSinkSwitching:
    def test_use_sink_context_manager_restores(self) -> None:
        original = get_sink()
        rec = RecordingSink()
        with use_sink(rec) as active:
            assert active is rec
            assert get_sink() is rec
        assert get_sink() is original

    def test_default_sink_is_log_structured(self) -> None:
        sink = get_sink()
        assert isinstance(sink, LogStructuredSink)

    def test_set_sink_replaces_global(self) -> None:
        original = get_sink()
        rec = RecordingSink()
        set_sink(rec)
        try:
            assert get_sink() is rec
        finally:
            set_sink(original)


def test_recording_sink_satisfies_protocol() -> None:
    assert isinstance(RecordingSink(), MetricsSink)


def test_log_structured_sink_satisfies_protocol() -> None:
    assert isinstance(LogStructuredSink(), MetricsSink)
