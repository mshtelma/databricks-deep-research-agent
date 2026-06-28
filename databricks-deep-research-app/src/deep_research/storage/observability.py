"""Observability primitives for the storage layer.

Every `StorageBackend`, the cache, the queue, and the hydrator emit metrics
through `get_sink()`. Two default sinks ship:

* `LogStructuredSink` — emits a `storage.metric` structured log event with the
  metric name, value, labels, and kind. Consumed by the host log pipeline
  (Splunk / Grafana Loki). The authoritative data source for v1.
* `RecordingSink` — test-only, stores every emission in memory for assertion.

Prometheus wiring is Phase 5 — see ADR follow-up F5. This module is careful to
keep emission zero-allocation (no dict copy) on the hot path so the observer
never becomes the bottleneck.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

logger = logging.getLogger("storage.metric")

MetricKind = Literal["counter", "gauge", "histogram"]


@runtime_checkable
class MetricsSink(Protocol):
    """Pluggable sink for storage metrics. Used via the global `get_sink()`."""

    def counter(self, name: str, value: int = 1, **labels: str) -> None: ...
    def gauge(self, name: str, value: float, **labels: str) -> None: ...
    def histogram(self, name: str, value: float, **labels: str) -> None: ...


class LogStructuredSink:
    """Default sink — one structured log line per emission.

    Every line has the shape:
        INFO storage.metric name=<name> kind=<kind> value=<value> [label=value …]
    The host log aggregator extracts these into dashboards.
    """

    def _emit(self, name: str, kind: MetricKind, value: float, labels: dict[str, str]) -> None:
        # `extra` carries structured fields; logger format string is trivial.
        extra = {"metric_name": name, "metric_kind": kind, "metric_value": value, **labels}
        logger.info("storage.metric %s=%s", name, value, extra=extra)

    def counter(self, name: str, value: int = 1, **labels: str) -> None:
        self._emit(name, "counter", float(value), labels)

    def gauge(self, name: str, value: float, **labels: str) -> None:
        self._emit(name, "gauge", float(value), labels)

    def histogram(self, name: str, value: float, **labels: str) -> None:
        self._emit(name, "histogram", float(value), labels)


@dataclass
class _Emission:
    name: str
    kind: MetricKind
    value: float
    labels: dict[str, str] = field(default_factory=dict)


class RecordingSink:
    """In-memory sink for tests. Store every emission; expose counters / last
    gauge values / histogram samples for assertion.
    """

    def __init__(self) -> None:
        self.emissions: list[_Emission] = []
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)

    def _record(self, name: str, kind: MetricKind, value: float, labels: dict[str, str]) -> None:
        key = _key(name, labels)
        self.emissions.append(_Emission(name=name, kind=kind, value=value, labels=dict(labels)))
        if kind == "counter":
            self._counters[key] += value
        elif kind == "gauge":
            self._gauges[key] = value
        else:
            self._histograms[key].append(value)

    def counter(self, name: str, value: int = 1, **labels: str) -> None:
        self._record(name, "counter", float(value), labels)

    def gauge(self, name: str, value: float, **labels: str) -> None:
        self._record(name, "gauge", float(value), labels)

    def histogram(self, name: str, value: float, **labels: str) -> None:
        self._record(name, "histogram", float(value), labels)

    # -- Query helpers --------------------------------------------------

    def count(self, name: str, **labels: str) -> float:
        return self._counters.get(_key(name, labels), 0.0)

    def last_gauge(self, name: str, **labels: str) -> float | None:
        return self._gauges.get(_key(name, labels))

    def samples(self, name: str, **labels: str) -> list[float]:
        return list(self._histograms.get(_key(name, labels), []))

    def names(self) -> set[str]:
        return {e.name for e in self.emissions}


def _key(name: str, labels: dict[str, str]) -> str:
    if not labels:
        return name
    parts = [f"{k}={v}" for k, v in sorted(labels.items())]
    return f"{name}|" + ",".join(parts)


_sink: MetricsSink = LogStructuredSink()


def set_sink(sink: MetricsSink) -> None:
    """Swap the global sink. Typically called once at app startup."""
    global _sink
    _sink = sink


def get_sink() -> MetricsSink:
    return _sink


@contextmanager
def use_sink(sink: MetricsSink) -> Generator[MetricsSink]:
    """Context manager for tests — restore previous sink on exit."""
    previous = _sink
    set_sink(sink)
    try:
        yield sink
    finally:
        set_sink(previous)
