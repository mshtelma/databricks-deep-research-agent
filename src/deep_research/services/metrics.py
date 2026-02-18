"""Source Query Metrics Collection.

Provides in-memory metrics tracking for enterprise data source queries.
Tracks query counts, latencies, and errors per source type.

Part of 007-enterprise-data-sources feature (T108).

Usage:
    from deep_research.services.metrics import source_metrics

    # Record a successful query
    source_metrics.record_query(
        source_type="vector_search",
        source_name="product_docs",
        latency_ms=150.5,
        success=True,
    )

    # Get metrics summary
    summary = source_metrics.get_summary()
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceMetrics:
    """Metrics for a single source type or source name."""

    query_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float | None = None
    max_latency_ms: float | None = None
    last_query_time: float | None = None
    last_error: str | None = None

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.query_count == 0:
            return 0.0
        return self.total_latency_ms / self.query_count

    @property
    def error_rate(self) -> float:
        """Calculate error rate (0.0 to 1.0)."""
        if self.query_count == 0:
            return 0.0
        return self.error_count / self.query_count

    def record(
        self,
        latency_ms: float,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record a query result.

        Args:
            latency_ms: Query latency in milliseconds.
            success: Whether the query succeeded.
            error: Optional error message if not successful.
        """
        self.query_count += 1
        self.total_latency_ms += latency_ms
        self.last_query_time = time.time()

        if success:
            self.success_count += 1
        else:
            self.error_count += 1
            self.last_error = error

        # Update min/max
        if self.min_latency_ms is None or latency_ms < self.min_latency_ms:
            self.min_latency_ms = latency_ms
        if self.max_latency_ms is None or latency_ms > self.max_latency_ms:
            self.max_latency_ms = latency_ms

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "query_count": self.query_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "error_rate": round(self.error_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2) if self.min_latency_ms else None,
            "max_latency_ms": round(self.max_latency_ms, 2) if self.max_latency_ms else None,
            "last_query_time": self.last_query_time,
            "last_error": self.last_error,
        }


@dataclass
class SourceMetricsCollector:
    """Thread-safe collector for source query metrics.

    Tracks metrics at two levels:
    1. By source type (vector_search, genie, knowledge_assistant)
    2. By individual source name

    Metrics are kept in-memory and can be exposed via API.
    """

    _by_type: dict[str, SourceMetrics] = field(default_factory=lambda: defaultdict(SourceMetrics))
    _by_name: dict[str, SourceMetrics] = field(default_factory=lambda: defaultdict(SourceMetrics))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Percentile tracking (optional, stores last N latencies)
    _latency_history: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _max_history_size: int = 1000

    def record_query(
        self,
        source_type: str,
        source_name: str,
        latency_ms: float,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record a source query result.

        Thread-safe. Records metrics for both source type and source name.

        Args:
            source_type: Type of source (vector_search, genie, knowledge_assistant).
            source_name: Specific source name.
            latency_ms: Query latency in milliseconds.
            success: Whether the query succeeded.
            error: Optional error message if not successful.
        """
        with self._lock:
            # Record by type
            self._by_type[source_type].record(latency_ms, success, error)

            # Record by name
            self._by_name[source_name].record(latency_ms, success, error)

            # Store latency for percentile calculation
            type_history = self._latency_history[source_type]
            type_history.append(latency_ms)
            if len(type_history) > self._max_history_size:
                type_history.pop(0)

    def get_metrics_by_type(self, source_type: str) -> SourceMetrics | None:
        """Get metrics for a specific source type.

        Args:
            source_type: Type of source.

        Returns:
            SourceMetrics if found, None otherwise.
        """
        with self._lock:
            return self._by_type.get(source_type)

    def get_metrics_by_name(self, source_name: str) -> SourceMetrics | None:
        """Get metrics for a specific source name.

        Args:
            source_name: Name of the source.

        Returns:
            SourceMetrics if found, None otherwise.
        """
        with self._lock:
            return self._by_name.get(source_name)

    def get_all_type_metrics(self) -> dict[str, dict[str, Any]]:
        """Get all metrics grouped by source type.

        Returns:
            Dictionary mapping source type to metrics dict.
        """
        with self._lock:
            return {
                source_type: metrics.to_dict()
                for source_type, metrics in self._by_type.items()
            }

    def get_all_name_metrics(self) -> dict[str, dict[str, Any]]:
        """Get all metrics grouped by source name.

        Returns:
            Dictionary mapping source name to metrics dict.
        """
        with self._lock:
            return {
                source_name: metrics.to_dict()
                for source_name, metrics in self._by_name.items()
            }

    def get_summary(self) -> dict[str, Any]:
        """Get complete metrics summary.

        Returns:
            Dictionary with by_type and by_name sections.
        """
        with self._lock:
            total_queries = sum(m.query_count for m in self._by_type.values())
            total_errors = sum(m.error_count for m in self._by_type.values())

            error_rate = (
                round(total_errors / total_queries, 4) if total_queries > 0 else 0.0
            )
            return {
                "total_queries": total_queries,
                "total_errors": total_errors,
                "overall_error_rate": error_rate,
                "by_type": {
                    source_type: metrics.to_dict()
                    for source_type, metrics in self._by_type.items()
                },
                "by_name": {
                    source_name: metrics.to_dict()
                    for source_name, metrics in self._by_name.items()
                },
            }

    def get_percentile(self, source_type: str, percentile: float) -> float | None:
        """Calculate latency percentile for a source type.

        Args:
            source_type: Type of source.
            percentile: Percentile value (0-100).

        Returns:
            Latency at the specified percentile, or None if no data.
        """
        with self._lock:
            history = self._latency_history.get(source_type, [])
            if not history:
                return None

            sorted_latencies = sorted(history)
            index = int(len(sorted_latencies) * percentile / 100)
            index = min(index, len(sorted_latencies) - 1)
            return sorted_latencies[index]

    def get_latency_percentiles(
        self,
        source_type: str,
        percentiles: list[float] | None = None,
    ) -> dict[str, float | None]:
        """Get multiple latency percentiles for a source type.

        Args:
            source_type: Type of source.
            percentiles: List of percentiles to calculate. Defaults to [50, 90, 95, 99].

        Returns:
            Dictionary mapping percentile label to latency value.
        """
        if percentiles is None:
            percentiles = [50, 90, 95, 99]

        return {
            f"p{int(p)}": self.get_percentile(source_type, p)
            for p in percentiles
        }

    def reset(self) -> None:
        """Reset all metrics.

        Clears all collected data. Use with caution.
        """
        with self._lock:
            self._by_type.clear()
            self._by_name.clear()
            self._latency_history.clear()

    def reset_source(self, source_name: str) -> None:
        """Reset metrics for a specific source.

        Args:
            source_name: Name of the source to reset.
        """
        with self._lock:
            if source_name in self._by_name:
                del self._by_name[source_name]


# Global metrics collector instance
source_metrics = SourceMetricsCollector()


def record_source_query(
    source_type: str,
    source_name: str,
    latency_ms: float,
    success: bool,
    error: str | None = None,
) -> None:
    """Convenience function to record a source query.

    Args:
        source_type: Type of source (vector_search, genie, knowledge_assistant).
        source_name: Specific source name.
        latency_ms: Query latency in milliseconds.
        success: Whether the query succeeded.
        error: Optional error message if not successful.
    """
    source_metrics.record_query(
        source_type=source_type,
        source_name=source_name,
        latency_ms=latency_ms,
        success=success,
        error=error,
    )
