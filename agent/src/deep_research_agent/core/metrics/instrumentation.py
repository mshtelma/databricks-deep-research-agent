"""Instrumentation and observability for the metric calculation pipeline.

Provides metrics tracking, performance monitoring, and event emission
for production observability.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from .. import get_logger


logger = get_logger(__name__)


class CalculationStatus(str, Enum):
    """Status of a calculation execution."""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"
    TIMEOUT = "timeout"
    RETRY = "retry"


@dataclass
class ExecutionMetric:
    """Single execution metric record."""
    task_id: str
    status: CalculationStatus
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    retry_count: int = 0
    cache_hit: bool = False


@dataclass
class PipelineMetrics:
    """Aggregated metrics for a pipeline run."""
    run_id: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    cached_tasks: int
    total_duration_ms: float
    research_feedback_triggered: bool
    research_queries_count: int
    iteration_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_tasks == 0:
            return 1.0
        return self.successful_tasks / self.total_tasks
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_tasks == 0:
            return 0.0
        return self.cached_tasks / self.total_tasks


class MetricPipelineInstrumentation:
    """Instrumentation system for metric pipeline observability.
    
    Tracks execution metrics, cache efficiency, and emits events for
    production monitoring and debugging.
    """
    
    def __init__(self, enable_detailed_logging: bool = False):
        self._enable_detailed_logging = enable_detailed_logging
        self._execution_metrics: List[ExecutionMetric] = []
        self._cache_stats: Dict[str, int] = {
            "plan_hits": 0,
            "plan_misses": 0,
            "result_hits": 0,
            "result_misses": 0
        }
        self._research_feedback_count: int = 0
        self._current_run_start: Optional[float] = None
    
    def start_run(self, run_id: str) -> None:
        """Start tracking a pipeline run."""
        self._current_run_start = time.time()
        logger.info(f"[INSTRUMENTATION] Pipeline run started: {run_id}")
    
    def record_execution(
        self,
        task_id: str,
        duration_ms: float,
        status: CalculationStatus,
        error_message: Optional[str] = None,
        retry_count: int = 0,
        cache_hit: bool = False
    ) -> None:
        """Record a calculation execution."""
        metric = ExecutionMetric(
            task_id=task_id,
            status=status,
            duration_ms=duration_ms,
            error_message=error_message,
            retry_count=retry_count,
            cache_hit=cache_hit
        )
        self._execution_metrics.append(metric)
        
        if self._enable_detailed_logging:
            logger.info(
                f"[INSTRUMENTATION] Task {task_id}: {status.value} "
                f"in {duration_ms:.2f}ms (cache_hit={cache_hit}, retries={retry_count})"
            )
        
        # Emit event for monitoring systems
        self._emit_event("calculation_execution", {
            "task_id": task_id,
            "status": status.value,
            "duration_ms": duration_ms,
            "cache_hit": cache_hit,
            "retry_count": retry_count
        })
    
    def record_cache_hit(self, cache_type: str) -> None:
        """Track cache hit."""
        key = f"{cache_type}_hits"
        if key in self._cache_stats:
            self._cache_stats[key] += 1
            logger.debug(f"[INSTRUMENTATION] Cache hit: {cache_type}")
    
    def record_cache_miss(self, cache_type: str) -> None:
        """Track cache miss."""
        key = f"{cache_type}_misses"
        if key in self._cache_stats:
            self._cache_stats[key] += 1
            logger.debug(f"[INSTRUMENTATION] Cache miss: {cache_type}")
    
    def record_research_feedback(self, queries_generated: int) -> None:
        """Track research feedback loop usage."""
        self._research_feedback_count += 1
        logger.info(
            f"[INSTRUMENTATION] Research feedback triggered: {queries_generated} queries"
        )
        self._emit_event("research_feedback_triggered", {
            "queries_count": queries_generated,
            "feedback_count": self._research_feedback_count
        })
    
    def get_pipeline_metrics(
        self,
        run_id: str,
        iteration_count: int = 1
    ) -> PipelineMetrics:
        """Generate aggregated metrics for the current run."""
        total_duration_ms = 0.0
        if self._current_run_start:
            total_duration_ms = (time.time() - self._current_run_start) * 1000
        
        total_tasks = len(self._execution_metrics)
        successful = sum(1 for m in self._execution_metrics if m.status == CalculationStatus.COMPLETED)
        failed = sum(1 for m in self._execution_metrics if m.status == CalculationStatus.FAILED)
        cached = sum(1 for m in self._execution_metrics if m.cache_hit)
        
        metrics = PipelineMetrics(
            run_id=run_id,
            total_tasks=total_tasks,
            successful_tasks=successful,
            failed_tasks=failed,
            cached_tasks=cached,
            total_duration_ms=total_duration_ms,
            research_feedback_triggered=self._research_feedback_count > 0,
            research_queries_count=self._research_feedback_count,
            iteration_count=iteration_count
        )
        
        logger.info(
            f"[INSTRUMENTATION] Pipeline metrics: {total_tasks} tasks, "
            f"{metrics.success_rate:.1%} success rate, "
            f"{metrics.cache_hit_rate:.1%} cache hit rate, "
            f"{total_duration_ms:.0f}ms total"
        )
        
        return metrics
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = dict(self._cache_stats)
        
        # Calculate rates
        plan_total = stats["plan_hits"] + stats["plan_misses"]
        result_total = stats["result_hits"] + stats["result_misses"]
        
        stats["plan_hit_rate"] = (
            stats["plan_hits"] / plan_total if plan_total > 0 else 0.0
        )
        stats["result_hit_rate"] = (
            stats["result_hits"] / result_total if result_total > 0 else 0.0
        )
        
        return stats
    
    def reset(self) -> None:
        """Reset all metrics for a new run."""
        self._execution_metrics.clear()
        self._cache_stats = {
            "plan_hits": 0,
            "plan_misses": 0,
            "result_hits": 0,
            "result_misses": 0
        }
        self._research_feedback_count = 0
        self._current_run_start = None
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event for external monitoring systems.
        
        In production, this could push to a metrics system like Prometheus,
        CloudWatch, or DataDog. For now, it just logs.
        """
        if self._enable_detailed_logging:
            logger.debug(f"[EVENT] {event_type}: {data}")
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history for debugging."""
        return [
            {
                "task_id": m.task_id,
                "status": m.status.value,
                "duration_ms": m.duration_ms,
                "timestamp": m.timestamp.isoformat(),
                "error_message": m.error_message,
                "retry_count": m.retry_count,
                "cache_hit": m.cache_hit
            }
            for m in self._execution_metrics
        ]


__all__ = [
    "MetricPipelineInstrumentation",
    "CalculationStatus",
    "ExecutionMetric",
    "PipelineMetrics"
]


