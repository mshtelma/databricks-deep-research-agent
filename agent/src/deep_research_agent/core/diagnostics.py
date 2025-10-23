"""
Diagnostic context for tracking execution flow and metrics.

Provides comprehensive logging and metrics collection throughout agent execution.
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticEvent:
    """Single diagnostic event with timestamp and context."""
    timestamp: str
    phase: str
    component: str
    event_type: str  # info, warning, error, metric
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class DiagnosticContext:
    """Collect diagnostics throughout execution."""

    def __init__(self, run_id: str):
        """
        Initialize diagnostic context.

        Args:
            run_id: Unique identifier for this execution run
        """
        self.run_id = run_id
        self.events: List[DiagnosticEvent] = []
        self.metrics: Dict[str, Any] = {}
        self.start_time = datetime.utcnow()
        self._event_counts = {"info": 0, "warning": 0, "error": 0, "metric": 0}

    def log(
        self,
        phase: str,
        component: str,
        message: str,
        event_type: str = "info",
        **details
    ):
        """
        Log diagnostic event.

        Args:
            phase: Execution phase (e.g., "initialization", "metric_pipeline", "report_generation")
            component: Component name (e.g., "reporter", "validator", "fetcher")
            message: Human-readable message
            event_type: Type of event - "info", "warning", "error", or "metric"
            **details: Additional context as keyword arguments
        """
        if event_type not in ["info", "warning", "error", "metric"]:
            logger.warning(f"Invalid event_type '{event_type}', using 'info'")
            event_type = "info"

        event = DiagnosticEvent(
            timestamp=datetime.utcnow().isoformat(),
            phase=phase,
            component=component,
            event_type=event_type,
            message=message,
            details=details if details else None
        )
        self.events.append(event)
        self._event_counts[event_type] += 1

        # Also log to standard logger with appropriate level
        log_func = {
            "error": logger.error,
            "warning": logger.warning,
            "info": logger.info,
            "metric": logger.debug
        }.get(event_type, logger.info)

        details_str = f" | {json.dumps(details)}" if details else ""
        log_func(f"[{phase}:{component}] {message}{details_str}")

    def metric(self, name: str, value: Any):
        """
        Record metric value.

        Args:
            name: Metric name (e.g., "report_length", "fetch_success_rate")
            value: Metric value (number, boolean, string)
        """
        self.metrics[name] = value
        self.log("metrics", "collector", f"{name}={value}", "metric")

    def error(self, phase: str, component: str, message: str, **details):
        """
        Log error event (convenience method).

        Args:
            phase: Execution phase
            component: Component name
            message: Error message
            **details: Additional error context
        """
        self.log(phase, component, message, "error", **details)

    def warning(self, phase: str, component: str, message: str, **details):
        """
        Log warning event (convenience method).

        Args:
            phase: Execution phase
            component: Component name
            message: Warning message
            **details: Additional context
        """
        self.log(phase, component, message, "warning", **details)

    def info(self, phase: str, component: str, message: str, **details):
        """
        Log info event (convenience method).

        Args:
            phase: Execution phase
            component: Component name
            message: Info message
            **details: Additional context
        """
        self.log(phase, component, message, "info", **details)

    def get_errors(self) -> List[DiagnosticEvent]:
        """Get all error events."""
        return [e for e in self.events if e.event_type == "error"]

    def get_warnings(self) -> List[DiagnosticEvent]:
        """Get all warning events."""
        return [e for e in self.events if e.event_type == "warning"]

    def has_errors(self) -> bool:
        """Check if any errors were logged."""
        return self._event_counts["error"] > 0

    def has_warnings(self) -> bool:
        """Check if any warnings were logged."""
        return self._event_counts["warning"] > 0

    def get_summary(self) -> dict:
        """
        Get diagnostic summary.

        Returns:
            Dictionary with summary statistics and sample events
        """
        errors = self.get_errors()
        warnings = self.get_warnings()
        duration = (datetime.utcnow() - self.start_time).total_seconds()

        return {
            "run_id": self.run_id,
            "duration_seconds": round(duration, 2),
            "total_events": len(self.events),
            "event_counts": self._event_counts.copy(),
            "errors": len(errors),
            "warnings": len(warnings),
            "metrics": self.metrics.copy(),
            "error_details": [e.to_dict() for e in errors[:5]],  # First 5 errors
            "warning_details": [e.to_dict() for e in warnings[:5]],  # First 5 warnings
            "status": "failed" if errors else ("warning" if warnings else "success")
        }

    def get_summary_json(self, indent: int = 2) -> str:
        """
        Get diagnostic summary as formatted JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation of summary
        """
        return json.dumps(self.get_summary(), indent=indent)

    def assert_healthy(self, fail_on_warnings: bool = False):
        """
        Raise exception if critical errors detected.

        Args:
            fail_on_warnings: If True, also fail on warnings

        Raises:
            RuntimeError: If errors (or warnings if fail_on_warnings=True) detected
        """
        errors = self.get_errors()

        if errors:
            error_summary = "\n".join(f"- [{e.phase}:{e.component}] {e.message}" for e in errors[:10])
            raise RuntimeError(
                f"Diagnostic check failed with {len(errors)} errors:\n{error_summary}"
            )

        if fail_on_warnings:
            warnings = self.get_warnings()
            if warnings:
                warning_summary = "\n".join(
                    f"- [{w.phase}:{w.component}] {w.message}" for w in warnings[:10]
                )
                raise RuntimeError(
                    f"Diagnostic check failed with {len(warnings)} warnings:\n{warning_summary}"
                )

    def log_summary(self, log_level: str = "info"):
        """
        Log the diagnostic summary.

        Args:
            log_level: Log level to use ("debug", "info", "warning", "error")
        """
        summary = self.get_summary()
        log_func = getattr(logger, log_level, logger.info)

        log_func("=" * 80)
        log_func(f"Diagnostic Summary (Run ID: {self.run_id})")
        log_func("=" * 80)
        log_func(f"Duration: {summary['duration_seconds']}s")
        log_func(f"Status: {summary['status'].upper()}")
        log_func(f"Events: {summary['total_events']} total "
                f"({summary['event_counts']['error']} errors, "
                f"{summary['event_counts']['warning']} warnings)")

        if summary["metrics"]:
            log_func("Metrics:")
            for name, value in summary["metrics"].items():
                log_func(f"  - {name}: {value}")

        if summary["error_details"]:
            log_func(f"Recent Errors ({len(summary['error_details'])} shown):")
            for error in summary["error_details"]:
                log_func(f"  - [{error['phase']}:{error['component']}] {error['message']}")

        if summary["warning_details"]:
            log_func(f"Recent Warnings ({len(summary['warning_details'])} shown):")
            for warning in summary["warning_details"]:
                log_func(f"  - [{warning['phase']}:{warning['component']}] {warning['message']}")

        log_func("=" * 80)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - log summary on exit."""
        if exc_type is not None:
            self.error(
                "execution",
                "context_manager",
                f"Exception occurred: {exc_type.__name__}",
                exception=str(exc_val)
            )

        self.log_summary("info" if not self.has_errors() else "error")
        return False  # Don't suppress exceptions
