"""
Base classes and protocols for report generation strategies.

Defines the interface that all strategies must implement,
plus common functionality shared across strategies.
"""

import time
import tracemalloc
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Any
from contextlib import asynccontextmanager
from uuid import uuid4

from ...core import get_logger
from .types import (
    ReportGenerationRequest,
    ReportGenerationResult,
    ReporterConfig,
    GenerationMode,
    ReportQuality,
    GenerationMetrics,
    GenerationError,
)

logger = get_logger(__name__)


class ReportGenerationStrategy(ABC):
    """
    Protocol for report generation strategies.

    All strategies must implement this interface for type safety
    and to work with the factory.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging/metrics."""
        ...

    @property
    @abstractmethod
    def generation_mode(self) -> GenerationMode:
        """Generation mode this strategy implements."""
        ...

    @property
    @abstractmethod
    def quality_level(self) -> ReportQuality:
        """Quality level this strategy provides."""
        ...

    @abstractmethod
    def can_handle(
        self,
        request: ReportGenerationRequest
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if this strategy can handle the request.

        Args:
            request: The generation request

        Returns:
            Tuple of (can_handle, reason_if_not)
        """
        ...

    @abstractmethod
    async def generate(
        self,
        request: ReportGenerationRequest,
        config: ReporterConfig
    ) -> ReportGenerationResult:
        """
        Generate report from request.

        Args:
            request: Typed generation request
            config: Typed reporter configuration

        Returns:
            Typed generation result

        Raises:
            GenerationError: If generation fails and cannot recover
        """
        ...

    async def validate_output(
        self,
        result: ReportGenerationResult
    ) -> Tuple[bool, List[str]]:
        """
        Validate the generated output.

        Default implementation does basic checks.
        Strategies can override for more specific validation.

        Args:
            result: The generation result to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check minimum report length
        if len(result.final_report.strip()) < 100:
            issues.append("Report is too short (< 100 characters)")

        # Check sections were generated
        if result.total_sections == 0:
            issues.append("No sections generated")

        # Check section count matches
        if len(result.sections) != result.total_sections:
            issues.append(
                f"Section count mismatch: {len(result.sections)} != {result.total_sections}"
            )

        # Warnings don't invalidate, but should be logged
        if result.warnings:
            logger.warning(f"Report generated with warnings: {result.warnings}")

        return (len(issues) == 0, issues)


class BaseReportGenerator(ReportGenerationStrategy):
    """
    Base class with common functionality for all strategies.

    Provides:
    - Context management for metrics/tracing
    - Progress emission
    - Error handling utilities
    - Resource tracking
    """

    def __init__(
        self,
        llm: Any,
        config: ReporterConfig,
        metrics_collector: Optional[Any] = None,
        event_emitter: Optional[Any] = None
    ):
        """
        Initialize base generator.

        Args:
            llm: Language model instance
            config: Typed reporter configuration
            metrics_collector: Optional metrics collector
            event_emitter: Optional event emitter for UI updates
        """
        self.llm = llm
        self.config = config
        self.metrics_collector = metrics_collector or NullMetricsCollector()
        self.event_emitter = event_emitter or NullEventEmitter()

        # Request context (set during generation)
        self._request_context: Optional[ReportGenerationRequest] = None
        self._trace_id: Optional[str] = None

        # Metrics tracking
        self._llm_call_count = 0
        self._tokens_used = 0
        self._start_time: Optional[float] = None

    @asynccontextmanager
    async def _generation_context(self, request: ReportGenerationRequest):
        """
        Context manager for generation with metrics/tracing.

        Automatically tracks:
        - Generation time
        - Memory usage
        - Error recovery
        - Event emission

        Usage:
            async with self._generation_context(request) as trace_id:
                # Generation code here
                ...
        """
        self._start_time = time.time()
        self._trace_id = str(uuid4())
        self._llm_call_count = 0
        self._tokens_used = 0

        # Start memory tracking if enabled
        if self.config.enable_performance_monitoring:
            tracemalloc.start()
            snapshot_start = tracemalloc.take_snapshot()

        # Emit start event
        if self.config.enable_tracing:
            self._emit_event('generation_start', {
                'strategy': self.name,
                'request_id': request.request_id,
                'trace_id': self._trace_id,
                'topic': request.research_topic
            })

        try:
            self._request_context = request
            yield self._trace_id

        except Exception as e:
            # Record error metrics
            self.metrics_collector.record_error(self.name, type(e).__name__)

            # Emit error event
            self._emit_event('generation_error', {
                'strategy': self.name,
                'error': str(e),
                'error_type': type(e).__name__,
                'trace_id': self._trace_id
            })

            # Re-raise for handling by caller
            raise

        finally:
            # Calculate duration
            duration_ms = (time.time() - self._start_time) * 1000

            # Calculate memory usage
            memory_mb = 0.0
            if self.config.enable_performance_monitoring:
                snapshot_end = tracemalloc.take_snapshot()
                memory_mb = sum(stat.size for stat in snapshot_end.statistics('lineno')) / (1024 * 1024)
                tracemalloc.stop()

            # Record performance metrics
            if self.config.enable_metrics:
                metrics = GenerationMetrics(
                    strategy_name=self.name,
                    request_id=request.request_id,
                    start_time=self._start_time,
                    end_time=time.time(),
                    duration_ms=duration_ms,
                    sections_generated=0,  # Updated by strategy
                    llm_invocations=self._llm_call_count,
                    observations_processed=len(request.observations),
                    memory_mb_used=memory_mb,
                    tokens_consumed=self._tokens_used
                )
                self.metrics_collector.record_metrics(metrics)

            # Emit completion event
            if self.config.enable_tracing:
                self._emit_event('generation_complete', {
                    'strategy': self.name,
                    'duration_ms': duration_ms,
                    'trace_id': self._trace_id,
                    'llm_calls': self._llm_call_count
                })

            # Clean up context
            self._request_context = None
            self._trace_id = None

    def _emit_event(self, event_type: str, data: dict):
        """Emit event to UI via event emitter."""
        if self.event_emitter:
            try:
                self.event_emitter.emit(event_type, data)
            except Exception as e:
                logger.warning(f"Failed to emit event {event_type}: {e}")

    def _emit_progress(self, message: str, progress: float):
        """
        Emit progress event for UI.

        Args:
            message: Progress message
            progress: Progress value (0.0 to 1.0)
        """
        if self._request_context:
            self._emit_event('generation_progress', {
                'strategy': self.name,
                'message': message,
                'progress': min(1.0, max(0.0, progress)),
                'request_id': self._request_context.request_id,
                'trace_id': self._trace_id
            })

    def _count_llm_call(self, tokens: int = 0):
        """Track LLM call and token usage."""
        self._llm_call_count += 1
        self._tokens_used += tokens

    def _parse_sections(self, report_text: str) -> List['ReportSection']:
        """
        Parse markdown report into sections.

        Utility method for extracting sections from generated text.
        """
        from .types import ReportSection

        sections = []
        current_section = None
        current_content = []

        for line in report_text.split('\n'):
            # Check for heading
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    sections.append(ReportSection(
                        title=current_section['title'],
                        content='\n'.join(current_content).strip(),
                        level=current_section['level']
                    ))

                # Start new section
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                current_section = {'title': title, 'level': level}
                current_content = []
            else:
                current_content.append(line)

        # Add last section
        if current_section:
            sections.append(ReportSection(
                title=current_section['title'],
                content='\n'.join(current_content).strip(),
                level=current_section['level']
            ))

        return sections

    def _count_tables(self, text: str) -> int:
        """Count markdown tables in text."""
        # Count separator rows (| --- |)
        import re
        separator_pattern = r'\|\s*[-:]+\s*\|'
        return len(re.findall(separator_pattern, text))

    @abstractmethod
    async def generate(
        self,
        request: ReportGenerationRequest,
        config: ReporterConfig
    ) -> ReportGenerationResult:
        """Subclasses must implement generation logic."""
        pass


# Null implementations for optional dependencies

class NullMetricsCollector:
    """Null implementation of metrics collector."""

    def record_error(self, strategy: str, error_type: str):
        """No-op error recording."""
        pass

    def record_metrics(self, metrics: GenerationMetrics):
        """No-op metrics recording."""
        pass

    def record_duration(self, strategy: str, duration_ms: float):
        """No-op duration recording."""
        pass


class NullEventEmitter:
    """Null implementation of event emitter."""

    def emit(self, event_type: str, data: dict):
        """No-op event emission."""
        pass
