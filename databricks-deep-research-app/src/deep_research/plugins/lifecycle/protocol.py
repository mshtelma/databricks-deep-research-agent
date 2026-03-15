"""
Plugin lifecycle callback protocol.

Plugins implement this protocol to receive notifications about job
execution events. All methods are OPTIONAL - implement only what you need.

Framework uses hasattr() to check which hooks each plugin implements.
"""

from typing import Protocol, runtime_checkable

from .events import (
    CustomAgentSelectedEvent,
    DataLandscapeBuiltEvent,
    DataSourceQueryEvent,
    JobCompletedEvent,
    JobFailedEvent,
    JobStartedEvent,
    JobSubmittedEvent,
    StreamEvent,
    SynthesisChunkEvent,
    SynthesisCompletedEvent,
    SynthesisConfigEvent,
    SynthesisStartedEvent,
    TemplateAppliedEvent,
    ValidationErrorEvent,
    ValidationSuccessEvent,
)


@runtime_checkable
class JobLifecycleListener(Protocol):
    """
    Optional protocol for plugins to receive job lifecycle events.

    Plugins can implement ANY SUBSET of these methods - framework
    uses hasattr() to check which hooks plugin implements.

    All hooks can be sync OR async - framework handles both.

    Hook errors are caught and logged but don't crash the framework.

    Example:
    --------
    class MyPlugin(JobLifecycleListener):
        def on_job_submitted(self, event: JobSubmittedEvent) -> None:
            logger.info("Job submitted: %s", event.job_id)

        def on_validation_error(self, event: ValidationErrorEvent) -> None:
            # Only implement hooks you need
            logger.error("Validation failed: %s", event.failed_fields)
    """

    # ========================================================================
    # Job Lifecycle Hooks
    # ========================================================================

    def on_job_submitted(self, event: JobSubmittedEvent) -> None:
        """
        Called when job is submitted to system (after DB insert).

        Use for: Job tracking, external notifications, metrics.

        Timing: After job exists in database, before research starts.
        """
        ...

    def on_job_started(self, event: JobStartedEvent) -> None:
        """
        Called when job execution begins (before stream_research).

        Use for: Start timers, initialize resources.

        Timing: After job fetched from DB, before first research call.
        """
        ...

    def on_job_completed(self, event: JobCompletedEvent) -> None:
        """
        Called when job completes successfully.

        Use for: Cleanup, metrics, success notifications.

        Timing: After all events processed, before returning to client.
        """
        ...

    def on_job_failed(self, event: JobFailedEvent) -> None:
        """
        Called when job fails with error.

        Use for: Error tracking, alerting, cleanup.

        Timing: In exception handler, before error propagated to client.
        """
        ...

    # ========================================================================
    # Synthesis Lifecycle Hooks
    # ========================================================================

    def on_synthesis_config(self, event: SynthesisConfigEvent) -> None:
        """
        Called before synthesis starts with full configuration.

        Use for: Logging synthesis parameters, debugging schema constraints.

        Timing: After ResearchConfig created, before stream_research called.
        """
        ...

    def on_synthesis_started(self, event: SynthesisStartedEvent) -> None:
        """
        Called when first synthesis event is received.

        Use for: Confirming synthesis started, latency tracking.

        Timing: When first event arrives from LLM (after initial latency).
        """
        ...

    def on_synthesis_chunk(self, event: SynthesisChunkEvent) -> None:
        """
        Called for each synthesis progress chunk.

        Use for: Progress tracking, milestone logging.

        WARNING: Called frequently (potentially hundreds of times per job).
        Keep implementation FAST! Framework applies throttling by default.

        Timing: Every time synthesisProgress event received (or at milestones).
        """
        ...

    def on_synthesis_completed(self, event: SynthesisCompletedEvent) -> None:
        """
        Called when synthesis finishes.

        Use for: Synthesis metrics, duration tracking.

        Timing: After all synthesis chunks received, before validation.
        """
        ...

    # ========================================================================
    # Validation Lifecycle Hooks
    # ========================================================================

    def on_validation_error(self, event: ValidationErrorEvent) -> None:
        """
        Called when Pydantic validation fails.

        Use for: Detailed error logging, schema debugging.

        Event includes unpacked ValidationError with field-level details:
        - validation_errors: Raw Pydantic error list
        - failed_fields: Simplified list with path, type, message
        - raw_output: What LLM actually returned
        - attempt_number: Which retry this is

        Timing: In exception handler during structured output recovery.
        """
        ...

    def on_validation_success(self, event: ValidationSuccessEvent) -> None:
        """
        Called when validation passes.

        Use for: Success rate tracking, field coverage metrics.

        Timing: After successful Pydantic validation.
        """
        ...

    # ========================================================================
    # Generic Hooks
    # ========================================================================

    def on_stream_event(self, event: StreamEvent) -> None:
        """
        Called for ANY stream event (catch-all).

        Use for: Event replay, debugging, comprehensive logging.

        WARNING: Called VERY frequently. Keep implementation fast!

        Timing: For every event from stream_research.
        """
        ...

    # ========================================================================
    # Enterprise Data Source Hooks (007-enterprise-data-sources, T064)
    # ========================================================================

    def on_data_source_query(self, event: DataSourceQueryEvent) -> None:
        """
        Called when an enterprise data source is queried.

        Use for: Query tracking, analytics, performance monitoring.

        Timing: After each Vector Search, Genie, or Knowledge Assistant query.

        Event includes:
        - source_type: 'vector_search', 'genie', 'knowledge_assistant'
        - source_name: Specific source identifier
        - query: The query that was executed
        - result_count: Number of results returned
        - duration_ms: Query execution time
        """
        ...

    def on_template_applied(self, event: TemplateAppliedEvent) -> None:
        """
        Called when a prompt template is applied.

        Use for: Template usage analytics, debugging.

        Timing: When a template is rendered during research.

        Event includes:
        - template_id: ID of the applied template
        - template_type: 'system', 'step', 'synthesis', 'query'
        - template_source: 'system', 'plugin', 'user'
        - variables: Variables used in rendering
        """
        ...

    def on_custom_agent_selected(self, event: CustomAgentSelectedEvent) -> None:
        """
        Called when a custom agent is selected for research.

        Use for: Agent usage tracking, analytics.

        Timing: When user or system selects a custom agent.

        Event includes:
        - agent_id: ID of the selected agent
        - agent_name: Human-readable agent name
        - agent_source: 'plugin' or 'user'
        """
        ...

    def on_data_landscape_built(self, event: DataLandscapeBuiltEvent) -> None:
        """
        Called when discovery phase builds the data landscape.

        Use for: Discovery performance tracking, source effectiveness analysis.

        Timing: After discovery phase completes source relevance assessment.

        Event includes:
        - sources_queried: Number of sources queried
        - sources_with_results: Number of sources that returned results
        - top_source: Most relevant source name
        - top_source_relevance: Relevance score of top source
        - total_duration_ms: Total discovery duration
        """
        ...
