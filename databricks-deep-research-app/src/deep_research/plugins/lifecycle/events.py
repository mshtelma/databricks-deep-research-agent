"""
Plugin lifecycle event dataclasses.

All events are frozen (immutable) to prevent accidental modification
by plugin callbacks. Events are generic and work with any research mode
or output type.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

# ============================================================================
# Job Lifecycle Events
# ============================================================================


@dataclass(frozen=True)
class JobSubmittedEvent:
    """Emitted when a job is submitted to the system (after DB insert)."""

    job_id: UUID
    chat_id: str
    query: str
    user_id: str | None
    research_config: dict[str, Any]  # Generic config (query_mode, research_depth, etc.)
    timestamp: datetime

    # Optional plugin-specific data
    plugin_data: dict[str, Any] | None = None


@dataclass(frozen=True)
class JobStartedEvent:
    """Emitted when job execution begins (before stream_research)."""

    job_id: UUID
    timestamp: datetime


@dataclass(frozen=True)
class JobCompletedEvent:
    """Emitted when job completes successfully."""

    job_id: UUID
    duration_seconds: float
    output: Any  # The final research output
    output_type: str  # "meeting_prep", "generic", etc.
    metrics: dict[str, Any]  # Generic metrics dict
    timestamp: datetime


@dataclass(frozen=True)
class JobFailedEvent:
    """Emitted when job fails with error."""

    job_id: UUID
    error_message: str
    error_type: str  # Exception class name
    error_category: str  # "validation", "network", "timeout", "database", "unknown"
    is_recoverable: bool
    error_context: dict[str, Any]  # Additional error details
    timestamp: datetime


# ============================================================================
# Synthesis Lifecycle Events
# ============================================================================


@dataclass(frozen=True)
class SynthesisConfigEvent:
    """Emitted before synthesis starts with full configuration.

    Use this to understand what synthesis parameters were used.
    Helpful for debugging schema constraints and verification settings.
    """

    job_id: UUID
    output_type: str  # "meeting_prep", "generic", etc.
    model_tier: str  # "synthesis", "reasoning", etc.
    temperature: float
    max_tokens: int
    query_preview: str  # First 200 chars of query

    # Schema information (if available)
    schema_name: str | None = None
    schema_fields: list[str] | None = None
    schema_required_fields: list[str] | None = None

    # Verification settings
    verify_sources: bool = False
    enable_post_verification: bool = False

    timestamp: datetime = None  # type: ignore


@dataclass(frozen=True)
class SynthesisStartedEvent:
    """Emitted when first synthesis event is received.

    Useful for tracking synthesis latency (time from start to first event).
    """

    job_id: UUID
    first_event_type: str
    elapsed_ms: float  # Time from synthesis start to first event
    timestamp: datetime


@dataclass(frozen=True)
class SynthesisChunkEvent:
    """Emitted for each synthesis progress chunk.

    WARNING: This hook is called VERY frequently (potentially hundreds of
    times per job). Keep implementations fast and consider throttling.
    """

    job_id: UUID
    content_chunk: str
    chunk_length: int
    cumulative_length: int  # Total chars accumulated so far
    timestamp: datetime

    # Optional metadata (for batching, etc.)
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class SynthesisCompletedEvent:
    """Emitted when synthesis finishes."""

    job_id: UUID
    total_chunks: int
    total_length: int
    duration_ms: float
    timestamp: datetime


# ============================================================================
# Validation Lifecycle Events
# ============================================================================


@dataclass(frozen=True)
class ValidationErrorEvent:
    """Emitted when Pydantic validation fails.

    This is THE CRITICAL event for debugging structured output failures.
    Includes unpacked ValidationError with field-level details.
    """

    job_id: UUID
    error_type: str  # Exception class name
    error_message: str

    # Pydantic-specific details (if ValidationError)
    validation_errors: list[dict[str, Any]] | None = None  # Raw Pydantic errors
    failed_fields: list[dict[str, str]] | None = None  # Simplified field list
    error_count: int | None = None

    # What the LLM actually returned
    raw_output: str | None = None
    raw_output_preview: str | None = None  # First 500 chars

    # Retry context
    attempt_number: int = 1
    max_attempts: int = 3

    timestamp: datetime = None  # type: ignore


@dataclass(frozen=True)
class ValidationSuccessEvent:
    """Emitted when validation passes."""

    job_id: UUID
    output_type: str
    field_count: int
    attempt_number: int = 1
    timestamp: datetime = None  # type: ignore


# ============================================================================
# Generic Stream Event
# ============================================================================


@dataclass(frozen=True)
class StreamEvent:
    """Generic event for any stream event (catch-all).

    Use this for comprehensive logging, event replay, or debugging.
    WARNING: Called VERY frequently - keep implementations fast!
    """

    job_id: UUID
    event_type: str
    event_data: dict[str, Any]
    sequence: int
    timestamp: datetime


@dataclass(frozen=True)
class GenericEvent:
    """Fallback event when event creation fails.

    Used by EventEmitter when specific event creation crashes
    but we still want to emit something for observability.
    """

    hook_name: str
    data: dict[str, Any]
    timestamp: datetime


# ============================================================================
# Helper Functions for Event Creation
# ============================================================================


def create_synthesis_config_event(
    job_id: UUID,
    research_config: Any,
    query: str,
) -> SynthesisConfigEvent:
    """
    Helper to create SynthesisConfigEvent with safe schema introspection.

    Extracts schema details without crashing if schema is unavailable.

    Args:
        job_id: Job UUID
        research_config: ResearchConfig object with output_schema attribute
        query: User query string

    Returns:
        SynthesisConfigEvent with schema details if available
    """

    schema_name = None
    schema_fields = None
    schema_required = None

    # Safely extract schema information
    if hasattr(research_config, "output_schema") and research_config.output_schema:
        schema_name = research_config.output_schema.__name__

        try:
            # Try to get JSON schema
            schema_json = research_config.output_schema.model_json_schema()
            properties = schema_json.get("properties", {})
            schema_fields = list(properties.keys())
            schema_required = schema_json.get("required", [])
        except Exception:
            # Schema introspection failed - continue without it
            pass

    return SynthesisConfigEvent(
        job_id=job_id,
        output_type=getattr(research_config, "output_type", "generic"),
        model_tier=getattr(research_config, "model_tier", "synthesis"),
        temperature=getattr(research_config, "temperature", 0.7),
        max_tokens=getattr(research_config, "max_tokens", 8000),
        query_preview=query[:200] if query else "",
        schema_name=schema_name,
        schema_fields=schema_fields,
        schema_required_fields=schema_required,
        verify_sources=getattr(research_config, "verify_sources", False),
        enable_post_verification=getattr(research_config, "enable_post_verification", False),
        timestamp=datetime.now(UTC),
    )


# ============================================================================
# Data Source Lifecycle Events (007-enterprise-data-sources, T009)
# ============================================================================


@dataclass(frozen=True)
class DataSourceQueryEvent:
    """Emitted when a data source is queried (T009).

    Tracks queries to enterprise data sources (Vector Search, Genie,
    Knowledge Assistants) for observability and analytics.
    """

    job_id: str
    source_type: str  # DataSourceType value
    source_name: str
    query: str
    filters: dict[str, Any] | None
    result_count: int
    duration_ms: float
    timestamp: datetime

    # Optional success/error info
    success: bool = True
    error_message: str | None = None


@dataclass(frozen=True)
class TemplateAppliedEvent:
    """Emitted when a template is applied (T009).

    Tracks template usage for analytics and debugging.
    """

    job_id: str
    template_id: str
    template_type: str  # 'system', 'step', 'synthesis', 'query'
    template_source: str  # 'system', 'plugin', 'user'
    variables: dict[str, Any]
    timestamp: datetime


@dataclass(frozen=True)
class CustomAgentSelectedEvent:
    """Emitted when a custom agent is selected (T009).

    Tracks custom agent usage for analytics.
    """

    job_id: str
    agent_id: str
    agent_name: str
    agent_source: str  # 'plugin', 'user'
    timestamp: datetime


@dataclass(frozen=True)
class DataLandscapeBuiltEvent:
    """Emitted when discovery builds the DataLandscape.

    Tracks discovery phase performance and source relevance.
    """

    job_id: str
    sources_queried: int
    sources_with_results: int
    top_source: str | None
    top_source_relevance: float | None
    total_duration_ms: float
    timestamp: datetime


def create_validation_error_event(
    job_id: UUID,
    error: Exception,
    raw_output: str | None = None,
    attempt_number: int = 1,
    max_attempts: int = 3,
) -> ValidationErrorEvent:
    """
    Helper to create ValidationErrorEvent with unpacked Pydantic details.

    Automatically unpacks ValidationError to extract field-level failures.
    Handles API changes across Pydantic versions defensively.

    Args:
        job_id: Job UUID
        error: The exception (ValidationError or other)
        raw_output: What the LLM actually returned (for debugging)
        attempt_number: Which retry attempt this is
        max_attempts: Maximum retries allowed

    Returns:
        ValidationErrorEvent with unpacked field details if ValidationError
    """

    validation_errors: list[dict[str, Any]] | None = None
    failed_fields: list[dict[str, str]] | None = None
    error_count: int | None = None

    # Try to unpack ValidationError
    try:
        from pydantic import ValidationError

        if isinstance(error, ValidationError):
            # Get raw error list - cast ErrorDetails (TypedDict) to dict[str, Any]
            raw_errors = error.errors()
            validation_errors = [{str(k): v for k, v in err.items()} for err in raw_errors]
            error_count = len(validation_errors)

            # Unpack to simplified field list
            failed_fields = []
            for err in raw_errors:
                field_path = ".".join(str(loc) for loc in err.get("loc", []))
                failed_fields.append({
                    "path": field_path,
                    "type": err.get("type", "unknown"),
                    "message": str(err.get("msg", ""))[:200],
                })

    except ImportError:
        # Pydantic not available
        pass
    except Exception:
        # Unpacking failed - continue with None values
        pass

    return ValidationErrorEvent(
        job_id=job_id,
        error_type=type(error).__name__,
        error_message=str(error)[:2000],
        validation_errors=validation_errors,
        failed_fields=failed_fields,
        error_count=error_count,
        raw_output=raw_output,
        raw_output_preview=raw_output[:500] if raw_output else None,
        attempt_number=attempt_number,
        max_attempts=max_attempts,
        timestamp=datetime.now(UTC),
    )
