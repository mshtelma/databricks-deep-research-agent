"""
EventEmitter - Centralized event creation and emission.

Reduces boilerplate at integration points and provides defensive
event creation with error recovery.

Usage:
------
# Before (OLD - 10+ lines of boilerplate):
from deep_research.plugins.lifecycle import JobSubmittedEvent
from datetime import datetime, timezone

event = JobSubmittedEvent(
    job_id=job.id,
    chat_id=chat_id,
    query=query,
    timestamp=datetime.now(timezone.utc),
    # ... many fields ...
)
await plugin_manager.emit_hook("on_job_submitted", event)

# After (NEW - 1 line):
await event_emitter.job_submitted(job.id, chat_id, query)
"""

import contextlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from .events import (
    GenericEvent,
    JobCompletedEvent,
    JobFailedEvent,
    JobStartedEvent,
    JobSubmittedEvent,
    StreamEvent,
    SynthesisChunkEvent,
    SynthesisCompletedEvent,
    SynthesisConfigEvent,
    SynthesisStartedEvent,
    ValidationErrorEvent,
    ValidationSuccessEvent,
)

if TYPE_CHECKING:
    from deep_research.plugins.manager import PluginManager

logger = logging.getLogger(__name__)


def unpack_validation_error(error: Exception) -> dict[str, Any]:
    """
    Safely unpack Pydantic ValidationError.

    Handles API changes across Pydantic versions defensively.
    Returns dict with error_type, message, errors list, and simplified fields.

    Args:
        error: Exception to unpack (typically ValidationError)

    Returns:
        Dict with keys: error_type, message, errors, fields
    """
    error_info: dict[str, Any] = {
        "error_type": type(error).__name__,
        "message": str(error)[:500],
        "errors": [],
        "fields": [],
    }

    # Try Pydantic ValidationError
    try:
        from pydantic import ValidationError

        if isinstance(error, ValidationError):
            try:
                # Try Pydantic v2 API
                error_info["errors"] = error.errors()

                # Extract simplified field list
                for err in error.errors():
                    loc = err.get("loc", ())
                    if loc:
                        field_path = ".".join(str(x) for x in loc)
                        error_info["fields"].append({
                            "path": field_path,
                            "type": err.get("type", "unknown"),
                            "msg": err.get("msg", "")[:200],
                        })

            except AttributeError:
                # Pydantic v1 fallback
                with contextlib.suppress(Exception):
                    error_info["errors"] = error.errors()

            except Exception as unpack_err:
                logger.warning(
                    "Failed to unpack ValidationError: %s",
                    str(unpack_err)[:200],
                )

    except ImportError:
        # Pydantic not available
        pass

    return error_info


class EventEmitter:
    """
    Centralized event emission with error handling.

    Reduces boilerplate at integration points and provides
    consistent error handling and metrics.

    Each method creates the appropriate event and emits it via
    PluginManager.emit_hook(), with defensive error handling.
    """

    def __init__(self, plugin_manager: "PluginManager"):
        """
        Initialize EventEmitter.

        Args:
            plugin_manager: PluginManager instance for hook emission
        """
        self._pm = plugin_manager

    # ========================================================================
    # Job Lifecycle Events
    # ========================================================================

    async def job_submitted(
        self,
        job_id: UUID,
        chat_id: str,
        query: str,
        user_id: str | None = None,
        **research_config: Any,
    ) -> None:
        """
        Emit job_submitted event.

        Args:
            job_id: Job UUID
            chat_id: Chat session ID
            query: User query string
            user_id: Optional user ID
            **research_config: Generic research configuration (query_mode, etc.)
        """
        await self._safe_emit(
            "on_job_submitted",
            lambda: JobSubmittedEvent(
                job_id=job_id,
                chat_id=chat_id,
                query=query,
                user_id=user_id,
                research_config=research_config,
                timestamp=datetime.now(UTC),
            ),
            fallback_data={"job_id": str(job_id), "chat_id": chat_id},
        )

    async def job_started(self, job_id: UUID) -> None:
        """
        Emit job_started event.

        Args:
            job_id: Job UUID
        """
        await self._safe_emit(
            "on_job_started",
            lambda: JobStartedEvent(
                job_id=job_id,
                timestamp=datetime.now(UTC),
            ),
            fallback_data={"job_id": str(job_id)},
        )

    async def job_completed(
        self,
        job_id: UUID,
        duration_seconds: float,
        output: Any,
        output_type: str,
        **metrics: Any,
    ) -> None:
        """
        Emit job_completed event.

        Args:
            job_id: Job UUID
            duration_seconds: Total job duration
            output: The final research output
            output_type: Output type name ("meeting_prep", etc.)
            **metrics: Generic metrics dict
        """
        await self._safe_emit(
            "on_job_completed",
            lambda: JobCompletedEvent(
                job_id=job_id,
                duration_seconds=duration_seconds,
                output=output,
                output_type=output_type,
                metrics=metrics,
                timestamp=datetime.now(UTC),
            ),
            fallback_data={
                "job_id": str(job_id),
                "duration": duration_seconds,
            },
        )

    async def job_failed(
        self,
        job_id: UUID,
        error: Exception,
        error_category: str,
        is_recoverable: bool = False,
        **error_context: Any,
    ) -> None:
        """
        Emit job_failed event.

        Args:
            job_id: Job UUID
            error: The exception that caused failure
            error_category: Category ("validation", "network", "timeout", etc.)
            is_recoverable: Whether error can be retried
            **error_context: Additional error details
        """
        await self._safe_emit(
            "on_job_failed",
            lambda: JobFailedEvent(
                job_id=job_id,
                error_message=str(error)[:500],
                error_category=error_category,
                error_type=type(error).__name__,
                is_recoverable=is_recoverable,
                error_context=error_context,
                timestamp=datetime.now(UTC),
            ),
            fallback_data={
                "job_id": str(job_id),
                "error": str(error)[:200],
            },
        )

    # ========================================================================
    # Synthesis Lifecycle Events
    # ========================================================================

    async def synthesis_config(
        self,
        job_id: UUID,
        output_type: str,
        model_tier: str,
        temperature: float,
        max_tokens: int,
        query_preview: str,
        schema_name: str | None = None,
        schema_fields: list[str] | None = None,
        schema_required_fields: list[str] | None = None,
        verify_sources: bool = False,
        enable_post_verification: bool = False,
    ) -> None:
        """
        Emit synthesis_config event.

        Args:
            job_id: Job UUID
            output_type: Output type name
            model_tier: Model tier ("synthesis", etc.)
            temperature: LLM temperature
            max_tokens: Max output tokens
            query_preview: First 200 chars of query
            schema_name: Output schema name (if structured)
            schema_fields: List of schema fields
            schema_required_fields: Required fields
            verify_sources: Whether source verification enabled
            enable_post_verification: Whether post-verification enabled
        """
        await self._safe_emit(
            "on_synthesis_config",
            lambda: SynthesisConfigEvent(
                job_id=job_id,
                output_type=output_type,
                model_tier=model_tier,
                temperature=temperature,
                max_tokens=max_tokens,
                query_preview=query_preview,
                schema_name=schema_name,
                schema_fields=schema_fields,
                schema_required_fields=schema_required_fields,
                verify_sources=verify_sources,
                enable_post_verification=enable_post_verification,
                timestamp=datetime.now(UTC),
            ),
            fallback_data={"job_id": str(job_id), "output_type": output_type},
        )

    async def synthesis_started(
        self,
        job_id: UUID,
        first_event_type: str,
        elapsed_ms: float,
    ) -> None:
        """
        Emit synthesis_started event.

        Args:
            job_id: Job UUID
            first_event_type: Type of first event received
            elapsed_ms: Latency from synthesis start to first event
        """
        await self._safe_emit(
            "on_synthesis_started",
            lambda: SynthesisStartedEvent(
                job_id=job_id,
                first_event_type=first_event_type,
                elapsed_ms=elapsed_ms,
                timestamp=datetime.now(UTC),
            ),
            fallback_data={"job_id": str(job_id)},
        )

    async def synthesis_chunk(
        self,
        job_id: UUID,
        content_chunk: str,
        cumulative_length: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Emit synthesis_chunk event (throttled automatically).

        Args:
            job_id: Job UUID
            content_chunk: The chunk content
            cumulative_length: Total chars accumulated
            metadata: Optional metadata (batching info, etc.)
        """
        await self._safe_emit(
            "on_synthesis_chunk",
            lambda: SynthesisChunkEvent(
                job_id=job_id,
                content_chunk=content_chunk,
                chunk_length=len(content_chunk),
                cumulative_length=cumulative_length,
                metadata=metadata,
                timestamp=datetime.now(UTC),
            ),
            fallback_data={"job_id": str(job_id)},
        )

    async def synthesis_completed(
        self,
        job_id: UUID,
        total_chunks: int,
        total_length: int,
        duration_ms: float,
    ) -> None:
        """
        Emit synthesis_completed event.

        Args:
            job_id: Job UUID
            total_chunks: Total number of chunks received
            total_length: Total character count
            duration_ms: Total synthesis duration
        """
        await self._safe_emit(
            "on_synthesis_completed",
            lambda: SynthesisCompletedEvent(
                job_id=job_id,
                total_chunks=total_chunks,
                total_length=total_length,
                duration_ms=duration_ms,
                timestamp=datetime.now(UTC),
            ),
            fallback_data={"job_id": str(job_id)},
        )

    # ========================================================================
    # Validation Events
    # ========================================================================

    async def validation_error(
        self,
        job_id: UUID,
        error: Exception,
        raw_output: str | None = None,
        attempt_number: int = 1,
        max_attempts: int = 3,
    ) -> None:
        """
        Emit validation_error event with safe error unpacking.

        Args:
            job_id: Job UUID
            error: Validation exception
            raw_output: What LLM returned
            attempt_number: Which retry this is
            max_attempts: Max retries allowed
        """
        error_info = unpack_validation_error(error)

        await self._safe_emit(
            "on_validation_error",
            lambda: ValidationErrorEvent(
                job_id=job_id,
                error_type=error_info["error_type"],
                error_message=error_info["message"],
                validation_errors=error_info.get("errors"),
                failed_fields=error_info.get("fields"),
                error_count=len(error_info.get("errors", [])) or None,
                raw_output=raw_output,
                raw_output_preview=raw_output[:500] if raw_output else None,
                attempt_number=attempt_number,
                max_attempts=max_attempts,
                timestamp=datetime.now(UTC),
            ),
            fallback_data={
                "job_id": str(job_id),
                "error": str(error)[:200],
            },
        )

    async def validation_success(
        self,
        job_id: UUID,
        output_type: str,
        field_count: int,
        attempt_number: int = 1,
    ) -> None:
        """
        Emit validation_success event.

        Args:
            job_id: Job UUID
            output_type: Output type name
            field_count: Number of fields validated
            attempt_number: Which attempt succeeded
        """
        await self._safe_emit(
            "on_validation_success",
            lambda: ValidationSuccessEvent(
                job_id=job_id,
                output_type=output_type,
                field_count=field_count,
                attempt_number=attempt_number,
                timestamp=datetime.now(UTC),
            ),
            fallback_data={"job_id": str(job_id)},
        )

    # ========================================================================
    # Generic Stream Events
    # ========================================================================

    async def stream_event(
        self,
        job_id: UUID,
        event_type: str,
        event_data: dict[str, Any],
        sequence: int,
    ) -> None:
        """
        Emit generic stream event.

        Args:
            job_id: Job UUID
            event_type: Type of stream event
            event_data: Event payload
            sequence: Event sequence number
        """
        await self._safe_emit(
            "on_stream_event",
            lambda: StreamEvent(
                job_id=job_id,
                event_type=event_type,
                event_data=event_data,
                sequence=sequence,
                timestamp=datetime.now(UTC),
            ),
            fallback_data={"job_id": str(job_id), "event_type": event_type},
        )

    # ========================================================================
    # Internal: Safe Emission with Error Recovery
    # ========================================================================

    async def _safe_emit(
        self,
        hook_name: str,
        event_factory: Callable[[], Any],
        fallback_data: dict[str, Any] | None = None,
    ) -> None:
        """
        Safely create and emit hook with error recovery.

        Tries to create event using event_factory. If creation fails,
        falls back to GenericEvent with fallback_data.

        Args:
            hook_name: Name of hook to call
            event_factory: Callable that creates the event
            fallback_data: Minimal data for GenericEvent fallback
        """
        try:
            # Try to create event
            event = event_factory()
        except Exception as e:
            logger.error(
                "PLUGIN_EVENT_CREATION_FAILED hook=%s error=%s",
                hook_name,
                str(e)[:200],
            )

            if fallback_data:
                # Create minimal fallback event
                event = GenericEvent(
                    hook_name=hook_name,
                    data=fallback_data,
                    timestamp=datetime.now(UTC),
                )
            else:
                # Skip emission entirely
                return

        try:
            await self._pm.emit_hook(hook_name, event)
        except Exception:
            logger.exception(
                "PLUGIN_HOOK_EMISSION_FAILED hook=%s",
                hook_name,
            )
