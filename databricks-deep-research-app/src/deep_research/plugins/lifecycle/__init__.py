"""
Plugin lifecycle callback system.

Provides lifecycle hooks for plugins to receive notifications about
job execution events. Plugins can implement any subset of callbacks.

Key Components:
- Event dataclasses: Immutable event objects with job/synthesis data
- JobLifecycleListener: Protocol defining all available callbacks
- EventEmitter: Helper for creating and emitting events
- Helper functions: Defensive event creation with error handling

Example:
--------
from deep_research.plugins.lifecycle import (
    JobLifecycleListener,
    JobSubmittedEvent,
    ValidationErrorEvent,
)

class MyPlugin(JobLifecycleListener):
    def on_job_submitted(self, event: JobSubmittedEvent) -> None:
        logger.info("Job %s submitted", event.job_id)

    def on_validation_error(self, event: ValidationErrorEvent) -> None:
        logger.error("Validation failed: %s", event.failed_fields)
"""

from .emitter import EventEmitter, unpack_validation_error
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
    create_synthesis_config_event,
    create_validation_error_event,
)
from .protocol import JobLifecycleListener

__all__ = [
    # Events
    "JobSubmittedEvent",
    "JobStartedEvent",
    "JobCompletedEvent",
    "JobFailedEvent",
    "SynthesisConfigEvent",
    "SynthesisStartedEvent",
    "SynthesisChunkEvent",
    "SynthesisCompletedEvent",
    "ValidationErrorEvent",
    "ValidationSuccessEvent",
    "StreamEvent",
    "GenericEvent",
    # Protocol
    "JobLifecycleListener",
    # Emitter
    "EventEmitter",
    # Helpers
    "create_synthesis_config_event",
    "create_validation_error_event",
    "unpack_validation_error",
]
