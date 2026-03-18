"""
Output Module
=============

Provides infrastructure for custom output types in research pipelines.

This module enables:
- Custom output schema definitions
- Output type registration and discovery
- Custom synthesizer configuration
- Plugin-provided output types

Example usage:
    from deep_research.output import (
        OutputTypeProvider,
        OutputTypeRegistry,
        SynthesisReport,
        SynthesizerConfig,
    )

    # Register custom output type
    registry = OutputTypeRegistry()
    registry.register("meeting_prep", MeetingPrepProvider())

    # Get synthesizer config for output type
    config = registry.get_synthesizer_config("meeting_prep")
"""

from deep_research.output.base import (
    SynthesisReport,
    SynthesizerConfig,
)
from deep_research.output.protocol import OutputTypeProvider
from deep_research.output.registry import OutputTypeRegistry

__all__ = [
    # Base types
    "SynthesisReport",
    "SynthesizerConfig",
    # Protocol
    "OutputTypeProvider",
    # Registry
    "OutputTypeRegistry",
]
