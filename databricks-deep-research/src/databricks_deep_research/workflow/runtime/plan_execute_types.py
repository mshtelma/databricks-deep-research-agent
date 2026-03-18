from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AvailableSourceDescriptor:
    source_name: str
    tool_kind: str
    source_kind: str
    description: str = ""
    endpoint: str = ""


@dataclass(frozen=True)
class ReplanFeedbackEntry:
    reason: str
    cycle: int
    message: str
    step_title: str = ""


@dataclass
class PlanCycleContext:
    cycle: int = 0
    completed_steps: list[str] = field(default_factory=list)
    feedback_history: list[ReplanFeedbackEntry] = field(default_factory=list)
    available_sources: list[AvailableSourceDescriptor] = field(default_factory=list)
