"""Shared data model for benchmark infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class BenchmarkQuestion:
    """A single benchmark question — immutable input."""

    uid: str
    question: str
    expected_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata carries benchmark-specific fields: difficulty, source_files, etc.


@dataclass
class QuestionResult:
    """Result of running a single benchmark question."""

    uid: str
    question: str
    expected_answer: str
    predicted_answer: str | None  # Extracted answer (None if extraction failed)
    raw_output: str  # Full workflow output
    status: Literal["success", "error", "timeout", "no_answer"]
    wall_time_seconds: float
    num_sources: int
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata preserves difficulty, requires_external, model_used, etc.


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    concurrency: int = 3
    timeout_per_question: int = 300
    results_dir: str = "results"
    resume: bool = True
    retry_statuses: frozenset[str] = field(default_factory=frozenset)
