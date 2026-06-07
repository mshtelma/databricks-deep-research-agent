"""Exception hierarchy for the databricks-deep-research framework."""

from __future__ import annotations

from typing import Any


class WorkflowError(Exception):
    """Base exception for all framework errors."""


class WorkflowConditionEvaluationError(WorkflowError):
    """Raised when a workflow condition cannot be evaluated safely."""


class PlanningContractError(WorkflowError):
    """Raised when a planning loop cannot satisfy its execution contract."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


class WorkflowValidationError(WorkflowError):
    """Raised when workflow definition fails load-time validation."""

    def __init__(self, message: str = "Workflow validation failed", errors: list[str] | None = None) -> None:
        self.errors: list[str] = errors or []
        full = f"{message}: {'; '.join(self.errors)}" if self.errors else message
        super().__init__(full)


class WorkflowCancelledError(WorkflowError):
    """Raised when workflow is cancelled mid-execution."""


class TokenBudgetExceededError(WorkflowError):
    """Raised when token budget is exhausted."""

    def __init__(self, used: int, limit: int) -> None:
        self.used = used
        self.limit = limit
        super().__init__(f"Token budget exceeded: used {used} of {limit}")


class ContextWindowExceededError(WorkflowError):
    """Raised when an assembled prompt cannot fit any available model's context window.

    Only raised when overflow handling is configured to ``fail`` (the default
    behavior escalates to a larger-context endpoint and, as a last resort,
    truncates). ``tried`` lists the endpoints considered, each as
    ``(endpoint, window)``.
    """

    def __init__(
        self,
        required_tokens: int,
        best_window: int,
        tried: list[tuple[str, int]] | None = None,
    ) -> None:
        self.required_tokens = required_tokens
        self.best_window = best_window
        self.tried = tried or []
        super().__init__(
            f"Prompt requires ~{required_tokens} tokens but the largest "
            f"available context window is {best_window}. "
            f"Tried endpoints: {self.tried}"
        )


class NodeBudgetExceededError(WorkflowError):
    """Raised when a node exceeds its configured wall-clock budget."""

    def __init__(self, node_id: str, budget_seconds: float, elapsed_ms: float) -> None:
        self.node_id = node_id
        self.budget_seconds = budget_seconds
        self.elapsed_ms = elapsed_ms
        super().__init__(
            f"Node budget exceeded for {node_id}: "
            f"{elapsed_ms:.1f} ms elapsed of {budget_seconds:.3f} s budget"
        )


class WorkflowExecutionError(WorkflowError):
    """Raised when a workflow fails after emitting partial progress."""

    def __init__(
        self,
        message: str,
        *,
        state: Any,
        events: list[Any],
        cause: Exception,
    ) -> None:
        self.state = state
        self.events = events
        self.cause = cause
        super().__init__(message)
