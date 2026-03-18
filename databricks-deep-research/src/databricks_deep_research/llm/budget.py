"""Workflow-level token budget tracking for LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from databricks_deep_research.errors import TokenBudgetExceededError


@dataclass
class NodeTokenUsage:
    """Per-node token usage tracking."""

    node_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0


@dataclass
class TokenBudget:
    """Workflow-level token budget tracker.

    If *max_total_tokens* is ``0`` the budget is unlimited and calls to
    :meth:`track_usage` will never raise.
    """

    max_total_tokens: int = 0  # 0 = unlimited
    _total_used: int = 0
    _node_usage: dict[str, NodeTokenUsage] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Core tracking
    # ------------------------------------------------------------------

    def track_usage(
        self, node_id: str, prompt_tokens: int, completion_tokens: int
    ) -> None:
        """Record token usage from an LLM call.

        Raises :class:`TokenBudgetExceededError` if the cumulative usage
        exceeds *max_total_tokens* (when a budget is configured).
        """
        total = prompt_tokens + completion_tokens
        self._total_used += total

        usage = self._node_usage.get(node_id)
        if usage is None:
            usage = NodeTokenUsage(node_id=node_id)
            self._node_usage[node_id] = usage

        usage.prompt_tokens += prompt_tokens
        usage.completion_tokens += completion_tokens
        usage.total_tokens += total
        usage.call_count += 1

        if self.max_total_tokens > 0 and self._total_used > self.max_total_tokens:
            raise TokenBudgetExceededError(
                used=self._total_used, limit=self.max_total_tokens
            )

    def check_budget(self, estimated_tokens: int = 0) -> bool:
        """Return ``True`` if *estimated_tokens* would fit within the remaining budget.

        Always returns ``True`` when the budget is unlimited.
        """
        if self.max_total_tokens <= 0:
            return True
        return (self._total_used + estimated_tokens) <= self.max_total_tokens

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_used(self) -> int:
        """Total tokens consumed so far across all nodes."""
        return self._total_used

    @property
    def remaining(self) -> int:
        """Remaining tokens.  Returns ``-1`` when the budget is unlimited."""
        if self.max_total_tokens <= 0:
            return -1
        return max(0, self.max_total_tokens - self._total_used)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_node_usage(self, node_id: str) -> NodeTokenUsage | None:
        """Return usage for a specific node, or ``None`` if it has not been recorded."""
        return self._node_usage.get(node_id)

    def get_all_usage(self) -> dict[str, NodeTokenUsage]:
        """Return a shallow copy of per-node usage data."""
        return dict(self._node_usage)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the budget state to a plain dictionary."""
        return {
            "max_total_tokens": self.max_total_tokens,
            "total_used": self._total_used,
            "remaining": self.remaining,
            "nodes": {
                nid: {
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens": u.total_tokens,
                    "call_count": u.call_count,
                }
                for nid, u in self._node_usage.items()
            },
        }
