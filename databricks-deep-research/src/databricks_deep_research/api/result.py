"""``AgentResult`` — the structured return value of :meth:`Agent.arun`.

Generic over the user's ``output_type`` so type checkers can narrow
``result.output`` for typed pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from databricks_deep_research.citation.extraction import VerificationSummary
from databricks_deep_research.events.types import StreamEvent

T = TypeVar("T", bound=BaseModel)


@dataclass
class AgentResult(Generic[T]):
    """Result of a completed :class:`Agent` run.

    Attributes:
        content: Raw textual output from the agent (typically the final
            content of the last LLM response).
        output: When ``Agent.output_type`` is set and the LLM produced
            valid structured output, this is the parsed Pydantic instance.
            Otherwise ``content`` is mirrored here.
        events: Ordered list of :class:`StreamEvent` emitted during the run.
        verification: Structured verification summary from the framework's
            7-stage citation pipeline. Populated only when the agent's
            ``subtype="synthesizer"``; ``None`` otherwise.
        tool_calls: Best-effort list of tool calls observed in events.
        sources: Best-effort list of sources surfaced via tool results.
        usage: Aggregate LLM token usage (``prompt_tokens``,
            ``completion_tokens``, ``total_tokens``).
        ok: ``True`` when the run completed without a fatal error or
            structured-output validation failure.
        run_id: Unique identifier for this run (workflow id by default).
    """

    content: str = ""
    output: Any = None
    events: list[StreamEvent] = field(default_factory=list)
    verification: VerificationSummary | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    sources: list[Any] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    ok: bool = True
    run_id: str = ""


__all__ = ["AgentResult"]
