"""Agent I/O isolation boundary.

Every agent node receives an :class:`AgentInput` and produces an
:class:`AgentOutput`.  These dataclasses form the strict boundary between
the workflow engine and the agent execution logic, ensuring that agents
never touch global state directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from databricks_deep_research.agents.prompt_context import CompiledPoolSection


@dataclass(frozen=True)
class AgentInput:
    """Immutable input envelope handed to an agent node.

    Attributes:
        query: The user's original query (always available).
        context: Values resolved from ``input_keys`` on the workflow state.
        system_prompt: Fully rendered system prompt.
        user_prompt: Fully rendered user prompt.
        tools: Tool definitions the agent is allowed to call.
        pool_sections: Injected pool sections keyed by pool name.
        conversation_history: Prior messages for multi-turn agents.
    """

    query: str
    context: dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    user_prompt: str = ""
    tools: list[Any] = field(default_factory=list)  # list[ToolDefinition]
    pool_sections: dict[str, CompiledPoolSection] = field(default_factory=dict)
    conversation_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentOutput:
    """Mutable output envelope produced by an agent node.

    Attributes:
        content: Raw agent output (str, dict, or Pydantic model instance).
        output_key: State key where ``content`` will be stored.
        pool_writes: Items to append to shared pools, keyed by pool name.
        sources: Source-tracking records (SourceInfo instances).
        token_usage: Token counts (prompt_tokens, completion_tokens, etc.).
        events: Domain events emitted during execution.
    """

    content: Any = None
    output_key: str = "output"
    pool_writes: dict[str, list[Any]] = field(default_factory=dict)
    sources: list[Any] = field(default_factory=list)  # list[SourceInfo]
    token_usage: dict[str, int] = field(default_factory=dict)
    events: list[Any] = field(default_factory=list)  # Domain events to emit
