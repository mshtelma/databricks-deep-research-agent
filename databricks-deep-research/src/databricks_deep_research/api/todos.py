"""DeepAgents-style ``write_todos`` primitive.

A :class:`Todo` carries minimal scheduling metadata (id, title, status). The
schema is intentionally **independent** from
``workflow/runtime/plan_execute_contracts.NormalizedPlanContract`` — the
``plan_and_execute`` runtime is unchanged. ``write_todos`` is a free-form
LLM-driven scratchpad; ``plan_and_execute`` remains the structural runtime.

Usage::

    from databricks_deep_research.api import (
        Agent, InMemoryTodoStore, write_todos_tool, tool,
    )

    store = InMemoryTodoStore()
    agent = Agent(
        name="researcher",
        instructions="Plan your work via write_todos.",
        tools=[write_todos_tool(store)],
        extras={"_framework_todos_store": store},
    )

The ``write_todos`` tool reads ``_framework_todos_store`` and
``_framework_thread_id`` from ``ToolContext.extras``.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from databricks_deep_research.tools.api import _DecoratedTool, tool

TodoStatus = Literal["pending", "in_progress", "completed", "blocked"]


class Todo(BaseModel):
    """A single todo item.

    Independent schema — DOES NOT extend the IR's ``NormalizedPlanContract``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    description: str = ""
    status: TodoStatus = "pending"
    created_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


class TodoStore(Protocol):
    """Protocol for an async todo store keyed by thread_id."""

    async def list(self, thread_id: str) -> list[Todo]: ...
    async def write(self, thread_id: str, todos: Iterable[Todo]) -> None: ...
    async def update(self, thread_id: str, todo_id: str, **fields: Any) -> None: ...
    async def clear(self, thread_id: str) -> None: ...


class InMemoryTodoStore:
    """In-process :class:`TodoStore` keyed by thread_id."""

    def __init__(self) -> None:
        self._by_thread: dict[str, list[Todo]] = {}

    async def list(self, thread_id: str) -> list[Todo]:
        return [t.model_copy() for t in self._by_thread.get(thread_id, [])]

    async def write(self, thread_id: str, todos: Iterable[Todo]) -> None:
        self._by_thread[thread_id] = [t.model_copy() for t in todos]

    async def update(self, thread_id: str, todo_id: str, **fields: Any) -> None:
        for t in self._by_thread.get(thread_id, []):
            if t.id == todo_id:
                for k, v in fields.items():
                    if hasattr(t, k):
                        setattr(t, k, v)
                t.updated_at = datetime.now(tz=UTC).isoformat()

    async def clear(self, thread_id: str) -> None:
        self._by_thread.pop(thread_id, None)


def write_todos_tool(store: TodoStore | None = None) -> _DecoratedTool:
    """Factory returning a ``@tool``-decorated ``write_todos`` callable.

    Args:
        store: Optional default store. When provided, the tool falls back to
            this when ``ctx.extras["_framework_todos_store"]`` is unset.

    The returned tool accepts a list of :class:`Todo` and persists it
    keyed by ``ctx.extras["_framework_thread_id"]``.
    """

    @tool(
        name="write_todos",
        inject={
            "_inject_store": "_framework_todos_store",
            "_inject_thread_id": "_framework_thread_id",
        },
    )
    async def write_todos(
        todos: list[Todo],
        *,
        _inject_store: TodoStore | None = None,
        _inject_thread_id: str | None = None,
    ) -> str:
        """Replace the agent's todo list for this thread.

        Args:
            todos: New ordered list of todos.

        Returns:
            A human-readable summary like ``"wrote 3 todos"``.
        """
        active_store: TodoStore | None = _inject_store or store
        if active_store is None:
            return "no todos store configured"
        thread_id = _inject_thread_id or "_default_thread"
        await active_store.write(thread_id, todos)
        return f"wrote {len(list(todos))} todos"

    return write_todos


__all__ = [
    "InMemoryTodoStore",
    "Todo",
    "TodoStatus",
    "TodoStore",
    "write_todos_tool",
]
