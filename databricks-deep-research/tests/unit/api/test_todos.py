"""TodoStore + write_todos tool tests."""

from __future__ import annotations

import pytest

from databricks_deep_research.api import (
    InMemoryTodoStore,
    Todo,
    write_todos_tool,
)
from databricks_deep_research.tools.protocol import ToolContext


@pytest.mark.asyncio
async def test_inmemory_store_round_trip() -> None:
    store = InMemoryTodoStore()
    todos = [
        Todo(id="a", title="task A"),
        Todo(id="b", title="task B", status="in_progress"),
    ]
    await store.write("t1", todos)
    items = await store.list("t1")
    assert len(items) == 2
    assert items[0].id == "a"
    assert items[1].status == "in_progress"


@pytest.mark.asyncio
async def test_inmemory_store_thread_isolation() -> None:
    store = InMemoryTodoStore()
    await store.write("t1", [Todo(id="a", title="A")])
    await store.write("t2", [Todo(id="b", title="B")])

    t1 = await store.list("t1")
    t2 = await store.list("t2")
    assert [x.id for x in t1] == ["a"]
    assert [x.id for x in t2] == ["b"]


@pytest.mark.asyncio
async def test_inmemory_store_update_and_clear() -> None:
    store = InMemoryTodoStore()
    await store.write("t1", [Todo(id="a", title="A")])
    await store.update("t1", "a", status="completed")
    items = await store.list("t1")
    assert items[0].status == "completed"

    await store.clear("t1")
    assert await store.list("t1") == []


@pytest.mark.asyncio
async def test_write_todos_tool_factory() -> None:
    store = InMemoryTodoStore()
    tool_obj = write_todos_tool(store)
    assert tool_obj.definition.name == "write_todos"

    ctx = ToolContext()
    ctx.extras["_framework_thread_id"] = "thread-1"
    ctx.extras["_framework_todos_store"] = store
    result = await tool_obj.execute(
        {"todos": [{"id": "x", "title": "wax car"}]},
        ctx,
    )
    assert "wrote" in result.content
    listed = await store.list("thread-1")
    assert len(listed) == 1
    assert listed[0].id == "x"


@pytest.mark.asyncio
async def test_write_todos_falls_back_to_default_store() -> None:
    store = InMemoryTodoStore()
    tool_obj = write_todos_tool(store)
    ctx = ToolContext()  # No extras → fall back to factory's `store`
    ctx.extras["_framework_thread_id"] = "tA"
    result = await tool_obj.execute(
        {"todos": [{"id": "x", "title": "tA work"}]}, ctx,
    )
    assert "wrote" in result.content
    assert (await store.list("tA"))[0].id == "x"


def test_todo_schema_independent_from_plan_step() -> None:
    """Todo must NOT inherit from NormalizedPlanContract / PlanStep."""
    # Sanity: Todo is a plain BaseModel; doesn't share schema with the plan-and-execute runtime.
    assert Todo.model_fields.keys() == {
        "id",
        "title",
        "description",
        "status",
        "created_at",
        "updated_at",
    }
