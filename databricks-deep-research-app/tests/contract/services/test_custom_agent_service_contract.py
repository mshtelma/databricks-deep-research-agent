"""Contract tests for `CachedCustomAgentService` (F-CA).

Exercises create / get / update / delete agent, add / update / delete preset
step, and reorder semantics. Runs against the parametric ``stack`` fixture from
conftest.py (FakeBackend by default; real backends via env vars).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from deep_research.services.cached.custom_agent import CachedCustomAgentService


class TestCachedCustomAgentServiceContract:
    """Agent CRUD lifecycle."""

    @pytest.mark.asyncio
    async def test_create_and_get_accessible(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        agent = await svc.create_agent(owner_id=owner, name="MyAgent")

        assert agent.id is not None
        assert agent.owner_id == owner
        assert agent.name == "MyAgent"
        assert agent.source_scope == "all"
        assert agent.visibility == "private"
        assert agent.preset_steps == []

        fetched = await svc.get_accessible(agent.id, owner)
        assert fetched is not None
        assert fetched.id == agent.id

    @pytest.mark.asyncio
    async def test_get_for_user_ownership(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner_a = f"user_{uuid4().hex[:8]}"
        owner_b = f"user_{uuid4().hex[:8]}"

        agent = await svc.create_agent(owner_id=owner_a, name="PrivateAgent")

        found = await svc.get_for_user(agent.id, owner_a)
        assert found is not None

        not_found = await svc.get_for_user(agent.id, owner_b)
        assert not_found is None

    @pytest.mark.asyncio
    async def test_get_accessible_workspace_visibility(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"
        other = f"user_{uuid4().hex[:8]}"

        agent = await svc.create_agent(
            owner_id=owner, name="WorkspaceAgent", visibility="workspace"
        )

        # Owner can access
        assert await svc.get_accessible(agent.id, owner) is not None
        # Other user can also access (workspace)
        assert await svc.get_accessible(agent.id, other) is not None
        # But other cannot access via get_for_user (strict ownership)
        assert await svc.get_for_user(agent.id, other) is None

    @pytest.mark.asyncio
    async def test_get_by_name(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        await svc.create_agent(owner_id=owner, name="Alpha")
        await svc.create_agent(owner_id=owner, name="Beta")

        found = await svc.get_by_name(owner, "Alpha")
        assert found is not None
        assert found.name == "Alpha"

        not_found = await svc.get_by_name(owner, "Gamma")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_get_accessible_agents_list(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"
        other = f"user_{uuid4().hex[:8]}"

        await svc.create_agent(owner_id=owner, name="Private1", visibility="private")
        await svc.create_agent(owner_id=owner, name="Shared", visibility="workspace")

        # Owner sees both
        agents, total = await svc.get_accessible_agents(user_id=owner)
        own_names = {a.name for a in agents if a.owner_id == owner}
        assert "Private1" in own_names
        assert "Shared" in own_names

        # Other user sees only workspace agents
        other_agents, _ = await svc.get_accessible_agents(user_id=other)
        other_names = {a.name for a in other_agents}
        assert "Shared" in other_names
        assert "Private1" not in other_names

    @pytest.mark.asyncio
    async def test_update_agent(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        agent = await svc.create_agent(owner_id=owner, name="Original")
        agent.name = "Updated"
        agent.description = "New description"

        updated = await svc.update(agent)
        assert updated.name == "Updated"
        assert updated.description == "New description"

        # Verify persistence
        fetched = await svc.get_accessible(agent.id, owner)
        assert fetched is not None
        assert fetched.name == "Updated"

    @pytest.mark.asyncio
    async def test_delete_agent(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        agent = await svc.create_agent(owner_id=owner, name="ToDelete")
        await svc.delete(agent)

        fetched = await svc.get_accessible(agent.id, owner)
        assert fetched is None

    @pytest.mark.asyncio
    async def test_create_with_preset_steps(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        agent = await svc.create_agent(
            owner_id=owner,
            name="SteppedAgent",
            preset_steps=[
                {"title": "Step A", "order": 1, "description": "First step"},
                {"title": "Step B", "order": 2, "description": "Second step"},
            ],
        )

        assert len(agent.preset_steps) == 2
        assert agent.preset_steps[0].title == "Step A"
        assert agent.preset_steps[1].title == "Step B"
        # UUIDs are assigned
        assert agent.preset_steps[0].id is not None
        assert agent.preset_steps[1].id is not None
        # IDs are different
        assert agent.preset_steps[0].id != agent.preset_steps[1].id


class TestPresetStepManagement:
    """Preset step CRUD and reorder."""

    @pytest.mark.asyncio
    async def test_add_and_get_step(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        agent = await svc.create_agent(owner_id=owner, name="AgentWithSteps")
        step = await svc.create_preset_step(
            agent_id=agent.id,
            title="Research competitors",
            description="Find top 5 competitors",
            order=1,
        )

        assert step.id is not None
        assert step.agent_id == agent.id
        assert step.title == "Research competitors"
        assert step.order == 1

        fetched_step = await svc.get_preset_step(step.id, agent.id)
        assert fetched_step is not None
        assert fetched_step.id == step.id

    @pytest.mark.asyncio
    async def test_get_agent_preset_steps_ordered(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        agent = await svc.create_agent(owner_id=owner, name="OrderedAgent")
        await svc.create_preset_step(agent_id=agent.id, title="Step 3", order=3)
        await svc.create_preset_step(agent_id=agent.id, title="Step 1", order=1)
        await svc.create_preset_step(agent_id=agent.id, title="Step 2", order=2)

        steps = await svc.get_agent_preset_steps(agent.id)
        assert len(steps) == 3
        assert steps[0].title == "Step 1"
        assert steps[1].title == "Step 2"
        assert steps[2].title == "Step 3"

    @pytest.mark.asyncio
    async def test_update_preset_step(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        agent = await svc.create_agent(owner_id=owner, name="UpdateStepAgent")
        step = await svc.create_preset_step(
            agent_id=agent.id, title="Original title", order=1
        )

        step.title = "Updated title"
        step.description = "Now with description"
        updated = await svc.update_preset_step(step)

        assert updated.title == "Updated title"
        assert updated.description == "Now with description"

        # Verify persistence round-trip
        fetched = await svc.get_preset_step(step.id, agent.id)
        assert fetched is not None
        assert fetched.title == "Updated title"

    @pytest.mark.asyncio
    async def test_delete_preset_step(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        agent = await svc.create_agent(owner_id=owner, name="DeleteStepAgent")
        step1 = await svc.create_preset_step(agent_id=agent.id, title="Keep", order=1)
        step2 = await svc.create_preset_step(agent_id=agent.id, title="Delete", order=2)

        await svc.delete_preset_step(step2)

        steps = await svc.get_agent_preset_steps(agent.id)
        assert len(steps) == 1
        assert steps[0].id == step1.id

    @pytest.mark.asyncio
    async def test_reorder_preserves_uuids(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        agent = await svc.create_agent(owner_id=owner, name="ReorderAgent")
        s1 = await svc.create_preset_step(agent_id=agent.id, title="A", order=1)
        s2 = await svc.create_preset_step(agent_id=agent.id, title="B", order=2)
        s3 = await svc.create_preset_step(agent_id=agent.id, title="C", order=3)

        # Reorder: C, A, B
        reordered = await svc.reorder_preset_steps(agent.id, [s3.id, s1.id, s2.id])

        assert len(reordered) == 3
        # UUIDs preserved, order reflects new position
        id_to_order = {s.id: s.order for s in reordered}
        assert id_to_order[s3.id] == 1
        assert id_to_order[s1.id] == 2
        assert id_to_order[s2.id] == 3

        # Verify persistence
        steps = await svc.get_agent_preset_steps(agent.id)
        assert steps[0].id == s3.id
        assert steps[1].id == s1.id
        assert steps[2].id == s2.id

    @pytest.mark.asyncio
    async def test_reorder_partial_step_ids(self, stack) -> None:
        """reorder_preset_steps silently ignores unknown step IDs."""
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        agent = await svc.create_agent(owner_id=owner, name="PartialReorderAgent")
        s1 = await svc.create_preset_step(agent_id=agent.id, title="X", order=1)
        s2 = await svc.create_preset_step(agent_id=agent.id, title="Y", order=2)

        # Pass an extra unknown UUID — should not raise
        unknown = uuid4()
        result = await svc.reorder_preset_steps(agent.id, [s2.id, s1.id, unknown])

        assert len(result) == 2
        assert result[0].id == s2.id
        assert result[1].id == s1.id

    @pytest.mark.asyncio
    async def test_step_round_trip_source_hints(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        agent = await svc.create_agent(owner_id=owner, name="HintsAgent")
        hints = {"preferred_sources": ["vs_index_1"], "search_queries": ["market trends"]}
        step = await svc.create_preset_step(
            agent_id=agent.id,
            title="With Hints",
            order=1,
            source_hints=hints,
            source_scope="enterprise_only",
        )

        fetched = await svc.get_preset_step(step.id, agent.id)
        assert fetched is not None
        assert fetched.source_hints == hints
        assert fetched.source_scope == "enterprise_only"
        assert fetched.get_preferred_sources() == ["vs_index_1"]
        assert fetched.get_search_queries() == ["market trends"]

    @pytest.mark.asyncio
    async def test_get_preset_step_nonexistent_returns_none(self, stack) -> None:
        svc = CachedCustomAgentService(stack)
        owner = f"user_{uuid4().hex[:8]}"
        agent = await svc.create_agent(owner_id=owner, name="EmptyAgent")

        result = await svc.get_preset_step(uuid4(), agent.id)
        assert result is None
