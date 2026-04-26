"""Contract tests for ``CachedTemplateService`` (F-OTHER.1).

Exercises create / get / update / delete / set_default / render semantics.
Runs against the parametric ``stack`` fixture from conftest.py
(FakeBackend by default; real backends via env vars).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from deep_research.services.cached.template import CachedTemplateService


class TestCachedTemplateServiceContract:
    """Template CRUD lifecycle."""

    @pytest.mark.asyncio
    async def test_create_and_get_for_user(self, stack) -> None:
        svc = CachedTemplateService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        tpl = await svc.create_template(
            owner_id=owner,
            name="My Template",
            template_type="system",
            content="Hello {{name}}",
        )

        assert tpl.id is not None
        assert tpl.owner_id == owner
        assert tpl.name == "My Template"
        assert tpl.type == "system"
        assert tpl.visibility == "private"
        # Variable auto-extracted
        assert any(v["name"] == "name" for v in tpl.variables)

        fetched = await svc.get_for_user(tpl.id, owner)
        assert fetched is not None
        assert fetched.id == tpl.id
        assert fetched.name == "My Template"

    @pytest.mark.asyncio
    async def test_get_for_user_ownership(self, stack) -> None:
        svc = CachedTemplateService(stack)
        owner_a = f"user_{uuid4().hex[:8]}"
        owner_b = f"user_{uuid4().hex[:8]}"

        tpl = await svc.create_template(
            owner_id=owner_a, name="Private", template_type="system", content="x"
        )

        assert await svc.get_for_user(tpl.id, owner_a) is not None
        assert await svc.get_for_user(tpl.id, owner_b) is None

    @pytest.mark.asyncio
    async def test_get_accessible_workspace_visibility(self, stack) -> None:
        svc = CachedTemplateService(stack)
        owner = f"user_{uuid4().hex[:8]}"
        other = f"user_{uuid4().hex[:8]}"

        tpl = await svc.create_template(
            owner_id=owner,
            name="Shared",
            template_type="synthesis",
            content="x",
            visibility="workspace",
        )

        assert await svc.get_accessible(tpl.id, owner) is not None
        assert await svc.get_accessible(tpl.id, other) is not None
        assert await svc.get_for_user(tpl.id, other) is None

    @pytest.mark.asyncio
    async def test_get_by_name(self, stack) -> None:
        svc = CachedTemplateService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        await svc.create_template(owner_id=owner, name="Alpha", template_type="system", content="x")
        await svc.create_template(owner_id=owner, name="Beta", template_type="system", content="y")

        found = await svc.get_by_name(owner, "Alpha")
        assert found is not None
        assert found.name == "Alpha"

        assert await svc.get_by_name(owner, "Gamma") is None

    @pytest.mark.asyncio
    async def test_get_accessible_templates_list(self, stack) -> None:
        svc = CachedTemplateService(stack)
        owner = f"user_{uuid4().hex[:8]}"
        other = f"user_{uuid4().hex[:8]}"

        await svc.create_template(owner_id=owner, name="Priv", template_type="system", content="x")
        await svc.create_template(
            owner_id=owner, name="Pub", template_type="system", content="x", visibility="workspace"
        )

        owned, total_owned = await svc.get_accessible_templates(user_id=owner)
        names = {t.name for t in owned}
        assert "Priv" in names
        assert "Pub" in names

        other_tpls, _ = await svc.get_accessible_templates(user_id=other)
        other_names = {t.name for t in other_tpls}
        assert "Pub" in other_names
        assert "Priv" not in other_names

    @pytest.mark.asyncio
    async def test_update_template(self, stack) -> None:
        svc = CachedTemplateService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        tpl = await svc.create_template(
            owner_id=owner, name="Original", template_type="step", content="x"
        )
        tpl.name = "Updated"
        tpl.description = "A description"

        updated = await svc.update(tpl)
        assert updated.name == "Updated"
        assert updated.description == "A description"

        fetched = await svc.get_for_user(tpl.id, owner)
        assert fetched is not None
        assert fetched.name == "Updated"

    @pytest.mark.asyncio
    async def test_delete_template(self, stack) -> None:
        svc = CachedTemplateService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        tpl = await svc.create_template(
            owner_id=owner, name="ToDelete", template_type="query", content="x"
        )
        await svc.delete(tpl)

        assert await svc.get_for_user(tpl.id, owner) is None
        assert await svc.get_accessible(tpl.id, owner) is None

    @pytest.mark.asyncio
    async def test_set_default_template(self, stack) -> None:
        svc = CachedTemplateService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        t1 = await svc.create_template(
            owner_id=owner, name="T1", template_type="system", content="x"
        )
        t2 = await svc.create_template(
            owner_id=owner, name="T2", template_type="system", content="y"
        )

        result = await svc.set_default_template(t1.id, owner)
        assert result is not None
        assert result.is_default is True

        # Setting t2 as default must unset t1
        result2 = await svc.set_default_template(t2.id, owner)
        assert result2 is not None
        assert result2.is_default is True

        t1_fetched = await svc.get_for_user(t1.id, owner)
        assert t1_fetched is not None
        assert t1_fetched.is_default is False

    @pytest.mark.asyncio
    async def test_get_default_template(self, stack) -> None:
        svc = CachedTemplateService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        # No default yet
        default = await svc.get_default_template(owner, "synthesis")
        assert default is None

        tpl = await svc.create_template(
            owner_id=owner,
            name="DefaultSynth",
            template_type="synthesis",
            content="x",
            is_default=True,
        )

        default = await svc.get_default_template(owner, "synthesis")
        assert default is not None
        assert default.id == tpl.id

    @pytest.mark.asyncio
    async def test_render_template(self, stack) -> None:
        svc = CachedTemplateService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        tpl = await svc.create_template(
            owner_id=owner,
            name="Render",
            template_type="system",
            content="Hello {{name}}, you are {{age}} years old.",
        )

        rendered, missing, defaults = svc.render_template(tpl, {"name": "Alice", "age": "30"})
        assert "Alice" in rendered
        assert "30" in rendered
        assert missing == []

    @pytest.mark.asyncio
    async def test_search_by_tags(self, stack) -> None:
        svc = CachedTemplateService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        await svc.create_template(
            owner_id=owner, name="Tagged", template_type="system", content="x", tags=["ai", "research"]
        )
        await svc.create_template(
            owner_id=owner, name="Untagged", template_type="system", content="x", tags=[]
        )

        results = await svc.search_by_tags(owner, ["ai"])
        names = {t.name for t in results}
        assert "Tagged" in names
        assert "Untagged" not in names

    # ------------------------------------------------------------------
    # set_as_default — atomic default-flag flip (replaces _unset_defaults
    # introspection from api/v1/templates.py:235).
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_set_as_default_promotes_one_and_demotes_others(self, stack) -> None:
        svc = CachedTemplateService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        a = await svc.create_template(
            owner_id=owner, name="A", template_type="system", content="a", is_default=True
        )
        b = await svc.create_template(
            owner_id=owner, name="B", template_type="system", content="b", is_default=False
        )
        c = await svc.create_template(
            owner_id=owner, name="C", template_type="system", content="c", is_default=False
        )

        # Initially A is the default.
        assert (await svc.get_for_user(a.id, owner)).is_default is True
        assert (await svc.get_for_user(b.id, owner)).is_default is False
        assert (await svc.get_for_user(c.id, owner)).is_default is False

        await svc.set_as_default(template_id=b.id, owner_id=owner, type_="system")

        # B is now default; A is demoted; C unchanged.
        assert (await svc.get_for_user(a.id, owner)).is_default is False
        assert (await svc.get_for_user(b.id, owner)).is_default is True
        assert (await svc.get_for_user(c.id, owner)).is_default is False

        # Only one default exists for (owner, type) — no scenario where
        # the demote of A and the promote of B both leave is_default=True.
        defaults = [
            t for t in [
                await svc.get_for_user(a.id, owner),
                await svc.get_for_user(b.id, owner),
                await svc.get_for_user(c.id, owner),
            ] if t and t.is_default
        ]
        assert len(defaults) == 1
        assert defaults[0].id == b.id

    @pytest.mark.asyncio
    async def test_set_as_default_scoped_per_type(self, stack) -> None:
        svc = CachedTemplateService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        sys_default = await svc.create_template(
            owner_id=owner, name="SysDefault", template_type="system", content="x", is_default=True
        )
        synth_default = await svc.create_template(
            owner_id=owner, name="SynDefault", template_type="synthesis", content="y", is_default=True
        )
        sys_other = await svc.create_template(
            owner_id=owner, name="SysOther", template_type="system", content="z"
        )

        # Promoting sys_other must NOT touch the synthesis default.
        await svc.set_as_default(template_id=sys_other.id, owner_id=owner, type_="system")

        assert (await svc.get_for_user(sys_default.id, owner)).is_default is False
        assert (await svc.get_for_user(sys_other.id, owner)).is_default is True
        # synthesis default untouched
        assert (await svc.get_for_user(synth_default.id, owner)).is_default is True

    @pytest.mark.asyncio
    async def test_set_as_default_idempotent(self, stack) -> None:
        svc = CachedTemplateService(stack)
        owner = f"user_{uuid4().hex[:8]}"

        a = await svc.create_template(
            owner_id=owner, name="A", template_type="system", content="a", is_default=True
        )
        # Setting the already-default as default again is a no-op.
        await svc.set_as_default(template_id=a.id, owner_id=owner, type_="system")
        assert (await svc.get_for_user(a.id, owner)).is_default is True
