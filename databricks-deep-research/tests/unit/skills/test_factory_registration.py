"""Verify read_skill is registered with the BuiltinToolFactory."""

from __future__ import annotations

import pytest

from databricks_deep_research.skills.store import FilesystemSkillStore
from databricks_deep_research.tools.builtins.read_skill import ReadSkillTool
from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.workflow.definition import ToolDeclaration


class TestReadSkillRegistration:
    def test_factory_supports_read_skill(self) -> None:
        assert BuiltinToolFactory().supports("read_skill")

    def test_catalog_card_present(self) -> None:
        assert "read_skill" in BuiltinToolFactory.catalog_cards

    async def test_create_builds_read_skill_tool(self) -> None:
        factory = BuiltinToolFactory()
        ctx = ToolFactoryContext(extras={"_skill_store": FilesystemSkillStore()})
        decl = ToolDeclaration(name="read_skill", kind="read_skill")
        tool = await factory.create(decl, ctx)
        assert isinstance(tool, ReadSkillTool)
        assert tool.definition.name == "read_skill"

    async def test_create_requires_skill_store(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(name="read_skill", kind="read_skill")
        with pytest.raises(ValueError, match="_skill_store"):
            await factory.create(decl, ToolFactoryContext())
