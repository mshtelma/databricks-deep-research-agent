"""Tests for YAML-first tool declarations: ToolDeclaration, ToolKind, ToolResolver, factories."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from databricks_deep_research.agents.config import SUBTYPE_DEFAULTS, AgentNodeConfig
from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
from databricks_deep_research.tools.factories.databricks import DatabricksToolFactory
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import (
    ResearchTool,
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolKind,
    ToolResult,
    tool_kind_to_source_kind,
)
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.tools.resolver import ToolResolver
from databricks_deep_research.workflow.definition import (
    NodeType,
    ToolDeclaration,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.loader import load_workflow_from_string

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeTool:
    """Minimal ResearchTool for testing."""

    def __init__(self, name: str = "fake") -> None:
        self._name = name
        self._definition = ToolDefinition(
            name=name,
            description=f"Fake tool {name}",
            parameters={"type": "object", "properties": {}},
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return arguments

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        return ToolResult(content="ok")


class _BrokenFactory:
    """Factory that claims support but fails at creation time."""

    def supports(self, kind: str) -> bool:
        return kind == "web_search"

    async def create(
        self,
        declaration: ToolDeclaration,
        context: ToolFactoryContext,
    ) -> ResearchTool:
        raise RuntimeError("factory exploded")


# ---------------------------------------------------------------------------
# Milestone 1: ToolKind + ToolDeclaration
# ---------------------------------------------------------------------------


class TestToolKind:
    def test_enum_values(self) -> None:
        assert ToolKind.web_search == "web_search"
        assert ToolKind.web_research == "web_research"
        assert ToolKind.vector_search == "vector_search"
        assert ToolKind.genie == "genie"
        assert ToolKind.knowledge_assistant == "knowledge_assistant"
        assert ToolKind.delta_context == "delta_context"

    def test_tool_kind_to_source_kind_known(self) -> None:
        assert tool_kind_to_source_kind("web_search") == SourceKind.web
        assert tool_kind_to_source_kind("vector_search") == SourceKind.vector_index
        assert tool_kind_to_source_kind("genie") == SourceKind.sql_analytics
        assert tool_kind_to_source_kind("knowledge_assistant") == SourceKind.qa_assistant
        assert tool_kind_to_source_kind("file_search") == SourceKind.file
        assert tool_kind_to_source_kind("web_crawl") == SourceKind.builtin
        assert tool_kind_to_source_kind("web_research") == SourceKind.web
        assert tool_kind_to_source_kind("delta_context") == SourceKind.delta_table

    def test_tool_kind_to_source_kind_unknown(self) -> None:
        assert tool_kind_to_source_kind("custom_thing") == SourceKind.builtin
        assert tool_kind_to_source_kind("") == SourceKind.builtin


class TestToolDeclaration:
    def test_basic_validation(self) -> None:
        decl = ToolDeclaration(
            name="my_index",
            kind="vector_search",
            config={"index_name": "cat.schema.idx"},
            description="Test index",
        )
        assert decl.name == "my_index"
        assert decl.kind == "vector_search"
        assert decl.config["index_name"] == "cat.schema.idx"

    def test_defaults(self) -> None:
        decl = ToolDeclaration(name="x", kind="web_search")
        assert decl.config == {}
        assert decl.description == ""

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ToolDeclaration(name="x", kind="y", unknown_field="z")  # type: ignore[call-arg]


class TestWorkflowDefinitionTools:
    def test_tools_field_default_empty(self) -> None:
        defn = WorkflowDefinition(
            id="test",
            name="Test",
            root=WorkflowNode(id="r", type=NodeType.agent, label="R"),
        )
        assert defn.tools == []

    def test_tools_field_accepts_declarations(self) -> None:
        decls = [
            ToolDeclaration(name="a", kind="web_search"),
            ToolDeclaration(name="b", kind="genie", config={"space_id": "x"}),
        ]
        defn = WorkflowDefinition(
            id="test",
            name="Test",
            root=WorkflowNode(id="r", type=NodeType.agent, label="R"),
            tools=decls,
        )
        assert len(defn.tools) == 2
        assert defn.tools[0].name == "a"
        assert defn.tools[1].kind == "genie"


# ---------------------------------------------------------------------------
# Milestone 2: ToolResolver
# ---------------------------------------------------------------------------


class TestToolResolver:
    @pytest.mark.asyncio
    async def test_override_priority(self) -> None:
        """Override wins over declaration."""
        decl = ToolDeclaration(name="web_search", kind="web_search")
        override_tool = _FakeTool("web_search")

        resolver = ToolResolver(declarations=[decl])
        resolver.override("web_search", override_tool)

        result = await resolver.resolve("web_search")
        assert result is override_tool

    @pytest.mark.asyncio
    async def test_string_ref(self) -> None:
        """Resolves plain string name via override."""
        tool = _FakeTool("my_tool")
        resolver = ToolResolver()
        resolver.override("my_tool", tool)

        result = await resolver.resolve("my_tool")
        assert result is tool

    @pytest.mark.asyncio
    async def test_legacy_dict_ref(self) -> None:
        """Resolves legacy {type, name} dict via ToolRegistry fallback."""
        registry = ToolRegistry()
        tool = _FakeTool("web_search")
        registry.register_builtin("web_search", tool)

        resolver = ToolResolver(legacy_registry=registry)
        result = await resolver.resolve({"type": "builtin", "name": "web_search"})
        assert result.definition.name == "web_search"

    @pytest.mark.asyncio
    async def test_missing_tool_raises(self) -> None:
        """Raises ValueError with helpful message for missing tool."""
        resolver = ToolResolver()
        with pytest.raises(ValueError, match="Tool not found: 'nonexistent'"):
            await resolver.resolve("nonexistent")

    @pytest.mark.asyncio
    async def test_no_factory_for_kind_raises(self) -> None:
        """Raises ValueError listing factories when no factory supports the kind."""
        decl = ToolDeclaration(name="x", kind="salesforce_query")
        resolver = ToolResolver(declarations=[decl])

        with pytest.raises(ValueError, match="No factory supports kind='salesforce_query'"):
            await resolver.resolve("x")

    @pytest.mark.asyncio
    async def test_empty_name_raises(self) -> None:
        resolver = ToolResolver()
        with pytest.raises(ValueError, match="empty name"):
            await resolver.resolve("")
        with pytest.raises(ValueError, match="empty name"):
            await resolver.resolve({"type": "builtin", "name": ""})

    @pytest.mark.asyncio
    async def test_resolve_many_collects_errors(self) -> None:
        tool = _FakeTool("good")
        resolver = ToolResolver()
        resolver.override("good", tool)

        results = await resolver.resolve_many(["good", "missing1", "missing2"])
        assert len(results) == 1
        assert results[0] is tool

    @pytest.mark.asyncio
    async def test_cache_hit(self) -> None:
        """Second resolve of same name hits cache."""
        tool = _FakeTool("cached")
        resolver = ToolResolver()
        resolver.override("cached", tool)

        r1 = await resolver.resolve("cached")
        r2 = await resolver.resolve("cached")
        assert r1 is r2

    def test_list_available(self) -> None:
        registry = ToolRegistry()
        registry.register_builtin("web_search", _FakeTool("web_search"))

        resolver = ToolResolver(
            declarations=[ToolDeclaration(name="my_index", kind="vector_search")],
            legacy_registry=registry,
        )
        resolver.override("override_tool", _FakeTool("override_tool"))

        available = resolver.list_available()
        assert "my_index" in available
        assert "override_tool" in available
        assert "web_search" in available

    @pytest.mark.asyncio
    async def test_legacy_name_lookup(self) -> None:
        """String name resolves through legacy registry when tool is registered."""
        registry = ToolRegistry()
        tool = _FakeTool("web_search")
        registry.register_builtin("web_search", tool)

        resolver = ToolResolver(legacy_registry=registry)
        result = await resolver.resolve("web_search")
        assert result.definition.name == "web_search"

    @pytest.mark.asyncio
    async def test_factory_failure_falls_back_to_legacy_registry(self) -> None:
        """Declaration resolution should fall back to the legacy registry on factory failure."""
        decl = ToolDeclaration(name="web_search", kind="web_search")
        registry = ToolRegistry()
        tool = _FakeTool("web_search")
        registry.register_builtin("web_search", tool)

        resolver = ToolResolver(
            declarations=[decl],
            factories=[_BrokenFactory()],
            factory_context=ToolFactoryContext(),
            legacy_registry=registry,
        )

        result = await resolver.resolve("web_search")

        assert result is tool


# ---------------------------------------------------------------------------
# Milestone 3: Loader + AgentConfig
# ---------------------------------------------------------------------------


class TestLoaderToolDeclarations:
    def test_parses_tools_section(self) -> None:
        yaml_content = """
id: test
name: Test
tools:
  - name: my_index
    kind: vector_search
    config:
      index_name: cat.schema.idx
    description: My vector index
root:
  id: r
  type: agent
  label: R
  config:
    subtype: researcher
"""
        defn = load_workflow_from_string(yaml_content)
        assert len(defn.tools) == 1
        assert defn.tools[0].name == "my_index"
        assert defn.tools[0].kind == "vector_search"
        assert defn.tools[0].config["index_name"] == "cat.schema.idx"

    def test_auto_populates_sources(self) -> None:
        yaml_content = """
id: test
name: Test
tools:
  - name: my_index
    kind: vector_search
    config:
      index_name: cat.schema.idx
    description: My vector index
  - name: sales_genie
    kind: genie
    config:
      space_id: abc-123
    description: Sales analytics
root:
  id: r
  type: agent
  label: R
  config:
    subtype: researcher
"""
        defn = load_workflow_from_string(yaml_content)
        assert len(defn.sources) == 2
        # vector_search → vector_index source kind
        assert defn.sources[0].name == "my_index"
        assert defn.sources[0].kind == "vector_index"
        assert defn.sources[0].endpoint == "cat.schema.idx"
        # genie → sql_analytics source kind
        assert defn.sources[1].name == "sales_genie"
        assert defn.sources[1].kind == "sql_analytics"
        assert defn.sources[1].endpoint == "abc-123"

    def test_explicit_sources_not_overridden(self) -> None:
        yaml_content = """
id: test
name: Test
tools:
  - name: my_index
    kind: vector_search
    config:
      index_name: cat.schema.idx
sources:
  - name: explicit_source
    kind: web
root:
  id: r
  type: agent
  label: R
  config:
    subtype: researcher
"""
        defn = load_workflow_from_string(yaml_content)
        assert len(defn.sources) == 1
        assert defn.sources[0].name == "explicit_source"

    def test_no_tools_section(self) -> None:
        yaml_content = """
id: test
name: Test
root:
  id: r
  type: agent
  label: R
  config:
    subtype: coordinator
"""
        defn = load_workflow_from_string(yaml_content)
        assert defn.tools == []


class TestAgentConfigTools:
    def test_string_tools(self) -> None:
        config = AgentNodeConfig(subtype="researcher", tools=["web_search", "web_crawl"])
        assert config.tools == ["web_search", "web_crawl"]

    def test_dict_tools_still_valid(self) -> None:
        config = AgentNodeConfig(
            subtype="researcher",
            tools=[{"type": "builtin", "name": "web_search"}],
        )
        assert len(config.tools) == 1

    def test_mixed_tools(self) -> None:
        config = AgentNodeConfig(
            subtype="researcher",
            tools=["web_search", {"type": "builtin", "name": "web_crawl"}],
        )
        assert len(config.tools) == 2

    def test_researcher_defaults_use_strings(self) -> None:
        defaults = SUBTYPE_DEFAULTS["researcher"]["tools"]
        assert defaults == ["web_search", "web_crawl"]


# ---------------------------------------------------------------------------
# Milestone 5: Factory tests
# ---------------------------------------------------------------------------


class TestBuiltinToolFactory:
    def test_supports(self) -> None:
        f = BuiltinToolFactory()
        assert f.supports("web_search")
        assert f.supports("web_crawl")
        assert f.supports("file_search")
        assert not f.supports("vector_search")
        assert not f.supports("genie")

    @pytest.mark.asyncio
    async def test_missing_search_client(self) -> None:
        f = BuiltinToolFactory()
        decl = ToolDeclaration(name="ws", kind="web_search")
        with pytest.raises(ValueError, match="search_client required"):
            await f.create(decl, ToolFactoryContext())

    @pytest.mark.asyncio
    async def test_web_crawl_uses_builtin_crawler_when_not_injected(self) -> None:
        f = BuiltinToolFactory()
        decl = ToolDeclaration(name="wc", kind="web_crawl")
        tool = await f.create(decl, ToolFactoryContext())
        assert tool.definition.name == "web_crawl"

    @pytest.mark.asyncio
    async def test_missing_file_index(self) -> None:
        f = BuiltinToolFactory()
        decl = ToolDeclaration(name="fs", kind="file_search")
        with pytest.raises(ValueError, match="file_index required"):
            await f.create(decl, ToolFactoryContext())

    @pytest.mark.asyncio
    async def test_creates_web_search(self) -> None:
        f = BuiltinToolFactory()
        mock_client = MagicMock()
        decl = ToolDeclaration(
            name="ws",
            kind="web_search",
            config={"max_results": 3},
        )
        ctx = ToolFactoryContext(search_client=mock_client)
        tool = await f.create(decl, ctx)
        assert tool.definition.name == "web_search"


class TestDatabricksToolFactory:
    def test_supports(self) -> None:
        f = DatabricksToolFactory()
        assert f.supports("vector_search")
        assert f.supports("genie")
        assert f.supports("knowledge_assistant")
        assert not f.supports("web_search")

    @pytest.mark.asyncio
    async def test_missing_workspace_client(self) -> None:
        f = DatabricksToolFactory()
        decl = ToolDeclaration(
            name="vs",
            kind="vector_search",
            config={"index_name": "x"},
        )
        with pytest.raises(ValueError, match="workspace_client required"):
            await f.create(decl, ToolFactoryContext())

    @pytest.mark.asyncio
    async def test_missing_index_name(self) -> None:
        f = DatabricksToolFactory()
        decl = ToolDeclaration(name="vs", kind="vector_search")
        ctx = ToolFactoryContext(workspace_client=MagicMock())
        with pytest.raises(ValueError, match="'index_name' required"):
            await f.create(decl, ctx)

    @pytest.mark.asyncio
    async def test_missing_space_id(self) -> None:
        f = DatabricksToolFactory()
        decl = ToolDeclaration(name="g", kind="genie")
        ctx = ToolFactoryContext(workspace_client=MagicMock())
        with pytest.raises(ValueError, match="'space_id' required"):
            await f.create(decl, ctx)

    @pytest.mark.asyncio
    async def test_missing_endpoint_name(self) -> None:
        f = DatabricksToolFactory()
        decl = ToolDeclaration(name="ka", kind="knowledge_assistant")
        ctx = ToolFactoryContext(workspace_client=MagicMock())
        with pytest.raises(ValueError, match="'endpoint_name' required"):
            await f.create(decl, ctx)

    @pytest.mark.asyncio
    async def test_creates_vector_search(self) -> None:
        f = DatabricksToolFactory()
        decl = ToolDeclaration(
            name="my_vs",
            kind="vector_search",
            config={"index_name": "cat.schema.idx", "num_results": 5},
            description="My index",
        )
        ctx = ToolFactoryContext(workspace_client=MagicMock())
        tool = await f.create(decl, ctx)
        assert tool.definition.name == "my_vs"
        assert tool.definition.source_kind == SourceKind.vector_index

    @pytest.mark.asyncio
    async def test_creates_genie(self) -> None:
        f = DatabricksToolFactory()
        decl = ToolDeclaration(
            name="sales",
            kind="genie",
            config={"space_id": "abc-123"},
            description="Sales analytics",
        )
        ctx = ToolFactoryContext(workspace_client=MagicMock())
        tool = await f.create(decl, ctx)
        assert tool.definition.name == "sales"
        assert tool.definition.source_kind == SourceKind.sql_analytics

    @pytest.mark.asyncio
    async def test_creates_knowledge_assistant(self) -> None:
        f = DatabricksToolFactory()
        decl = ToolDeclaration(
            name="docs",
            kind="knowledge_assistant",
            config={"endpoint_name": "my-endpoint"},
            description="Documentation Q&A",
        )
        ctx = ToolFactoryContext(workspace_client=MagicMock())
        tool = await f.create(decl, ctx)
        assert tool.definition.name == "docs"
        assert tool.definition.source_kind == SourceKind.qa_assistant


# ---------------------------------------------------------------------------
# Milestone 6: Tool implementation unit tests
# ---------------------------------------------------------------------------


class TestDatabricksVectorSearchTool:
    def test_validate_arguments(self) -> None:
        from databricks_deep_research.tools.builtins.vector_search import (
            DatabricksVectorSearchTool,
        )

        tool = DatabricksVectorSearchTool(
            workspace_client=MagicMock(),
            name="test",
            index_name="cat.schema.idx",
        )
        validated = tool.validate_arguments({"query": "test query"})
        assert validated["query"] == "test query"
        assert validated["num_results"] == 10

    def test_validate_empty_query_raises(self) -> None:
        from databricks_deep_research.tools.builtins.vector_search import (
            DatabricksVectorSearchTool,
        )

        tool = DatabricksVectorSearchTool(
            workspace_client=MagicMock(),
            name="test",
            index_name="cat.schema.idx",
        )
        with pytest.raises(ValueError, match="query"):
            tool.validate_arguments({"query": ""})


class TestDatabricksGenieTool:
    def test_validate_arguments(self) -> None:
        from databricks_deep_research.tools.builtins.genie import DatabricksGenieTool

        tool = DatabricksGenieTool(
            workspace_client=MagicMock(),
            name="test",
            space_id="abc",
        )
        validated = tool.validate_arguments({"question": "What is revenue?"})
        assert validated["question"] == "What is revenue?"

    def test_validate_empty_question_raises(self) -> None:
        from databricks_deep_research.tools.builtins.genie import DatabricksGenieTool

        tool = DatabricksGenieTool(
            workspace_client=MagicMock(),
            name="test",
            space_id="abc",
        )
        with pytest.raises(ValueError, match="question"):
            tool.validate_arguments({"question": ""})


class TestDatabricksKnowledgeAssistantTool:
    def test_validate_arguments(self) -> None:
        from databricks_deep_research.tools.builtins.knowledge_assistant import (
            DatabricksKnowledgeAssistantTool,
        )

        tool = DatabricksKnowledgeAssistantTool(
            workspace_client=MagicMock(),
            name="test",
            endpoint_name="my-endpoint",
        )
        validated = tool.validate_arguments({"question": "How do I deploy?"})
        assert validated["question"] == "How do I deploy?"


# ---------------------------------------------------------------------------
# Milestone 4: Executor integration
# ---------------------------------------------------------------------------


class TestExecutorResolverIntegration:
    @pytest.mark.asyncio
    async def test_executor_resolves_string_refs(self) -> None:
        """Executor with ToolResolver resolves tool name strings."""
        from databricks_deep_research.workflow.executor import WorkflowExecutor

        tool = _FakeTool("web_search")
        resolver = ToolResolver()
        resolver.override("web_search", tool)

        defn = WorkflowDefinition(
            id="test",
            name="Test",
            root=WorkflowNode(
                id="agent1",
                type=NodeType.agent,
                label="Agent",
                config={
                    "subtype": "researcher",
                    "tools": ["web_search"],
                    "max_tool_calls": 0,
                },
            ),
        )

        llm = MagicMock()
        llm.resolve_model = MagicMock(return_value="test-model")
        llm.complete = AsyncMock(
            return_value=MagicMock(content="mock", usage={})
        )

        executor = WorkflowExecutor(defn, llm, tool_resolver=resolver)
        # Verify executor was created without error
        assert executor._resolver is resolver

    @pytest.mark.asyncio
    async def test_executor_backward_compat(self) -> None:
        """Executor with old tool_registry still works."""
        from databricks_deep_research.workflow.executor import WorkflowExecutor

        registry = ToolRegistry()
        tool = _FakeTool("web_search")
        registry.register_builtin("web_search", tool)

        defn = WorkflowDefinition(
            id="test",
            name="Test",
            root=WorkflowNode(
                id="agent1",
                type=NodeType.agent,
                label="Agent",
                config={"subtype": "coordinator"},
            ),
        )

        llm = MagicMock()
        llm.resolve_model = MagicMock(return_value="test-model")

        executor = WorkflowExecutor(defn, llm, tool_registry=registry)
        # Verify resolver wraps the registry
        assert executor._resolver._legacy is registry


# ---------------------------------------------------------------------------
# Example YAML loading
# ---------------------------------------------------------------------------


class TestExamplesLoad:
    """Verify all example YAML files load without errors."""

    _examples_dir = Path(__file__).parent.parent / "examples"

    @pytest.mark.parametrize(
        "yaml_file",
        sorted(Path(__file__).parent.parent.joinpath("examples").glob("*.yaml")),
        ids=lambda p: p.stem,
    )
    def test_example_loads(self, yaml_file: Path) -> None:
        from databricks_deep_research.workflow.loader import load_workflow

        defn = load_workflow(yaml_file)
        assert defn.id
        assert defn.name
        assert defn.root is not None
