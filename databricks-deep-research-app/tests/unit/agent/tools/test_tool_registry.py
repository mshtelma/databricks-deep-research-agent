"""Unit tests for ToolRegistry."""

from typing import Any
from uuid import uuid4

import pytest

from deep_research.agent.tools import (
    ResearchContext,
    ResearchTool,
    ToolDefinition,
    ToolRegistry,
    ToolRegistryError,
    ToolResult,
)


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str = "mock_tool", description: str = "A mock tool") -> None:
        self._definition = ToolDefinition(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ResearchContext,
    ) -> ToolResult:
        return ToolResult(
            content=f"Mock result for: {arguments.get('query', '')}",
            success=True,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        errors = []
        if "query" not in arguments:
            errors.append("Missing required argument: query")
        return errors


@pytest.fixture
def registry() -> ToolRegistry:
    """Create a fresh tool registry."""
    return ToolRegistry()


@pytest.fixture
def mock_tool() -> MockTool:
    """Create a mock tool."""
    return MockTool()


@pytest.fixture
def research_context() -> ResearchContext:
    """Create a research context."""
    return ResearchContext(
        chat_id=uuid4(),
        user_id="test-user",
    )


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test registering a tool."""
        registry.register(mock_tool)
        assert "mock_tool" in registry
        assert len(registry) == 1

    def test_register_duplicate_raises_error(
        self, registry: ToolRegistry, mock_tool: MockTool
    ) -> None:
        """Test that registering a duplicate tool raises an error."""
        registry.register(mock_tool)
        with pytest.raises(ToolRegistryError) as exc_info:
            registry.register(mock_tool)
        assert "already registered" in str(exc_info.value)

    def test_register_with_prefix(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test registering a tool with a prefix."""
        prefixed_name = registry.register_with_prefix(mock_tool, "my_plugin")
        assert prefixed_name == "my_plugin_mock_tool"
        assert "my_plugin_mock_tool" in registry
        assert "mock_tool" not in registry

    def test_get_tool(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test getting a tool by name."""
        registry.register(mock_tool)
        tool = registry.get("mock_tool")
        assert tool is not None
        assert tool.definition.name == "mock_tool"

    def test_get_nonexistent_tool(self, registry: ToolRegistry) -> None:
        """Test getting a nonexistent tool returns None."""
        tool = registry.get("nonexistent")
        assert tool is None

    def test_unregister_tool(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test unregistering a tool."""
        registry.register(mock_tool)
        result = registry.unregister("mock_tool")
        assert result is True
        assert "mock_tool" not in registry
        assert len(registry) == 0

    def test_unregister_nonexistent_tool(self, registry: ToolRegistry) -> None:
        """Test unregistering a nonexistent tool returns False."""
        result = registry.unregister("nonexistent")
        assert result is False

    def test_list_tools(self, registry: ToolRegistry) -> None:
        """Test listing all tool definitions."""
        tool1 = MockTool("tool1", "Tool 1")
        tool2 = MockTool("tool2", "Tool 2")
        registry.register(tool1)
        registry.register(tool2)

        definitions = registry.list_tools()
        assert len(definitions) == 2
        names = [d.name for d in definitions]
        assert "tool1" in names
        assert "tool2" in names

    def test_get_tool_names(self, registry: ToolRegistry) -> None:
        """Test getting all tool names."""
        tool1 = MockTool("tool1", "Tool 1")
        tool2 = MockTool("tool2", "Tool 2")
        registry.register(tool1)
        registry.register(tool2)

        names = registry.get_tool_names()
        assert len(names) == 2
        assert "tool1" in names
        assert "tool2" in names

    def test_get_openai_tools(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test converting to OpenAI function calling format."""
        registry.register(mock_tool)
        openai_tools = registry.get_openai_tools()

        assert len(openai_tools) == 1
        tool = openai_tools[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "mock_tool"
        assert tool["function"]["description"] == "A mock tool"
        assert "parameters" in tool["function"]
        assert tool["function"]["parameters"]["type"] == "object"

    def test_clear_registry(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test clearing the registry."""
        registry.register(mock_tool)
        assert len(registry) == 1
        registry.clear()
        assert len(registry) == 0

    def test_iterate_registry(self, registry: ToolRegistry) -> None:
        """Test iterating over registry."""
        tool1 = MockTool("tool1", "Tool 1")
        tool2 = MockTool("tool2", "Tool 2")
        registry.register(tool1)
        registry.register(tool2)

        tools = list(registry)
        assert len(tools) == 2


class TestResearchContext:
    """Tests for ResearchContext."""

    def test_create_context(self) -> None:
        """Test creating a research context."""
        chat_id = uuid4()
        context = ResearchContext(
            chat_id=chat_id,
            user_id="test-user",
        )
        assert context.chat_id == chat_id
        assert context.user_id == "test-user"
        assert context.research_type == "medium"  # default
        assert context.tool_call_count == 0
        assert context.max_tool_calls == 20

    def test_context_with_custom_values(self) -> None:
        """Test creating a context with custom values."""
        chat_id = uuid4()
        session_id = uuid4()
        context = ResearchContext(
            chat_id=chat_id,
            user_id="test-user",
            research_session_id=session_id,
            research_type="extended",
            tool_call_count=5,
            max_tool_calls=30,
        )
        assert context.research_session_id == session_id
        assert context.research_type == "extended"
        assert context.tool_call_count == 5
        assert context.max_tool_calls == 30


class TestToolDefinition:
    """Tests for ToolDefinition."""

    def test_create_definition(self) -> None:
        """Test creating a tool definition."""
        definition = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        assert definition.name == "test_tool"
        assert definition.description == "A test tool"
        assert definition.parameters["type"] == "object"

    def test_definition_is_frozen(self) -> None:
        """Test that ToolDefinition is immutable."""
        definition = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={},
        )
        with pytest.raises(AttributeError):
            definition.name = "new_name"  # type: ignore[misc]


class TestToolResult:
    """Tests for ToolResult."""

    def test_create_success_result(self) -> None:
        """Test creating a successful result."""
        result = ToolResult(
            content="Result content",
            success=True,
        )
        assert result.content == "Result content"
        assert result.success is True
        assert result.sources is None
        assert result.error is None

    def test_create_failure_result(self) -> None:
        """Test creating a failure result."""
        result = ToolResult(
            content="",
            success=False,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_create_result_with_sources(self) -> None:
        """Test creating a result with sources."""
        result = ToolResult(
            content="Result with sources",
            success=True,
            sources=[
                {"type": "web", "url": "https://example.com", "title": "Example"},
            ],
        )
        assert result.sources is not None
        assert len(result.sources) == 1
        assert result.sources[0]["type"] == "web"


class TestMockToolExecution:
    """Tests for mock tool execution."""

    @pytest.mark.asyncio
    async def test_execute_tool(
        self, mock_tool: MockTool, research_context: ResearchContext
    ) -> None:
        """Test executing a mock tool."""
        result = await mock_tool.execute(
            {"query": "test query"},
            research_context,
        )
        assert result.success is True
        assert "test query" in result.content

    def test_validate_arguments(self, mock_tool: MockTool) -> None:
        """Test validating arguments."""
        # Valid arguments
        errors = mock_tool.validate_arguments({"query": "test"})
        assert errors == []

        # Missing required argument
        errors = mock_tool.validate_arguments({})
        assert len(errors) == 1
        assert "query" in errors[0]


class TestResearchToolProtocol:
    """Tests for ResearchTool protocol."""

    def test_mock_tool_is_research_tool(self, mock_tool: MockTool) -> None:
        """Test that MockTool satisfies the ResearchTool protocol."""
        assert isinstance(mock_tool, ResearchTool)
