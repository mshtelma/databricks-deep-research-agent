"""Unit tests for KnowledgeAssistantTool class."""

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from deep_research.agent.tools.base import ResearchContext, ResearchTool
from deep_research.agent.tools.knowledge_assistant import (
    KnowledgeAssistantTool,
    create_knowledge_assistant_tools_from_config,
)


def create_test_context() -> ResearchContext:
    """Create a test ResearchContext."""
    return ResearchContext(
        chat_id=uuid4(),
        user_id="test-user",
        research_type="medium",
    )


def create_mock_client(
    answer: str = "Test answer",
    citations: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Create a mock WorkspaceClient with serving_endpoints."""
    mock_client = MagicMock()

    if citations is None:
        citations = [
            {
                "source": "docs",
                "title": "Documentation",
                "url": "https://example.com/docs",
                "snippet": "Relevant text...",
            }
        ]

    mock_response = MagicMock()
    mock_response.predictions = [
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    }
                }
            ]
        }
    ]

    mock_client.serving_endpoints.query.return_value = mock_response
    return mock_client


class TestKnowledgeAssistantToolDefinition:
    """Tests for KnowledgeAssistantTool definition property."""

    def test_implements_research_tool_protocol(self) -> None:
        """KnowledgeAssistantTool should implement ResearchTool protocol."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        assert isinstance(tool, ResearchTool)

    def test_definition_has_generated_name(self) -> None:
        """Tool name should be generated from endpoint name."""
        tool = KnowledgeAssistantTool(endpoint_name="product-assistant")
        assert tool.definition.name == "ask_product_assistant"

    def test_definition_with_custom_name(self) -> None:
        """Should use custom tool name when provided."""
        tool = KnowledgeAssistantTool(
            endpoint_name="test-ka",
            tool_name="custom_ask",
        )
        assert tool.definition.name == "custom_ask"

    def test_definition_has_description(self) -> None:
        """Tool definition should have a description."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        assert tool.definition.description
        assert "test-ka" in tool.definition.description

    def test_definition_with_custom_description(self) -> None:
        """Should use custom description when provided."""
        tool = KnowledgeAssistantTool(
            endpoint_name="test-ka",
            description="Custom KA description",
        )
        assert tool.definition.description == "Custom KA description"

    def test_definition_has_question_parameter(self) -> None:
        """Tool definition should require 'question' parameter."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        params = tool.definition.parameters
        assert "question" in params["properties"]
        assert params["required"] == ["question"]


class TestKnowledgeAssistantToolValidation:
    """Tests for KnowledgeAssistantTool argument validation."""

    def test_valid_question(self) -> None:
        """Should accept valid question."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        errors = tool.validate_arguments({"question": "What is X?"})
        assert errors == []

    def test_missing_question(self) -> None:
        """Should reject missing question."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        errors = tool.validate_arguments({})
        assert len(errors) == 1
        assert "question" in errors[0]

    def test_empty_question(self) -> None:
        """Should reject empty question."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        errors = tool.validate_arguments({"question": ""})
        assert len(errors) == 1
        assert "question" in errors[0]

    def test_question_too_long(self) -> None:
        """Should reject question over 2000 characters."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        errors = tool.validate_arguments({"question": "x" * 2001})
        assert len(errors) == 1
        assert "2000" in errors[0]

    def test_non_string_question(self) -> None:
        """Should reject non-string question."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        errors = tool.validate_arguments({"question": 123})
        assert len(errors) == 1
        assert "string" in errors[0]


class TestKnowledgeAssistantToolExecution:
    """Tests for KnowledgeAssistantTool execute method."""

    @pytest.mark.asyncio
    async def test_successful_query(self) -> None:
        """Should execute query and return answer with citations."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        mock_client = create_mock_client(answer="The answer is 42.")
        tool._client = mock_client

        context = create_test_context()
        result = await tool.execute({"question": "What is the answer?"}, context)

        assert result.success
        assert "The answer is 42." in result.content
        assert "Sources:" in result.content
        assert "Documentation" in result.content

    @pytest.mark.asyncio
    async def test_query_returns_sources(self) -> None:
        """Should include sources for citation tracking."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        mock_client = create_mock_client()
        tool._client = mock_client

        context = create_test_context()
        result = await tool.execute({"question": "Test question"}, context)

        assert result.sources is not None
        assert len(result.sources) == 1
        assert result.sources[0]["type"] == "knowledge_assistant"
        assert result.sources[0]["endpoint_name"] == "test-ka"
        assert result.sources[0]["title"] == "Documentation"

    @pytest.mark.asyncio
    async def test_query_returns_data(self) -> None:
        """Should include data with query info."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        mock_client = create_mock_client()
        tool._client = mock_client

        context = create_test_context()
        result = await tool.execute({"question": "Test question"}, context)

        assert result.data is not None
        assert result.data["question"] == "Test question"
        assert result.data["has_answer"] is True
        assert result.data["citation_count"] == 1

    @pytest.mark.asyncio
    async def test_no_answer(self) -> None:
        """Should handle empty answer."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        mock_client = create_mock_client(answer="", citations=[])
        tool._client = mock_client

        context = create_test_context()
        result = await tool.execute({"question": "Unanswerable"}, context)

        assert result.success
        assert "could not provide an answer" in result.content

    @pytest.mark.asyncio
    async def test_no_citations(self) -> None:
        """Should handle answer without citations."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        mock_client = create_mock_client(answer="Answer without sources", citations=[])
        tool._client = mock_client

        context = create_test_context()
        result = await tool.execute({"question": "Test"}, context)

        assert result.success
        assert "Answer without sources" in result.content
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_query_error_handling(self) -> None:
        """Should handle query errors gracefully."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        mock_client = create_mock_client()
        mock_client.serving_endpoints.query.side_effect = Exception("API error")
        tool._client = mock_client

        context = create_test_context()
        result = await tool.execute({"question": "Test"}, context)

        assert not result.success
        assert result.error is not None
        assert "API error" in result.error

    @pytest.mark.asyncio
    async def test_multiple_citations(self) -> None:
        """Should handle multiple citations."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")
        mock_client = create_mock_client(
            answer="Multi-source answer",
            citations=[
                {"source": "doc1", "title": "First Doc", "url": None, "snippet": None},
                {"source": "doc2", "title": "Second Doc", "url": None, "snippet": None},
            ],
        )
        tool._client = mock_client

        context = create_test_context()
        result = await tool.execute({"question": "Test"}, context)

        assert result.success
        assert len(result.sources) == 2
        assert "[1] First Doc" in result.content
        assert "[2] Second Doc" in result.content


class TestKnowledgeAssistantToolResponseParsing:
    """Tests for response parsing edge cases."""

    @pytest.mark.asyncio
    async def test_parses_dict_response(self) -> None:
        """Should parse dict-based response."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")

        # Create response with dict structure
        mock_response = MagicMock()
        mock_response.predictions = [
            {
                "choices": [
                    {
                        "message": {
                            "content": "Dict answer",
                            "citations": [],
                        }
                    }
                ]
            }
        ]

        mock_client = MagicMock()
        mock_client.serving_endpoints.query.return_value = mock_response
        tool._client = mock_client

        context = create_test_context()
        result = await tool.execute({"question": "Test"}, context)

        assert result.success
        assert "Dict answer" in result.content

    @pytest.mark.asyncio
    async def test_handles_empty_predictions(self) -> None:
        """Should handle empty predictions list."""
        tool = KnowledgeAssistantTool(endpoint_name="test-ka")

        mock_response = MagicMock()
        mock_response.predictions = []

        mock_client = MagicMock()
        mock_client.serving_endpoints.query.return_value = mock_response
        tool._client = mock_client

        context = create_test_context()
        result = await tool.execute({"question": "Test"}, context)

        assert result.success
        assert "could not provide" in result.content


class TestCreateKnowledgeAssistantToolsFromConfig:
    """Tests for create_knowledge_assistant_tools_from_config function."""

    def test_returns_empty_when_disabled(self) -> None:
        """Should return empty list when KA is disabled."""
        mock_config = MagicMock()
        mock_config.enabled = False

        tools = create_knowledge_assistant_tools_from_config(mock_config)
        assert tools == []

    def test_returns_empty_when_no_config(self) -> None:
        """Should return empty list when config is None."""
        tools = create_knowledge_assistant_tools_from_config(None)
        assert tools == []

    def test_creates_tools_from_endpoints(self) -> None:
        """Should create a tool for each enabled endpoint."""
        mock_endpoint1 = MagicMock()
        mock_endpoint1.endpoint_name = "ka1"
        mock_endpoint1.enabled = True
        mock_endpoint1.tool_name = None
        mock_endpoint1.description = None

        mock_endpoint2 = MagicMock()
        mock_endpoint2.endpoint_name = "ka2"
        mock_endpoint2.enabled = True
        mock_endpoint2.tool_name = "custom_ka"
        mock_endpoint2.description = "Custom description"

        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.endpoints = {
            "ep1": mock_endpoint1,
            "ep2": mock_endpoint2,
        }

        tools = create_knowledge_assistant_tools_from_config(mock_config)

        assert len(tools) == 2
        assert tools[0].definition.name == "ask_ka1"
        assert tools[1].definition.name == "custom_ka"

    def test_skips_disabled_endpoints(self) -> None:
        """Should skip disabled endpoints."""
        mock_endpoint = MagicMock()
        mock_endpoint.enabled = False

        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.endpoints = {"ep1": mock_endpoint}

        tools = create_knowledge_assistant_tools_from_config(mock_config)
        assert tools == []
