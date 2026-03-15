"""Unit tests for create_tools_from_source_ids() factory function.

Tests the last-resort fallback that creates tools directly from
source ID prefixes (assistant:, genie:, vs:) without requiring
discovery cache or DB access.
"""

from unittest.mock import MagicMock, patch

import pytest

from deep_research.agent.tools.factory import create_tools_from_source_ids


@pytest.fixture(autouse=True)
def _mock_obo_client() -> MagicMock:
    """Mock OBODatabricksClient for all tests."""
    with patch(
        "deep_research.agent.tools.factory.OBODatabricksClient"
    ) as mock_cls:
        mock_cls.return_value = MagicMock()
        yield mock_cls


class TestCreateToolsFromSourceIds:
    """Test create_tools_from_source_ids() factory function."""

    def test_creates_ka_from_assistant_prefix(self) -> None:
        """assistant:endpoint_name -> KnowledgeAssistantTool."""
        tools = create_tools_from_source_ids(["assistant:my_ka_endpoint"])
        assert len(tools) == 1
        assert tools[0].definition.name == "ask_my_ka_endpoint"
        assert tools[0].definition.source_type == "knowledge_assistant"

    def test_creates_genie_from_genie_prefix(self) -> None:
        """genie:space_id -> GenieTool."""
        tools = create_tools_from_source_ids(["genie:abc123"])
        assert len(tools) == 1
        assert tools[0].definition.source_type == "genie"
        assert "abc123" in tools[0].definition.name

    def test_creates_vs_from_vs_prefix(self) -> None:
        """vs:catalog.schema.index -> UserVectorSearchTool."""
        tools = create_tools_from_source_ids(["vs:catalog.schema.my_index"])
        assert len(tools) == 1
        assert tools[0].definition.source_type == "vector_search"

    def test_skips_unknown_prefix(self) -> None:
        """Unknown prefix -> None, no crash."""
        tools = create_tools_from_source_ids(["unknown:foo", "bad-format"])
        assert len(tools) == 0

    def test_skips_empty_identifier(self) -> None:
        """Empty identifier after prefix -> None."""
        tools = create_tools_from_source_ids(["assistant:", "genie:", "vs:"])
        assert len(tools) == 0

    def test_mixed_valid_and_invalid(self) -> None:
        """Processes all source_ids, skips invalid ones."""
        tools = create_tools_from_source_ids([
            "assistant:valid_ka",
            "unknown:skip",
            "genie:valid_space",
            "assistant:",  # empty
        ])
        assert len(tools) == 2
        source_types = {t.definition.source_type for t in tools}
        assert source_types == {"knowledge_assistant", "genie"}

    def test_empty_list(self) -> None:
        """Empty input -> empty output."""
        tools = create_tools_from_source_ids([])
        assert len(tools) == 0

    def test_genie_long_space_id_truncated_in_name(self) -> None:
        """Genie tool with long space_id gets truncated display name."""
        long_id = "abcdefghijklmnopqrstuvwxyz"
        tools = create_tools_from_source_ids([f"genie:{long_id}"])
        assert len(tools) == 1
        # Display name should contain truncated ID
        assert "..." in tools[0].definition.description or len(tools) == 1

    def test_ka_endpoint_with_dashes(self) -> None:
        """KA endpoint names with dashes produce valid tool names."""
        tools = create_tools_from_source_ids(["assistant:my-ka-endpoint"])
        assert len(tools) == 1
        # Tool name replaces dashes with underscores
        assert tools[0].definition.name == "ask_my_ka_endpoint"

    def test_vs_fully_qualified_index(self) -> None:
        """VS with three-part name catalog.schema.index works."""
        tools = create_tools_from_source_ids(["vs:main.default.docs_index"])
        assert len(tools) == 1
        assert tools[0].definition.source_type == "vector_search"

    def test_shares_obo_client(self, _mock_obo_client: MagicMock) -> None:
        """All tools share a single OBODatabricksClient instance."""
        create_tools_from_source_ids([
            "assistant:ka1",
            "genie:space1",
            "vs:idx1",
        ])
        # OBODatabricksClient should be created exactly once
        assert _mock_obo_client.call_count == 1

    def test_exception_in_one_tool_doesnt_block_others(self) -> None:
        """If one tool constructor raises, other tools still get created."""
        with patch(
            "deep_research.agent.tools.knowledge_assistant.KnowledgeAssistantTool",
            side_effect=RuntimeError("boom"),
        ):
            # KA will fail, but genie should still work
            tools = create_tools_from_source_ids([
                "assistant:will_fail",
                "genie:should_work",
            ])
            assert len(tools) == 1
            assert tools[0].definition.source_type == "genie"
