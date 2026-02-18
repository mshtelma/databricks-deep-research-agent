"""Unit tests for FileSearchTool."""

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from deep_research.agent.tools.base import ResearchContext
from deep_research.agent.tools.file_search import FileSearchTool


class TestFileSearchTool:
    """Tests for uploaded file search behavior."""

    def test_normalize_file_ids_filters_invalid_values(self) -> None:
        """Only valid UUIDs should be kept in normalized file IDs."""
        valid = str(uuid4())
        normalized = FileSearchTool._normalize_file_ids([valid, "not-a-uuid"])
        assert normalized == {UUID(valid)}

    @pytest.mark.asyncio
    async def test_execute_formats_uploaded_file_source_url(self) -> None:
        """Search results should include stable uploaded-file citation URLs."""
        file_id = uuid4()
        chunk_id = uuid4()
        tool = FileSearchTool(session=MagicMock(), owner_id="user-123")
        tool._search_files = AsyncMock(
            return_value=[
                {
                    "file_id": file_id,
                    "filename": "notes.pdf",
                    "chunk_id": chunk_id,
                    "chunk_index": 2,
                    "content": "Revenue increased by 20 percent year over year.",
                    "score": 0.42,
                    "page_number": 4,
                    "section": "Financial Results",
                    "highlight": "Revenue increased by 20 percent...",
                }
            ]
        )

        context = ResearchContext(chat_id=uuid4(), user_id="user-123")
        result = await tool.execute({"query": "revenue growth"}, context)

        assert result.success is True
        assert result.sources is not None
        assert result.sources[0]["url"] == f"uploaded-file://{file_id}#chunk-2"
        assert "notes.pdf" in result.content

    @pytest.mark.asyncio
    async def test_execute_passes_selected_file_ids(self) -> None:
        """Configured file IDs should be forwarded to the internal search query."""
        selected_file_id = str(uuid4())
        tool = FileSearchTool(
            session=MagicMock(),
            owner_id="user-123",
            file_ids=[selected_file_id],
        )
        tool._search_files = AsyncMock(return_value=[])

        context = ResearchContext(chat_id=uuid4(), user_id="user-123")
        await tool.execute({"query": "summary"}, context)

        kwargs = tool._search_files.await_args.kwargs
        assert kwargs["file_ids"] == {UUID(selected_file_id)}

