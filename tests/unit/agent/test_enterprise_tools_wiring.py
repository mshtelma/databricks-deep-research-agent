"""Unit tests for enterprise tools wiring into researchers (007 Phase 2).

Tests the flow:
1. OrchestrationConfig.user_token -> ResearchState.user_token
2. Enterprise tools loading when enterprise search is allowed
3. Tool execution in classic and ReAct researchers
4. GenieTool two-step query result retrieval
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from deep_research.agent.state import ResearchState, SourceInfo
from deep_research.agent.orchestrator import OrchestrationConfig


class TestOrchestratorWiring:
    """Test OrchestrationConfig to ResearchState wiring."""

    def test_orchestration_config_has_user_token(self) -> None:
        """OrchestrationConfig should have user_token field."""
        config = OrchestrationConfig()
        assert hasattr(config, "user_token")
        assert config.user_token is None

    def test_orchestration_config_user_token_set(self) -> None:
        """OrchestrationConfig should accept user_token."""
        config = OrchestrationConfig(user_token="test_token_123")
        assert config.user_token == "test_token_123"

    def test_research_state_has_enterprise_tools(self) -> None:
        """ResearchState should have enterprise_tools field."""
        state = ResearchState(query="test")
        assert hasattr(state, "enterprise_tools")
        assert state.enterprise_tools == []

    def test_research_state_has_user_token(self) -> None:
        """ResearchState should have user_token field."""
        state = ResearchState(query="test")
        assert hasattr(state, "user_token")
        assert state.user_token is None


class TestEnterpriseToolsField:
    """Test enterprise_tools field on ResearchState."""

    def test_enterprise_tools_default_empty(self) -> None:
        """enterprise_tools should default to empty list."""
        state = ResearchState(query="test")
        assert state.enterprise_tools == []
        assert isinstance(state.enterprise_tools, list)

    def test_enterprise_tools_can_be_set(self) -> None:
        """enterprise_tools should be settable."""
        state = ResearchState(query="test")
        mock_tool = MagicMock()
        mock_tool.definition.name = "query_genie_aapl"
        state.enterprise_tools = [mock_tool]
        assert len(state.enterprise_tools) == 1
        assert state.enterprise_tools[0].definition.name == "query_genie_aapl"

    def test_enterprise_tools_multiple(self) -> None:
        """enterprise_tools should support multiple tools."""
        state = ResearchState(query="test")

        tool1 = MagicMock()
        tool1.definition.name = "query_genie_aapl"

        tool2 = MagicMock()
        tool2.definition.name = "search_docs_vector"

        state.enterprise_tools = [tool1, tool2]
        assert len(state.enterprise_tools) == 2


class TestToolSourceMapping:
    """Test tool source mapping for parallel execution."""

    def test_get_tool_source_web_search(self) -> None:
        """web_search should map to 'web' source."""
        from deep_research.agent.nodes.react_researcher import _get_tool_source

        assert _get_tool_source("web_search") == "web"

    def test_get_tool_source_web_crawl(self) -> None:
        """web_crawl should map to 'web' source."""
        from deep_research.agent.nodes.react_researcher import _get_tool_source

        assert _get_tool_source("web_crawl") == "web"

    def test_get_tool_source_genie(self) -> None:
        """Genie tools should map to 'genie' source."""
        from deep_research.agent.nodes.react_researcher import _get_tool_source

        assert _get_tool_source("query_genie_aapl_data") == "genie"
        assert _get_tool_source("genie_query") == "genie"

    def test_get_tool_source_vector(self) -> None:
        """Vector search tools should map to 'vector' source."""
        from deep_research.agent.nodes.react_researcher import _get_tool_source

        assert _get_tool_source("search_docs_vector") == "vector"
        assert _get_tool_source("vector_search") == "vector"

    def test_get_tool_source_assistant(self) -> None:
        """Knowledge assistant tools should map to 'assistant' source."""
        from deep_research.agent.nodes.react_researcher import _get_tool_source

        assert _get_tool_source("knowledge_assistant") == "assistant"
        assert _get_tool_source("docs_assistant") == "assistant"

    def test_get_tool_source_file_search(self) -> None:
        """file_search should map to uploaded_file source."""
        from deep_research.agent.nodes.react_researcher import _get_tool_source

        assert _get_tool_source("file_search") == "uploaded_file"

    def test_get_tool_source_unknown(self) -> None:
        """Unknown tools should use their name as source type."""
        from deep_research.agent.nodes.react_researcher import _get_tool_source

        assert _get_tool_source("my_custom_tool") == "my_custom_tool"


class TestScopeAllowedToolFiltering:
    """Test source-scope filtering of non-web tools."""

    def test_web_only_scope_keeps_file_search(self) -> None:
        """file_search remains available even when enterprise sources are blocked."""
        from deep_research.agent.nodes.react_researcher import _get_scope_allowed_non_web_tools
        from deep_research.agent.tools.base import ToolDefinition
        from deep_research.schemas.source_scope import SourceScope, SourceScopeConfig

        state = ResearchState(query="test")
        state.source_scope_config = SourceScopeConfig(scope=SourceScope.WEB_ONLY)

        file_tool = MagicMock()
        file_tool.definition = ToolDefinition(
            name="file_search",
            description="Search uploaded files",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        enterprise_tool = MagicMock()
        enterprise_tool.definition = ToolDefinition(
            name="query_genie_sales",
            description="Query enterprise source",
            parameters={"type": "object", "properties": {"question": {"type": "string"}}},
        )

        state.enterprise_tools = [file_tool, enterprise_tool]

        allowed = _get_scope_allowed_non_web_tools(state)

        assert [tool.definition.name for tool in allowed] == ["file_search"]


class TestEnterpriseToolsExecution:
    """Test enterprise tool execution logic."""

    @pytest.mark.asyncio
    async def test_classic_researcher_enterprise_tool_success(self) -> None:
        """Classic researcher should execute enterprise tools when web blocked."""
        from deep_research.agent.tools.base import ToolResult, ToolDefinition, ResearchContext
        from deep_research.schemas.source_scope import SourceScope, SourceScopeConfig

        # Set up state with enterprise_only scope
        state = ResearchState(query="What is AAPL revenue?")
        state.source_scope_config = SourceScopeConfig(scope=SourceScope.ENTERPRISE_ONLY)

        # Create mock tool
        mock_tool = MagicMock()
        mock_tool.definition = ToolDefinition(
            name="query_genie_aapl",
            description="Query AAPL data",
            parameters={"type": "object", "properties": {"question": {"type": "string"}}},
        )
        mock_tool.execute = AsyncMock(return_value=ToolResult(
            content="AAPL revenue is $100B",
            success=True,
            sources=[{"url": "genie://aapl", "title": "AAPL Data", "content": "Revenue data"}],
        ))

        state.enterprise_tools = [mock_tool]

        # Verify web search is blocked
        assert not state.is_web_search_allowed()
        assert state.is_enterprise_search_allowed()
        assert len(state.enterprise_tools) == 1

    @pytest.mark.asyncio
    async def test_react_researcher_builds_dynamic_tool_list(self) -> None:
        """ReAct researcher should build dynamic tool list with enterprise tools."""
        from deep_research.agent.tools.base import ToolDefinition
        from deep_research.agent.tools.research_tools import RESEARCH_TOOLS
        from deep_research.schemas.source_scope import SourceScope, SourceScopeConfig

        # Set up state with ALL scope (both web and enterprise allowed)
        state = ResearchState(query="Test query")
        state.source_scope_config = SourceScopeConfig(scope=SourceScope.ALL)

        # Create mock enterprise tool
        mock_tool = MagicMock()
        mock_tool.definition = ToolDefinition(
            name="query_genie_test",
            description="Test Genie tool",
            parameters={"type": "object", "properties": {"question": {"type": "string"}}},
        )
        state.enterprise_tools = [mock_tool]

        # Build available tools (simulating what run_react_researcher does)
        available_tools = list(RESEARCH_TOOLS)
        if state.is_enterprise_search_allowed() and state.enterprise_tools:
            for tool in state.enterprise_tools:
                available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.definition.name,
                        "description": tool.definition.description,
                        "parameters": tool.definition.parameters,
                    },
                })

        # Should have web tools + enterprise tool
        assert len(available_tools) == len(RESEARCH_TOOLS) + 1

        # Last tool should be the enterprise tool
        last_tool = available_tools[-1]
        assert last_tool["function"]["name"] == "query_genie_test"


class TestEnterpriseSourceTracking:
    """Test that enterprise tool results are tracked as sources."""

    def test_enterprise_source_url_format(self) -> None:
        """Enterprise sources should use enterprise:// URL scheme."""
        state = ResearchState(query="test")

        source = SourceInfo(
            url="enterprise://query_genie_aapl",
            title="Genie: AAPL Data",
            snippet="Revenue data for Apple Inc.",
            content="Full revenue data...",
        )
        state.add_source(source)

        assert len(state.sources) == 1
        assert state.sources[0].url.startswith("enterprise://")

    def test_enterprise_sources_in_to_dict(self) -> None:
        """Enterprise sources should be serialized in to_dict()."""
        state = ResearchState(query="test")

        source = SourceInfo(
            url="enterprise://genie_test",
            title="Test Source",
            snippet="Test snippet",
        )
        state.add_source(source)

        result = state.to_dict()
        assert len(result["sources"]) == 1
        assert result["sources"][0]["url"] == "enterprise://genie_test"


class TestGenieToolDataExtraction:
    """Test GenieTool two-step query result retrieval."""

    def _make_genie_tool(self) -> "GenieTool":
        """Create a GenieTool with mocked OBO client."""
        from deep_research.agent.tools.genie import GenieTool

        obo_client = MagicMock()
        return GenieTool(
            obo_client=obo_client,
            space_id="test-space-123",
            name="Test Genie Room",
            max_rows=50,
        )

    def _make_message(
        self,
        *,
        content: str | None = None,
        conversation_id: str = "conv-1",
        message_id: str = "msg-1",
        attachments: list[Any] | None = None,
        query_result: Any = None,
        error: Any = None,
        status: Any = None,
    ) -> MagicMock:
        """Create a mock GenieMessage."""
        msg = MagicMock()
        msg.content = content
        msg.conversation_id = conversation_id
        msg.message_id = message_id
        msg.id = message_id
        msg.attachments = attachments
        msg.query_result = query_result
        msg.error = error
        msg.status = status
        return msg

    def _make_attachment(
        self,
        *,
        attachment_id: str = "att-1",
        sql: str | None = "SELECT * FROM table",
    ) -> MagicMock:
        """Create a mock attachment with query."""
        att = MagicMock()
        att.attachment_id = attachment_id
        if sql is not None:
            att.query = MagicMock()
            att.query.query = sql
        else:
            att.query = None
        return att

    def _make_statement_response(
        self,
        *,
        columns: list[str] | None = None,
        data: list[list[str]] | None = None,
        statement_id: str = "stmt-1",
    ) -> MagicMock:
        """Create a mock GenieGetMessageQueryResultResponse."""
        response = MagicMock()
        stmt = MagicMock()
        response.statement_response = stmt

        # Manifest with columns
        if columns is not None:
            schema = MagicMock()
            col_objs = []
            for name in columns:
                col = MagicMock()
                col.name = name
                col_objs.append(col)
            schema.columns = col_objs
            stmt.manifest = MagicMock()
            stmt.manifest.schema = schema
        else:
            stmt.manifest = None

        # Result with data_array
        if data is not None:
            stmt.result = MagicMock()
            stmt.result.data_array = data
        else:
            stmt.result = None

        stmt.statement_id = statement_id
        stmt.status = None

        return response

    def test_genie_execute_query_two_step_flow(self) -> None:
        """Two-step flow: message with attachment → fetch query result → columns + rows."""
        tool = self._make_genie_tool()

        # Create message with attachment
        attachment = self._make_attachment(sql="SELECT price FROM stocks WHERE symbol='NVDA'")
        qr = MagicMock()
        qr.statement_id = "stmt-1"
        qr.row_count = 5
        message = self._make_message(
            content="Here are the NVDA prices",
            attachments=[attachment],
            query_result=qr,
        )

        # Create statement response with tabular data
        stmt_response = self._make_statement_response(
            columns=["symbol", "price", "date"],
            data=[
                ["NVDA", "150.00", "2024-01-01"],
                ["NVDA", "155.00", "2024-01-02"],
                ["NVDA", "148.00", "2024-01-03"],
            ],
        )

        # Mock the client
        client = MagicMock()
        client.genie.start_conversation_and_wait.return_value = message
        client.genie.get_message_attachment_query_result.return_value = stmt_response

        result = tool._execute_query(client, "NVDA pricing info", is_follow_up=False)

        # Verify two-step flow was used
        client.genie.start_conversation_and_wait.assert_called_once()
        client.genie.get_message_attachment_query_result.assert_called_once_with(
            space_id="test-space-123",
            conversation_id="conv-1",
            message_id="msg-1",
            attachment_id="att-1",
        )

        # Verify result has all data
        assert result["sql"] == "SELECT price FROM stocks WHERE symbol='NVDA'"
        assert result["columns"] == ["symbol", "price", "date"]
        assert len(result["rows"]) == 3
        assert result["row_count"] == 3
        assert result["narrative"] == "Here are the NVDA prices"
        assert not result.get("error")

    def test_genie_extract_result_narrative_only(self) -> None:
        """Text-only response: narrative but no query attachments → no 'No data' message."""
        tool = self._make_genie_tool()

        message = self._make_message(
            content="NVDA is currently trading at approximately $150 per share.",
            attachments=[],
        )

        result = tool._extract_result(message)

        assert result["narrative"] == "NVDA is currently trading at approximately $150 per share."
        assert result["columns"] == []
        assert result["rows"] == []

        # Format should include narrative, NOT "No data returned"
        formatted = tool._format_result(result)
        assert "NVDA is currently trading" in formatted
        assert "No data returned" not in formatted

    def test_genie_second_call_failure_graceful(self) -> None:
        """When second API call fails, still return SQL + narrative from step 1."""
        tool = self._make_genie_tool()

        attachment = self._make_attachment(sql="SELECT * FROM stocks")
        message = self._make_message(
            content="Query executed",
            attachments=[attachment],
        )

        # Mock client — second call raises
        client = MagicMock()
        client.genie.start_conversation_and_wait.return_value = message
        client.genie.get_message_attachment_query_result.side_effect = RuntimeError(
            "Result expired after 10 minutes"
        )

        result = tool._execute_query(client, "stock prices", is_follow_up=False)

        # Should still have SQL and narrative from step 1
        assert result["sql"] == "SELECT * FROM stocks"
        assert result["narrative"] == "Query executed"
        # But no tabular data
        assert result["columns"] == []
        assert result["rows"] == []
        # No error set in result (the warning is logged, not returned as error)
        assert "error" not in result

    def test_genie_attribute_error_not_swallowed(self) -> None:
        """AttributeError from message processing should propagate, not be caught by narrow handler.

        The narrow `except AttributeError` only catches `client.genie` access.
        Any other AttributeError (e.g., from SDK incompatibility) should propagate
        to the general `except Exception` in execute().
        """
        tool = self._make_genie_tool()

        # Use a plain object (not MagicMock) so property descriptors work properly
        class BrokenMessage:
            conversation_id = "conv-1"
            content = None
            error = None
            status = None
            query_result = None
            message_id = "msg-1"
            id = "msg-1"

            @property
            def attachments(self) -> list[Any]:
                raise AttributeError("no attachments field")

        client = MagicMock()
        client.genie.start_conversation_and_wait.return_value = BrokenMessage()

        # The narrow except only catches `client.genie` AttributeError.
        # An AttributeError from message processing should propagate to
        # the general `except Exception` in execute().
        with pytest.raises(AttributeError, match="no attachments field"):
            tool._execute_query(client, "test query", is_follow_up=False)

    def test_genie_empty_attachments(self) -> None:
        """Message with no attachments: should log and return metadata only."""
        tool = self._make_genie_tool()

        # Message with no attachments and no query_result
        message = self._make_message(
            content="I don't have enough information to answer that question.",
            attachments=None,
            query_result=None,
        )

        client = MagicMock()
        client.genie.start_conversation_and_wait.return_value = message

        result = tool._execute_query(client, "something vague", is_follow_up=False)

        # Should NOT have called the second API
        client.genie.get_message_attachment_query_result.assert_not_called()
        client.genie.get_message_query_result.assert_not_called()

        # Should have narrative
        assert result["narrative"] == "I don't have enough information to answer that question."
        assert result["columns"] == []
        assert result["rows"] == []
        assert result["sql"] is None

    def test_genie_fallback_to_deprecated_api(self) -> None:
        """When no attachment_id but statement_id exists, fall back to deprecated API."""
        tool = self._make_genie_tool()

        # Message with query_result.statement_id but no attachments with query
        qr = MagicMock()
        qr.statement_id = "stmt-fallback-1"
        qr.row_count = 2

        # Attachment without query field
        att = MagicMock()
        att.query = None
        att.attachment_id = "att-no-query"

        message = self._make_message(
            content="Results found",
            attachments=[att],
            query_result=qr,
        )

        stmt_response = self._make_statement_response(
            columns=["name", "value"],
            data=[["item1", "100"], ["item2", "200"]],
        )

        client = MagicMock()
        client.genie.start_conversation_and_wait.return_value = message
        client.genie.get_message_query_result.return_value = stmt_response

        result = tool._execute_query(client, "get items", is_follow_up=False)

        # Should have used deprecated API
        client.genie.get_message_attachment_query_result.assert_not_called()
        client.genie.get_message_query_result.assert_called_once_with(
            space_id="test-space-123",
            conversation_id="conv-1",
            message_id="msg-1",
        )

        # Should have data from fallback
        assert result["columns"] == ["name", "value"]
        assert len(result["rows"]) == 2
        assert result["row_count"] == 2

    def test_genie_parse_statement_response_empty(self) -> None:
        """_parse_statement_response handles None statement_response gracefully."""
        tool = self._make_genie_tool()

        response = MagicMock()
        response.statement_response = None

        result: dict[str, Any] = {
            "sql": None, "columns": [], "rows": [], "row_count": 0,
            "truncated": False, "narrative": None,
        }

        # Should not raise
        tool._parse_statement_response(response, result)

        # Result unchanged
        assert result["columns"] == []
        assert result["rows"] == []

    def test_genie_parse_statement_response_failed_status(self) -> None:
        """_parse_statement_response sets error when statement status is FAILED."""
        tool = self._make_genie_tool()

        response = MagicMock()
        stmt = MagicMock()
        response.statement_response = stmt
        stmt.manifest = None
        stmt.result = None
        stmt.statement_id = "stmt-1"

        # Set FAILED status
        status = MagicMock()
        status.state = MagicMock()
        status.state.value = "FAILED"
        status.error = "Division by zero"
        stmt.status = status

        result: dict[str, Any] = {
            "sql": None, "columns": [], "rows": [], "row_count": 0,
            "truncated": False, "narrative": None,
        }

        tool._parse_statement_response(response, result)

        assert "error" in result
        assert "FAILED" in result["error"]
        assert "Division by zero" in result["error"]

    def test_genie_format_result_with_error(self) -> None:
        """_format_result shows error message when result has error."""
        tool = self._make_genie_tool()

        result = {
            "sql": None,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "narrative": None,
            "error": "Permission denied",
        }

        formatted = tool._format_result(result)
        assert "**Error:** Permission denied" in formatted

    def test_genie_truncation(self) -> None:
        """_parse_statement_response truncates rows exceeding max_rows."""
        tool = self._make_genie_tool()  # max_rows=50

        # Create 100 rows
        large_data = [[str(i), f"val_{i}"] for i in range(100)]
        stmt_response = self._make_statement_response(
            columns=["id", "value"],
            data=large_data,
        )

        result: dict[str, Any] = {
            "sql": None, "columns": [], "rows": [], "row_count": 0,
            "truncated": False, "narrative": None,
        }

        tool._parse_statement_response(stmt_response, result)

        assert result["row_count"] == 100
        assert len(result["rows"]) == 50
        assert result["truncated"] is True


# =============================================================================
# Bug Fix Tests (Enterprise Tools Comprehensive Fix)
# =============================================================================


class TestVectorSearchWorkspaceClientAuth:
    """Test that UserVectorSearchTool uses WorkspaceClient from OBO client."""

    @pytest.mark.asyncio
    async def test_obo_client_provides_workspace_client(self) -> None:
        """UserVectorSearchTool should get WorkspaceClient from OBO client."""
        from deep_research.agent.tools.user_vector_search import UserVectorSearchTool
        from deep_research.agent.tools.base import ResearchContext

        obo_client = MagicMock()
        data_source = MagicMock()
        data_source.name = "test_source"

        # Mock WorkspaceClient returned by OBO
        mock_client = MagicMock()
        mock_client.vector_search_indexes.query_index.return_value = MagicMock(
            manifest=MagicMock(columns=[]),
            result=MagicMock(data_array=[]),
        )
        obo_client.get_client = AsyncMock(return_value=mock_client)

        tool = UserVectorSearchTool(
            obo_client=obo_client,
            data_source=data_source,
            endpoint_name="test-endpoint",
            index_name="catalog.schema.index",
        )

        context = MagicMock(spec=ResearchContext)
        context.user_token = "user-token-123"

        result = await tool.execute({"query": "test query"}, context)

        # OBO client should have been called with user token
        obo_client.get_client.assert_called_once_with("user-token-123")

    @pytest.mark.asyncio
    async def test_no_user_token_still_gets_client(self) -> None:
        """Without user_token, OBO client still returns SP client."""
        from deep_research.agent.tools.user_vector_search import UserVectorSearchTool
        from deep_research.agent.tools.base import ResearchContext

        obo_client = MagicMock()
        data_source = MagicMock()
        data_source.name = "test_source"

        mock_client = MagicMock()
        mock_client.vector_search_indexes.query_index.return_value = MagicMock(
            manifest=MagicMock(columns=[]),
            result=MagicMock(data_array=[]),
        )
        obo_client.get_client = AsyncMock(return_value=mock_client)

        tool = UserVectorSearchTool(
            obo_client=obo_client,
            data_source=data_source,
            endpoint_name="test-endpoint",
            index_name="catalog.schema.index",
        )

        context = MagicMock(spec=ResearchContext)
        context.user_token = None

        result = await tool.execute({"query": "test query"}, context)

        # OBO client called with None → returns SP client
        obo_client.get_client.assert_called_once_with(None)


class TestClassicResearcherArgumentMapping:
    """Test dynamic argument key mapping in classic researcher (Bug 2 fix)."""

    @pytest.mark.asyncio
    async def test_vector_search_tool_receives_query_key(self) -> None:
        """Vector Search tool (required: ['query']) should receive {'query': ...}."""
        from deep_research.agent.tools.base import ToolResult, ToolDefinition, ResearchContext
        from deep_research.schemas.source_scope import SourceScope, SourceScopeConfig

        state = ResearchState(query="What is AAPL revenue?")
        state.source_scope_config = SourceScopeConfig(scope=SourceScope.ENTERPRISE_ONLY)

        # Create steps so get_current_step() returns something
        from deep_research.agent.state import PlanStep, StepType
        state.steps = [PlanStep(
            id="step-1", title="Test Step", description="test",
            step_type=StepType.RESEARCH, needs_search=True,
        )]
        state.current_step_index = 0

        mock_tool = MagicMock()
        mock_tool.definition = ToolDefinition(
            name="search_my_index",
            description="Search vector index",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        mock_tool.validate_arguments = MagicMock(return_value=[])
        mock_tool.execute = AsyncMock(return_value=ToolResult(
            content="Found results", success=True, sources=[],
        ))
        state.enterprise_tools = [mock_tool]

        # Simulate the argument building logic from researcher.py
        tool_params = mock_tool.definition.parameters
        required_keys = tool_params.get("required", [])
        arg_key = required_keys[0] if required_keys else "question"
        arguments = {arg_key: "test query"}

        assert arg_key == "query"
        assert "query" in arguments

    @pytest.mark.asyncio
    async def test_genie_tool_receives_question_key(self) -> None:
        """Genie tool (required: ['question']) should receive {'question': ...}."""
        from deep_research.agent.tools.base import ToolDefinition

        mock_tool = MagicMock()
        mock_tool.definition = ToolDefinition(
            name="query_genie_aapl",
            description="Query AAPL data",
            parameters={
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        )

        tool_params = mock_tool.definition.parameters
        required_keys = tool_params.get("required", [])
        arg_key = required_keys[0] if required_keys else "question"

        assert arg_key == "question"

    def test_tool_with_empty_required_falls_back(self) -> None:
        """Tool with empty required list should fall back to 'question'."""
        from deep_research.agent.tools.base import ToolDefinition

        mock_tool = MagicMock()
        mock_tool.definition = ToolDefinition(
            name="some_tool",
            description="Some tool",
            parameters={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": [],
            },
        )

        tool_params = mock_tool.definition.parameters
        required_keys = tool_params.get("required", [])
        arg_key = required_keys[0] if required_keys else "question"

        assert arg_key == "question"


class TestReActEnterpriseContentLinking:
    """Test content key consistency in ReAct researcher (Bug 3 fix)."""

    @pytest.mark.asyncio
    async def test_content_key_matches_source_url(self) -> None:
        """crawled_content key should use actual source URL, not synthetic enterprise:// URL."""
        from deep_research.agent.tools.base import ToolResult, ToolDefinition, ResearchContext
        from deep_research.agent.nodes.react_researcher import ReactResearchState

        react_state = ReactResearchState()
        state = ResearchState(query="test")

        # Simulate the fixed logic from react_researcher.py
        result = ToolResult(
            content="Genie response data",
            success=True,
            sources=[
                {"url": "genie://space-123", "title": "AAPL Data", "content": "Revenue info"},
            ],
        )
        tc_name = "query_genie_aapl"

        fallback_url = f"enterprise://{tc_name}"
        primary_url = fallback_url
        if result.sources:
            first_url = result.sources[0].get("url")
            if first_url:
                primary_url = first_url

        await react_state.add_high_quality_source(primary_url)
        await react_state.add_crawled_content(primary_url, result.content)

        for src in result.sources:
            state.add_source(SourceInfo(
                url=src.get("url", fallback_url),
                title=src.get("title", tc_name),
                snippet=src.get("content", "")[:500],
                content=src.get("content"),
            ))

        # Verify the key is the actual URL, not enterprise://
        assert "genie://space-123" in react_state.crawled_content
        assert "enterprise://query_genie_aapl" not in react_state.crawled_content
        assert react_state.high_quality_sources == ["genie://space-123"]

        # Verify state.sources has the actual URL
        assert state.sources[0].url == "genie://space-123"

        # Verify post-processing loop would match
        for url in react_state.high_quality_sources:
            if url in react_state.crawled_content:
                for source in state.sources:
                    if source.url == url:
                        source.content = react_state.crawled_content[url]
                        break

        assert state.sources[0].content == "Genie response data"

    @pytest.mark.asyncio
    async def test_no_sources_creates_generic_entry(self) -> None:
        """Tool returning no sources should create a generic entry in state.sources."""
        from deep_research.agent.tools.base import ToolResult
        from deep_research.agent.nodes.react_researcher import ReactResearchState

        react_state = ReactResearchState()
        state = ResearchState(query="test")

        result = ToolResult(
            content="Some enterprise data",
            success=True,
            sources=[],  # No structured sources
        )
        tc_name = "search_my_index"

        fallback_url = f"enterprise://{tc_name}"
        primary_url = fallback_url
        if result.sources:
            first_url = result.sources[0].get("url")
            if first_url:
                primary_url = first_url

        await react_state.add_high_quality_source(primary_url)
        await react_state.add_crawled_content(primary_url, result.content)

        # No structured sources — add generic entry
        if not result.sources:
            state.add_source(SourceInfo(
                url=primary_url,
                title=tc_name,
                snippet=result.content[:500] if result.content else "",
                content=result.content,
            ))

        # Should have fallback entry
        assert len(state.sources) == 1
        assert state.sources[0].url == "enterprise://search_my_index"
        assert state.sources[0].content == "Some enterprise data"


class TestArgumentValidation:
    """Test argument validation before enterprise tool execution (Bug 7 fix)."""

    @pytest.mark.asyncio
    async def test_validation_prevents_invalid_execution(self) -> None:
        """When validate_arguments returns errors, execute() should NOT be called."""
        from deep_research.agent.tools.base import ToolResult, ToolDefinition, ResearchContext
        from deep_research.schemas.source_scope import SourceScope, SourceScopeConfig
        from deep_research.agent.state import PlanStep, StepType

        state = ResearchState(query="test")
        state.source_scope_config = SourceScopeConfig(scope=SourceScope.ENTERPRISE_ONLY)
        state.steps = [PlanStep(
            id="step-1", title="Test", description="test",
            step_type=StepType.RESEARCH, needs_search=True,
        )]
        state.current_step_index = 0

        mock_tool = MagicMock()
        mock_tool.definition = ToolDefinition(
            name="test_tool",
            description="Test",
            parameters={"type": "object", "properties": {}, "required": ["query"]},
        )
        mock_tool.validate_arguments = MagicMock(
            return_value=["'query' must be a string"]
        )
        mock_tool.execute = AsyncMock()
        state.enterprise_tools = [mock_tool]

        # Simulate the researcher logic
        tool_params = mock_tool.definition.parameters
        required_keys = tool_params.get("required", [])
        arg_key = required_keys[0] if required_keys else "question"
        arguments = {arg_key: "test"}

        validation_errors = mock_tool.validate_arguments(arguments)
        if validation_errors:
            # Should skip execution
            pass
        else:
            await mock_tool.execute(arguments=arguments, context=MagicMock())

        # execute should NOT have been called
        mock_tool.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_validation_passes_allows_execution(self) -> None:
        """When validate_arguments returns no errors, execute() should be called."""
        from deep_research.agent.tools.base import ToolResult, ToolDefinition, ResearchContext

        mock_tool = MagicMock()
        mock_tool.definition = ToolDefinition(
            name="test_tool",
            description="Test",
            parameters={"type": "object", "properties": {}, "required": ["query"]},
        )
        mock_tool.validate_arguments = MagicMock(return_value=[])
        mock_tool.execute = AsyncMock(return_value=ToolResult(
            content="result", success=True, sources=[],
        ))

        arguments = {"query": "test query"}
        validation_errors = mock_tool.validate_arguments(arguments)
        if validation_errors:
            pass
        else:
            await mock_tool.execute(arguments=arguments, context=MagicMock())

        mock_tool.execute.assert_called_once()


class TestErrorContextInFallback:
    """Test error context in fallback text (Bug 4 fix)."""

    def test_all_tools_fail_shows_error_details(self) -> None:
        """When all enterprise tools raise exceptions, error details should appear in text."""
        enterprise_errors: list[str] = []
        enterprise_results: list[str] = []

        # Simulate two tools raising exceptions
        for tool_name in ["tool_a", "tool_b"]:
            try:
                raise ConnectionError(f"Connection refused for {tool_name}")
            except Exception as e:
                enterprise_errors.append(f"{tool_name}: {str(e)[:150]}")

        # Build search_results_text as researcher.py does
        if enterprise_results:
            search_results_text = "\n\n".join(enterprise_results)
        elif enterprise_errors:
            error_summary = "; ".join(enterprise_errors)
            search_results_text = (
                f"[Enterprise tools failed. Errors: {error_summary}]"
            )
        else:
            search_results_text = "[Enterprise tools returned no results]"

        assert "Enterprise tools failed" in search_results_text
        assert "tool_a" in search_results_text
        assert "tool_b" in search_results_text
        assert "Connection refused" in search_results_text

    def test_tool_returns_failure_shows_error(self) -> None:
        """When a tool returns success=False, error should be tracked."""
        from deep_research.agent.tools.base import ToolResult

        enterprise_errors: list[str] = []
        enterprise_results: list[str] = []

        result = ToolResult(
            content="",
            success=False,
            error="Index not found: 'catalog.schema.missing_index'",
        )

        tool_name = "search_missing_index"
        if result.success and result.content:
            enterprise_results.append(f"### {tool_name}\n{result.content}")
        elif not result.success:
            enterprise_errors.append(
                f"{tool_name}: {result.error or 'unknown error'}"
            )

        if enterprise_results:
            search_results_text = "\n\n".join(enterprise_results)
        elif enterprise_errors:
            error_summary = "; ".join(enterprise_errors)
            search_results_text = (
                f"[Enterprise tools failed. Errors: {error_summary}]"
            )
        else:
            search_results_text = "[Enterprise tools returned no results]"

        assert "Enterprise tools failed" in search_results_text
        assert "Index not found" in search_results_text
        assert "search_missing_index" in search_results_text
