"""Unit tests for the framework orchestrator (framework_orchestrator.py).

Tests the simple-mode short-circuit, file search tool loading,
and existing sources loading helpers.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from deep_research.agent.framework_orchestrator import (
    _build_state_proxy,
    _extract_verification_from_framework_state,
    _load_enterprise_tools,
    _load_existing_sources,
    _load_file_search_tool,
    _safe_uuid,
    _to_sse_event,
    _to_uuid,
    stream_research_via_framework,
)
from deep_research.schemas.streaming import (
    AgentCompletedEvent,
    AgentStartedEvent,
    ResearchCompletedEvent,
    StreamErrorEvent,
    SynthesisProgressEvent,
    SynthesisStartedEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_config(**overrides: Any) -> MagicMock:
    """Build a MagicMock OrchestrationConfig with sensible defaults."""
    defaults: dict[str, Any] = {
        "query_mode": "deep_research",
        "research_depth": "auto",
        "system_instructions": None,
        "message_id": uuid4(),
        "research_session_id": uuid4(),
        "is_draft": False,
        "session_pre_created": False,
        "verify_sources": True,
        "output_format": "markdown",
        "output_schema": None,
        "synthesis_mode": "simple",
        "enable_post_verification": False,
        "file_ids": None,
        "source_scope": None,
        "enabled_sources": None,
        "disabled_sources": None,
        "user_token": None,
        "model_overrides": None,
        "domain_filter": None,
        "agent_id": None,
        "workflow_ref": None,
        "research_timeout_seconds": 1800,
    }
    defaults.update(overrides)

    config = MagicMock()
    for key, value in defaults.items():
        setattr(config, key, value)
    return config


# ---------------------------------------------------------------------------
# Tests — Simple mode short-circuit
# ---------------------------------------------------------------------------


class TestSimpleModeShortCircuit:
    """Test the CoordinatorClassifiedEvent handling for simple queries."""

    @pytest.mark.asyncio
    async def test_simple_response_yields_synthesis_events(self) -> None:
        """Mock the WorkflowExecutor to emit a CoordinatorClassifiedEvent with
        is_simple=True, verify that synthesis and agent events are yielded."""
        from databricks_deep_research.events.types import CoordinatorClassifiedEvent

        # Build a mock CoordinatorClassifiedEvent
        classified_evt = CoordinatorClassifiedEvent(
            node_id="coordinator",
            timestamp="2025-01-01T00:00:00Z",
            complexity="simple",
            recommended_depth="none",
            is_simple=True,
            direct_response="Hello world",
        )

        # Create an async generator that yields the classified event
        async def _mock_execute(state: Any) -> Any:
            yield classified_evt

        mock_executor_instance = MagicMock()
        mock_executor_instance.execute = _mock_execute

        config = _mock_config(query_mode="simple")

        with (
            patch(
                "deep_research.agent.framework_orchestrator.WorkflowExecutor",
                return_value=mock_executor_instance,
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_llm_client",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_file_search_tool",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_enterprise_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_existing_sources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator.translate",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.ExecutionContext",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.DomainContextTracker",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.safe_mlflow_run",
            ),
            patch(
                "deep_research.agent.framework_orchestrator.safe_tool_span",
            ),
            patch(
                "deep_research.agent.framework_orchestrator.safe_update_trace",
            ),
            patch(
                "deep_research.agent.framework_orchestrator._persist_simple_response",
                new_callable=AsyncMock,
            ),
        ):
            events: list[Any] = []
            async for evt in stream_research_via_framework(
                query="Hello",
                llm=MagicMock(),
                brave_client=MagicMock(),
                crawler=MagicMock(),
                config=config,
            ):
                events.append(evt)

            # Check that we got the expected event types
            event_types = [type(e) for e in events]
            assert SynthesisStartedEvent in event_types
            assert SynthesisProgressEvent in event_types
            assert AgentStartedEvent in event_types
            assert AgentCompletedEvent in event_types
            assert ResearchCompletedEvent in event_types

    @pytest.mark.asyncio
    async def test_simple_response_sets_final_report(self) -> None:
        """Verify final_report in ResearchCompletedEvent matches the direct_response."""
        from databricks_deep_research.events.types import CoordinatorClassifiedEvent

        direct_text = "The answer is 42"
        classified_evt = CoordinatorClassifiedEvent(
            node_id="coordinator",
            timestamp="2025-01-01T00:00:00Z",
            complexity="simple",
            recommended_depth="none",
            is_simple=True,
            direct_response=direct_text,
        )

        async def _mock_execute(state: Any) -> Any:
            yield classified_evt

        mock_executor_instance = MagicMock()
        mock_executor_instance.execute = _mock_execute

        config = _mock_config(query_mode="simple")

        with (
            patch(
                "deep_research.agent.framework_orchestrator.WorkflowExecutor",
                return_value=mock_executor_instance,
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_llm_client",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_file_search_tool",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_enterprise_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_existing_sources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator.translate",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.ExecutionContext",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.DomainContextTracker",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.safe_mlflow_run",
            ),
            patch(
                "deep_research.agent.framework_orchestrator.safe_tool_span",
            ),
            patch(
                "deep_research.agent.framework_orchestrator.safe_update_trace",
            ),
            patch(
                "deep_research.agent.framework_orchestrator._persist_simple_response",
                new_callable=AsyncMock,
            ),
        ):
            events: list[Any] = []
            async for evt in stream_research_via_framework(
                query="What is the meaning of life?",
                llm=MagicMock(),
                brave_client=MagicMock(),
                crawler=MagicMock(),
                config=config,
            ):
                events.append(evt)

            # Find the ResearchCompletedEvent
            completed = [e for e in events if isinstance(e, ResearchCompletedEvent)]
            assert len(completed) == 1
            assert completed[0].final_report == direct_text


# ---------------------------------------------------------------------------
# Tests — File search tool loading
# ---------------------------------------------------------------------------


class TestFileSearchLoading:
    """Test _load_file_search_tool helper."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_file_ids(self) -> None:
        """Config without file_ids, no db -> returns None."""
        config = _mock_config(file_ids=None)
        result = await _load_file_search_tool(config, db=None, user_id=None, chat_id=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_creates_tool_from_explicit_file_ids(self) -> None:
        """Config with file_ids, mock db + user_id -> returns FileSearchTool (mock)."""
        config = _mock_config(file_ids=["file-1", "file-2"])
        mock_db = MagicMock()
        mock_tool = MagicMock()

        with patch(
            "deep_research.agent.tools.file_search.create_file_search_tool",
            return_value=mock_tool,
        ) as mock_create:
            result = await _load_file_search_tool(
                config, db=mock_db, user_id="user-123", chat_id="chat-456",
            )

        assert result is mock_tool
        mock_create.assert_called_once_with(
            session=mock_db,
            owner_id="user-123",
            file_ids=["file-1", "file-2"],
        )

    @pytest.mark.asyncio
    async def test_auto_discovers_files(self) -> None:
        """No file_ids but chat_id present -> auto-discovers via FileUploadService."""
        config = _mock_config(file_ids=None)
        mock_db = MagicMock()
        chat_id = str(uuid4())

        # Create mock files with is_ready=True
        mock_file = MagicMock()
        mock_file.is_ready = True
        mock_file.id = uuid4()

        mock_service = AsyncMock()
        mock_service.get_session_files = AsyncMock(return_value=([mock_file], None))

        mock_tool = MagicMock()

        with (
            patch(
                "deep_research.agent.framework_orchestrator.make_file_upload_service",
                return_value=mock_service,
            ) as mock_factory,
            patch(
                "deep_research.agent.tools.file_search.create_file_search_tool",
                return_value=mock_tool,
            ) as mock_create,
        ):
            result = await _load_file_search_tool(
                config, db=mock_db, user_id="user-123", chat_id=chat_id,
            )

        assert result is mock_tool
        assert mock_factory.call_count == 1
        mock_create.assert_called_once_with(
            session=mock_db,
            owner_id="user-123",
            file_ids=[str(mock_file.id)],
        )


# ---------------------------------------------------------------------------
# Tests — Existing sources loading
# ---------------------------------------------------------------------------


class TestExistingSourcesLoading:
    """Test _load_existing_sources helper."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_db(self) -> None:
        """Both storage_stack=None and db=None -> returns []."""
        result = await _load_existing_sources(
            storage_stack=None, db=None, chat_id="some-id",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_chat_id(self) -> None:
        """chat_id=None -> returns []."""
        result = await _load_existing_sources(
            storage_stack=None, db=MagicMock(), chat_id=None,
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_loads_sources_from_service(self) -> None:
        """Legacy ORM path: mock db.execute -> returns list of source dicts."""
        chat_id = str(uuid4())

        # Create mock source objects
        mock_source_1 = SimpleNamespace(
            url="https://example.com/1",
            title="Source 1",
            snippet="Snippet 1",
            content="Content 1",
        )
        mock_source_2 = SimpleNamespace(
            url="https://example.com/2",
            title="Source 2",
            snippet=None,
            content=None,
        )

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_source_1, mock_source_2]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await _load_existing_sources(
            storage_stack=None, db=mock_db, chat_id=chat_id,
        )

        assert len(result) == 2
        assert result[0]["url"] == "https://example.com/1"
        assert result[0]["title"] == "Source 1"
        assert result[0]["snippet"] == "Snippet 1"
        assert result[0]["content"] == "Content 1"
        assert result[1]["url"] == "https://example.com/2"
        assert result[1]["title"] == "Source 2"
        assert result[1]["snippet"] is None
        assert result[1]["content"] is None

    @pytest.mark.asyncio
    async def test_loads_sources_from_cached_stack(self) -> None:
        """F-SOURCES: cached path reads doc.state.sources from storage_stack."""
        chat_id = str(uuid4())
        mock_source_high = SimpleNamespace(
            url="https://example.com/high",
            title="High",
            metadata={
                "snippet": "hs",
                "content": "hc",
                "relevance_score": 0.9,
            },
        )
        mock_source_low = SimpleNamespace(
            url="https://example.com/low",
            title="Low",
            metadata={
                "snippet": "ls",
                "content": "lc",
                "relevance_score": 0.1,
            },
        )
        mock_state = SimpleNamespace(sources=[mock_source_low, mock_source_high])
        mock_doc = SimpleNamespace(state=mock_state)
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=mock_doc)
        mock_stack = SimpleNamespace(cache=mock_cache)

        result = await _load_existing_sources(
            storage_stack=mock_stack, db=None, chat_id=chat_id,
        )

        assert len(result) == 2
        # High relevance_score first.
        assert result[0]["url"] == "https://example.com/high"
        assert result[0]["snippet"] == "hs"
        assert result[0]["content"] == "hc"
        assert result[1]["url"] == "https://example.com/low"


# ---------------------------------------------------------------------------
# Tests — Enterprise tool loading
# ---------------------------------------------------------------------------


class TestEnterpriseToolLoading:
    """Test framework enterprise tool loading fallbacks."""

    @pytest.mark.asyncio
    async def test_loads_db_tools_when_scope_defaults_to_all(self) -> None:
        """Default scope should still load DB-backed enterprise tools."""
        config = _mock_config(source_scope=None, enabled_sources=None, disabled_sources=None)
        mock_tool = MagicMock()
        mock_tool.definition.name = "search_finance_docs"

        with patch(
            "deep_research.agent.tools.factory.get_enabled_tools_for_user",
            new_callable=AsyncMock,
            return_value=[mock_tool],
        ) as mock_get_tools:
            result = await _load_enterprise_tools(
                config,
                db=MagicMock(),
                user_id="user-123",
                chat_id=None,
            )

        assert result == [mock_tool]
        mock_get_tools.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_loads_selected_sources_from_discovery_cache(self) -> None:
        """Selected enterprise source IDs should resolve via discovery cache."""
        selected_source_id = "vs:main.finance_docs"
        config = _mock_config(
            source_scope="enterprise_only",
            enabled_sources=[selected_source_id],
            disabled_sources=[],
        )
        mock_db = MagicMock()
        mock_discovered_source = SimpleNamespace(source_id=selected_source_id)
        mock_tool = MagicMock()
        mock_tool.definition.name = "search_main_finance_docs"
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=[mock_discovered_source])
        mock_service = MagicMock()
        mock_service.get_accessible_sources = AsyncMock(return_value=([], None))

        with (
            patch(
                "deep_research.services.data_source_service.DataSourceService",
                return_value=mock_service,
            ),
            patch(
                "deep_research.agent.tools.factory.create_tools_from_user_sources",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_create_user_tools,
            patch(
                "deep_research.services.discovery_cache.get_discovery_cache",
                return_value=mock_cache,
            ),
            patch(
                "deep_research.agent.tools.factory.create_tools_from_discovered_sources",
                new_callable=AsyncMock,
                return_value=[mock_tool],
            ) as mock_create_discovery_tools,
            patch(
                "deep_research.agent.tools.factory.create_tools_from_source_ids",
                return_value=[],
            ) as mock_create_source_id_tools,
        ):
            result = await _load_enterprise_tools(
                config,
                db=mock_db,
                user_id="user-123",
                chat_id=None,
            )

        assert result == [mock_tool]
        mock_create_user_tools.assert_not_awaited()
        mock_cache.get.assert_awaited_once_with(user_id="user-123")
        mock_create_discovery_tools.assert_awaited_once_with([mock_discovered_source])
        mock_create_source_id_tools.assert_not_called()

    @pytest.mark.asyncio
    async def test_loads_selected_sources_directly_when_cache_misses(self) -> None:
        """Selected enterprise source IDs should fall back to direct construction."""
        selected_source_id = "vs:main.finance_docs"
        config = _mock_config(
            source_scope="enterprise_only",
            enabled_sources=[selected_source_id],
            disabled_sources=[],
        )
        mock_db = MagicMock()
        mock_tool = MagicMock()
        mock_tool.definition.name = "search_main_finance_docs"
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_service = MagicMock()
        mock_service.get_accessible_sources = AsyncMock(return_value=([], None))

        with (
            patch(
                "deep_research.services.data_source_service.DataSourceService",
                return_value=mock_service,
            ),
            patch(
                "deep_research.agent.tools.factory.create_tools_from_user_sources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.services.discovery_cache.get_discovery_cache",
                return_value=mock_cache,
            ),
            patch(
                "deep_research.agent.tools.factory.create_tools_from_discovered_sources",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_create_discovery_tools,
            patch(
                "deep_research.agent.tools.factory.create_tools_from_source_ids",
                return_value=[mock_tool],
            ) as mock_create_source_id_tools,
        ):
            result = await _load_enterprise_tools(
                config,
                db=mock_db,
                user_id="user-123",
                chat_id=None,
            )

        assert result == [mock_tool]
        mock_create_discovery_tools.assert_not_awaited()
        mock_create_source_id_tools.assert_called_once_with([selected_source_id])


# ---------------------------------------------------------------------------
# Tests — WorkflowExecutor instantiation regression
# ---------------------------------------------------------------------------


class TestExecutorInstantiation:
    """Regression tests for correct WorkflowExecutor and WorkflowState usage."""

    @pytest.mark.asyncio
    async def test_executor_receives_correct_args(self) -> None:
        """Verify WorkflowExecutor is instantiated with (definition, llm, ...)
        not (context) — regression for the Bug A fix."""
        from databricks_deep_research.events.types import CoordinatorClassifiedEvent

        classified_evt = CoordinatorClassifiedEvent(
            node_id="coordinator",
            timestamp="2025-01-01T00:00:00Z",
            complexity="simple",
            recommended_depth="none",
            is_simple=True,
            direct_response="Hello",
        )

        async def _mock_execute(state: Any) -> Any:
            yield classified_evt

        mock_executor_cls = MagicMock()
        mock_executor_instance = MagicMock()
        mock_executor_instance.execute = _mock_execute
        mock_executor_cls.return_value = mock_executor_instance

        mock_workflow_def = MagicMock(name="workflow_def")
        mock_llm = MagicMock(name="framework_llm")

        config = _mock_config(query_mode="simple")

        with (
            patch(
                "deep_research.agent.framework_orchestrator.WorkflowExecutor",
                mock_executor_cls,
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_llm_client",
                return_value=mock_llm,
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_file_search_tool",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_enterprise_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_existing_sources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator.translate",
                return_value=mock_workflow_def,
            ),
            patch(
                "deep_research.agent.framework_orchestrator.ExecutionContext",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.DomainContextTracker",
                return_value=MagicMock(),
            ),
            patch("deep_research.agent.framework_orchestrator.safe_mlflow_run"),
            patch("deep_research.agent.framework_orchestrator.safe_tool_span"),
            patch("deep_research.agent.framework_orchestrator.safe_update_trace"),
            patch(
                "deep_research.agent.framework_orchestrator._persist_simple_response",
                new_callable=AsyncMock,
            ),
        ):
            events: list[Any] = []
            async for evt in stream_research_via_framework(
                query="test",
                llm=MagicMock(),
                brave_client=MagicMock(),
                crawler=MagicMock(),
                config=config,
            ):
                events.append(evt)

            # Verify WorkflowExecutor was called with definition and llm_client
            mock_executor_cls.assert_called_once()
            args, kwargs = mock_executor_cls.call_args
            assert args[0] is mock_workflow_def, "1st arg should be workflow_def"
            assert args[1] is mock_llm, "2nd arg should be framework_llm"
            assert "tool_resolver" in kwargs, "should pass pre-populated tool_resolver"
            assert "context" in kwargs

    @pytest.mark.asyncio
    async def test_executor_receives_workflow_state(self) -> None:
        """Verify executor.execute() receives a WorkflowState, not a WorkflowDefinition."""
        from databricks_deep_research.events.types import CoordinatorClassifiedEvent
        from databricks_deep_research.workflow.state import WorkflowState

        classified_evt = CoordinatorClassifiedEvent(
            node_id="coordinator",
            timestamp="2025-01-01T00:00:00Z",
            complexity="simple",
            recommended_depth="none",
            is_simple=True,
            direct_response="Hello",
        )

        captured_state: dict[str, Any] = {}

        async def _mock_execute(state: Any) -> Any:
            captured_state["state"] = state
            yield classified_evt

        mock_executor_instance = MagicMock()
        mock_executor_instance.execute = _mock_execute

        config = _mock_config(query_mode="simple")

        with (
            patch(
                "deep_research.agent.framework_orchestrator.WorkflowExecutor",
                return_value=mock_executor_instance,
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_llm_client",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_file_search_tool",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_enterprise_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_existing_sources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator.translate",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.ExecutionContext",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.DomainContextTracker",
                return_value=MagicMock(),
            ),
            patch("deep_research.agent.framework_orchestrator.safe_mlflow_run"),
            patch("deep_research.agent.framework_orchestrator.safe_tool_span"),
            patch("deep_research.agent.framework_orchestrator.safe_update_trace"),
            patch(
                "deep_research.agent.framework_orchestrator._persist_simple_response",
                new_callable=AsyncMock,
            ),
        ):
            events: list[Any] = []
            async for evt in stream_research_via_framework(
                query="What is quantum computing?",
                llm=MagicMock(),
                brave_client=MagicMock(),
                crawler=MagicMock(),
                config=config,
            ):
                events.append(evt)

            state = captured_state["state"]
            assert isinstance(state, WorkflowState)
            assert state.query == "What is quantum computing?"


# ---------------------------------------------------------------------------
# Tests — _safe_uuid helper
# ---------------------------------------------------------------------------


class TestSafeUuid:
    """Tests for the _safe_uuid helper."""

    def test_valid_uuid_string_preserved(self) -> None:
        u = uuid4()
        assert _safe_uuid(str(u)) == u

    def test_uuid_object_passthrough(self) -> None:
        u = uuid4()
        assert _safe_uuid(u) is u

    def test_non_uuid_string_returns_valid_uuid(self) -> None:
        result = _safe_uuid("research_cycle_planner_c0_iter1")
        assert isinstance(result, UUID)

    def test_non_uuid_string_is_deterministic(self) -> None:
        a = _safe_uuid("research_cycle_planner_c0_iter1")
        b = _safe_uuid("research_cycle_planner_c0_iter1")
        assert a == b

    def test_different_strings_produce_different_uuids(self) -> None:
        a = _safe_uuid("plan_iter1")
        b = _safe_uuid("plan_iter2")
        assert a != b

    def test_none_fallback(self) -> None:
        """_safe_uuid(uuid4()) when data.get returns None — the OR pattern."""
        result = _safe_uuid(uuid4())
        assert isinstance(result, UUID)


# ---------------------------------------------------------------------------
# Tests — Event roundtrip (framework event → tracker → _to_sse_event → app model)
# ---------------------------------------------------------------------------

_TS = "2026-01-01T00:00:00Z"


class TestEventRoundtrip:
    """Full-chain regression: fw event → real tracker → _to_sse_event() → valid app model.

    These tests catch type mismatches (str→UUID, dict→Pydantic) that
    isolated unit tests miss because they mock one side of the boundary.
    """

    def test_plan_created_non_uuid_plan_id(self) -> None:
        """The exact crash from production: string plan_id must become UUID."""
        from databricks_deep_research.events.types import (
            PlanCreatedEvent as FwPlanCreatedEvent,
        )

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        fw = FwPlanCreatedEvent(
            node_id="research_cycle_planner_c0",
            timestamp=_TS,
            plan_id="research_cycle_planner_c0_iter1",
            title="Plan",
            thought="Thinking",
            steps=[
                {
                    "id": "s1",
                    "title": "Step 1",
                    "step_type": "research",
                    "needs_search": True,
                }
            ],
            iteration=1,
            has_enough_context=False,
        )
        [app_evt] = tracker.process_event(fw)
        sse = _to_sse_event(app_evt)
        assert sse is not None
        assert isinstance(sse.plan_id, UUID)

    def test_plan_created_valid_uuid_preserved(self) -> None:
        from databricks_deep_research.events.types import (
            PlanCreatedEvent as FwPlanCreatedEvent,
        )

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        uid = str(uuid4())
        tracker = DomainContextTracker()
        fw = FwPlanCreatedEvent(
            node_id="planner",
            timestamp=_TS,
            plan_id=uid,
            title="Plan",
            thought="T",
            steps=[],
            iteration=1,
        )
        [app_evt] = tracker.process_event(fw)
        sse = _to_sse_event(app_evt)
        assert sse is not None
        assert str(sse.plan_id) == uid

    def test_plan_created_steps_with_missing_fields(self) -> None:
        """LLM omits needs_search — must not crash, uses defaults."""
        from databricks_deep_research.events.types import (
            PlanCreatedEvent as FwPlanCreatedEvent,
        )

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        fw = FwPlanCreatedEvent(
            node_id="planner",
            timestamp=_TS,
            plan_id="plan_iter1",
            title="Plan",
            thought="T",
            steps=[{"title": "Incomplete step"}],  # Missing id, step_type, needs_search
            iteration=1,
        )
        [app_evt] = tracker.process_event(fw)
        sse = _to_sse_event(app_evt)
        assert sse is not None
        assert len(sse.steps) == 1
        assert sse.steps[0].id == "step-0"  # Default
        assert sse.steps[0].needs_search is True  # Default

    def test_plan_created_extra_fields_in_steps_ignored(self) -> None:
        """Framework steps include description, source_hints — silently dropped."""
        from databricks_deep_research.events.types import (
            PlanCreatedEvent as FwPlanCreatedEvent,
        )

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        fw = FwPlanCreatedEvent(
            node_id="planner",
            timestamp=_TS,
            plan_id="plan_iter1",
            title="Plan",
            thought="T",
            steps=[
                {
                    "id": "s1",
                    "title": "Step",
                    "step_type": "research",
                    "needs_search": True,
                    "description": "Details",
                    "source_hints": [],
                }
            ],
            iteration=1,
        )
        [app_evt] = tracker.process_event(fw)
        sse = _to_sse_event(app_evt)
        assert sse is not None
        assert len(sse.steps) == 1

    def test_coordinator_roundtrip(self) -> None:
        from databricks_deep_research.events.types import CoordinatorClassifiedEvent

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        fw = CoordinatorClassifiedEvent(
            node_id="coordinator",
            timestamp=_TS,
            complexity="complex",
            recommended_depth="extended",
            is_simple=False,
            direct_response=None,
        )
        [app_evt] = tracker.process_event(fw)
        sse = _to_sse_event(app_evt)
        assert sse is not None

    def test_background_roundtrip(self) -> None:
        from databricks_deep_research.events.types import BackgroundCompletedEvent

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        fw = BackgroundCompletedEvent(
            node_id="bg",
            timestamp=_TS,
            sources_discovered=3,
            data_landscape_summary="Summary",
            data_landscape={},
            query_decomposition=["q1"],
        )
        [app_evt] = tracker.process_event(fw)
        sse = _to_sse_event(app_evt)
        assert sse is not None

    def test_item_started_roundtrip(self) -> None:
        from databricks_deep_research.events.types import ItemStartedEvent

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        fw = ItemStartedEvent(
            node_id="researcher",
            timestamp=_TS,
            item_index=0,
            item_summary="Search",
            total_items=3,
        )
        [app_evt] = tracker.process_event(fw)
        sse = _to_sse_event(app_evt)
        assert sse is not None
        assert sse.event_type == "step_started"

    def test_item_completed_roundtrip(self) -> None:
        from databricks_deep_research.events.types import ItemCompletedEvent

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        fw = ItemCompletedEvent(
            node_id="researcher",
            timestamp=_TS,
            item_index=1,
            items_processed=2,
        )
        [app_evt] = tracker.process_event(fw)
        sse = _to_sse_event(app_evt)
        assert sse is not None
        assert sse.event_type == "step_completed"
        assert sse.sources_found == 0  # Default when no sources tracked

    def test_item_completed_roundtrip_with_sources(self) -> None:
        """sources_found from ToolResultEvent flows through to StepCompletedEvent."""
        from databricks_deep_research.events.types import (
            ItemCompletedEvent,
            ToolResultEvent,
        )

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        # Simulate tool result before item completed
        tracker.process_event(ToolResultEvent(
            node_id="research_cycle", timestamp=_TS,
            tool_name="web_search", result_summary="Found results",
            source_count=8,
        ))
        fw = ItemCompletedEvent(
            node_id="research_cycle", timestamp=_TS,
            item_index=0, items_processed=1,
        )
        [app_evt] = tracker.process_event(fw)
        sse = _to_sse_event(app_evt)
        assert sse is not None
        assert sse.sources_found == 8

    def test_reflection_roundtrip(self) -> None:
        from databricks_deep_research.events.types import ReflectionDecisionEvent

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        fw = ReflectionDecisionEvent(
            node_id="reflector",
            timestamp=_TS,
            decision="continue",
            reasoning="Need more data",
        )
        [app_evt] = tracker.process_event(fw)
        sse = _to_sse_event(app_evt)
        assert sse is not None
        assert sse.event_type == "reflection_decision"

    def test_node_error_roundtrip(self) -> None:
        from databricks_deep_research.events.types import NodeErrorEvent

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        fw = NodeErrorEvent(
            node_id="researcher",
            timestamp=_TS,
            error_message="Rate limit",
            will_retry=True,
            retry_attempt=1,
        )
        [app_evt] = tracker.process_event(fw)
        sse = _to_sse_event(app_evt)
        assert sse is not None
        assert sse.event_type == "error"

    def test_agent_output_returns_none(self) -> None:
        """agent_output is handled via final_report tracking, not SSE."""
        from databricks_deep_research.events.types import AgentOutputEvent

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        fw = AgentOutputEvent(
            node_id="synth",
            timestamp=_TS,
            output_key="report",
            output_preview="Report...",
        )
        [app_evt] = tracker.process_event(fw)
        sse = _to_sse_event(app_evt)
        assert sse is None

    def test_workflow_completed_returns_none(self) -> None:
        """workflow_completed handled by ResearchCompletedEvent at end."""
        from databricks_deep_research.events.types import WorkflowCompletedEvent

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        fw = WorkflowCompletedEvent(
            node_id="main",
            timestamp=_TS,
            workflow_id="wf-1",
            duration_ms=1000.0,
            total_tokens=100,
        )
        [app_evt] = tracker.process_event(fw)
        sse = _to_sse_event(app_evt)
        assert sse is None

    def test_synthesis_started_roundtrip(self) -> None:
        from databricks_deep_research.events.types import (
            SynthesisStartedEvent as FwSynthesisStartedEvent,
        )

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        fw = FwSynthesisStartedEvent(
            node_id="synth",
            timestamp=_TS,
            total_observations=5,
            total_sources=3,
        )
        [app_evt] = tracker.process_event(fw)
        sse = _to_sse_event(app_evt)
        assert sse is not None
        assert sse.event_type == "synthesis_started"


# ---------------------------------------------------------------------------
# Tests — _build_state_proxy contract
# ---------------------------------------------------------------------------


class TestBuildStateProxy:
    """Contract test: proxy must expose every attribute persistence reads.

    This test class is the safety net against the fundamental fragility of
    duck-typing a SimpleNamespace as ResearchState.  If persistence.py adds
    a new ``state.X`` access, add ``X`` to the appropriate list below.

    Cross-ref: _build_state_proxy() docstring, persistence.py:89-748.
    """

    # persist_research_session_complete_update_independent (L690-718)
    _COMPLETE_UPDATE_ATTRS = [
        "final_report", "all_observations", "current_plan",
        "reflection_history", "current_step_index", "plan_iterations",
    ]

    # persist_complete_research (L367-451) — superset of above
    _COMPLETE_RESEARCH_ATTRS = _COMPLETE_UPDATE_ATTRS + ["query_mode"]

    # persist_research_data (L89-158) — called by both above
    _RESEARCH_DATA_ATTRS = ["sources", "claims", "verification_summary"]

    # Union of all — the full contract
    _ALL_PERSISTENCE_ATTRS = list(set(
        _COMPLETE_RESEARCH_ATTRS + _RESEARCH_DATA_ATTRS
    ))

    def test_has_all_persistence_attrs(self) -> None:
        """Every attribute accessed by persistence must exist on proxy."""
        config = _mock_config()
        proxy = _build_state_proxy(config, "report")
        for attr in self._ALL_PERSISTENCE_ATTRS:
            assert hasattr(proxy, attr), (
                f"Missing '{attr}' on state proxy — persistence.py reads state.{attr} "
                f"but _build_state_proxy() doesn't set it. Add it to the proxy."
            )

    def test_reflection_history_is_iterable_and_falsy(self) -> None:
        """L711: `[r.to_dict() for r in state.reflection_history] if state.reflection_history`
        Empty list must be falsy (skip iteration) and iterable (safe if loop runs)."""
        proxy = _build_state_proxy(_mock_config(), "report")
        assert not proxy.reflection_history  # falsy
        assert list(proxy.reflection_history) == []  # iterable

    def test_claims_is_iterable_and_supports_len(self) -> None:
        """L112: `len(state.claims)` and L120: `for claim in state.claims:`"""
        proxy = _build_state_proxy(_mock_config(), "report")
        assert len(proxy.claims) == 0
        assert list(proxy.claims) == []

    def test_verification_summary_is_falsy(self) -> None:
        """L249: `if not state.claims and not state.verification_summary: return None`
        L266: `state.verification_summary.to_dict() if state.verification_summary`
        Must be falsy so .to_dict() is never called."""
        proxy = _build_state_proxy(_mock_config(), "report")
        assert not proxy.verification_summary

    def test_current_plan_is_falsy(self) -> None:
        """L430: `state.current_plan.to_dict() if state.current_plan else None`
        Must be falsy so .to_dict() is never called."""
        proxy = _build_state_proxy(_mock_config(), "report")
        assert not proxy.current_plan

    def test_query_mode_from_config(self) -> None:
        """L427: `query_mode=state.query_mode` — must match config."""
        proxy = _build_state_proxy(_mock_config(query_mode="web_search"), "report")
        assert proxy.query_mode == "web_search"

    def test_final_report_passthrough(self) -> None:
        proxy = _build_state_proxy(_mock_config(), "My research report")
        assert proxy.final_report == "My research report"

    def test_sources_is_empty_list(self) -> None:
        """L183: `for source_info in state.sources:` — must be iterable."""
        proxy = _build_state_proxy(_mock_config(), "report")
        assert proxy.sources == []

    def test_sources_are_loaded_from_framework_sources_pool(self) -> None:
        """Framework sources pool should feed final persistence state."""
        from databricks_deep_research.pools.pool_state import PoolConfig, PoolState

        sources_pool = PoolState(PoolConfig(name="sources", dedup_content_hash=False))
        sources_pool.add({
            "url": "vs://main.finance_docs/doc-1",
            "title": "Quarterly Earnings",
            "snippet": "Revenue grew 10 percent year over year.",
            "content": "Revenue grew 10 percent year over year.",
            "source_type": "vector_search",
            "relevance_score": 0.91,
        })
        wf_state = SimpleNamespace(pools={"sources": sources_pool})

        proxy = _build_state_proxy(_mock_config(), "report", wf_state)

        assert len(proxy.sources) == 1
        source = proxy.sources[0]
        assert source.url == "vs://main.finance_docs/doc-1"
        assert source.title == "Quarterly Earnings"
        assert source.snippet == "Revenue grew 10 percent year over year."
        assert source.content == "Revenue grew 10 percent year over year."
        assert source.source_type == "vector_search"
        assert source.relevance_score == 0.91


class TestFrameworkVerificationExtraction:
    def test_prefers_framework_state_claims_and_summary(self) -> None:
        from databricks_deep_research.workflow.state import WorkflowState

        wf_state = WorkflowState(query="kroger earnings")
        wf_state.append(
            "synth",
            "claims",
            [
                {
                    "claim_text": "Kroger reported identical sales growth of 2.6%.",
                    "claim_type": "numeric",
                    "position_start": 0,
                    "position_end": 48,
                    "evidence": {
                        "source_url": "enterprise://vector_search/earnings/0",
                        "quote_text": "Kroger reported identical sales growth of 2.6%.",
                    },
                    "verification_verdict": "supported",
                    "citation_key": "0",
                    "citation_keys": ["0"],
                }
            ],
        )
        wf_state.append(
            "synth",
            "verification_summary",
            {
                "total_claims": 1,
                "verified_claims": 1,
                "corrected_citations": 0,
                "removed_claims": 0,
                "softened_claims": 0,
                "overall_confidence": 1.0,
            },
        )

        claims, summary = _extract_verification_from_framework_state(
            wf_state,
            [
                {
                    "url": "enterprise://vector_search/earnings/0",
                    "snippet": "Kroger reported identical sales growth of 2.6%.",
                }
            ],
        )

        assert len(claims) == 1
        assert claims[0].claim_text.startswith("Kroger reported")
        assert claims[0].evidence is not None
        assert claims[0].evidence.source_url == "enterprise://vector_search/earnings/0"
        assert summary is not None
        assert summary.total_claims == 1
        assert summary.supported_count == 1


# ---------------------------------------------------------------------------
# Tests — BUG-17: _to_uuid helper
# ---------------------------------------------------------------------------


class TestToUuid:
    """Tests for the _to_uuid helper."""

    def test_string_conversion(self) -> None:
        u = uuid4()
        assert _to_uuid(str(u)) == u

    def test_uuid_passthrough(self) -> None:
        u = uuid4()
        assert _to_uuid(u) is u


# ---------------------------------------------------------------------------
# Tests — BUG-7: fail fast on empty model mapping
# ---------------------------------------------------------------------------


class TestBuildModelMapping:
    """Tests for _build_model_mapping in llm_adapter."""

    def test_empty_mapping_raises(self) -> None:
        """All tiers fail → ValueError('LLM_ADAPTER_NO_TIERS')."""
        from deep_research.agent.adapters.llm_adapter import _build_model_mapping

        mock_llm = MagicMock()
        mock_config = MagicMock()
        # Make get_role always return None for every tier
        mock_config.get_role.return_value = None
        mock_llm._config = mock_config

        with pytest.raises(ValueError, match="LLM_ADAPTER_NO_TIERS"):
            _build_model_mapping(mock_llm, overrides=None)

    def test_overrides_prevent_error(self) -> None:
        """Overrides prevent error even if config has no tiers."""
        from deep_research.agent.adapters.llm_adapter import _build_model_mapping

        mock_llm = MagicMock()
        mock_config = MagicMock()
        mock_config.get_role.return_value = None
        mock_llm._config = mock_config

        mapping = _build_model_mapping(
            mock_llm,
            overrides={"analytical": "my-model"},
        )
        assert "analytical" in mapping
        assert mapping["analytical"] == "my-model"


# ---------------------------------------------------------------------------
# Tests — BUG-1: _persist_delta warning
# ---------------------------------------------------------------------------


class TestPersistDelta:
    """Tests for the _persist_delta no-op warning."""

    @pytest.mark.asyncio
    async def test_dirty_delta_emits_warning(self, caplog: Any) -> None:
        """Dirty delta → WARNING log with FWK_INCREMENTAL_PERSIST_NOT_IMPLEMENTED."""
        from deep_research.agent.adapters.domain_context import PersistenceDelta
        from deep_research.agent.framework_orchestrator import _persist_delta

        delta = PersistenceDelta()
        delta.new_sources = [{"url": "https://example.com"}]
        delta._dirty = True

        config = _mock_config()
        import logging
        with caplog.at_level(logging.WARNING, logger="deep_research.agent.framework_orchestrator"):
            await _persist_delta(delta, config, db=None, chat_id=None, user_id=None)

        assert "FWK_INCREMENTAL_PERSIST_NOT_IMPLEMENTED" in caplog.text

    @pytest.mark.asyncio
    async def test_clean_delta_no_warning(self, caplog: Any) -> None:
        """Clean delta → no warning."""
        from deep_research.agent.adapters.domain_context import PersistenceDelta
        from deep_research.agent.framework_orchestrator import _persist_delta

        delta = PersistenceDelta()
        delta._dirty = False

        config = _mock_config()
        import logging
        with caplog.at_level(logging.WARNING, logger="deep_research.agent.framework_orchestrator"):
            await _persist_delta(delta, config, db=None, chat_id=None, user_id=None)

        assert "FWK_INCREMENTAL_PERSIST_NOT_IMPLEMENTED" not in caplog.text


# ---------------------------------------------------------------------------
# Tests — BUG-4: isinstance-based progress tracking
# ---------------------------------------------------------------------------


class TestProgressTracking:
    """Tests for isinstance-based progress tracking in the executor loop."""

    def test_item_completed_isinstance(self) -> None:
        """ItemCompletedEvent is correctly detected via isinstance."""
        from databricks_deep_research.events.types import ItemCompletedEvent

        evt = ItemCompletedEvent(
            node_id="researcher",
            timestamp="2026-01-01T00:00:00Z",
            item_index=1,
            items_processed=2,
        )
        assert isinstance(evt, ItemCompletedEvent)

    def test_replan_triggered_isinstance(self) -> None:
        """ReplanTriggeredEvent is correctly detected via isinstance."""
        from databricks_deep_research.events.types import ReplanTriggeredEvent

        evt = ReplanTriggeredEvent(
            node_id="reflector",
            timestamp="2026-01-01T00:00:00Z",
            cycle=1,
            reason="Insufficient coverage",
            items_remaining=3,
        )
        assert isinstance(evt, ReplanTriggeredEvent)


# ---------------------------------------------------------------------------
# Tests — BUG-11: unhandled event logging + new handlers
# ---------------------------------------------------------------------------


class TestUnhandledEventLogging:
    """Tests for framework event handling in domain_context."""

    def test_replan_triggered_produces_progress(self) -> None:
        """ReplanTriggeredEvent → research_progress with 'replan_triggered'."""
        from databricks_deep_research.events.types import ReplanTriggeredEvent

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        evt = ReplanTriggeredEvent(
            node_id="reflector",
            timestamp="2026-01-01T00:00:00Z",
            cycle=2,
            reason="Need more data",
            items_remaining=5,
        )
        results = tracker.process_event(evt)
        assert len(results) == 1
        assert results[0].event_type == "research_progress"
        assert results[0].data["progress_type"] == "replan_triggered"

    def test_tool_call_produces_progress_with_tool_name(self) -> None:
        """ToolCallEvent → research_progress with correct tool_name."""
        from databricks_deep_research.events.types import ToolCallEvent

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        evt = ToolCallEvent(
            node_id="researcher",
            timestamp="2026-01-01T00:00:00Z",
            tool_name="web_search",
            arguments={"query": "test"},
        )
        results = tracker.process_event(evt)
        assert len(results) == 1
        assert results[0].data["tool_name"] == "web_search"

    def test_unknown_event_returns_empty_with_debug_log(self, caplog: Any) -> None:
        """Truly unknown event type → returns [] + DEBUG log."""
        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        # Create a mock event that's not in _HANDLERS
        mock_event = MagicMock()
        mock_event.node_id = "test"
        # Use a type not registered in _HANDLERS
        type(mock_event).__name__ = "SomeUnknownEvent"

        import logging
        with caplog.at_level(logging.DEBUG, logger="deep_research.agent.adapters.domain_context"):
            results = tracker.process_event(mock_event)

        assert results == []
        assert "FWK_EVENT_UNHANDLED" in caplog.text

    def test_noop_handler_returns_empty(self) -> None:
        """No-op handler (e.g., WorkflowStartedEvent) → returns []."""
        from databricks_deep_research.events.types import WorkflowStartedEvent

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        evt = WorkflowStartedEvent(
            node_id="main",
            timestamp="2026-01-01T00:00:00Z",
            workflow_id="wf-1",
            workflow_name="test",
        )
        results = tracker.process_event(evt)
        assert results == []


# ---------------------------------------------------------------------------
# Tests — BUG-5: EnterpriseToolAdapter calling convention
# ---------------------------------------------------------------------------


class TestEnterpriseToolAdapter:
    """Tests for EnterpriseToolAdapter.execute() calling convention."""

    @pytest.mark.asyncio
    async def test_execute_calls_with_arguments_dict_and_context(self) -> None:
        """Verify execute called with (arguments=dict, context=ResearchContext), NOT **kwargs."""
        from deep_research.agent.adapters.tool_adapter import EnterpriseToolAdapter

        mock_tool = AsyncMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.parameters = {"type": "object", "properties": {}}
        mock_tool.execute = AsyncMock(
            return_value=SimpleNamespace(content="result", success=True),
        )

        adapter = EnterpriseToolAdapter(
            app_tool=mock_tool,
            user_token="tok-123",
            chat_id=str(uuid4()),
            user_id="user-1",
        )

        await adapter.execute({"query": "test"})

        mock_tool.execute.assert_called_once()
        call_kwargs = mock_tool.execute.call_args.kwargs
        assert "arguments" in call_kwargs
        assert call_kwargs["arguments"] == {"query": "test"}
        assert "context" in call_kwargs

    @pytest.mark.asyncio
    async def test_context_has_user_token(self) -> None:
        """Verify context.user_token is set correctly from adapter."""
        from deep_research.agent.adapters.tool_adapter import EnterpriseToolAdapter

        captured_context: dict[str, Any] = {}

        async def _capture_execute(*, arguments: dict, context: Any) -> Any:
            captured_context["ctx"] = context
            return SimpleNamespace(content="ok", success=True)

        mock_tool = MagicMock()
        mock_tool.name = "genie"
        mock_tool.description = "Genie tool"
        mock_tool.parameters = {"type": "object", "properties": {}}
        mock_tool.execute = _capture_execute

        adapter = EnterpriseToolAdapter(
            app_tool=mock_tool,
            user_token="obo-token-xyz",
            chat_id=str(uuid4()),
            user_id="user-1",
        )

        await adapter.execute({"space_id": "123"})

        ctx = captured_context["ctx"]
        assert ctx.user_token == "obo-token-xyz"
        assert ctx.user_id == "user-1"


# ---------------------------------------------------------------------------
# Tests — BUG-2/3: final_report truncation fix
# ---------------------------------------------------------------------------


class TestFinalReportNotTruncated:
    """Tests for final report not being truncated to 200 chars."""

    def test_workflow_completed_extracts_full_report(self) -> None:
        """WorkflowCompletedEvent with 5000-char report → delta.final_report is 5000 chars."""
        from databricks_deep_research.events.types import WorkflowCompletedEvent

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        full_report = "A" * 5000

        evt = WorkflowCompletedEvent(
            node_id="main",
            timestamp="2026-01-01T00:00:00Z",
            workflow_id="wf-1",
            duration_ms=1000.0,
            total_tokens=100,
            final_report=full_report,
        )
        tracker.process_event(evt)

        delta = tracker.get_persistence_delta()
        assert delta.final_report == full_report
        assert len(delta.final_report) == 5000

    def test_workflow_completed_overwrites_truncated_agent_output(self) -> None:
        """AgentOutputEvent (200 chars) then WorkflowCompletedEvent (5000 chars) → full report."""
        from databricks_deep_research.events.types import (
            AgentOutputEvent,
            WorkflowCompletedEvent,
        )

        from deep_research.agent.adapters.domain_context import DomainContextTracker

        tracker = DomainContextTracker()
        truncated = "B" * 200
        full_report = "C" * 5000

        # First: truncated AgentOutputEvent
        tracker.process_event(AgentOutputEvent(
            node_id="synth",
            timestamp="2026-01-01T00:00:00Z",
            output_key="report",
            output_preview=truncated,
        ))

        # Then: full WorkflowCompletedEvent
        tracker.process_event(WorkflowCompletedEvent(
            node_id="main",
            timestamp="2026-01-01T00:00:01Z",
            workflow_id="wf-1",
            duration_ms=2000.0,
            total_tokens=200,
            final_report=full_report,
        ))

        delta = tracker.get_persistence_delta()
        assert delta.final_report == full_report
        assert len(delta.final_report) == 5000


# ---------------------------------------------------------------------------
# Tests — BUG-16: research timeout enforcement
# ---------------------------------------------------------------------------


class TestResearchTimeout:
    """Test asyncio.timeout enforcement around the executor loop."""

    @pytest.mark.asyncio
    async def test_slow_executor_yields_timeout_error(self) -> None:
        """Slow executor (sleeps 100s) + 1s timeout → StreamErrorEvent(RESEARCH_TIMEOUT)."""

        async def _slow_execute(state: Any) -> Any:
            await asyncio.sleep(100)
            yield  # never reached

        mock_executor_instance = MagicMock()
        mock_executor_instance.execute = _slow_execute

        config = _mock_config(research_timeout_seconds=1)

        with (
            patch(
                "deep_research.agent.framework_orchestrator.WorkflowExecutor",
                return_value=mock_executor_instance,
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_llm_client",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_file_search_tool",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_enterprise_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_existing_sources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator.translate",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.ExecutionContext",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.DomainContextTracker",
                return_value=MagicMock(
                    process_event=MagicMock(return_value=[]),
                    should_persist=MagicMock(return_value=False),
                    get_persistence_delta=MagicMock(
                        return_value=MagicMock(final_report=None, _dirty=False),
                    ),
                ),
            ),
            patch("deep_research.agent.framework_orchestrator.safe_mlflow_run"),
            patch("deep_research.agent.framework_orchestrator.safe_tool_span"),
            patch("deep_research.agent.framework_orchestrator.safe_update_trace"),
        ):
            events: list[Any] = []
            async for evt in stream_research_via_framework(
                query="test timeout",
                llm=MagicMock(),
                brave_client=MagicMock(),
                crawler=MagicMock(),
                config=config,
            ):
                events.append(evt)

            # Find the StreamErrorEvent
            error_events = [
                e for e in events
                if isinstance(e, StreamErrorEvent) and e.error_code == "RESEARCH_TIMEOUT"
            ]
            assert len(error_events) == 1
            assert "timed out" in error_events[0].error_message


# ---------------------------------------------------------------------------
# Tests — Tool Registration (Bug 1 fix: builtin vs external)
# ---------------------------------------------------------------------------


class TestToolRegistration:
    """Tests that all framework tools are registered as overrides in the ToolResolver.

    Bug 1: config_translator emits string tool refs, but tools were
    registered as "external" via WorkflowExecutor(enterprise_tools=). This
    caused resolution to fail.

    Fix: framework_orchestrator creates a ToolResolver, registers all tools
    as overrides, and passes tool_resolver= (not enterprise_tools=) to executor.
    """

    @pytest.mark.asyncio
    async def test_all_tools_registered_as_builtin(self) -> None:
        """Verify ToolResolver.override called for each tool."""
        from databricks_deep_research.tools.resolver import ToolResolver

        # Create mock tools with definition.name
        mock_tool_1 = MagicMock()
        mock_tool_1.definition.name = "web_search"
        mock_tool_2 = MagicMock()
        mock_tool_2.definition.name = "web_crawl"

        captured: dict[str, Any] = {}

        def _capture_executor(*args: Any, **kwargs: Any) -> MagicMock:
            captured["tool_resolver"] = kwargs.get("tool_resolver")
            captured["enterprise_tools"] = kwargs.get("enterprise_tools")
            mock_exec = MagicMock()

            async def _empty_execute(state: Any) -> Any:
                return
                yield  # noqa: F841 - makes it an async generator

            mock_exec.execute = _empty_execute
            return mock_exec

        config = _mock_config()

        with (
            patch(
                "deep_research.agent.framework_orchestrator.WorkflowExecutor",
                side_effect=_capture_executor,
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_llm_client",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_tools",
                new_callable=AsyncMock,
                return_value=[mock_tool_1, mock_tool_2],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_file_search_tool",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_enterprise_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_existing_sources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator.translate",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.ExecutionContext",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.DomainContextTracker",
                return_value=MagicMock(
                    process_event=MagicMock(return_value=[]),
                    should_persist=MagicMock(return_value=False),
                    get_persistence_delta=MagicMock(
                        return_value=MagicMock(final_report=None, _dirty=False),
                    ),
                ),
            ),
            patch("deep_research.agent.framework_orchestrator.safe_mlflow_run"),
            patch("deep_research.agent.framework_orchestrator.safe_tool_span"),
            patch("deep_research.agent.framework_orchestrator.safe_update_trace"),
        ):
            events: list[Any] = []
            async for evt in stream_research_via_framework(
                query="test tools",
                llm=MagicMock(),
                brave_client=MagicMock(),
                crawler=MagicMock(),
                config=config,
            ):
                events.append(evt)

            # Verify tool_resolver was passed (not enterprise_tools for resolution)
            resolver = captured.get("tool_resolver")
            assert resolver is not None
            assert isinstance(resolver, ToolResolver)

            # Verify both tools are registered as overrides
            assert resolver._overrides.get("web_search") is mock_tool_1
            assert resolver._overrides.get("web_crawl") is mock_tool_2

            # Verify enterprise_tools kwarg was NOT passed
            assert captured.get("enterprise_tools") is None

    @pytest.mark.asyncio
    async def test_executor_receives_pre_populated_resolver(self) -> None:
        """Verify WorkflowExecutor receives tool_resolver= with tools already overridden."""
        from databricks_deep_research.tools.resolver import ToolResolver

        mock_tool = MagicMock()
        mock_tool.definition.name = "web_search"

        captured: dict[str, Any] = {}

        def _capture_executor(*args: Any, **kwargs: Any) -> MagicMock:
            captured["tool_resolver"] = kwargs.get("tool_resolver")
            mock_exec = MagicMock()

            async def _empty_execute(state: Any) -> Any:
                return
                yield

            mock_exec.execute = _empty_execute
            return mock_exec

        config = _mock_config()

        with (
            patch(
                "deep_research.agent.framework_orchestrator.WorkflowExecutor",
                side_effect=_capture_executor,
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_llm_client",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_tools",
                new_callable=AsyncMock,
                return_value=[mock_tool],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_file_search_tool",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_enterprise_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_existing_sources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator.translate",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.ExecutionContext",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.DomainContextTracker",
                return_value=MagicMock(
                    process_event=MagicMock(return_value=[]),
                    should_persist=MagicMock(return_value=False),
                    get_persistence_delta=MagicMock(
                        return_value=MagicMock(final_report=None, _dirty=False),
                    ),
                ),
            ),
            patch("deep_research.agent.framework_orchestrator.safe_mlflow_run"),
            patch("deep_research.agent.framework_orchestrator.safe_tool_span"),
            patch("deep_research.agent.framework_orchestrator.safe_update_trace"),
        ):
            async for _ in stream_research_via_framework(
                query="test resolver",
                llm=MagicMock(),
                brave_client=MagicMock(),
                crawler=MagicMock(),
                config=config,
            ):
                pass

            # Verify resolution works via override
            resolver = captured["tool_resolver"]
            assert isinstance(resolver, ToolResolver)
            resolved = await resolver.resolve("web_search")
            assert resolved is mock_tool


# ---------------------------------------------------------------------------
# Tests — Final report capture from WorkflowCompletedEvent
# ---------------------------------------------------------------------------


class TestFinalReportCapture:
    """Regression test for the bug where periodic delta reset discards final_report.

    Root cause: get_persistence_delta() resets the delta (destructive read).
    When WorkflowCompletedEvent lands at an event count divisible by 5,
    the periodic-persist check fires, calls get_persistence_delta(), and
    the subsequent post-loop read gets an empty delta with final_report=None.

    Fix: Direct capture from FwkWorkflowCompletedEvent in the event loop,
    BEFORE the periodic-persist check. Also, carry final_report across
    delta resets in DomainContextTracker.

    IMPORTANT: These tests use the REAL DomainContextTracker (not mocked)
    to exercise the delta reset path end-to-end.
    """

    @pytest.mark.asyncio
    async def test_report_captured_at_persist_boundary(self) -> None:
        """Emit exactly 4 filler events + WorkflowCompletedEvent (total=5,
        hits %5 boundary). Assert ResearchCompletedEvent.final_report has
        the full report despite the periodic delta reset."""
        from databricks_deep_research.events.types import (
            ItemCompletedEvent as FwkItemCompletedEvent,
        )
        from databricks_deep_research.events.types import (
            WorkflowCompletedEvent as FwkWorkflowCompletedEvent,
        )

        full_report = "# Research Report\n\n" + "Content. " * 500  # >200 chars

        async def _mock_execute(state: Any) -> Any:
            # Emit 4 filler events to reach count=4
            for i in range(4):
                yield FwkItemCompletedEvent(
                    node_id="researcher",
                    timestamp="2026-01-01T00:00:00Z",
                    item_index=i,
                    items_processed=i + 1,
                )
            # Event #5 — hits the %5 boundary
            yield FwkWorkflowCompletedEvent(
                node_id="main",
                timestamp="2026-01-01T00:00:01Z",
                workflow_id="wf-1",
                duration_ms=30000.0,
                total_tokens=50000,
                final_report=full_report,
            )

        mock_executor_instance = MagicMock()
        mock_executor_instance.execute = _mock_execute

        config = _mock_config()

        with (
            patch(
                "deep_research.agent.framework_orchestrator.WorkflowExecutor",
                return_value=mock_executor_instance,
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_llm_client",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_file_search_tool",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_enterprise_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_existing_sources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator.translate",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.ExecutionContext",
                return_value=MagicMock(),
            ),
            # Do NOT mock DomainContextTracker — use the real one
            patch("deep_research.agent.framework_orchestrator.safe_mlflow_run"),
            patch("deep_research.agent.framework_orchestrator.safe_tool_span"),
            patch("deep_research.agent.framework_orchestrator.safe_update_trace"),
        ):
            events: list[Any] = []
            async for evt in stream_research_via_framework(
                query="test persist boundary",
                llm=MagicMock(),
                brave_client=MagicMock(),
                crawler=MagicMock(),
                config=config,
            ):
                events.append(evt)

            # Find the ResearchCompletedEvent
            completed = [e for e in events if isinstance(e, ResearchCompletedEvent)]
            assert len(completed) == 1
            assert completed[0].final_report == full_report
            assert len(completed[0].final_report) > 200  # Not truncated

    @pytest.mark.asyncio
    async def test_report_captured_off_boundary(self) -> None:
        """Emit 1 filler event + WorkflowCompletedEvent (total=2, no %5
        boundary). Control test — report should still be captured."""
        from databricks_deep_research.events.types import (
            ItemCompletedEvent as FwkItemCompletedEvent,
        )
        from databricks_deep_research.events.types import (
            WorkflowCompletedEvent as FwkWorkflowCompletedEvent,
        )

        full_report = "# Off-boundary Report\n\n" + "Data. " * 500

        async def _mock_execute(state: Any) -> Any:
            yield FwkItemCompletedEvent(
                node_id="researcher",
                timestamp="2026-01-01T00:00:00Z",
                item_index=0,
                items_processed=1,
            )
            yield FwkWorkflowCompletedEvent(
                node_id="main",
                timestamp="2026-01-01T00:00:01Z",
                workflow_id="wf-1",
                duration_ms=15000.0,
                total_tokens=25000,
                final_report=full_report,
            )

        mock_executor_instance = MagicMock()
        mock_executor_instance.execute = _mock_execute

        config = _mock_config()

        with (
            patch(
                "deep_research.agent.framework_orchestrator.WorkflowExecutor",
                return_value=mock_executor_instance,
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_llm_client",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.create_framework_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_file_search_tool",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_enterprise_tools",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator._load_existing_sources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "deep_research.agent.framework_orchestrator.translate",
                return_value=MagicMock(),
            ),
            patch(
                "deep_research.agent.framework_orchestrator.ExecutionContext",
                return_value=MagicMock(),
            ),
            # Do NOT mock DomainContextTracker — use the real one
            patch("deep_research.agent.framework_orchestrator.safe_mlflow_run"),
            patch("deep_research.agent.framework_orchestrator.safe_tool_span"),
            patch("deep_research.agent.framework_orchestrator.safe_update_trace"),
        ):
            events: list[Any] = []
            async for evt in stream_research_via_framework(
                query="test off boundary",
                llm=MagicMock(),
                brave_client=MagicMock(),
                crawler=MagicMock(),
                config=config,
            ):
                events.append(evt)

            completed = [e for e in events if isinstance(e, ResearchCompletedEvent)]
            assert len(completed) == 1
            assert completed[0].final_report == full_report


# ---------------------------------------------------------------------------
# Tests — research_progress event forwarding in _to_sse_event
# ---------------------------------------------------------------------------


class TestResearchProgressForwarding:
    """Test that _to_sse_event correctly forwards research_progress sub-events."""

    def test_tool_call_event_forwarded(self) -> None:
        """research_progress with progress_type=tool_call -> ToolCallEvent."""
        from deep_research.agent.adapters.domain_context import AppSSEEvent
        from deep_research.schemas.streaming import ToolCallEvent

        app_evt = AppSSEEvent(
            event_type="research_progress",
            data={
                "progress_type": "tool_call",
                "tool_name": "web_search",
                "node_id": "researcher",
            },
        )
        result = _to_sse_event(app_evt)
        assert result is not None
        assert isinstance(result, ToolCallEvent)
        assert result.tool_name == "web_search"
        assert result.event_type == "tool_call"

    def test_tool_result_event_forwarded(self) -> None:
        """research_progress with progress_type=tool_result -> ToolResultEvent."""
        from deep_research.agent.adapters.domain_context import AppSSEEvent
        from deep_research.schemas.streaming import ToolResultEvent

        app_evt = AppSSEEvent(
            event_type="research_progress",
            data={
                "progress_type": "tool_result",
                "tool_name": "web_crawl",
                "result_summary": "Found 5 results",
                "source_count": 5,
            },
        )
        result = _to_sse_event(app_evt)
        assert result is not None
        assert isinstance(result, ToolResultEvent)
        assert result.tool_name == "web_crawl"
        assert result.sources_crawled == 5
        assert result.result_preview == "Found 5 results"

    def test_tool_result_source_count_mapped_to_sources_crawled(self) -> None:
        """Verify source_count from framework maps to sources_crawled in app event."""
        from deep_research.agent.adapters.domain_context import AppSSEEvent
        from deep_research.schemas.streaming import ToolResultEvent

        app_evt = AppSSEEvent(
            event_type="research_progress",
            data={
                "progress_type": "tool_result",
                "tool_name": "web_search",
                "result_summary": "",
                "source_count": 12,
            },
        )
        result = _to_sse_event(app_evt)
        assert isinstance(result, ToolResultEvent)
        assert result.sources_crawled == 12

    def test_claim_verified_forwarded(self) -> None:
        """research_progress with progress_type=claim_verified -> ClaimVerifiedEvent."""
        from deep_research.agent.adapters.domain_context import AppSSEEvent
        from deep_research.schemas.streaming import ClaimVerifiedEvent

        app_evt = AppSSEEvent(
            event_type="research_progress",
            data={
                "progress_type": "claim_verified",
                "claim_index": 0,
                "verdict": "supported",
                "confidence": "high",
                "evidence_snippet": "Test evidence",
                "claim_text": "AI is transformative",
            },
        )
        result = _to_sse_event(app_evt)
        assert result is not None
        assert isinstance(result, ClaimVerifiedEvent)
        assert result.verdict == "supported"
        assert result.confidence_level == "high"
        assert result.claim_text == "AI is transformative"

    def test_claim_verified_float_confidence(self) -> None:
        """Float confidence should be converted to string level."""
        from deep_research.agent.adapters.domain_context import AppSSEEvent
        from deep_research.schemas.streaming import ClaimVerifiedEvent

        app_evt = AppSSEEvent(
            event_type="research_progress",
            data={
                "progress_type": "claim_verified",
                "claim_index": 0,
                "verdict": "supported",
                "confidence": 0.92,
                "evidence_snippet": "Test",
            },
        )
        result = _to_sse_event(app_evt)
        assert result is not None
        assert isinstance(result, ClaimVerifiedEvent)
        assert result.confidence_level == "high"  # 0.92 >= 0.8 → "high"

    def test_claim_verified_low_float_confidence(self) -> None:
        """Low float confidence should map to 'low'."""
        from deep_research.agent.adapters.domain_context import AppSSEEvent
        from deep_research.schemas.streaming import ClaimVerifiedEvent

        app_evt = AppSSEEvent(
            event_type="research_progress",
            data={
                "progress_type": "claim_verified",
                "claim_index": 1,
                "verdict": "unsupported",
                "confidence": 0.3,
                "evidence_snippet": "",
            },
        )
        result = _to_sse_event(app_evt)
        assert result is not None
        assert isinstance(result, ClaimVerifiedEvent)
        assert result.confidence_level == "low"  # 0.3 <= 0.4 → "low"

    def test_verification_summary_forwarded(self) -> None:
        """research_progress with progress_type=verification_summary."""
        from deep_research.agent.adapters.domain_context import AppSSEEvent
        from deep_research.schemas.streaming import VerificationSummaryEvent

        app_evt = AppSSEEvent(
            event_type="research_progress",
            data={
                "progress_type": "verification_summary",
                "total_claims": 10,
                "verified_claims": 8,
                "corrected_citations": 1,
                "removed_claims": 1,
                "softened_claims": 0,
                "overall_confidence": 0.85,
                "warning": True,
            },
        )
        result = _to_sse_event(app_evt)
        assert result is not None
        assert isinstance(result, VerificationSummaryEvent)
        assert result.total_claims == 10
        assert result.supported == 8
        assert result.unsupported == 0        # softened_claims=0
        assert result.contradicted == 1       # removed_claims=1
        assert result.warning is True

    def test_citation_corrected_forwarded(self) -> None:
        """research_progress with progress_type=citation_corrected."""
        from deep_research.agent.adapters.domain_context import AppSSEEvent
        from deep_research.schemas.streaming import CitationCorrectedEvent

        app_evt = AppSSEEvent(
            event_type="research_progress",
            data={
                "progress_type": "citation_corrected",
                "claim_index": 2,
                "action": "replace",
                "original_key": "Arxiv",
                "corrected_key": "Github",
            },
        )
        result = _to_sse_event(app_evt)
        assert result is not None
        assert isinstance(result, CitationCorrectedEvent)
        assert result.correction_type == "replace"

    def test_replan_triggered_returns_none(self) -> None:
        """research_progress with progress_type=replan_triggered -> None."""
        from deep_research.agent.adapters.domain_context import AppSSEEvent

        app_evt = AppSSEEvent(
            event_type="research_progress",
            data={
                "progress_type": "replan_triggered",
                "cycle": 1,
                "reason": "insufficient coverage",
                "items_remaining": 3,
            },
        )
        result = _to_sse_event(app_evt)
        assert result is None

    def test_unknown_progress_type_returns_none(self) -> None:
        """research_progress with unknown progress_type -> None."""
        from deep_research.agent.adapters.domain_context import AppSSEEvent

        app_evt = AppSSEEvent(
            event_type="research_progress",
            data={"progress_type": "unknown_type"},
        )
        result = _to_sse_event(app_evt)
        assert result is None


# ---------------------------------------------------------------------------
# Tests — PR4 inline fix: failure-persistence logging (lines 902, 1715)
# ---------------------------------------------------------------------------


class TestFailurePersistenceLogging:
    """Regression tests for PR4 inline fix.

    Before PR4, lines 902 and 1715 contained ``except Exception: pass`` which
    silently dropped failures of ``persist_research_session_failed_independent``.
    DB rows would stay in 'running' forever after a primary error, with NO log
    line emitted (silent data loss).

    PR4 replaced both ``pass`` statements with ``logger.exception(...)`` so the
    failure path is observable. These tests assert the structured log line is
    emitted when the failure-persist call itself raises.
    """

    @pytest.mark.asyncio
    async def test_persist_completion_inner_failure_emits_log(
        self, caplog: Any
    ) -> None:
        """When persist_research_session_complete_update_independent AND the
        inner persist_research_session_failed_independent both raise, the
        outer except logs ``FWK_PERSISTENCE_FAILED`` (existing) AND the inner
        recovery emits ``FWK_FAILURE_PERSISTENCE_FAILED`` (PR4 fix at line 1715).
        """
        from deep_research.agent.framework_orchestrator import _persist_completion

        config = _mock_config()
        chat_id_uuid = uuid4()
        event_buffer = MagicMock()  # truthy → two-phase path

        async def _raise_complete(**_kwargs: Any) -> None:
            raise RuntimeError("boom-complete")

        async def _raise_failed(**_kwargs: Any) -> None:
            raise RuntimeError("boom-failed")

        import logging

        with patch(
            "deep_research.agent.persistence.persist_research_session_complete_update_independent",
            side_effect=_raise_complete,
        ), patch(
            "deep_research.agent.persistence.persist_research_session_failed_independent",
            side_effect=_raise_failed,
        ), caplog.at_level(
            logging.WARNING, logger="deep_research.agent.framework_orchestrator"
        ):
            result = await _persist_completion(
                config,
                chat_id_uuid,
                user_id="alice",
                query="q",
                final_report="report",
                event_buffer=event_buffer,
                wf_state=None,
                claims=None,
                verification_summary=None,
                storage_stack=None,
            )

        # Returns None on persistence failure (existing contract).
        assert result is None
        # Outer except line 1701 — pre-existing log line.
        assert "FWK_PERSISTENCE_FAILED" in caplog.text
        # PR4 fix: inner except line 1715 now also logs instead of silent pass.
        assert "FWK_FAILURE_PERSISTENCE_FAILED" in caplog.text

    @pytest.mark.asyncio
    async def test_persist_completion_inner_succeeds_no_failure_log(
        self, caplog: Any
    ) -> None:
        """If persist_research_session_failed_independent succeeds, the inner
        except is never entered and FWK_FAILURE_PERSISTENCE_FAILED is NOT
        emitted (no false-positive log noise on the happy failure path).
        """
        from deep_research.agent.framework_orchestrator import _persist_completion

        config = _mock_config()
        chat_id_uuid = uuid4()
        event_buffer = MagicMock()

        async def _raise_complete(**_kwargs: Any) -> None:
            raise RuntimeError("boom-complete")

        async def _ok_failed(**_kwargs: Any) -> None:
            return None

        import logging

        with patch(
            "deep_research.agent.persistence.persist_research_session_complete_update_independent",
            side_effect=_raise_complete,
        ), patch(
            "deep_research.agent.persistence.persist_research_session_failed_independent",
            side_effect=_ok_failed,
        ), caplog.at_level(
            logging.WARNING, logger="deep_research.agent.framework_orchestrator"
        ):
            await _persist_completion(
                config,
                chat_id_uuid,
                user_id="alice",
                query="q",
                final_report="report",
                event_buffer=event_buffer,
                wf_state=None,
                claims=None,
                verification_summary=None,
                storage_stack=None,
            )

        # Outer except still fires (the primary failure).
        assert "FWK_PERSISTENCE_FAILED" in caplog.text
        # Inner except is NOT entered → PR4 log line is absent.
        assert "FWK_FAILURE_PERSISTENCE_FAILED" not in caplog.text
