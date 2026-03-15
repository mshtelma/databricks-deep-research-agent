"""Unit tests for orchestrator shared helper functions (C1 fix)."""

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from deep_research.agent.framework_orchestrator import (
    _extract_verification_from_report,
)
from deep_research.agent.orchestrator import (
    OrchestrationConfig,
    _create_research_state,
    _handle_reflection_complete,
    stream_research,
    _wire_config_fields,
    _wire_manual_mode,
    _wire_source_scope,
)
from deep_research.agent.state import (
    Plan,
    PlanStep,
    ReflectionDecision,
    ReflectionResult,
    ResearchState,
    StepStatus,
    StepType,
)
from deep_research.schemas.streaming import ResearchCompletedEvent


# =============================================================================
# _create_research_state tests
# =============================================================================


class TestCreateResearchState:
    """Tests for _create_research_state helper."""

    def test_basic_state_creation(self) -> None:
        """State is created with all config fields transferred."""
        config = OrchestrationConfig(
            query_mode="deep_research",
            research_depth="extended",
            enable_clarification=False,
            verify_sources=True,
            output_format="json",
            synthesis_mode="reclaim",
            enable_post_verification=True,
        )
        state = _create_research_state("test query", config)

        assert state.query == "test query"
        assert state.query_mode == "deep_research"
        assert state.research_depth == "extended"
        assert state.enable_clarification is False
        assert state.enable_citation_verification is True
        assert state.output_format == "json"
        assert state.synthesis_mode == "reclaim"
        assert state.enable_post_verification is True
        assert state.conversation_history == []

    def test_workflow_mode_transferred(self) -> None:
        """BUG FIX: workflow_mode and manual_steps must be transferred."""
        config = OrchestrationConfig(
            workflow_mode="manual",
            manual_steps=[{"id": "s1", "title": "Step 1"}],
        )
        state = _create_research_state("query", config)

        assert state.workflow_mode == "manual"
        assert state.manual_steps == [{"id": "s1", "title": "Step 1"}]

    def test_manual_steps_defaults_to_empty_list(self) -> None:
        """When manual_steps is None in config, state gets empty list."""
        config = OrchestrationConfig(manual_steps=None)
        state = _create_research_state("query", config)

        assert state.manual_steps == []

    def test_session_id_set_when_provided(self) -> None:
        """Session ID is set when provided."""
        sid = uuid4()
        config = OrchestrationConfig()
        state = _create_research_state("query", config, session_id=sid)

        assert state.session_id == sid

    def test_session_id_generated_when_not_provided(self) -> None:
        """Session ID is auto-generated when not provided."""
        config = OrchestrationConfig()
        state = _create_research_state("query", config)

        assert isinstance(state.session_id, UUID)

    def test_conversation_history_passed_through(self) -> None:
        """Conversation history is passed to state."""
        history = [{"role": "user", "content": "hello"}]
        config = OrchestrationConfig()
        state = _create_research_state("query", config, conversation_history=history)

        assert state.conversation_history == history


# =============================================================================
# _wire_source_scope tests
# =============================================================================


class TestWireSourceScope:
    """Tests for _wire_source_scope helper."""

    def test_noop_when_no_scope(self) -> None:
        """Does nothing when config has no source_scope."""
        config = OrchestrationConfig(source_scope=None)
        state = _create_research_state("query", config)

        _wire_source_scope(state, config)

        assert state.source_scope_config is None

    def test_valid_scope_creates_config(self) -> None:
        """Valid scope string creates SourceScopeConfig on state."""
        config = OrchestrationConfig(
            source_scope="enterprise_only",
            enabled_sources=["src1", "src2"],
            disabled_sources=["src3"],
        )
        state = _create_research_state("query", config)

        _wire_source_scope(state, config)

        assert state.source_scope_config is not None
        scope = state.source_scope_config.scope
        assert (scope.value if hasattr(scope, "value") else scope) == "enterprise_only"
        assert state.source_scope_config.enabled_sources == ["src1", "src2"]
        assert state.source_scope_config.disabled_sources == ["src3"]

    def test_invalid_scope_defaults_to_all(self) -> None:
        """Invalid scope string defaults to ALL."""
        config = OrchestrationConfig(source_scope="invalid_scope_value")
        state = _create_research_state("query", config)

        _wire_source_scope(state, config)

        assert state.source_scope_config is not None
        scope = state.source_scope_config.scope
        assert (scope.value if hasattr(scope, "value") else scope) == "all"


# =============================================================================
# _wire_config_fields tests
# =============================================================================


class TestWireConfigFields:
    """Tests for _wire_config_fields helper."""

    def test_transfers_all_fields(self) -> None:
        """All simple config fields are transferred to state."""
        config = OrchestrationConfig(
            user_token="tok-123",
            file_ids=["f1", "f2"],
            agent_id="agent-1",
            model_overrides={"analytical": "custom-endpoint"},
            domain_filter=MagicMock(),
        )
        state = _create_research_state("query", config)

        _wire_config_fields(state, config)

        assert state.user_token == "tok-123"
        assert state.file_ids == ["f1", "f2"]
        assert state.agent_id == "agent-1"
        assert state.model_overrides == {"analytical": "custom-endpoint"}
        assert state.domain_filter is not None

    def test_skips_none_fields(self) -> None:
        """None/empty fields are not transferred."""
        config = OrchestrationConfig()
        state = _create_research_state("query", config)

        _wire_config_fields(state, config)

        assert state.user_token is None
        assert state.file_ids == []
        assert state.agent_id is None
        assert state.model_overrides is None
        assert state.domain_filter is None


# =============================================================================
# _wire_manual_mode tests
# =============================================================================


class TestWireManualMode:
    """Tests for _wire_manual_mode helper."""

    def test_noop_for_planner_mode(self) -> None:
        """Does nothing when workflow is planner mode."""
        state = ResearchState(query="query", workflow_mode="planner")
        _wire_manual_mode(state, "query")
        assert state.current_plan is None

    def test_noop_when_no_manual_steps(self) -> None:
        """Does nothing when manual mode but no steps defined."""
        state = ResearchState(query="query", workflow_mode="manual", manual_steps=[])
        _wire_manual_mode(state, "query")
        assert state.current_plan is None

    def test_creates_plan_from_manual_steps(self) -> None:
        """Creates plan from manual step definitions."""

        @dataclass
        class FakeStep:
            id: str
            title: str
            objective: str
            sources: list = None  # type: ignore[assignment]
            constraints: object | None = None

            def __post_init__(self) -> None:
                if self.sources is None:
                    self.sources = []

        steps = [
            FakeStep(id="s1", title="Search web", objective="Find info"),
            FakeStep(id="s2", title="Analyze", objective="Analyze data"),
        ]
        state = ResearchState(
            query="test query",
            workflow_mode="manual",
            manual_steps=steps,
        )

        _wire_manual_mode(state, "test query")

        assert state.current_plan is not None
        assert len(state.current_plan.steps) == 2
        assert state.current_plan.steps[0].title == "Search web"
        assert state.current_plan.steps[1].title == "Analyze"


class TestFrameworkDelegation:
    """Ensure the production orchestrator path delegates to the framework runtime."""

    @pytest.mark.asyncio
    async def test_stream_research_uses_framework_path(self) -> None:
        """stream_research should yield framework events and skip legacy setup."""
        config = OrchestrationConfig()
        terminal_event = ResearchCompletedEvent(
            session_id=uuid4(),
            total_steps_executed=1,
            total_steps_skipped=0,
            plan_iterations=1,
            total_duration_ms=1,
            final_report="done",
        )

        async def _mock_framework_stream(*args, **kwargs):  # type: ignore[no-untyped-def]
            yield terminal_event

        with (
            patch(
                "deep_research.agent.framework_orchestrator.stream_research_via_framework",
                side_effect=_mock_framework_stream,
            ) as framework_stream,
            patch(
                "deep_research.agent.orchestrator._create_research_state",
                new_callable=MagicMock,
            ) as create_state,
        ):
            events = [
                event
                async for event in stream_research(
                    query="framework only",
                    llm=MagicMock(),
                    brave_client=MagicMock(),
                    crawler=MagicMock(),
                    config=config,
                )
            ]

        assert events == [terminal_event]
        framework_stream.assert_called_once()
        create_state.assert_not_called()


# =============================================================================
# _handle_reflection_complete tests
# =============================================================================


def _make_state_with_plan(
    num_steps: int,
    num_completed: int,
    depth: str = "medium",
) -> ResearchState:
    """Create a state with a plan for testing reflection handling."""
    steps = [
        PlanStep(
            id=f"s{i}",
            title=f"Step {i}",
            description=f"Description {i}",
            step_type=StepType.RESEARCH,
            needs_search=True,
            status=StepStatus.COMPLETED if i < num_completed else StepStatus.PENDING,
        )
        for i in range(num_steps)
    ]
    state = ResearchState(
        query="test",
        research_depth=depth,
        current_plan=Plan(
            id="p1",
            title="Test Plan",
            thought="Test",
            steps=steps,
            has_enough_context=False,
            iteration=1,
        ),
        current_step_index=num_completed,
        last_reflection=ReflectionResult(
            decision=ReflectionDecision.COMPLETE,
            reasoning="Done",
        ),
    )
    return state


class TestHandleReflectionComplete:
    """Tests for _handle_reflection_complete helper."""

    @patch("deep_research.agent.config.get_step_limits")
    def test_override_when_below_minimum(self, mock_limits: MagicMock) -> None:
        """Early completion is overridden when below min steps."""
        mock_limits.return_value = MagicMock(min=3)
        state = _make_state_with_plan(num_steps=5, num_completed=1)

        should_break, skipped = _handle_reflection_complete(state)

        assert should_break is False
        assert skipped == 0
        assert state.last_reflection is not None
        assert state.last_reflection.decision == ReflectionDecision.CONTINUE

    @patch("deep_research.agent.config.get_step_limits")
    def test_allows_completion_when_above_minimum(self, mock_limits: MagicMock) -> None:
        """Completion is allowed when at or above min steps."""
        mock_limits.return_value = MagicMock(min=2)
        state = _make_state_with_plan(num_steps=5, num_completed=3)

        should_break, skipped = _handle_reflection_complete(state)

        assert should_break is True
        assert skipped == 2  # 2 remaining steps skipped

    @patch("deep_research.agent.config.get_step_limits")
    def test_zero_min_steps_allows_immediate_completion(self, mock_limits: MagicMock) -> None:
        """Zero min steps allows completion at any point."""
        mock_limits.return_value = MagicMock(min=0)
        state = _make_state_with_plan(num_steps=5, num_completed=1)

        should_break, skipped = _handle_reflection_complete(state)

        assert should_break is True
        assert skipped == 4  # 4 remaining steps skipped

    @patch("deep_research.agent.config.get_step_limits")
    def test_all_steps_completed_no_skip(self, mock_limits: MagicMock) -> None:
        """When all steps done, should break with 0 skipped."""
        mock_limits.return_value = MagicMock(min=2)
        state = _make_state_with_plan(num_steps=3, num_completed=3)

        should_break, skipped = _handle_reflection_complete(state)

        assert should_break is True
        assert skipped == 0


# =============================================================================
# _extract_verification_from_report tests
# =============================================================================


def _make_source(url: str, title: str) -> object:
    """Create a lightweight source object for testing."""
    from types import SimpleNamespace

    return SimpleNamespace(url=url, title=title, snippet=f"Snippet for {title}")


class TestExtractVerificationFromReport:
    """Tests for _extract_verification_from_report."""

    def test_1indexed_markers_map_to_correct_sources(self) -> None:
        """LLM uses [1], [2] (1-indexed) → sources[0], sources[1]."""
        report = "Revenue grew 2% year over year [1]. EPS guidance was raised for Q4 [2]."
        sources = [
            _make_source("https://a.com", "Source A"),
            _make_source("https://b.com", "Source B"),
        ]
        claims, summary = _extract_verification_from_report(report, sources)
        assert len(claims) == 2
        assert claims[0].citation_key == "1"
        assert claims[0].evidence is not None
        assert claims[0].evidence.source_url == "https://a.com"
        assert claims[1].citation_key == "2"
        assert claims[1].evidence is not None
        assert claims[1].evidence.source_url == "https://b.com"

    def test_0indexed_markers_map_to_correct_sources(self) -> None:
        """LLM uses [0], [1] (0-indexed) → sources[0], sources[1]."""
        report = "Revenue grew 2% year over year [0]. EPS guidance was raised for Q4 [1]."
        sources = [
            _make_source("https://a.com", "Source A"),
            _make_source("https://b.com", "Source B"),
        ]
        claims, summary = _extract_verification_from_report(report, sources)
        assert len(claims) == 2
        assert claims[0].citation_key == "0"
        assert claims[0].evidence is not None
        assert claims[0].evidence.source_url == "https://a.com"
        assert claims[1].citation_key == "1"
        assert claims[1].evidence is not None
        assert claims[1].evidence.source_url == "https://b.com"

    def test_citation_keys_are_numeric_strings(self) -> None:
        """Citation keys must be numeric strings matching markdown parser."""
        report = "Multi-source claim about earnings [1][3]. Single claim about revenue growth [2]."
        sources = [_make_source(f"https://{i}.com", f"S{i}") for i in range(4)]
        claims, _ = _extract_verification_from_report(report, sources)
        assert len(claims) == 2
        assert claims[0].citation_key == "1"
        assert claims[0].citation_keys == ["1", "3"]
        assert claims[1].citation_key == "2"
        assert claims[1].citation_keys is None  # single marker → None

    def test_out_of_bounds_marker_no_crash(self) -> None:
        """Markers beyond source count produce claims without evidence."""
        report = "This claim references a nonexistent source [5]."
        sources = [_make_source("https://a.com", "A")]
        claims, _ = _extract_verification_from_report(report, sources)
        assert len(claims) == 1
        assert claims[0].evidence is None
        assert claims[0].citation_key == "5"

    def test_empty_sources_returns_empty(self) -> None:
        """Empty sources list returns no claims."""
        report = "Some text with markers [1]."
        claims, summary = _extract_verification_from_report(report, [])
        assert claims == []
        assert summary is None

    def test_no_markers_returns_empty(self) -> None:
        """Report without markers produces no claims."""
        report = "Just plain text without any citations at all."
        sources = [_make_source("https://a.com", "A")]
        claims, _ = _extract_verification_from_report(report, sources)
        assert claims == []

    def test_none_report_returns_empty(self) -> None:
        """None report returns no claims."""
        claims, summary = _extract_verification_from_report(None, [_make_source("https://a.com", "A")])
        assert claims == []
        assert summary is None

    def test_single_1indexed_marker(self) -> None:
        """Single [1] marker with 1-indexed convention maps to sources[0]."""
        report = "The company reported strong earnings growth [1]."
        sources = [_make_source("https://only.com", "Only Source")]
        claims, _ = _extract_verification_from_report(report, sources)
        assert len(claims) == 1
        assert claims[0].citation_key == "1"
        assert claims[0].evidence is not None
        assert claims[0].evidence.source_url == "https://only.com"

    def test_verification_summary_counts(self) -> None:
        """Summary has correct claim counts."""
        report = "First claim here [1]. Second claim here [2]. Third claim here [1][2]."
        sources = [
            _make_source("https://a.com", "A"),
            _make_source("https://b.com", "B"),
        ]
        claims, summary = _extract_verification_from_report(report, sources)
        assert len(claims) == 3
        assert summary is not None
        assert summary.total_claims == 3
        assert summary.supported_count == 3
        assert summary.unsupported_count == 0
