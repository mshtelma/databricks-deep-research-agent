"""Integration tests for custom agent feature (009-custom-agent-config).

End-to-end tests covering:
- CRUD operations via CustomAgentService against real database
- Config application with real source IDs and real app.test.yaml
- Enterprise tool execution scoped by agent config
- Preset step conversion to ManualStepDefinition
- Agent resolution and edge cases
- Model override validation with real YAML endpoints

Requirements:
- .env with DATABRICKS_TOKEN or DATABRICKS_CONFIG_PROFILE
- Access to KA endpoint, Genie space, VS index (for pipeline tests)
- Database configured (LAKEBASE_* or DATABASE_URL) for DB tests

Run with:
    uv run pytest tests/integration/test_custom_agent_e2e.py -v -s
    uv run pytest tests/integration/test_custom_agent_e2e.py -v -s -k "not Pipeline"
"""

import asyncio
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from tests.integration.conftest import requires_databricks

from deep_research.agent.orchestrator import (
    OrchestrationConfig,
    apply_custom_agent_to_config,
)
from deep_research.agent.tools.base import ResearchContext, ToolResult
from deep_research.agent.tools.factory import create_tools_from_source_ids
from deep_research.services.custom_agent_service import CustomAgentService

# ---------------------------------------------------------------------------
# Constants — reuse same enterprise sources as test_enterprise_source_tools.py
# ---------------------------------------------------------------------------

KA_ENDPOINT_NAME = "ka-99a12b9d-endpoint"
KA_SOURCE_ID = f"assistant:{KA_ENDPOINT_NAME}"

GENIE_SPACE_ID = "01f0b5ab5b841281858ae25da3f58125"
GENIE_SOURCE_ID = f"genie:{GENIE_SPACE_ID}"

VS_INDEX_NAME = "anthony_ivan.demo-toolsapp.pdf_chunks_index"
VS_SOURCE_ID = f"vs:{VS_INDEX_NAME}"

# Timeouts (seconds)
KA_TIMEOUT = 60
GENIE_TIMEOUT = 120
VS_TIMEOUT = 60
ALL_TIMEOUT = 240

# Test user IDs
TEST_OWNER_A = "integ-test-user-alpha"
TEST_OWNER_B = "integ-test-user-beta"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> ResearchContext:
    """Create a minimal ResearchContext for tool execution."""
    return ResearchContext(
        chat_id=uuid4(),
        user_id="integration-test-user",
        research_type="light",
        user_token=None,
    )


async def _execute_tool(tool: object, params: dict[str, str], timeout: float) -> ToolResult:
    """Execute a tool with timeout."""
    ctx = _make_context()
    return await asyncio.wait_for(
        tool.execute(params, ctx),  # type: ignore[union-attr]
        timeout=timeout,
    )


async def _create_test_agent(
    service: CustomAgentService,
    owner_id: str,
    name: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create a test agent via CustomAgentService with a unique name."""
    agent_name = name or f"Test Agent {uuid4().hex[:8]}"
    return await service.create_agent(
        owner_id=owner_id,
        name=agent_name,
        **kwargs,
    )


def _make_mock_agent(**kwargs: Any) -> MagicMock:
    """Create a mock CustomAgent with sensible defaults (same pattern as unit tests)."""
    agent = MagicMock()
    agent.id = kwargs.get("id", uuid4())
    agent.name = kwargs.get("name", "Test Agent")
    agent.owner_id = kwargs.get("owner_id", "user-1")

    agent.source_scope = kwargs.get("source_scope", "all")
    agent.enabled_sources = kwargs.get("enabled_sources")
    agent.disabled_sources = kwargs.get("disabled_sources")

    agent.default_depth = kwargs.get("default_depth", "medium")
    agent.default_mode = kwargs.get("default_mode", "planner")
    agent.enable_clarification = kwargs.get("enable_clarification", True)
    agent.use_planner = kwargs.get("use_planner", True)

    agent.output_format = kwargs.get("output_format", "markdown")
    agent.output_schema = kwargs.get("output_schema")

    agent.model_overrides = kwargs.get("model_overrides")
    agent.domain_filter_mode = kwargs.get("domain_filter_mode")
    agent.include_domains = kwargs.get("include_domains")
    agent.exclude_domains = kwargs.get("exclude_domains")

    agent.system_prompt_template = kwargs.get("system_prompt_template")
    agent.synthesis_template = kwargs.get("synthesis_template")
    agent.preset_steps = kwargs.get("preset_steps", [])

    return agent


def _make_mock_step(
    title: str,
    order: int,
    source_scope: str | None = None,
    source_hints: dict[str, Any] | None = None,
    description: str | None = None,
    is_required: bool = True,
) -> MagicMock:
    """Create a mock AgentPresetStep."""
    step = MagicMock()
    step.id = uuid4()
    step.title = title
    step.description = description or title
    step.order = order
    step.is_required = is_required
    step.source_scope = source_scope
    step.source_hints = source_hints
    return step


# ---------------------------------------------------------------------------
# Class 1: Agent CRUD via CustomAgentService against real DB
# ---------------------------------------------------------------------------


@requires_databricks
class TestAgentCrudServiceLayer:
    """Tests CustomAgentService against a real database via db_session fixture."""

    @pytest.mark.asyncio
    async def test_create_agent_with_all_fields(self, db_session: Any) -> None:
        """All columns round-trip through DB."""
        service = CustomAgentService(db_session)

        agent = await _create_test_agent(
            service,
            owner_id=TEST_OWNER_A,
            description="A test agent",
            source_scope="enterprise_only",
            enabled_sources=[VS_SOURCE_ID, GENIE_SOURCE_ID],
            disabled_sources=[KA_SOURCE_ID],
            default_depth="extended",
            default_mode="manual",
            model_overrides={"complex": "haiku"},
            domain_filter_mode="include",
            include_domains=["*.gov", "*.edu"],
            exclude_domains=["spam.com"],
        )

        assert agent.id is not None
        assert agent.owner_id == TEST_OWNER_A
        assert agent.source_scope == "enterprise_only"
        assert agent.enabled_sources == [VS_SOURCE_ID, GENIE_SOURCE_ID]
        assert agent.disabled_sources == [KA_SOURCE_ID]
        assert agent.default_depth == "extended"
        assert agent.default_mode == "manual"
        assert agent.model_overrides == {"complex": "haiku"}
        assert agent.domain_filter_mode == "include"
        assert agent.include_domains == ["*.gov", "*.edu"]
        assert agent.exclude_domains == ["spam.com"]

    @pytest.mark.asyncio
    async def test_create_agent_with_inline_preset_steps(self, db_session: Any) -> None:
        """Agent + 2 preset steps created atomically."""
        service = CustomAgentService(db_session)

        agent = await _create_test_agent(
            service,
            owner_id=TEST_OWNER_A,
            default_mode="manual",
            preset_steps=[
                {
                    "title": "Research competitors",
                    "description": "Analyze competitor products",
                    "order": 1,
                    "source_scope": "enterprise_only",
                    "source_hints": {"preferred_sources": [VS_SOURCE_ID]},
                },
                {
                    "title": "Web validation",
                    "description": "Cross-check with web sources",
                    "order": 2,
                    "source_scope": "web_only",
                },
            ],
        )

        assert len(agent.preset_steps) == 2

        steps = sorted(agent.preset_steps, key=lambda s: s.order)
        assert steps[0].title == "Research competitors"
        assert steps[0].order == 1
        assert steps[0].source_scope == "enterprise_only"
        assert steps[0].source_hints == {"preferred_sources": [VS_SOURCE_ID]}
        assert steps[1].title == "Web validation"
        assert steps[1].order == 2
        assert steps[1].source_scope == "web_only"

    @pytest.mark.asyncio
    async def test_get_accessible_agents_respects_private_visibility(
        self, db_session: Any
    ) -> None:
        """Private agent visible to owner, invisible to other user."""
        service = CustomAgentService(db_session)

        agent = await _create_test_agent(
            service,
            owner_id=TEST_OWNER_A,
            visibility="private",
        )

        # Owner can see it
        agents_a, count_a = await service.get_accessible_agents(TEST_OWNER_A)
        agent_ids_a = {a.id for a in agents_a}
        assert agent.id in agent_ids_a

        # Other user cannot see it
        agents_b, _ = await service.get_accessible_agents(TEST_OWNER_B)
        agent_ids_b = {a.id for a in agents_b}
        assert agent.id not in agent_ids_b

    @pytest.mark.asyncio
    async def test_get_accessible_agents_workspace_visible_to_all(
        self, db_session: Any
    ) -> None:
        """Workspace agent visible to both owner and other user."""
        service = CustomAgentService(db_session)

        agent = await _create_test_agent(
            service,
            owner_id=TEST_OWNER_A,
            visibility="workspace",
        )

        agents_a, _ = await service.get_accessible_agents(TEST_OWNER_A)
        assert agent.id in {a.id for a in agents_a}

        agents_b, _ = await service.get_accessible_agents(TEST_OWNER_B)
        assert agent.id in {a.id for a in agents_b}

    @pytest.mark.asyncio
    async def test_update_agent_fields_persists(self, db_session: Any) -> None:
        """Modify name/description/source_scope/depth, re-fetch, changes persisted."""
        service = CustomAgentService(db_session)

        agent = await _create_test_agent(service, owner_id=TEST_OWNER_A)
        original_id = agent.id

        agent.name = f"Updated Agent {uuid4().hex[:8]}"
        agent.description = "Updated description"
        agent.source_scope = "web_only"
        agent.default_depth = "light"
        await service.update(agent)

        fetched = await service.get(original_id)
        assert fetched is not None
        assert fetched.description == "Updated description"
        assert fetched.source_scope == "web_only"
        assert fetched.default_depth == "light"

    @pytest.mark.asyncio
    async def test_delete_agent_cascades_to_preset_steps(self, db_session: Any) -> None:
        """Delete agent -> both agent and steps gone."""
        service = CustomAgentService(db_session)

        agent = await _create_test_agent(
            service,
            owner_id=TEST_OWNER_A,
            preset_steps=[
                {"title": "Step 1", "order": 1},
                {"title": "Step 2", "order": 2},
            ],
        )

        agent_id = agent.id
        step_ids = [s.id for s in agent.preset_steps]
        assert len(step_ids) == 2

        await service.delete(agent)

        assert await service.get(agent_id) is None
        for step_id in step_ids:
            step = await service.get_preset_step(step_id, agent_id)
            assert step is None

    @pytest.mark.asyncio
    async def test_name_uniqueness_per_owner_enforced(self, db_session: Any) -> None:
        """Second agent with same name + same owner -> IntegrityError."""
        service = CustomAgentService(db_session)
        shared_name = f"Unique Agent {uuid4().hex[:8]}"

        await _create_test_agent(service, owner_id=TEST_OWNER_A, name=shared_name)

        with pytest.raises(IntegrityError):
            await _create_test_agent(service, owner_id=TEST_OWNER_A, name=shared_name)

        await db_session.rollback()


# ---------------------------------------------------------------------------
# Class 2: Agent config application with real sources and real YAML config
# ---------------------------------------------------------------------------


@requires_databricks
class TestAgentConfigWithRealSources:
    """Tests apply_custom_agent_to_config() with real source IDs and real app.test.yaml."""

    def test_enterprise_only_scope_with_enabled_sources(self) -> None:
        """Enterprise-only scope with enabled sources propagates to config."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            source_scope="enterprise_only",
            enabled_sources=[VS_SOURCE_ID, GENIE_SOURCE_ID, KA_SOURCE_ID],
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.source_scope == "enterprise_only"
        assert result.enabled_sources == [VS_SOURCE_ID, GENIE_SOURCE_ID, KA_SOURCE_ID]

    def test_disabled_sources_propagate_to_config(self) -> None:
        """Disabled sources from agent propagate to config."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(disabled_sources=[KA_SOURCE_ID])

        result = apply_custom_agent_to_config(config, agent)

        assert result.disabled_sources == [KA_SOURCE_ID]

    def test_domain_filter_include_mode(self) -> None:
        """Include-mode domain filter builds correct DomainFilterConfig."""
        from deep_research.core.app_config import DomainFilterMode

        config = OrchestrationConfig()
        agent = _make_mock_agent(
            domain_filter_mode="include",
            include_domains=["*.gov"],
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.domain_filter is not None
        assert result.domain_filter.mode == DomainFilterMode("include")
        assert result.domain_filter.include_domains == ["*.gov"]

    def test_domain_filter_exclude_mode(self) -> None:
        """Exclude-mode domain filter builds correct DomainFilterConfig."""
        from deep_research.core.app_config import DomainFilterMode

        config = OrchestrationConfig()
        agent = _make_mock_agent(
            domain_filter_mode="exclude",
            exclude_domains=["spam.com"],
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.domain_filter is not None
        assert result.domain_filter.mode == DomainFilterMode("exclude")
        assert result.domain_filter.exclude_domains == ["spam.com"]

    def test_domain_filter_both_mode(self) -> None:
        """Both-mode domain filter populates include + exclude lists."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            domain_filter_mode="both",
            include_domains=["*.gov"],
            exclude_domains=["spam.gov"],
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.domain_filter is not None
        assert result.domain_filter.mode.value == "both"
        assert result.domain_filter.include_domains == ["*.gov"]
        assert result.domain_filter.exclude_domains == ["spam.gov"]

    def test_model_overrides_valid_yaml_endpoint(self) -> None:
        """Model override with YAML endpoint name ('haiku') is accepted."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(model_overrides={"complex": "haiku"})

        result = apply_custom_agent_to_config(config, agent)

        assert result.model_overrides == {"complex": "haiku"}

    def test_model_overrides_direct_endpoint_passthrough(self) -> None:
        """Model override with non-YAML endpoint passes through as direct identifier."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            model_overrides={"analytical": "my-custom-llama"},
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.model_overrides == {"analytical": "my-custom-llama"}


# ---------------------------------------------------------------------------
# Class 3: Enterprise tool execution scoped by agent config
# ---------------------------------------------------------------------------


@requires_databricks
class TestAgentEnterprisePipeline:
    """Full pipeline: mock agent -> apply config -> create tools -> execute."""

    @pytest.mark.asyncio
    async def test_enterprise_agent_vs_tool_executes(self) -> None:
        """Agent with enterprise_only + VS source -> VS tool executes successfully."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            source_scope="enterprise_only",
            enabled_sources=[VS_SOURCE_ID],
        )

        config = apply_custom_agent_to_config(config, agent)
        tools = create_tools_from_source_ids(config.enabled_sources or [])

        assert len(tools) == 1
        result = await _execute_tool(
            tools[0], {"query": "data protection regulations"}, VS_TIMEOUT
        )

        print(f"\n[AgentPipeline] VS: success={result.success}, len={len(result.content)}")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_enterprise_agent_genie_tool_returns_data(self) -> None:
        """Agent with Genie source -> result.data has space_id."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            source_scope="enterprise_only",
            enabled_sources=[GENIE_SOURCE_ID],
        )

        config = apply_custom_agent_to_config(config, agent)
        tools = create_tools_from_source_ids(config.enabled_sources or [])

        assert len(tools) == 1
        result = await _execute_tool(
            tools[0], {"question": "What are the latest news about NVDA?"}, GENIE_TIMEOUT
        )

        print(f"\n[AgentPipeline] Genie: success={result.success}, len={len(result.content)}")
        assert result.success is True
        assert result.data is not None
        assert "space_id" in result.data

    @pytest.mark.asyncio
    async def test_enterprise_agent_ka_tool_returns_answer(self) -> None:
        """Agent with KA source -> answer content is substantive."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            source_scope="enterprise_only",
            enabled_sources=[KA_SOURCE_ID],
        )

        config = apply_custom_agent_to_config(config, agent)
        tools = create_tools_from_source_ids(config.enabled_sources or [])

        assert len(tools) == 1
        result = await _execute_tool(
            tools[0], {"question": "What is the general sentiment for NVDA stock?"}, KA_TIMEOUT
        )

        print(f"\n[AgentPipeline] KA: success={result.success}, len={len(result.content)}")
        assert result.success is True
        assert len(result.content) > 50

    @pytest.mark.asyncio
    async def test_disabled_sources_excluded_from_tools(self) -> None:
        """Agent with all 3 enabled but KA disabled -> only 2 tools created."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            source_scope="enterprise_only",
            enabled_sources=[VS_SOURCE_ID, GENIE_SOURCE_ID, KA_SOURCE_ID],
            disabled_sources=[KA_SOURCE_ID],
        )

        config = apply_custom_agent_to_config(config, agent)

        # Filter disabled from enabled before factory
        effective_sources = [
            s for s in (config.enabled_sources or [])
            if s not in (config.disabled_sources or [])
        ]
        tools = create_tools_from_source_ids(effective_sources)

        assert len(tools) == 2
        source_types = {t.definition.source_type for t in tools}
        assert "knowledge_assistant" not in source_types

    @pytest.mark.asyncio
    async def test_all_enterprise_tools_execute_from_agent_config(self) -> None:
        """Agent with all 3 sources -> all tools execute successfully."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            source_scope="enterprise_only",
            enabled_sources=[VS_SOURCE_ID, GENIE_SOURCE_ID, KA_SOURCE_ID],
        )

        config = apply_custom_agent_to_config(config, agent)
        tools = create_tools_from_source_ids(config.enabled_sources or [])
        assert len(tools) == 3

        vs_tool = next(t for t in tools if t.definition.source_type == "vector_search")
        genie_tool = next(t for t in tools if t.definition.source_type == "genie")
        ka_tool = next(t for t in tools if t.definition.source_type == "knowledge_assistant")

        ctx = _make_context()

        vs_result = await asyncio.wait_for(
            vs_tool.execute({"query": "data protection"}, ctx), timeout=VS_TIMEOUT
        )
        ka_result = await asyncio.wait_for(
            ka_tool.execute({"question": "NVDA stock sentiment"}, ctx), timeout=KA_TIMEOUT
        )
        genie_result = await asyncio.wait_for(
            genie_tool.execute({"question": "NVDA latest news"}, ctx), timeout=GENIE_TIMEOUT
        )

        print(f"\n[AllTools] VS: success={vs_result.success}, len={len(vs_result.content)}")
        print(f"[AllTools] KA: success={ka_result.success}, len={len(ka_result.content)}")
        print(f"[AllTools] Genie: success={genie_result.success}, len={len(genie_result.content)}")

        assert vs_result.success is True
        assert ka_result.success is True
        assert genie_result.success is True


# ---------------------------------------------------------------------------
# Class 4: Preset steps -> ManualStepDefinition conversion
# ---------------------------------------------------------------------------


class TestAgentPresetStepsConversion:
    """Tests preset step -> ManualStepDefinition conversion via apply_custom_agent_to_config()."""

    def test_manual_mode_populates_manual_steps(self) -> None:
        """2 mock steps -> config.manual_steps has 2 entries with correct titles."""
        step1 = _make_mock_step("Research competitors", 1)
        step2 = _make_mock_step("Analyze market trends", 2)

        config = OrchestrationConfig()
        agent = _make_mock_agent(default_mode="manual", preset_steps=[step1, step2])

        result = apply_custom_agent_to_config(config, agent)

        assert result.manual_steps is not None
        assert len(result.manual_steps) == 2
        assert result.manual_steps[0].title == "Research competitors"
        assert result.manual_steps[1].title == "Analyze market trends"

    def test_per_step_source_scope_preserved(self) -> None:
        """Per-step source_scope survives conversion."""
        step1 = _make_mock_step("Enterprise step", 1, source_scope="enterprise_only")
        step2 = _make_mock_step("Web step", 2, source_scope="web_only")

        config = OrchestrationConfig()
        agent = _make_mock_agent(default_mode="manual", preset_steps=[step1, step2])

        result = apply_custom_agent_to_config(config, agent)

        assert result.manual_steps is not None
        assert result.manual_steps[0].source_scope == "enterprise_only"
        assert result.manual_steps[1].source_scope == "web_only"

    def test_hybrid_mode_also_converts_steps(self) -> None:
        """Hybrid mode with preset steps also populates manual_steps."""
        step1 = _make_mock_step("Step A", 1)

        config = OrchestrationConfig()
        agent = _make_mock_agent(default_mode="hybrid", preset_steps=[step1])

        result = apply_custom_agent_to_config(config, agent)

        assert result.workflow_mode == "hybrid"
        assert result.manual_steps is not None
        assert len(result.manual_steps) == 1

    def test_planner_mode_ignores_preset_steps(self) -> None:
        """Planner mode does not convert preset steps."""
        step1 = _make_mock_step("Some step", 1)

        config = OrchestrationConfig()
        agent = _make_mock_agent(default_mode="planner", preset_steps=[step1])

        result = apply_custom_agent_to_config(config, agent)

        assert result.workflow_mode == "planner"
        assert result.manual_steps is None

    def test_source_hints_converted_to_attachments(self) -> None:
        """Step with source_hints.preferred_sources -> manual_steps[0].sources populated."""
        step1 = _make_mock_step(
            "VS step",
            1,
            source_hints={"preferred_sources": ["vs:idx1"]},
        )

        config = OrchestrationConfig()
        agent = _make_mock_agent(default_mode="manual", preset_steps=[step1])

        result = apply_custom_agent_to_config(config, agent)

        assert result.manual_steps is not None
        assert len(result.manual_steps) == 1
        assert len(result.manual_steps[0].sources) == 1
        assert result.manual_steps[0].sources[0].source_name == "vs:idx1"

    def test_empty_preset_steps_produces_none(self) -> None:
        """Manual mode with empty preset_steps -> manual_steps is None."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(default_mode="manual", preset_steps=[])

        result = apply_custom_agent_to_config(config, agent)

        assert result.manual_steps is None


# ---------------------------------------------------------------------------
# Class 5: Agent resolution and edge cases
# ---------------------------------------------------------------------------


@requires_databricks
class TestAgentResolutionAndEdgeCases:
    """Mix of DB + mock tests for resolution, name conflicts, and stale sources."""

    @pytest.mark.asyncio
    async def test_same_name_different_owners_coexist(self, db_session: Any) -> None:
        """Same name for different owners is allowed."""
        service = CustomAgentService(db_session)
        shared_name = f"My Agent {uuid4().hex[:8]}"

        agent_a = await _create_test_agent(
            service, owner_id=TEST_OWNER_A, name=shared_name
        )
        agent_b = await _create_test_agent(
            service, owner_id=TEST_OWNER_B, name=shared_name
        )

        assert agent_a.id != agent_b.id
        assert agent_a.name == agent_b.name

    @pytest.mark.asyncio
    async def test_same_name_same_owner_rejected(self, db_session: Any) -> None:
        """Duplicate name + owner -> IntegrityError."""
        service = CustomAgentService(db_session)
        shared_name = f"Dup Agent {uuid4().hex[:8]}"

        await _create_test_agent(service, owner_id=TEST_OWNER_A, name=shared_name)

        with pytest.raises(IntegrityError):
            await _create_test_agent(service, owner_id=TEST_OWNER_A, name=shared_name)

        await db_session.rollback()

    def test_stale_source_id_gracefully_handled(self) -> None:
        """Nonexistent source ID -> returns empty, no crash."""
        tools = create_tools_from_source_ids(["vs:nonexistent.catalog.schema.fake_index"])

        # The factory logs a warning but returns empty or a tool that will fail at execute
        # The key assertion: no exception raised during creation
        assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_resolve_agent_by_id(self, db_session: Any) -> None:
        """Create agent -> resolve_agent_for_request(agent_id=X) -> found."""
        service = CustomAgentService(db_session)

        agent = await _create_test_agent(service, owner_id=TEST_OWNER_A)
        resolved = await service.resolve_agent_for_request(
            user_id=TEST_OWNER_A, agent_id=agent.id
        )

        assert resolved is not None
        assert resolved.id == agent.id

    @pytest.mark.asyncio
    async def test_resolve_agent_by_name(self, db_session: Any) -> None:
        """Create 'Named Agent' -> resolve by name -> found."""
        service = CustomAgentService(db_session)
        agent_name = f"Named Agent {uuid4().hex[:8]}"

        agent = await _create_test_agent(
            service, owner_id=TEST_OWNER_A, name=agent_name
        )
        resolved = await service.resolve_agent_for_request(
            user_id=TEST_OWNER_A, agent_name=agent_name
        )

        assert resolved is not None
        assert resolved.id == agent.id

    @pytest.mark.asyncio
    async def test_resolve_workspace_agent_by_name_for_other_user(
        self, db_session: Any
    ) -> None:
        """Workspace agent by owner A -> resolved by owner B via name."""
        service = CustomAgentService(db_session)
        agent_name = f"Workspace Agent {uuid4().hex[:8]}"

        await _create_test_agent(
            service,
            owner_id=TEST_OWNER_A,
            name=agent_name,
            visibility="workspace",
        )

        resolved = await service.resolve_agent_for_request(
            user_id=TEST_OWNER_B, agent_name=agent_name
        )

        assert resolved is not None
        assert resolved.name == agent_name

    def test_query_overrides_precedence(self) -> None:
        """Agent source_scope='enterprise_only', query override -> config reflects override."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(source_scope="enterprise_only")

        result = apply_custom_agent_to_config(
            config, agent, query_overrides={"source_scope": "web_only"}
        )

        assert result.source_scope == "web_only"


# ---------------------------------------------------------------------------
# Class 6: Model overrides validation with real YAML config
# ---------------------------------------------------------------------------


@requires_databricks
class TestAgentModelOverridesValidation:
    """Uses real app.test.yaml config (NOT mocked get_app_config)."""

    def test_valid_yaml_endpoint_accepted(self) -> None:
        """'haiku' from app.test.yaml -> accepted as model override."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(model_overrides={"complex": "haiku"})

        result = apply_custom_agent_to_config(config, agent)

        assert result.model_overrides == {"complex": "haiku"}

    def test_direct_endpoint_passthrough(self) -> None:
        """Non-YAML endpoint passes through as direct identifier."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(model_overrides={"analytical": "my-custom-llama"})

        result = apply_custom_agent_to_config(config, agent)

        assert result.model_overrides == {"analytical": "my-custom-llama"}

    def test_empty_whitespace_overrides_dropped(self) -> None:
        """Empty or whitespace-only overrides are dropped entirely."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            model_overrides={"complex": "", "analytical": "   "},
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.model_overrides is None

    def test_mixed_valid_and_empty(self) -> None:
        """Only non-empty overrides survive."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            model_overrides={"complex": "haiku", "analytical": ""},
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.model_overrides == {"complex": "haiku"}

    def test_query_overrides_prevent_agent_overrides(self) -> None:
        """When query_overrides has model_overrides key, agent overrides are skipped."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(model_overrides={"complex": "haiku"})

        result = apply_custom_agent_to_config(
            config, agent, query_overrides={"model_overrides": None}
        )

        assert result.model_overrides is None

    def test_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace on endpoint IDs is stripped."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(model_overrides={"complex": "  haiku  "})

        result = apply_custom_agent_to_config(config, agent)

        assert result.model_overrides == {"complex": "haiku"}
