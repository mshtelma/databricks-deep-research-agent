"""Unit tests for apply_custom_agent_to_config() (009-custom-agent-config T010).

Tests the function that merges CustomAgent settings into OrchestrationConfig:
1. Source scope from agent overrides config
2. Model overrides from agent are set on config
3. Domain filter from agent columns constructs DomainFilterConfig on config
4. System instructions from template relationship are set on config
5. Query overrides take precedence over agent
6. Null/empty agent fields leave config defaults unchanged
7. Preset steps are converted in manual/hybrid mode
"""

from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

from deep_research.agent.orchestrator import (
    OrchestrationConfig,
    apply_custom_agent_to_config,
)


def _make_mock_agent(**kwargs: Any) -> MagicMock:
    """Create a mock CustomAgent with sensible defaults.

    All CustomAgent columns default to safe values. Override via kwargs.
    """
    agent = MagicMock()
    agent.id = kwargs.get("id", uuid4())
    agent.name = kwargs.get("name", "Test Agent")
    agent.owner_id = kwargs.get("owner_id", "user-1")

    # Source scope
    agent.source_scope = kwargs.get("source_scope", "all")
    agent.enabled_sources = kwargs.get("enabled_sources")
    agent.disabled_sources = kwargs.get("disabled_sources")

    # Workflow
    agent.default_depth = kwargs.get("default_depth", "medium")
    agent.default_mode = kwargs.get("default_mode", "planner")
    agent.enable_clarification = kwargs.get("enable_clarification", True)
    agent.use_planner = kwargs.get("use_planner", True)

    # Output
    agent.output_format = kwargs.get("output_format", "markdown")
    agent.output_schema = kwargs.get("output_schema")

    # Model overrides (009)
    agent.model_overrides = kwargs.get("model_overrides")

    # Domain filter (009)
    agent.domain_filter_mode = kwargs.get("domain_filter_mode")
    agent.include_domains = kwargs.get("include_domains")
    agent.exclude_domains = kwargs.get("exclude_domains")

    # Template relationships
    agent.system_prompt_template = kwargs.get("system_prompt_template")
    agent.synthesis_template = kwargs.get("synthesis_template")

    # Preset steps
    agent.preset_steps = kwargs.get("preset_steps", [])

    return agent


class TestSourceScopeOverride:
    """Test that source_scope from agent overrides config."""

    def test_agent_source_scope_overrides_config(self) -> None:
        config = OrchestrationConfig()
        agent = _make_mock_agent(source_scope="enterprise_only")

        result = apply_custom_agent_to_config(config, agent)

        assert result.source_scope == "enterprise_only"

    def test_agent_web_only_scope(self) -> None:
        config = OrchestrationConfig()
        agent = _make_mock_agent(source_scope="web_only")

        result = apply_custom_agent_to_config(config, agent)

        assert result.source_scope == "web_only"

    def test_agent_enabled_sources_set(self) -> None:
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            source_scope="enterprise_only",
            enabled_sources=["vs_1", "genie_2"],
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.source_scope == "enterprise_only"
        assert result.enabled_sources == ["vs_1", "genie_2"]

    def test_agent_disabled_sources_set(self) -> None:
        config = OrchestrationConfig()
        agent = _make_mock_agent(disabled_sources=["web_general"])

        result = apply_custom_agent_to_config(config, agent)

        assert result.disabled_sources == ["web_general"]


class TestModelOverrides:
    """Test that model_overrides from agent are set on config."""

    @patch("deep_research.core.app_config.get_app_config")
    def test_model_overrides_applied(self, mock_get_config: MagicMock) -> None:
        """Valid model overrides should be set on config."""
        mock_config = MagicMock()
        mock_config.endpoints = {
            "databricks-haiku": MagicMock(),
            "databricks-opus": MagicMock(),
        }
        mock_get_config.return_value = mock_config

        config = OrchestrationConfig()
        agent = _make_mock_agent(
            model_overrides={"complex": "databricks-haiku"},
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.model_overrides == {"complex": "databricks-haiku"}

    @patch("deep_research.core.app_config.get_app_config")
    def test_model_override_non_yaml_endpoint_passes_through(
        self, mock_get_config: MagicMock
    ) -> None:
        """Non-YAML endpoint ID is NOT rejected — passes through as direct identifier."""
        mock_config = MagicMock()
        mock_config.endpoints = {
            "databricks-haiku": MagicMock(),
        }
        mock_get_config.return_value = mock_config

        config = OrchestrationConfig()
        agent = _make_mock_agent(
            model_overrides={
                "complex": "databricks-haiku",
                "analytical": "my-custom-fine-tuned-llama",
            },
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.model_overrides == {
            "complex": "databricks-haiku",
            "analytical": "my-custom-fine-tuned-llama",
        }

    @patch("deep_research.core.app_config.get_app_config")
    def test_model_override_empty_string_skipped(
        self, mock_get_config: MagicMock
    ) -> None:
        """Empty or whitespace-only overrides are dropped."""
        mock_config = MagicMock()
        mock_config.endpoints = {}
        mock_get_config.return_value = mock_config

        config = OrchestrationConfig()
        agent = _make_mock_agent(
            model_overrides={
                "complex": "",
                "analytical": "   ",
            },
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.model_overrides is None

    @patch("deep_research.core.app_config.get_app_config")
    def test_model_override_yaml_alias_still_works(
        self, mock_get_config: MagicMock
    ) -> None:
        """YAML alias like 'haiku' continues to pass validation."""
        mock_config = MagicMock()
        mock_config.endpoints = {
            "haiku": MagicMock(),
        }
        mock_get_config.return_value = mock_config

        config = OrchestrationConfig()
        agent = _make_mock_agent(
            model_overrides={"complex": "haiku"},
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.model_overrides == {"complex": "haiku"}

    @patch("deep_research.core.app_config.get_app_config")
    def test_model_override_strips_whitespace(
        self, mock_get_config: MagicMock
    ) -> None:
        """Endpoint IDs with leading/trailing whitespace are stripped."""
        mock_config = MagicMock()
        mock_config.endpoints = {}
        mock_get_config.return_value = mock_config

        config = OrchestrationConfig()
        agent = _make_mock_agent(
            model_overrides={"complex": "  my-endpoint  "},
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.model_overrides == {"complex": "my-endpoint"}

    def test_null_model_overrides_unchanged(self) -> None:
        """Null model_overrides on agent leaves config default."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(model_overrides=None)

        result = apply_custom_agent_to_config(config, agent)

        assert result.model_overrides is None


class TestDomainFilter:
    """Test that domain filter from agent columns constructs DomainFilterConfig."""

    def test_include_domain_filter_applied(self) -> None:
        """Agent with include domain filter should create DomainFilterConfig."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            domain_filter_mode="include",
            include_domains=["*.gov", "*.edu"],
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.domain_filter.mode.value == "include"
        assert result.domain_filter.include_domains == ["*.gov", "*.edu"]

    def test_exclude_domain_filter_applied(self) -> None:
        """Agent with exclude domain filter should create DomainFilterConfig."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            domain_filter_mode="exclude",
            exclude_domains=["spam.com", "ads.net"],
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.domain_filter.mode.value == "exclude"
        assert result.domain_filter.exclude_domains == ["spam.com", "ads.net"]

    def test_both_domain_filter_applied(self) -> None:
        """Agent with both mode constructs DomainFilterConfig with both lists."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            domain_filter_mode="both",
            include_domains=["*.gov"],
            exclude_domains=["spam.gov"],
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.domain_filter.mode.value == "both"
        assert result.domain_filter.include_domains == ["*.gov"]
        assert result.domain_filter.exclude_domains == ["spam.gov"]

    def test_null_domain_filter_unchanged(self) -> None:
        """Null domain_filter_mode leaves config.domain_filter as None."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(domain_filter_mode=None)

        result = apply_custom_agent_to_config(config, agent)

        assert result.domain_filter is None

    def test_invalid_domain_filter_mode_ignored(self) -> None:
        """Invalid domain_filter_mode should be ignored gracefully."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(domain_filter_mode="invalid_mode")

        result = apply_custom_agent_to_config(config, agent)

        assert result.domain_filter is None


class TestTemplateWiring:
    """Test system instructions from template relationship."""

    def test_system_prompt_template_applied(self) -> None:
        """Agent with system_prompt_template sets config.system_instructions."""
        template = MagicMock()
        template.content = "Be formal and academic."

        config = OrchestrationConfig()
        agent = _make_mock_agent(system_prompt_template=template)

        result = apply_custom_agent_to_config(config, agent)

        assert result.system_instructions == "Be formal and academic."

    def test_synthesis_template_applied(self) -> None:
        """Agent with synthesis_template sets config.structured_system_prompt."""
        template = MagicMock()
        template.content = "Synthesize with citations."

        config = OrchestrationConfig()
        agent = _make_mock_agent(synthesis_template=template)

        result = apply_custom_agent_to_config(config, agent)

        assert result.structured_system_prompt == "Synthesize with citations."

    def test_null_template_leaves_default(self) -> None:
        """Null templates leave config defaults unchanged."""
        config = OrchestrationConfig()
        original_instructions = config.system_instructions

        agent = _make_mock_agent(
            system_prompt_template=None,
            synthesis_template=None,
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.system_instructions == original_instructions

    def test_template_with_no_content_ignored(self) -> None:
        """Template with no content should not override config."""
        template = MagicMock()
        template.content = None

        config = OrchestrationConfig()
        original = config.system_instructions

        agent = _make_mock_agent(system_prompt_template=template)

        result = apply_custom_agent_to_config(config, agent)

        assert result.system_instructions == original


class TestQueryOverridePrecedence:
    """Test that query overrides take precedence over agent settings."""

    def test_query_source_scope_overrides_agent(self) -> None:
        """Query-level source_scope should override agent source_scope."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(source_scope="enterprise_only")

        result = apply_custom_agent_to_config(
            config, agent, query_overrides={"source_scope": "web_only"}
        )

        assert result.source_scope == "web_only"

    def test_query_research_depth_overrides_agent(self) -> None:
        """Query-level research_depth should override agent default."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(default_depth="light")

        result = apply_custom_agent_to_config(
            config, agent, query_overrides={"research_depth": "extended"}
        )

        assert result.research_depth == "extended"

    def test_query_enabled_sources_overrides_agent(self) -> None:
        """Query-level enabled_sources should override agent sources."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(enabled_sources=["vs_1"])

        result = apply_custom_agent_to_config(
            config, agent, query_overrides={"enabled_sources": ["vs_2", "vs_3"]}
        )

        assert result.enabled_sources == ["vs_2", "vs_3"]

    @patch("deep_research.core.app_config.get_app_config")
    def test_query_model_overrides_prevents_agent_overrides(
        self, mock_get_config: MagicMock
    ) -> None:
        """When query_overrides contains model_overrides key, agent overrides are skipped."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            model_overrides={"complex": "databricks-haiku"},
        )

        result = apply_custom_agent_to_config(
            config, agent, query_overrides={"model_overrides": None}
        )

        # get_app_config should not have been called since the key exists in overrides
        mock_get_config.assert_not_called()
        assert result.model_overrides is None


class TestNullFieldsDefault:
    """Test that null/empty agent fields leave config defaults unchanged."""

    def test_empty_agent_preserves_defaults(self) -> None:
        """Agent with all defaults should not change config meaningfully."""
        config = OrchestrationConfig()
        agent = _make_mock_agent()

        result = apply_custom_agent_to_config(config, agent)

        # Defaults should flow through
        assert result.research_depth == "medium"
        assert result.workflow_mode == "planner"
        assert result.enable_clarification is True
        assert result.output_format == "markdown"
        assert result.model_overrides is None
        assert result.domain_filter is None

    def test_null_source_scope_leaves_default(self) -> None:
        """Null source_scope on agent should not override config."""
        config = OrchestrationConfig()
        config.source_scope = "web_only"

        # source_scope is falsy (empty string) — should not override
        agent = _make_mock_agent(source_scope="")

        result = apply_custom_agent_to_config(config, agent)

        # Empty string is falsy, so the elif branch is skipped
        assert result.source_scope == "web_only"


class TestPresetStepsConversion:
    """Test that preset steps are converted in manual/hybrid mode."""

    def test_manual_mode_converts_preset_steps(self) -> None:
        """Manual mode with preset steps should convert to manual_steps."""
        step1 = MagicMock()
        step1.id = uuid4()
        step1.title = "Research competitors"
        step1.description = "Analyze competitor products"
        step1.order = 1
        step1.is_required = True
        step1.source_hints = None
        step1.source_scope = None

        config = OrchestrationConfig()
        agent = _make_mock_agent(
            default_mode="manual",
            preset_steps=[step1],
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.workflow_mode == "manual"
        assert result.manual_steps is not None
        assert len(result.manual_steps) == 1
        assert result.manual_steps[0].title == "Research competitors"

    def test_planner_mode_ignores_preset_steps(self) -> None:
        """Planner mode should not convert preset steps."""
        step1 = MagicMock()
        step1.id = uuid4()
        step1.title = "Some step"
        step1.description = None
        step1.order = 1
        step1.is_required = True
        step1.source_hints = None
        step1.source_scope = None

        config = OrchestrationConfig()
        agent = _make_mock_agent(
            default_mode="planner",
            preset_steps=[step1],
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.workflow_mode == "planner"
        assert result.manual_steps is None

    def test_hybrid_mode_converts_preset_steps(self) -> None:
        """Hybrid mode with preset steps should also convert."""
        step1 = MagicMock()
        step1.id = uuid4()
        step1.title = "Step A"
        step1.description = "Do A"
        step1.order = 1
        step1.is_required = True
        step1.source_hints = None
        step1.source_scope = None

        config = OrchestrationConfig()
        agent = _make_mock_agent(
            default_mode="hybrid",
            preset_steps=[step1],
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.workflow_mode == "hybrid"
        assert result.manual_steps is not None
        assert len(result.manual_steps) == 1


class TestPresetStepsSourceScope:
    """Test that preset steps with source_scope overrides are correctly converted (T043)."""

    def test_step_with_source_scope_override_converted(self) -> None:
        """Preset step with source_scope should carry it through to ManualStepDefinition."""
        step1 = MagicMock()
        step1.id = uuid4()
        step1.title = "Enterprise research"
        step1.description = "Query internal data"
        step1.order = 1
        step1.is_required = True
        step1.source_hints = None
        step1.source_scope = "enterprise_only"

        step2 = MagicMock()
        step2.id = uuid4()
        step2.title = "Web research"
        step2.description = "Search the web"
        step2.order = 2
        step2.is_required = True
        step2.source_hints = None
        step2.source_scope = "web_only"

        config = OrchestrationConfig()
        agent = _make_mock_agent(
            default_mode="manual",
            preset_steps=[step1, step2],
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.manual_steps is not None
        assert len(result.manual_steps) == 2
        assert result.manual_steps[0].source_scope == "enterprise_only"
        assert result.manual_steps[1].source_scope == "web_only"

    def test_step_without_source_scope_inherits_none(self) -> None:
        """Preset step without source_scope should have None (inherit agent default)."""
        step1 = MagicMock()
        step1.id = uuid4()
        step1.title = "Default step"
        step1.description = "Uses agent default"
        step1.order = 1
        step1.is_required = True
        step1.source_hints = None
        step1.source_scope = None

        config = OrchestrationConfig()
        agent = _make_mock_agent(
            default_mode="manual",
            preset_steps=[step1],
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.manual_steps is not None
        assert len(result.manual_steps) == 1
        assert result.manual_steps[0].source_scope is None

    def test_mixed_source_scope_steps(self) -> None:
        """Mix of steps with and without source_scope overrides."""
        step1 = MagicMock()
        step1.id = uuid4()
        step1.title = "Enterprise step"
        step1.description = "Only enterprise"
        step1.order = 1
        step1.is_required = True
        step1.source_hints = None
        step1.source_scope = "enterprise_only"

        step2 = MagicMock()
        step2.id = uuid4()
        step2.title = "Default step"
        step2.description = "Uses default"
        step2.order = 2
        step2.is_required = False
        step2.source_hints = None
        step2.source_scope = None

        config = OrchestrationConfig()
        agent = _make_mock_agent(
            default_mode="hybrid",
            preset_steps=[step1, step2],
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.manual_steps is not None
        assert len(result.manual_steps) == 2
        assert result.manual_steps[0].source_scope == "enterprise_only"
        assert result.manual_steps[1].source_scope is None


class TestWorkflowAndOutput:
    """Test workflow and output configuration settings."""

    def test_agent_depth_applied(self) -> None:
        config = OrchestrationConfig()
        agent = _make_mock_agent(default_depth="extended")

        result = apply_custom_agent_to_config(config, agent)

        assert result.research_depth == "extended"

    def test_agent_clarification_disabled(self) -> None:
        config = OrchestrationConfig()
        agent = _make_mock_agent(enable_clarification=False)

        result = apply_custom_agent_to_config(config, agent)

        assert result.enable_clarification is False

    def test_agent_json_output_format(self) -> None:
        config = OrchestrationConfig()
        agent = _make_mock_agent(
            output_format="json",
            output_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
        )

        result = apply_custom_agent_to_config(config, agent)

        assert result.output_format == "json"
        assert result.output_schema is not None


class TestConfigReturnValue:
    """Test that the function returns the modified config."""

    def test_returns_same_config_object(self) -> None:
        """Should return the same config object, mutated in place."""
        config = OrchestrationConfig()
        agent = _make_mock_agent(source_scope="enterprise_only")

        result = apply_custom_agent_to_config(config, agent)

        assert result is config

    def test_config_has_expected_new_fields(self) -> None:
        """OrchestrationConfig should have model_overrides and domain_filter fields."""
        config = OrchestrationConfig()
        assert hasattr(config, "model_overrides")
        assert hasattr(config, "domain_filter")
        assert config.model_overrides is None
        assert config.domain_filter is None
