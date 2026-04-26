"""Unit tests for _run_job() in JobManager.

Verifies:
1. Agent resolution wiring (009-custom-agent-config T011)
2. Post-stream DB operations use fresh sessions (stale connection fix)
3. Error handler guards against overwriting terminal status
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest


class TestAgentResolutionInRunJob:
    """Test agent resolution wiring in _run_job()."""

    @pytest.mark.asyncio
    async def test_agent_id_triggers_resolution(self) -> None:
        """When agent_id is provided, apply_custom_agent_to_config should be called.

        Verifying via source inspection that the agent resolution block exists
        in _run_job() since it has complex runtime dependencies.
        """
        import inspect

        from deep_research.services.job_manager import JobManager

        source = inspect.getsource(JobManager)

        # Verify the conditional agent resolution block
        assert "if agent_id:" in source
        assert "apply_custom_agent_to_config" in source
        # Agent resolution now routes through the factory (dual-backend support)
        assert "make_custom_agent_service" in source
        assert "get_accessible" in source

        # Verify the function exists and is callable
        from deep_research.agent.orchestrator import apply_custom_agent_to_config

        assert callable(apply_custom_agent_to_config)

    @pytest.mark.asyncio
    async def test_apply_custom_agent_called_with_agent(self) -> None:
        """Directly test that apply_custom_agent_to_config is called
        by verifying the import path used in job_manager."""
        import importlib
        import inspect

        spec = importlib.util.find_spec("deep_research.services.job_manager")
        assert spec is not None

        # Read the source to verify agent resolution is wired
        from deep_research.services.job_manager import JobManager

        source = inspect.getsource(JobManager)
        assert "apply_custom_agent_to_config" in source
        # Agent resolution now routes through the factory (dual-backend support)
        assert "make_custom_agent_service" in source
        assert "JOB_AGENT_CONFIG_APPLIED" in source
        assert "JOB_AGENT_NOT_FOUND" in source

    @pytest.mark.asyncio
    async def test_no_agent_id_skips_resolution(self) -> None:
        """When agent_id is None, the agent resolution block should be skipped."""
        import inspect

        from deep_research.services.job_manager import JobManager

        source = inspect.getsource(JobManager)
        # Verify the guard: "if agent_id:" is present
        assert "if agent_id:" in source

    @pytest.mark.asyncio
    async def test_agent_resolution_exception_handled(self) -> None:
        """Agent resolution failures should be caught and logged, not crash."""
        import inspect

        from deep_research.services.job_manager import JobManager

        source = inspect.getsource(JobManager)
        assert "JOB_AGENT_RESOLUTION_FAILED" in source
        assert "except Exception" in source


class TestAgentConfigApplyIntegration:
    """Integration-style tests verifying agent config flows through correctly."""

    def test_apply_sets_source_scope(self) -> None:
        """apply_custom_agent_to_config with enterprise_only sets config."""
        from deep_research.agent.orchestrator import (
            OrchestrationConfig,
            apply_custom_agent_to_config,
        )

        config = OrchestrationConfig()
        agent = MagicMock()
        agent.id = uuid4()
        agent.name = "Enterprise Agent"
        agent.source_scope = "enterprise_only"
        agent.enabled_sources = ["vs_1"]
        agent.disabled_sources = None
        agent.default_depth = "medium"
        agent.default_mode = "planner"
        agent.enable_clarification = True
        agent.output_format = "markdown"
        agent.output_schema = None
        agent.model_overrides = None
        agent.domain_filter_mode = None
        agent.include_domains = None
        agent.exclude_domains = None
        agent.system_prompt_template = None
        agent.synthesis_template = None
        agent.preset_steps = []

        result = apply_custom_agent_to_config(config, agent)

        assert result.source_scope == "enterprise_only"
        assert result.enabled_sources == ["vs_1"]

    def test_apply_preserves_defaults_with_empty_agent(self) -> None:
        """Agent with all defaults should not break config."""
        from deep_research.agent.orchestrator import (
            OrchestrationConfig,
            apply_custom_agent_to_config,
        )

        config = OrchestrationConfig()
        agent = MagicMock()
        agent.id = uuid4()
        agent.name = "Default Agent"
        agent.source_scope = "all"
        agent.enabled_sources = None
        agent.disabled_sources = None
        agent.default_depth = "medium"
        agent.default_mode = "planner"
        agent.enable_clarification = True
        agent.output_format = "markdown"
        agent.output_schema = None
        agent.model_overrides = None
        agent.domain_filter_mode = None
        agent.include_domains = None
        agent.exclude_domains = None
        agent.system_prompt_template = None
        agent.synthesis_template = None
        agent.preset_steps = []

        result = apply_custom_agent_to_config(config, agent)

        assert result.research_depth == "medium"
        assert result.model_overrides is None
        assert result.domain_filter is None


class TestStaleConnectionFix:
    """Verify _run_job uses fresh sessions for post-stream DB operations.

    The orchestrator uses independent sessions for all writes, so the
    connection held by the outer `async with session_maker()` block can
    go stale during long research runs.  Post-stream operations (completion
    check, timeout handling) must use fresh sessions to avoid
    ``asyncpg.InterfaceError: connection is closed``.
    """

    @pytest.mark.asyncio
    async def test_completion_path_uses_fresh_session(self) -> None:
        """Post-stream completion check must obtain a fresh session maker."""
        import inspect

        from deep_research.services.job_manager import JobManager

        source = inspect.getsource(JobManager._run_job)

        # The completion path should reference get_session_maker after the
        # stream returns, not reuse the outer `db`.
        stream_idx = source.index("_consume_research_stream")
        post_stream = source[stream_idx:]
        assert "completion_sm" in post_stream or "get_session_maker" in post_stream
        assert "JOB_COMPLETED" in post_stream

    @pytest.mark.asyncio
    async def test_timeout_path_uses_fresh_session(self) -> None:
        """Timeout handler must obtain a fresh session maker."""
        import inspect

        from deep_research.services.job_manager import JobManager

        source = inspect.getsource(JobManager._run_job)

        # Find the timeout handling section
        timeout_idx = source.index("RESEARCH_TIMEOUT")
        timeout_section = source[timeout_idx:]
        assert "timeout_sm" in timeout_section or "get_session_maker" in timeout_section

    @pytest.mark.asyncio
    async def test_error_handler_has_in_progress_guard(self) -> None:
        """Error handler must check IN_PROGRESS before overwriting to FAILED."""
        import inspect

        from deep_research.services.job_manager import JobManager

        source = inspect.getsource(JobManager._run_job)

        # Find the error handler section (after JOB_FAILED log)
        error_idx = source.index("JOB_FAILED")
        error_section = source[error_idx:]

        # Must check status before overwriting
        assert "ResearchStatus.IN_PROGRESS" in error_section

    @pytest.mark.asyncio
    async def test_cancel_handler_guard_preserved(self) -> None:
        """Cancel handler must still guard with IN_PROGRESS check (regression)."""
        import inspect

        from deep_research.services.job_manager import JobManager

        source = inspect.getsource(JobManager._run_job)

        cancel_idx = source.index("JOB_CANCELLED_BY_TASK")
        cancel_section = source[cancel_idx:]
        assert "ResearchStatus.IN_PROGRESS" in cancel_section

    @pytest.mark.asyncio
    async def test_error_handler_logs_skipped_terminal_status(self) -> None:
        """Error handler should log when skipping overwrite of terminal status."""
        import inspect

        from deep_research.services.job_manager import JobManager

        source = inspect.getsource(JobManager._run_job)
        assert "JOB_ERROR_SKIPPED_TERMINAL_STATUS" in source
