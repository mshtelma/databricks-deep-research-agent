"""Unit tests for _run_job() in JobManager.

Verifies:
1. Post-stream DB operations use fresh sessions (stale connection fix)
2. Error handler guards against overwriting terminal status
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest


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
