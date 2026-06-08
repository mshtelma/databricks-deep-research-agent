"""Unit tests for _merge_config() reasoning behavior.

Tests the provider-aware reasoning translation:
- Claude endpoints → thinking + budget_tokens via extra_body
- GPT/Gemini endpoints → reasoning_effort direct param
- supports_reasoning=False → no reasoning params
"""

from deep_research.services.llm.types import (
    CLAUDE_THINKING_BUDGETS,
    ModelEndpoint,
    ModelRole,
    ReasoningEffort,
)


def _make_endpoint(
    endpoint_id: str = "databricks-claude-opus-4-6",
    supports_reasoning: bool = True,
    reasoning_effort: ReasoningEffort | None = None,
    **kwargs: object,
) -> ModelEndpoint:
    """Create a ModelEndpoint with test defaults."""
    return ModelEndpoint(
        id="test",
        endpoint_identifier=endpoint_id,
        max_context_window=128000,
        tokens_per_minute=200000,
        supports_reasoning=supports_reasoning,
        reasoning_effort=reasoning_effort,
        **kwargs,  # type: ignore[arg-type]
    )


def _make_role(
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH,
    reasoning_budget: int | None = None,
    max_tokens: int = 32000,
) -> ModelRole:
    """Create a ModelRole with test defaults."""
    return ModelRole(
        name="complex",
        endpoints=["test"],
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_budget=reasoning_budget,
    )


def _call_merge_config(
    role: ModelRole,
    endpoint: ModelEndpoint,
    *,
    force_tool_use: bool = False,
) -> dict:
    """Call _merge_config via a minimal LLMClient mock.

    We test _merge_config as a standalone function by extracting it
    from the class. It only uses self for nothing — pure function of args.
    """
    from deep_research.services.llm.client import LLMClient

    # _merge_config is an instance method but doesn't use self beyond type
    return LLMClient._merge_config(  # type: ignore[arg-type]
        None, role, endpoint, force_tool_use=force_tool_use
    )


class TestClaudeThinking:
    """Claude endpoints: emulate reasoning effort via thinking + budget_tokens."""

    def test_claude_thinking_high(self) -> None:
        endpoint = _make_endpoint(endpoint_id="databricks-claude-opus-4-6")
        role = _make_role(reasoning_effort=ReasoningEffort.HIGH)
        config = _call_merge_config(role, endpoint)

        assert "extra_body" in config
        thinking = config["extra_body"]["thinking"]
        assert thinking["type"] == "enabled"
        assert thinking["budget_tokens"] == CLAUDE_THINKING_BUDGETS["high"]

    def test_claude_thinking_low(self) -> None:
        endpoint = _make_endpoint(endpoint_id="databricks-claude-sonnet-4-6")
        role = _make_role(reasoning_effort=ReasoningEffort.LOW)
        config = _call_merge_config(role, endpoint)

        assert config["extra_body"]["thinking"]["budget_tokens"] == CLAUDE_THINKING_BUDGETS["low"]

    def test_claude_thinking_medium(self) -> None:
        endpoint = _make_endpoint(endpoint_id="databricks-claude-opus-4-6")
        role = _make_role(reasoning_effort=ReasoningEffort.MEDIUM)
        config = _call_merge_config(role, endpoint)

        assert config["extra_body"]["thinking"]["budget_tokens"] == CLAUDE_THINKING_BUDGETS["medium"]

    def test_claude_thinking_max_capped_to_max_tokens(self) -> None:
        """MAX budget (32000) capped to max_tokens - 1024."""
        endpoint = _make_endpoint(endpoint_id="databricks-claude-opus-4-6")
        role = _make_role(reasoning_effort=ReasoningEffort.MAX, max_tokens=8000)
        config = _call_merge_config(role, endpoint)

        assert config["extra_body"]["thinking"]["budget_tokens"] == 8000 - 1024

    def test_claude_thinking_max_large_max_tokens(self) -> None:
        """MAX with large max_tokens uses full budget."""
        endpoint = _make_endpoint(endpoint_id="databricks-claude-opus-4-6")
        role = _make_role(reasoning_effort=ReasoningEffort.MAX, max_tokens=32000)
        config = _call_merge_config(role, endpoint)

        assert config["extra_body"]["thinking"]["budget_tokens"] == 32000 - 1024

    def test_claude_none_no_thinking(self) -> None:
        """NONE effort → no thinking dict."""
        endpoint = _make_endpoint(endpoint_id="databricks-claude-opus-4-6")
        role = _make_role(reasoning_effort=ReasoningEffort.NONE)
        config = _call_merge_config(role, endpoint)

        assert "extra_body" not in config

    def test_claude_minimal_no_thinking(self) -> None:
        """MINIMAL effort → no thinking dict (too small to be useful)."""
        endpoint = _make_endpoint(endpoint_id="databricks-claude-opus-4-6")
        role = _make_role(reasoning_effort=ReasoningEffort.MINIMAL)
        config = _call_merge_config(role, endpoint)

        assert "extra_body" not in config

    def test_claude_no_reasoning_effort_in_config(self) -> None:
        """Claude endpoints should NOT have reasoning_effort key."""
        endpoint = _make_endpoint(endpoint_id="databricks-claude-opus-4-6")
        role = _make_role(reasoning_effort=ReasoningEffort.HIGH)
        config = _call_merge_config(role, endpoint)

        assert "reasoning_effort" not in config


class TestGPTReasoningEffort:
    """GPT/Gemini endpoints: pass reasoning_effort directly."""

    def test_gpt_effort_high(self) -> None:
        endpoint = _make_endpoint(endpoint_id="databricks-gpt-5-4")
        role = _make_role(reasoning_effort=ReasoningEffort.HIGH)
        config = _call_merge_config(role, endpoint)

        assert config["reasoning_effort"] == "high"
        assert "extra_body" not in config

    def test_gpt_effort_max(self) -> None:
        """MAX passes through as 'max' — Databricks decides validity."""
        endpoint = _make_endpoint(endpoint_id="databricks-gpt-5-4")
        role = _make_role(reasoning_effort=ReasoningEffort.MAX)
        config = _call_merge_config(role, endpoint)

        assert config["reasoning_effort"] == "max"

    def test_gpt_effort_none(self) -> None:
        endpoint = _make_endpoint(endpoint_id="databricks-gpt-5-4")
        role = _make_role(reasoning_effort=ReasoningEffort.NONE)
        config = _call_merge_config(role, endpoint)

        assert config["reasoning_effort"] == "none"

    def test_gpt_effort_minimal(self) -> None:
        endpoint = _make_endpoint(endpoint_id="databricks-gpt-5-mini")
        role = _make_role(reasoning_effort=ReasoningEffort.MINIMAL)
        config = _call_merge_config(role, endpoint)

        assert config["reasoning_effort"] == "minimal"

    def test_gemini_effort_medium(self) -> None:
        endpoint = _make_endpoint(endpoint_id="databricks-gemini-3-flash")
        role = _make_role(reasoning_effort=ReasoningEffort.MEDIUM)
        config = _call_merge_config(role, endpoint)

        assert config["reasoning_effort"] == "medium"


class TestSupportsReasoningGuard:
    """supports_reasoning=False → no reasoning params regardless of effort."""

    def test_unsupported_claude_no_thinking(self) -> None:
        endpoint = _make_endpoint(
            endpoint_id="databricks-claude-haiku-4-5",
            supports_reasoning=False,
        )
        role = _make_role(reasoning_effort=ReasoningEffort.HIGH)
        config = _call_merge_config(role, endpoint)

        assert "extra_body" not in config
        assert "reasoning_effort" not in config

    def test_unsupported_gpt_no_effort(self) -> None:
        endpoint = _make_endpoint(
            endpoint_id="databricks-gpt-5-nano",
            supports_reasoning=False,
        )
        role = _make_role(reasoning_effort=ReasoningEffort.HIGH)
        config = _call_merge_config(role, endpoint)

        assert "reasoning_effort" not in config


class TestForceToolUseSuppressesThinking:
    """force_tool_use=True → Claude thinking suppressed (structured output is a
    forced tool call; the gateway rejects thinking + forced tool use).
    """

    def test_claude_force_tool_use_drops_thinking(self) -> None:
        endpoint = _make_endpoint(endpoint_id="databricks-claude-haiku-4-5")
        role = _make_role(reasoning_effort=ReasoningEffort.LOW)
        config = _call_merge_config(role, endpoint, force_tool_use=True)

        assert "extra_body" not in config
        # temperature=1 override only happens inside the thinking block
        assert config.get("temperature") != 1

    def test_claude_force_tool_use_drops_thinking_high(self) -> None:
        endpoint = _make_endpoint(endpoint_id="databricks-claude-opus-4-6")
        role = _make_role(reasoning_effort=ReasoningEffort.HIGH)
        config = _call_merge_config(role, endpoint, force_tool_use=True)

        assert "extra_body" not in config

    def test_claude_no_force_keeps_thinking(self) -> None:
        """Default (force_tool_use=False) preserves existing thinking behavior."""
        endpoint = _make_endpoint(endpoint_id="databricks-claude-opus-4-6")
        role = _make_role(reasoning_effort=ReasoningEffort.HIGH)
        config = _call_merge_config(role, endpoint, force_tool_use=False)

        assert config["extra_body"]["thinking"]["type"] == "enabled"

    def test_gpt_force_tool_use_keeps_reasoning_effort(self) -> None:
        """Forced tool use is a Claude-only constraint; GPT is unaffected."""
        endpoint = _make_endpoint(endpoint_id="databricks-gpt-5-4")
        role = _make_role(reasoning_effort=ReasoningEffort.HIGH)
        config = _call_merge_config(role, endpoint, force_tool_use=True)

        assert config["reasoning_effort"] == "high"


class TestEndpointOverridesRole:
    """Endpoint-level reasoning_effort takes precedence over role-level."""

    def test_endpoint_effort_overrides_role(self) -> None:
        endpoint = _make_endpoint(
            endpoint_id="databricks-claude-opus-4-6",
            reasoning_effort=ReasoningEffort.MEDIUM,
        )
        role = _make_role(reasoning_effort=ReasoningEffort.HIGH)
        config = _call_merge_config(role, endpoint)

        # MEDIUM budget (4096) not HIGH (10240)
        assert config["extra_body"]["thinking"]["budget_tokens"] == CLAUDE_THINKING_BUDGETS["medium"]

    def test_gpt_endpoint_effort_overrides_role(self) -> None:
        endpoint = _make_endpoint(
            endpoint_id="databricks-gpt-5-4",
            reasoning_effort=ReasoningEffort.LOW,
        )
        role = _make_role(reasoning_effort=ReasoningEffort.HIGH)
        config = _call_merge_config(role, endpoint)

        assert config["reasoning_effort"] == "low"
