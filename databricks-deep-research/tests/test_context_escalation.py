"""Tests for context-window-aware model escalation in FrameworkLLMClient."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from databricks_deep_research.errors import ContextWindowExceededError
from databricks_deep_research.llm.budget import estimate_message_tokens
from databricks_deep_research.llm.client import (
    FrameworkLLMClient,
    ModelTierConfig,
    _truncate_messages_to_tokens,
    parse_model_config,
)


def _client(
    model_mapping: dict[str, str | ModelTierConfig],
    registry: dict[str, int] | None = None,
) -> FrameworkLLMClient:
    return FrameworkLLMClient(
        openai_client=AsyncMock(),
        model_mapping=model_mapping,
        endpoint_registry=registry,
    )


def _msg(role: str, chars: int) -> dict[str, str]:
    return {"role": role, "content": "x" * chars}


# -- estimator --------------------------------------------------------------


def test_estimate_message_tokens_counts_content_and_tools() -> None:
    messages = [_msg("system", 400), _msg("user", 400)]  # ~800 chars
    base = estimate_message_tokens(messages)
    assert base >= 200  # ~800/4, plus overhead
    with_tools = estimate_message_tokens(
        messages, tools=[{"function": {"name": "x", "parameters": {"a": "b" * 400}}}]
    )
    assert with_tools > base


def test_estimate_message_tokens_minimum_one() -> None:
    assert estimate_message_tokens([]) == 1


# -- registry backfill ------------------------------------------------------


def test_registry_backfilled_from_tier_windows() -> None:
    cfg = ModelTierConfig(
        endpoints=["small"], endpoint_context_windows={"small": 1000}
    )
    client = _client({"analytical": cfg})
    assert client._window_of("small") == 1000


# -- selection --------------------------------------------------------------


def test_primary_fits_no_escalation() -> None:
    cfg = ModelTierConfig(endpoints=["big"], endpoint_context_windows={"big": 100_000})
    client = _client({"analytical": cfg})
    chosen, escalated_from = client._select_context_fit_endpoint("analytical", 5_000)
    assert chosen == "big"
    assert escalated_from is None


def test_same_tier_escalation() -> None:
    cfg = ModelTierConfig(
        endpoints=["small", "medium"],
        endpoint_context_windows={"small": 10_000, "medium": 300_000},
    )
    client = _client({"analytical": cfg})
    # required exceeds 'small' (primary) but fits 'medium' in the same tier.
    chosen, escalated_from = client._select_context_fit_endpoint("analytical", 120_000)
    assert chosen == "medium"
    assert escalated_from == "small"


def test_global_escalation_picks_smallest_fitting() -> None:
    cfg = ModelTierConfig(
        endpoints=["small"], endpoint_context_windows={"small": 10_000}
    )
    # Global registry has two larger options; the smallest fitting one wins.
    client = _client(
        {"analytical": cfg},
        registry={"small": 10_000, "huge": 1_000_000, "medium": 200_000},
    )
    chosen, escalated_from = client._select_context_fit_endpoint("analytical", 120_000)
    assert chosen == "medium"
    assert escalated_from == "small"


def test_nothing_fits_returns_primary() -> None:
    cfg = ModelTierConfig(
        endpoints=["small"], endpoint_context_windows={"small": 10_000}
    )
    client = _client({"analytical": cfg})
    chosen, escalated_from = client._select_context_fit_endpoint("analytical", 999_999)
    assert chosen == "small"
    assert escalated_from is None


def test_unknown_primary_window_no_escalation() -> None:
    # Bare-string tier (no window known) → escalation is a no-op.
    client = _client({"analytical": "mystery-model"})
    chosen, escalated_from = client._select_context_fit_endpoint("analytical", 999_999)
    assert chosen == "mystery-model"
    assert escalated_from is None


# -- fallback window guard --------------------------------------------------


def test_find_fallback_skips_too_small_window() -> None:
    cfg = ModelTierConfig(
        endpoints=["small", "tiny", "big"],
        endpoint_context_windows={"small": 10_000, "tiny": 5_000, "big": 500_000},
    )
    client = _client({"analytical": cfg})
    # required_total excludes 'tiny'; 'big' is the only viable fallback.
    fb = client._find_fallback("analytical", "small", required_total=120_000)
    assert fb == "big"


def test_find_fallback_no_constraint_keeps_legacy_behavior() -> None:
    cfg = ModelTierConfig(
        endpoints=["a", "b"], endpoint_context_windows={"a": 10_000, "b": 10_000}
    )
    client = _client({"analytical": cfg})
    fb = client._find_fallback("analytical", "a")  # required_total defaults to 0
    assert fb == "b"


# -- _resolve_for_context ---------------------------------------------------


def test_resolve_escalates_to_big_endpoint() -> None:
    cfg = ModelTierConfig(
        endpoints=["small"], endpoint_context_windows={"small": 10_000}
    )
    client = _client({"analytical": cfg}, registry={"small": 10_000, "big": 1_000_000})
    big_messages = [_msg("user", 200_000)]  # ~50k tokens, over 'small'
    model, messages, required = client._resolve_for_context(
        "analytical", big_messages, None, max_tokens=1000
    )
    assert model == "big"
    assert messages is big_messages  # not truncated — it fits 'big'
    assert required > 10_000


def test_resolve_truncates_when_nothing_fits() -> None:
    cfg = ModelTierConfig(
        endpoints=["small"],
        endpoint_context_windows={"small": 2_000},
        on_context_overflow="truncate",
    )
    client = _client({"analytical": cfg})
    big_messages = [_msg("system", 100), _msg("user", 400_000)]
    model, messages, _ = client._resolve_for_context(
        "analytical", big_messages, None, max_tokens=200
    )
    assert model == "small"
    # Truncated to fit ~2000-token window.
    assert estimate_message_tokens(messages) <= 2_000


def test_resolve_fails_fast_when_configured() -> None:
    cfg = ModelTierConfig(
        endpoints=["small"],
        endpoint_context_windows={"small": 2_000},
        on_context_overflow="fail",
    )
    client = _client({"analytical": cfg})
    big_messages = [_msg("user", 400_000)]
    with pytest.raises(ContextWindowExceededError):
        client._resolve_for_context("analytical", big_messages, None, max_tokens=200)


# -- truncation helper ------------------------------------------------------


def test_truncate_preserves_system_and_fits_budget() -> None:
    messages = [
        _msg("system", 200),
        _msg("user", 4_000),
        _msg("assistant", 4_000),
        _msg("user", 4_000),
    ]
    out = _truncate_messages_to_tokens(messages, max_input_tokens=300)
    assert out[0]["role"] == "system"
    assert estimate_message_tokens(out) <= 300


def test_truncate_drops_orphan_tool_leading_message() -> None:
    messages = [
        _msg("system", 100),
        {"role": "tool", "content": "x" * 4_000, "tool_call_id": "t1"},
        _msg("user", 100),
    ]
    out = _truncate_messages_to_tokens(messages, max_input_tokens=80)
    # The kept suffix must not begin on an orphan tool message.
    non_system = [m for m in out if m.get("role") != "system"]
    assert non_system[0]["role"] != "tool"


# -- parse_model_config parity ----------------------------------------------


def test_parse_model_config_reads_windows_and_overflow() -> None:
    parsed = parse_model_config(
        {
            "analytical": {
                "endpoints": ["a", "b"],
                "endpoint_context_windows": {"a": 128_000, "b": 1_000_000},
                "on_context_overflow": "fail",
            }
        }
    )
    cfg = parsed["analytical"]
    assert isinstance(cfg, ModelTierConfig)
    assert cfg.endpoint_context_windows == {"a": 128_000, "b": 1_000_000}
    assert cfg.on_context_overflow == "fail"


def test_parse_model_config_rejects_bad_overflow() -> None:
    with pytest.raises(ValueError, match="on_context_overflow"):
        parse_model_config(
            {"analytical": {"endpoints": ["a"], "on_context_overflow": "explode"}}
        )
