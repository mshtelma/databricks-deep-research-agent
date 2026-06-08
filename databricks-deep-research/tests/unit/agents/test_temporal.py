"""Unit tests for the ``PromptTemporalContext`` abstraction and its
integration into agent prompts via the shared TEMPORAL_ANCHOR_BLOCK.

These tests pin:
  * Production-clock construction emits well-formed ISO strings.
  * Tests can fix the clock by direct instantiation.
  * Prompts that embed ``{current_date}`` survive ``SafeTemplateRenderer``
    when the variable is missing (renders as empty, no literal ``{var}``).
  * Each builtin prompt that uses the temporal anchor block renders the
    actual injected date string after the harness's substitution pass.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta

import pytest
from pydantic import ValidationError

from databricks_deep_research.agents.prompts._shared import TEMPORAL_ANCHOR_BLOCK
from databricks_deep_research.agents.prompts.coordinator import COORDINATOR_SYSTEM_PROMPT
from databricks_deep_research.agents.prompts.reflector import REFLECTOR_SYSTEM_PROMPT
from databricks_deep_research.agents.prompts.researcher import RESEARCHER_DEFAULT_METHOD
from databricks_deep_research.agents.prompts.synthesizer import (
    STREAMING_SYNTHESIZER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
)
from databricks_deep_research.agents.temporal import PromptTemporalContext
from databricks_deep_research.templates.renderer import SafeTemplateRenderer

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}([+-]\d{2}:\d{2}|Z)$")


# ---------------------------------------------------------------------------
# PromptTemporalContext shape
# ---------------------------------------------------------------------------


def test_prompt_temporal_context_now_iso_shapes() -> None:
    """``now()`` produces well-formed ISO strings."""
    ctx = PromptTemporalContext.now()
    assert _ISO_DATE_RE.match(ctx.current_date), ctx.current_date
    assert _ISO_DATETIME_RE.match(ctx.current_iso_datetime), ctx.current_iso_datetime
    assert ctx.timezone_name == "UTC"
    # current_iso_datetime should round-trip through fromisoformat
    parsed = datetime.fromisoformat(ctx.current_iso_datetime)
    assert parsed.tzinfo is not None


def test_prompt_temporal_context_explicit_timezone() -> None:
    """Passing an explicit tz argument shifts the produced ISO datetime."""
    plus5 = timezone(timedelta(hours=5), name="custom_plus5")
    ctx = PromptTemporalContext.now(tz=plus5, tz_name="custom_plus5")
    assert ctx.timezone_name == "custom_plus5"
    # Offset string in the ISO datetime reflects the timezone we passed.
    assert "+05:00" in ctx.current_iso_datetime


def test_prompt_temporal_context_direct_construction() -> None:
    """Direct instantiation (the test/mock-clock path) works and freezes."""
    ctx = PromptTemporalContext(
        current_date="2024-01-15",
        current_iso_datetime="2024-01-15T00:00:00+00:00",
        timezone_name="UTC",
    )
    assert ctx.current_date == "2024-01-15"
    with pytest.raises(Exception):  # noqa: BLE001 — Pydantic ValidationError / AttributeError on frozen mutation
        ctx.current_date = "2025-01-01"  # type: ignore[misc]


def test_prompt_temporal_context_rejects_malformed_date() -> None:
    """current_date is length-pinned to 10 chars (YYYY-MM-DD)."""
    with pytest.raises(ValidationError):
        PromptTemporalContext(
            current_date="2024-01",  # too short
            current_iso_datetime="2024-01-15T00:00:00+00:00",
        )


def test_as_context_keys_emits_three_substitution_keys() -> None:
    """The dict produced by as_context_keys has the three expected slots."""
    ctx = PromptTemporalContext.now()
    keys = ctx.as_context_keys()
    assert set(keys.keys()) == {"current_date", "current_iso_datetime", "current_timezone"}
    assert keys["current_date"] == ctx.current_date
    assert keys["current_iso_datetime"] == ctx.current_iso_datetime
    assert keys["current_timezone"] == ctx.timezone_name


# ---------------------------------------------------------------------------
# Renderer behaviour — required for the missing-var safety story
# ---------------------------------------------------------------------------


def test_safe_renderer_missing_var_substitutes_empty() -> None:
    """``SafeTemplateRenderer`` substitutes missing vars to empty string.

    This is the contract that makes adding ``{current_date}`` to existing
    prompts safe even for callers who don't inject it — the prompt simply
    renders with a blank slot, not a literal ``{current_date}`` artifact
    reaching the LLM.
    """
    renderer = SafeTemplateRenderer()
    out = renderer.render("Hello {missing}!", {})
    assert out == "Hello !"


# ---------------------------------------------------------------------------
# TEMPORAL_ANCHOR_BLOCK presence in builtin prompts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "prompt"),
    [
        ("researcher", RESEARCHER_DEFAULT_METHOD),
        ("synthesizer", SYNTHESIZER_SYSTEM_PROMPT),
        ("streaming_synthesizer", STREAMING_SYNTHESIZER_SYSTEM_PROMPT),
        ("reflector", REFLECTOR_SYSTEM_PROMPT),
        ("coordinator", COORDINATOR_SYSTEM_PROMPT),
    ],
)
def test_builtin_prompts_include_temporal_anchor(name: str, prompt: str) -> None:
    """Every builtin system prompt embeds the shared TEMPORAL_ANCHOR_BLOCK.

    Locks the contract so a future prompt-rewrite that strips the block
    fails this test immediately.
    """
    # Use a stable signature substring — the exact wording may change, but
    # ``Today's date:`` is the load-bearing anchor we depend on.
    assert "## Temporal Anchor" in prompt, name
    assert "Today's date: {current_date}" in prompt, name
    # The block itself is identical across prompts.
    assert TEMPORAL_ANCHOR_BLOCK.strip() in prompt, name


def test_temporal_block_renders_with_injected_context() -> None:
    """Rendering a prompt with PromptTemporalContext keys produces a concrete date."""
    renderer = SafeTemplateRenderer()
    ctx = PromptTemporalContext(
        current_date="2026-05-19",
        current_iso_datetime="2026-05-19T13:36:25+00:00",
        timezone_name="UTC",
    )
    rendered = renderer.render(TEMPORAL_ANCHOR_BLOCK, ctx.as_context_keys())
    assert "Today's date: 2026-05-19 (UTC)" in rendered
    # No unresolved template artifacts.
    assert "{current_date}" not in rendered
    assert "{current_timezone}" not in rendered


def test_temporal_block_renders_with_missing_context_is_blank_safe() -> None:
    """Rendering the block with no temporal context produces an empty-slot
    string, not a crash and not a literal ``{current_date}`` leak."""
    renderer = SafeTemplateRenderer()
    rendered = renderer.render(TEMPORAL_ANCHOR_BLOCK, {})
    assert "Today's date:  ()" in rendered
    assert "{current_date}" not in rendered
