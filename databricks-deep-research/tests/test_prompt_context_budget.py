"""Tests for the three-tier synth-context preservation policy.

Background: the previous synth-context assembly applied a 300-char cap to
each observation and emitted title+URL only for each source. That discarded
most researcher output and primed the synthesiser to hallucinate. The new
policy keeps the first ``keep_full_top_k`` items verbatim, keeps subsequent
items as long as ``total_budget_chars`` allows, and only then applies
``truncation_policy``.
"""
from __future__ import annotations

from types import SimpleNamespace

from databricks_deep_research.agents.config import (
    SynthesisContextConfig,
    SynthesisContextFieldConfig,
)
from databricks_deep_research.agents.prompt_context import (
    _apply_budget,
    _render_observations_with_budget,
    _render_source_bullet,
    _render_sources_with_budget,
    compile_typed_synthesis_context,
    default_synthesis_context,
)
from databricks_deep_research.workflow.runtime_core.models import (
    EvidenceState,
    ObservationRecord,
    SourceRecord,
)


def _make_cfg(**overrides) -> SynthesisContextFieldConfig:
    base: dict[str, object] = {
        "max_items": 100,
        "max_item_chars": 0,
        "keep_full_top_k": 0,
        "total_budget_chars": 0,
        "truncation_policy": "soft_tail",
        "per_item_hard_cap": 0,
        "compaction": None,
    }
    base.update(overrides)
    return SynthesisContextFieldConfig(**base)


def test_apply_budget_keeps_everything_when_under_budget() -> None:
    cfg = _make_cfg(total_budget_chars=10_000)
    items = ["- a" * 100, "- b" * 100, "- c" * 100]
    kept, stats = _apply_budget(items, cfg)
    assert len(kept) == 3
    assert kept == items
    assert stats["kept_full"] == 3
    assert stats["soft_tail_trimmed"] == 0


def test_apply_budget_unlimited_when_total_budget_zero() -> None:
    cfg = _make_cfg(total_budget_chars=0)
    items = ["X" * 50_000 for _ in range(4)]
    kept, stats = _apply_budget(items, cfg)
    assert len(kept) == 4
    assert stats["soft_tail_trimmed"] == 0


def test_apply_budget_top_k_always_preserved_even_when_budget_too_small() -> None:
    cfg = _make_cfg(total_budget_chars=500, keep_full_top_k=2)
    items = ["A" * 1_000, "B" * 2_000, "C" * 3_000]
    kept, stats = _apply_budget(items, cfg)
    assert len(kept) == 2
    assert kept[0].startswith("A")
    assert kept[1].startswith("B")
    # Items past top-K never get emitted because the budget is exhausted.
    assert stats["kept_full"] == 2
    assert stats["soft_tail_trimmed"] == 0


def test_apply_budget_soft_tail_trims_only_the_overflowing_item() -> None:
    cfg = _make_cfg(total_budget_chars=1_800, truncation_policy="soft_tail")
    items = ["A" * 1_000, "B" * 2_000, "C" * 3_000]
    kept, stats = _apply_budget(items, cfg)
    # Item 0 fits; item 1 gets trimmed to the remaining budget; item 2 is dropped.
    assert len(kept) == 2
    assert kept[0] == "A" * 1_000
    assert "…[truncated]" in kept[1]
    assert stats["kept_full"] == 1
    assert stats["soft_tail_trimmed"] == 1


def test_apply_budget_hard_clip_matches_legacy_behaviour() -> None:
    cfg = _make_cfg(
        total_budget_chars=0,  # hard_clip ignores budget by design (legacy parity)
        truncation_policy="hard_clip",
        max_item_chars=300,
    )
    items = ["A" * 1_000, "B" * 2_000, "C" * 100]
    kept, stats = _apply_budget(items, cfg)
    # Every item clipped to <= 303 chars ("..." suffix for the two oversized);
    # all items retained (no item dropped) — this is the legacy [:300] semantics.
    assert len(kept) == 3
    assert kept[0] == "A" * 300 + "..."
    assert kept[1] == "B" * 300 + "..."
    assert kept[2] == "C" * 100
    assert stats["soft_tail_trimmed"] == 2


def test_apply_budget_per_item_hard_cap_applies_before_top_k() -> None:
    cfg = _make_cfg(per_item_hard_cap=200, keep_full_top_k=3)
    items = ["Z" * 10_000]
    kept, _stats = _apply_budget(items, cfg)
    assert len(kept) == 1
    # Hard cap wins even for top-K items; suffix is the multi-line marker.
    assert len(kept[0]) < 300
    assert "…[truncated]" in kept[0]


def test_render_observations_with_budget_uses_observation_record_text() -> None:
    cfg = _make_cfg(keep_full_top_k=2, total_budget_chars=100_000)
    obs = [
        ObservationRecord(observation_id=f"o{i}", text=f"insight_{i} " * 100)
        for i in range(3)
    ]
    text, stats = _render_observations_with_budget(obs, cfg)
    for i in range(3):
        assert f"insight_{i}" in text
    assert text.startswith("- insight_0")
    assert stats["kept_full"] == 3


def test_render_observations_skips_empty_items() -> None:
    cfg = _make_cfg(total_budget_chars=0)
    obs = [
        ObservationRecord(observation_id="o0", text="  "),
        ObservationRecord(observation_id="o1", text="real finding"),
    ]
    text, stats = _render_observations_with_budget(obs, cfg)
    assert text == "- real finding"
    assert stats["items_out"] == 1


def test_render_source_bullet_emits_snippet_and_content_sections() -> None:
    cfg = _make_cfg(
        include_snippet=True,
        include_content=True,
        max_content_chars_top_k=5_000,
        max_content_chars_other=1_000,
    )
    src = {
        "title": "Docs",
        "url": "https://example.com",
        "snippet": "the snippet",
        "content": "the full page " * 200,
        "source_kind": "web",
        "relevance_score": 0.73,
    }
    bullet, snip_in, cont_in = _render_source_bullet(src, is_top_k=True, cfg=cfg)
    assert "[Docs](https://example.com)" in bullet
    assert "(web, rel=0.73)" in bullet
    assert "Snippet: the snippet" in bullet
    assert "Content: the full page" in bullet
    assert snip_in is True
    assert cont_in is True


def test_render_source_bullet_uses_smaller_content_cap_outside_top_k() -> None:
    cfg = _make_cfg(
        include_snippet=True,
        include_content=True,
        max_content_chars_top_k=5_000,
        max_content_chars_other=50,
    )
    content = "x" * 10_000
    src = {"title": "T", "url": "u", "snippet": "s", "content": content}
    top_bullet, _, _ = _render_source_bullet(src, is_top_k=True, cfg=cfg)
    tail_bullet, _, _ = _render_source_bullet(src, is_top_k=False, cfg=cfg)
    assert len(top_bullet) > len(tail_bullet)
    assert "…" in tail_bullet


def test_render_sources_with_budget_counts_snippet_content_flags() -> None:
    cfg = _make_cfg(
        max_items=10,
        keep_full_top_k=5,
        total_budget_chars=50_000,
        include_snippet=True,
        include_content=True,
        max_content_chars_top_k=1_000,
        max_content_chars_other=500,
    )
    sources = [
        {"title": f"T{i}", "url": f"u{i}", "snippet": f"s{i}", "content": f"c{i}"}
        for i in range(3)
    ]
    text, stats = _render_sources_with_budget(sources, cfg)
    for i in range(3):
        assert f"T{i}" in text
        assert f"s{i}" in text
        assert f"c{i}" in text
    assert stats["snippets_included"] == 3
    assert stats["content_included"] == 3


def test_compile_typed_synthesis_context_preserves_full_observations() -> None:
    """Regression test for the Sagacity hallucination bug.

    Seeds an observation containing ``"Genie"`` well past the legacy 300-char
    cap and asserts the word survives into the assembled observations block.
    """
    obs_text = "preamble " * 50 + "Genie is the right name for this feature."
    evidence = EvidenceState(
        observations=[ObservationRecord(observation_id="o", text=obs_text)],
        sources=[
            SourceRecord(
                source_id="s1",
                url="https://docs.databricks.com/structured-retrieval-tools",
                title="Connect agents to structured data",
                snippet=(
                    "Create a structured retrieval tool using Unity Catalog SQL "
                    "functions. The following example creates a function called "
                    "lookup_customer_info."
                ),
            )
        ],
    )

    runtime = SimpleNamespace(
        capabilities=SimpleNamespace(evidence=evidence),
    )
    result = compile_typed_synthesis_context(runtime)  # type: ignore[arg-type]
    assert result is not None
    assert "Genie" in result.all_observations, (
        "Regression: observation text was truncated before reaching synth"
    )
    assert "lookup_customer_info" in result.sources_list, (
        "Regression: source snippet was dropped before reaching synth"
    )
    # Sanity: neither of the LLM-hallucinated tool names ever existed in input.
    assert "execute_sql" not in result.all_observations
    assert "execute_sql" not in result.sources_list


def test_compile_typed_synthesis_context_respects_config_override() -> None:
    """Explicit config overrides framework defaults."""
    tiny_cfg = SynthesisContextConfig(
        observations=SynthesisContextFieldConfig(
            max_items=10,
            max_item_chars=0,
            keep_full_top_k=0,
            total_budget_chars=50,
            truncation_policy="soft_tail",
        ),
        sources=SynthesisContextFieldConfig(
            max_items=5,
            max_item_chars=0,
            keep_full_top_k=0,
            total_budget_chars=0,
        ),
    )
    evidence = EvidenceState(
        observations=[
            ObservationRecord(observation_id="o1", text="A" * 500),
            ObservationRecord(observation_id="o2", text="B" * 500),
        ],
        sources=[],
    )
    runtime = SimpleNamespace(
        capabilities=SimpleNamespace(evidence=evidence),
    )
    result = compile_typed_synthesis_context(runtime, config=tiny_cfg)  # type: ignore[arg-type]
    assert result is not None
    # Soft-tail drops everything after the first overflow item.
    assert result.stats.observations_kept_full == 0
    assert result.stats.observations_soft_tail_trimmed >= 0
    assert result.stats.observation_items_out <= 1


def test_default_synthesis_context_has_generous_budgets() -> None:
    cfg = default_synthesis_context()
    assert cfg.observations is not None
    assert cfg.observations.total_budget_chars >= 100_000
    assert cfg.observations.keep_full_top_k >= 5
    assert cfg.observations.max_item_chars == 0  # no per-item cap under normal flow
    assert cfg.sources is not None
    assert cfg.sources.include_snippet is True
    assert cfg.sources.include_content is True
