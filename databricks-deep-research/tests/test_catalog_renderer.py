"""Tests for the pure tool-catalog renderer (B2, B3, B4).

The renderer must be deterministic (same input → same output regardless of
declaration iteration order), brace-safe at the output boundary so probe
samples carrying user-touched payloads don't corrupt downstream
substitution, and well-behaved under the character budget — progressively
shedding from full → summary → drop while keeping the output stable.
"""

from __future__ import annotations

from datetime import UTC, datetime

from databricks_deep_research.tools.catalog_renderer import (
    REGISTRY_VERSION,
    CatalogConfig,
    render_tool_catalog,
)
from databricks_deep_research.tools.catalog_types import CatalogCard, ProbeSample
from databricks_deep_research.workflow.definition import ToolDeclaration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_card(prefix: str = "") -> CatalogCard:
    return CatalogCard(
        summary=f"{prefix}Summary describing the tool affordance.",
        input_prose=f"{prefix}Input prose describing what to pass.",
        output_prose=f"{prefix}Output prose describing what comes back.",
    )


def _decl(name: str, kind: str = "vector_search") -> ToolDeclaration:
    return ToolDeclaration(name=name, kind=kind, config={})


# ---------------------------------------------------------------------------
# B2 — Deterministic ordering by tool name
# ---------------------------------------------------------------------------


class TestDeterministicOrdering:
    """B2: same set of tools, any iteration order → same rendered output."""

    def test_alphabetical_ordering_regardless_of_input_order(self) -> None:
        cards = {"vector_search": _make_card()}
        decls_a = [_decl("zeta_tool"), _decl("alpha_tool"), _decl("mid_tool")]
        decls_b = [_decl("alpha_tool"), _decl("zeta_tool"), _decl("mid_tool")]

        result_a = render_tool_catalog(decls_a, cards)
        result_b = render_tool_catalog(decls_b, cards)

        assert result_a.text == result_b.text
        # Verify alphabetical ordering inside the rendered text.
        idx_alpha = result_a.text.index("alpha_tool")
        idx_mid = result_a.text.index("mid_tool")
        idx_zeta = result_a.text.index("zeta_tool")
        assert idx_alpha < idx_mid < idx_zeta

    def test_unknown_kinds_are_silently_skipped(self) -> None:
        """A declaration whose kind has no card is not catalog-aware and is
        skipped without error (e.g., legacy ``decorated`` callables)."""
        cards = {"vector_search": _make_card("VS:")}
        decls = [
            _decl("vec1", kind="vector_search"),
            _decl("custom_callable", kind="decorated"),
        ]
        result = render_tool_catalog(decls, cards)
        assert "vec1" in result.text
        assert "custom_callable" not in result.text
        assert result.rendered_tool_count == 1

    def test_empty_input_returns_no_tools_line(self) -> None:
        result = render_tool_catalog([], {"vector_search": _make_card()})
        assert result.text == "(no tools wired)"
        assert result.rendered_tool_count == 0
        assert result.omitted_count == 0

    def test_no_matching_kinds_returns_empty_text(self) -> None:
        result = render_tool_catalog([_decl("x", kind="decorated")], {"vector_search": _make_card()})
        assert result.text == ""
        assert result.rendered_tool_count == 0


# ---------------------------------------------------------------------------
# B3 — Brace escaping at output boundary
# ---------------------------------------------------------------------------


class TestBraceEscaping:
    """B3: probe samples may carry user-touched payloads; the renderer must
    escape literal ``{`` / ``}`` so downstream ``str.format``-style
    substitution does not crash on the rendered block."""

    def test_card_prose_already_brace_safe(self) -> None:
        # Card validation rejects braces in card prose, and the header /
        # intro contain no braces either, so a card-only render must
        # produce text without any brace characters.
        cards = {"vector_search": _make_card()}
        result = render_tool_catalog([_decl("v")], cards)
        assert "{" not in result.text
        assert "}" not in result.text
        # And it round-trips through .format() unchanged.
        assert result.text.format() == result.text

    def test_probe_sample_braces_are_escaped(self) -> None:
        cards = {"vector_search": _make_card()}
        probe = ProbeSample(
            sample_input={"query": "find {something} now"},
            sample_output='passages: [{"text": "result"}]',
            probed_at=datetime.now(UTC),
            status="ok",
        )
        result = render_tool_catalog(
            [_decl("v")],
            cards,
            config=CatalogConfig(include_probe_samples=True),
            probe_samples_by_name={"v": probe},
        )
        # The probe-injected user text contained both '{' and '}'. After
        # escaping, every brace in the rendered text is part of a doubled
        # pair, so feeding the result through ``str.format()`` with no
        # arguments yields a string with single braces — and never raises
        # IndexError / KeyError due to a stray placeholder.
        text = result.text
        # Sanity: the probe payloads actually made it into the rendered text
        # (in their escaped form).
        assert "{{something}}" in text
        assert '{{"text": "result"}}' in text
        # The acid test: round-tripping through .format() must succeed and
        # restore the original single-brace user content.
        round_tripped = text.format()
        assert "{something}" in round_tripped
        assert '{"text": "result"}' in round_tripped

    def test_probe_samples_omitted_when_flag_false(self) -> None:
        """Probe samples are omitted when the renderer knob disables them."""
        cards = {"vector_search": _make_card()}
        probe = ProbeSample(
            sample_input={"query": "secret"},
            sample_output="sensitive output",
            probed_at=datetime.now(UTC),
            status="ok",
        )
        result = render_tool_catalog(
            [_decl("v")],
            cards,
            config=CatalogConfig(include_probe_samples=False),
            probe_samples_by_name={"v": probe},
        )
        assert "sensitive output" not in result.text
        assert "secret" not in result.text

    def test_probe_with_error_status_is_not_rendered(self) -> None:
        cards = {"vector_search": _make_card()}
        probe = ProbeSample(
            sample_input={},
            sample_output="should not appear",
            probed_at=datetime.now(UTC),
            status="error",
            reason="boom",
        )
        result = render_tool_catalog(
            [_decl("v")],
            cards,
            config=CatalogConfig(include_probe_samples=True),
            probe_samples_by_name={"v": probe},
        )
        assert "should not appear" not in result.text


# ---------------------------------------------------------------------------
# B4 — Budget shedding to summary-only and beyond
# ---------------------------------------------------------------------------


class TestBudgetShedding:
    """B4: under a constrained ``max_chars`` budget, the renderer
    progressively sheds detail in a fixed order — full → summary →
    drop — and surfaces the omitted count."""

    def test_summary_only_threshold_collapses_large_catalogs(self) -> None:
        cards = {"vector_search": _make_card()}
        # 9 tools — over the default threshold of 8.
        decls = [_decl(f"tool_{i:02d}") for i in range(9)]
        result = render_tool_catalog(decls, cards)
        assert result.used_summary_only is True
        # No 'Input.' / 'Output.' headers in summary-only mode.
        assert "**Input:**" not in result.text
        assert "**Output:**" not in result.text
        # All 9 tools represented as bullet items.
        for i in range(9):
            assert f"tool_{i:02d}" in result.text
        assert result.rendered_tool_count == 9

    def test_max_chars_forces_progressive_shedding(self) -> None:
        cards = {"vector_search": _make_card()}
        decls = [_decl(f"tool_{i:02d}") for i in range(5)]
        # Tight budget — full rendering would exceed it; renderer must shed.
        cfg = CatalogConfig(max_chars=400, summary_only_threshold=99)
        result = render_tool_catalog(decls, cfg=None) if False else render_tool_catalog(decls, cards, config=cfg)
        assert len(result.text) <= cfg.max_chars
        # Some tools should have been collapsed to summary or dropped.
        assert result.used_summary_only or result.omitted_count > 0

    def test_extreme_budget_drops_tools_and_reports_omissions(self) -> None:
        cards = {"vector_search": _make_card()}
        decls = [_decl(f"tool_{i:02d}") for i in range(10)]
        cfg = CatalogConfig(max_chars=300, summary_only_threshold=99)
        result = render_tool_catalog(decls, cards, config=cfg)
        # Strict budget — at least some tools must be dropped or summarized.
        # The header + intro alone is ~250 chars; 10 full tools cannot fit.
        assert (
            result.omitted_count > 0 or result.used_summary_only
        ), "Expected shedding under tight budget"

    def test_default_full_rendering_under_threshold(self) -> None:
        cards = {"vector_search": _make_card()}
        decls = [_decl(f"tool_{i:02d}") for i in range(3)]
        result = render_tool_catalog(decls, cards)
        # Below threshold and budget; full prose should be present.
        assert "**Input:**" in result.text
        assert "**Output:**" in result.text
        assert result.used_summary_only is False
        assert result.omitted_count == 0
        assert result.rendered_tool_count == 3

    def test_registry_version_propagates_to_result(self) -> None:
        cards = {"vector_search": _make_card()}
        result = render_tool_catalog([_decl("v")], cards)
        assert result.registry_version == REGISTRY_VERSION


class TestRendererPurity:
    """Sanity: the renderer is referentially transparent."""

    def test_same_inputs_same_outputs_across_calls(self) -> None:
        cards = {"vector_search": _make_card()}
        decls = [_decl("a"), _decl("b"), _decl("c")]
        first = render_tool_catalog(decls, cards)
        second = render_tool_catalog(decls, cards)
        third = render_tool_catalog(decls, cards)
        assert first.text == second.text == third.text
        assert first.rendered_tool_count == second.rendered_tool_count == third.rendered_tool_count
