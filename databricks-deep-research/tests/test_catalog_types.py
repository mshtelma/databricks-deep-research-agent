"""Tests for tool-catalog metadata types — :class:`CatalogCard` invariants
and the no-corpus-specific-strings stop-list (B1, B20).

Phase-0 of the tool-catalog auto-injection plan. These tests ensure the
foundational types refuse known-bad inputs at construction time so a
corrupted card cannot silently slip into a rendered prompt block.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from databricks_deep_research.tools.catalog_types import (
    CatalogCard,
    CatalogProvider,
)
from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
from databricks_deep_research.tools.factories.databricks import DatabricksToolFactory

# ---------------------------------------------------------------------------
# B1 — CatalogCard validation
# ---------------------------------------------------------------------------


class TestCatalogCardValidation:
    """Construction-time invariants on :class:`CatalogCard`."""

    def test_summary_must_not_be_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CatalogCard(summary="", input_prose="x", output_prose="y")

    def test_summary_must_not_be_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CatalogCard(summary="   ", input_prose="x", output_prose="y")

    def test_summary_max_length(self) -> None:
        # 121 chars — one over the documented ≤120 cap.
        too_long = "x" * 121
        with pytest.raises(ValueError, match="≤120 chars"):
            CatalogCard(summary=too_long, input_prose="x", output_prose="y")

    def test_summary_at_max_length_is_accepted(self) -> None:
        exactly_120 = "x" * 120
        card = CatalogCard(summary=exactly_120, input_prose="x", output_prose="y")
        assert card.summary == exactly_120

    def test_summary_no_braces(self) -> None:
        with pytest.raises(ValueError, match="literal"):
            CatalogCard(summary="hello {world}", input_prose="x", output_prose="y")

    def test_input_prose_no_braces(self) -> None:
        with pytest.raises(ValueError, match="literal"):
            CatalogCard(
                summary="ok summary",
                input_prose="provide a {var} here",
                output_prose="y",
            )

    def test_output_prose_no_braces(self) -> None:
        with pytest.raises(ValueError, match="literal"):
            CatalogCard(
                summary="ok summary",
                input_prose="x",
                output_prose="returns {data}",
            )

    def test_well_formed_card_constructs(self) -> None:
        card = CatalogCard(
            summary="A clean summary describing affordance.",
            input_prose="A description of the input shape.",
            output_prose="A description of the output shape.",
        )
        assert card.summary == "A clean summary describing affordance."
        assert card.input_prose == "A description of the input shape."
        assert card.output_prose == "A description of the output shape."

    def test_card_is_frozen(self) -> None:
        card = CatalogCard(summary="ok", input_prose="x", output_prose="y")
        with pytest.raises(FrozenInstanceError):
            card.summary = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# B20 — Card prose contains no corpus-specific or customer-identifying strings
# ---------------------------------------------------------------------------


# These stop-list entries are tokens we never want shipped in framework cards
# because they bias the LLM toward a particular customer's data shape. The
# list is intentionally small and conservative — it covers values that have
# leaked into prior prompts, not every conceivable proper noun.
_STOP_LIST: tuple[str, ...] = (
    "officeqa",
    "office_qa",
    "office-qa",
    "demo_corpus",
    "acme",
    "acmecorp",
    "internal_only",
)


def _all_factory_cards() -> dict[str, CatalogCard]:
    """Union of cards across all framework factories."""
    merged: dict[str, CatalogCard] = {}
    merged.update(dict(BuiltinToolFactory.catalog_cards))
    merged.update(dict(DatabricksToolFactory.catalog_cards))
    return merged


class TestCardCorpusNeutrality:
    """B20: framework cards may not name a specific corpus or customer."""

    @pytest.mark.parametrize("kind,card", list(_all_factory_cards().items()))
    def test_no_stop_list_strings_in_card(self, kind: str, card: CatalogCard) -> None:
        blob = (
            card.summary.lower()
            + "\n"
            + card.input_prose.lower()
            + "\n"
            + card.output_prose.lower()
        )
        for token in _STOP_LIST:
            assert token not in blob, (
                f"Card for kind={kind!r} contains stop-list token "
                f"{token!r}; cards must reference tool kinds and asset "
                "kinds, never corpus or customer names."
            )


# ---------------------------------------------------------------------------
# CatalogProvider — factories satisfy the structural protocol
# ---------------------------------------------------------------------------


class TestFactoriesAreCatalogProviders:
    """Every framework factory satisfies :class:`CatalogProvider`."""

    def test_builtin_factory_is_catalog_provider(self) -> None:
        # Class-level attributes are enough — the Protocol is structural.
        assert isinstance(BuiltinToolFactory(), CatalogProvider)

    def test_databricks_factory_is_catalog_provider(self) -> None:
        assert isinstance(DatabricksToolFactory(), CatalogProvider)

    def test_builtin_factory_supports_implies_card(self) -> None:
        """If a factory says it supports a kind, it MUST declare a card.

        This invariant prevents 'silent' kinds — kinds that are creatable
        through the factory but invisible in the rendered catalog block.
        """
        factory = BuiltinToolFactory()
        for kind in factory.catalog_cards:
            assert factory.supports(kind), (
                f"BuiltinToolFactory declares a card for kind={kind!r} "
                "but supports() returns False"
            )

    def test_databricks_factory_supports_implies_card(self) -> None:
        factory = DatabricksToolFactory()
        for kind in factory.catalog_cards:
            assert factory.supports(kind), (
                f"DatabricksToolFactory declares a card for kind={kind!r} "
                "but supports() returns False"
            )

    def test_safe_probes_keys_match_card_keys(self) -> None:
        """``safe_probes`` must declare an entry for every kind that has a
        card, even if the entry is ``None`` (indicating no probe supplied)."""
        for factory_cls in (BuiltinToolFactory, DatabricksToolFactory):
            card_kinds = set(factory_cls.catalog_cards.keys())
            probe_kinds = set(factory_cls.safe_probes.keys())
            assert card_kinds == probe_kinds, (
                f"{factory_cls.__name__}: catalog_cards and safe_probes must "
                f"cover the same kinds. cards-only={card_kinds - probe_kinds}, "
                f"probes-only={probe_kinds - card_kinds}"
            )
