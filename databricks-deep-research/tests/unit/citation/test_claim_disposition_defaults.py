"""Unit tests pinning the default ClaimDispositionConfig verdict→action mapping.

The defaults were flipped in PR 4a to favour SOFTEN over REMOVE/KEEP for any
verdict the verifier marked uncertain. These tests pin the mapping so an
accidental future revert is caught immediately, and they document via
parametrization which verdicts soften / keep / remove by default.

The qualitative softening behaviour (hedge text, table-cell preservation,
list-numbering preservation) is exercised by snapshot tests on
``_build_softened_fact_text`` elsewhere; here we only pin the *mapping*.
"""
from __future__ import annotations

import pytest

from databricks_deep_research.citation.config import (
    ClaimDisposition,
    ClaimDispositionConfig,
)


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("supported", ClaimDisposition.KEEP),
        ("partial", ClaimDisposition.SOFTEN),
        ("unsupported", ClaimDisposition.SOFTEN),
        ("contradicted", ClaimDisposition.REMOVE),
        ("abstained", ClaimDisposition.SOFTEN),
        ("analysis_partial", ClaimDisposition.SOFTEN),
        ("analysis_unsupported", ClaimDisposition.SOFTEN),
    ],
)
def test_default_disposition_mapping(verdict: str, expected: ClaimDisposition) -> None:
    """Every verdict-keyed field defaults to the documented action.

    Locks the contract for downstream pipeline behaviour: only
    ``contradicted`` is removed; everything else is softened or kept.
    """
    cfg = ClaimDispositionConfig()
    assert getattr(cfg, verdict) is expected, (
        f"Default disposition for verdict={verdict!r} should be {expected!r}; "
        f"got {getattr(cfg, verdict)!r}. If this is intentional, update the "
        "ClaimDispositionConfig docstring and this test."
    )


def test_default_contradicted_stays_remove() -> None:
    """Spelled-out: ``contradicted`` is the only default REMOVE.

    Captured separately so a future refactor that re-balances the table
    cannot silently lose the "wrong = remove" rule for contradicted claims.
    """
    cfg = ClaimDispositionConfig()
    assert cfg.contradicted is ClaimDisposition.REMOVE


def test_default_supported_stays_keep() -> None:
    """Spelled-out: ``supported`` is the only default KEEP.

    Any future "soften everything by default" sweep should leave supported
    claims alone — they're, by definition, verified.
    """
    cfg = ClaimDispositionConfig()
    assert cfg.supported is ClaimDisposition.KEEP


def test_callers_can_override_to_legacy_remove() -> None:
    """The back-out path documented in the config docstring works.

    Compliance pipelines wanting strict REMOVE for unsupported claims pass
    an explicit override; the rest of the table is unchanged.
    """
    cfg = ClaimDispositionConfig(unsupported=ClaimDisposition.REMOVE)
    assert cfg.unsupported is ClaimDisposition.REMOVE
    # Sanity-check other defaults survived the partial override.
    assert cfg.partial is ClaimDisposition.SOFTEN
    assert cfg.contradicted is ClaimDisposition.REMOVE


def test_config_is_frozen() -> None:
    """``ClaimDispositionConfig`` is frozen; mutation must raise.

    Frozen-ness is the only safeguard against accidental mid-pipeline
    rebinding — pin it so a future ``model_config`` edit can't silently
    unfreeze the model.
    """
    cfg = ClaimDispositionConfig()
    with pytest.raises(Exception):  # noqa: BLE001 — Pydantic raises ValidationError or AttributeError
        cfg.unsupported = ClaimDisposition.REMOVE  # type: ignore[misc]
