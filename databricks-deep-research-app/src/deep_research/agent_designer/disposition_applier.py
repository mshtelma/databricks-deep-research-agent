"""PR3-E R1 — Conditional claim_disposition_applier.

Thin wrapper exposing the existing
``databricks_deep_research/citation/pipeline.py:process_unverified_claims``
as a standalone callable used by the framework workflow's synthesizer
chain when ``SYNTH_PIPELINE_V2=true``.

The plan's intent: relocate Stage 8 (post-verification disposition) to
AFTER the draft synthesizer instead of AFTER a regen-then-verify cycle.
The regen pass otherwise re-introduces softened/abstained claims as
fresh prose that the verifier then has to chase again — which is
exactly the ``removed_claims=0`` hole the failing officeqa run
exhibited.

This module imports the framework module lazily so app-side test
collection does not pay framework-import overhead on every test run.
"""

from __future__ import annotations

from typing import Any


async def claim_disposition_applier(
    draft_report: str,
    claims: list[Any],
    policy: Any,
) -> tuple[str, dict[str, Any]]:
    """Apply claim disposition to *draft_report* using *policy*.

    Delegates to
    ``CitationVerificationPipeline.process_unverified_claims`` (Stage 8),
    constructing a minimal pipeline instance via ``object.__new__`` so we
    bypass the heavyweight ``__init__`` that requires all 7-stage
    protocol implementations. Stage 8 only reads ``self.config``, so the
    bypass is safe.

    Returns ``(final_markdown, disposition_summary)`` where
    ``disposition_summary`` is a dict with keys
    ``{removed_claims, softened_claims, rewritten_claims, total_claims}``.
    """
    from databricks_deep_research.citation.config import CitationConfig
    from databricks_deep_research.citation.pipeline import (
        CitationVerificationPipeline,
    )

    # Build a config carrying the disposition policy. CitationConfig is
    # a frozen pydantic model so we construct via the constructor.
    config = CitationConfig(claim_disposition=policy) if policy is not None else CitationConfig()
    # Stage 8 only reads ``self.config`` — bypass __init__ to avoid the
    # heavy 7-stage protocol dependencies (only relevant when running
    # the FULL pipeline; not needed here).
    pipeline = object.__new__(CitationVerificationPipeline)
    pipeline.config = config
    (
        final_markdown,
        removed,
        softened,
        rewritten,
    ) = await pipeline.process_unverified_claims(draft_report, claims)
    summary = {
        "removed_claims": removed,
        "softened_claims": softened,
        "rewritten_claims": rewritten,
        "total_claims": len(claims),
    }
    return final_markdown, summary
