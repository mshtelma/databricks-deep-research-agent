"""Bounded, run-start seeding of prior-turn sources into the working pool.

Codex-aligned (UNIFIED_CHAT_MEMORY_PLAN.md §5): seed only the top-K relevant
prior sources, NOT all history — the researcher hybrid-searches them and the
synthesizer cites them through the unchanged 7-stage pipeline. Distinct from the
recovery-path ``hydrate_pools_from_discovered_sources`` which fires only on an
empty plan; this seeds at run start.

The 2b-0 spike proved the synthesizer collects citation sources from the
``sources`` pool and assigns citation indices from each source's own URL (no
UrlRegistry pre-seed needed). It also proved a seeded record must carry an
``evidence_quality`` in the substantive set (the app builder stamps
``"cached"``); a bare URL with no body is dropped here as a backstop.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_deep_research.pools.pool_state import PoolState
from databricks_deep_research.workflow.runtime_core.selectors import select_prior_sources


def seed_pools_with_prior_sources(
    pools: dict[str, PoolState],
    prior: list[dict[str, Any]],
    *,
    top_k: int = 20,
) -> int:
    """Add at most ``top_k`` prior sources (already ranked + canonicalized
    upstream) to the ``sources`` pool. Returns the count actually seeded.

    Deduplication is the pool's own ``url`` key. Records with no evidence body
    (``content``/``snippet``) are skipped — they cannot ground a citation and
    would only crowd the working set.
    """
    if not prior:
        return 0
    sources_pool = pools.get("sources")
    if sources_pool is None:
        return 0
    seeded = 0
    for item in prior[:top_k]:
        if type(item) is not dict or not item.get("url"):
            continue
        # Citable-snapshot guard (Codex §2): never seed a bare URL with no
        # evidence body — it can't ground a citation and invites the model to
        # paraphrase memory instead of citing a source. The app-side builder
        # already drops these; this is the framework-boundary backstop.
        if not (item.get("content") or item.get("snippet")):
            continue
        if sources_pool.add(item):
            seeded += 1
    return seeded


def seed_prior_sources_at_run_start(
    state: Any,
    pools: dict[str, PoolState],
    *,
    top_k: int = 20,
    logger: logging.Logger | None = None,
) -> int:
    """Run-start hook: seed the pool from ``prior_sources_for_seed`` iff the app
    set the ``seed_prior_sources`` state flag.

    The framework stays flag-agnostic — the app gates on ``CHAT_MEMORY_UNIFIED``
    and only sets the state flag when the feature is enabled, so with the flag
    absent this is a no-op and behaviour is byte-identical to before.
    """
    requested = state.get("seed_prior_sources") if hasattr(state, "get") else None
    if not requested:
        return 0
    prior_raw = select_prior_sources(state)
    prior = [item for item in prior_raw if isinstance(item, dict)]
    seeded = seed_pools_with_prior_sources(pools, prior, top_k=top_k)
    if seeded and logger is not None:
        logger.info("PRIOR_SOURCES_SEEDED count=%d", seeded)
    return seeded
