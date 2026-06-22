"""Regression + (now-promoted) meter for cross-turn pool seeding.

History: on a follow-up turn the app loaded prior-turn sources but seeded them
under ``existing_sources``, while the pool hydrator only read
``discovered_sources`` — so prior sources were silently dropped and every
follow-up started with an EMPTY pool.

Phase-2b read-path fix: the app now stages a *bounded, canonicalized, citable*
seed under a DEDICATED key ``prior_sources_for_seed`` (NOT ``existing_sources``,
which stays "durable history" for the appendix/retrieval tool). A new selector
``select_prior_sources`` / ``coerce_prior_sources`` reads that key, and the
run-start seeder hydrates the pool from it.

* ``test_discovered_sources_key_hydrates_pool`` — the discovered-key path still
  works (pure regression).
* ``test_existing_sources_key_is_dropped`` — ``existing_sources`` is HISTORY,
  not a seed instruction: neither coercion may pull it into the pool. Permanent
  regression so a stray history payload never silently seeds.
* ``test_prior_sources_for_seed_hydrates_pool_after_readpath`` — PROMOTED from a
  strict xfail: the dedicated seed key now hydrates the follow-up pool.
"""

from __future__ import annotations

from databricks_deep_research.pools.pool_state import PoolConfig, PoolState
from databricks_deep_research.workflow.runtime.plan_execute_recovery import (
    coerce_discovered_sources,
    coerce_prior_sources,
    hydrate_pools_from_discovered_sources,
)
from databricks_deep_research.workflow.state import WorkflowState

_PRIOR_SOURCES = [
    {"url": "https://a.example/1", "title": "Prior A", "snippet": "sa", "content": "ca"},
    {"url": "https://b.example/2", "title": "Prior B", "snippet": "sb", "content": "cb"},
    {"url": "https://c.example/3", "title": "Prior C", "snippet": "sc", "content": "cc"},
]


def _fresh_pools() -> dict[str, PoolState]:
    return {
        "sources": PoolState(PoolConfig(name="sources", item_type="source", dedup_key="url")),
        "observations": PoolState(PoolConfig(name="observations", item_type="text")),
        "discovery_sources": PoolState(
            PoolConfig(name="discovery_sources", item_type="source", dedup_key="url")
        ),
    }


def test_discovered_sources_key_hydrates_pool() -> None:
    """The consumer reads ``discovered_sources`` — seeding it fills the pool."""
    state = WorkflowState(query="follow up")
    state.append("init", "discovered_sources", _PRIOR_SOURCES)

    discovered = coerce_discovered_sources(state)
    pools = _fresh_pools()
    hydrated = hydrate_pools_from_discovered_sources(pools, discovered)

    assert hydrated is True
    assert pools["sources"].count() == len(_PRIOR_SOURCES)


def test_existing_sources_key_is_dropped() -> None:
    """``existing_sources`` is durable HISTORY, not a seed instruction. Neither
    coercion may pull it into the pool — otherwise a stray history payload would
    silently seed the working set (the very ambiguity Codex §3 warned against)."""
    state = WorkflowState(query="follow up")
    state.append("init", "existing_sources", _PRIOR_SOURCES)

    assert coerce_discovered_sources(state) == []
    assert coerce_prior_sources(state) == []
    pools = _fresh_pools()
    assert hydrate_pools_from_discovered_sources(pools, coerce_prior_sources(state)) is False
    assert pools["sources"].count() == 0


def test_prior_sources_for_seed_hydrates_pool_after_readpath() -> None:
    """PROMOTED (was strict-xfail). Prior sources staged under the dedicated
    ``prior_sources_for_seed`` key now hydrate the follow-up pool (Phase 2b)."""
    state = WorkflowState(query="follow up")
    state.append("init", "prior_sources_for_seed", _PRIOR_SOURCES)

    prior = coerce_prior_sources(state)
    pools = _fresh_pools()
    hydrate_pools_from_discovered_sources(pools, prior)

    assert pools["sources"].count() == len(_PRIOR_SOURCES)
