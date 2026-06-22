"""Phase 2b-2: bounded, citable run-start seeding of prior sources into the pool."""

from __future__ import annotations

from databricks_deep_research.pools.pool_state import PoolConfig, PoolState
from databricks_deep_research.workflow.runtime.prior_source_seed import (
    seed_pools_with_prior_sources,
    seed_prior_sources_at_run_start,
)
from databricks_deep_research.workflow.state import WorkflowState


def _pools() -> dict[str, PoolState]:
    return {"sources": PoolState(PoolConfig(name="sources", item_type="source", dedup_key="url"))}


def test_seed_bounds_to_top_k() -> None:
    prior = [
        {"url": f"https://x/{i}", "title": str(i), "snippet": f"body {i}"} for i in range(30)
    ]
    pools = _pools()
    n = seed_pools_with_prior_sources(pools, prior, top_k=20)
    assert n == 20
    assert pools["sources"].count() == 20


def test_seed_dedups_and_handles_empty() -> None:
    pools = _pools()
    assert seed_pools_with_prior_sources(pools, [], top_k=20) == 0
    dupes = [{"url": "https://x/1", "snippet": "b"}, {"url": "https://x/1", "snippet": "b"}]
    assert seed_pools_with_prior_sources(pools, dupes, top_k=20) == 1


def test_seed_skips_bare_urls_without_evidence() -> None:
    """Citable-snapshot guard (Codex §2): a record with no content/snippet cannot
    ground a citation, so it is never seeded."""
    pools = _pools()
    assert seed_pools_with_prior_sources(pools, [{"url": "https://x/1"}], top_k=20) == 0


def test_seed_no_sources_pool_is_noop() -> None:
    assert seed_pools_with_prior_sources({}, [{"url": "https://x/1", "snippet": "b"}]) == 0


def test_run_start_seeds_only_when_flag_set() -> None:
    """The run-start hook seeds the pool from ``prior_sources_for_seed`` only when
    the app sets the ``seed_prior_sources`` state flag (framework stays
    flag-agnostic; app gates on CHAT_MEMORY_UNIFIED)."""
    state = WorkflowState(query="follow up")
    state.append("init", "prior_sources_for_seed", [{"url": "https://x/1", "snippet": "b"}])

    pools = _pools()
    # Flag absent → no seed (byte-identical to today's behavior).
    assert seed_prior_sources_at_run_start(state, pools) == 0
    assert pools["sources"].count() == 0

    # Flag set → seeds.
    state.append("init", "seed_prior_sources", True)
    assert seed_prior_sources_at_run_start(state, pools) == 1
    assert pools["sources"].count() == 1

